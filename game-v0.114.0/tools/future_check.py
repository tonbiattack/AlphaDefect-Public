"""整局预生成内容（记录类型 17，决策记录 259）的上线前检验。

一、帧（--dll-base 旧引擎、--dll-new 新引擎，各开一个进程，按同一串随机动作同步走）：
  1. 不带 future_content 时，新引擎的每一帧与旧引擎逐字节相同；
  2. 带 future_content 时，去掉类型 17 的记录、头部记录数改回之后，仍与旧引擎逐字节相同；类型 17 只出现在整局模式战斗之外的决策帧
     （结束、卡住的帧没有，与 features.macro_rows 一致）。头部 12、14 是进程级累计量，比较时不看（见 mask12）；
  3. 本幕第 0 条普通战、精英与全局记录特权特征 0、1（预抽好的下一场普通战、精英战）一致；
  4. 与动作无关：同一种子换一串动作（另一个随机数种子）走，开局帧与之后每一帧里“后面几幕”的内容完全相同，
     本幕内容等于开局时本幕表里从已走场数往后的那几项。
二、模型（--ckpt，CPU）：future_content 打开时旧检查点照常载入；清零初始化下有没有类型 17 的记录，价值与策略输出完全相同；
  把新层换成随机权重后，策略输出仍与没有这些记录时完全相同，只有局外价值变了；反向时新层有梯度。

    .venv/bin/python -I tools/future_check.py --dll-base 旧/Sts2Headless.dll --dll-new 新/Sts2Headless.dll [--seeds 12] [--steps 600] [--ckpt 检查点.pt]
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

from sts2env.env import _Process  # noqa: E402
from sts2env.protocol import HEADER_INTS, TOK_INTS, TOK_PRIV, tok_feats_n, DecisionKind, TokenKind, parse  # noqa: E402

FUT = int(TokenKind.FUTURE_CONTENT)
fails: list[str] = []


def check(ok: bool, msg: str) -> bool:
    if not ok:
        fails.append(msg)
        print("  不通过：", msg, flush=True)
    return ok


def mask12(buf: bytes) -> bytes:
    """头部 12、14 是引擎进程级的累计量（12 累计赢的战斗场数 UiFlows.CombatsWon，14 遇到过的未知词表项个数 Vocab.Missing），
    跨局不清零、模型不用，同一进程里先打过几局就会不同，比较时置 0。"""
    h = np.frombuffer(buf, np.int32, HEADER_INTS).copy(); h[12] = 0; h[14] = 0
    return h.tobytes() + buf[HEADER_INTS * 4:]


def strip_future(buf: bytes) -> bytes:
    """去掉类型 17 的记录，头部记录数改回，其余字节原样拼回（布局见 protocol.parse）；头部 12、14 置 0（见 mask12）。"""
    nh = HEADER_INTS
    h = np.frombuffer(buf, np.int32, nh).copy(); h[12] = 0; h[14] = 0
    n, m = int(h[2]), int(h[3])
    off = nh * 4
    ti = np.frombuffer(buf, np.int32, n * TOK_INTS, off).reshape(n, TOK_INTS); off += n * TOK_INTS * 4
    nf = tok_feats_n(int(h[0]))      # 版本 27 起每条记录 22 个特征
    tf = np.frombuffer(buf, np.float32, n * nf, off).reshape(n, nf); off += n * nf * 4
    tp = np.frombuffer(buf, np.float32, n * TOK_PRIV, off).reshape(n, TOK_PRIV); off += n * TOK_PRIV * 4
    keep = ti[:, 0] != FUT
    h[2] = int(keep.sum())
    return h.tobytes() + ti[keep].tobytes() + tf[keep].tobytes() + tp[keep].tobytes() + buf[off:]


def future_records(o) -> list[tuple[int, int, int, int, int]]:
    """(第几幕之后, 类别, 先后, 弱池位, 词表 ID)，按引擎输出的顺序。"""
    k = o.tok_ints[:, 0] == FUT
    tf = o.tok_feats[k]
    return [(int(round(a)), int(round(b)), int(round(c)), int(round(d)), int(i))
            for (a, b, c, d), i in zip(tf[:, :4], o.tok_ints[k, 1])]


def play(proc: _Process, seed: str, act_seed: int, steps: int, flag: bool, pol=None, eps: float = 0.0):
    """走一局（至多 steps 步），返回每一帧的原始字节。pol 为 None 时按随机动作；否则按检查点贪心，另以概率 eps 随机换一个动作。"""
    rng = np.random.default_rng(act_seed)
    req = {"op": "reset", "mode": "run", "character": "DEFECT", "ascension": 10, "seed": seed, "privileged": True,
           "route_summary": True, "snapshots": True}
    if flag:
        req["future_content"] = True
    proc.send(req); out = [proc.recv()]
    for _ in range(steps):
        o = parse(out[-1])
        if o.done or o.stuck or o.num_actions == 0:
            break
        act = int(rng.integers(o.num_actions)) if pol is None or rng.random() < eps else pol(o)
        proc.send({"op": "step", "action": act}); out.append(proc.recv())
    return out


def greedy(ckpt: str):
    """检查点的贪心动作（CPU，逐帧推理；不带 future_content，策略本来就看不到类型 17）。"""
    import torch
    from sts2rl.features import Ragged, collate_ragged
    from sts2rl.model import ModelConfig, PolicyValueNet, load_state
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    net = PolicyValueNet(ModelConfig(**blob["config"]["model"])).eval()
    load_state(net, blob["state"])
    torch.set_num_threads(4)

    def pol(o) -> int:
        with torch.no_grad():
            logits, _ = net(collate_ragged(Ragged.from_obs([o], np.ones(1, np.int64))).to("cpu"))
        return int(logits[0, :o.num_actions].argmax())
    return pol


def frames_check(a) -> list[list[bytes]]:
    base, new = _Process(Path(a.dll_base), open("/dev/null", "wb")), _Process(Path(a.dll_new), open("/dev/null", "wb"))
    seeds = [f"FUTCHK{i:04d}" for i in range(a.seeds)]
    kept = []
    n_frames = n_fut_frames = n_fut = 0
    try:
        for si, seed in enumerate(seeds):
            fb = play(base, seed, si, a.steps, False)
            fn = play(new, seed, si, a.steps, False)
            check(len(fb) == len(fn) and all(mask12(x) == mask12(y) for x, y in zip(fb, fn)), f"{seed}：不带开关时新旧引擎的帧不同")
            ff = play(new, seed, si, a.steps, True)
            check(len(fb) == len(ff), f"{seed}：带开关时局长不同（{len(ff)} 对 {len(fb)}）")
            for t, (x, y) in enumerate(zip(fb, ff)):
                o = parse(y)
                recs = future_records(o)
                n_frames += 1; n_fut += len(recs); n_fut_frames += bool(recs)
                if not check(strip_future(y) == mask12(x), f"{seed} 第 {t} 帧：去掉类型 17 后与旧引擎不同"):
                    break
                if o.in_combat or o.done or o.stuck or o.decision == DecisionKind.COMBAT:
                    check(not recs, f"{seed} 第 {t} 帧：战斗中、结束或卡住的帧出现了类型 17")
                    continue
                check(bool(recs), f"{seed} 第 {t} 帧：战斗外的整局帧没有类型 17")
                g = o.tok_ints[:, 0] == TokenKind.GLOBAL
                pn, pe = (int(round(v)) for v in o.tok_priv[g][0, :2])
                cur_n = [r[4] for r in recs if r[0] == 0 and r[1] == 0 and r[2] == 0]
                cur_e = [r[4] for r in recs if r[0] == 0 and r[1] == 1 and r[2] == 0]
                if cur_n:
                    check(cur_n[0] == pn, f"{seed} 第 {t} 帧：本幕下一场普通战 {cur_n[0]} 与特权特征 {pn} 不一致")
                if cur_e:
                    check(cur_e[0] == pe, f"{seed} 第 {t} 帧：本幕下一场精英 {cur_e[0]} 与特权特征 {pe} 不一致")
            kept.append(ff)
            print(f"  {seed}：{len(ff)} 帧，第 {parse(ff[-1]).act + 1} 幕第 {parse(ff[-1]).floor} 层结束", flush=True)
    finally:
        base.close(); new.close()
    print(f"帧：{n_frames} 帧，带类型 17 的 {n_fut_frames} 帧，平均每帧 {n_fut / max(1, n_fut_frames):.1f} 条", flush=True)
    return kept


def exogeneity_check(a) -> None:
    """同一种子换几串不同的动作：每条记录换算成（第几幕, 类别, 绝对位置）→ 词表 ID，所有帧、所有动作串里同一个位置必须是同一个 ID。
    本幕普通战、精英的绝对位置 = 已走场数（头部 18、19）+ 先后；后面几幕从 0 算；本幕事件不知道已走几个，不比。
    走到下一幕时，头部 16、17（本幕 Boss，公开）必须等于此前预告的那一幕的 Boss。"""
    new = _Process(Path(a.dll_new), open("/dev/null", "wb"))
    pol = greedy(a.ckpt) if a.ckpt else None
    n_keys = n_boss = 0
    try:
        for si in range(a.exo_seeds):
            seed = f"FUTCHK{si:04d}"
            plan = [(None, 0.0), (None, 0.0)] if pol is None else [(pol, 0.0), (pol, 0.15), (None, 0.0)]
            table: dict[tuple[int, int, int], int] = {}
            lens = []
            for j, (pp, eps) in enumerate(plan):
                r = play(new, seed, 1000 + si * 7 + j, a.steps if pp is None else 6000, True, pp, eps)
                lens.append(f"{len(r)} 帧到第 {parse(r[-1]).act + 1} 幕")
                for t, buf in enumerate(r):
                    o = parse(buf)
                    for (off, cat, order, _weak, vid) in future_records(o):
                        act = o.act + off
                        if off == 0 and cat == 3:
                            continue
                        pos = order + (int(o.header[18]) if off == 0 and cat == 0 else int(o.header[19]) if off == 0 and cat == 1 else 0)
                        key = (act, cat, pos)
                        if key in table:
                            check(table[key] == vid, f"{seed} 第 {j} 串第 {t} 帧：第 {act + 1} 幕类别 {cat} 第 {pos} 项 {vid} 与先前的 {table[key]} 不同")
                        else:
                            table[key] = vid
                    b1, b2 = int(o.header[16]), int(o.header[17])
                    if (o.act, 2, 0) in table and b1:
                        n_boss += 1
                        check(table[(o.act, 2, 0)] == b1, f"{seed} 第 {j} 串第 {t} 帧：第 {o.act + 1} 幕 Boss {b1} 与预告的 {table[(o.act, 2, 0)]} 不同")
                        if (o.act, 2, 1) in table:
                            check(table[(o.act, 2, 1)] == b2, f"{seed} 第 {j} 串第 {t} 帧：第 {o.act + 1} 幕第二个 Boss {b2} 与预告的 {table[(o.act, 2, 1)]} 不同")
            n_keys += len(table)
            print(f"  {seed}：{'、'.join(lens)}；比对了 {len(table)} 个位置", flush=True)
    finally:
        new.close()
    print(f"与动作无关：{n_keys} 个位置，Boss 预告与实际核对 {n_boss} 帧", flush=True)


def model_check(a, kept: list[list[bytes]]) -> None:
    import torch
    from sts2rl.features import Ragged, collate_ragged
    from sts2rl.model import ModelConfig, PolicyValueNet, load_state
    blob = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    mc = dict(blob["config"]["model"]); mc["future_content"] = True
    net = PolicyValueNet(ModelConfig(**mc)).eval()
    load_state(net, blob["state"])
    print(f"模型：{a.ckpt} 载入（future_content 打开，新层 {sum(p.numel() for n, p in net.named_parameters() if n.startswith('fut_')):,} 个参数）", flush=True)
    obs = [parse(b) for r in kept for b in r]
    obs = [o for o in obs if not o.done and o.num_actions > 0][:600]
    mode = np.ones(len(obs), np.int64)
    with_f = Ragged.from_obs(obs, mode)
    stripped = []
    for o in obs:
        k = o.tok_ints[:, 0] != FUT
        h = o.header.copy(); h[2] = int(k.sum())
        stripped.append(type(o)(h, o.tok_ints[k], o.tok_feats[k], o.tok_priv[k], o.cand_ints, o.cand_feats, o.labels, o.extra))
    without = Ragged.from_obs(stripped, mode)
    bw, bo = collate_ragged(with_f), collate_ragged(without)
    macro = bw.macro.astype(bool)
    dk = with_f.drop_kind(FUT)      # 价值探针“去特权”用的去记录（features.Ragged.drop_kind）应与逐帧去掉的结果完全相同
    check(all(np.array_equal(getattr(dk, f), getattr(without, f)) for f in ("header", "tok_ints", "tok_feats", "tok_priv", "tok_off", "cand_ints", "cand_feats", "cand_off")),
          "Ragged.drop_kind 与逐帧去掉类型 17 的结果不同")
    check(len(bw.fut_tok) > 0 and len(bo.fut_tok) == 0, "组批后类型 17 的下标不对")
    with torch.no_grad():
        lw, vw = net(bw.to("cpu")); lo, vo = net(bo.to("cpu"))
    finite = torch.isfinite(lw)
    check(torch.equal(lw[finite], lo[finite]) and torch.equal(finite, torch.isfinite(lo)), "清零初始化下有没有类型 17 的记录，策略输出不同")
    check(torch.allclose(vw, vo, atol=0, rtol=0), f"清零初始化下有没有类型 17 的记录，价值不同（最大差 {float((vw - vo).abs().max()):.3g}）")
    torch.manual_seed(0)
    with torch.no_grad():
        for p in net.fut_out.parameters():
            p.normal_(0, 0.05)
        lw2, vw2 = net(bw.to("cpu")); lo2, vo2 = net(bo.to("cpu"))
    check(torch.equal(lw2[finite], lo2[finite]), "新层换成随机权重后，策略输出随类型 17 的记录变了（泄漏）")
    dv = (vw2 - vo2).abs().numpy()
    check(dv[macro].max() > 1e-4 and dv[~macro].max() == 0, f"随机权重下局外价值应变、战斗价值不应变（局外最大差 {dv[macro].max():.3g}，战斗 {dv[~macro].max():.3g}）")
    net.train(); net.zero_grad()
    _, v = net(bw.to("cpu"))
    (v[torch.from_numpy(macro)] ** 2).mean().backward()
    gf = sum(float(p.grad.abs().sum()) for n, p in net.named_parameters() if n.startswith("fut_") and p.grad is not None)
    check(gf > 0, "新层没有梯度")
    print(f"模型：{len(obs)} 帧（局外 {int(macro.sum())}），清零时输出不变、随机权重时只有局外价值变（最大差 {dv[macro].max():.3g}），新层梯度和 {gf:.3g}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dll-base", required=True)
    ap.add_argument("--dll-new", required=True)
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--ckpt", default=None, help="给了就做模型检验，并在“与动作无关”的检验里用它贪心打局（打得到后几幕）")
    ap.add_argument("--exo-seeds", type=int, default=3)
    a = ap.parse_args()
    print("一、帧", flush=True)
    kept = frames_check(a)
    print("一（续）、与动作无关", flush=True)
    exogeneity_check(a)
    if a.ckpt:
        print("二、模型", flush=True)
        model_check(a, kept)
    print(json.dumps({"fails": len(fails)}, ensure_ascii=False))
    if fails:
        print("\n".join(fails[:30]))
        sys.exit(1)
    print("全部通过")


if __name__ == "__main__":
    main()
