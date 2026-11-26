"""局外帧扩展注意力（决策记录 262，ModelConfig.macro_attend）的上线前检验，本机 CPU。

用检查点贪心打几局整局（带 privileged、route_summary），在真实帧上核对：
  1. 组批：attend 关时与原来相同；开时只有局外帧多了核心记录，且恰好是牌组、已有遗物、出怪池、买不起的商品和补上的 Boss 记录
     （Boss 记录数 = 头部 16、17 里非 0 的个数）；战斗帧的核心记录不变；选牌候选引用的牌组牌改为引用上下文表示。
  2. 模型：macro_attend 打开时旧检查点照常载入（kind_x 是新表）；战斗帧的策略与价值输出与原来相同（逐帧比，容差 1e-5）；
     局外帧的输出必然改变，报出策略的 KL、首选一致率与价值差，作为上线冲击的量级；组批与网络配置不一致时报错；kind_x 有梯度。

    .venv/bin/python -I tools/macro_attend_check.py --dll 新引擎/Sts2Headless.dll --ckpt 检查点.pt [--games 4]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

from sts2env.env import _Process  # noqa: E402
from sts2env.protocol import DecisionKind, TokenKind, Zone, parse  # noqa: E402

fails: list[str] = []


def check(ok: bool, msg: str) -> bool:
    if not ok:
        fails.append(msg)
        print("  不通过：", msg, flush=True)
    return ok


def load(ckpt: str, **over):
    import torch
    from sts2rl.model import ModelConfig, PolicyValueNet, load_state
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    net = PolicyValueNet(ModelConfig(**{**blob["config"]["model"], **over})).eval()
    load_state(net, blob["state"])
    return net


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dll", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--games", type=int, default=4)
    ap.add_argument("--steps", type=int, default=3000)
    a = ap.parse_args()
    import torch
    import torch.nn.functional as F
    from sts2rl.actor import MODE_RUN
    from sts2rl.features import BOSS_REF, Ragged, collate_ragged, macro_rows
    torch.set_num_threads(4)
    base = load(a.ckpt)
    new = load(a.ckpt, macro_attend=True)

    def pol(o) -> int:
        with torch.no_grad():
            lg, _ = base(collate_ragged(Ragged.from_obs([o], np.full(1, MODE_RUN)), attend=False).to("cpu"))
        return int(lg[0, :o.num_actions].argmax())
    proc = _Process(Path(a.dll), open("/dev/null", "wb"))
    obs = []
    for gi in range(a.games):
        proc.send({"op": "reset", "mode": "run", "character": "DEFECT", "ascension": 10, "seed": f"MATT{gi:04d}", "privileged": True, "route_summary": True})
        o = parse(proc.recv())
        for _ in range(a.steps):
            if o.done or o.stuck or o.num_actions == 0:
                break
            obs.append(o)
            proc.send({"op": "step", "action": pol(o)}); o = parse(proc.recv())
        print(f"  第 {gi} 局：第 {o.floor} 层，{'通关' if o.victory else '输'}，累计 {len(obs)} 帧", flush=True)
    r = Ragged.from_obs(obs, np.full(len(obs), MODE_RUN))
    mac = macro_rows(r.header, r.modes())

    # 一、组批
    b0, b1 = collate_ragged(r, attend=False), collate_ragged(r, attend=True)
    n0, n1 = b0.core_mask.sum(1), b1.core_mask.sum(1)
    # 战斗帧：只有候选引用了牌堆里的牌（从弃牌堆、抽牌堆、消耗堆选牌）的帧会变，多出的正好是被引用的那几张
    chg = (~mac) & (n0 != n1)
    for i in np.flatnonzero(~mac):
        ti = r.tok_ints[r.tok_off[i]:r.tok_off[i + 1]]; ci = r.cand_ints[r.cand_off[i]:r.cand_off[i + 1]]
        refs = {int(x) for x in np.concatenate([ci[:, 1], ci[:, 2]]) if 0 <= x < len(ti)}
        pooled = {x for x in refs if ti[x, 0] == TokenKind.CARD and ti[x, 2] in (Zone.DRAW, Zone.DISCARD, Zone.EXHAUST, Zone.DECK)}
        check(n1[i] - n0[i] == len(pooled), f"战斗第 {i} 帧：多出 {n1[i] - n0[i]} 条核心记录，被引用的牌堆里的牌 {len(pooled)} 张")
    print(f"战斗帧 {int((~mac).sum())} 帧，其中 {int(chg.sum())} 帧（选牌：{sorted({DecisionKind(int(r.header[i, 1])).name for i in np.flatnonzero(chg)})}）"
          f"多了被引用的牌，平均每帧 {float((n1 - n0)[chg].mean()) if chg.any() else 0:.1f} 张", flush=True)
    nb = (r.header[:, 16:18] > 0).sum(1) * mac
    kinds = {"牌组": 0, "遗物": 0, "出怪池": 0, "买不起的商品": 0, "Boss 记录": 0}
    for i in np.flatnonzero(mac):
        ti = r.tok_ints[r.tok_off[i]:r.tok_off[i + 1]]; tf = r.tok_feats[r.tok_off[i]:r.tok_off[i + 1]]
        k, z = ti[:, 0], ti[:, 2]
        add = {"牌组": int(((k == TokenKind.CARD) & (z == Zone.DECK)).sum()), "遗物": int(((k == TokenKind.RELIC) & (z == Zone.ALLY)).sum()),
               "出怪池": int((k == TokenKind.ENCOUNTER_POOL).sum()), "买不起的商品": int(((k == TokenKind.POWER) & (tf[:, 17] > 0)).sum()),
               "Boss 记录": int(nb[i])}
        for x, c in add.items():
            kinds[x] += c
        check(n1[i] - n0[i] == sum(add.values()), f"第 {i} 帧（{DecisionKind(int(r.header[i, 1])).name}）：多出的核心记录 {n1[i] - n0[i]}，应为 {sum(add.values())}")
    boss_tok = b1.tok_ints[:, 0] == BOSS_REF
    check(int(boss_tok.sum()) == int(nb.sum()), f"补上的 Boss 记录 {int(boss_tok.sum())} 条，应为 {int(nb.sum())}")
    sel = np.flatnonzero((r.header[:, 1] == DecisionKind.CARD_SELECT) & mac)     # 局外的选牌（战斗里从弃牌堆、抽牌堆选牌的不在这次范围内）
    if len(sel):
        S1 = len(b1.tok_ints); cs = np.isin(b1.cand_step, sel) & (b1.cand_a >= 0)
        frac = float((b1.cand_a[cs] < S1).mean()) if cs.any() else float("nan")
        S0 = len(b0.tok_ints); cs0 = np.isin(b0.cand_step, sel) & (b0.cand_a >= 0)
        frac0 = float((b0.cand_a[cs0] < S0).mean()) if cs0.any() else float("nan")
        check(frac > 0.95, f"选牌候选引用上下文表示的比例 {frac:.2f}（原来 {frac0:.2f}）")
        print(f"选牌：{len(sel)} 帧，候选引用上下文表示 {frac0:.0%} → {frac:.0%}", flush=True)
    print(f"组批：{len(obs)} 帧（局外 {int(mac.sum())}），局外帧每帧核心记录 {n0[mac].mean():.1f} → {n1[mac].mean():.1f}；"
          + "，".join(f"{k} {v / max(mac.sum(), 1):.1f}" for k, v in kinds.items()), flush=True)

    # 二、模型
    with torch.no_grad():
        lb, vb = base(b0.to("cpu"))
        ln, vn = new(b1.to("cpu"))
    fin = lambda x: torch.nan_to_num(x, neginf=-1e9)
    c = torch.as_tensor((~mac) & (n0 == n1))
    check(bool(torch.allclose(fin(lb[c]), fin(ln[c]), atol=1e-5)) and bool(torch.allclose(vb[c], vn[c], atol=1e-5)),
          f"战斗帧输出变了：策略最大差 {float((fin(lb[c]) - fin(ln[c])).abs().max()):.2e}，价值最大差 {float((vb[c] - vn[c]).abs().max()):.2e}")
    m = torch.as_tensor(mac)
    pb, pn = F.log_softmax(lb[m], -1), F.log_softmax(ln[m], -1)
    kl = (pb.exp() * (pb - pn)).nan_to_num().sum(-1)
    na = torch.as_tensor([o.num_actions for o in obs])[m]
    multi = na > 1
    agree = (lb[m].argmax(-1) == ln[m].argmax(-1))[multi].float().mean()
    print(f"上线冲击（局外 {int(m.sum())} 帧，多选 {int(multi.sum())} 帧）：KL 平均 {float(kl[multi].mean()):.4f}、中位 {float(kl[multi].median()):.4f}、"
          f"最大 {float(kl[multi].max()):.3f}；首选一致 {float(agree):.1%}；价值差平均 {float((vb[m] - vn[m]).abs().mean()):.4f}", flush=True)
    try:
        new(b0.to("cpu")); check(False, "组批没扩展、网络打开了扩展，却没有报错")
    except RuntimeError:
        pass
    new.train()
    _, v, _, _ = new.forward_all(b1.to("cpu"))
    v[m].sum().backward()
    check(new.kind_x.weight.grad is not None and float(new.kind_x.weight.grad.abs().sum()) > 0, "kind_x 没有梯度")
    print("全部通过" if not fails else f"共 {len(fails)} 项不通过", flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
