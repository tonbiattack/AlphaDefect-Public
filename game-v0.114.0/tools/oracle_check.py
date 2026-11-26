"""预知记录（记录类型 20，决策记录 280）的上线前检验。

一、帧（--dll-base 旧引擎、--dll-new 新引擎，按同一串随机动作同步走，两边都带 privileged、future_content，与训练采集相同）：
  1. 不带 oracle 时，新引擎的每一帧与旧引擎逐字节相同；
  2. 带 oracle 时，去掉类型 20 的记录、头部记录数改回之后，仍与旧引擎逐字节相同，局长相同——推算没有改动对局；
     类型 20 只出现在整局模式战斗之外的决策帧。头部 12、14 是进程级累计量，比较时不看。
二、准不准：每次出现卡牌奖励界面（CARD_REWARD），拿它给的三张牌和战斗前最后一个战斗外帧里对应房间的预览比（三种房间都看，
   取对上最多的那一种），统计三张全对、对上几张；药水掉落与精英遗物同理。只是统计，不判通过与否（个别遗物、事件会改动奖励）。

    .venv/bin/python -I tools/oracle_check.py --dll-base 旧/Sts2Headless.dll --dll-new 新/Sts2Headless.dll [--seeds 20] [--steps 800]
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

from sts2env.env import _Process  # noqa: E402
from sts2env.protocol import HEADER_INTS, TOK_INTS, TOK_PRIV, tok_feats_n, DecisionKind, TokenKind, Zone, parse  # noqa: E402

ORC = int(TokenKind.ORACLE)
DEBUG = "--debug" in sys.argv
VOC = (REPO / "engine/catalog/vocab.txt").read_text(encoding="utf-8").split("\n")
fails: list[str] = []


def check(ok: bool, msg: str) -> bool:
    if not ok:
        fails.append(msg)
        print("  不通过：", msg, flush=True)
    return ok


def mask12(buf: bytes) -> bytes:
    h = np.frombuffer(buf, np.int32, HEADER_INTS).copy(); h[12] = 0; h[14] = 0
    return h.tobytes() + buf[HEADER_INTS * 4:]


def strip_oracle(buf: bytes) -> bytes:
    nh = HEADER_INTS
    h = np.frombuffer(buf, np.int32, nh).copy(); h[12] = 0; h[14] = 0
    n = int(h[2])
    off = nh * 4
    ti = np.frombuffer(buf, np.int32, n * TOK_INTS, off).reshape(n, TOK_INTS); off += n * TOK_INTS * 4
    nf = tok_feats_n(int(h[0]))      # 版本 27 起每条记录 22 个特征
    tf = np.frombuffer(buf, np.float32, n * nf, off).reshape(n, nf); off += n * nf * 4
    tp = np.frombuffer(buf, np.float32, n * TOK_PRIV, off).reshape(n, TOK_PRIV); off += n * TOK_PRIV * 4
    keep = ti[:, 0] != ORC
    h[2] = int(keep.sum())
    return h.tobytes() + ti[keep].tobytes() + tf[keep].tobytes() + tp[keep].tobytes() + buf[off:]


def records(o) -> list[tuple[int, int, int, float, int]]:
    """(类别, 先后, 子类, 附加数值, 词表 ID)。"""
    k = o.tok_ints[:, 0] == ORC
    return [(int(round(a)), int(round(b)), int(round(c)), float(d), int(i)) for (a, b, c, d), i in zip(o.tok_feats[k, :4], o.tok_ints[k, 1])]


def play(proc: _Process, seed: str, act_seed: int, steps: int, oracle: bool) -> list[bytes]:
    rng = np.random.default_rng(act_seed)
    req = {"op": "reset", "mode": "run", "character": "DEFECT", "ascension": 10, "seed": seed, "privileged": True, "future_content": True,
           "route_summary": True, "snapshots": True}
    if oracle:
        req["oracle"] = True
    proc.send(req); out = [proc.recv()]
    for _ in range(steps):
        o = parse(out[-1])
        if o.done or o.stuck or o.num_actions == 0:
            break
        # 战斗里优先结束回合以外的动作会拖很长；随机动作即可，走得到卡牌奖励就够
        proc.send({"op": "step", "action": int(rng.integers(o.num_actions))}); out.append(proc.recv())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dll-base", required=True)
    ap.add_argument("--dll-new", required=True)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()
    base, new = _Process(Path(a.dll_base), open("/dev/null", "wb")), _Process(Path(a.dll_new), open("/dev/null", "wb"))
    n_frames = n_orc_frames = n_orc = 0
    acc = Counter()
    cats = Counter()
    try:
        for si in range(a.seeds):
            seed = f"ORCCHK{si:04d}"
            fb = play(base, seed, si, a.steps, False)
            fn = play(new, seed, si, a.steps, False)
            check(len(fb) == len(fn) and all(mask12(x) == mask12(y) for x, y in zip(fb, fn)), f"{seed}：不带 oracle 时新旧引擎的帧不同")
            fo = play(new, seed, si, a.steps, True)
            check(len(fb) == len(fo), f"{seed}：带 oracle 时局长不同（{len(fo)} 对 {len(fb)}），推算改动了对局")
            last = pending = None
            for t, (x, y) in enumerate(zip(fb, fo)):
                o = parse(y)
                recs = records(o)
                n_frames += 1; n_orc += len(recs); n_orc_frames += bool(recs)
                for r in recs:
                    cats[r[0]] += 1
                if not check(strip_oracle(y) == mask12(x), f"{seed} 第 {t} 帧：去掉类型 20 后与旧引擎不同"):
                    break
                if o.in_combat or o.done or o.stuck or o.decision == DecisionKind.COMBAT:
                    check(not recs, f"{seed} 第 {t} 帧：战斗中、结束或卡住的帧出现了类型 20")
                    if o.in_combat and pending is not None:
                        last, pending = pending, None      # 进战斗前最后一个战斗外帧的预览，就是这场战斗奖励的预测
                elif recs:
                    pending = recs
                if o.decision == DecisionKind.CARD_REWARD and last is not None:
                    offer = o.tok_ints[(o.tok_ints[:, 0] == TokenKind.CARD) & (o.tok_ints[:, 2] == Zone.OFFER), 1].tolist()
                    if len(offer) == 3:
                        best = max(len(set(offer) & {r[4] for r in last if r[0] == c}) for c in (0, 1, 2))
                        acc[f"卡牌奖励对上 {best} 张"] += 1
                        if DEBUG:
                            print("    奖励", t, "层", o.floor, "给的", [VOC[i] for i in offer], "预览", {c: [VOC[r[4]] for r in last if r[0] == c] for c in (0, 1, 2)}, flush=True)
                    last = None      # 每次战斗只比一次（奖励界面可能连着几帧）
            print(f"  {seed}：{len(fo)} 帧，第 {parse(fo[-1]).act + 1} 幕第 {parse(fo[-1]).floor} 层结束", flush=True)
    finally:
        base.close(); new.close()
    print(f"帧：{n_frames} 帧，带类型 20 的 {n_orc_frames} 帧，平均每帧 {n_orc / max(1, n_orc_frames):.1f} 条；各类别条数 {dict(sorted(cats.items()))}")
    print("准确：", dict(sorted(acc.items())))
    print("全部通过" if not fails else f"不通过 {len(fails)} 项")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
