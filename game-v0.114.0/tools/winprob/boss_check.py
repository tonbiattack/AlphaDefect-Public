"""胜率模型在 Boss 战上准不准（2026-11-06，用户问：模型预测胜率低时有多少真输了、真输了的有多少开打前就料到）。

每局评测回放里找出各幕 Boss 战（第一幕、第二幕、第三幕第一个、第二个），记三个时刻的预测与这场 Boss 的输赢：
  开打前：Boss 战前最后一个战斗外的决策（多为选路线走进 Boss），“打过下一个 Boss”头；
  开打时：Boss 战第一步（敌人、手牌都看得到了），“赢下当前战斗”头与“打过下一个 Boss”头。
输出每类 Boss 的场数、输率、AUC，按阈值的“预测低 → 真输了的比例”“真输了 → 预测低的比例”，以及校准五档。

    python -I tools/winprob/boss_check.py --model 胜率模型.pt --replays 评测回放目录 [--min-version 169360] [--workers 4] [--out 明细.jsonl]
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from winprob import WinProb, N_SCALAR, featurize_step, pack, read_replay, select, to_torch, labels  # noqa: E402

ROOM_BOSS = None
_net = None


def init(model_path: str) -> None:
    global _net, ROOM_BOSS
    import torch
    import winprob
    torch.set_num_threads(1)
    ROOM_BOSS = winprob.ROOM_BOSS
    b = torch.load(model_path, map_location="cpu", weights_only=False)
    am = b["model"]["mlp.0.weight"].shape[1] == 9 * b["args"].get("d", 16) + N_SCALAR
    _net = WinProb(**{**b["args"], "amounts": am}).eval()
    _net.load_state_dict(b["model"])


def one(path: str) -> list[dict]:
    import torch
    r = read_replay(Path(path))
    if r is None:
        return []
    head, steps = r
    g = [s["g"] for s in steps]
    n = len(steps)
    y = labels(head, steps)
    # Boss 段：连续的战斗步、同一层、房间为 Boss
    segs, i = [], 0
    while i < n:
        if g[i][8]:
            j = i
            while j + 1 < n and g[j + 1][8] and g[j + 1][1] == g[i][1]:
                j += 1
            if g[i][9] == ROOM_BOSS:
                segs.append((i, j))
            i = j + 1
        else:
            i += 1
    if not segs:
        return []
    want = set()
    pre = {}
    for a, _ in segs:
        k = a - 1
        while k >= 0 and g[k][8]:
            k -= 1
        pre[a] = k if k >= 0 else None
        want.add(a)
        if pre[a] is not None:
            want.add(pre[a])
    rows, idx, deck = [], [], []
    for t, s in enumerate(steps):
        if "deck" in s:
            deck = s["deck"]
        if t in want:
            rows.append(featurize_step(s, deck)); idx.append(t)
    with torch.no_grad():
        D = pack(rows)
        p = torch.sigmoid(_net(to_torch(select(D, np.arange(D["n"])), "cpu"))).numpy()
    P = {t: p[k] for k, t in enumerate(idx)}
    out, act_count = [], {}
    for a, e in segs:
        act = int(g[a][0])
        act_count[act] = act_count.get(act, 0) + 1
        kind = f"第{act + 1}幕" + (f"第{act_count[act]}个" if act == 2 else "")
        rec = {"file": Path(path).name, "seed": head.get("seed"), "kind": kind, "floor": int(g[a][1]), "won": bool(y[a, 0]),
               "hp": int(g[a][2]), "max_hp": int(g[a][3]),
               "start_fight": float(P[a][0]), "start_boss": float(P[a][1]), "run_win": bool(head.get("win"))}
        if pre[a] is not None:
            rec["pre_boss"] = float(P[pre[a]][1])
        out.append(rec)
    return out


def auc(score_loss: np.ndarray, lost: np.ndarray) -> float:
    """“预测越低越可能输”的 AUC：输的局的预测比赢的局低的比例。"""
    if lost.all() or not lost.any():
        return float("nan")
    a, b = score_loss[lost], score_loss[~lost]
    # 输的那场预测低于赢的那场的比例（相等算一半）
    return float(((a[:, None] < b[None, :]).sum() + 0.5 * (a[:, None] == b[None, :]).sum()) / (len(a) * len(b)))


def report(recs: list[dict]) -> None:
    kinds = ["第1幕", "第2幕", "第3幕第1个", "第3幕第2个"]
    for col, name in (("pre_boss", "开打前·打过这个 Boss"), ("start_fight", "开打时·赢下这场"), ("start_boss", "开打时·打过这个 Boss")):
        print(f"\n=== {name} ===")
        for kind in kinds:
            rs = [r for r in recs if r["kind"] == kind and col in r]
            if not rs:
                continue
            p = np.array([r[col] for r in rs]); lost = np.array([not r["won"] for r in rs])
            print(f"{kind}：{len(rs):,} 场，输 {lost.sum():,}（{lost.mean():.1%}），平均预测 {p.mean():.1%}，AUC {auc(p, lost):.3f}")
            for thr in (0.3, 0.5, 0.7):
                low = p < thr
                if low.any():
                    print(f"   预测 < {thr:.0%}：{low.sum():,} 场（{low.mean():.1%}），其中真输 {lost[low].mean():.1%}；真输的里面预测 < {thr:.0%} 的 {low[lost].mean():.1%}")
            qs = [0, 0.2, 0.4, 0.6, 0.8, 1.01]
            cal = []
            for lo, hi in zip(qs, qs[1:]):
                m = (p >= lo) & (p < hi)
                if m.sum():
                    cal.append(f"{lo:.0%}–{min(hi, 1):.0%}: {m.sum()} 场 预测 {p[m].mean():.0%} 实际赢 {1 - lost[m].mean():.0%}")
            print("   校准：" + "；".join(cal))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--replays")
    ap.add_argument("--min-version", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--report", default=None, help="只按已有明细出报告")
    a = ap.parse_args()
    if a.report:
        report([json.loads(l) for l in open(a.report, encoding="utf-8")])
        return
    files = [str(f) for f in sorted(Path(a.replays).glob("v*_*.jsonl.gz")) if int(f.name[1:].split("_")[0]) >= a.min_version]
    print(f"{len(files):,} 局", flush=True)
    recs = []
    with ProcessPoolExecutor(a.workers, initializer=init, initargs=(a.model,)) as pool:
        for k, rs in enumerate(pool.map(one, files, chunksize=16)):
            recs += rs
            if (k + 1) % 1000 == 0:
                print(f"  {k + 1:,} 局", flush=True)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    report(recs)


if __name__ == "__main__":
    main()
