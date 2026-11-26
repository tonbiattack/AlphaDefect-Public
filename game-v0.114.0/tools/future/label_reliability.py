"""树搜索标签的复测可靠性（2026-11-17）：同一批题用两组世界种子（--policy-seed 0 / 1）各搜一遍（tree_search --labels），
取两边都有、根帧逐字节相同的根（实机轨迹在第一次选法不同之前完全相同），比较两次搜索给出的“候选 − 贪心项”的 Q 差。
老师（PIMC，16 + 16 个世界）的同类读数：奇偶世界分半相关 0.28，折算全部世界 0.43（愚者决策记录第 25 条）；蒸馏学不进网络的根因就是这个。

  python3 tools/future/label_reliability.py 目录A 目录B [--min-visits 4]
"""
import argparse
import glob
import json

import numpy as np


def load(d: str) -> dict:
    out = {}
    for f in glob.glob(f"{d}/labels_s*.jsonl"):
        for l in open(f, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                out[(r["id"], r["t"])] = r
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("--min-visits", type=int, default=4)
    a = ap.parse_args()
    A, B = load(a.a), load(a.b)
    keys = [k for k in A if k in B and A[k]["frame"] == B[k]["frame"]]
    print(f"根：A {len(A)}，B {len(B)}，两边都有且根帧相同 {len(keys)}")
    xa, xb, same, chg_a, chg_b, both_chg_same = [], [], 0, 0, 0, 0
    for k in keys:
        ra, rb = A[k], B[k]
        g = str(ra["greedy"])
        qa, qb = ra.get("q", {}), rb.get("q", {})
        same += int(ra["best"] == rb["best"])
        ca, cb = ra["best"] != ra["greedy"], rb["best"] != rb["greedy"]
        chg_a += ca; chg_b += cb; both_chg_same += int(ca and cb and ra["best"] == rb["best"])
        if g not in qa or g not in qb:
            continue
        for c in qa:
            if c == g or c not in qb:
                continue
            if min(qa[c][1], qb[c][1], qa[g][1], qb[g][1]) < a.min_visits:
                continue
            xa.append(qa[c][0] - qa[g][0]); xb.append(qb[c][0] - qb[g][0])
    xa, xb = np.array(xa), np.array(xb)
    n = len(keys)
    print(f"最终动作相同 {same} / {n}（{same / max(1, n):.3f}）；改选 A {chg_a}、B {chg_b}，两边都改且改成同一个 {both_chg_same}"
          f"（占改选的 {2 * both_chg_same / max(1, chg_a + chg_b):.3f}）")
    if len(xa) > 2:
        r = np.corrcoef(xa, xb)[0, 1]
        print(f"候选 − 贪心项的 Q 差：{len(xa)} 对（两边各至少 {a.min_visits} 次模拟），复测相关 {r:.3f}；"
              f"均值 A {xa.mean():+.4f} B {xb.mean():+.4f}，标准差 A {xa.std():.4f} B {xb.std():.4f}；符号一致 {np.mean(np.sign(xa) == np.sign(xb)):.3f}")
        big = np.abs(xa) > 0.05
        if big.any():
            print(f"  A 里 |Q 差| > 0.05 的 {big.sum()} 对：复测相关 {np.corrcoef(xa[big], xb[big])[0, 1]:.3f}，符号一致 {np.mean(np.sign(xa[big]) == np.sign(xb[big])):.3f}")


if __name__ == "__main__":
    main()
