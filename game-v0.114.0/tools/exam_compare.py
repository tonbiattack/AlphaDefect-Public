"""对照线与主线的固定考卷按题配对比较（考卷见 tools/exam_build.py，结果在各运行目录的 exam.jsonl）。

两条线各取版本在 [起, 止] 内的考卷记录，每道题算出各自的通过比例（同一条线考了几次就平均几次），
再按题取差（对照 − 主线），报告各部分的平均差与题目间的标准误；另给出只考一次时可用的不一致题数（对照过、主线没过 / 反过来）。
题目顺序与 exam.json 一致，第二幕 Boss、第三幕 Boss 各一组（exam.jsonl 的 act2、act3 字段给出每组题数）。

    python3 tools/exam_compare.py --from 40073            # 默认对照为 defect（A），主线为 defect-b（B），止于两边都考过的最新版本
    python3 tools/exam_compare.py --from 40073 --to 41572 --a defect --b defect-b
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

REMOTE = Path.home() / ".local/share/sls2/runs/remote"


def load(d: str) -> list[dict]:
    p = REMOTE / d / "exam.jsonl"
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()] if p.exists() else []


def per_item(rows: list[dict]) -> list[float | None]:
    """每道题的通过比例（"-" 为没考成，不计）。"""
    n = len(rows[0]["res"])
    out: list[float | None] = []
    for k in range(n):
        xs = [r["res"][k] for r in rows if k < len(r["res"]) and r["res"][k] in "01"]
        out.append(sum(x == "1" for x in xs) / len(xs) if xs else None)
    return out


def compare(a_rows: list[dict], b_rows: list[dict]) -> dict:
    groups = {}
    n_act2 = max(r["act2"][1] for r in a_rows + b_rows)     # 题组边界：前 n_act2 题是第二幕 Boss，其余是第三幕 Boss
    n_all = len(a_rows[0]["res"])
    pa, pb = per_item(a_rows), per_item(b_rows)
    for name, lo, hi in [("第二幕 Boss", 0, n_act2), ("第三幕 Boss", n_act2, n_all), ("合计", 0, n_all)]:
        d = [(x, y) for x, y in zip(pa[lo:hi], pb[lo:hi]) if x is not None and y is not None]
        if not d:
            continue
        diffs = [x - y for x, y in d]
        m = sum(diffs) / len(diffs)
        sd = math.sqrt(sum((v - m) ** 2 for v in diffs) / max(len(diffs) - 1, 1))
        groups[name] = dict(n=len(d), a=sum(x for x, _ in d) / len(d), b=sum(y for _, y in d) / len(d), diff=m, se=sd / math.sqrt(len(d)),
                            a_only=sum(1 for x, y in d if x > y), b_only=sum(1 for x, y in d if y > x))
    return groups


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="lo", type=int, required=True, help="起始版本（含）")
    ap.add_argument("--to", dest="hi", type=int, default=None, help="截止版本（含；默认两边都考过的最新版本）")
    ap.add_argument("--a", default="defect", help="对照线的拉取目录")
    ap.add_argument("--b", default="defect-b", help="主线的拉取目录")
    a = ap.parse_args()
    A, B = load(a.a), load(a.b)
    hi = a.hi if a.hi is not None else min(max((r["version"] for r in A), default=0), max((r["version"] for r in B), default=0))
    A = [r for r in A if a.lo <= r["version"] <= hi]
    B = [r for r in B if a.lo <= r["version"] <= hi]
    if not A or not B:
        print(f"第 {a.lo}–{hi} 版里考卷记录不够：对照 {len(A)} 次，主线 {len(B)} 次")
        return
    print(f"考卷配对（第 {a.lo:,}–{hi:,} 版）：对照 {a.a} {len(A)} 次（{', '.join(f'{r['version']:,}' for r in A)}），"
          f"主线 {a.b} {len(B)} 次（{', '.join(f'{r['version']:,}' for r in B)}）；平均版本差（对照 − 主线）"
          f"{sum(r['version'] for r in A) / len(A) - sum(r['version'] for r in B) / len(B):+.0f}")
    for name, g in compare(A, B).items():
        print(f"  {name}（{g['n']} 题）：对照 {g['a'] * 100:.1f}%，主线 {g['b'] * 100:.1f}%，差 {g['diff'] * 100:+.1f} ± {g['se'] * 100:.1f} 个百分点；"
              f"对照更好 {g['a_only']} 题、主线更好 {g['b_only']} 题")


if __name__ == "__main__":
    main()
