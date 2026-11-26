"""成套偏离测试（决策记录 265）的汇总：/root/diag/DEV_<流派>/ 与基线 DEV_base 按种子配对。

每组：局数、通关率与 95% 区间、平均层数、每局选卡奖励里拿到的本流派牌数（runs 的 cards 字段：各牌在三幕的提供与选中次数，
后三项为选中）、与基线配对（只这组赢、只基线赢、双侧二项检验 p、配对差）、偏置命中与改了选择的次数（各分片日志最后一行）。

    python3 tools/deviation_report.py [--dir /root/diag] [--prefix DEV_]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import re
from pathlib import Path

ARCH = json.loads(Path(__file__).with_name("defect_archetypes.json").read_text(encoding="utf-8"))


def load(d: str) -> dict[str, dict]:
    out = {}
    for f in glob.glob(f"{d}/runs_s*.jsonl"):
        for line in open(f, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                out[r["seed"]] = r
    return out


def picks(r: dict, cards: set[str]) -> int:
    n = 0
    for k, v in (r.get("cards") or {}).items():
        if k in cards and isinstance(v, list) and len(v) >= 6:
            n += sum(v[3:6])
    return n


def bias_counts(d: str) -> tuple[int, int]:
    hit = chg = 0
    for f in glob.glob(f"{d}/log_s*.txt"):
        m = re.findall(r"偏离：命中 (\d+) 次、改了选择 (\d+) 次", open(f, encoding="utf-8", errors="ignore").read())
        if m:
            hit += int(m[-1][0]); chg += int(m[-1][1])
    return hit, chg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/root/diag")
    ap.add_argument("--prefix", default="DEV_")
    a = ap.parse_args()
    base = load(f"{a.dir}/{a.prefix}base")
    print(f"基线：{len(base)} 局，通关 {sum(r['win'] for r in base.values()) / max(1, len(base)):.1%}")
    rows = []
    for k, v in ARCH.items():
        if k.startswith("_"):
            continue
        d = f"{a.dir}/{a.prefix}{k}"
        runs = load(d)
        if not runs:
            continue
        cards = set(v["cards"])
        n = len(runs); p = sum(r["win"] for r in runs.values()) / n
        common = [s for s in runs if s in base]
        x = sum(runs[s]["win"] and not base[s]["win"] for s in common)
        y = sum(base[s]["win"] and not runs[s]["win"] for s in common)
        kk, m = min(x, y), x + y
        pv = min(1.0, 2 * sum(math.comb(m, i) for i in range(kk + 1)) / 2 ** m) if m else 1.0
        diff = (sum(runs[s]["win"] for s in common) - sum(base[s]["win"] for s in common)) / max(1, len(common))
        pk = sum(picks(r, cards) for r in runs.values()) / n
        pkb = sum(picks(base[s], cards) for s in common) / max(1, len(common))
        fl = sum(r.get("floor", 0) for r in runs.values()) / n
        hit, chg = bias_counts(d)
        rows.append((k, v["名字"], n, p, 1.96 * math.sqrt(p * (1 - p) / n), fl, pkb, pk, x, y, pv, diff, hit, chg))
    print("| 流派 | 局数 | 通关率 | 平均层数 | 每局拿本流派牌（基线 → 偏置） | 只偏置赢 / 只基线赢 | p | 配对差 | 偏置命中 / 改了选择 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for k, name, n, p, ci, fl, pkb, pk, x, y, pv, diff, hit, chg in sorted(rows, key=lambda r: -r[11]):
        print(f"| {name}（{k}） | {n} | {p:.1%} ± {ci:.1%} | {fl:.2f} | {pkb:.2f} → {pk:.2f} | {x} / {y} | {pv:.3f} | {diff:+.1%} | {hit} / {chg} |")


if __name__ == "__main__":
    main()
