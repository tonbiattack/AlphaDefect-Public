"""配对复测（2026-10-28，决策记录 168）：几个冻结检查点在同一批种子上各打一遍（tools/ability_test.py --seed-prefix），只比所有模型都打完的种子。
每个模型：通关率（Wilson 95% 区间）、平均层数、进第三幕 Boss、打过第三幕第一个 Boss、进第二个 Boss 后通关的条件胜率；
两两配对：只有一方通关的局数与精确二项检验（McNemar）p 值，层数差的均值与 95% 区间。

    python3 tools/ability_compare.py 目录1 目录2 ... [--asc 10] [--json 输出.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from itertools import combinations
from pathlib import Path


def load(d: str, asc: int) -> tuple[int, dict]:
    runs, ver = {}, None
    for f in glob.glob(f"{d}/runs_s*.jsonl"):
        for l in open(f, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                if r["asc"] == asc:
                    runs[r["seed"]] = r; ver = r["version"]
    return ver, runs


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def binom_p(a: int, b: int) -> float:
    """只有一方通关的局数 a、b：双侧精确二项检验（p=0.5）。"""
    n = a + b
    if n == 0:
        return 1.0
    k = min(a, b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def act3_bosses(r: dict) -> int:
    return sum(1 for c in r.get("combats") or [] if c["act"] == 2 and c["room"] == "Boss")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--asc", type=int, default=10)
    ap.add_argument("--json")
    a = ap.parse_args()
    models = [load(d, a.asc) for d in a.dirs]
    seeds = sorted(set.intersection(*(set(m[1]) for m in models)))
    n = len(seeds)
    out = {"asc": a.asc, "seeds": n, "models": [], "pairs": []}
    print(f"进阶 {a.asc}，{len(models)} 个模型都打完的种子 {n} 个")
    for ver, runs in models:
        rs = [runs[s] for s in seeds]
        w = sum(r["win"] for r in rs); lo, hi = wilson(w, n)
        b = [act3_bosses(r) for r in rs]
        r1 = sum(x >= 1 for x in b); r2 = sum(x >= 2 for x in b)
        w2 = sum(r["win"] for r, x in zip(rs, b) if x >= 2)
        m = {"version": ver, "wins": w, "win_rate": w / max(n, 1), "ci": [lo, hi], "floor_mean": sum(r["floor"] for r in rs) / max(n, 1),
             "reach_a3_boss1": r1 / max(n, 1), "beat_a3_boss1": r2 / max(r1, 1), "win_given_boss2": w2 / max(r2, 1), "stuck": sum(r["stuck"] for r in rs)}
        out["models"].append(m)
        print(f"  第 {ver:>6} 版：通关 {w}/{n} = {m['win_rate']:.1%}（{lo:.1%}–{hi:.1%}），平均层数 {m['floor_mean']:.2f}，"
              f"进第三幕 Boss {m['reach_a3_boss1']:.1%}，打过第一个 {m['beat_a3_boss1']:.1%}，进第二个后通关 {m['win_given_boss2']:.1%}，卡住 {m['stuck']}")
    for (va, ra), (vb, rb) in combinations(models, 2):
        only_a = sum(ra[s]["win"] and not rb[s]["win"] for s in seeds); only_b = sum(rb[s]["win"] and not ra[s]["win"] for s in seeds)
        d = [ra[s]["floor"] - rb[s]["floor"] for s in seeds]
        mu = sum(d) / max(n, 1); sd = math.sqrt(sum((x - mu) ** 2 for x in d) / max(n - 1, 1)); se = sd / math.sqrt(max(n, 1))
        p = binom_p(only_a, only_b)
        out["pairs"].append({"a": va, "b": vb, "only_a": only_a, "only_b": only_b, "p": p, "floor_diff": mu, "floor_ci": [mu - 1.96 * se, mu + 1.96 * se]})
        print(f"  {va} 对 {vb}：只有前者通关 {only_a} 局、只有后者 {only_b} 局，p = {p:.3f}；层数差 {mu:+.2f}（{mu - 1.96 * se:+.2f} 至 {mu + 1.96 * se:+.2f}）")
    if a.json:
        Path(a.json).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
