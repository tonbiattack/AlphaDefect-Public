"""复测快照的通关并集（2026-11-02 起，决策记录 247）：同一种子池上历次定时复测的逐种子结果合在一起看，
有多少种子被至少一个快照通关、被全部快照通关、一个都没通关。允许读档重来的支线版本要用这个数（用户：“这个数据比较重要”）。

只用 retest/state.json 里同一种子池（seed_pool，默认 P1）的记录；同一版本记了两次的（如第 141,229 版的计划外基线）只算一次。
除了全部快照的累计并集，另给最近 4 个、最近 8 个快照的窗口并集：累计并集随快照数单调上升，窗口并集才能前后比较。
决策记录 248 起，累计并集（all）还算进并集补测纳入的过往快照（retest/union_backfill.json 的 accepted，tools/union_backfill.py）；
只算定时复测的累计并集另记为 sched，窗口并集只用定时复测。
结果写 retest/union.json（最新一次）并追加 retest/union.jsonl，打印一行摘要（retest.sh 记进 retest.log）。

    python3 tools/retest_union.py [--pool P1]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

AT = Path.home() / ".local/share/sls2/ability_test"
WINDOWS = (4, 8)


def load(d: str) -> dict[str, int]:
    w = {}
    for f in Path(d).glob("runs_s*.jsonl"):
        for line in open(f, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                w[r["seed"]] = int(r["win"])
    return w


def stats(snaps: list[tuple[int, dict[str, int]]]) -> dict:
    seeds = sorted(set.intersection(*(set(w) for _, w in snaps)))
    n = len(seeds)
    wins = [sum(w[s] for _, w in snaps) for s in seeds]
    k = len(snaps)
    return {"snapshots": k, "versions": [v for v, _ in snaps], "seeds": n,
            "union": round(sum(c > 0 for c in wins) / n, 4), "all": sum(c == k for c in wins), "none": sum(c == 0 for c in wins),
            "mean_single": round(sum(wins) / (n * k), 4), "by_count": [sum(c == i for c in wins) for i in range(k + 1)]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="P1")
    a = ap.parse_args()
    runs = json.loads((AT / "retest/state.json").read_text(encoding="utf-8"))["runs"]
    snaps, seen = [], set()
    for r in runs:
        if r.get("seed_pool", "RT") != a.pool or r["version"] in seen:
            continue
        w = load(r["dir"])
        if len(w) >= 1000:
            snaps.append((r["version"], w)); seen.add(r["version"])
    if not snaps:
        print(f"种子池 {a.pool} 还没有复测记录"); return
    sched = list(snaps)
    bf = AT / "retest/union_backfill.json"
    for x in (json.loads(bf.read_text(encoding="utf-8")).get("accepted", []) if bf.exists() else []):
        if x["version"] not in seen:
            snaps.append((x["version"], load(x["dir"]))); seen.add(x["version"])
    out = {"time": round(time.time(), 1), "pool": a.pool, "all": stats(snaps), "sched": stats(sched), "backfill": len(snaps) - len(sched)}
    # 首通探测（tools/unsolved_probe.py，决策记录 250）：只打全局未通关的种子，首通的种子算进累计并集（all），全部通关的口径只看打满的快照
    pr = AT / "retest/unsolved_probe.json"
    if pr.exists():
        p = json.loads(pr.read_text(encoding="utf-8"))
        seeds = set.intersection(*(set(w) for _, w in snaps))
        won = {x for x in seeds if any(w[x] for _, w in snaps)}
        extra = sorted({f["seed"] for f in p.get("firsts", [])} & seeds - won)
        a_ = out["all"]
        a_["union"] = round((len(won) + len(extra)) / len(seeds), 4); a_["none"] = len(seeds) - len(won) - len(extra)
        out["probe"] = {"probes": len(p.get("probes", [])), "firsts": p.get("firsts", [])[-10:], "first_n": len(p.get("firsts", [])), "extra": len(extra)}
    for k in WINDOWS:
        if len(sched) >= k:
            out[f"last{k}"] = stats(sched[-k:])
    (AT / "retest/union.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    with open(AT / "retest/union.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(out, ensure_ascii=False) + "\n")
    s = out["all"]
    q = out["sched"]
    parts = [f"种子池 {a.pool} 的 {s['snapshots']} 个快照（定时复测 {q['snapshots']} 个 + 并集补测 {out['backfill']} 个，第 {min(s['versions']):,}–{max(s['versions']):,} 版）："
             f"至少一个通关 {s['union']:.1%}，全部通关 {s['all']}，都没通关 {s['none']}，单个平均 {s['mean_single']:.1%}",
             f"只算定时复测：至少一个 {q['union']:.1%}"]
    if "probe" in out:
        parts.append(f"首通探测已测 {out['probe']['probes']} 个快照、首通 {out['probe']['first_n']} 个种子")
    for k in WINDOWS:
        if f"last{k}" in out:
            w = out[f"last{k}"]
            parts.append(f"最近 {k} 个：至少一个 {w['union']:.1%}、全部 {w['all']}、都没 {w['none']}")
    print("  并集：" + "；".join(parts))


if __name__ == "__main__":
    main()
