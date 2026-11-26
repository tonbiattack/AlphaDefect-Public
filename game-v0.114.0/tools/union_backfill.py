"""并集补测的选题与判定（2026-11-02 起，决策记录 248）：tools/remote/union_backfill.sh 在定时复测的空档里调它。
用户：“在复测间隔阶段，从近往前用过往强快照去做新种子池复测，只要对并集有提升，就纳入记录。”

候选（默认 --mode recent，11-02 用户：“A 上加测不要再测过去版本，而是有加测窗口时测最近快照”）：A 上常规评测最新留存的权重
（eval_runs/，只留最近 2 轮）里，比已在种子池上测过的版本（定时复测、补测）都新的那一版；沿用评测已打的 384 局。
原来的从近往前补过往强快照改为 --mode past：
观测格式 19 上线（第 130,420 版）以后的过往强快照，从近往前——
  - 以前的定时复测（retest/state.json，RT 种子）通关不低于 73% 的，用本机结果目录里的 model.pt（LOCAL）；
  - A 上常规评测留档的前几名（best/）与偏高留档（high/）：种子池口径的前几名全收，RT 口径的通关不低于 75% 才收（EVAL）。
  已在种子池上定时复测过、已补测过、已判过的版本跳过；同一版本两处都有时用本机的。
判定：这个快照的逐种子结果并进“定时复测 + 已纳入的补测”之后，至少一个通关的种子数有没有增加；增加就纳入
（retest/union_backfill.json 的 accepted），否则记 rejected。没打满的记 failed，最多重试一次。

    python3 tools/union_backfill.py next --best best_index.json --high high_index.json     # 打印 "LOCAL 路径 版本" 或 "EVAL 版本"，没有了打印空行
    python3 tools/union_backfill.py judge 版本 结果目录 [--gap 上一次定时复测的版本]
    python3 tools/union_backfill.py gapcount 上一次定时复测的版本                                # 这个空档里已经补测了几个
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

AT = Path.home() / ".local/share/sls2/ability_test"
REG = AT / "retest/union_backfill.json"
MIN_V = 130420          # 观测格式 19 上线后第一版：更早的检查点在现在的引擎上会看到没训练过的记录（决策记录 219）
POOL = "P1"


def reg() -> dict:
    d = json.loads(REG.read_text(encoding="utf-8")) if REG.exists() else {}
    for k in ("accepted", "rejected", "failed"):
        d.setdefault(k, [])
    return d


def wins(d: str | Path) -> dict[str, int]:
    w = {}
    for f in Path(d).glob("runs_s*.jsonl"):
        for line in open(f, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                w[r["seed"]] = int(r["win"])
    return w


def sched_runs() -> list[dict]:
    return json.loads((AT / "retest/state.json").read_text(encoding="utf-8"))["runs"]


def cmd_next(a) -> None:
    r = reg()
    runs = sched_runs()
    if a.mode == "recent":
        tested = [x["version"] for x in runs if x.get("seed_pool") == POOL] + [x["version"] for x in r["accepted"] + r["rejected"] + r["failed"]]
        top = max(tested, default=0)
        avail = []
        if a.eval_runs and Path(a.eval_runs).exists():
            for line in Path(a.eval_runs).read_text(encoding="utf-8").split():
                if line.startswith("v") and line.endswith(".pt") and line[1:-3].isdigit():
                    avail.append(int(line[1:-3]))
        new = [v for v in avail if v > top]
        print(f"EVAL {max(new)}" if new else "")
        return
    skip = {x["version"] for x in runs if x.get("seed_pool") == POOL}
    skip |= {x["version"] for x in r["accepted"] + r["rejected"]}
    tries: dict[int, int] = {}
    for x in r["failed"]:
        tries[x["version"]] = tries.get(x["version"], 0) + 1
    skip |= {v for v, n in tries.items() if n >= 2}
    cand: dict[int, tuple[str, str]] = {}
    for idx, key, thr in ((a.best, "best", 0.75), (a.high, "high", 0.75)):
        if not idx or not Path(idx).exists():
            continue
        try:
            items = json.loads(Path(idx).read_text(encoding="utf-8")).get(key, [])
        except ValueError:
            continue
        for x in items:
            v = int(x["version"])
            pool_ok = POOL in str(x.get("seed_set", ""))
            if v >= MIN_V and (pool_ok or x.get("win_rate", 0) >= thr):
                cand[v] = ("EVAL", str(v))
    for x in runs:
        v = x["version"]
        ck = Path(x["dir"]) / "model.pt"
        if x.get("seed_pool", "RT") == "RT" and v >= MIN_V and x["win_rate"] >= 0.73 and ck.exists():
            cand[v] = ("LOCAL", str(ck))
    for v in sorted(cand, reverse=True):
        if v not in skip:
            kind, arg = cand[v]
            print(f"LOCAL {arg} {v}" if kind == "LOCAL" else f"EVAL {v}")
            return
    print("")


def cmd_judge(a) -> None:
    r = reg()
    w = wins(a.dir)
    rec = {"version": a.version, "dir": a.dir, "time": round(time.time(), 1), "gap": a.gap, "seeds": len(w)}
    if len(w) < 1000:
        r["failed"].append(rec)
        REG.write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  并集补测：第 {a.version:,} 版只打完 {len(w)} 局，记为失败")
        return
    base: list[dict[str, int]] = []
    seen = set()
    for x in sched_runs():
        if x.get("seed_pool") == POOL and x["version"] not in seen:
            base.append(wins(x["dir"])); seen.add(x["version"])
    base += [wins(x["dir"]) for x in r["accepted"]]
    seeds = set(w).intersection(*(set(b) for b in base)) if base else set(w)
    before = sum(any(b[s] for b in base) for s in seeds)
    after = sum(any(b[s] for b in base) or w[s] for s in seeds)
    rec.update({"win_rate": round(sum(w[s] for s in seeds) / max(len(seeds), 1), 4), "union_before": before, "union_after": after,
                "gain": after - before, "base_snapshots": len(base)})
    (r["accepted"] if after > before else r["rejected"]).append(rec)
    REG.write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    verdict = "纳入记录" if after > before else "没有提升，不纳入"
    print(f"  并集补测：第 {a.version:,} 版单独通关 {rec['win_rate']:.1%}；并集 {before} → {after}（+{after - before}，"
          f"在 {len(base)} 个快照之上），{verdict}")


def cmd_gapcount(a) -> None:
    r = reg()
    print(sum(1 for x in r["accepted"] + r["rejected"] + r["failed"] if x.get("gap") == a.gap))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("next"); n.add_argument("--best"); n.add_argument("--high"); n.add_argument("--eval-runs", help="A 上 eval_runs/ 的文件列表")
    n.add_argument("--mode", default="recent", choices=["recent", "past"])
    j = sub.add_parser("judge"); j.add_argument("version", type=int); j.add_argument("dir"); j.add_argument("--gap", type=int, default=0)
    g = sub.add_parser("gapcount"); g.add_argument("gap", type=int)
    a = ap.parse_args()
    {"next": cmd_next, "judge": cmd_judge, "gapcount": cmd_gapcount}[a.cmd](a)


if __name__ == "__main__":
    main()
