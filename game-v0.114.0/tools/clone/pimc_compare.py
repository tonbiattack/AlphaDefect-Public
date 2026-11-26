"""对照几次 tools/future/boss_pimc.py run 的结果：每次搜索的耗时、引擎分项、搜索结果是否一致（复制版 PIMC 的核对与测速用）。

读每个输出目录里的 res_s*.jsonl（每题一行：胜负、生命、步数、搜索次数、改选次数）、labels_s*.jsonl（每次搜索一行：
根的步号 t、改选 best、贪心 greedy、各世界的读数 meas、耗时 ms）、engine_s*.log（引擎退出时的 [timing] 行，要 STS2_TIMING=1）。

  - 耗时：每次搜索的平均、中位、去掉每个分片第一次搜索后的平均（第一次含 JIT 与第一次拍快照的一次性开销）。
  - 引擎分项：[timing] 行按进程分块（关闭顺序：实机进程在前，推演进程在后），推演进程合计每项的次数、平均毫秒、总秒数；
    再折算成“每个世界”：reset、replay、restore、snapshot 等按世界数平均，step、encode 按推演帧数平均。
  - 一致性（每个目录都和第一个比）：逐题按搜索的先后对齐，直到第一处不同为止——同一个根（t 相同）且各世界读数 meas 逐项相同算“全同”，
    只改选相同算“改选相同”；第一处不同之后两边的对局已经分开，不再对齐。另比每题的胜负、生命、步数。

  python -I tools/clone/pimc_compare.py 默认目录 复制目录 [复制+异步目录 …] [--names 默认,复制,…]
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

TIMING = re.compile(r"^\[timing\]\s+(\S+)\s+(\d+)\s+次\s+平均\s+([\d.]+)\s+ms")


def load(d: Path) -> dict:
    res = {}
    for f in sorted(d.glob("res_s*.jsonl")):
        for l in open(f, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                res[(r["id"], r.get("rep", 0))] = r
    labs: dict = {}
    firsts = set()
    for f in sorted(d.glob("labels_s*.jsonl")):
        first = True
        for l in open(f, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                r.pop("frame", None)
                labs.setdefault(r["id"], []).append(r)
                if first:
                    firsts.add((r["id"], r["t"]))
                    first = False
    blocks = []
    for f in sorted(d.glob("engine_s*.log")):
        cur: dict = {}
        for l in open(f, encoding="utf-8", errors="replace"):
            m = TIMING.match(l.strip())
            if not m:
                continue
            k, n, avg = m.group(1), int(m.group(2)), float(m.group(3))
            if k in cur:
                blocks.append(cur)
                cur = {}
            cur[k] = (n, avg)
        if cur:
            blocks.append(cur)
    return {"res": res, "labs": labs, "firsts": firsts, "blocks": blocks}


def timing_summary(D: dict) -> dict:
    ms = [r["ms"] for rs in D["labs"].values() for r in rs]
    ms2 = [r["ms"] for rs in D["labs"].values() for r in rs if (r["id"], r["t"]) not in D["firsts"]]
    worlds = sum(len(r["meas"]) for rs in D["labs"].values() for r in rs)
    eng = {}
    for b in D["blocks"][1:]:          # 第一块是实机进程
        for k, (n, avg) in b.items():
            t = eng.get(k, (0, 0.0))
            eng[k] = (t[0] + n, t[1] + n * avg)
    return {"searches": len(ms), "mean": statistics.fmean(ms) if ms else 0, "median": statistics.median(ms) if ms else 0,
            "mean_wo_first": statistics.fmean(ms2) if ms2 else 0, "worlds": worlds, "engine": eng}


def compare(A: dict, B: dict) -> dict:
    out = {"aligned": 0, "same_meas": 0, "same_best": 0, "first_diff": [], "res_same": 0, "res_n": 0}
    for i, la in A["labs"].items():
        lb = B["labs"].get(i, [])
        for x, y in zip(la, lb):
            if x["t"] != y["t"]:
                out["first_diff"].append((i, x["t"], "根不同（之前的改选不同，对局已分开）"))
                break
            out["aligned"] += 1
            sm = x["meas"] == y["meas"]
            out["same_meas"] += int(sm)
            out["same_best"] += int(x["best"] == y["best"])
            if x["best"] != y["best"]:
                out["first_diff"].append((i, x["t"], f"改选不同 {x['best']} / {y['best']}"))
                break
            if not sm:
                out["first_diff"].append((i, x["t"], "改选相同、世界读数不同"))
                # 改选相同时对局不分开，继续对齐
    for k, r in A["res"].items():
        s = B["res"].get(k)
        if s is None:
            continue
        out["res_n"] += 1
        out["res_same"] += int((r["win"], r.get("hp"), r["steps"]) == (s["win"], s.get("hp"), s["steps"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--names", default="")
    a = ap.parse_args()
    names = a.names.split(",") if a.names else [Path(d).name for d in a.dirs]
    Ds = [load(Path(d)) for d in a.dirs]
    print("| 配置 | 题 | 胜 | 搜索次数 | 改选 | 每次搜索 ms（平均 / 中位 / 去掉首次） | 世界数 |")
    print("|---|---|---|---|---|---|---|")
    for n, D in zip(names, Ds):
        t = timing_summary(D)
        res = list(D["res"].values())
        print(f"| {n} | {len(res)} | {sum(r['win'] for r in res)} | {t['searches']} | {sum(r['changed'] for r in res)} | "
              f"{t['mean']:.0f} / {t['median']:.0f} / {t['mean_wo_first']:.0f} | {t['worlds']} |")
    print()
    print("推演进程的引擎分项（[timing]，次数 / 平均 ms / 合计 s；每个世界折算 = 合计 ÷ 世界数）：")
    for n, D in zip(names, Ds):
        t = timing_summary(D)
        w = max(1, t["worlds"])
        parts = []
        for k in ("reset", "restore", "replay", "snapshot", "auto", "step", "decision", "encode"):
            if k in t["engine"]:
                c, tot = t["engine"][k]
                parts.append(f"{k} {c} / {tot / max(1, c):.2f} / {tot / 1000:.1f}（每世界 {tot / w:.1f} ms）")
        print(f"  {n}：" + "；".join(parts))
        st = [r.get("stat") for r in D["res"].values() if r.get("stat")]
        if st:
            agg = {k: sum(s.get(k, 0) for s in st) for k in st[0]}
            print(f"      推演计数：{agg}（批 {agg.get('batches', 0)}，平均每批 {agg.get('rows', 0) / max(1, agg.get('batches', 1)):.2f} 行）")
    print()
    for n, D in zip(names[1:], Ds[1:]):
        c = compare(Ds[0], D)
        print(f"{n} 对 {names[0]}：对齐的搜索 {c['aligned']}，世界读数全同 {c['same_meas']}，改选相同 {c['same_best']}；"
              f"每题结果（胜负、生命、步数）相同 {c['res_same']}/{c['res_n']}")
        for i, t, why in c["first_diff"][:8]:
            print(f"    第 {i} 题第 {t} 步：{why}")


if __name__ == "__main__":
    main()
