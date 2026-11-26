"""把 tools/sim_pimc_eval.py run 的结果目录拆成两组逐种子结果（2026-11-21，论文补充实验）：贪心主干与第三幕 PIMC，
各写一个 jsonl（每行 {"seed", "win"}），给 tools/paper_stats.py 配对用。

sim_pimc_eval 每个种子在 seeds_s*.jsonl 里一行（win 为贪心主干的结果）；打到第三幕第一个 Boss 的种子另在 items_s*.jsonl 里一行，
greedy.win / search.win 为两组各自是否打赢第三幕两个 Boss（即通关）。没有 items 行的种子在第三幕 Boss 之前就输了，两组都算输。
核对：seeds 行的 win 必须等于 items 行的 greedy.win（不等就报出来）。

    python3 tools/pimc_to_arms.py 结果目录 输出前缀      # 写 输出前缀_greedy.jsonl、输出前缀_search.jsonl
"""
from __future__ import annotations

import glob
import json
import os
import sys


def main() -> None:
    d, pre = sys.argv[1], sys.argv[2]
    seeds = {}
    for f in sorted(glob.glob(os.path.join(d, "seeds_s*.jsonl"))):
        for l in open(f, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                seeds[r["seed"]] = bool(r["win"])
    items = {}
    for f in sorted(glob.glob(os.path.join(d, "items_s*.jsonl"))):
        for l in open(f, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                items[r["seed"]] = r
    bad = [s for s, it in items.items() if s in seeds and bool(it["greedy"]["win"]) != seeds[s]]
    orphan = [s for s in items if s not in seeds]
    with open(pre + "_greedy.jsonl", "w", encoding="utf-8") as fg, open(pre + "_search.jsonl", "w", encoding="utf-8") as fs:
        for s, w in seeds.items():
            it = items.get(s)
            fg.write(json.dumps({"seed": s, "win": w}) + "\n")
            fs.write(json.dumps({"seed": s, "win": bool(it["search"]["win"]) if it else False,
                                 "reached_act3_boss": it is not None}) + "\n")
    sw = sum(bool(items[s]["search"]["win"]) for s in seeds if s in items)
    print(f"{d}: 种子 {len(seeds)}，到第三幕 Boss {sum(s in items for s in seeds)}，贪心主干通关 {sum(seeds.values())}，搜索通关 {sw}；"
          f"主干与 items 不一致 {len(bad)}，items 无主干行 {len(orphan)}")
    if bad or orphan:
        print("  不一致：", bad[:10], " 无主干行：", orphan[:10])


if __name__ == "__main__":
    main()
