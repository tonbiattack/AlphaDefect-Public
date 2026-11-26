"""第三幕双 Boss 专项的题目文件（决策记录 287）：把 ability_test 复测结果（runs_s*.jsonl 的 boss3 项）整理成训练端读的 jsonl。

每行 {"id", "encounter", "next", "snap", "w", "model_win"}：第三幕第一个 Boss 战前的玩家快照与这一局的两个 Boss；model_win 为出题模型在这一局里
是否连赢两场（只作记录）；w 为抽题权重（默认 1）。按种子去重，--exclude 里的种子（留出验收卷）不收。先写临时文件再改名，训练端按修改时间重读。

    python3 tools/specialist_problems.py 复测目录1 [复测目录2 ...] --out 题目.jsonl [--exclude 种子文件.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude", default=None)
    a = ap.parse_args()
    ex = set(json.loads(Path(a.exclude).read_text(encoding="utf-8"))["seeds"]) if a.exclude else set()
    seen, n_win, items = set(), 0, []
    for d in a.dirs:
        for f in sorted(glob.glob(os.path.join(d, "runs_s*.jsonl"))):
            for line in open(f, encoding="utf-8"):
                if not line.strip():
                    continue
                r = json.loads(line)
                b = r.get("boss3")
                if not b or not b.get("snap") or not b.get("next_encounter") or r["seed"] in ex or r["seed"] in seen:
                    continue
                seen.add(r["seed"])
                n_win += bool(r["win"])
                items.append({"id": r["seed"], "encounter": b["encounter"], "next": b["next_encounter"], "snap": b["snap"], "w": 1.0,
                              "model_win": bool(r["win"])})
    tmp = a.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    os.replace(tmp, a.out)
    print(f"{len(items)} 题（出题模型连赢 {n_win}，{n_win / max(1, len(items)):.1%}）→ {a.out}")


if __name__ == "__main__":
    main()
