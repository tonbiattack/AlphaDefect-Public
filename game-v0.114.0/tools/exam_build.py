"""固定考卷：冻结第二幕、第三幕 Boss 战前局面各 256 个，写成 exam.json，传到主线租用机的运行目录。评测每 exam_every 版用贪心
把考卷打一遍（evaluator），局面和种子固定，可以按局面配对比较不同版本；求解器练习池（solver_pool）读到 exam.json 后不再用这些局面出题。

- 第 1 版（2026-10-26）：取本机求解器存档（labeled.jsonl）里各幕最近的 256 个。
- 第 2 版（2026-10-27，用户要求重做）：第 1 版的牌组是当时的策略攒的，比现在的弱（考卷第二幕 Boss 约 48%，评测里约 80%），
  而且第二幕的快照因为求解器优先解第三幕，已标注的最近一批也是一天多以前的。改为从已标注存档和本机待标注收件箱里
  取最近 --hours 小时内导出的快照，按导出时刻均匀抽各 256 个（不要求标注过；solver_win 为 null 表示还没解）。

  python3 tools/exam_build.py [输出路径] [--hours 6]
"""
import argparse
import json
from pathlib import Path

SOLVER = Path.home() / ".local/share/sls2/solver"
N = 256


def t_of(rid: str) -> int:
    """快照编号 r<采集进程>_<导出时刻>_<序号> 里的导出时刻。"""
    return int(rid.split("_")[1])


ap = argparse.ArgumentParser()
ap.add_argument("out", nargs="?", default="exam.json")
ap.add_argument("--hours", type=float, default=6.0)
a = ap.parse_args()
pool: dict[str, dict] = {}
for l in open(SOLVER / "labeled.jsonl", encoding="utf-8"):
    if l.strip():
        r = json.loads(l)
        if r.get("spec", {}).get("player_snapshot"):
            pool[r["id"]] = {"id": r["id"], "act": r["act"], "encounter": r["encounter"], "solver_win": r.get("win"), "spec": r["spec"]}
for f in (SOLVER / "inbox").glob("*.json"):
    try:
        r = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        continue
    if r["id"] not in pool and r.get("spec", {}).get("player_snapshot"):
        pool[r["id"]] = {"id": r["id"], "act": r["label"]["act"], "encounter": r["label"]["encounter"], "solver_win": None, "spec": r["spec"]}
newest = max(t_of(i) for i in pool)
items = []
for act in (1, 2):
    cand = sorted((x for x in pool.values() if x["act"] == act and newest - t_of(x["id"]) <= a.hours * 3600), key=lambda x: t_of(x["id"]))
    if len(cand) < N:
        raise SystemExit(f"第{'一二三'[act]}幕最近 {a.hours} 小时只有 {len(cand)} 份快照，不够 {N} 份")
    pick = [cand[round(k * (len(cand) - 1) / (N - 1))] for k in range(N)]     # 按导出时刻均匀抽
    for x in pick:
        spec = dict(x["spec"]); spec.update(room="Boss")
        items.append(dict(x, spec=spec))
out = Path(a.out)
out.write_text(json.dumps({"version": 2, "built": newest, "hours": a.hours, "items": items}, ensure_ascii=False), encoding="utf-8")
print(f"考卷 {len(items)} 题，写到 {out}：" + "，".join(
    f"第{'一二三'[act]}幕 {sum(1 for x in items if x['act'] == act)} 题（已标注 {sum(1 for x in items if x['act'] == act and x['solver_win'] is not None)}，"
    f"其中求解器能赢 {sum(1 for x in items if x['act'] == act and x['solver_win'])}）" for act in (1, 2)))
