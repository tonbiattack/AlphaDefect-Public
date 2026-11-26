# 评测回放里进第三幕第二个 Boss 时的血量与胜负（只读；铁甲战士决策记录第 25 条的基线与读数）。在 B 上跑：
#   ssh B "nice -n 10 python3 - <版本下限> <版本上限>" < tools/remote/ironclad/diag/boss3b_hp.py
import glob, gzip, json, os, sys
lo, hi = int(sys.argv[1]), int(sys.argv[2])
D = "/root/.local/share/sls2/runs/ironclad-eval/eval_replays"
rows = []
for f in glob.glob(f"{D}/v*_*.jsonl.gz"):
    v = int(os.path.basename(f)[1:].split("_")[0])
    if not (lo <= v <= hi):
        continue
    with gzip.open(f, "rt") as fh:
        r = json.loads(fh.readline())
    b3 = [c for c in r.get("combats") or [] if c.get("room") == "Boss" and c.get("act") == 2]
    if len(b3) >= 2:
        c = b3[1]
        rows.append((c["hp"] / max(1, c["max_hp"]), c["hp"], bool(r.get("win"))))
n = len(rows)
if not n:
    print("没有"); raise SystemExit
print(f"第 {lo}–{hi} 版：进第二个 Boss {n} 场，平均血量 {sum(x[1] for x in rows) / n:.1f}，平均比例 {sum(x[0] for x in rows) / n:.3f}，第二个 Boss 胜率 {sum(x[2] for x in rows) / n:.3f}")
for a, b in ((0, .3), (.3, .5), (.5, .7), (.7, .9), (.9, 1.01)):
    w = [x for x in rows if a <= x[0] < b]
    if w:
        print(f"  比例 {a:.1f}–{min(b, 1):.1f}：{len(w):5d} 场（{len(w) / n:.1%}） 胜率 {sum(x[2] for x in w) / len(w):.3f}")
