"""报告第 4 名（按预算分配搜索）的离线小分析：PIMC 搜过的根按原首选概率分档，看各档的搜索次数、改选比例、老师自评收益（同一批世界，偏乐观）、平均耗时。"""
import glob, json, sys
import numpy as np
rows = []
for f in glob.glob(sys.argv[1]):
    for l in open(f):
        r = json.loads(l)
        p = max(r["p"]); g = r["greedy"]; b = r["best"]
        gain = r["diff"].get(str(b), [0.0])[0] if b != g else 0.0
        rows.append((p, int(b != g), gain, r["ms"], r["stage"]))
a = np.array(rows)
edges = [0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
print(f"共 {len(a)} 次搜索；改选 {a[:,1].mean():.3f}；老师自评收益合计 {a[:,2].sum():.1f}")
for lo, hi in zip(edges, edges[1:]):
    m = (a[:, 0] >= lo) & (a[:, 0] < hi)
    if m.any():
        print(f"首选概率 [{lo:.1f}, {hi:.1f})：搜索 {m.sum():5d}（{m.mean():.1%}）  改选 {a[m,1].mean():.3f}  自评收益占比 {a[m,2].sum()/a[:,2].sum():.1%}  平均 {a[m,3].mean()/1000:.1f} 秒")
for st in (0, 1):
    m = a[:, 4] == st
    print(f"第 {st+1} 个 Boss：搜索 {m.sum()}，改选 {a[m,1].mean():.3f}，自评收益占比 {a[m,2].sum()/a[:,2].sum():.1%}")
