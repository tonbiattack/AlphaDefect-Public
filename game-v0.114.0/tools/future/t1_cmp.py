"""T1 配对比较：同一批题（同种子）贪心 vs PIMC，按题号配对；另按第一个/第二个 Boss 分开。"""
import glob, json, math, sys
def load(d):
    r = {}
    for f in glob.glob(f"{d}/res_s*.jsonl"):
        for l in open(f):
            if l.strip():
                x = json.loads(l); r[(x["id"], x.get("rep", 0))] = x
    return r
A, B = load(sys.argv[1]), load(sys.argv[2])
ks = sorted(set(A) & set(B))
n = len(ks); wa = sum(A[k]["win"] for k in ks); wb = sum(B[k]["win"] for k in ks)
ob = sum(1 for k in ks if B[k]["win"] and not A[k]["win"]); oa = sum(1 for k in ks if A[k]["win"] and not B[k]["win"])
s1a = sum(A[k]["stage2"] for k in ks); s1b = sum(B[k]["stage2"] for k in ks)
d = (wb - wa) / max(1, n); se = math.sqrt(max(1e-12, (ob + oa) / n - d * d) / max(1, n))
from math import comb
m = ob + oa; p = min(1.0, 2 * sum(comb(m, i) for i in range(0, min(ob, oa) + 1)) / 2 ** m) if m else 1.0
print(f"配对 {n} 题：{sys.argv[1].split('/')[-1]} 赢 {wa}（{wa/n:.1%}）  {sys.argv[2].split('/')[-1]} 赢 {wb}（{wb/n:.1%}）  差 {d*100:+.1f}pp ±{1.96*se*100:.1f}；只后者赢 {ob}、只前者赢 {oa}，符号检验 p≈{p:.3g}")
print(f"  打到第二个 Boss（第一个赢）：{s1a/n:.1%} → {s1b/n:.1%}；第二个 Boss 条件胜率 {wa/max(1,s1a):.1%} → {wb/max(1,s1b):.1%}")
sr = sum(B[k].get("searches", 0) for k in ks); ch = sum(B[k].get("changed", 0) for k in ks)
if sr: print(f"  搜索 {sr} 次、改选 {ch} 次（{ch/sr:.1%}）；平均每题 {sum(B[k]['s'] for k in ks)/n:.0f} 秒")
