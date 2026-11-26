# 铁甲战士集成测试各组同种子逐局配对比较（2026-11-19，铁甲战士决策记录第 12 条；照 tools/remote/fool/finish/cmp_arms.py）：通关率、平均层数、到第三幕、卡住；两两“只该组赢 / 只对方赢”与符号检验 p
#   python3 cmp_arms.py /root/ens_silent M E S
import glob, json, math, sys
from pathlib import Path
T = Path(sys.argv[1]); arms = sys.argv[2:]
R = {}
for a in arms:
    d = {}
    for f in glob.glob(str(T / a / "runs_s*.jsonl")):
        for l in open(f):
            if l.strip():
                r = json.loads(l); d[r["seed"]] = r
    R[a] = d
def binom_p(k, n):     # 双侧符号检验
    if n == 0: return 1.0
    c = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * c)
for a in arms:
    d = R[a]; n = len(d)
    if not n: continue
    w = sum(r["win"] for r in d.values())
    print(f"{a}: {n} 局 通关 {w} = {w/n:.1%}，平均层数 {sum(r['floor'] for r in d.values())/n:.2f}，到第三幕 {sum(r.get('act', 0) >= 2 for r in d.values())/n:.1%}，卡住 {sum(bool(r.get('stuck')) for r in d.values())}")
for i, a in enumerate(arms):
    for b in arms[i + 1:]:
        s = sorted(set(R[a]) & set(R[b]))
        if not s: continue
        oa = sum(R[a][x]["win"] and not R[b][x]["win"] for x in s); ob = sum(R[b][x]["win"] and not R[a][x]["win"] for x in s)
        n = len(s); diff = (oa - ob) / n
        se = math.sqrt(max(oa + ob - (oa - ob) ** 2 / n, 0)) / n
        print(f"{a} 对 {b}（{n} 个共同种子）：{diff*100:+.1f} ± {1.96*se*100:.1f} 个百分点（只 {a} 赢 {oa} / 只 {b} 赢 {ob}），p = {binom_p(oa, oa + ob):.3f}")
