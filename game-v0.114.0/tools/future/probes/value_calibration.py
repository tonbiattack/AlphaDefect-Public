import json, glob, numpy as np, sys
d = sys.argv[1]
V, Y, ST, W, FR = [], [], [], [], []
for f in glob.glob(f"{d}/res_s*.jsonl"):
    for l in open(f):
        r = json.loads(l)
        tr = r.get("traj") or []
        if r.get("stuck") or not tr: continue
        win = bool(r["win"]); left_end = 0.0 if win else tr[-1][5]
        term = 1.5 if win else -1.0 + 0.25 * (tr[-1][6] + 1.0 - left_end)
        T = len(tr)
        for t, x in enumerate(tr):
            V.append(x[7]); Y.append(term); ST.append(x[6]); W.append(win); FR.append(t / T)
V, Y, ST, W, FR = map(np.array, (V, Y, ST, W, FR))
print(f"帧 {len(V)}；价值头均值 {V.mean():.3f}，实际终局得分均值 {Y.mean():.3f}，相关 {np.corrcoef(V, Y)[0,1]:.3f}")
for s in (0, 1):
    m = ST == s
    print(f" 第 {s+1} 个 Boss：帧 {m.sum()}，价值头均值 {V[m].mean():.3f}，实际 {Y[m].mean():.3f}，胜率 {W[m].mean():.3f}，相关 {np.corrcoef(V[m], Y[m])[0,1]:.3f}")
print(" 价值头分箱 → 实际终局得分均值、胜率、帧数")
qs = np.quantile(V, np.linspace(0, 1, 11))
for i in range(10):
    m = (V >= qs[i]) & (V <= qs[i+1])
    print(f"  [{qs[i]:+.2f}, {qs[i+1]:+.2f}]  实际 {Y[m].mean():+.3f}  胜率 {W[m].mean():.3f}  {m.sum()}")
for lo, hi in ((0, .1), (.45, .55), (.9, 1.01)):
    m = (FR >= lo) & (FR < hi)
    print(f" 局内进度 {lo:.2f}–{hi:.2f}：价值头 {V[m].mean():+.3f}，实际 {Y[m].mean():+.3f}，相关 {np.corrcoef(V[m], Y[m])[0,1]:.3f}")
