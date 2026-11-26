"""战斗结果模型复测的两项补充统计（reports/战斗结果模型复测.md）：在 B 的 /root/cmbpred 下跑。
  1. 单张牌改动的误差比与自助法区间（旧辅助头，T1、T2 与合并），以及残差最大的局面；
  2. 药水预测的校准（两个新模型，D1 验证整局）。
  /root/silent-deploy/sls2/.venv/bin/python -I cmbpred_extra.py
"""
import base64, collections, json, math, sys
import numpy as np
sys.path.insert(0, "/root/cmbpred")
from cmbpred_report import _read
def load(truth_files, old_files, meta_file, kpred=8):
    truth = collections.defaultdict(dict); pred = collections.defaultdict(dict)
    for fn in truth_files:
        for r in _read(fn):
            if "error" not in r and r.get("win") is not None: truth[(r["sid"], r["var"].split(":")[0])][r["k"]] = r
    for fn in old_files:
        for r in _read(fn):
            if r.get("p") is not None and "error" not in r and r["k"] < kpred: pred[(r["sid"], r["var"].split(":")[0])][r["k"]] = r
    meta = {m["sid"]: m for m in json.load(open(meta_file))}
    return truth, pred, meta
def resid(truth, pred, meta, kind="upgrade"):
    out = []
    for (sid, var), d in truth.items():
        if var != kind: continue
        b = truth.get((sid, "base"), {}); ks = sorted(set(d) & set(b))
        P, PB = pred.get((sid, var), {}), pred.get((sid, "base"), {}); kp = sorted(set(P) & set(PB))
        if len(ks) < 4 or len(kp) < 2: continue
        dT = np.mean([d[k]["win"] - b[k]["win"] for k in ks]); dP = np.mean([P[k]["p"] - PB[k]["p"] for k in kp])
        nT = np.var([d[k]["win"] - b[k]["win"] for k in ks], ddof=1) / len(ks)
        out.append((dP - dT, dT, dP, nT, len(ks), sid, meta[sid]["room"], meta[sid]["act"], meta[sid]["encounter"]))
    return out
for tag, tf, of, mf in (("T1", [f"/root/cmbpred/cf_{i}.jsonl" for i in range(4)] + [f"/root/cmbpred/cfb_{i}.jsonl" for i in range(4)], ["/root/cmbpred/t1f_0.jsonl", "/root/cmbpred/t1f_1.jsonl"], "/root/cmbpred/cf_0.jsonl.meta.json"),
                        ("T2", [f"/root/cmbpred/t2_{i}.jsonl" for i in range(4)], [f"/root/cmbpred/t2_{i}.jsonl" for i in range(4)], "/root/cmbpred/t2_0.jsonl.meta.json")):
    truth, pred, meta = load(tf, of, mf)
    for kind in ("upgrade", "add_silent"):
        R = resid(truth, pred, meta, kind)
        r = np.array([x[0] for x in R]); nT = np.array([x[3] for x in R]); dT = np.array([x[1] for x in R])
        Ks = collections.Counter(x[4] for x in R)
        print(f"{tag} {kind} n{len(R)} K分布{dict(Ks)}  残差SD {r.std():.4f} 噪声均值SD {math.sqrt(nT.mean()):.4f} 真实变化SD {dT.std():.4f}  |残差|中位 {np.median(abs(r)):.4f} 90分位 {np.quantile(abs(r),0.9):.4f}")
        top = sorted(R, key=lambda x: -abs(x[0]))[:5]
        for x in top: print(f"    残差 {x[0]:+.3f} 真 {x[1]:+.3f} 预 {x[2]:+.3f} K{x[4]} 噪声SD {math.sqrt(x[3]):.3f} sid{x[5]} {x[7]}:{x[6]} {x[8]}")
    # 按 K 分组看 T1 的残差
    if tag == "T1":
        R = resid(truth, pred, meta, "upgrade")
        for K in sorted({x[4] for x in R}):
            rr = np.array([x[0] for x in R if x[4] == K]); nn = np.array([x[3] for x in R if x[4] == K])
            print(f"    T1 upgrade K={K}: n{len(rr)} 残差方差 {rr.var():.5f} 噪声 {nn.mean():.5f}")

print("\n==== 自助法（按局面重抽 2,000 次）：单张牌改动的误差比（扣噪声的预测误差离散 ÷ 扣噪声的真实变化离散）")
def ratio_stats(R):
    r = np.array([x[0] for x in R]); dT = np.array([x[1] for x in R]); nT = np.array([x[3] for x in R])
    ev = r.var(ddof=1) - nT.mean(); tv = dT.var(ddof=1) - nT.mean()
    return math.sqrt(max(ev, 0)) / math.sqrt(tv) if tv > 0 else float("nan")
rng = np.random.default_rng(0)
sets = {}
for tag, tf, of, mf in (("T1", [f"/root/cmbpred/cf_{i}.jsonl" for i in range(4)] + [f"/root/cmbpred/cfb_{i}.jsonl" for i in range(4)], ["/root/cmbpred/t1f_0.jsonl", "/root/cmbpred/t1f_1.jsonl"], "/root/cmbpred/cf_0.jsonl.meta.json"),
                        ("T2", [f"/root/cmbpred/t2_{i}.jsonl" for i in range(4)], [f"/root/cmbpred/t2_{i}.jsonl" for i in range(4)], "/root/cmbpred/t2_0.jsonl.meta.json")):
    truth, pred, meta = load(tf, of, mf)
    for kind in ("upgrade", "add_silent", "remove", "remove_nb", "add_curse"):
        sets[(tag, kind)] = resid(truth, pred, meta, kind)
for kind in ("upgrade", "add_silent", "remove", "remove_nb", "add_curse"):
    for tag in ("T1", "T2", "合并"):
        R = sets[("T1", kind)] + sets[("T2", kind)] if tag == "合并" else sets[(tag, kind)]
        est = ratio_stats(R)
        bs = [ratio_stats([R[i] for i in rng.integers(0, len(R), len(R))]) for _ in range(2000)]
        bs = np.array([b for b in bs if b == b])
        print(f"  {kind:10s} {tag:4s} n{len(R):4d} 误差比 {est:.2f}  95% 区间 [{np.quantile(bs, 0.025):.2f}, {np.quantile(bs, 0.975):.2f}]  噪声占残差方差 {np.mean([x[3] for x in R]) / np.var([x[0] for x in R], ddof=1):.0%}")


print("\n==== 药水预测的校准（D1 验证整局）")
import numpy as np
sys.argv = ["x"]
import combat_model as cm
from sts2env.protocol import parse
vi = cm.vocab_index()
lab = {}
for fn in [f"/root/cmbpred/d1_{s}.jsonl.gz" for s in (11, 12, 13, 14)]:
    for r in cm.read_lines(fn):
        if not cm.is_val(r["seed"]): continue
        for ci, c in enumerate(r["combats"]):
            if c.get("frame") and c.get("win") and c.get("snap1"):
                o = parse(base64.b64decode(c["frame"]))
                y = cm.potion_labels(o, cm.snap_potions(c["snap1"], vi))
                if y: lab[(r["seed"], ci)] = (y, c["room"], c["act"])
for M in ("frozen", "full"):
    P, Y, room = [], [], []
    for r in cm.read_lines(f"/root/cmbpred/p_d1_{M}.jsonl"):
        k = (r["seed"], r["ci"])
        if k in lab and len(r["pot"]) == len(lab[k][0]):
            P += r["pot"]; Y += lab[k][0]; room += [lab[k][1]] * len(r["pot"])
    P, Y = np.array(P), np.array(Y, float)
    from cmbpred_report import auc
    print(f"{M}: 药水 {len(P)} 瓶 实际用掉 {Y.mean():.3f} 预测 {P.mean():.3f} Brier {np.mean((P - Y) ** 2):.4f}（只猜均值 {Y.var():.4f}） AUC {auc(P, Y):.3f}")
    for lo, hi in ((0, .1), (.1, .3), (.3, .5), (.5, .7), (.7, .9), (.9, 1.01)):
        m = (P >= lo) & (P < hi)
        if m.sum(): print(f"    [{lo:.1f},{hi:.1f}) {m.sum():5d} 瓶 预测 {P[m].mean():.3f} 实际 {Y[m].mean():.3f}")
    for rm in ("Monster", "Elite", "Boss"):
        m = np.array([x == rm for x in room])
        if m.sum(): print(f"    {rm:8s} {m.sum():5d} 瓶 预测 {P[m].mean():.3f} 实际 {Y[m].mean():.3f} AUC {auc(P[m], Y[m]):.3f}")
