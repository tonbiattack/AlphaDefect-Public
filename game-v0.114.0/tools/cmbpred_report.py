"""战斗结果预测检验的汇总（tools/cmbpred_check.py 的输出）。纯 Python 加 numpy，在 B 上跑：
  python3 cmbpred_report.py runs runs.jsonl            检验一（开战第一帧的校准与区分度）+ 检验三（方差账）
  python3 cmbpred_report.py cf cf.jsonl                检验二（改牌组、改血量后，预测变化对真实变化）
"""
from __future__ import annotations

import collections
import json
import math
import sys

import numpy as np


def auc(p: np.ndarray, y: np.ndarray) -> float:
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    # 并列取平均秩
    allv = np.concatenate([pos, neg])
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks); ranks = (sums / cnt)[inv]
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def strat_key(c: dict) -> str:
    r = c["room"] or "?"
    if r == "Boss" and c["act"] == 2:
        return f"幕{c['act'] + 1} Boss{'2' if c.get('second') else '1'}"
    return f"幕{c['act'] + 1} {r}"


def report_runs(path: str) -> None:
    runs = [json.loads(l) for fn in path.split(",") for l in open(fn, encoding="utf-8") if l.strip()]
    C = [dict(c, _run=r) for r in runs for c in r["combats"] if c.get("p") is not None and c.get("win") is not None]
    print(f"整局 {len(runs)}，通关 {np.mean([r['win'] for r in runs]):.3f}，战斗 {len(C)}（每局 {len(C) / len(runs):.1f}），权重第 {runs[0]['version']} 版")
    groups = collections.defaultdict(list)
    for c in C:
        groups[strat_key(c)].append(c)
        groups["全部"].append(c)
    print("\n== 检验一：开战第一帧的辅助头（p = 预测本场胜率，h = 预测到本场结束要掉的血 / 最大生命，死了按掉光算）")
    print(f"{'分层':10s} {'场数':>6s} {'实际胜':>7s} {'预测胜':>7s} {'Brier':>7s} {'只猜均值':>8s} {'技巧':>6s} {'AUC':>6s} | {'实际掉血':>8s} {'预测掉血':>8s} {'R²':>6s}")
    order = ["全部"] + sorted(k for k in groups if k != "全部")
    for k in order:
        g = groups[k]
        p = np.array([c["p"] for c in g]); y = np.array([c["win"] for c in g], float)
        h = np.array([c["h"] for c in g]); l = np.array([c["loss"] for c in g])
        br = float(np.mean((p - y) ** 2)); b0 = float(np.mean((y.mean() - y) ** 2))
        skill = 1 - br / b0 if b0 > 0 else float("nan")
        r2 = 1 - np.mean((h - l) ** 2) / max(np.var(l), 1e-12)
        print(f"{k:10s} {len(g):6d} {y.mean():7.3f} {p.mean():7.3f} {br:7.4f} {b0:8.4f} {skill:6.2f} {auc(p, y):6.3f} | {l.mean():8.3f} {h.mean():8.3f} {r2:6.2f}")
    # 校准表
    print("\n校准（全部战斗，按预测胜率分箱）")
    allc = groups["全部"]
    p = np.array([c["p"] for c in allc]); y = np.array([c["win"] for c in allc], float)
    for lo, hi in ((0, .3), (.3, .5), (.5, .7), (.7, .8), (.8, .9), (.9, .95), (.95, .98), (.98, .995), (.995, 1.01)):
        m = (p >= lo) & (p < hi)
        if m.sum():
            se = math.sqrt(max(y[m].mean() * (1 - y[m].mean()), 1e-9) / m.sum())
            print(f"  [{lo:.3f},{hi:.3f}) {m.sum():6d} 场 预测 {p[m].mean():.3f} 实际 {y[m].mean():.3f} ± {1.96 * se:.3f}")
    # 第三幕 Boss 按遭遇
    print("\n第三幕 Boss 按遭遇")
    eb = collections.defaultdict(list)
    for c in allc:
        if c["room"] == "Boss" and c["act"] == 2:
            eb[(c["encounter"], 2 if c.get("second") else 1)].append(c)
    for (e, pos), g in sorted(eb.items()):
        p = np.array([c["p"] for c in g]); y = np.array([c["win"] for c in g], float)
        print(f"  {e:22s} 第{pos}个 {len(g):4d} 场 实际 {y.mean():.3f} 预测 {p.mean():.3f} AUC {auc(p, y):.3f}")

    # 检验三：方差账（下界）。整局通关 Y 的方差 = 各处“信息揭晓”的跳变方差之和（鞅分解）。
    # 一场战斗的输赢揭晓：开战时 M = p·V⁺，打完跳到 V⁺ 或 0，跳变方差 p(1−p)·V⁺²；V⁺ 为赢下这场后整局通关的概率，
    # 用“活过第 f 层的整局里最后通关的比例”近似。不算血量起伏带来的那部分，所以是下界；p 用辅助头的预测（校准好时可用）。
    print("\n== 检验三：整局通关方差里，战斗输赢彩票占多少（下界，不含血量起伏）")
    Y = np.array([r["win"] for r in runs], float)
    vy = Y.mean() * (1 - Y.mean())
    alive = collections.Counter(); wins = collections.Counter()
    for r in runs:
        last = max((c["floor"] for c in r["combats"]), default=0)
        for f in range(0, last + 1):
            # 活过第 f 层：这一局在第 f 层之后还有动作（死在 f 层的不算）
            if f < r["floor"] or r["win"]:
                alive[f] += 1; wins[f] += int(r["win"])
    W = {f: wins[f] / alive[f] for f in alive if alive[f] >= 5}
    contrib = collections.defaultdict(float)
    tot = 0.0
    for r in runs:
        for c in r["combats"]:
            if c.get("p") is None:
                continue
            vplus = W.get(c["floor"], Y.mean())
            j = c["p"] * (1 - c["p"]) * vplus ** 2
            tot += j; contrib[strat_key(c)] += j
    share = tot / len(runs) / vy
    print(f"  整局通关 {Y.mean():.3f}，方差 {vy:.4f}；战斗输赢彩票的跳变方差合计每局 {tot / len(runs):.4f}，占 {share:.1%}")
    for k, v in sorted(contrib.items(), key=lambda x: -x[1])[:8]:
        print(f"    {k:10s} {v / len(runs):.4f}（{v / len(runs) / vy:.1%}）")
    # 用实际结果直接算的对照（不依赖 p 的校准）：每场的 (y − p)² 之和
    tot2 = sum((c["win"] - c["p"]) ** 2 * W.get(c["floor"], Y.mean()) ** 2 for r in runs for c in r["combats"] if c.get("p") is not None)
    print(f"  对照：用实际输赢算的残差平方（含预测误差）每局 {tot2 / len(runs):.4f}，占 {tot2 / len(runs) / vy:.1%}")


def report_cf(path: str) -> None:
    files = path.split(",")
    rows = []
    for fn in files:
        for l in open(fn, encoding="utf-8"):
            try:
                rows.append(json.loads(l))
            except json.JSONDecodeError:
                pass      # 进程被停时写了半行
    err = [r for r in rows if "error" in r]
    rows = [r for r in rows if "error" not in r and r.get("p") is not None]
    meta = {m["sid"]: m for m in json.load(open(files[0] + ".meta.json", encoding="utf-8"))}
    print(f"单场战斗 {len(rows)}（引擎报错 {len(err)}，卡住 {sum(r['stuck'] for r in rows)}），局面 {len(meta)}")
    by = collections.defaultdict(dict)
    for r in rows:
        by[(r["sid"], r["var"])][r["k"]] = r
    # 原样：按快照重开的战斗与整局里那一场的预测对比；原样的预测对 K 场平均的真实结果
    print("\n== 原样：按快照重开（每个局面 K 场）")
    print(f"{'分层':10s} {'局面':>4s} {'整局里p':>8s} {'重开p':>7s} {'重开实际胜':>9s} {'重开h':>7s} {'重开实际掉血':>11s}")
    st = collections.defaultdict(list)
    for (sid, var), d in by.items():
        if var == "base" and d:
            m = meta[sid]
            st[strat_key(m)].append((m["run_p"], np.mean([x["p"] for x in d.values()]), np.mean([x["win"] for x in d.values()]),
                                     np.mean([x["h"] for x in d.values()]), np.mean([x["loss"] for x in d.values()])))
    for k in sorted(st):
        a = np.array(st[k])
        print(f"{k:10s} {len(a):4d} {a[:, 0].mean():8.3f} {a[:, 1].mean():7.3f} {a[:, 2].mean():9.3f} {a[:, 3].mean():7.3f} {a[:, 4].mean():11.3f}")
    # 逐局面的误差（预测减真实），扣掉 K 场平均的抽样噪声后的离散：开战前（抽牌之前）的预测能不能认准“这个局面”
    print("\n  逐局面：K 场平均的预测减 K 场平均的实际（扣抽样噪声）")
    per = collections.defaultdict(list)
    for (sid, var), d in by.items():
        if var != "base" or len(d) < 4:
            continue
        w = np.array([x["win"] for x in d.values()], float); p = np.array([x["p"] for x in d.values()])
        l = np.array([x["loss"] for x in d.values()]); h = np.array([x["h"] for x in d.values()])
        K = len(w)
        e = (p - w); eL = (h - l)
        per[strat_key(meta[sid])].append((e.mean(), e.var(ddof=1) / K, eL.mean(), eL.var(ddof=1) / K, w.mean(), l.mean()))
        per["全部"].append(per[strat_key(meta[sid])][-1])
    for k in ["全部"] + sorted(x for x in per if x != "全部"):
        a = np.array(per[k])
        sd = math.sqrt(max(0.0, a[:, 0].var(ddof=1) - a[:, 1].mean())) if len(a) > 2 else float("nan")
        sdL = math.sqrt(max(0.0, a[:, 2].var(ddof=1) - a[:, 3].mean())) if len(a) > 2 else float("nan")
        sdT = math.sqrt(max(0.0, a[:, 4].var(ddof=1) - np.mean(a[:, 4] * (1 - a[:, 4]) / 48))) if len(a) > 2 else float("nan")
        print(f"    {k:10s} {len(a):4d} 个 胜率：平均误差 {a[:, 0].mean():+.4f} 误差离散 {sd:.4f}（局面间真实胜率离散 {sdT:.4f}）  "
              f"掉血：平均误差 {a[:, 2].mean():+.4f} 误差离散 {sdL:.4f}")
    a = np.array([x for v in st.values() for x in v])
    if len(a) > 2:
        sl = np.polyfit(a[:, 1], a[:, 2], 1)[0]
        print(f"  全部 {len(a)} 个局面：预测胜 {a[:, 1].mean():.3f} 实际胜 {a[:, 2].mean():.3f}；局面间 实际对预测的回归斜率 {sl:.2f}，"
              f"相关 {np.corrcoef(a[:, 1], a[:, 2])[0, 1]:.3f}；掉血 预测 {a[:, 3].mean():.3f} 实际 {a[:, 4].mean():.3f} 相关 {np.corrcoef(a[:, 3], a[:, 4])[0, 1]:.3f}")

    # 变体减原样：同一种子（共同随机数）逐场配对
    print("\n== 检验二：变体减原样（同种子配对）。Δ真 = 实际的变化，Δ预 = 开战第一帧预测的变化；单位：胜率为概率，掉血为最大生命的比例")
    kinds = collections.defaultdict(list)
    for (sid, var), d in by.items():
        if var == "base":
            continue
        b = by.get((sid, "base"), {})
        ks = sorted(set(d) & set(b))
        if len(ks) < 4:
            continue
        dw = np.array([d[k]["win"] - b[k]["win"] for k in ks], float)
        dp = np.array([d[k]["p"] - b[k]["p"] for k in ks])
        dl = np.array([d[k]["loss"] - b[k]["loss"] for k in ks])
        dh = np.array([d[k]["h"] - b[k]["h"] for k in ks])
        kind = var.split(":")[0]
        kinds[kind].append({"sid": sid, "var": var, "K": len(ks), "dT": dw.mean(), "dP": dp.mean(), "seT2": dw.var(ddof=1) / len(ks),
                            "r": (dp - dw).mean(), "ser2": (dp - dw).var(ddof=1) / len(ks),
                            "dL": dl.mean(), "dH": dh.mean(), "seL2": dl.var(ddof=1) / len(ks),
                            "rL": (dh - dl).mean(), "serL2": (dh - dl).var(ddof=1) / len(ks), "room": strat_key(meta[sid])})
    kinds["全部改牌组"] = [x for k, v in kinds.items() if k in ("upgrade", "remove", "remove_nb", "add_silent", "add_curse") for x in v]
    for kind in ("upgrade", "remove", "remove_nb", "add_silent", "add_curse", "全部改牌组", "hp-15%", "hp+15%"):
        xs = kinds.get(kind, [])
        if len(xs) < 3:
            continue
        for tag, T, P, seT2, R, seR2 in (("胜率", "dT", "dP", "seT2", "r", "ser2"), ("掉血", "dL", "dH", "seL2", "rL", "serL2")):
            t = np.array([x[T] for x in xs]); p = np.array([x[P] for x in xs]); r = np.array([x[R] for x in xs])
            noiseT = np.mean([x[seT2] for x in xs]); noiseR = np.mean([x[seR2] for x in xs])
            sig = math.sqrt(max(0.0, t.var(ddof=1) - noiseT))          # 真实变化在局面之间的标准差（扣掉抽样噪声）
            bias_sd = math.sqrt(max(0.0, r.var(ddof=1) - noiseR))      # 预测误差在局面之间的标准差（扣掉抽样噪声）
            se_mean = math.sqrt(r.var(ddof=1) / len(r))
            slope = np.cov(p, t)[0, 1] / max(np.var(p, ddof=1), 1e-12)
            corr = np.corrcoef(p, t)[0, 1] if np.std(p) > 0 and np.std(t) > 0 else float("nan")
            # 去掉抽样噪声后的相关（真实变化的方差扣噪声）
            corr_adj = np.cov(p, t)[0, 1] / max(np.std(p, ddof=1) * sig, 1e-12) if sig > 0 else float("nan")
            print(f"  {kind:10s} {tag} {len(xs):4d} 个  Δ真 {t.mean():+.4f}  Δ预 {p.mean():+.4f}  平均偏差 {r.mean():+.4f} ± {1.96 * se_mean:.4f}  "
                  f"真变化离散 {sig:.4f}  预测误差离散 {bias_sd:.4f}  斜率 {slope:.2f}  相关 {corr:.2f}（扣噪声 {corr_adj:.2f}）")
    # 按房间看全部改牌组的胜率
    print("\n  全部改牌组按分层（胜率）")
    rk = collections.defaultdict(list)
    for x in kinds["全部改牌组"]:
        rk[x["room"]].append(x)
    for k in sorted(rk):
        xs = rk[k]
        t = np.array([x["dT"] for x in xs]); p = np.array([x["dP"] for x in xs]); r = np.array([x["r"] for x in xs])
        print(f"    {k:10s} {len(xs):4d} 个 Δ真 {t.mean():+.4f} Δ预 {p.mean():+.4f} 平均偏差 {r.mean():+.4f} ± {1.96 * math.sqrt(r.var(ddof=1) / len(r)):.4f}")

    # 选项排序：同一局面两个变体谁更好（像局外决策在几个选项里挑一个），只看真实差别显著（|配对差均值| > 2 倍标准误）的选项对
    import itertools
    by2 = collections.defaultdict(dict)
    for r in rows:
        by2[(r["sid"], r["var"].split(":")[0])][r["k"]] = r
    sids = sorted({s for s, _ in by2})
    deck = ["base", "upgrade", "remove", "remove_nb", "add_silent", "add_curse"]
    for tag, names in (("只改牌组（含原样）", deck), ("全部变体（含血量）", deck + ["hp-15%", "hp+15%"])):
        for key_t, key_p, nm in (("win", "p", "胜率"), ("loss", "h", "掉血")):
            res = collections.defaultdict(lambda: [0, 0])
            for s in sids:
                for x, y in itertools.combinations(names, 2):
                    A, B = by2.get((s, x)), by2.get((s, y))
                    if not A or not B:
                        continue
                    ks = sorted(set(A) & set(B))
                    if len(ks) < 8:
                        continue
                    d = np.array([A[k][key_t] - B[k][key_t] for k in ks], float)
                    se = d.std(ddof=1) / math.sqrt(len(ks))
                    if se == 0 or abs(d.mean()) < 2 * se:
                        continue
                    pd = np.mean([A[k][key_p] for k in ks]) - np.mean([B[k][key_p] for k in ks])
                    mag = abs(d.mean())
                    bk = "<0.05" if mag < .05 else "0.05-0.10" if mag < .10 else "0.10-0.20" if mag < .2 else ">0.20"
                    for b in (bk, "全部"):
                        res[b][0] += 1; res[b][1] += int(np.sign(pd) == np.sign(d.mean()))
            print(f"\n  选项排序 · {tag} · {nm}：真实差别显著的选项对里，预测排对的比例")
            for b in ("全部", "<0.05", "0.05-0.10", "0.10-0.20", ">0.20"):
                n, ok = res.get(b, [0, 0])
                if n:
                    print(f"     真实差 {b:10s} {n:5d} 对  排对 {ok / n:.1%}")


# ---------------------------------------------------------------- 战斗结果模型的复测（reports/战斗结果模型方案.md）

def _read(path: str):
    import gzip
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                pass


def eval_predictor(name: str, truth: dict, meta: dict, pred: dict, kpred: int) -> dict:
    """truth[(sid, 变体类)][k] = 真打的一场（win、loss）；pred[(sid, 变体类)][k] = (p, h)，只用 k < kpred 的开战帧。
    口径同检验二：预测取开战帧的平均（相当于抽牌之前），真实取全部种子的平均；变化按同种子配对，误差离散扣掉两边的抽样噪声。"""
    out = {"name": name}
    def pmean(d, j):
        xs = [v[j] for k, v in d.items() if k < kpred]
        return (float(np.mean(xs)), float(np.var(xs, ddof=1) / len(xs)) if len(xs) > 1 else 0.0) if xs else (None, None)
    # 原样
    per = collections.defaultdict(list)
    for (sid, var), d in truth.items():
        if var != "base" or (sid, "base") not in pred or len(d) < 4:
            continue
        w = np.array([x["win"] for x in d.values()], float); l = np.array([x["loss"] for x in d.values()])
        pm, pv = pmean(pred[(sid, "base")], 0); hm, hv = pmean(pred[(sid, "base")], 1)
        if pm is None:
            continue
        row = (pm - w.mean(), w.var(ddof=1) / len(w) + pv, hm - l.mean(), l.var(ddof=1) / len(l) + hv, w.mean(), l.mean(), pm)
        per[strat_key(meta[sid])].append(row); per["全部"].append(row)
    out["base"] = {}
    for k, rows in per.items():
        a = np.array(rows)
        if len(a) < 3:
            continue
        out["base"][k] = {"n": len(a), "bias": a[:, 0].mean(), "sd": math.sqrt(max(0.0, a[:, 0].var(ddof=1) - a[:, 1].mean())),
                          "biasL": a[:, 2].mean(), "sdL": math.sqrt(max(0.0, a[:, 2].var(ddof=1) - a[:, 3].mean())),
                          "corr": float(np.corrcoef(a[:, 6], a[:, 4])[0, 1])}
    # 变化
    kinds = collections.defaultdict(list)
    for (sid, var), d in truth.items():
        if var == "base" or (sid, var) not in pred or (sid, "base") not in pred:
            continue
        b = truth.get((sid, "base"), {})
        ks = sorted(set(d) & set(b))
        kp = sorted(k for k in set(pred[(sid, var)]) & set(pred[(sid, "base")]) if k < kpred)
        if len(ks) < 4 or len(kp) < 2:
            continue
        dw = np.array([d[k]["win"] - b[k]["win"] for k in ks], float); dl = np.array([d[k]["loss"] - b[k]["loss"] for k in ks])
        dp = np.array([pred[(sid, var)][k][0] - pred[(sid, "base")][k][0] for k in kp])
        dh = np.array([pred[(sid, var)][k][1] - pred[(sid, "base")][k][1] for k in kp])
        x = {"dT": dw.mean(), "nT": dw.var(ddof=1) / len(ks), "dP": dp.mean(), "nP": dp.var(ddof=1) / len(kp),
             "dL": dl.mean(), "nL": dl.var(ddof=1) / len(ks), "dH": dh.mean(), "nH": dh.var(ddof=1) / len(kp), "room": strat_key(meta[sid])}
        kinds[var].append(x)
        if var in ("upgrade", "remove", "remove_nb", "add_silent", "add_curse"):
            kinds["全部改牌组"].append(x)
    out["delta"] = {}
    for kind, xs in kinds.items():
        if len(xs) < 3:
            continue
        res = {"n": len(xs)}
        for tag, T, P, nT, nP in (("win", "dT", "dP", "nT", "nP"), ("loss", "dL", "dH", "nL", "nH")):
            t = np.array([x[T] for x in xs]); pp = np.array([x[P] for x in xs]); r = pp - t
            noise_t = np.mean([x[nT] for x in xs]); noise_r = noise_t + np.mean([x[nP] for x in xs])
            sig = math.sqrt(max(0.0, t.var(ddof=1) - noise_t)); esd = math.sqrt(max(0.0, r.var(ddof=1) - noise_r))
            cov = np.cov(pp, t)[0, 1]
            res[tag] = {"dT": t.mean(), "dP": pp.mean(), "bias": r.mean(), "ci": 1.96 * math.sqrt(r.var(ddof=1) / len(r)), "sig": sig, "esd": esd,
                        "ratio": esd / sig if sig > 0 else float("nan"), "slope": cov / max(np.var(pp, ddof=1), 1e-12),
                        "corr_adj": cov / max(np.std(pp, ddof=1) * sig, 1e-12) if sig > 0 else float("nan")}
        out["delta"][kind] = res
    # 选项排序（真实差别显著的选项对）
    import itertools
    sids = sorted({s for s, _ in truth})
    names = ["base", "upgrade", "remove", "remove_nb", "add_silent", "add_curse", "hp-15%", "hp+15%"]
    rank = collections.defaultdict(lambda: [0, 0])
    for s in sids:
        for x, y in itertools.combinations(names, 2):
            A, B = truth.get((s, x)), truth.get((s, y))
            PA, PB = pred.get((s, x)), pred.get((s, y))
            if not A or not B or not PA or not PB:
                continue
            ks = sorted(set(A) & set(B))
            if len(ks) < 8:
                continue
            for tag, kt, j in (("win", "win", 0), ("loss", "loss", 1)):
                d = np.array([A[k][kt] - B[k][kt] for k in ks], float)
                se = d.std(ddof=1) / math.sqrt(len(ks))
                if se == 0 or abs(d.mean()) < 2 * se:
                    continue
                pd = pmean(PA, j)[0] - pmean(PB, j)[0]
                small = abs(d.mean()) < (0.10 if tag == "win" else 0.05)
                for key in ((tag, "全部"), (tag, "小") if small else None):
                    if key:
                        rank[key][0] += 1; rank[key][1] += int(np.sign(pd) == np.sign(d.mean()))
    out["rank"] = {k: (v[0], v[1] / v[0]) for k, v in rank.items() if v[0]}
    return out


def print_eval(res: list[dict]) -> None:
    names = [r["name"] for r in res]
    print("\n原样局面（胜率：平均误差 / 误差离散；掉血：平均误差 / 误差离散；局面间预测与实际的相关）")
    keys = sorted({k for r in res for k in r["base"]}, key=lambda k: (k != "全部", k))
    for k in keys:
        cells = []
        for r in res:
            b = r["base"].get(k)
            cells.append(f"{b['bias']:+.3f}/{b['sd']:.3f} {b['biasL']:+.3f}/{b['sdL']:.3f} r{b['corr']:.2f}" if b else "-")
        print(f"  {k:10s} " + " | ".join(f"{n}: {c}" for n, c in zip(names, cells)))
    for tag, nm in (("win", "胜率"), ("loss", "掉血")):
        print(f"\n改动的效果（{nm}）：平均偏差 ± 95% / 误差离散 ÷ 真实离散 / 斜率 / 扣噪声相关")
        for kind in ("upgrade", "remove", "remove_nb", "add_silent", "add_curse", "全部改牌组", "hp-15%", "hp+15%"):
            cells = []
            for r in res:
                d = r["delta"].get(kind)
                x = d[tag] if d else None
                cells.append(f"{x['bias']:+.4f}±{x['ci']:.4f} {x['ratio']:.2f} 斜{x['slope']:.2f} 相{x['corr_adj']:.2f}" if x else "-")
            n = next((r["delta"][kind]["n"] for r in res if kind in r["delta"]), 0)
            print(f"  {kind:10s}({n:4d}) " + " | ".join(f"{nm2}: {c}" for nm2, c in zip(names, cells)))
    print("\n选项排序（真实差别显著的选项对里排对的比例；小 = 胜率差 < 10 个百分点、掉血差 < 5%）")
    for key in (("win", "全部"), ("win", "小"), ("loss", "全部"), ("loss", "小")):
        print(f"  {key[0]:4s} {key[1]:2s} " + " | ".join(f"{r['name']}: {r['rank'][key][1]:.1%}（{r['rank'][key][0]}）" if key in r["rank"] else f"{r['name']}: -" for r in res))


def report_pred(argv: list[str]) -> None:
    """pred --truth 真打结果文件（逗号分隔） --meta 局面元数据 --old 只取开战帧的文件（含旧辅助头的 p、h） --preds 名字=预测文件,... --kpred 8"""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", required=True); ap.add_argument("--meta", required=True)
    ap.add_argument("--old", default=""); ap.add_argument("--preds", default=""); ap.add_argument("--kpred", type=int, default=8)
    a = ap.parse_args(argv)
    truth = collections.defaultdict(dict)
    for fn in a.truth.split(","):
        for r in _read(fn):
            if "error" not in r and r.get("win") is not None:
                truth[(r["sid"], r["var"].split(":")[0])][r["k"]] = r
    meta = {m["sid"]: m for m in json.load(open(a.meta, encoding="utf-8"))}
    res = []
    if a.old:       # 旧辅助头：在同一批开战帧上的 p、h
        pr = collections.defaultdict(dict)
        for fn in a.old.split(","):
            for r in _read(fn):
                if r.get("p") is not None and "error" not in r:
                    pr[(r["sid"], r["var"].split(":")[0])][r["k"]] = (r["p"], r["h"])
        res.append(eval_predictor("旧辅助头", truth, meta, pr, a.kpred))
    for spec in [x for x in a.preds.split(",") if x]:
        nm, fn = spec.split("=", 1)
        pr = collections.defaultdict(dict)
        for r in _read(fn):
            pr[(r["sid"], r["var"].split(":")[0])][r["k"]] = (r["p"], r["h"])
        res.append(eval_predictor(nm, truth, meta, pr, a.kpred))
    print(f"真打 {sum(len(d) for d in truth.values())} 场，局面 {len(meta)}，预测用每个变体前 {a.kpred} 个种子的开战帧")
    print_eval(res)


def report_cal(argv: list[str]) -> None:
    """cal --d1 collect 的输出（逗号分隔） --preds 名字=预测文件,...：D1 验证整局（与模型训练同一划分）上，按层的校准与区分度。"""
    import argparse, hashlib
    ap = argparse.ArgumentParser(); ap.add_argument("--d1", required=True); ap.add_argument("--preds", default="")
    a = ap.parse_args(argv)
    isval = lambda g: int(hashlib.md5(g.encode()).hexdigest(), 16) % 20 == 0
    rows = {}
    for fn in a.d1.split(","):
        for r in _read(fn):
            if not isval(r["seed"]):
                continue
            for ci, c in enumerate(r["combats"]):
                if c.get("win") is not None and c.get("p") is not None:
                    rows[(r["seed"], ci)] = c
    preds = {"旧辅助头": {k: (c["p"], c["h"]) for k, c in rows.items()}}
    for spec in [x for x in a.preds.split(",") if x]:
        nm, fn = spec.split("=", 1)
        preds[nm] = {(r["seed"], r["ci"]): (r["p"], r["h"]) for r in _read(fn) if "seed" in r}
    print(f"D1 验证整局里的战斗 {len(rows)} 场")
    groups = collections.defaultdict(list)
    for k, c in rows.items():
        groups[strat_key(c)].append(k); groups["全部"].append(k)
    for g in ["全部"] + sorted(x for x in groups if x != "全部"):
        ks = groups[g]
        y = np.array([rows[k]["win"] for k in ks], float); l = np.array([rows[k]["loss"] for k in ks])
        cells = []
        for nm, pr in preds.items():
            kk = [i for i, k in enumerate(ks) if k in pr]
            if not kk:
                continue
            p = np.array([pr[ks[i]][0] for i in kk]); h = np.array([pr[ks[i]][1] for i in kk]); yy = y[kk]; ll = l[kk]
            br = np.mean((p - yy) ** 2); b0 = np.mean((yy.mean() - yy) ** 2)
            cells.append(f"{nm}: 预测 {p.mean():.3f} 技巧 {1 - br / b0 if b0 > 0 else float('nan'):.2f} AUC {auc(p, yy):.3f} 掉血R² {1 - np.mean((h - ll) ** 2) / max(np.var(ll), 1e-12):.2f}")
        print(f"  {g:10s} {len(ks):5d} 场 实际 {y.mean():.3f} | " + " | ".join(cells))


def report_stats(argv: list[str]) -> None:
    """stats --d1 collect 的输出：战斗中的永久变化（比对开战与战后的玩家快照）各类出现的频率，决定模型要不要预测它们。"""
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--d1", required=True)
    a = ap.parse_args(argv)
    cnt = collections.Counter(); n = 0
    ex = collections.defaultdict(collections.Counter)
    for fn in a.d1.split(","):
        for r in _read(fn):
            for c in r["combats"]:
                if not c.get("win") or not c.get("snap0") or not c.get("snap1"):
                    continue
                s0, s1 = json.loads(c["snap0"]), json.loads(c["snap1"])
                n += 1
                if s1.get("max_hp") != s0.get("max_hp"):
                    cnt["最大生命变了"] += 1; ex["最大生命变了"][f"{c['encounter']}"] += 1
                d0 = collections.Counter((x["id"], x.get("current_upgrade_level", 0)) for x in s0.get("deck") or [])
                d1 = collections.Counter((x["id"], x.get("current_upgrade_level", 0)) for x in s1.get("deck") or [])
                if d0 != d1:
                    cnt["牌组变了"] += 1
                    for k in (d1 - d0):
                        ex["牌组多了"][k[0]] += 1
                    for k in (d0 - d1):
                        ex["牌组少了"][k[0]] += 1
                if s1.get("gold") != s0.get("gold"):
                    cnt["金币变了"] += 1
                r0 = collections.Counter(x["id"] for x in s0.get("relics") or []); r1 = collections.Counter(x["id"] for x in s1.get("relics") or [])
                if r0 != r1:
                    cnt["遗物变了"] += 1
                p0 = collections.Counter(x["id"] for x in s0.get("potions") or []); p1 = collections.Counter(x["id"] for x in s1.get("potions") or [])
                if p1 - p0:
                    cnt["战斗中得到药水"] += 1
                if p0 - p1:
                    cnt["用了药水"] += 1
    print(f"赢下且有前后快照的战斗 {n} 场")
    for k, v in cnt.most_common():
        print(f"  {k:10s} {v:6d}（{v / max(n, 1):.2%}）")
    for k, c in ex.items():
        print(f"  {k} 最常见：" + "，".join(f"{x} {y}" for x, y in c.most_common(8)))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd in ("runs", "cf"):
        {"runs": report_runs, "cf": report_cf}[cmd](sys.argv[2])
    else:
        {"pred": report_pred, "cal": report_cal, "stats": report_stats}[cmd](sys.argv[2:])
