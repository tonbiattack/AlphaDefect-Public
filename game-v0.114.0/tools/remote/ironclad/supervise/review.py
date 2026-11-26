# 铁甲战士从零线复盘（在 C 上跑：python3 - SINCE [PREV] < review.py；2026-11-17，铁甲战士决策记录第 1 条，照恐惧模式决策记录第 10 条的复盘脚本）。
# SINCE 为上一次改训练设置后的第一版：评测按每 6 轮分段（之前那段作对照），各 Boss 过关率前后对比，学习端熵、KL、训练整局与练习胜率，卡住与引擎报错。
import collections, json, statistics as S, sys, time
R = "/root/.local/share/sls2/runs/ironclad"; E = "/root/.local/share/sls2/runs/ironclad-eval"
SINCE = int(sys.argv[1]) if len(sys.argv) > 1 else 1
PREV = int(sys.argv[2]) if len(sys.argv) > 2 else 0      # 对照段的起点
print("C 时间", time.strftime("%F %T"), "，对照段 第", PREV, "版起，改后 第", SINCE, "版起")
ev = []
for l in open(f"{E}/eval.jsonl"):
    if not l.strip():
        continue
    r = json.loads(l)
    while ev and ev[-1]["version"] >= r["version"]:
        ev.pop()
    ev.append(r)
def agg(rs):
    return (len(rs), round(S.mean(r["win_rate"] for r in rs) * 100, 1), round(S.mean(r["floor_mean"] for r in rs), 1),
            round(S.mean(r["reach_act2"] for r in rs) * 100, 1), round(S.mean(r["reach_act3"] for r in rs) * 100, 1), sum(r.get("stuck") or 0 for r in rs))
pre = [r for r in ev if PREV <= r["version"] < SINCE]; post = [r for r in ev if r["version"] >= SINCE]
print("轮数/通关/层/二幕/三幕/卡住：对照", agg(pre) if pre else None, "改后", agg(post) if post else None)
for i in range(0, len(post), 6):
    rs = post[i:i + 6]; print(f"  第 {rs[0]['version']}–{rs[-1]['version']} 版", agg(rs))
def bosses(rs):
    c = collections.defaultdict(lambda: [0, 0])
    for r in rs:
        for k, (w, n) in (r.get("bosses") or {}).items():
            c[k][0] += w; c[k][1] += n
    return c
b0, b1 = bosses(pre), bosses(post)
print("Boss 过关率（对照 → 改后）：" + "；".join(f"{k.split(':')[1].replace('_BOSS', '')}{int(k[0]) + 1} {b0[k][0] / max(1, b0[k][1]):.0%}→{b1[k][0] / max(1, b1[k][1]):.0%}" for k in sorted(b1)))
rows = [json.loads(l) for l in open(f"{R}/log.jsonl") if l.strip()]
last = rows[-1]["version"]
for name, rs in (("对照段末 100 版", [r for r in rows if SINCE - 100 <= r["version"] < SINCE]), ("最近 100 版", rows[-100:])):
    if not rs:
        continue
    ent_c = [(r["entropy"] - r["macro_frac"] * r["entropy_macro"]) / (1 - r["macro_frac"]) for r in rs
             if all(isinstance(r.get(k), (int, float)) for k in ("entropy", "entropy_macro", "macro_frac")) and r["macro_frac"] < 1]
    ep = sum(r.get("run_episodes") or 0 for r in rs)
    win = sum((r.get("run_win_rate") or 0) * (r.get("run_episodes") or 0) for r in rs) / max(1, ep)
    bw = [sum(r["boss_wins"][i] for r in rs if r.get("boss_wins")) / max(1, sum(r["boss_n"][i] for r in rs if r.get("boss_n"))) for i in range(3)]
    print(f"{name}（第 {rs[0]['version']}–{rs[-1]['version']} 版）：战斗熵 {S.median(ent_c):.3f} 局外熵 {S.median([r['entropy_macro'] for r in rs]):.3f} "
          f"KL {S.median([r['approx_kl'] for r in rs]):.4f} 裁剪 {S.median([r['clip_frac'] for r in rs]):.3f} 训练整局通关 {win:.1%} 练习 Boss {bw[0]:.2f}/{bw[1]:.2f}/{bw[2]:.2f} 卡住 {sum(r.get('stuck') or 0 for r in rs)}")
print("最新版本", last)
