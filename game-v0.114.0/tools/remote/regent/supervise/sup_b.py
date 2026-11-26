# 储君监管：学习端检查（在 B 上跑，读日志末尾）。输出一行读数；有问题的项以 ALERT: 开头，到了决策条件的以 TRIGGER: 开头
import json, os, statistics as S, time
L = "/root/.local/share/sls2/runs/regent/log.jsonl"
with open(L, "rb") as f:
    f.seek(max(0, os.path.getsize(L) - 26_000_000))
    raw = f.read().split(b"\n")[1:]
rows = []
for l in raw:
    try:
        rows.append(json.loads(l))
    except ValueError:
        pass
# 回退过权重时日志里先后有两段同号的版本（储君决策记录第 27 条：第 54,604–55,715 版的 Muon 段作废，从第 54,603 版接着训）：
# 按文件顺序，后面的记录版本号不大于前面的，就把前面版本号不小于它的那些（作废的时间线）去掉（同 tools/remote/lite_stats.py 的 drop_reverted）
out = []
for r in rows:
    if "version" not in r: continue
    while out and out[-1]["version"] >= r["version"]: out.pop()
    out.append(r)
rows = out
w50, w200 = rows[-50:], rows[-200:]
out, flags = [], []
age = time.time() - (rows[-1].get("time") or 0)
med = lambda k, w=w50: S.median([r[k] for r in w if isinstance(r.get(k), (int, float))] or [0])
kl, ent, entm, lvm = med("approx_kl"), med("entropy"), med("entropy_macro"), med("loss_v_macro")
nan = sum(r.get("nan_skipped") or 0 for r in w50)
early = sum(1 for r in w50 if r.get("epochs", 2) < 2)
ep = sum(r.get("run_episodes") or 0 for r in rows[-100:])
floor = sum((r.get("run_floor_mean") or 0) * (r.get("run_episodes") or 0) for r in rows[-100:]) / max(1, ep)
def rate(k):
    x = sum(r[k][0] for r in w200 if r.get(k)); n = sum(r[k][1] for r in w200 if r.get(k))
    return x / n if n else None
def brate(i):
    x = sum(r["boss_wins"][i] for r in w200 if r.get("boss_wins")); n = sum(r["boss_n"][i] for r in w200 if r.get("boss_n"))
    return x / n if n else None
a1m, a1e, a2m, a2e, b1, b2, b3 = rate("act1_monster"), rate("act1_elite"), rate("act2_monster"), rate("act2_elite"), brate(0), brate(1), brate(2)
out.append(f"B v{rows[-1]['version']} 日志{age:.0f}s前 KL{kl:.4f} 熵{ent:.3f}/{entm:.3f} 局外价值损失{lvm:.4f} 早停{early}/50 非有限{nan} 整局{floor:.2f}层"
           f" 练习 一幕{a1m:.3f}/{a1e:.3f} 二幕{a2m:.3f}/{a2e:.3f} Boss{b1:.3f}/{b2:.3f}/{b3:.3f}")
if age > 600: flags.append(f"ALERT:学习端日志 {age:.0f} 秒没更新")
if nan: flags.append(f"ALERT:非有限梯度 {nan}")
if early > 15: flags.append(f"ALERT:早停 {early}/50")
if kl > 0.04: flags.append(f"ALERT:KL 中位 {kl:.4f}")
# 第 25 条 Muon 时加的线，第 27 条退回 Adam 后留作异常线：Adam 下 KL 中位约 0.005，超过 0.010 叫醒
if kl > 0.010: flags.append(f"ALERT:KL中位{kl:.4f}>0.010")
if ent < 0.55: flags.append(f"ALERT:战斗熵 {ent:.3f}")
if entm < 0.22: flags.append(f"ALERT:局外熵 {entm:.3f}")
# 局外价值损失与解释方差：训练里开始有通关后，通关局的终局奖励（通关 2 + 第二个 Boss 2）让回报方差变大，两者跟着变差是正常的。
# 故障机器人同阶段（训练通关 0.3%）局外价值损失中位约 0.06，之后随通关率涨到 0.2 时模型仍在变强；第 18,776 版储君 0.04、解释方差 0.80 时报过一次（第 18,730 版一批 2 局通关，损失 0.20）。
# 线放宽到损失 0.12、解释方差 0.65（2026-11-07）
# 第 25,316 版例行巡检：训练整局通关每 500 版 4.4% → 5.9% → 6.7%，局外价值损失 0.106、解释方差 0.849；故障机器人训练通关 7.6% 时约 0.20，线 0.12 → 0.25
# 第 41,440 版报 0.2519：每 500 版中位 0.186 → 0.225 跟着训练通关 20.8% → 29.2% 平稳上升，局外熵没有趋势；通关率往 50% 走时回报方差还会涨约 25%，线 0.25 → 0.35，护栏改靠解释方差（线 0.65 → 0.75）
if lvm > 0.35: flags.append(f"ALERT:局外价值损失 {lvm:.4f}")
# 局外价值的解释方差（第 14,068 版巡检：几十版读数 0.833，此前 0.855–0.863），掉到 0.80 以下叫醒
evm = S.median([r["health"]["ev"]["macro"] for r in w50 if (r.get("health") or {}).get("ev")] or [1])
out.append(f"局外解释方差{evm:.3f}")
if evm < 0.75: flags.append(f"ALERT:局外解释方差 {evm:.3f}")
if floor < 22: flags.append(f"ALERT:训练整局 {floor:.2f} 层")
if a1m is not None and a1m < 0.98: flags.append(f"ALERT:第一幕普通战练习 {a1m:.3f}（遗忘？）")
if a1e is not None and a1e < 0.90: flags.append(f"ALERT:第一幕精英练习 {a1e:.3f}（遗忘？）")
# 挪份额按最近 500 版判断（200 版噪声大：2026-11-07 第 10,760 版 200 版 96.0%，500、1,000 版都是 95.2%）
x5 = sum(r["act2_monster"][0] for r in rows[-500:] if r.get("act2_monster")); n5 = sum(r["act2_monster"][1] for r in rows[-500:] if r.get("act2_monster"))
a2m5 = x5 / n5 if n5 else None
out.append(f"二幕普通500版{a2m5:.3f}" if a2m5 is not None else "")
# 第 18 条已挪（第二幕普通战份额 0.20 → 0.10）；之后改看它会不会回落：改前 500 版 96.0%，掉到 94.5% 以下叫醒
# 下一次挪份额照故障机器人第 120 条：它在第二幕精英练习约 94% 时把第二幕真实局面 0.37 → 0.25、腾给第三幕 Boss；储君第 18,396 版时 84.6%，到 92% 叫醒
x5e = sum(r["act2_elite"][0] for r in rows[-500:] if r.get("act2_elite")); n5e = sum(r["act2_elite"][1] for r in rows[-500:] if r.get("act2_elite"))
# 第 19 条已挪（第 27,073 版起第二幕真实局面 0.37 → 0.25，精英绝对份额 0.27 → 0.18）：改前最近 500 版 92.3%，掉到 89% 以下叫醒
# 第 22 条（第 45,281 版起第二幕真实局面 0.25 → 0.15，精英绝对份额 0.18 → 0.11）：改前 500 版 97.5%，掉到 95% 以下叫醒
if n5e and x5e / n5e < 0.95: flags.append(f"ALERT:第二幕精英练习（500版）{x5e / n5e:.3f}<95%（第22条挪份额后回落？）")
out.append(f"二幕精英500版{x5e / n5e:.3f}" if n5e else "")
# 下一次挪份额：第二幕精英练习 500 版到 97% 叫醒，考虑把第二幕真实局面 0.25 再挪一部分给第三幕 Boss（第 40,321 版时 96.2%；照故障机器人第 105 条在约 97% 时挪的节奏）
# 第 44,222 版第一次到线时只在第 21 条（第 43,999 版）改后约 220 版，窗口一半是改后的数据、作用分不开：改后满 750 版（第 44,749 版）再判断
# 97% 的挪份额触发已用掉（第 22 条，用户：“挪动练习份额吧。”）；第二幕普通战改前 500 版 99.5%，掉到 98% 以下叫醒
if a2m5 is not None and a2m5 < 0.98: flags.append(f"ALERT:第二幕普通战练习（500 版）{a2m5:.3f} < 98%（第22条挪份额后回落？）")
if b3 is not None and b3 >= 0.03: flags.append(f"TRIGGER:第三幕 Boss 练习 {b3:.3f} ≥ 3%（看评测到第三幕）")
# 第 24 条（第 49,389 版起第一幕真实局面 0.08 → 0.04、第三幕非 Boss 0.10 → 0.06）：改前 500 版第一幕精英 99.0%、第三幕普通 99.5%、精英 98.5%，
# 少练之后回落到 97.5% / 98.5% / 97.0% 以下叫醒
for key, lab, line in (("act1_elite", "一幕精英", 0.975), ("act3_monster", "三幕普通战", 0.985), ("act3_elite", "三幕精英", 0.970)):
    xk = sum(r[key][0] for r in rows[-500:] if r.get(key)); nk = sum(r[key][1] for r in rows[-500:] if r.get(key))
    if nk >= 2000 and xk / nk < line: flags.append(f"ALERT:{lab}练习（500版）{xk / nk:.3f}<{line:.1%}（第24条少练后回落？）")
V18 = 54604  # 最近一次改动生效的版本：第 27 条退回第 54,603 版的检查点、继续 Adam（第 25 条的 Muon 段作废），配置同第 24 条；两个 1,000 版窗口都落在改后（2,000 版）才判走平
V18_WAIT = 2000
# 练习走平（学习率判断改以练习胜率为准）：最近 400 版对前 400 版，第二幕精英与第二幕 Boss 练习胜率都涨不到 0.5 个百分点
def rate_w(w, k, i=None):
    if i is None:
        x = sum(r[k][0] for r in w if r.get(k)); n = sum(r[k][1] for r in w if r.get(k))
    else:
        x = sum(r["boss_wins"][i] for r in w if r.get("boss_wins")); n = sum(r["boss_n"][i] for r in w if r.get("boss_n"))
    return x / n if n else None
if len(rows) >= 2000:
    new, old = rows[-1000:], rows[-2000:-1000]
    de = rate_w(new, "act2_elite") - rate_w(old, "act2_elite"); db = rate_w(new, None, 1) - rate_w(old, None, 1); d3 = rate_w(new, None, 2) - rate_w(old, None, 2)
    out.append(f"练习1000版涨幅 二幕精英{de*100:+.1f} 二幕Boss{db*100:+.1f} 三幕Boss{d3*100:+.1f}")
    # 第 7,988 版起训练判输线 20 → 8（储君决策记录第 13 条补记），先让它跑约 800 版再判断学习率，免得两项改动混在一起
    # 第 25,865 版起看第二、三幕 Boss（第二幕精英练习约 91% 已近饱和，现在卡在第三幕 Boss）
    if rows[-1]["version"] >= V18 + V18_WAIT and db < 0.005 and d3 < 0.005: flags.append(f"TRIGGER:练习走平（二幕Boss{db*100:+.1f}、三幕Boss{d3*100:+.1f}个百分点/1000版）")
import subprocess
sc = subprocess.run(["grep", "-c", "stallcut", "/root/.local/share/sls2/runs/regent/engine.log"], capture_output=True, text=True).stdout.strip()
out.append(f"B采集判输累计{sc}")
print(" ".join(out + flags))
