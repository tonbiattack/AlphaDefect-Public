# 猎人监管：学习端检查（在 B 上跑，读日志末尾）。输出一行读数；有问题的项以 ALERT: 开头，到了决策条件的以 TRIGGER: 开头（名字里不能有空格）
import json, os, statistics as S, time
L = "/root/.local/share/sls2/runs/silent/log.jsonl"
if not os.path.exists(L):
    print("S 还没有日志"); raise SystemExit
with open(L, "rb") as f:
    f.seek(max(0, os.path.getsize(L) - 30_000_000))
    raw = f.read().split(b"\n")[1:]
rows = []
for l in raw:
    try:
        r = json.loads(l)
    except ValueError:
        continue
    if "version" not in r:
        continue
    while rows and rows[-1]["version"] >= r["version"]:     # 回退过权重时去掉作废的时间线（同 lite_stats.drop_reverted）
        rows.pop()
    rows.append(r)
if not rows:
    print("S 日志里还没有版本"); raise SystemExit
w50, w200 = rows[-50:], rows[-200:]
flags = []
v = rows[-1]["version"]
age = time.time() - (rows[-1].get("time") or 0)
med = lambda k, w=w50: S.median([r[k] for r in w if isinstance(r.get(k), (int, float))] or [float("nan")])
kl, ent, entm, lvm, clip = med("approx_kl"), med("entropy"), med("entropy_macro"), med("loss_v_macro"), med("clip_frac")
nan = sum(r.get("nan_skipped") or 0 for r in w50)
early = sum(1 for r in w50 if r.get("epochs", 2) < 2)
ep = sum(r.get("run_episodes") or 0 for r in rows[-100:])
floor = sum((r.get("run_floor_mean") or 0) * (r.get("run_episodes") or 0) for r in rows[-100:]) / max(1, ep)
win = sum((r.get("run_win_rate") or 0) * (r.get("run_episodes") or 0) for r in rows[-100:]) / max(1, ep)
def rate(k):
    x = sum(r[k][0] for r in w200 if r.get(k)); n = sum(r[k][1] for r in w200 if r.get(k))
    return x / n if n else float("nan")
def brate(i):
    x = sum(r["boss_wins"][i] for r in w200 if r.get("boss_wins")); n = sum(r["boss_n"][i] for r in w200 if r.get("boss_n"))
    return x / n if n else float("nan")
evm = S.median([r["health"]["ev"]["macro"] for r in w50 if (r.get("health") or {}).get("ev", {}).get("macro") is not None] or [float("nan")])
upd = (rows[-1].get("health") or {}).get("upd") or {}
print(f"S v{v} 日志{age:.0f}s前 KL{kl:.4f} 裁剪{clip:.3f} 熵{ent:.3f}/{entm:.3f} 局外价值损失{lvm:.4f} 局外解释方差{evm:.3f} 早停{early}/50 非有限{nan} "
      f"整局{floor:.2f}层/通关{win:.3f} 练习 一幕{rate('act1_monster'):.3f}/{rate('act1_elite'):.3f} 二幕{rate('act2_monster'):.3f}/{rate('act2_elite'):.3f} "
      f"Boss{brate(0):.3f}/{brate(1):.3f}/{brate(2):.3f} muon_lr{rows[-1].get('muon_lr')} 每版更新L0{upd.get('L0', float('nan')):.4f}/L5{upd.get('L5', float('nan')):.4f}", end="")
if age > 600: flags.append(f"ALERT:学习端日志{age:.0f}秒没更新")
if nan: flags.append(f"ALERT:非有限梯度{nan}")
if early > 15: flags.append(f"ALERT:早停{early}/50")
if kl == kl and kl > 0.04: flags.append(f"ALERT:KL中位{kl:.4f}")
if ent == ent and ent < 0.3: flags.append(f"ALERT:熵（全部步）{ent:.3f}")
if entm == entm and entm < 0.15: flags.append(f"ALERT:局外熵{entm:.3f}")
# 熵坍塌（2026-11-11，第 11 条扩批后；用户：“这样子扩批的话，我们需要关注一下策略熵是否坍塌”）：最近 50 版中位。日志的 entropy 是全部步的平均，
# 战斗步按 (entropy − macro_frac × entropy_macro) / (1 − macro_frac) 推算（同看板 lite_stats.ent_combat）。线的依据（全程每 500 版中位）：
# 战斗步第 14,000 版起 0.67–0.68，第 8 条（每批加倍）后最低 0.644 再回升，单版 σ 约 0.015 → 0.60；局外步第 10,000 版起 0.284–0.31 → 0.25；
# 地图选路步全程 0.052–0.070、单版 σ 约 0.008 → 0.04。看板“训练健康”有这三项的全程与逐版走势
entc = S.median([(r["entropy"] - r["macro_frac"] * r["entropy_macro"]) / (1 - r["macro_frac"]) for r in w50
                 if all(isinstance(r.get(k), (int, float)) for k in ("entropy", "entropy_macro", "macro_frac")) and 0 <= r["macro_frac"] < 1] or [float("nan")])
entp = med("entropy_map")
print(f" 熵 战斗步{entc:.3f}/局外{entm:.3f}/选路{entp:.3f}", end="")
if entc == entc and entc < 0.60: flags.append(f"ALERT:熵坍塌？战斗步熵50版中位{entc:.3f}<0.60")
if entm == entm and entm < 0.25: flags.append(f"ALERT:熵坍塌？局外熵50版中位{entm:.3f}<0.25")
if entp == entp and entp < 0.04: flags.append(f"ALERT:熵坍塌？选路熵50版中位{entp:.3f}<0.04")
# 第 2 条（第 1,060 版起 Adam 部分 3e-4 → 4e-4，Muon 6e-4 不变）：改前 50 版 KL 中位 0.0123、裁剪 0.113；KL 中位超过 0.02 叫醒（考虑退回 3e-4）
if kl == kl and kl > 0.02 and v >= 1060: flags.append(f"ALERT:第2条后KL中位{kl:.4f}>0.02")
# 第 7 条（第 17,853 版起第二幕真实局面 0.37 → 0.25，腾出的 0.12 给第三幕 Boss）的护栏（同储君第 19 条）：
# 第二幕精英练习最近 500 版掉到 89% 以下（改前 93.5%）、普通战掉到 94.5% 以下（改前 97.9%）叫醒
# 下一次挪份额（用户：“练习份额什么的，你看情况自己调”）：照储君第 22 条，第二幕精英练习最近 500 版到 97%，再把第二幕真实局面 0.25 → 0.15 挪给第三幕 Boss
w500 = rows[-500:]
def rate_w(w, k):
    x = sum(r[k][0] for r in w if r.get(k)); n = sum(r[k][1] for r in w if r.get(k)); return x / n if n else float("nan")
r2m, r2e = rate_w(w500, "act2_monster"), rate_w(w500, "act2_elite")
print(f" 500版二幕{r2m:.3f}/{r2e:.3f}", end="")
if r2m < 0.945: flags.append(f"ALERT:第7条后二幕普通战练习500版{r2m:.3f}<94.5%")
if r2e < 0.89: flags.append(f"ALERT:第7条后二幕精英练习500版{r2e:.3f}<89%")
# “第二幕精英练习到 97%”的挪份额触发在第 21,277 版报过，已用掉（第 10 条）
# 第 10 条（第 21,286 版起第二幕真实局面 0.25 → 0.15，腾出的 0.10 给第三幕 Boss）的护栏（同储君第 22 条）：
# 第二幕精英练习最近 500 版掉到 95% 以下（改前 97.0%）、普通战掉到 97.5% 以下（改前 99.1%）叫醒；改后满 500 版才判，免得窗口里混着改前的数据
V10 = 21286
if v >= V10 + 500:
    if r2m < 0.975: flags.append(f"ALERT:第10条后二幕普通战练习500版{r2m:.3f}<97.5%")
    if r2e < 0.95: flags.append(f"ALERT:第10条后二幕精英练习500版{r2e:.3f}<95%")
# 第 8 条（第 18,638 版起学习率减半、每批加倍）的护栏（同储君第 21 条）：改后满 1,000 版，训练整局通关最近 500 版低于 16.2%（改前约 18.2%）叫醒，
# 再看评测最近 12 轮（sup_eval 低于 27.2% 叫醒）；两条都成立就退回 lr 4e-4、muon_lr 6e-4、每批 32,768
V8 = 18638
if v >= V8 + 1000:
    w5 = [r for r in rows[-500:] if r["version"] >= V8]
    e5 = sum(r.get("run_episodes") or 0 for r in w5)
    win5 = sum((r.get("run_win_rate") or 0) * (r.get("run_episodes") or 0) for r in w5) / max(1, e5)
    print(f" 第8条后500版整局通关{win5:.3f}", end="")
    if win5 < 0.162: flags.append(f"ALERT:第8条后训练整局通关500版{win5:.3f}<16.2%")
# 第 9 条（第 19,901 版起观测格式 21 → 23）的护栏：改后满 500 版，训练整局通关改后最近 500 版低于 30.4%（改前 500 版 34.44%）叫醒，
# 再看评测最近 12 轮（sup_eval 低于 41.9% 叫醒）；两条都成立就退回格式 21（还原 FormatFor 与 FORMAT_BY_CHARACTER、重新部署，从当时的权重接着训）
V9 = 19901
if v >= V9 + 500:
    w9 = [r for r in rows[-500:] if r["version"] >= V9]
    e9 = sum(r.get("run_episodes") or 0 for r in w9)
    win9 = sum((r.get("run_win_rate") or 0) * (r.get("run_episodes") or 0) for r in w9) / max(1, e9)
    print(f" 第9条后{len(w9)}版整局通关{win9:.3f}", end="")
    if win9 < 0.304: flags.append(f"ALERT:第9条后训练整局通关{len(w9)}版{win9:.3f}<30.4%")
# 第 11 条（第 23,321 版起学习率再减半、每批再加倍：Adam 2e-4 → 1e-4、Muon 3e-4 → 1.5e-4、131,072 步）的护栏（同第 8 条）：
# 改后满 500 版（每版的数据是第 8 条之后的两倍，500 版 ≈ 原来的 1,000 版），训练整局通关改后最近 500 版低于 38.9%（改前 500 版 40.97% 减 2 个百分点）叫醒，
# 再看评测最近 12 轮（sup_eval 低于 51.4% 叫醒）；两条都成立就退回 lr 2e-4、muon_lr 3e-4、每批 65,536（从当时的权重接着训，必要时还原 v23320 备份）。
# 入训吞吐：改后最近 50 版低于每秒 5,500 步叫醒（改前 60 版 6,732 步 / 秒；每批加倍不该改变吞吐）
V11 = 23321
if v >= V11 + 50:
    w50s = [r for r in w50 if r["version"] >= V11 + 2]
    if len(w50s) >= 2:
        # 相邻两版间隔超过 120 秒的跳过（停机、备份、重启编译的那几分钟不算吞吐；2026-11-11 第 24,335 版重启后误报过一次 5,346）
        pr = [(b["total_steps"] - a["total_steps"], b["time"] - a["time"]) for a, b in zip(w50s, w50s[1:]) if 0 < b["time"] - a["time"] <= 120]
        sps11 = sum(s for s, _ in pr) / max(1e-9, sum(t for _, t in pr))
        print(f" 入训{sps11:,.0f}步/秒", end="")
        lim = 5000 if v >= 24208 else 5500      # 第 14 条（跳过稳赢战斗）后每版 20 → 22.5 秒、入训约 5,800，线降到 5,000（见下面第 14 条的说明）
        if sps11 < lim: flags.append(f"ALERT:入训{sps11:.0f}步/秒<{lim}")
if v >= V11 + 500:
    w11 = [r for r in rows[-500:] if r["version"] >= V11]
    e11 = sum(r.get("run_episodes") or 0 for r in w11)
    win11 = sum((r.get("run_win_rate") or 0) * (r.get("run_episodes") or 0) for r in w11) / max(1, e11)
    print(f" 第11条后{len(w11)}版整局通关{win11:.3f}", end="")
    if win11 < 0.389: flags.append(f"ALERT:第11条后训练整局通关{len(w11)}版{win11:.3f}<38.9%")
# 第 12 条（第 23,828 版起整局分段 run_segment_steps 64 → 256；用户 2026-11-11：“读出后上 256”）的护栏。改前 500 版（第 23,326–23,825 版）：
# 训练整局通关 39,274 局 47.18%；每版中途段中位 718（691–749），每段约 94 步；队列里的步平均滞后 0.42 版（0 版 58%、1 版 42%），过期步 0；
# 局外价值损失 50 版中位 0.228–0.257（单版中位 0.238、σ 0.032），局外解释方差 50 版中位 0.845–0.864（单版中位 0.855、σ 0.021）。
# - 段数：改后满 20 版起，最近 50 版的每版中途段中位还在 400 以上，说明 256 没生效（预计约 230 段、每段约 290 步），叫醒
# - 策略滞后：改后满 20 版起，最近 50 版队列里的步平均滞后超过 1.5 版，或出现过期步（超过 max_policy_lag 6），叫醒（段长 256 预计多半版到一版）
# - 局外价值：改后满 50 版起，局外价值损失 50 版中位超过 0.52 或局外解释方差低于 0.72 叫醒。原定按改前定的 0.40 / 0.78，上线头几版就碰线了：
#   目标的视野变长、自举少了，目标本身的方差变大，损失涨、解释方差掉是预料中的（故障机器人第 267 条沿整局算时局外价值均方误差约 0.21 → 0.43）。
#   改后第 23,848–23,882 版（过渡 20 版之后）稳在损失中位 0.403（单版 σ 0.045）、解释方差中位 0.781（σ 0.028），之后没再变差；
#   线改按改后的平台定，损失高出约 0.12、解释方差低约 0.06 才算继续变差（50 版中位的抽样误差约 0.008 / 0.005）
# - 效果：改后满 500 版起，训练整局通关改后最近 500 版低于 45.2%（改前 47.18% 减 2 个百分点）叫醒，再看评测最近 12 轮（sup_eval 低于 60.5% 叫醒）；
#   两条都成立就退回 64（从当时的权重接着训，必要时还原 backups/v23827-20261111-1203-整局分段256前.pt）
# - 第 12 条满 500 版读数（2026-11-11 第 24,332 版；第 23,829–24,333 版对第 11 条后第 23,327–23,826 版）：训练整局通关 47.19% → 45.66%，
#   评测（第 12 条前 12 轮 62.00% → 改后最近 10 轮 61.28%）平均层数 44.91 → 44.07、每局死于第一幕 7.3% → 9.8%，练习第三幕 Boss 0.61 → 0.59，
#   局外熵 0.302 → 0.343、选路熵 0.064 → 0.080，没有一项变好。先在第 24,335 版把分段退回 64（从当时的权重接着训，跑了 10 版），
#   用户随即要求“请直接连着权重一起回退”：第 13 条把权重还原到第 23,827 版（第 12 条上线前的备份），第 23,828 版起是新的时间线，
#   第 12 条那段（第 23,828–24,344 版）作废，日志里同号的旧记录由上面的去重（drop_reverted）丢掉。第 12 条的护栏随之停用。
# 第 13 条（权重还原到第 23,827 版、整局分段 64，即第 11 条的配置与权重；B 2026-11-11 15:07 起第 23,828 版）的护栏。参照（第 11 条后第 23,327–23,826 版）：
#   训练整局通关 47.19%，每版中途段约 718，局外价值损失 0.238 / 解释方差 0.855（50 版中位 0.228–0.257 / 0.845–0.864），局外熵 0.302（每 25 版中位 0.300–0.312）、选路熵 0.064。
# - 段数：满 20 版起，最近 50 版每版中途段中位低于 500 叫醒（64 没生效）
# - 策略滞后：最近 50 版队列里的步平均滞后超过 1.5 版，或出现过期步，叫醒（同第 12 条）
# - 局外价值：满 50 版起，局外价值损失 50 版中位超过 0.32 或局外解释方差低于 0.80 叫醒（权重本来就是按 64 的目标训的，不用等过渡）
# - 局外熵：满 50 版起，局外熵 50 版中位超过 0.335 叫醒（同样的权重、同样的配置又往上漂，就说明第 12 条期间的上漂不全是 256 造成的，要另找原因）
# - 效果：满 500 版（第 24,328 版）起报 TRIGGER 读效果；训练整局通关最近 500 版低于 46.2%（第 12 条期间 45.66%；第 11 条后 47.19%）叫醒，
#   与评测（sup_eval）一起看：还原了权重还是掉，说明第 12 条期间的回落有别的原因
V12 = V13 = 23828     # 第 12 条那段作废后，第 13 条的新时间线也从第 23,828 版起
if v >= V13 + 20:
    w13 = [r for r in w50 if r["version"] >= V13]
    segm = S.median([r.get("run_segments") or 0 for r in w13])
    qh = [0] * 11
    for r in w13:
        for i, x in enumerate(((r.get("lag_split") or {}).get("local") or {}).get("queue", [])[:11]): qh[i] += x
    qlag = sum(i * x for i, x in enumerate(qh)) / max(1, sum(qh))
    stale13 = sum(r.get("stale_steps") or 0 for r in w13)
    print(f" 第13条后每版中途段{segm:.0f} 队列滞后{qlag:.2f}版 过期步{stale13}", end="")
    if segm < 500: flags.append(f"ALERT:第13条后每版中途段中位{segm:.0f}<500（64没生效？）")
    if qlag > 1.5: flags.append(f"ALERT:第13条后队列滞后{qlag:.2f}版>1.5")
    if stale13: flags.append(f"ALERT:第13条后过期步{stale13}")
# - 局外熵往上漂（2026-11-11 第 24,067 版加）：局外优势每批按均值、标准差标准化（train.py “分开标准化”），段变长后局外优势里实际回报的噪声多了，
#   标准化后真信号相对变小，固定的熵奖励（ent_coef_macro 0.05）相对变大。每 25 版中位：局外熵改前 0.300–0.312 → 改后 0.310、0.318、0.321、0.321、
#   0.334、0.332、0.330、0.335、0.338、0.357，选路熵 0.059–0.067 → 0.068–0.084，还在涨；战斗步 0.67–0.68 → 0.657–0.675。
#   局外熵第 10,000 版起一直在 0.284–0.31（开练初期最高约 0.37），选路全程 0.052–0.070。50 版中位局外熵超过 0.40 或选路熵超过 0.11 叫醒：
#   到时考虑把 ent_coef_macro 降下来，或退回 64
#   （这两条线第 13 条回退后照旧留着）
if v >= V12 + 50:
    if entm == entm and entm > 0.40: flags.append(f"ALERT:局外熵往上漂，50版中位{entm:.3f}>0.40")
    if entp == entp and entp > 0.11: flags.append(f"ALERT:选路熵往上漂，50版中位{entp:.3f}>0.11")
if v >= V13 + 50:
    if lvm == lvm and lvm > 0.32: flags.append(f"ALERT:第13条后局外价值损失50版中位{lvm:.4f}>0.32")
    if evm == evm and evm < 0.80: flags.append(f"ALERT:第13条后局外解释方差50版中位{evm:.3f}<0.80")
if v >= V13 + 50:
    if entm == entm and entm > 0.335: flags.append(f"ALERT:第13条后局外熵又往上漂，50版中位{entm:.3f}>0.335")
# 第 13 条满 500 版的读效果与训练整局 46.2% 那条线停用：第 14 条第 24,208 版上线，第 13 条只走了 380 版（第 23,828–24,207 版），
# 读数补在猎人决策记录第 13 条；第 14 条之后训练整局里混着跳过的战斗，整局通关要按局类别看（下面）。
# 第 14 条（第 24,208 版起跳过稳赢战斗；储君会话实现，用户：“直接全量开始做吧，然后现有训练可以继续停。”）的护栏，任一条成立就关掉跳过
# （配置 skip_easy 改 false 重启；回退备份 backups/v24207-20261111-1840-跳过稳赢战斗上线前.pt 与同名的滑动平均权重）：
# - 允许跳过的局与完全真打的局进第三幕第一个 Boss 时的血量比例：满 200 版起，第 14 条后（最多最近 500 版）两类相差超过 3 个百分点叫醒
#   （头 13 版 0.893 对 0.904；每版完全真打的局约 19 次进第三幕 Boss，200 版约 3,800 次，均值的标准误约 0.003）
# - 药水头校准（真打的够条件战斗上，预测用掉的比例对实际）：满 20 版起，最近 200 版相差超过 0.05 叫醒（头 13 版 0.237 对 0.235）
# - 卡住或引擎报错的局（采集端把引擎报错也按卡住结束，记进 stuck）：满 20 版起，第 14 条后超过整局数的 0.1% 叫醒（头 13 版 0）
# - 评测（sup_eval）：满 8 轮起最近 12 轮通关低于 59.5% 叫醒
# - 满 500 版（第 24,707 版）报 TRIGGER 读效果：常规评测通关、完全真打整局的通关、每小时整局数、局外熵（与上线前同样长度比）
# - 入训吞吐：跳过的战斗不产生步，每版整局多了、每版用时 20 → 22.5 秒，入训约 5,800 步 / 秒；第 11 条那条 5,500 的线第 14 条后改为 5,000
V14 = 24208
if v >= V14 + 1:
    w14 = [r for r in rows[-500:] if r["version"] >= V14]
    def chs(w, pre):
        c = {}
        for r in w:
            for k, x in (r.get("run_choices") or {}).items():
                if k.startswith(pre): c[k] = c.get(k, 0) + x
        return c
    c14 = chs(w14, ("SKIP", "B3_", "SKIPRUN_"))
    elig = sum(x for k, x in c14.items() if k.startswith("SKIP_ELIG:")); done = sum(x for k, x in c14.items() if k.startswith("SKIP_DONE:"))
    hr = {kd: c14.get(f"B3_HP:{kd}", 0) / max(1, c14.get(f"B3_N:{kd}", 0)) for kd in ("real", "skip")}
    wr = {kd: c14.get(f"SKIPRUN_WIN:{kd}", 0) / max(1, c14.get(f"SKIPRUN_N:{kd}", 0)) for kd in ("real", "skip")}
    c200 = chs([r for r in rows[-200:] if r["version"] >= V14], ("SKIP_POT_",))
    pn = c200.get("SKIP_POT_N", 0); pp, py = c200.get("SKIP_POT_P", 0) / max(1, pn), c200.get("SKIP_POT_Y", 0) / max(1, pn)
    w14a = [r for r in rows if r["version"] >= V14]
    stk, eps14 = sum(r.get("stuck") or 0 for r in w14a), sum(r.get("run_episodes") or 0 for r in w14a)
    print(f" 第14条后{len(w14a)}版 跳过{done / max(1, elig):.2f} 进三幕Boss血量 真打{hr['real']:.3f}/可跳{hr['skip']:.3f}（{c14.get('B3_N:real', 0):.0f}/{c14.get('B3_N:skip', 0):.0f}次）"
          f" 整局通关 真打{wr['real']:.3f}/可跳{wr['skip']:.3f} 药水 预测{pp:.3f}/实际{py:.3f} 卡住{stk}/{eps14}局", end="")
    if v >= V14 + 200 and abs(hr["skip"] - hr["real"]) > 0.03:
        flags.append(f"ALERT:第14条护栏 进三幕Boss血量 可跳{hr['skip']:.3f}对真打{hr['real']:.3f}，差超过3个百分点（关跳过）")
    if v >= V14 + 20 and pn and abs(pp - py) > 0.05:
        flags.append(f"ALERT:第14条护栏 药水头预测{pp:.3f}对实际{py:.3f}，差超过0.05（关跳过）")
    if v >= V14 + 20 and stk > 0.001 * max(1, eps14):
        flags.append(f"ALERT:第14条护栏 卡住或引擎报错{stk}/{eps14}局超过0.1%（关跳过）")
# 第 14 条满 500 版读效果已做（第 24,707 版，猎人决策记录第 14 条补记），这一行 TRIGGER 停用。
# 第 17 条（第 24,722 版起梯度累积 1 → 2，有效小批量 8,192；训练在第 24,721 版停过，做收尾的集成、搜索评测与胜率小模型，第 15、16 条）：
# - 满 500 版（第 25,221 版）报 TRIGGER 读效果：完全真打整局通关、评测最近 12 轮、Boss 练习、KL 与熵（对第 14 条那 500 版）；
# - 满 500 版起，最近 500 版完全真打整局通关低于 46.5%（第 14 条 48.48% 减 2 个百分点）叫醒；评测那一侧见 sup_eval.py。两条都成立就退回梯度累积 1。
V17 = 24722
V20 = 25228      # 第 20 条的起点（见下）
V21 = 25485      # 第 21 条（退回 skip_prob 0.8）的起点（见下）
if v >= V17 + 1:
    w17 = [r for r in rows[-500:] if V17 <= r["version"] < V20]
    c17 = chs(w17, ("SKIPRUN_",)); n17 = c17.get("SKIPRUN_N:real", 0); win17 = c17.get("SKIPRUN_WIN:real", 0) / max(1, n17)
    if v < V20: print(f" 第17条后{len([r for r in rows if r['version'] >= V17])}版 完全真打整局通关{win17:.3f}（{n17:.0f}局）", end="")
    if V17 + 500 <= v < V20 and n17 and win17 < 0.465: flags.append(f"ALERT:第17条后完全真打整局通关500版{win17:.3f}<46.5%")
# 第 17 条满 500 版读效果已做（第 25,223 版报出，读到第 25,221 版，猎人决策记录第 20 条）；上面这行 46.5% 的护栏与 TRIGGER 停用，由第 20 条的线接替。
# 第 20 条（第 25,228 版起跳过稳赢战斗的 skip_prob 0.8 → 1.0；用户：“读完再提到 1.0”）：
# - 满 250 版起，第 20 条后（最多最近 500 版）完全真打整局通关低于 51.1%（第 17 条 500 版 53.07% 减 2 个百分点；250 版约 6,600 局，标准误约 0.6 个百分点）
#   叫醒；评测那一侧见 sup_eval.py（62.3%）。两条都成立就退回 skip_prob 0.8（回退备份 backups/v25227-20261112-0600-跳过概率提到1前.pt）。
# - 第 14 条的三条护栏（进三幕 Boss 血量、药水头、卡住）照旧；第 20 条之后成立时先退回 0.8，不直接关掉跳过。
# - 满 500 版（第 25,727 版）报 TRIGGER 读效果：完全真打与可跳的局的整局通关、评测、Boss 练习、跳过比例、入训吞吐（对第 17 条那 500 版）。
if V20 + 1 <= v < V21:
    w20 = [r for r in rows[-500:] if r["version"] >= V20]
    c20 = chs(w20, ("SKIPRUN_",)); n20 = c20.get("SKIPRUN_N:real", 0); win20 = c20.get("SKIPRUN_WIN:real", 0) / max(1, n20)
    ns20 = c20.get("SKIPRUN_N:skip", 0); ws20 = c20.get("SKIPRUN_WIN:skip", 0) / max(1, ns20)
    print(f" 第20条后{len([r for r in rows if r['version'] >= V20])}版 完全真打整局通关{win20:.3f}（{n20:.0f}局）/可跳{ws20:.3f}", end="")
    if v >= V20 + 250 and n20 and win20 < 0.511: flags.append(f"ALERT:第20条后完全真打整局通关{win20:.3f}<51.1%")
    # 第 25,479 版补（猎人决策记录第 20 条补记）：上面这条在满 250 版时碰线（50.9%，7,471 局），评测最近 12 轮 64.8% 没碰 62.3%，按“两条都成立才退回”继续跑；
    # 巡检用 IGNORE='<51\.1%' 跳过上面这条，另加两条：完全真打低于 50.0%（第 17 条 53.07% 减 3 个百分点，约 4 个标准误）单独就退回 0.8；
    # 第三幕 Boss 练习最近 100 版低于 0.630（第 17 条 500 版 0.654，改后 250 版约 0.643）叫醒重新判断
    if v >= V20 + 250 and n20 and win20 < 0.500: flags.append(f"ALERT:第20条后完全真打整局通关{win20:.3f}<50.0%（单独退回0.8）")
    if v >= V20 + 100:
        w100 = [r for r in rows[-100:] if r["version"] >= V20]
        nb3 = sum(r["boss_n"][2] for r in w100 if r.get("boss_n")); pb3 = sum(r["boss_wins"][2] for r in w100 if r.get("boss_wins")) / max(1, nb3)
        print(f" 三幕Boss练习100版{pb3:.3f}", end="")
        if nb3 and pb3 < 0.630: flags.append(f"ALERT:第20条后三幕Boss练习100版{pb3:.3f}<0.630")
if V20 + 500 <= v < V21:
    flags.append("TRIGGER:第20条满500版读效果（完全真打与可跳整局通关、评测、Boss练习、跳过比例、入训）")
# 第 21 条（第 25,485 版起退回 skip_prob 0.8，配置与第 17 条相同 d56bbabd…；权重不回退）：第 20 条满 250 版时完全真打 50.9%、第三幕 Boss 练习一路降到 0.632，
# 降幅集中在进了第三幕 Boss 之后（进场血量、药水不变），评测没降。看退回之后练习与整局是否回升：
# - 满 250 版（第 25,734 版）报 TRIGGER 读：第三幕 Boss 练习、完全真打与可跳整局（对第 17 条 500 版与第 20 条 257 版），评测
# - 第三幕 Boss 练习最近 100 版（只算第 21 条之后）低于 0.620 叫醒（退回后还在往下，就不是跳过的问题，考虑权重回退到第 25,227 版）
# - 满 250 版起，第 21 条后完全真打整局通关低于 50.0% 叫醒
if v >= V21 + 1:
    w21 = [r for r in rows[-500:] if r["version"] >= V21]
    c21 = chs(w21, ("SKIPRUN_",)); n21 = c21.get("SKIPRUN_N:real", 0); win21 = c21.get("SKIPRUN_WIN:real", 0) / max(1, n21)
    ns21 = c21.get("SKIPRUN_N:skip", 0); ws21 = c21.get("SKIPRUN_WIN:skip", 0) / max(1, ns21)
    w100 = [r for r in rows[-100:] if r["version"] >= V21]
    nb3 = sum(r["boss_n"][2] for r in w100 if r.get("boss_n")); pb3 = sum(r["boss_wins"][2] for r in w100 if r.get("boss_wins")) / max(1, nb3)
    print(f" 第21条后{len([r for r in rows if r['version'] >= V21])}版 完全真打整局通关{win21:.3f}（{n21:.0f}局）/可跳{ws21:.3f} 三幕Boss练习最近{len(w100)}版{pb3:.3f}", end="")
    if len(w100) >= 50 and nb3 and pb3 < 0.620: flags.append(f"ALERT:第21条后三幕Boss练习{len(w100)}版{pb3:.3f}<0.620")
    if v >= V21 + 250 and n21 and win21 < 0.500: flags.append(f"ALERT:第21条后完全真打整局通关{win21:.3f}<50.0%")
# 第 21 条满 250 版读退回效果已做（第 25,749 版报出，读到第 25,734 版，猎人决策记录第 21 条补记：整局与评测都回来了），这一行 TRIGGER 停用；上面三条护栏照旧。
# 练习走平（同储君 sup_b）：第二、三幕 Boss 练习都涨不到 0.5 个百分点。第 8 条（第 18,638 版）每版的数据是原来的两倍，等两个窗口都落在改后才判；
# 第 9、10 条各推到改后 2,000 版。第 10 条之后（第 23,321 版）用户看了走势图判走平，直接做了预案的两步里的“学习率减半、每批加倍”（第 11 条）。
# 第 11 条之后每版的数据又翻一倍，窗口改为最近 500 版对前 500 版（≈ 第 8 条之前的 2,000 版数据），等两个窗口都落在改后（第 24,321 版）起判。
# 第 12 条那段作废、第 13 条把权重还原到第 23,827 版后，新时间线接着第 11 条往下走，原定仍从第 24,321 版起判（新窗口在第 13 条之后，旧窗口在第 11 条之后）。
# 第 14 条（第 24,208 版起跳过稳赢战斗）在判之前上线：整局里八成的稳赢战斗不再产生训练步，换成更多整局（每版整局 78 → 约 110；每版整局步约 67,000 没变），
# 训练数据变了，照前几条的做法等两个窗口都落在第 14 条之后（第 25,208 版）起判。
# 再走平时下一步是梯度累积 ×2（用户 11-10：“如果又出现走平，可以考虑再降学习力、提高批量，并增加梯度累积”；故障机器人第 225 条、储君第 23 条）
# 第 14 条满 500 版时训练已走平（第 23,900 版前后起完全真打约 48%），没等第 25,208 版就在第 17 条做了梯度累积 ×2；两个窗口都落在第 17 条之后（第 25,722 版）起再判，
# 再走平时下一步是学习率 1e-4 → 5e-5（muon_lr 同比例）。
# 第 20 条（第 25,228 版起 skip_prob 0.8 → 1.0）又改了整局里入训的战斗，照同样的做法等两个窗口都落在第 20 条之后（第 26,228 版）起判。
# 第 21 条（第 25,485 版）退回后，两个窗口都落在第 21 条之后（第 26,485 版）起判。
if v >= V21 + 1000 and len(rows) >= 1000:
    def bi(w, i):
        x = sum(r["boss_wins"][i] for r in w if r.get("boss_wins")); n = sum(r["boss_n"][i] for r in w if r.get("boss_n")); return x / n if n else float("nan")
    new, old = rows[-500:], rows[-1000:-500]
    d2, d3 = bi(new, 1) - bi(old, 1), bi(new, 2) - bi(old, 2)
    print(f" 练习500版涨幅 二幕Boss{d2*100:+.1f} 三幕Boss{d3*100:+.1f}", end="")
    if d2 < 0.005 and d3 < 0.005: flags.append(f"TRIGGER:练习走平（二幕Boss{d2*100:+.1f}、三幕Boss{d3*100:+.1f}个百分点/500版）")
print(" " + " ".join(flags) if flags else "")
