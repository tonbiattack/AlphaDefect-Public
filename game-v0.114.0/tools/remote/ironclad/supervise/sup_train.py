# 铁甲战士从零线监管：学习端检查（在训练机上跑，2026-11-20 起为 B；读日志末尾；2026-11-17，铁甲战士决策记录第 1 条，照恐惧模式故障机器人从零线 63fdcdd 与 tools/remote/fool/supervise/sup_train.py）。
# 输出一行读数；有问题的项以 ALERT: 开头（名字里不能有空格）。从零开训，优化器与练习份额同猎人从零（4163619）与愚者首版，另开愚者后来留下的收尾自举、格式 28 与读数头。
# 线的依据是猎人从零开训那段（标准模式，同一份配置，每版 32,768 步），每 100 版窗口：
#   第 500 版 17.3 层、第 1,000 版 20.1 层、第 2,000 版 21.4 层、第 4,000 版 24.1 层、第 8,000 版 29.1 层（通关 0.7%）；
#   KL 中位 0.010–0.014、裁剪 0.09–0.12；熵 战斗步 1.16 → 0.76、局外 0.62 → 0.32、选路 0.21 → 0.05。
# 愚者从零（同一份优化器设置，标准模式）：第 162 版 14.3 层，第 902 版 23.1 层；学得慢的两条线只是叫醒来看，不是自动处置的依据。
import json, os, re, shutil, statistics as S, time
R = "/root/.local/share/sls2/runs/ironclad"
L = f"{R}/log.jsonl"
flags = []
# 学习端反复崩溃、被守护拉起：train.out 里最近 1 小时“启动训练”超过 2 次
starts = []
try:
    for line in open(f"{R}/train.out", encoding="utf-8", errors="replace"):
        m = re.match(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) 启动训练", line)
        if m:
            starts.append(time.mktime(time.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")))
except OSError:
    pass
restarts = sum(1 for t in starts if time.time() - t < 3600)
if restarts > 2: flags.append(f"ALERT:学习端1小时内启动{restarts}次")
# 显存不够（亡灵契约师线 2026-11-17 查到：第二幕精英十足蜈蚣的节段死后回满血接回，无进展判输判不下来，单场练习打满 max_steps_combat，
# 整场一次交给学习端撑爆显存，约每小时崩一次被守护拉起，上面的“1 小时 2 次”看不出来）：train.out 里出现就叫醒
try:
    oom = sum(1 for line in open(f"{R}/train.out", encoding="utf-8", errors="replace") if "OutOfMemoryError" in line)
except OSError:
    oom = 0
if oom: flags.append(f"ALERT:学习端显存不够{oom}次")
# 打满步数上限的局（stuck.jsonl）：单场练习打满 max_steps_combat、整局里一场战斗打满 max_steps_fight（第 4 条起，亡灵契约师线 8baabf4 + 9cf8bd4），
# 出现就叫醒来看（读过的用 IGNORE 按场数跳过，场数再变会再叫醒）；整局打满 max_steps_run 到第 5,899 版只有 1 局（第 3,589 版，第二幕第 28 层选路来回 2 选 1 打满 20,000 步），再多就来看
lim = {"combat": 0, "fight": 0, "run": 0}
try:
    for line in open(f"{R}/stuck.jsonl", encoding="utf-8", errors="replace"):
        try:
            j = json.loads(line)
        except ValueError:
            continue
        if j.get("reason") == "超过步数上限": lim["run" if j.get("mode") == "run" else "combat"] += 1
        elif j.get("reason") == "单场战斗超过步数上限": lim["fight"] += 1
except OSError:
    pass
if lim["combat"]: flags.append(f"ALERT:单场练习打满步数上限{lim['combat']}场")
if lim["fight"]: flags.append(f"ALERT:整局里单场战斗打满步数上限{lim['fight']}场")
if lim["run"] > 1: flags.append(f"TRIGGER:整局打满步数上限{lim['run']}局")
# 引擎看门狗（engine_watchdog.sh，第 9 条补记）杀掉的失控引擎：有就叫醒来看（读过的用 IGNORE 按次数跳过）
try:
    wd = sum(1 for _ in open(f"{R}/engine_watchdog.log", encoding="utf-8", errors="replace"))
except OSError:
    wd = 0
if wd: flags.append(f"ALERT:引擎看门狗杀掉{wd}次")
if os.path.exists(f"{R}/STOP"): flags.append("ALERT:运行目录有STOP")
free = shutil.disk_usage("/root").free / 2**30
if free < 80: flags.append(f"ALERT:磁盘只剩{free:.0f}GB")
rows = []
if os.path.exists(L):
    with open(L, "rb") as f:
        f.seek(max(0, os.path.getsize(L) - 30_000_000))
        raw = f.read().split(b"\n")[1:] if os.path.getsize(L) > 30_000_000 else f.read().split(b"\n")
    for l in raw:
        try:
            r = json.loads(l)
        except ValueError:
            continue
        if "version" not in r:
            continue
        while rows and rows[-1]["version"] >= r["version"]:     # 回退过权重时去掉作废的时间线
            rows.pop()
        rows.append(r)
if not rows:
    since = time.time() - (starts[-1] if starts else time.time())
    if since > 900: flags.append(f"ALERT:启动{since:.0f}秒还没有版本")
    print(f"S 还没有版本（启动后{since:.0f}s） 磁盘余{free:.0f}GB" + "".join(" " + x for x in flags)); raise SystemExit
w50, w100, w200 = rows[-50:], rows[-100:], rows[-200:]
v = rows[-1]["version"]
age = time.time() - (rows[-1].get("time") or 0)
med = lambda k, w=w50: S.median([r[k] for r in w if isinstance(r.get(k), (int, float))] or [float("nan")])
kl, ent, entm, lvm, clip, entp = med("approx_kl"), med("entropy"), med("entropy_macro"), med("loss_v_macro"), med("clip_frac"), med("entropy_map")
entc = S.median([(r["entropy"] - r["macro_frac"] * r["entropy_macro"]) / (1 - r["macro_frac"]) for r in w50
                 if all(isinstance(r.get(k), (int, float)) for k in ("entropy", "entropy_macro", "macro_frac")) and 0 <= r["macro_frac"] < 1] or [float("nan")])
nan = sum(r.get("nan_skipped") or 0 for r in w50)
early = sum(1 for r in w50 if r.get("epochs", 2) < 2)
ep = sum(r.get("run_episodes") or 0 for r in w100)
floor = sum((r.get("run_floor_mean") or 0) * (r.get("run_episodes") or 0) for r in w100) / max(1, ep)
win = sum((r.get("run_win_rate") or 0) * (r.get("run_episodes") or 0) for r in w100) / max(1, ep)
a2 = sum((r.get("run_reach_act2") or 0) * (r.get("run_episodes") or 0) for r in w100) / max(1, ep)
def rate(k):
    x = sum(r[k][0] for r in w200 if r.get(k)); n = sum(r[k][1] for r in w200 if r.get(k))
    return x / n if n else float("nan")
def brate(i):
    x = sum(r["boss_wins"][i] for r in w200 if r.get("boss_wins")); n = sum(r["boss_n"][i] for r in w200 if r.get("boss_n"))
    return x / n if n else float("nan")
evm = S.median([r["health"]["ev"]["macro"] for r in w50 if (r.get("health") or {}).get("ev", {}).get("macro") is not None] or [float("nan")])
stk, eps200 = sum(r.get("stuck") or 0 for r in w200), sum(r.get("run_episodes") or 0 for r in w200)
# 入训吞吐：相邻两版间隔超过 120 秒的跳过（停机、重启的那几分钟不算）
pr = [(b["total_steps"] - a["total_steps"], b["time"] - a["time"]) for a, b in zip(w50, w50[1:])
      if isinstance(a.get("total_steps"), (int, float)) and isinstance(b.get("total_steps"), (int, float)) and 0 < b["time"] - a["time"] <= 120]
sps = sum(s for s, _ in pr) / max(1e-9, sum(t for _, t in pr)) if pr else float("nan")
print(f"S v{v} 日志{age:.0f}s前 KL{kl:.4f} 裁剪{clip:.3f} 熵 战斗步{entc:.3f}/局外{entm:.3f}/选路{entp:.3f} 局外价值损失{lvm:.4f} 局外解释方差{evm:.3f} "
      f"早停{early}/50 非有限{nan} 整局（100版{ep}局）{floor:.2f}层/进二幕{a2:.3f}/通关{win:.3f} 课程进度{rows[-1].get('progress')} "
      f"练习 一幕{rate('act1_monster'):.3f}/{rate('act1_elite'):.3f} 二幕{rate('act2_monster'):.3f}/{rate('act2_elite'):.3f} Boss{brate(0):.3f}/{brate(1):.3f}/{brate(2):.3f} "
      f"入训{sps:,.0f}步/秒 卡住{stk}/{eps200}局 磁盘余{free:.0f}GB", end="")
if age > 600: flags.append(f"ALERT:学习端日志{age:.0f}秒没更新")
if nan: flags.append(f"ALERT:非有限梯度{nan}")
if early > 15: flags.append(f"ALERT:早停{early}/50")
if kl == kl and kl > 0.04: flags.append(f"ALERT:KL中位{kl:.4f}")
# 熵坍塌：最近 50 版中位。猎人从零开训的头 8,000 版 战斗步 ≥ 0.76、局外 ≥ 0.32、选路 ≥ 0.05，之后续训到第 29,000 版 0.67 / 0.30 / 0.056
# 2026-11-18 第 8,148 版战斗步熵 0.589：第 2,600 版前后约 0.69 起缓慢下降（每 1,000 版约 0.02），同期评测 35 → 42 层、通关 0.1% → 9%，
# 与愚者 2026-11-15 第 21,134 版碰 0.55 时一样，是策略变笃定、不是坍塌。照愚者 sup_train.py 的改法：绝对线放到 0.50，
# 另加骤降检查：50 版中位比之前 500 版（倒数第 550–51 版）的中位低 0.04 以上就叫醒（铁甲战士决策记录第 4 条补记）
def entc_of(w):
    return S.median([(r["entropy"] - r["macro_frac"] * r["entropy_macro"]) / (1 - r["macro_frac"]) for r in w
                     if all(isinstance(r.get(k), (int, float)) for k in ("entropy", "entropy_macro", "macro_frac")) and 0 <= r["macro_frac"] < 1] or [float("nan")])
if v >= 50:
    # 2026-11-19 第 21,361 版碰到 0.50（0.496）：每 2,000 版中位 0.695 → 0.656 → 0.622 → 0.599 → 0.599 → 0.579 → 0.566 → 0.558 → 0.529 → 0.512，
    # 同期评测通关 0.2% → 63.1%、局外熵 0.38 → 0.32、选路熵约 0.07、KL 平稳，没有骤降，判为策略变笃定。照愚者的做法绝对线再放到 0.45，骤降检查不变（第 10 条补记）
    if entc == entc and entc < 0.45: flags.append(f"ALERT:熵坍塌？战斗步熵50版中位{entc:.3f}<0.45")
    if len(rows) >= 550:
        entc_prev = entc_of(rows[-550:-50])
        print(f" 战斗步熵前500版{entc_prev:.3f}", end="")
        if entc == entc and entc_prev == entc_prev and entc < entc_prev - 0.04:
            flags.append(f"ALERT:熵骤降？战斗步熵50版中位{entc:.3f}比之前500版{entc_prev:.3f}低0.04以上")
    if entm == entm and entm < 0.20: flags.append(f"ALERT:熵坍塌？局外熵50版中位{entm:.3f}<0.20")
    if entp == entp and entp < 0.03: flags.append(f"ALERT:熵坍塌？选路熵50版中位{entp:.3f}<0.03")
# 卡住或引擎报错的局（采集端把引擎报错也按卡住结束，记进 stuck）：最近 200 版超过整局数的 0.1%（猎人全程 0）
if eps200 >= 1000 and stk > 0.001 * eps200: flags.append(f"ALERT:卡住或引擎报错{stk}/{eps200}局超过0.1%")
# 吞吐：第 100 版起最近 50 版低于每秒 2,000 步（C 只有 6 × 27 个引擎）；B（8 × 42 个引擎，第 24 条起）低于 3,000
SPS_MIN = 2000 if v > 25462 else 2500 if v > 24279 else 2000      # 第 26 条回灌每步多一遍编码器（2,500）；第 27 条格式 29 编码更慢（2,000）
if v >= 100 and sps == sps and sps < SPS_MIN: flags.append(f"ALERT:入训{sps:.0f}步/秒<{SPS_MIN}")
# 学得慢：第 1,000 版起 100 版窗口整局平均低于 12 层（猎人此时 20.1 层）、第 3,000 版起低于 18 层（猎人 23.7 层）
if v >= 1000 and ep and floor < 12: flags.append(f"ALERT:学得慢？第{v}版整局{floor:.2f}层<12")
if v >= 3000 and ep and floor < 18: flags.append(f"ALERT:学得慢？第{v}版整局{floor:.2f}层<18")
# 候选未来读数头（愚者第 25 条）：归一化均方误差 50 版中位、残差输出层权重范数；损失不是有限值就叫醒
if any("fh_loss" in r for r in w50):
    fhl = med("fh_loss"); fhw = rows[-1].get("fh_res_w", float("nan"))
    print(f" 读数头 损失{fhl:.3f} 残差权重{fhw:.4f}", end="")
    if not (fhl == fhl and fhl < 1e6): flags.append(f"ALERT:读数头损失{fhl}")
# 战斗收尾自举（愚者第 8 条）：fight_end 每批按幕去均值前的平均，最近 50 版各幕的中位（看收尾落差有没有慢慢变小）
fe = {}
for r in w50:
    for a_, (n_, m_) in (r.get("fight_end") or {}).items():
        if n_ >= 32: fe.setdefault(a_, []).append(m_)
if fe: print(" 收尾落差 " + "/".join(f"{k}幕{S.median(x):+.3f}" for k, x in sorted(fe.items())), end="")
# 到了该读效果的版本报 TRIGGER（名字里不能有空格），读过的用 supervise.sh 的 IGNORE 跳过（如 IGNORE='第15条满125版'）：
# 铁甲战士决策记录第 25 条（千种池复测没有提升：撤销加血与余血奖励，权重退回第 24,279 版）与第 26 条（回灌上线）同时生效，
# 新时间线第 24,280 版起（作废的加血时间线第 24,280–24,733 版由上面按版本去重去掉）。满 125、250、500 版读效果
# 第 27 条（观测格式 29，第 25,463 版起）满 125、250、500 版读效果；第 26 条的读点已用完，去掉
for k in (125, 250, 500):
    if v >= 25462 + k: flags.append(f"TRIGGER:第27条满{k}版读效果")
# 第 28 条（用户定“只跑不改”）：从第 25,968 版起每满 1,000 版停训做一次千种池配对（读过的用 IGNORE 按次数跳过）
if v >= 25968 + 1000: flags.append(f"TRIGGER:千种池第{(v - 25968) // 1000}次")
# 训练端护栏：第 27 条起满 125 版，训练整局最近 125 版通关低于 51.2%（改前约 250 版 53.2% 减 2 个百分点）叫醒（第 26 条的 52.9% 由这条接替）
if v >= 25462 + 125:
    w125 = rows[-125:]; e125 = sum(r.get("run_episodes") or 0 for r in w125)
    win125 = sum((r.get("run_win_rate") or 0) * (r.get("run_episodes") or 0) for r in w125) / max(1, e125)
    print(f" 整局125版通关{win125:.3f}", end="")
    if win125 < 0.512: flags.append(f"ALERT:第27条训练端（整局125版通关{win125:.3f}<0.512）")
# 回灌读数（第 26 条）：第二遍对第一遍的 KL、首选相同的比例、rcy_ 各层权重范数，最近 50 版中位
rc = [r for r in w50 if r.get("rcy_kl") is not None]
if rc:
    print(f" 回灌 KL{S.median(r['rcy_kl'] for r in rc):.4f} 首选相同{S.median(r['rcy_agree'] for r in rc):.4f} 权重{rc[-1]['rcy_w']:.4f}", end="")
lk = [r["luck"] for r in rows[-50:] if r.get("luck")]
if lk:
    print(f" 运气校正 每版{S.median([x['n'] for x in lk]):.0f}步 运气均方根{S.median([x['rms'] for x in lk]):.3f} δ均方根{S.median([x['d_rms'] for x in lk]):.3f}→{S.median([x['d_rms_cv'] for x in lk]):.3f}", end="")
# 第 6 条把第二幕真实局面的份额又减了四成：改后满 100 版起，第二幕精英练习（最近 200 版）低于 0.94、普通战低于 0.97 叫醒（愚者第 13 条的线；改前 0.972 / 0.995）。
# 第 3 条的线（0.87 / 0.95）由这条接替；“到 0.97”的 TRIGGER 已用掉，去掉
if v >= 18145 + 100:
    if rate("act2_elite") < 0.94: flags.append(f"ALERT:二幕精英练习回落{rate('act2_elite'):.3f}<0.94")
    if rate("act2_monster") < 0.97: flags.append(f"ALERT:二幕普通战练习回落{rate('act2_monster'):.3f}<0.97")
# 练习走平的提醒（照愚者第 9 条补记的 TRIGGER“练习走平”）：最近 N 版对前 N 版，第二、三幕 Boss 练习都涨不到 0.5 个百分点就报。
# 第 7 条起每版数据翻倍，照愚者第 10 条窗口 N 从 1,000 改为 500；第 8 条起再翻倍，照愚者第 12 条改为 250；第 9 条（梯度累积）照愚者第 14 条推到改后满 500 版（第 20,487 版）起再判；第 10 条（学习率再减半）再推到第 21,783 版起；第 11 条（梯度累积 2 → 4）推到第 22,097 版起；第 14 条（免调度）推到第 22,944 版起，第 15 条（退回）照旧；不满时仍打印涨幅。
# 第 24 条（训练加血）作废、第 26 条（回灌）从第 24,280 版起：推到满 2 × 250 版（第 24,779 版）起再判。
# 降噪按第 2 条补记暂不做（用户：“我人工判定有学习趋势，暂时不要做降噪处理。”；第二次走平时：“再等等”），报出后只向用户报告，由用户决定
NW = 250 if v >= 19479 else 500 if v >= 18355 else 1000
if len(rows) >= 2 * NW:
    def br(i, w):
        x = sum(r["boss_wins"][i] for r in w if r.get("boss_wins")); n = sum(r["boss_n"][i] for r in w if r.get("boss_n"))
        return x / n if n else float("nan")
    a_, b_ = rows[-NW:], rows[-2 * NW:-NW]
    d2, d3 = br(1, a_) - br(1, b_), br(2, a_) - br(2, b_)
    print(f" {NW}版涨幅 二幕Boss{d2 * 100:+.1f}/三幕Boss{d3 * 100:+.1f}", end="")
    if v >= 25462 + 2 * NW and d2 < 0.005 and d3 < 0.005: flags.append(f"TRIGGER:练习走平（二幕Boss{d2 * 100:+.1f}/三幕Boss{d3 * 100:+.1f}）")
print("".join(" " + x for x in flags))
