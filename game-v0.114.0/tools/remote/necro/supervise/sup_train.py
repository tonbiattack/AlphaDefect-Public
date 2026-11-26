# 亡灵契约师监管：学习端检查（在 B 上跑，读日志末尾；2026-11-17 亡灵契约师从零开训，照愚者从零开训时那一版 sup_train.py（4237f40）的通用部分，
# 加愚者后来补的读数头读数、战斗熵骤降检查）。输出一行读数；有问题的项以 ALERT: 开头，到了决策条件的以 TRIGGER: 开头（名字里不能有空格）。
# 线的依据是愚者从零开训那段（同一份从零配置 2d0321d，每版 32,768 步；愚者第 1 版起的 log.jsonl，每 100 版窗口的训练整局）：
#   第 100 版 10.0 层、第 200 版 14.5 层、第 500 版 19.9 层（进第二幕 0.33）、第 1,000 版 24.5 层（0.58）、第 2,000 版 24.9 层、第 3,000 版 27.9 层（0.71）；
#   KL 中位 0.010–0.013；熵（总）0.90 → 0.67、局外 0.57 → 0.36、选路 0.12 → 0.06–0.09。
# 亡灵契约师是另一个角色，学得快慢可能不同，“学得慢”的线只是叫醒来看，不是自动处置的依据。
import json, os, re, shutil, statistics as S, time
R = "/root/.local/share/sls2/runs/necro"
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
if os.path.exists(f"{R}/STOP"): flags.append("ALERT:运行目录有STOP")
free = shutil.disk_usage("/root").free / 2**30
if free < 80: flags.append(f"ALERT:B磁盘只剩{free:.0f}GB")
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
def entc_of(w):
    return S.median([(r["entropy"] - r["macro_frac"] * r["entropy_macro"]) / (1 - r["macro_frac"]) for r in w
                     if all(isinstance(r.get(k), (int, float)) for k in ("entropy", "entropy_macro", "macro_frac")) and 0 <= r["macro_frac"] < 1] or [float("nan")])
entc = entc_of(w50)
nan = sum(r.get("nan_skipped") or 0 for r in w50)
early = sum(1 for r in w50 if r.get("epochs", 2) < 2)
ep = sum(r.get("run_episodes") or 0 for r in w100)
floor = sum((r.get("run_floor_mean") or 0) * (r.get("run_episodes") or 0) for r in w100) / max(1, ep)
win = sum((r.get("run_win_rate") or 0) * (r.get("run_episodes") or 0) for r in w100) / max(1, ep)
a2 = sum((r.get("run_reach_act2") or 0) * (r.get("run_episodes") or 0) for r in w100) / max(1, ep)
def rate(k, w=None):
    w = w200 if w is None else w
    x = sum(r[k][0] for r in w if r.get(k)); n = sum(r[k][1] for r in w if r.get(k))
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
# 候选未来读数头（愚者第 25 条，本线从第 1 版就开）：归一化均方误差 50 版中位、残差输出层权重范数；损失不是有限值就叫醒
if any("fh_loss" in r for r in w50):
    fhl = med("fh_loss"); fhw = rows[-1].get("fh_res_w", float("nan"))
    print(f" 读数头 损失{fhl:.3f} 残差权重{fhw:.4f}", end="")
    if not (fhl == fhl and fhl < 1e6): flags.append(f"ALERT:读数头损失{fhl}")
if age > 600: flags.append(f"ALERT:学习端日志{age:.0f}秒没更新")
if nan: flags.append(f"ALERT:非有限梯度{nan}")
if early > 15: flags.append(f"ALERT:早停{early}/50")
if kl == kl and kl > 0.04: flags.append(f"ALERT:KL中位{kl:.4f}")
# 熵坍塌：最近 50 版中位（愚者从零开训头 4,000 版 总熵 ≥ 0.66、局外 ≥ 0.36、选路 ≥ 0.06）。战斗步熵另查骤降（愚者 0b77de8）：
# 50 版中位比之前 500 版（倒数第 550–51 版）的中位低 0.04 以上就叫醒
if v >= 50:
    if entc == entc and entc < 0.50: flags.append(f"ALERT:熵坍塌？战斗步熵50版中位{entc:.3f}<0.50")
    if entm == entm and entm < 0.20: flags.append(f"ALERT:熵坍塌？局外熵50版中位{entm:.3f}<0.20")
    if entp == entp and entp < 0.03: flags.append(f"ALERT:熵坍塌？选路熵50版中位{entp:.3f}<0.03")
    # 骤降检查第 2,000 版起才判（2026-11-17 第 642 版误报 0.859 → 0.786：从零开训的头几百版熵本来就在快速下降，之前 500 版的窗口含着开训时的高熵；
    # 当时第 525、585、642 版的 50 版中位 0.783、0.791、0.786，已经平了）。之前只看上面的绝对线
    # 改配置、重启后之前 500 版的窗口跨着改动，满 550 版再判骤降；这段时间看绝对线（版本起点, 绝对线）：
    #   第 6 条补记：第 8,877 版练习构成改（Boss 0.20 → 0.30），改前约 0.70–0.72、改后约 0.67，线 0.62；
    #   第 7 条补记：第 9,553 版补回第一幕 Boss 份额后的重启，0.69–0.71 → 0.63–0.64 持续约 75 版（评测同时到 37.07 层的新高），线 0.58
    hold = [(c, line) for c, line in ((8877, 0.62), (9553, 0.58), (18367, 0.60), (19450, 0.58), (21310, 0.56), (23559, 0.56), (24044, 0.56)) if c <= v < c + 550]
    for c, line in hold:
        if entc == entc and entc < line: flags.append(f"ALERT:第{c}版改动后战斗步熵{entc:.3f}<{line}")
    if len(rows) >= 600 and v >= 2000 and not hold:
        before = entc_of(rows[-550:-50])
        if entc == entc and before == before and before - entc > 0.04: flags.append(f"ALERT:战斗步熵骤降{before:.3f}→{entc:.3f}")
# 卡住或引擎报错的局（采集端把引擎报错也按卡住结束，记进 stuck）：最近 200 版超过整局数的 0.1%（愚者全程约 0）
# 第 7 条补记二：十足蜈蚣节段反复接回的死循环被步数上限截住（练习“超过步数上限”、整局“单场战斗超过步数上限”）也记进 stuck，
# 第 9,750–9,910 版就有 11 局，会把这条线占满、盖住真正的引擎卡住或报错。按 stuck.jsonl 分开数（同一窗口：最近 200 版）：
#   引擎判卡住、引擎报错等：超过整局数的 0.1% 叫醒（原线）；步数上限截住的循环：超过 0.5% 叫醒（循环不只出在十足蜈蚣、或明显变多时）
caps = other = 0
v200 = rows[-200]["version"] if len(rows) >= 200 else 0
try:
    for line in open(f"{R}/stuck.jsonl", encoding="utf-8"):
        try:
            sr = json.loads(line)
        except ValueError:
            continue
        if sr.get("version", -1) >= v200:
            if "步数上限" in sr.get("reason", ""): caps += 1
            else: other += 1
except OSError:
    pass
print(f" 卡住分类 步数上限{caps}/其他{other}", end="")
if eps200 >= 1000 and other > 0.001 * eps200: flags.append(f"ALERT:卡住或引擎报错{other}/{eps200}局超过0.1%")
if eps200 >= 1000 and caps > 0.005 * eps200: flags.append(f"ALERT:步数上限截住的循环{caps}/{eps200}局超过0.5%")
# 吞吐：第 100 版起最近 50 版低于每秒 2,500 步（愚者从零约 6,800，换新引擎后的愚者约 7,300）
if v >= 100 and sps == sps and sps < 2500: flags.append(f"ALERT:入训{sps:.0f}步/秒<2500")
# 学得慢：愚者同期的大约六成（第 500 版 19.9 层、第 1,000 版 24.5 层、第 3,000 版 27.9 层）
if v >= 500 and ep and floor < 12: flags.append(f"ALERT:学得慢？第{v}版整局{floor:.2f}层<12")
if v >= 1000 and ep and floor < 15: flags.append(f"ALERT:学得慢？第{v}版整局{floor:.2f}层<15")
if v >= 3000 and ep and floor < 20: flags.append(f"ALERT:学得慢？第{v}版整局{floor:.2f}层<20")
# 照愚者走过的顺序调份额（亡灵契约师决策记录第 1 条“之后怎么走”）：第二幕普通战练习到 0.96（最近 200 版，第 1,000 版起）时报一次，
# 读评测死因再定要不要把第二幕份额挪给 Boss、第一幕 Boss 奖励降到 0.1（愚者第 4 条在第 3,765 版做，当时第二幕普通战练习 0.96）
# 第 3 条补记五：窗口 200 → 500 版（猎人第 6 条、储君第 18 条都按最近 500 版到 96% 才挪；第 8,284 版 200 版 0.961、500 版 0.956、1,500 版 0.952，还在涨）
# 第 5 条（第 8,877 版起，第二幕普通战份额已减半）：这条触发已用过，改为两项——
#   护栏：改后满 500 版起，第二幕普通战练习最近 500 版跌破 0.945（改前 0.9606；储君第 18 条的线）叫醒查；
#   下一次挪份额：第二幕精英练习最近 500 版到 0.92（猎人第 7 条），把第二幕真实局面的份额再挪给死得最多的 Boss（愚者第 9、13 条）
if v >= 8877 + 500 and rate("act2_monster", rows[-500:]) < 0.945: flags.append(f"ALERT:二幕普通战练习500版跌到{rate('act2_monster', rows[-500:]):.4f}")
# 第 8 条（第 18,367 版起，第二幕真实局面 0.37 → 0.25）：上面“到 0.92”的触发已用过，换成——
#   护栏：改后满 500 版起，第二幕精英练习最近 500 版跌破 0.90（改前 0.9204）叫醒；
#   下一次挪份额：第二幕精英练习最近 500 版到 0.97（猎人第 10 条），第二幕真实局面再挪给死得最多的 Boss（愚者第 13 条、猎人第 10 条：0.25 → 0.15）
# 第 8 条补记二：第 19,142 版 0.8989 报过一次（改前 0.9204），评测里每局死在第二幕精英 0.038 → 0.040、整体更好，线放到 0.88（改前 −4 个百分点）
# 第 14 条（第 24,044 版起，第二幕真实局面 0.25 → 0.15，照猎人第 10 条；先例里第二幕份额到 0.15 就不再挪）：“到 0.97”的触发已用过、撤掉；
# 护栏改为改后满 500 版起第二幕精英练习最近 500 版跌破 0.94（改前 0.9701，−3 个百分点）叫醒
if v >= 24044 + 500 and rate("act2_elite", rows[-500:]) < 0.94: flags.append(f"ALERT:二幕精英练习500版跌到{rate('act2_elite', rows[-500:]):.4f}")
# 评测与 Boss 练习都走平（决策记录第 3 条补记四）：第 4,600 版起，最近 1,000 版评测平均层数对版本号的直线回归折成每 500 版不到 0.3 层，
# 并且第二、三幕 Boss 练习每场胜率最近 1,000 版比之前 1,000 版都涨不到 0.5 个百分点（储君第 18 条补记的练习走平判法），两项同时成立才叫醒。
# 只看评测会在 0.3 层上下来回报（评测每 500 版涨约 0.5 层时斜率的标准误约 0.25 层）；第 3 条及其补记里三次判“不动”的理由都是 Boss 练习还在涨。
# 叫醒后按第 1 条“之后怎么走”与第 3 条末尾的备选定下一项（挪份额、第一幕 Boss 奖励、降噪阶梯）
def brate_w(i, w):
    x = sum(r["boss_wins"][i] for r in w if r.get("boss_wins")); n = sum(r["boss_n"][i] for r in w if r.get("boss_n"))
    return x / n if n else float("nan")
# 第 5 条：改后 1,000 版内（两个 1,000 版窗口都落在改后之前）不判，免得练习构成的变化混进来（储君第 18 条：改后 750 版内不判学习率走平）
# 第 9 条（第 19,450 版起学习率减半、每批加倍）：改后 2,000 版内不判（猎人第 8 条：练习走平的判断改为改后 2,000 版起，两个 1,000 版窗口都落在改后）；
# 再走平时下一步是降噪阶梯第 ② 步（再减半、再加倍，愚者第 12 条、猎人第 11 条）
# 第 11 条（第 21,310 版起降噪第 ② 步）：改后 2,000 版内不判；再走平时下一步是第 ③ 步梯度累积 ×2（愚者第 14 条；本线已是小批 2,048 × 累积 2，改为 × 4，有效小批 8,192）
# 第 13 条（第 23,559 版起降噪第 ③ 步）：改后 2,000 版内不判；降噪阶梯走完，再走平时下一步是免调度优化器（愚者第 15、16 条）
if v >= 23559 + 2000 and len(rows) >= 2000:
    d2 = brate_w(1, rows[-1000:]) - brate_w(1, rows[-2000:-1000]); d3 = brate_w(2, rows[-1000:]) - brate_w(2, rows[-2000:-1000])
    ev = []
    try:
        for l in open("/root/.local/share/sls2/runs/necro-eval/eval.jsonl"):
            if l.strip():
                r = json.loads(l)
                if r.get("runs") != 192: continue
                while ev and ev[-1]["version"] >= r["version"]: ev.pop()
                ev.append(r)
    except OSError:
        pass
    ew = [r for r in ev if ev and r["version"] > ev[-1]["version"] - 1000]
    d = float("nan")
    if len(ew) >= 16:
        xs = [r["version"] for r in ew]; ys = [r["floor_mean"] for r in ew]; mx, my = S.mean(xs), S.mean(ys)
        sxx = sum((x - mx) ** 2 for x in xs)
        d = 500 * sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else float("nan")
    # 第 7 条补记四：通关率涨起来以后层数说明不了进步（储君第 18 条补记：层数停在 40–42、通关 5% → 13%），评测再看通关率的回归（每 1,000 版）：
    # 层数每 500 版涨不到 0.3 层、且通关每 1,000 版涨不到 1 个百分点，才算评测走平
    dw = float("nan")
    if len(ew) >= 16:
        ws = [r["win_rate"] for r in ew]; mw = S.mean(ws)
        dw = 1000 * sum((x - mx) * (y - mw) for x, y in zip(xs, ws)) / sxx if sxx else float("nan")
    print(f" Boss练习1000版 二幕{d2 * 100:+.1f}/三幕{d3 * 100:+.1f}个百分点 评测回归每500版{d:+.2f}层 通关每1000版{dw * 100:+.1f}个百分点", end="")
    if d == d and dw == dw and d < 0.3 and dw < 0.01 and d2 < 0.005 and d3 < 0.005:
        flags.append(f"TRIGGER:评测与Boss练习都走平（评测{d:+.2f}层、通关{dw * 100:+.1f}，二幕{d2 * 100:+.1f}，三幕{d3 * 100:+.1f}）")
print("".join(" " + x for x in flags))
