# Fool 监管：学习端检查（在 B 上跑，读日志末尾；2026-11-13 Fool 从零开训，照 tools/remote/silent/supervise/sup_train.py 的通用部分）。
# 输出一行读数；有问题的项以 ALERT: 开头，到了决策条件的以 TRIGGER: 开头（名字里不能有空格）。
# 线的依据是猎人从零开训那段（2026-11-09 起，同一份从零配置 4163619，每版 32,768 步），每 100 版窗口：
#   第 500 版 17.3 层、第 1,000 版 20.1 层、第 2,000 版 21.4 层、第 4,000 版 24.1 层、第 8,000 版 29.1 层（通关 0.7%）；
#   KL 中位 0.010–0.014、裁剪 0.09–0.12；熵 战斗步 1.16 → 0.76、局外 0.62 → 0.32、选路 0.21 → 0.05；入训 5,900–7,500 步 / 秒；卡住 0、早停 0。
# Fool 是新角色，学得快慢可能和猎人不同，下面“学得慢”的两条线只是叫醒来看，不是自动处置的依据。
import json, os, re, shutil, statistics as S, time
R = "/root/.local/share/sls2/runs/fool"
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
if v >= 50:
    if entc == entc and entc < 0.55: flags.append(f"ALERT:熵坍塌？战斗步熵50版中位{entc:.3f}<0.55")
    if entm == entm and entm < 0.20: flags.append(f"ALERT:熵坍塌？局外熵50版中位{entm:.3f}<0.20")
    if entp == entp and entp < 0.03: flags.append(f"ALERT:熵坍塌？选路熵50版中位{entp:.3f}<0.03")
# 卡住或引擎报错的局（采集端把引擎报错也按卡住结束，记进 stuck）：最近 200 版超过整局数的 0.1%（猎人全程 0）
if eps200 >= 1000 and stk > 0.001 * eps200: flags.append(f"ALERT:卡住或引擎报错{stk}/{eps200}局超过0.1%")
# 吞吐：第 100 版起最近 50 版低于每秒 2,500 步（Fool 头几版约 4,700；猎人从零 5,900–7,500）
if v >= 100 and sps == sps and sps < 2500: flags.append(f"ALERT:入训{sps:.0f}步/秒<2500")
# 学得慢：第 1,000 版起 100 版窗口整局平均低于 12 层（猎人此时 20.1 层）、第 3,000 版起低于 18 层（猎人 23.7 层）
if v >= 1000 and ep and floor < 12: flags.append(f"ALERT:学得慢？第{v}版整局{floor:.2f}层<12")
if v >= 3000 and ep and floor < 18: flags.append(f"ALERT:学得慢？第{v}版整局{floor:.2f}层<18")
print("".join(" " + x for x in flags))
