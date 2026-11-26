# 储君监管：评测端检查（在 A 上跑）。输出一行读数；ALERT: 有问题，TRIGGER: 到了决策条件
import json, os, statistics as S, time
E = "/root/.local/share/sls2/runs/regent-eval"
rows = [json.loads(l) for l in open(f"{E}/eval.jsonl") if l.strip()]
rows = [r for r in rows if r.get("runs") == 384]
# 回退过权重时评测记录里有作废时间线上的轮次（储君决策记录第 27 条：Muon 段第 55,458–55,626 版 4 轮）：按文件顺序，后面一轮的版本号不大于前面的，就把前面版本号不小于它的去掉
_out = []
for r in rows:
    while _out and _out[-1]["version"] >= r["version"]: _out.pop()
    _out.append(r)
rows = _out
prog = json.load(open(f"{E}/eval_progress.json"))
running = time.time() - prog.get("start", time.time())
stuck_n = sum(1 for _ in open(f"{E}/stuck.jsonl")) if os.path.exists(f"{E}/stuck.jsonl") else 0
last5 = rows[-5:]
f5 = S.mean(r["floor_mean"] for r in last5); a3 = S.mean(r["reach_act3"] for r in last5); a2 = S.mean(r["reach_act2"] for r in last5)
flags = []
v = rows[-1]["version"]
# 750 版的两段各要 6 轮以上（2026-11-08：评测每轮约 95–100 版，750 版常只有 7 轮，原来要 8 轮时这两项一直是空的，走平判断失效）
cur = [r for r in rows if r["version"] > v - 750]; prev = [r for r in rows if v - 1500 < r["version"] <= v - 750]
d = S.mean(r["floor_mean"] for r in cur) - S.mean(r["floor_mean"] for r in prev) if len(cur) >= 6 and len(prev) >= 6 else None
dw = S.mean(r["win_rate"] for r in cur) - S.mean(r["win_rate"] for r in prev) if len(cur) >= 6 and len(prev) >= 6 else None
wins = sum(round(r.get("win_rate", 0) * r["runs"]) for r in rows[-20:])
# 第一幕非 Boss 战的死亡：2026-11-07 起多是对小叶史莱姆拖着不打、被“连续 20 回合不造成伤害”判输（训练里吃不到这个惩罚）
d1 = [sum(v for k, v in (r.get("deaths") or {}).items() if k.startswith("0:") and ":Boss:" not in k) for r in rows[-5:]]
print(f"A 评测v{v} 最近5轮 {f5:.2f}层 进二幕{a2:.3f} 到三幕{a3:.3f} 近20轮通关{wins} 750版涨幅{'' if d is None else f'{d:+.2f}'} 750版通关涨幅{'' if dw is None else f'{dw*100:+.1f}'} "
      f"本轮{prog.get('done')}/{prog.get('total')} 已{running:.0f}s 卡住记录{stuck_n} 一幕非Boss死亡(近5轮){d1}", end="")
if running > 900 and prog.get("done", 0) < prog.get("total", 0): flags.append(f"ALERT:评测这一轮已 {running:.0f} 秒（第 {prog.get('version')} 版，{prog.get('done')}/{prog.get('total')}）")
if f5 < 29.5: flags.append(f"ALERT:评测最近 5 轮 {f5:.2f} 层")
if d1 and d1[-1] > 100: flags.append(f"ALERT:评测第一幕非 Boss 死亡 {d1[-1]} 局（拖延判输在恶化？）")
if a2 < 0.74: flags.append(f"ALERT:评测进第二幕 {a2:.3f}")
# 第 22,075 版前后：评测第一幕选精英从每局约 1.0–1.4 次涨到 1.6–1.7 次，第一幕死亡每轮约 20 局 → 27 / 27 / 33 / 42，进第二幕 0.96 → 0.89；
# 同期第二、三幕 Boss 变强、通关率升。平常最近 5 轮进第二幕 0.94–0.96，跌破 0.90 叫醒
# 2026-11-08（储君决策记录第 19 条补记）：第一幕多打精英，进第二幕平常已在 0.90–0.92，5 轮的线 0.90 → 0.88，免得常误报；12 轮的常设线见下
if a2 < 0.88: flags.append(f"ALERT:评测最近 5 轮进第二幕 {a2:.3f} < 0.88（第一幕多打精英、死得多？）")
# 常设护栏（2026-11-08，第 19 条的护栏退役后换上，储君决策记录第 19 条补记）：最近 12 轮平均层数低于 41.2、进第二幕低于 90%、
# 到第三幕低于 68.3%、通关低于 22% 任一条就叫醒。名字里不能有空格（supervise.sh 按空格截取，ALERT 开头的都会叫醒）
last12 = rows[-12:]
pf = S.mean(r["floor_mean"] for r in last12); p2 = S.mean(r["reach_act2"] for r in last12); p3 = S.mean(r["reach_act3"] for r in last12)
pw = S.mean(r["win_rate"] for r in last12)
print(f" 近12轮 {pf:.2f}层/{p2:.3f}/{p3:.3f}/通关{pw:.3f}", end="")
if pf < 41.2 or p2 < 0.90 or p3 < 0.683 or pw < 0.22:
    flags.append(f"ALERT:常设护栏（近12轮{pf:.2f}层、进二幕{p2:.3f}、到三幕{p3:.3f}、通关{pw:.3f}）")
# 第 25 条护栏（原为 Muon 而设；第 27 条退回第 54,603 版、继续 Adam 后照用：从第 54,604 版起的新时间线对改前 12 轮，作废的 Muon 段已在上面去掉）：改后满 8 轮评测起，改后最近 12 轮与改前 12 轮
# （第 53,610–54,254 版：46.06 层、进第二幕 0.954、到第三幕 0.890、通关 0.668；每局死于第一幕非 Boss 2.0%、第三幕非 Boss 2.1%）比，
# 层数掉 1 层、进第二幕掉 3、到第三幕掉 4.5、通关掉 3 个百分点，或每局死于第一幕非 Boss、第三幕非 Boss 超过 3.5%，任一条叫醒（考虑退回 Adam）
post = [r for r in rows if r["version"] >= 54604][-12:]
if len(post) >= 8:
    qf = S.mean(r["floor_mean"] for r in post); q2 = S.mean(r["reach_act2"] for r in post); q3 = S.mean(r["reach_act3"] for r in post)
    qw = S.mean(r["win_rate"] for r in post)
    def dn(a):
        return sum(v for r in post for k, v in (r.get("deaths") or {}).items() if k.startswith(f"{a}:Elite:") or k.startswith(f"{a}:Monster:")) / (384 * len(post))
    d1, d3n = dn(0), dn(2)
    print(f" 第25条改后{len(post)}轮 {qf:.2f}层/{q2:.3f}/{q3:.3f}/通关{qw:.3f}/一幕非Boss死{d1:.3f}/三幕非Boss死{d3n:.3f}", end="")
    if qf < 45.06 or q2 < 0.924 or q3 < 0.845 or qw < 0.638 or d1 > 0.035 or d3n > 0.035:
        flags.append(f"ALERT:第25条护栏（改后{len(post)}轮{qf:.2f}层、进二幕{q2:.3f}、到三幕{q3:.3f}、通关{qw:.3f}、一幕非Boss死{d1:.3f}、三幕非Boss死{d3n:.3f}）")
if a3 >= 0.25: flags.append(f"TRIGGER:评测到第三幕 {a3:.3f} ≥ 25%")
# 走平判断改看通关率（第 25,865 版起）：评测层数停在 40–42 层、通关率还在涨，层数已说明不了进步；原来这条的名字有空格，
# supervise.sh 按第一个空格截取触发名，截出来是“TRIGGER:评测”，与练习走平同时成立的判断从没认出过它。名字里不能有空格
if dw is not None and dw < 0.01: flags.append(f"TRIGGER:评测通关750版涨幅{dw*100:+.1f}个百分点<1（看是否走平）")
if os.path.exists(f"{E}/PAUSE"): print(" 常规评测暂停中（PAUSE）", end="")
print(" " + " ".join(flags) if flags else "")
print(f"STUCK_N={stuck_n}")
