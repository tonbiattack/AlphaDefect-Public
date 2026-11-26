# 猎人监管：评测端检查（在 B 上跑）。输出一行读数；ALERT: 有问题
import json, os, statistics as S, time
E = "/root/.local/share/sls2/runs/silent-eval"
rows = [json.loads(l) for l in open(f"{E}/eval.jsonl") if l.strip()] if os.path.exists(f"{E}/eval.jsonl") else []
rows = [r for r in rows if r.get("runs") == 384]
out = []
for r in rows:
    while out and out[-1]["version"] >= r["version"]:
        out.pop()
    out.append(r)
rows = out
prog = json.load(open(f"{E}/eval_progress.json")) if os.path.exists(f"{E}/eval_progress.json") else {}
running = time.time() - prog.get("start", time.time())
flags = []
if rows:
    last5 = rows[-5:]
    f5 = S.mean(r["floor_mean"] for r in last5); w5 = S.mean(r["win_rate"] for r in last5); a2 = S.mean(r["reach_act2"] for r in last5)
    print(f"E 评测v{rows[-1]['version']} 已评{len(rows)}轮 最近5轮 {f5:.2f}层 进二幕{a2:.3f} 通关{w5:.3f} 本轮{prog.get('done')}/{prog.get('total')} 已{running:.0f}s", end="")
else:
    print(f"E 还没有评测轮次 本轮{prog.get('done')}/{prog.get('total')} 已{running:.0f}s", end="")
# 第 4 条起评测用显卡推理，一轮约 220 秒；超过 30 分钟没打完叫醒（原为 CPU 推理时的 3 小时）
if prog and running > 1800 and prog.get("done", 0) < prog.get("total", 0):
    flags.append(f"ALERT:评测这一轮已{running:.0f}秒")
# 第 7 条护栏（第 17,853 版起第二幕真实局面 0.37 → 0.25，腾出的 0.12 给第三幕 Boss）：改后满 8 轮起，最近 12 轮与改前 10 轮
# （第 17,103–17,747 版：40.84 层、进第二幕 0.888、到第三幕 0.713、通关 28.4%）比，层数掉 1 层、到第三幕掉 4.5 个百分点、通关掉 3 个百分点，
# 任一条就叫醒。进第二幕的线按第 15,000–17,747 版的平均（约 0.866）减 3 个百分点取 0.835：改前 10 轮的 0.888 是这段的高点，
# 第 14,000 段 1,000 版平均只有 0.848；第 7 条不动第一幕份额，储君第 19 条就在这条线上误报过一次（第 6 条的护栏一条没碰：改前 10 轮 37.37 层，第 7 条前最近 12 轮 40.48 层）
post = [r for r in rows if r["version"] >= 17853]
if len(post) >= 8:
    w = post[-12:]
    pf, pa2, pa3 = S.mean(r["floor_mean"] for r in w), S.mean(r["reach_act2"] for r in w), S.mean(r.get("reach_act3", 0) for r in w)
    pw = S.mean(r["win_rate"] for r in w)
    print(f" 第7条后{len(post)}轮 最近{len(w)}轮 {pf:.2f}层 进二幕{pa2:.3f} 到三幕{pa3:.3f} 通关{pw:.3f}", end="")
    if pf < 39.84 or pa2 < 0.835 or pa3 < 0.668 or pw < 0.254:
        flags.append(f"ALERT:第7条护栏（{pf:.2f}层/{pa2:.3f}/{pa3:.3f}/{pw:.3f}）")
# 第 8 条（第 18,638 版起学习率减半、每批加倍）的护栏：改后满 8 轮起，最近 12 轮通关低于 27.2%（改前 12 轮约 29.2% 减 2 个百分点）叫醒，
# 与学习端的训练整局通关护栏一起看（两条都成立就退回，同储君第 21 条）
post8 = [r for r in rows if r["version"] >= 18638]
if len(post8) >= 8:
    w8 = post8[-12:]
    pw8 = S.mean(r["win_rate"] for r in w8)
    print(f" 第8条后{len(post8)}轮 最近{len(w8)}轮通关{pw8:.3f}", end="")
    if pw8 < 0.272: flags.append(f"ALERT:第8条后评测12轮通关{pw8:.3f}<27.2%")
# 第 9 条（第 19,901 版起观测格式 21 → 23）的护栏：改后满 8 轮起，最近 12 轮通关低于 41.9%（改前 12 轮第 19,412–19,856 版 2,160 / 4,608 = 46.88% 减 5 个百分点）叫醒，
# 与学习端的训练整局通关护栏（30.4%）一起看，两条都成立就退回格式 21
post9 = [r for r in rows if r["version"] >= 19901]
if len(post9) >= 8:
    w9 = post9[-12:]
    pw9, pf9, pa39 = S.mean(r["win_rate"] for r in w9), S.mean(r["floor_mean"] for r in w9), S.mean(r.get("reach_act3", 0) for r in w9)
    print(f" 第9条后{len(post9)}轮 最近{len(w9)}轮 {pf9:.2f}层 到三幕{pa39:.3f} 通关{pw9:.3f}", end="")
    if pw9 < 0.419: flags.append(f"ALERT:第9条后评测12轮通关{pw9:.3f}<41.9%")
elif post9:
    print(f" 第9条后{len(post9)}轮 通关{S.mean(r['win_rate'] for r in post9):.3f}", end="")
# 第 10 条（第 21,286 版起第二幕真实局面 0.25 → 0.15，腾出的 0.10 给第三幕 Boss）的护栏（同储君第 22 条）：改后满 8 轮起，最近 12 轮与改前 12 轮
# （第 20,821–21,236 版 4,608 局：43.66 层、进第二幕 0.906、到第三幕 0.814、通关 2,334 / 4,608 = 50.65%、每局死于第二幕非 Boss 4.90%）比，
# 层数掉 1 层、进第二幕掉 3、到第三幕掉 4.5、通关掉 3 个百分点，或每局死于第二幕非 Boss 超过 6.9%（改前加 2 个百分点），任一条叫醒
post10 = [r for r in rows if r["version"] >= 21286]
if len(post10) >= 8:
    w10 = post10[-12:]
    n10 = sum(r["runs"] for r in w10)
    pf, pa2, pa3, pw = (S.mean(r["floor_mean"] for r in w10), S.mean(r["reach_act2"] for r in w10),
                        S.mean(r.get("reach_act3", 0) for r in w10), S.mean(r["win_rate"] for r in w10))
    d2 = sum(v for r in w10 for k, v in (r.get("deaths") or {}).items() if k.startswith(("1:Elite:", "1:Monster:"))) / max(1, n10)
    print(f" 第10条后{len(post10)}轮 最近{len(w10)}轮 {pf:.2f}层 进二幕{pa2:.3f} 到三幕{pa3:.3f} 通关{pw:.3f} 死于二幕非Boss{d2:.3f}", end="")
    if pf < 42.66 or pa2 < 0.876 or pa3 < 0.769 or pw < 0.4765 or d2 > 0.069:
        flags.append(f"ALERT:第10条护栏（{pf:.2f}层/{pa2:.3f}/{pa3:.3f}/{pw:.3f}/二幕非Boss死{d2:.3f}）")
elif post10:
    print(f" 第10条后{len(post10)}轮 通关{S.mean(r['win_rate'] for r in post10):.3f}", end="")
# 第 11 条（第 23,321 版起学习率再减半、每批再加倍）的护栏（同第 8 条）：改后满 8 轮起，最近 12 轮通关低于 51.4%（改前 12 轮第 22,823–23,240 版
# 2,462 / 4,608 = 53.43% 减 2 个百分点）叫醒，与学习端的训练整局通关护栏（38.9%）一起看，两条都成立就退回 lr 2e-4、muon_lr 3e-4、每批 65,536
post11 = [r for r in rows if r["version"] >= 23321]
if len(post11) >= 8:
    w11 = post11[-12:]
    pw11, pf11, pa311 = S.mean(r["win_rate"] for r in w11), S.mean(r["floor_mean"] for r in w11), S.mean(r.get("reach_act3", 0) for r in w11)
    print(f" 第11条后{len(post11)}轮 最近{len(w11)}轮 {pf11:.2f}层 到三幕{pa311:.3f} 通关{pw11:.3f}", end="")
    if pw11 < 0.514: flags.append(f"ALERT:第11条后评测12轮通关{pw11:.3f}<51.4%")
elif post11:
    print(f" 第11条后{len(post11)}轮 通关{S.mean(r['win_rate'] for r in post11):.3f}", end="")
# 第 12 条（第 23,828 版起整局分段 run_segment_steps 64 → 256）的护栏：改后满 8 轮起，最近 12 轮通关低于 60.5%（改前 12 轮第 23,571–23,798 版
# 2,880 / 4,608 = 62.50% 减 2 个百分点）叫醒，与学习端的训练整局通关护栏（45.2%）一起看，两条都成立就退回 64
# 第 12 条满 500 版没有正收益（改后最近 10 轮 61.28%，平均层数 44.91 → 44.07，每局死于第一幕 7.3% → 9.8%）。第 13 条把权重还原到第 23,827 版、分段退回 64，
# 第 23,828 版起是新的时间线；第 12 条那段的评测轮次由上面的去重丢掉，上面这条护栏停用。
# 第 13 条的护栏：满 8 轮起，最近 12 轮通关低于 60.0% 或平均层数低于 43.8 叫醒（第 12 条前 12 轮第 23,613–23,827 版 62.00%、44.91 层，
# 第 12 条最后 10 轮 61.28%、44.07 层）；满 500 版（约 24 轮）时与学习端一起读：同样的权重与配置，应回到 62% 上下、44.9 层上下
# 2026-11-11 09:10 补（第 24,017 版）：这段起初写成 >= 24335（只退回分段时的起点），权重回退后漏改，第 13 条的评测护栏一直没生效；改成 23828 后
# 头 9 轮（第 23,828–23,982 版）59.5%、44.35 层就碰了 60.0% 的线。回退的起点第 23,827 版自己那轮就只有 57.3%、44.22 层（前后几轮 61.7–63.0%），
# 同一起点走第 12 条时头 10 轮是 62.6%：贪心评测在两条独立的时间线之间差 3 个百分点，不全是配置的差别。线改为 58.0% / 43.6 层
# （第 11 条以来每 250 版的评测块最低 59.2%、44.33 层；12 轮平均低于 58% 就比第 11 条以来任何一段都低），只管“明显变差”，效果满 500 版再读
# 第 14 条（第 24,208 版）上线后第 13 条这段收口：只算第 23,828–24,207 版的轮次
post13 = [r for r in rows if 23828 <= r["version"] < 24208]
if len(post13) >= 8:
    w13 = post13[-12:]
    pw13, pf13, pa313 = S.mean(r["win_rate"] for r in w13), S.mean(r["floor_mean"] for r in w13), S.mean(r.get("reach_act3", 0) for r in w13)
    print(f" 第13条后{len(post13)}轮 最近{len(w13)}轮 {pf13:.2f}层 到三幕{pa313:.3f} 通关{pw13:.3f}", end="")
    if pw13 < 0.580 or pf13 < 43.6: flags.append(f"ALERT:第13条后评测12轮（{pf13:.2f}层/通关{pw13:.3f}）")
elif post13:
    print(f" 第13条后{len(post13)}轮 通关{S.mean(r['win_rate'] for r in post13):.3f}", end="")
# 第 14 条（第 24,208 版起跳过稳赢战斗，猎人决策记录第 14 条）的护栏：满 8 轮起，最近 12 轮通关低于 59.5%（上线前 12 轮 61.5% 减 2 个百分点）叫醒，
# 成立就关掉跳过（配置 skip_easy 改 false 重启）。评测本身不跳过战斗，照常整局真打
# 第 17 条（第 24,722 版起梯度累积 ×2）之后，跳过照开，第 14 条这条护栏改用第 17 条之后的评测（同一条线 59.5%）；
# 第 17 条自己的线：满 8 轮起最近 12 轮低于 60.4%（第 14 条最后 12 轮 62.43% 减 2 个百分点）叫醒，与训练那侧（sup_train.py）都成立就退回梯度累积 1
# 第 20 条（第 25,228 版起 skip_prob 0.8 → 1.0）：第 17 条满 500 版已读（最后 12 轮 64.34%），第 17 条那两条线收口到第 24,722–25,227 版的轮次；
# 第 20 条自己的线：满 8 轮起最近 12 轮低于 62.3%（第 17 条最后 12 轮 64.34% 减 2 个百分点）叫醒，与训练那侧（sup_train.py 51.1%）都成立就退回 skip_prob 0.8；
# 第 14 条那条 59.5% 的线照旧（成立时先退回 0.8）。满 8 轮之前第 20 条的轮次与第 17 条的连起来按最近 12 轮看这两条。
post14 = [r for r in rows if 24208 <= r["version"] < 24722]
post17 = [r for r in rows if 24722 <= r["version"] < 25228]
post20 = [r for r in rows if 25228 <= r["version"] < 25485]
# 第 21 条（第 25,485 版起退回 skip_prob 0.8）：第 20 条的轮次收口；退回后满 8 轮起最近 12 轮低于 62.3% 叫醒（同第 20 条的线），第 14 条 59.5% 照旧
post21 = [r for r in rows if r["version"] >= 25485]
if len(post21) >= 8:
    w21 = post21[-12:]
    pw21, pf21 = S.mean(r["win_rate"] for r in w21), S.mean(r["floor_mean"] for r in w21)
    print(f" 第21条后{len(post21)}轮 最近{len(w21)}轮 {pf21:.2f}层 通关{pw21:.3f}", end="")
    if pw21 < 0.623: flags.append(f"ALERT:第21条后评测12轮通关{pw21:.3f}<62.3%")
    if pw21 < 0.595: flags.append(f"ALERT:第14条护栏 评测12轮通关{pw21:.3f}<59.5%")
    post20 = []
elif post21:
    print(f" 第21条后{len(post21)}轮 通关{S.mean(r['win_rate'] for r in post21):.3f}", end="")
if len(post20) >= 8:
    w20 = post20[-12:]
    pw20, pf20 = S.mean(r["win_rate"] for r in w20), S.mean(r["floor_mean"] for r in w20)
    print(f" 第20条后{len(post20)}轮 最近{len(w20)}轮 {pf20:.2f}层 通关{pw20:.3f}", end="")
    if pw20 < 0.623: flags.append(f"ALERT:第20条后评测12轮通关{pw20:.3f}<62.3%")
    if pw20 < 0.595: flags.append(f"ALERT:第14条护栏 评测12轮通关{pw20:.3f}<59.5%（先退回skip_prob0.8）")
elif post20:
    wm = (post17 + post20)[-12:]; pwm = S.mean(r["win_rate"] for r in wm)
    print(f" 第20条后{len(post20)}轮 通关{S.mean(r['win_rate'] for r in post20):.3f}（连第17条最近12轮{pwm:.3f}）", end="")
    if pwm < 0.595: flags.append(f"ALERT:第14条护栏 评测12轮通关{pwm:.3f}<59.5%（先退回skip_prob0.8）")
elif len(post17) >= 8:
    w17 = post17[-12:]
    pw17, pf17 = S.mean(r["win_rate"] for r in w17), S.mean(r["floor_mean"] for r in w17)
    print(f" 第17条后{len(post17)}轮 最近{len(w17)}轮 {pf17:.2f}层 通关{pw17:.3f}", end="")
    if pw17 < 0.604: flags.append(f"ALERT:第17条后评测12轮通关{pw17:.3f}<60.4%")
    if pw17 < 0.595: flags.append(f"ALERT:第14条护栏 评测12轮通关{pw17:.3f}<59.5%（关跳过）")
elif post17:
    print(f" 第17条后{len(post17)}轮 通关{S.mean(r['win_rate'] for r in post17):.3f}", end="")
elif len(post14) >= 8:
    w14 = post14[-12:]
    print(f" 第14条{len(post14)}轮 最近{len(w14)}轮 通关{S.mean(r['win_rate'] for r in w14):.3f}", end="")
print(" " + " ".join(flags) if flags else "")
