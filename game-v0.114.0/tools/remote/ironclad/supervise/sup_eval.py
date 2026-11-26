# 铁甲战士从零线监管：评测端检查（在训练机上跑，2026-11-20 起为 B；2026-11-17，铁甲战士决策记录第 1 条，照恐惧模式故障机器人从零线 63fdcdd 与 tools/remote/fool/supervise/sup_eval.py）。输出一行读数；ALERT: 有问题
import json, os, statistics as S, subprocess, time
E = "/root/.local/share/sls2/runs/ironclad-eval"
rows = [json.loads(l) for l in open(f"{E}/eval.jsonl") if l.strip()] if os.path.exists(f"{E}/eval.jsonl") else []
rows = [r for r in rows if r.get("runs") == 384]      # 评测 384 局（种子池 P1 评测部分全部，铁甲战士决策记录第 1 条）
out = []
for r in rows:
    while out and out[-1]["version"] >= r["version"]:
        out.pop()
    out.append(r)
rows = out
prog = json.load(open(f"{E}/eval_progress.json")) if os.path.exists(f"{E}/eval_progress.json") else {}
running = time.time() - prog.get("start", time.time())
alive = subprocess.run(["pgrep", "-f", "[r]emote_eval.py --config /root/ironclad_train_config.json"], capture_output=True).returncode == 0
flags = []
if rows:
    last5 = rows[-5:]
    f5 = S.mean(r["floor_mean"] for r in last5); w5 = S.mean(r["win_rate"] for r in last5); a2 = S.mean(r["reach_act2"] for r in last5)
    a3 = S.mean(r.get("reach_act3", 0) for r in last5)
    print(f"E 评测v{rows[-1]['version']} 已评{len(rows)}轮 最近5轮 {f5:.2f}层 进二幕{a2:.3f} 到三幕{a3:.3f} 通关{w5:.3f} 本轮{prog.get('done')}/{prog.get('total')} 已{running:.0f}s", end="")
else:
    print(f"E 还没有评测轮次 本轮{prog.get('done')}/{prog.get('total')} 已{running:.0f}s", end="")
# 护栏：每次改训练设置时按那一条的改前读数加（照愚者第 4、9 条的做法），开训时没有。
# 铁甲战士决策记录第 24 条（B 上续训：训练加血 20、通关余血奖励，第 24,280 版起；评测照旧不加血）：改后满 12 轮、窗口全落在改后再判，
# 最近 12 轮任一条成立就叫醒：层数低于 45.44、进第二幕低于 0.923、到第三幕低于 0.858、通关低于 63.61%
# （改前 48 轮第 23,898–24,258 版 46.44 层 / 0.953 / 0.903 / 66.61% 减 1 层、3、4.5、3 个百分点）。
# 这是用户要试的有偏信号，碰线先报告用户、由用户定是否撤回，不像第 21 条那样碰线就自动关。第 22 条的护栏由这条接替。
# 第 24 条已作废（第 25 条：退回第 24,279 版），同一组线现在给第 26 条（回灌，新时间线第 24,280 版起）用：碰线先报告用户；
# 掉得很厉害（如十几个百分点）时先关回灌、退回第 24,279 版再报告。作废时间线的评测轮次由上面按版本去重去掉
# 第 27 条（观测格式 29，第 25,463 版起）沿用同一组线，窗口从第 25,463 版起满 12 轮再判；碰线先报告用户
post = [r for r in rows if r["version"] > 25462]
if post:
    print(f" 改后{len(post)}轮", end="")
if len(post) >= 12:
    p12 = post[-12:]; nn = sum(r["runs"] for r in p12)
    pf = sum(r["floor_mean"] * r["runs"] for r in p12) / nn
    pa2 = sum(r["reach_act2"] * r["runs"] for r in p12) / nn
    pa3 = sum(r.get("reach_act3", 0) * r["runs"] for r in p12) / nn
    pw = sum(r["win_rate"] * r["runs"] for r in p12) / nn
    print(f" 改后12轮{pf:.2f}层/{pa2:.3f}/{pa3:.3f}/通关{pw:.3f}", end="")
    if pf < 45.44 or pa2 < 0.923 or pa3 < 0.858 or pw < 0.6361:
        flags.append(f"ALERT:第27条护栏（{pf:.2f}层/{pa2:.3f}/{pa3:.3f}/通关{pw:.3f}）")
if not alive: flags.append("ALERT:评测进程不在")
# 384 局一轮约 5 分钟（4 × 6 显卡推理，整局短时更快）；超过 30 分钟没打完叫醒
if prog and running > 1800 and prog.get("done", 0) < prog.get("total", 0):
    flags.append(f"ALERT:评测这一轮已{running:.0f}秒")
if rows and time.time() - rows[-1].get("time", time.time()) > 3600:
    flags.append(f"ALERT:评测{(time.time() - rows[-1]['time']) / 60:.0f}分钟没有新轮次")
print("".join(" " + x for x in flags))
