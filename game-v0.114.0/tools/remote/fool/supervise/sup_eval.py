# Fool 监管：评测端检查（在 B 上跑；2026-11-13 Fool 从零开训）。输出一行读数；ALERT: 有问题
import json, os, statistics as S, subprocess, time
E = "/root/.local/share/sls2/runs/fool-eval"
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
alive = subprocess.run(["pgrep", "-f", "[r]emote_eval.py --config /root/fool_train_config.json"], capture_output=True).returncode == 0
flags = []
if rows:
    last5 = rows[-5:]
    f5 = S.mean(r["floor_mean"] for r in last5); w5 = S.mean(r["win_rate"] for r in last5); a2 = S.mean(r["reach_act2"] for r in last5)
    a3 = S.mean(r.get("reach_act3", 0) for r in last5)
    print(f"E 评测v{rows[-1]['version']} 已评{len(rows)}轮 最近5轮 {f5:.2f}层 进二幕{a2:.3f} 到三幕{a3:.3f} 通关{w5:.3f} 本轮{prog.get('done')}/{prog.get('total')} 已{running:.0f}s", end="")
else:
    print(f"E 还没有评测轮次 本轮{prog.get('done')}/{prog.get('total')} 已{running:.0f}s", end="")
# 愚者决策记录第 4 条（第 3,765 版起练习份额往 Boss 挪、第一幕 Boss 奖励 0.5 → 0.1）的护栏（幅度同猎人第 5 条）：改后满 8 轮起，最近 12 轮与
# 改前 10 轮（第 3,551–3,749 版，1,920 局：34.86 层、进第二幕 0.885、到第三幕 0.341）比，层数低于 33.86、进第二幕低于 0.855、到第三幕低于 0.296，任一条成立就叫醒
post = [r for r in rows if r["version"] >= 3765]
if len(post) >= 8:
    p12 = post[-12:]; nn = sum(r["runs"] for r in p12)
    pf = sum(r["floor_mean"] * r["runs"] for r in p12) / nn
    pa2 = sum(r["reach_act2"] * r["runs"] for r in p12) / nn
    pa3 = sum(r.get("reach_act3", 0) * r["runs"] for r in p12) / nn
    if pf < 33.86 or pa2 < 0.855 or pa3 < 0.296:
        flags.append(f"ALERT:第4条护栏（{pf:.2f}层/{pa2:.3f}/{pa3:.3f}）")
if not alive: flags.append("ALERT:评测进程不在")
# 显卡推理一轮约 220 秒（猎人）；超过 30 分钟没打完叫醒
if prog and running > 1800 and prog.get("done", 0) < prog.get("total", 0):
    flags.append(f"ALERT:评测这一轮已{running:.0f}秒")
if rows and time.time() - rows[-1].get("time", time.time()) > 3600:
    flags.append(f"ALERT:评测{(time.time() - rows[-1]['time']) / 60:.0f}分钟没有新轮次")
print("".join(" " + x for x in flags))
