# 亡灵契约师监管：评测端检查（在 B 上跑；2026-11-17 亡灵契约师从零开训，照愚者从零开训时那一版 sup_eval.py（4237f40））。
# 输出一行读数；ALERT: 有问题，TRIGGER: 到了决策条件。
import json, os, statistics as S, subprocess, time
E = "/root/.local/share/sls2/runs/necro-eval"
rows = [json.loads(l) for l in open(f"{E}/eval.jsonl") if l.strip()] if os.path.exists(f"{E}/eval.jsonl") else []
rows = [r for r in rows if r.get("runs") == 192]
out = []
for r in rows:
    while out and out[-1]["version"] >= r["version"]:     # 回退过权重时去掉作废的时间线
        out.pop()
    out.append(r)
rows = out
prog = json.load(open(f"{E}/eval_progress.json")) if os.path.exists(f"{E}/eval_progress.json") else {}
running = time.time() - prog.get("start", time.time())
alive = subprocess.run(["pgrep", "-f", "[r]emote_eval.py --config /root/necro_train_config.json"], capture_output=True).returncode == 0
flags = []
if rows:
    last5 = rows[-5:]
    f5 = S.mean(r["floor_mean"] for r in last5); w5 = S.mean(r["win_rate"] for r in last5); a2 = S.mean(r["reach_act2"] for r in last5)
    a3 = S.mean(r.get("reach_act3", 0) for r in last5)
    print(f"E 评测v{rows[-1]['version']} 已评{len(rows)}轮 最近5轮 {f5:.2f}层 进二幕{a2:.3f} 到三幕{a3:.3f} 通关{w5:.3f} 本轮{prog.get('done')}/{prog.get('total')} 已{running:.0f}s", end="")
    # 最近 10 轮三幕 Boss 每场胜率（决策记录第 3 条：卡点在一、二幕 Boss，叫醒时直接看）
    bw = [[0, 0], [0, 0], [0, 0]]
    for r in rows[-10:]:
        for k, x in (r.get("bosses") or {}).items():
            a = k.split(":")[0]
            if a.isdigit() and int(a) <= 2:
                bw[int(a)][0] += x[0]; bw[int(a)][1] += x[1]
    print(" Boss每场(10轮) " + "/".join(f"{w / n:.3f}" if n else "nan" for w, n in bw), end="")
    # 评测走平（照愚者第 4 条的起因：第 2,750 版起每 250 版的评测层数基本不涨），读死因与练习后按决策记录的顺序定下一项（挪份额、第一幕 Boss 奖励、降学习率）。
    # 判法（决策记录第 3 条补记三）：最近 1,000 版的评测平均层数对版本号做直线回归，折成每 500 版的涨幅，不到 0.3 层就报。
    # 原来是最近 500 版对之前 500 版的均值差，只看一条分界、一两轮高低就能让它跳：第 5,137 版 +0.83、5 分钟后第 5,274 版 +0.09（同期回归 +0.66、+0.51）。
    # 回归在历史上照样认得出真平台：本线第 2,000 / 2,250 版 −0.01 / −0.15，愚者第 4 条的第 3,765 版 −0.22。
    # 原定第 2,000 版起判；第 3 条改为第 2,750 版起，补记二改为第 4,600 版起
    v = rows[-1]["version"]
    if v >= 4600:
        w = [r for r in rows if r["version"] > v - 1000]
        if len(w) >= 16:
            xs = [r["version"] for r in w]; ys = [r["floor_mean"] for r in w]
            mx, my = S.mean(xs), S.mean(ys)
            sxx = sum((x - mx) ** 2 for x in xs)
            d = 500 * sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else float("nan")
            print(f" 1000版回归每500版{d:+.2f}层", end="")
            # 第 3 条补记四：只看评测不再叫醒，叫醒改由 sup_train.py 的“评测与 Boss 练习都走平”判（两项同时成立）
else:
    print(f"E 还没有评测轮次 本轮{prog.get('done')}/{prog.get('total')} 已{running:.0f}s", end="")
# 护栏（第 14 条，第 24,044 版起：第二幕真实局面份额 0.25 → 0.15、第三幕 Boss 0.235 → 0.335）：满 8 轮起，最近 12 轮与改前第 23,559–24,031 版 52 轮
# （43.07 层、进第二幕 0.886、到第三幕 0.797、通关 5,497 / 9,984 = 55.06%）比：通关低于 51.1%（−4 个百分点）、到第三幕低于 0.752（−0.045）、
# 进第二幕低于 0.826（−0.06），任一条成立就叫醒
post = [r for r in rows if r["version"] >= 24044]
if len(post) >= 8:
    p12 = post[-12:]
    pf = S.mean(r["floor_mean"] for r in p12); pa2 = S.mean(r["reach_act2"] for r in p12); pa3 = S.mean(r.get("reach_act3", 0) for r in p12)
    pw = S.mean(r["win_rate"] for r in p12)
    print(f" 第14条后{len(post)}轮 近12轮{pf:.2f}层/{pa2:.3f}/{pa3:.3f}/通关{pw:.3f}", end="")
    if pw < 0.511: flags.append(f"ALERT:第14条护栏通关{pw:.3f}<0.511")
    if pa3 < 0.752: flags.append(f"ALERT:第14条护栏到三幕{pa3:.3f}<0.752")
    if pa2 < 0.826: flags.append(f"ALERT:第14条护栏进二幕{pa2:.3f}<0.826")
if not alive: flags.append("ALERT:评测进程不在")
# 显卡推理一轮约 220 秒（猎人）；超过 30 分钟没打完叫醒
if prog and running > 1800 and prog.get("done", 0) < prog.get("total", 0):
    flags.append(f"ALERT:评测这一轮已{running:.0f}秒")
if rows and time.time() - rows[-1].get("time", time.time()) > 3600:
    flags.append(f"ALERT:评测{(time.time() - rows[-1]['time']) / 60:.0f}分钟没有新轮次")
print("".join(" " + x for x in flags))
