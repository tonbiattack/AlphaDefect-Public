# 第 14 条（跳过稳赢战斗，第 24,208 版起）满 500 版读效果：对上线前同长度（第 23,708–24,207 版）
import json, math, statistics as S, sys
V14 = 24208
END = int(sys.argv[1]) if len(sys.argv) > 1 else 24707
rows = []
with open("/root/.local/share/sls2/runs/silent/log.jsonl", "rb") as f:
    f.seek(-70_000_000, 2)
    for l in f.read().split(b"\n")[1:]:
        try: r = json.loads(l)
        except ValueError: continue
        if r.get("version") is None: continue
        while rows and rows[-1]["version"] >= r["version"]: rows.pop()
        rows.append(r)
print(f"日志版本 {rows[0]['version']}–{rows[-1]['version']}")
def ch(w, pre):
    c = {}
    for r in w:
        for k, x in (r.get("run_choices") or {}).items():
            if k.startswith(pre): c[k] = c.get(k, 0) + x
    return c
def hours(w):
    t = sum(b["time"] - a["time"] for a, b in zip(w, w[1:]) if 0 < b["time"] - a["time"] <= 120)
    return t / 3600, len(w) - 1
def blk(w, tag):
    med = lambda k: S.median([r[k] for r in w if isinstance(r.get(k), (int, float))])
    ep = sum(r.get("run_episodes") or 0 for r in w)
    win = sum((r.get("run_win_rate") or 0) * (r.get("run_episodes") or 0) for r in w) / max(1, ep)
    fl = sum((r.get("run_floor_mean") or 0) * (r.get("run_episodes") or 0) for r in w) / max(1, ep)
    h, nv = hours(w)
    hv = h * len(w) / max(1, nv)
    bn = [sum(r["boss_n"][i] for r in w if r.get("boss_n")) for i in range(3)]
    bw = [sum(r["boss_wins"][i] for r in w if r.get("boss_wins")) / max(1, bn[i]) for i in range(3)]
    act = lambda r: sum((r.get("lag_split") or {}).get("local", {}).get("actor", []))
    mac = sum((r.get("macro_frac") or 0) * act(r) for r in w)
    runst = sum(sum(a[0] for a in (r.get("run_fresh_act") or [])) for r in w)
    allst = sum(act(r) for r in w)
    evm = S.median([r["health"]["ev"]["macro"] for r in w if (r.get("health") or {}).get("ev", {}).get("macro") is not None])
    evc = S.median([r["health"]["ev"]["combat"] for r in w if (r.get("health") or {}).get("ev", {}).get("combat") is not None])
    ent_c = S.median([(r["entropy"] - r["macro_frac"] * r["entropy_macro"]) / (1 - r["macro_frac"]) for r in w if r.get("macro_frac") and r.get("entropy_macro") is not None])
    c = ch(w, ("SKIPRUN_",))
    nr, wr = c.get("SKIPRUN_N:real", 0), c.get("SKIPRUN_WIN:real", 0)
    ns, ws = c.get("SKIPRUN_N:skip", 0), c.get("SKIPRUN_WIN:skip", 0)
    print(f"\n[{tag}] v{w[0]['version']}–{w[-1]['version']} {len(w)}版，实际运行 {hv:.2f} 小时（每版 {hv*3600/len(w):.1f} 秒，入训中位 {med('sps'):,.0f} 步/秒）")
    print(f"  整局 {ep} 局（每小时 {ep/hv:,.0f} 局，每版 {ep/len(w):.1f} 局） 通关 {win:.4f}±{math.sqrt(win*(1-win)/max(1,ep)):.4f} 层 {fl:.2f}")
    if nr: print(f"  完全真打 {nr:.0f} 局（每小时 {nr/hv:,.0f}） 通关 {wr/nr:.4f}±{math.sqrt((wr/nr)*(1-wr/nr)/nr):.4f}；可跳的局 {ns:.0f} 局 通关 {ws/max(1,ns):.4f}")
    print(f"  每版：整局步 {runst/len(w):,.0f} / 全部入训步 {allst/len(w):,.0f}，局外步 {mac/len(w):,.0f}（每小时 {mac/hv:,.0f}），每局局外步 {mac/max(1,ep):.0f}")
    print(f"  Boss 练习 {bw[0]:.3f}/{bw[1]:.3f}/{bw[2]:.3f}（{bn[0]}/{bn[1]}/{bn[2]} 场）")
    print(f"  熵 战斗步 {ent_c:.3f} 局外 {med('entropy_macro'):.3f} 选路 {med('entropy_map'):.3f}；KL {med('approx_kl'):.4f} 裁剪 {med('clip_frac'):.3f}；"
          f"局外价值损失 {med('loss_v_macro'):.4f} 解释方差 局外 {evm:.3f} 战斗 {evc:.3f}")
    print(f"  耗时中位：采集 {med('t_collect'):.2f} 学习 {med('t_learn'):.2f}（PPO {med('t_ppo'):.2f} 准备 {med('t_prep'):.2f} 等小批 {med('t_mb_wait'):.2f}）秒；"
          f"结果头 血量损失 {med('co_hp_loss'):.3f}（每版 {med('co_n'):.0f} 场） 药水损失 {med('co_pot_loss'):.3f}" if w[-1].get("co_n") else
          f"  耗时中位：采集 {med('t_collect'):.2f} 学习 {med('t_learn'):.2f}（PPO {med('t_ppo'):.2f} 准备 {med('t_prep'):.2f} 等小批 {med('t_mb_wait'):.2f}）秒")
pre = [r for r in rows if V14 - 500 <= r["version"] < V14]
post = [r for r in rows if V14 <= r["version"] <= END]
blk(pre, "上线前 500 版"); blk(post, "第 14 条"); blk(post[:250], "第 14 条前一半"); blk(post[250:], "第 14 条后一半")
# 跳过明细
c = ch(post, ("SKIP_",))
eps = sum(r.get("run_episodes") or 0 for r in post)
print(f"\n跳过明细（第 14 条 {len(post)} 版，{eps} 局，完全真打 {c.get('SKIP_REAL_RUN', 0):.0f} 局）")
for k in sorted({k.split(":", 1)[1] for k in c if k.startswith("SKIP_ELIG:")}):
    el, dn = c.get(f"SKIP_ELIG:{k}", 0), c.get(f"SKIP_DONE:{k}", 0)
    rn, rs = c.get(f"SKIP_RES_N:{k}", 0), c.get(f"SKIP_RES_SUM:{k}", 0)
    print(f"  {k}: 够条件 {el:.0f}（每局 {el/max(1,eps):.2f}）跳过 {dn:.0f}（{dn/max(1,el):.2f}） 校准残差 {rs/max(1,rn):+.4f}（{rn:.0f} 场） 抽到血量变化 {c.get(f'SKIP_DHP_SUM:{k}', 0)/max(1,dn):+.4f}")
# 完全真打整局通关的 100 版分段
print("\n完全真打整局通关（每 100 版）：", end="")
for i in range(0, len(post), 100):
    cc = ch(post[i:i + 100], ("SKIPRUN_",)); n = cc.get("SKIPRUN_N:real", 0)
    print(f" v{post[i]['version']} {cc.get('SKIPRUN_WIN:real', 0)/max(1,n):.3f}({n:.0f})", end="")
print("\n上线前整局通关（每 100 版）：", end="")
for i in range(0, len(pre), 100):
    w = pre[i:i + 100]; ep = sum(r.get("run_episodes") or 0 for r in w)
    print(f" v{w[0]['version']} {sum((r.get('run_win_rate') or 0) * (r.get('run_episodes') or 0) for r in w)/max(1,ep):.3f}({ep})", end="")
print()
# 评测
ev = []
for l in open("/root/.local/share/sls2/runs/silent-eval/eval.jsonl"):
    if not l.strip(): continue
    e = json.loads(l)
    if e.get("runs") != 384: continue
    while ev and ev[-1]["version"] >= e["version"]: ev.pop()
    ev.append(e)
def evb(w, tag):
    if not w: return
    n = sum(e["runs"] for e in w); win = sum(e["win_rate"] * e["runs"] for e in w)
    p = win / n
    print(f"  {tag}：{len(w)} 轮 v{w[0]['version']}–{w[-1]['version']} 通关 {p:.4f} 层 {S.mean(e['floor_mean'] for e in w):.2f} 进二幕 {S.mean(e['reach_act2'] for e in w):.3f} 到三幕 {S.mean(e['reach_act3'] for e in w):.3f}"
          f"（单轮 σ {S.pstdev(e['win_rate'] for e in w):.4f}）")
p14 = [e for e in ev if V14 <= e["version"] <= END]
p13 = [e for e in ev if 23828 <= e["version"] < V14]
print("\n评测（种子池 P1 384 局，贪心）")
evb([e for e in ev if e["version"] < 23828][-12:], "第 13 条前 12 轮（第 11 条时间线）")
evb(p13, "第 13 条全部"); evb(p14, "第 14 条全部"); evb(p14[:len(p14)//2], "第 14 条前一半"); evb(p14[len(p14)//2:], "第 14 条后一半"); evb(p14[-12:], "第 14 条最后 12 轮")
# 种子级：评测轮间波动有多少是“翻面种子”
def seeds(w, tag):
    k = len(w); m = [[int(ch_) for ch_ in e["res"]] for e in w if len(e.get("res") or "") == 384]
    if len(m) < 4: return
    k = len(m); pi = [sum(col) / k for col in zip(*m)]
    always = sum(1 for p in pi if p == 1); never = sum(1 for p in pi if p == 0)
    sd_ind = math.sqrt(sum(p * (1 - p) for p in pi)) / 384
    sd_obs = S.pstdev(sum(r) / 384 for r in m)
    print(f"  {tag}（{k} 轮）：总赢 {always} 个种子、总输 {never} 个、翻面 {384-always-never} 个；轮间 σ 实测 {sd_obs:.4f}，按各种子独立翻面算 {sd_ind:.4f}")
print("种子级")
seeds(p13, "第 13 条"); seeds(p14, "第 14 条"); seeds(p14[-12:], "第 14 条最后 12 轮")
# 相邻两轮同种子翻面数
if len(p14) >= 2:
    fl = [sum(a != b for a, b in zip(x["res"], y["res"])) for x, y in zip(p14, p14[1:]) if len(x.get("res") or "") == 384 == len(y.get("res") or "")]
    print(f"  第 14 条相邻两轮同种子结果不同：平均 {S.mean(fl):.1f} 个（{S.mean(fl)/384:.3f}）")
