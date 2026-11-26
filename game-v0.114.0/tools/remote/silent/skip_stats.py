"""跳过稳赢战斗（猎人决策记录第 14 条）的统计：读运行目录的 log.jsonl（run_choices 里的 SKIP_*、SKIPRUN_*、B3_* 计数与结果头损失），在 B 上跑。
  python3 skip_stats.py /root/.local/share/sls2/runs/silent          # 最近 6 版明细 + 全部累计
  START=24208 python3 skip_stats.py /root/.local/share/sls2/runs/silent   # 另算第 START 版起的每版用时、按局类别的进第三幕 Boss 血量与通关
计数的含义见 python/sts2rl/actor.py（SKIP_ELIG / SKIP_DONE：够条件 / 跳过的场数；SKIP_RES_*：真打的够条件战斗上，实际战后血量比例减结果头预测期望；
SKIP_POT_*：同上的药水预测与实际；SKIPRUN_N / WIN：按局类别（real 完全真打、skip 允许跳过）的局数与通关；B3_*：进第三幕第一个 Boss 时的血量比例与药水瓶数）。
"""
import json, collections, sys
T = sys.argv[1] if len(sys.argv) > 1 else "/root/skipdev/run_test"
L0 = [json.loads(l) for l in open(f"{T}/log.jsonl") if l.strip()]
L = []
for r in L0:        # 回退过的时间线（如第 12 条作废的那段）：只留最新的一条，版本单调递增
    while L and L[-1]["version"] >= r["version"]:
        L.pop()
    L.append(r)
print("版数", len(L))
for r in L[-6:]:
    ch = r.get("run_choices") or {}
    sk = {k: v for k, v in ch.items() if k.startswith("SKIP")}
    done = sum(v for k, v in sk.items() if k.startswith("SKIP_DONE")); elig = sum(v for k, v in sk.items() if k.startswith("SKIP_ELIG:"))
    print(f"v{r['version']} sps {r['sps']} 整局 {r.get('run_episodes')} 段 {r.get('run_segments')} 整局步 {r.get('run_steps')} t_collect {r.get('t_collect')} "
          f"co_hp {r.get('co_hp_loss', float('nan')):.3f} co_pot {r.get('co_pot_loss', float('nan')):.3f} co_n {r.get('co_n')} | 够条件 {elig} 跳过 {done} "
          f"完全真打局 {sk.get('SKIP_REAL_RUN', 0)} 药水抽中 {sk.get('SKIP_POT_USED', 0)} KL {r.get('approx_kl')} 通关 {r.get('run_win_rate')}")
tot = collections.Counter()
for r in L:
    for k, v in (r.get("run_choices") or {}).items():
        if k.startswith("SKIP"): tot[k] += v
print("累计：", {k: round(v, 3) for k, v in sorted(tot.items())})
for key in sorted({k.split(":", 1)[1] for k in tot if k.startswith("SKIP_RES_N:")}):
    n = tot[f"SKIP_RES_N:{key}"]; print(f"  校准残差 {key}: n {n:.0f} 均值 {tot[f'SKIP_RES_SUM:{key}'] / max(n, 1):+.4f}；跳过 {tot.get(f'SKIP_DONE:{key}', 0):.0f} 场，抽到的血量变化均值 {tot.get(f'SKIP_DHP_SUM:{key}', 0) / max(tot.get(f'SKIP_DONE:{key}', 0), 1):+.4f}")
if tot.get("SKIP_POT_N"):
    print(f"  药水头（真打的够条件战斗）：{tot['SKIP_POT_N']:.0f} 瓶 预测用掉 {tot['SKIP_POT_P'] / tot['SKIP_POT_N']:.3f} 实际 {tot['SKIP_POT_Y'] / tot['SKIP_POT_N']:.3f}")
# 重启后（第 START 版起）的统计：每小时版数、按局类别的进第三幕 Boss 血量与通关、卡住
import os
start = int(os.environ.get("START", "0"))
W = [r for r in L if r["version"] >= start]
if len(W) >= 3:
    dt = (W[-1]["time"] - W[1]["time"]) / max(1, len(W) - 2)
    eps = sum(r.get("run_episodes", 0) for r in W[1:]) / max(1, len(W) - 1)
    print(f"重启后 {len(W)} 版：每版 {dt:.1f} 秒（每小时约 {3600 / dt:.0f} 版），每版整局 {eps:.1f}，每小时整局约 {3600 / dt * eps:.0f}；卡住合计 {sum(r.get('stuck', 0) or 0 for r in W)}")
    c = collections.Counter()
    for r in W:
        for k, v in (r.get("run_choices") or {}).items():
            if k.startswith(("B3_", "SKIPRUN_")): c[k] += v
    for kind in ("real", "skip"):
        n = c[f"B3_N:{kind}"]; m = c[f"SKIPRUN_N:{kind}"]
        print(f"  {kind}: 进第三幕 Boss {n:.0f} 次，血量 {c[f'B3_HP:{kind}'] / max(n, 1):.3f}，药水 {c[f'B3_POT:{kind}'] / max(n, 1):.2f}；整局 {m:.0f}，通关 {c[f'SKIPRUN_WIN:{kind}'] / max(m, 1):.3f}")
