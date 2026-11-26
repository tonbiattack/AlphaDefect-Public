#!/bin/bash
# 在协同采集机（A）上重启常规评测（tools/remote/remote_eval.py，决策记录 198）：杀掉旧的评测进程树（按进程名找 remote_eval 与其评测子进程），再起新的。
# 原来只放在 A 的 /root/a_eval_restart.sh，决策记录 203 起收进仓库并加 --keep-runs：每轮的逐局结果留最近 100 轮、权重留最近 2 轮，千种子复测沿用。
# 决策记录 240 起种子改用种子池 tools/remote/seed_pool.json（评测取其中的 eval 部分）；决策记录 276 起改用 P1 全部 1,000 个（seed_pool_p1_full.json）。
# 决策记录 223 起加过 --ema（用学习端的滑动平均权重）；2026-11-06 用户要求改回原始权重（滑动平均跟得慢，评测响应迟），去掉 --ema（决策记录 279）。
# 决策记录 206 起加过 --merge-equiv（贪心按等价动作合并），10-31 用户决定取消（决策记录 207），评测与复测都回到逐候选取最大。
#   bash /root/a_eval_restart.sh            # EXTRA_PIDS="…" 另杀几个进程（连同子进程）
desc() { for c in $(pgrep -P "$1"); do echo "$c"; desc "$c"; done; }
for p in $(pgrep -f '^\.venv/bin/python -I tools/remote/remote_eval\.py'); do kill -9 $p $(desc $p) 2>/dev/null; done
for p in ${EXTRA_PIDS:-}; do kill -9 $p $(desc $p) 2>/dev/null; done
sleep 3
cd /root/sls2-deploy/sls2 && export DOTNET_ROOT=$HOME/.dotnet PATH=$HOME/.dotnet:$PATH DOTNET_CLI_TELEMETRY_OPTOUT=1
setsid nohup .venv/bin/python -I tools/remote/remote_eval.py --config /root/b_train_config.json --run-dir /root/.local/share/sls2/runs/defect-eval \
  --seeds-file tools/remote/seed_pool_p1_full.json --workers 8 --envs 12 --device cuda --cores 20 --nice 5 --keep-runs 100 \
  >> /root/.local/share/sls2/runs/defect-eval/remote_eval.out 2>&1 < /dev/null &
sleep 5; tail -1 /root/.local/share/sls2/runs/defect-eval/remote_eval.out
