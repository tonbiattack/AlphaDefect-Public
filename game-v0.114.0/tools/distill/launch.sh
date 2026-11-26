#!/usr/bin/env bash
# A 上：起搜索蒸馏的标签进程（tools/distill/label_worker.py），已在跑的编号不重复起。由本机中转 tools/distill/relay.py 调用（进程不足时补起）。
# 前一批搜索评测（tools/mcts_eval.py、lookahead_eval.py）还在跑时不起，免得抢引擎和显卡。
#   N=16 ENVS=6 LIVE=4 bash tools/distill/launch.sh
N=${N:-16}; ENVS=${ENVS:-6}; LIVE=${LIVE:-4}
cd "$(dirname "$0")/../.."
export DOTNET_ROOT=$HOME/.dotnet PATH=$HOME/.dotnet:$PATH DOTNET_CLI_TELEMETRY_OPTOUT=1
if pgrep -f "^.venv/bin/python -I tools/(mcts|lookahead)_eval.py" >/dev/null; then echo "搜索评测还在跑，先不起"; exit 0; fi
mkdir -p ~/distill/logs ~/distill/problems ~/distill/out ~/distill/ckpt
n=0
for i in $(seq 0 $((N - 1))); do
  pgrep -f -- "label_worker.py --worker $i --" >/dev/null && continue
  setsid nohup .venv/bin/python -I tools/distill/label_worker.py --worker "$i" --envs "$ENVS" --live "$LIVE" --device cuda \
    >> ~/distill/logs/w$i.log 2>&1 < /dev/null &
  n=$((n + 1))
done
echo "起了 $n 个"
