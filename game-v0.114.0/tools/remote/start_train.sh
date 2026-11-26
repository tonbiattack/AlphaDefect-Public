#!/usr/bin/env bash
# 租用机上启动（或续跑）训练：独立会话、不随 SSH 断开，日志在运行目录的 train.out。已在跑就什么也不做。
#   CONFIG=配置.json [RUN_DIR=...] [EXTRA="--参数 值 ..."] bash start_train.sh
# 停止：touch $RUN_DIR/STOP 后 kill 学习进程（会先存检查点；守护看到 STOP 就不再拉起）；继续：rm $RUN_DIR/STOP 后再运行本脚本。
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
RUN_DIR=${RUN_DIR:-$HOME/.local/share/sls2/runs/defect}
CONFIG=${CONFIG:-$RUN_DIR/train_config.json}
mkdir -p "$RUN_DIR"
[ -f "$CONFIG" ] && [ "$CONFIG" != "$RUN_DIR/train_config.json" ] && cp "$CONFIG" "$RUN_DIR/train_config.json"
PIDF=$RUN_DIR/learner.pid
if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then echo "训练已在运行（pid $(cat "$PIDF")）"; exit 0; fi
[ -f "$RUN_DIR/STOP" ] && { echo "有 STOP 标记，不启动（rm $RUN_DIR/STOP 后再试）"; exit 0; }
# CUDA MPS：学习进程与各采集进程的推理并发使用显卡，而不是分时轮流（实测学习快约 25%）。MPS=0 可关闭。
if [ "${MPS:-1}" = 1 ] && command -v nvidia-cuda-mps-control >/dev/null && ! pgrep -x nvidia-cuda-mps >/dev/null && ! pgrep -f "nvidia-cuda-mps-control -d" >/dev/null; then
  nvidia-cuda-mps-control -d && echo "$(date '+%F %T') 已启动 CUDA MPS" >> "$RUN_DIR/train.out"
fi
cd "$REPO/python"
export DOTNET_ROOT=$HOME/.dotnet PATH=$HOME/.dotnet:$PATH DOTNET_CLI_TELEMETRY_OPTOUT=1
echo "$(date '+%F %T') 启动训练 config=$RUN_DIR/train_config.json ${EXTRA:-}" >> "$RUN_DIR/train.out"
setsid nohup ../.venv/bin/python -I train_defect.py --config "$RUN_DIR/train_config.json" --run-dir "$RUN_DIR" ${EXTRA:-} >> "$RUN_DIR/train.out" 2>&1 < /dev/null &
echo $! > "$PIDF"
echo "已启动 pid $(cat "$PIDF")，日志 $RUN_DIR/train.out"
