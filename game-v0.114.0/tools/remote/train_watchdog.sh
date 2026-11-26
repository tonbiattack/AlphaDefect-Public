#!/usr/bin/env bash
# 租用机守护（cron 每 2 分钟）：训练进程不在、且没有 STOP 标记时，从检查点续跑；状态写 $RUN_DIR/watchdog.json。
#   安装：(crontab -l 2>/dev/null; echo "*/2 * * * * bash $HOME/sls2-deploy/sls2/tools/remote/train_watchdog.sh") | crontab -
RUN_DIR=${RUN_DIR:-$HOME/.local/share/sls2/runs/defect}
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
PIDF=$RUN_DIR/learner.pid
alive=0; [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null && alive=1
action=none
if [ $alive = 0 ] && [ ! -f "$RUN_DIR/STOP" ] && [ -f "$RUN_DIR/train_config.json" ]; then
  # 清理上一次留下的引擎（学习进程不在时采集进程会自行退出，这里兜底）。只清本部署目录下的引擎：
  # 同一台机器上可能还有别的训练线（储君与故障机器人各一个部署目录）或评测在用 Sts2Headless（2026-11-06 起）
  pkill -u "$(id -u)" -f "$REPO/engine/Sts2Headless/bin/[R]elease/net9.0/Sts2Headless.dll serve" 2>/dev/null || true
  RUN_DIR=$RUN_DIR bash "$HERE/start_train.sh" >> "$RUN_DIR/watchdog.log" 2>&1
  action=restart
  echo "$(date '+%F %T') 训练不在，已重新拉起" >> "$RUN_DIR/watchdog.log"
fi
df_free=$(df -BG --output=avail "$RUN_DIR" | tail -1 | tr -dc 0-9)
printf '{"time": %s, "alive": %s, "action": "%s", "stop_flag": %s, "disk_free_gb": %s}\n' "$(date +%s)" $alive $action \
  "$([ -f "$RUN_DIR/STOP" ] && echo true || echo false)" "${df_free:-0}" > "$RUN_DIR/watchdog.json"
