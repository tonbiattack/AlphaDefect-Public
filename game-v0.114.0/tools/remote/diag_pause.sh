#!/usr/bin/env bash
# 本机运行：为平台期诊断暂停全部训练（2026-11-03，用户：“暂停所有训练进程，开始报告建议的诊断，充分利用性能”）。
#   bash tools/remote/diag_pause.sh pause     # B 存检查点后停学习进程与采集引擎、备份；A 停采集与评测
#   bash tools/remote/diag_pause.sh resume    # 恢复（只在用户下令时用）
# 和万种测试（w10k.sh）的停训做法一样，但不设“到点自动恢复”：用户要求停到另行通知。
# 标记：B 上运行目录的 STOP.diag，A 上 /root/w10k/PAUSED 与评测目录的 PAUSE；直播服务不动。
set -u
cd "$(dirname "$0")/../.." && . tools/remote/hosts.sh
SA=$(host_ssh a); SB=$(host_ssh b); DB=$(host_deploy b); RB=$(host_run b)
log() { echo "$(date '+%F %T') $*"; }

case "${1:-}" in
pause)
  log "B：存检查点后停学习进程，停采集引擎，备份"
  $SB "set -e; cd \$HOME/$RB
$REMOTE_ALIVE
$REMOTE_STOP
touch STOP.diag
sleep 5; pkill -9 -f '[s]ls2-deploy/sls2/engine/Sts2Headless' || true; sleep 2; echo B 剩余训练引擎 \$(pgrep -fc '[s]ls2-deploy/sls2/engine/Sts2Headless' || true)
TAG=诊断前 RUN_DIR=\$HOME/$RB bash $DB/tools/remote/ckpt_backup.sh && echo 已备份 \$(tail -1 backups/backup.log)" < /dev/null \
    || { log "B 停训失败"; exit 1; }
  log "A：停采集与评测"
  $SA "mkdir -p /root/w10k; touch /root/w10k/PAUSED \$HOME/.local/share/sls2/runs/defect-eval/PAUSE
pkill -f '[t]ools/remote/remote_actors.py --config'; desc() { for c in \$(pgrep -P \$1); do echo \$c; desc \$c; done; }
for p in \$(pgrep -f '^\.venv/bin/python -I tools/remote/remote_eval\.py'); do kill -9 \$p \$(desc \$p) 2>/dev/null; done
sleep 8; pkill -9 -f '[t]ools/remote/remote_actors.py --config'; pkill -9 -f '[s]ls2-deploy/sls2/engine/Sts2Headless'; true
echo A 剩余引擎 \$(pgrep -fc '[s]ls2-deploy/sls2/engine/Sts2Headless' || true); nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader" < /dev/null
  log "已暂停"
  ;;
resume)
  log "恢复训练"
  $SB "cd \$HOME/$RB && rm -f STOP STOP.diag && RUN_DIR=\$HOME/$RB bash $DB/tools/remote/start_train.sh" < /dev/null 2>&1 | tail -2
  $SA "rm -f /root/w10k/PAUSED \$HOME/.local/share/sls2/runs/defect-eval/PAUSE; bash /root/collector_restart.sh; bash /root/a_eval_restart.sh" < /dev/null 2>&1 | tail -2
  log "已恢复"
  ;;
*) echo "用法：$0 pause|resume"; exit 2 ;;
esac
