#!/usr/bin/env bash
# 本机运行：停铁甲战士训练线（2026-11-20 起默认在 B，HOST=c 停 C 上的）（2026-11-17，铁甲战士决策记录第 1 条；照 tools/remote/fool/stop.sh 与恐惧线的 stop.sh）。学习端放 STOP（守护不再拉起）后 TERM 学习进程（会先存检查点）；
# PARTS 含 eval 时连评测一起停。只按本线的运行目录与配置文件名匹配进程，恐惧线（runs/defect-dread*、/root/dread-deploy）不动。
#   bash tools/remote/ironclad/stop.sh               # 只停学习端（评测等新权重，不用停）
#   PARTS="train eval" bash tools/remote/ironclad/stop.sh
set -euo pipefail
. "$(dirname "$0")/../hosts.sh"
H=${HOST:-b}
host_require "$H"
RUN=/root/.local/share/sls2/runs/ironclad
PARTS=${PARTS:-train}
if [[ " $PARTS " == *" train "* ]]; then
  $(host_ssh "$H") "cd $RUN 2>/dev/null || exit 0
    touch STOP
    if [ -f learner.pid ] && P=\$(cat learner.pid) && kill -0 \$P 2>/dev/null; then
      kill -TERM \$P; for i in \$(seq 90); do kill -0 \$P 2>/dev/null || break; sleep 2; done
      kill -0 \$P 2>/dev/null && { echo 学习进程停不下来（STOP 仍在）; exit 1; }
    fi
    echo 铁甲战士学习进程已停（STOP 在 $RUN/STOP）"
fi
if [[ " $PARTS " == *" eval "* ]]; then
  $(host_ssh "$H") "pkill -TERM -f '[r]emote_eval.py --config /root/ironclad_train_config.json' || true
    for i in \$(seq 30); do pgrep -f '[r]emote_eval.py --config /root/ironclad_train_config.json' >/dev/null || break; sleep 2; done
    desc() { for c in \$(pgrep -P \"\$1\"); do echo \"\$c\"; desc \"\$c\"; done; }
    for p in \$(pgrep -f '[r]emote_eval.py --config /root/ironclad_train_config.json'); do kill -9 \$p \$(desc \$p) 2>/dev/null; done
    echo 铁甲战士评测已停"
fi
