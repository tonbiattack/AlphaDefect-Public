#!/usr/bin/env bash
# 本机运行：停储君训练线（2026-11-06）。B：放 STOP（守护不再拉起）后 TERM 学习进程（会先存检查点），等它退出；A：停远程采集与评测。
# 只按储君的部署目录、配置文件名匹配进程，故障机器人的不动。crontab 里储君的两条留着（STOP 在时守护什么也不做）。
#   bash tools/remote/regent/stop.sh             # 全部
#   PARTS="a" bash tools/remote/regent/stop.sh   # 只停 A
# A 到期后（hosts.json 里 a 标为 pending，储君决策记录第 26 条）默认只停 B 的学习进程；B 上的常规评测不用停（等新权重），要停时 PARTS="beval"
set -euo pipefail
. "$(dirname "$0")/../hosts.sh"
RUN=/root/.local/share/sls2/runs/regent
if host_pending a; then PARTS=${PARTS:-b}; else PARTS=${PARTS:-b a}; fi
if [[ " $PARTS " == *" b "* ]]; then
  echo "== B"
  $(host_ssh b) "cd $RUN 2>/dev/null || exit 0
    touch STOP
    if [ -f learner.pid ] && P=\$(cat learner.pid) && kill -0 \$P 2>/dev/null; then
      kill -TERM \$P; for i in \$(seq 90); do kill -0 \$P 2>/dev/null || break; sleep 2; done
      kill -0 \$P 2>/dev/null && { echo 学习进程停不下来（STOP 仍在）; exit 1; }
    fi
    echo 学习进程已停（STOP 在 $RUN/STOP）"
fi
if [[ " $PARTS " == *" beval "* ]]; then
  echo "== B：常规评测"
  $(host_ssh b) "pkill -TERM -f '[r]emote_eval.py --config /root/regent_train_config.json' || true
    for i in \$(seq 30); do pgrep -f '[r]emote_eval.py --config /root/regent_train_config.json' >/dev/null || break; sleep 2; done
    desc() { for c in \$(pgrep -P \"\$1\"); do echo \"\$c\"; desc \"\$c\"; done; }
    for p in \$(pgrep -f '[r]emote_eval.py --config /root/regent_train_config.json'); do echo 强制结束 \$p; kill -9 \$p \$(desc \$p) 2>/dev/null; done
    pgrep -fa '[r]emote_eval.py --config /root/regent_train_config.json' && echo 还有进程没退出 || echo B 上评测已停"
fi
if [[ " $PARTS " == *" a "* ]]; then
  echo "== A"
  $(host_ssh a) "pkill -TERM -f '[r]emote_actors.py --config /root/regent_train_config.json' || true
    pkill -TERM -f '[r]emote_eval.py --config /root/regent_train_config.json' || true
    for i in \$(seq 30); do pgrep -f '[r]emote_(actors|eval).py --config /root/regent_train_config.json' >/dev/null || break; sleep 2; done
    # 60 秒还没退出（评测进程等手上的对局）：连同子进程（评测工作进程、引擎）强制结束
    desc() { for c in \$(pgrep -P \"\$1\"); do echo \"\$c\"; desc \"\$c\"; done; }
    for p in \$(pgrep -f '[r]emote_(actors|eval).py --config /root/regent_train_config.json'); do echo 强制结束 \$p; kill -9 \$p \$(desc \$p) 2>/dev/null; done
    sleep 2; pgrep -fa '[r]emote_(actors|eval).py --config /root/regent_train_config.json' && echo 还有进程没退出 || echo 远程采集与评测已停"
fi
