#!/usr/bin/env bash
# 亡灵契约师训练监管（2026-11-17，照 tools/remote/fool/supervise/supervise.sh）：每 5 分钟在 B 上跑 sup_train.py（学习端）与 sup_eval.py（评测端），
# 遇到 ALERT、没被 IGNORE 的 TRIGGER、连续 3 次取不到读数就打印并退出，让会话醒来处理；最长 HOURS 小时（默认 3）例行退出一次，
# 给了 MINUTES 就按分钟（2026-11-17 用户：“即使没有唤醒事件，也每半个小时简单巡检一次”，用 MINUTES=30）。
#   bash tools/remote/necro/supervise/supervise.sh                       # 日志在 ~/.local/share/sls2/runs/remote/necro_supervise.log
#   IGNORE='二幕普通战练习到' HOURS=6 bash tools/remote/necro/supervise/supervise.sh
#   MINUTES=30 bash tools/remote/necro/supervise/supervise.sh
set -u
SP=$(dirname "$0")
. "$SP/../../hosts.sh"
MIN=${MINUTES:-$(( ${HOURS:-3} * 60 ))}
END=$(( $(date +%s) + MIN * 60 ))
IGNORE=${IGNORE:-'^$'}
FAIL=0
while :; do
  t=$($(host_ssh b) "nice -n 10 python3 -" < "$SP/sup_train.py" 2>&1 | tail -1)
  e=$($(host_ssh b) "nice -n 10 python3 -" < "$SP/sup_eval.py" 2>&1 | tail -1)
  echo "$(date '+%F %T') $t || $e" >> "$HOME/.local/share/sls2/runs/remote/necro_supervise.log"
  if [[ "$t" != S* ]] || [[ "$e" != E* ]]; then FAIL=$((FAIL + 1)); else FAIL=0; fi
  [ $FAIL -ge 3 ] && { echo "ALERT:连续3次取不到读数"; echo "$t"; echo "$e"; exit 0; }
  flags=$(echo "$t $e" | grep -o '\(ALERT\|TRIGGER\):[^ ]*' | grep -v -E "$IGNORE")
  [ -n "$flags" ] && { echo "$flags"; echo "$t"; echo "$e"; exit 0; }
  [ $(date +%s) -ge $END ] && { echo "例行检查（${MIN} 分钟到）"; echo "$t"; echo "$e"; exit 0; }
  sleep 300
done
