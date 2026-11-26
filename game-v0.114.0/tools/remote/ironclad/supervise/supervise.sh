#!/usr/bin/env bash
# 铁甲战士从零线监管（2026-11-17，铁甲战士决策记录第 1 条，照恐惧模式故障机器人从零线的 supervise.sh）：每 5 分钟在训练机（2026-11-20 起为 B，HOST 可改）上跑 sup_train.py（学习端）与 sup_eval.py（评测端），
# 遇到 ALERT、没被 IGNORE 的 TRIGGER、连续 3 次取不到读数就打印并退出，让会话醒来处理；最长 HOURS 小时（默认 3）例行退出一次。
# MINUTES 优先于 HOURS（2026-11-17 用户要求没有唤醒事件也每半小时巡检一次：MINUTES=30）。
set -u
SP=$(dirname "$0")
. "$SP/../../hosts.sh"
H=${HOST:-b}
MIN=${MINUTES:-$(( ${HOURS:-3} * 60 ))}
END=$(( $(date +%s) + MIN * 60 ))
IGNORE=${IGNORE:-'^$'}
FAIL=0
while :; do
  t=$($(host_ssh $H) "nice -n 10 python3 -" < "$SP/sup_train.py" 2>&1 | tail -1)
  e=$($(host_ssh $H) "nice -n 10 python3 -" < "$SP/sup_eval.py" 2>&1 | tail -1)
  echo "$(date '+%F %T') $t || $e" >> "$HOME/.local/share/sls2/runs/remote/ironclad_supervise.log"
  if [[ "$t" != S* ]] || [[ "$e" != E* ]]; then FAIL=$((FAIL + 1)); else FAIL=0; fi
  [ $FAIL -ge 3 ] && { echo "ALERT:连续3次取不到读数"; echo "$t"; echo "$e"; exit 0; }
  flags=$(echo "$t $e" | grep -o '\(ALERT\|TRIGGER\):[^ ]*' | grep -v -E "$IGNORE")
  [ -n "$flags" ] && { echo "$flags"; echo "$t"; echo "$e"; exit 0; }
  [ $(date +%s) -ge $END ] && { echo "例行检查（${MIN} 分钟到）"; echo "$t"; echo "$e"; exit 0; }
  sleep 300
done
