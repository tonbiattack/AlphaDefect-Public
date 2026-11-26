#!/usr/bin/env bash
# 猎人训练监管（2026-11-09，猎人决策记录第 1 条）：每 5 分钟在 B 上跑 sup_train.py（学习端）与 sup_eval.py（评测端），
# 遇到 ALERT、没被 IGNORE 的 TRIGGER、连续 3 次取不到读数就打印并退出，让会话醒来处理；最长 HOURS 小时（默认 3）例行退出一次。
set -u
SP=$(dirname "$0")
. "$HOME/dev/sls2/tools/remote/hosts.sh"
END=$(( $(date +%s) + ${HOURS:-3} * 3600 ))
IGNORE=${IGNORE:-'^$'}
FAIL=0
while :; do
  t=$($(host_ssh b) "nice -n 10 python3 -" < "$SP/sup_train.py" 2>&1 | tail -1)
  e=$($(host_ssh b) "nice -n 10 python3 -" < "$SP/sup_eval.py" 2>&1 | tail -1)
  echo "$(date '+%F %T') $t || $e" >> "$SP/supervise.log"
  if [[ "$t" != S* ]] || [[ "$e" != E* ]]; then FAIL=$((FAIL + 1)); else FAIL=0; fi
  [ $FAIL -ge 3 ] && { echo "ALERT:连续3次取不到读数"; echo "$t"; echo "$e"; exit 0; }
  flags=$(echo "$t $e" | grep -o '\(ALERT\|TRIGGER\):[^ ]*' | grep -v -E "$IGNORE")
  [ -n "$flags" ] && { echo "$flags"; echo "$t"; echo "$e"; exit 0; }
  [ $(date +%s) -ge $END ] && { echo "例行检查（${HOURS:-3} 小时到）"; echo "$t"; echo "$e"; exit 0; }
  sleep 300
done
