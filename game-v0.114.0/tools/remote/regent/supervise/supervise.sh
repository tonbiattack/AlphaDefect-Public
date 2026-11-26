#!/usr/bin/env bash
# 储君训练监管（2026-11-07，用户：“你帮我监管训练，挪额度什么的你自己看着来……我暂时不监管。”）
# 每 5 分钟跑一次 sup_b.py（学习端）与 sup_a.py（评测端）。遇到下列情况就打印并退出，让会话醒来处理：
#   任何 ALERT；第二幕普通战练习 ≥ 96%；评测 750 版涨幅 < 0.15；第三幕 Boss 练习 ≥ 3% 且评测到第三幕 ≥ 25%（两条同时）；
#   出现新的卡住记录；连续 3 次连不上机器。最长 HOURS 小时（默认 3）例行退出一次。IGNORE 为要忽略的触发的正则（已处理过的）。
set -u
SP=$(dirname "$0")
. "$HOME/dev/sls2/tools/remote/hosts.sh"
END=$(( $(date +%s) + ${HOURS:-3} * 3600 ))
IGNORE=${IGNORE:-'^$'}
STUCK0=""
FAIL=0
FLAT=0
SV0=""; SC0=""
while :; do
  b=$($(host_ssh b) "nice -n 10 python3 -" < "$SP/sup_b.py" 2>&1 | tail -1)
  a=$($(host_ssh a) "nice -n 10 python3 -" < "$SP/sup_a.py" 2>&1)
  n=$(echo "$a" | sed -n 's/^STUCK_N=//p'); a=$(echo "$a" | grep -v '^STUCK_N=' | tail -1)
  echo "$(date '+%F %T') $b || $a" >> "$SP/supervise.log"
  if [[ "$b" != B* ]] || [[ "$a" != A* ]]; then FAIL=$((FAIL + 1)); else FAIL=0; fi
  [ $FAIL -ge 3 ] && { echo "ALERT:连续 3 次取不到读数"; echo "$b"; echo "$a"; exit 0; }
  [ -z "$STUCK0" ] && STUCK0=$n
  wake=""
  flags=$(echo "$b $a" | grep -o '\(ALERT\|TRIGGER\):[^ ]*' | grep -v -E "$IGNORE")
  echo "$flags" | grep -q '^ALERT' && wake=1
  echo "$flags" | grep -q '第二幕普通战\|第二幕精英练习' && wake=1
  # 走平（降学习率的判断）：练习 1,000 版走平与评测 750 版通关涨不到 1 个百分点同时成立才算；单独一项噪声太大（2026-11-07）
  # 两项都会随分段边界来回摆，要连续 FLAT_N 次检查（默认 6 次，约 30 分钟、500 版）都成立才叫醒（第 34,124 版误报后，2026-11-08）
  if echo "$flags" | grep -q '练习走平' && echo "$flags" | grep -q '750 版涨幅\|750版涨幅'; then FLAT=$((FLAT + 1)); else FLAT=0; fi
  [ $FLAT -ge ${FLAT_N:-6} ] && wake=1
  echo "$flags" | grep -q '第三幕 Boss 练习' && echo "$flags" | grep -q '评测到第三幕' && wake=1
  [ -n "$n" ] && [ -n "$STUCK0" ] && [ "$n" -gt "$STUCK0" ] && { wake=1; flags="$flags ALERT:新的卡住记录（$STUCK0 → $n）"; }
  # 训练判输频率：每满 1,000 版算一次，超过每千版 100 次叫醒（平常 15–25 次，第 36,000 版起约 50 次，多是补刀拖延；2026-11-08）
  bv=$(echo "$b" | sed -n 's/^B v\([0-9]*\).*/\1/p'); sc=$(echo "$b" | sed -n 's/.*B采集判输累计\([0-9]*\).*/\1/p')
  if [ -n "$bv" ] && [ -n "$sc" ]; then
    if [ -z "$SV0" ]; then SV0=$bv; SC0=$sc
    elif [ $((bv - SV0)) -ge 1000 ]; then
      r=$(( (sc - SC0) * 1000 / (bv - SV0) ))
      # 第 43,999 版起每版数据翻倍（65,536 步），线 100 → 200
      [ $r -gt 200 ] && { wake=1; flags="$flags ALERT:训练判输每千版${r}次（第${SV0}–${bv}版）"; }
      SV0=$bv; SC0=$sc
    fi
  fi
  if [ -n "$wake" ]; then echo "$flags"; echo "$b"; echo "$a"; exit 0; fi
  [ $(date +%s) -ge $END ] && { echo "例行检查（${HOURS:-3} 小时到）"; echo "$b"; echo "$a"; exit 0; }
  sleep 300
done
