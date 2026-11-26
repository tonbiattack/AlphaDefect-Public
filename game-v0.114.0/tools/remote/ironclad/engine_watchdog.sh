#!/usr/bin/env bash
# 铁甲战士线的引擎看门狗（2026-11-19，铁甲战士决策记录第 9 条补记）：每分钟由 cron 调用，杀掉本线部署目录里内存超过 LIMIT_MB（默认 3,072）的引擎进程。
# 起因：引擎偶尔在一步里失控、内存一直涨，步数上限管不到（卡在单步里）：
#   第 3,503 版 第二幕练习 HUNTER_KILLER_NORMAL 第 12 步，涨到 27.6 GB 被系统 OOM 杀掉；
#   第 20,230 版 第三幕 Boss 练习 AEONGLASS_BOSS 第 98 步，涨到 25 GB，16 GB 交换区占满，学习端与评测停了约半小时，手动杀掉。
# 正常引擎约 150–220 MB（2026-11-19 量的 188 个：中位 171、最大 217 MB）。杀掉后采集端按“引擎报错”结束那一局（记进 stuck.jsonl）并重开引擎，
# 评测按卡住算负，与系统 OOM 杀进程的结果相同。只匹配 /root/ironclad-deploy 的引擎，恐惧线、亡灵契约师线的不碰。
#   * * * * * bash /root/ironclad-deploy/sls2/tools/remote/ironclad/engine_watchdog.sh
set -u
LIMIT_KB=$(( ${LIMIT_MB:-3072} * 1024 ))
LOG=${LOG:-/root/.local/share/sls2/runs/ironclad/engine_watchdog.log}
for p in $(pgrep -f 'ironclad-deploy/sls2/engine/Sts2Headless'); do
  r=$(awk '/^VmRSS/{print $2}' "/proc/$p/status" 2>/dev/null)
  if [ "${r:-0}" -gt "$LIMIT_KB" ]; then
    kill -9 "$p" 2>/dev/null && echo "$(date '+%F %T') 杀掉引擎 $p，内存 $((r / 1024)) MB" >> "$LOG"
  fi
done
