#!/usr/bin/env bash
# 亡灵契约师线的引擎看门狗（2026-11-19，亡灵契约师决策记录第 12 条；照铁甲战士线 ironclad-train c5d18d9 的 engine_watchdog.sh）：
# 每分钟由 cron 调用，杀掉本线部署目录里内存超过 LIMIT_MB（默认 3,072）的引擎进程。
# 起因（铁甲线转来）：引擎偶尔在一步里失控、内存一直涨到 25–27 GB，步数上限管不到（卡在单步里），交换区占满后学习端与评测都停；
# 铁甲线 4 次都在单场练习（第二幕 HUNTER_KILLER_NORMAL、第三幕 AEONGLASS_BOSS、QUEEN_BOSS）。本线到 2026-11-19 还没碰到（dmesg 没有 OOM，引擎最大约 0.29 GB）。
# 杀掉后采集端按“引擎报错”结束那一局（记进 stuck.jsonl）并重开引擎，评测按卡住算负，与系统 OOM 杀进程的结果相同。
# 只匹配 /root/necro-deploy 的引擎（采集与常规评测都用它），别的线的不碰。DRY=1 只记录、不杀（核对用）。
#   * * * * * bash /root/necro-deploy/sls2/tools/remote/necro/engine_watchdog.sh
set -u
LIMIT_KB=$(( ${LIMIT_MB:-3072} * 1024 ))
LOG=${LOG:-/root/.local/share/sls2/runs/necro/engine_watchdog.log}
for p in $(pgrep -f 'necro-deploy/sls2/engine/Sts2Headless'); do
  r=$(awk '/^VmRSS/{print $2}' "/proc/$p/status" 2>/dev/null)
  if [ "${r:-0}" -gt "$LIMIT_KB" ]; then
    if [ "${DRY:-0}" = 1 ]; then
      echo "$(date '+%F %T') [演练] 会杀引擎 $p，内存 $((r / 1024)) MB" >> "$LOG"
    else
      kill -9 "$p" 2>/dev/null && echo "$(date '+%F %T') 杀掉引擎 $p，内存 $((r / 1024)) MB；$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | cut -c1-120)" >> "$LOG"
    fi
  fi
done
