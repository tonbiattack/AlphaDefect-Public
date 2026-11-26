#!/usr/bin/env bash
# 模型能力测试（tools/ability_test.py，在 A 上跑）的报告实时更新：每 EVERY 秒（默认 120）从 A 取回逐局结果，重新生成报告，
# 写到原目录（Windows 侧）并放进看板公开副本的 replay/ability.html 推到香港服务器（https://dashboard.example/replay/ability.html）。
# 全部打完后把 A 上的整个测试目录（逐局结果、回放、引擎日志、所用检查点）取回本机留档，生成最终报告后退出。
#   bash tools/ability_live.sh v44820 1000 0-10
set -u
TAG=$1; TARGET=${2:-1000}; ASC=${3:-0-10}; EVERY=${EVERY:-120}
cd "$(dirname "$0")/.." && . tools/remote/hosts.sh
L=$HOME/.local/share/sls2/ability_test/$TAG; mkdir -p "$L"
AREM=${AREM:-ability_test}       # A 上的测试目录（相对 A 的家目录或绝对路径）
WIN=/mnt/c/主存储文件/脚本开发/sls2/模型能力测试报告_$TAG.html
PUB=$HOME/.local/share/sls2/runs/remote/public/replay/ability.html
E=$(host_rsync_e a); AD=$(host_addr a)
lo=${ASC%-*}; hi=${ASC#*-}; TOTAL=$((TARGET * (hi - lo + 1)))
report() {
  python3 tools/ability_report.py "$L" "$L/report.html" --target "$TARGET" --asc "$ASC" ${NOTES:+--notes "$NOTES"} >/dev/null || return 1
  cp "$L/report.html" "$WIN"; cp "$L/report.summary.json" "${WIN%.html}.summary.json"; cp "$L/report.html" "$PUB"
  bash tools/remote/publish_replay.sh || true
}
while true; do
  rsync -a --include='runs_s*.jsonl' --exclude='*' -e "$E" "$AD:$AREM/" "$L/" 2>/dev/null
  n=$(cat "$L"/runs_s*.jsonl 2>/dev/null | wc -l)
  if [ "$n" -gt 0 ]; then report; fi
  echo "$(date '+%F %T') $n / $TOTAL 局"
  if [ "$n" -ge "$TOTAL" ]; then break; fi
  sleep "$EVERY"
done
rsync -a -e "$E" "$AD:$AREM/" "$L/" && echo "已取回 A 上的测试目录：$(du -sh "$L" | cut -f1)"
python3 -c "import json,sys,time; json.dump({'finished': time.time(), 'archive': sys.argv[1]}, open(sys.argv[1] + '/meta.json', 'w'), ensure_ascii=False)" "$L"
report && echo "最终报告：$WIN"
