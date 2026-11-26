#!/usr/bin/env bash
# 价值诊断专用采集：追加日志保留远端；原子完成的 NPZ 经 rsync 成功送达本机后清理远端缓冲。
# 不删本机数据、不接触训练检查点。默认执行一次，LOOP=1 每 30 秒继续。
set -u
. "$(dirname "$0")/hosts.sh"
SOURCE_ROOT=$REPO
if [ -f "$REPO/.hotdev-source" ]; then SOURCE_ROOT=$(cat "$REPO/.hotdev-source"); fi
VP_ROOT=${VALUE_PROBE_ROOT:-$SOURCE_ROOT/runs/value_probe}
VP_INDEX=${VALUE_PROBE_INDEX:-$REPO/runs/value-probe-index.sqlite3}
mkdir -p "$VP_ROOT" "$(dirname "$VP_INDEX")"
exec 8>"$(dirname "$VP_INDEX")/value-probe-pull.lock"
flock -n 8 || exit 0
VP_SCRIPT=$(readlink -f "${BASH_SOURCE[0]}")
VP_SCRIPT_MTIME=$(stat -c %Y "$VP_SCRIPT")
while true; do
  if [ "$(stat -c %Y "$VP_SCRIPT")" != "$VP_SCRIPT_MTIME" ]; then exec bash "$VP_SCRIPT" "$@"; fi
  sync_failed=0
  VP_HOSTS=$(host_active)
  # 仅同步已确认启用探针的机器；例如 A 只中转权重、actors=0 时没有诊断目录。
  if [ -f "$VP_ROOT/activation.json" ]; then
    VP_HOSTS=$(python3 - "$VP_ROOT/activation.json" <<'PY'
import json,sys
print(' '.join(json.load(open(sys.argv[1]))['hosts']))
PY
)
  fi
  # 探针启用尚待批准时，只留存本机已有日志，不执行任何远端操作。
  if [ -n "${VALUE_PROBE_LOCAL_ONLY:-}" ] || [ -f "$VP_ROOT/activation_pending.json" ]; then VP_HOSTS=""; fi
  for h in $VP_HOSTS; do
    A=$(host_addr "$h"); R=$(host_run "$h"); E=$(host_rsync_e "$h"); L=$VP_ROOT/$h
    mkdir -p "$L/metrics" "$L/episodes" "$L/batches"
    # 只有本探针创建的子目录；不存在代表该机器尚未启用采集。
    timeout 45 rsync -az --ignore-missing-args --exclude='*.tmp' -e "$E" "$A:$R/value_probe/metrics/" "$L/metrics/" || sync_failed=1
    timeout 30 rsync -az --ignore-missing-args -e "$E" "$A:$R/value_probe/*_status.json" "$L/" || sync_failed=1
    for part in episodes batches; do
      timeout 60 rsync -az --ignore-missing-args --remove-source-files --include='*.npz' --exclude='*' -e "$E" \
        "$A:$R/value_probe/$part/" "$L/$part/" || sync_failed=1
    done
  done
  VP_MAIN=$(main_host)
  python3 "$REPO/tools/value_probe_summary.py" --root "$VP_ROOT" --index "$VP_INDEX" --host "$VP_MAIN" --legacy-log "$(host_local "$VP_MAIN")/log.jsonl"
  # 只读线性诊断每小时最多一次，不训练策略网络；在本机运行，单线程、低优先级。
  VP_BASE_TS="$(dirname "$VP_INDEX")/value-probe-baseline.ts"
  if [ ! -f "$VP_BASE_TS" ] || [ $(( $(date +%s) - $(stat -c %Y "$VP_BASE_TS") )) -ge 3600 ]; then
    if OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 nice -n 15 "$REPO/.venv/bin/python" -I "$REPO/tools/value_probe_analysis.py" fit-baseline --root "$VP_ROOT" > "$VP_ROOT/baseline.log" 2>&1; then
      touch "$VP_BASE_TS"
    fi
  fi
  python3 - "$VP_ROOT" "$sync_failed" <<'PY'
import json,os,sys,time
from pathlib import Path
p=Path(sys.argv[1])/'sync.json';t=p.with_suffix('.tmp')
t.write_text(json.dumps({'time':time.time(),'ok':sys.argv[2]=='0'}));os.replace(t,p)
PY
  [ -n "${LOOP:-}" ] || break
  sleep 30
done
