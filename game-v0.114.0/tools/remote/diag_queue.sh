#!/usr/bin/env bash
# 租用机上运行：平台期诊断的任务队列（2026-11-03）。逐个任务用全部分片打 P1 的 1,000 个种子（tools/ability_test.py），打完一个再打下一个。
#   setsid nohup bash tools/remote/diag_queue.sh /root/diag/jobs.txt > /root/diag/queue.log 2>&1 &
# 任务文件每行：名字|检查点|ability_test 的其余参数（如 --temp 1.0 --policy-seed 3、--perturb 0.01 --perturb-seed 1、--ensemble a.pt,b.pt）
# SEEDS 换种子文件（默认 P1）、RUNS 取其前多少个。结果在 /root/diag/<名字>/runs_s*.jsonl；已打过的局会跳过（可续跑）。SH 个分片、每片 ENVS 个引擎。
set -u
cd "$(dirname "$0")/../.."
export DOTNET_ROOT=$HOME/.dotnet PATH=$HOME/.dotnet:$PATH DOTNET_CLI_TELEMETRY_OPTOUT=1
JOBS=$1; SEEDS=${SEEDS:-tools/remote/seed_pool.json}; SH=${SH:-12}; ENVS=${ENVS:-20}; RUNS=${RUNS:-1000}; D=/root/diag
log() { echo "$(date '+%F %T') $*"; }
while IFS='|' read -r name ck extra <&3; do
  [ -z "$name" ] || [ "${name:0:1}" = "#" ] && continue
  O=$D/$name; mkdir -p "$O"
  n=$(cat "$O"/runs_s*.jsonl 2>/dev/null | grep -c . || true)
  [ "${n:-0}" -ge "$RUNS" ] && { log "$name 已打完（$n 局），跳过"; continue; }
  log "开始 $name：$ck $extra"
  for k in $(seq 0 $((SH - 1))); do
    .venv/bin/python -I tools/ability_test.py --ckpt "$ck" --out "$O" --asc 10-10 --runs $RUNS --shard $k/$SH --envs $ENVS --device cuda \
      --seeds-file $SEEDS $extra > "$O/log_s$k.txt" 2>&1 < /dev/null &
  done
  wait
  log "完成 $name：$(cat "$O"/runs_s*.jsonl | grep -c .) 局，通关 $(cat "$O"/runs_s*.jsonl | grep -c '"win": true')"
done 3< "$JOBS"
log "队列结束"
