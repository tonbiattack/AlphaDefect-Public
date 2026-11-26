#!/usr/bin/env bash
# 在 B 上顺序跑搜索第 0 步诊断（研究报告/2026-10-30_战斗搜索方案调查 第 4.0 节），训练停着的这段时间用。
# 各阶段结束时、整个脚本退出时（含出错、被杀、超时）都会删掉 STOP 并拉起训练，所以训练不会因为这个脚本一直停着。
#   nohup bash /root/sls2-diag/sls2/tools/search_diag_run.sh > /root/sls2-diag/out/driver.out 2>&1 &
# 环境变量：SHARDS（分片数，默认 8）、ENVS（每片引擎数，默认 4）、MAXH（总时限小时，默认 2，到时剩下的阶段跳过）、
#          CK（被测检查点）、RESUME=0 时不自动恢复训练。
set -u
D=/root/sls2-diag; O=$D/out; PY=/root/sls2-deploy/sls2/.venv/bin/python
CK=${CK:-$D/ckpt.pt}; EX=$D/boss3_exam.json
SHARDS=${SHARDS:-8}; ENVS=${ENVS:-4}
DEADLINE=$(( $(date +%s) + ${MAXH:-2} * 3600 ))
R=$HOME/.local/share/sls2/runs/defect
export DOTNET_ROOT=$HOME/.dotnet PATH=$HOME/.dotnet:$PATH DOTNET_CLI_TELEMETRY_OPTOUT=1
cd $D/sls2
log() { echo "$(date '+%F %T') $*" | tee -a $O/driver.log; }
resume() {
  [ "${RESUME:-1}" = 1 ] || { log "RESUME=0，不恢复训练"; return; }
  pkill -f 'tools/search_diag.py' 2>/dev/null; sleep 3
  pkill -u "$(id -u)" -f "$D/sls2/engine/Sts2Headless" 2>/dev/null
  rm -f $R/STOP
  RUN_DIR=$R bash /root/sls2-deploy/sls2/tools/remote/start_train.sh >> $O/driver.log 2>&1
  log "训练已恢复"
}
trap resume EXIT

# phase 名称 子命令 参数...   （每个分片一个进程，全部结束后进入下一阶段）
phase() {
  local name=$1; shift
  [ "$(date +%s)" -lt "$DEADLINE" ] || { log "超时，跳过 $name"; return; }
  log "开始 $name"
  for k in $(seq 0 $((SHARDS - 1))); do
    nohup nice -n 0 $PY -I tools/search_diag.py "$@" --ckpt $CK --exam $EX --out $O/${name}_s$k.jsonl --shard $k/$SHARDS \
      --envs $ENVS --live 2 --threads 1 --device cuda > $O/${name}_s$k.log 2>&1 &
  done
  wait
  log "结束 $name：$(cat $O/${name}_s*.jsonl 2>/dev/null | wc -l) 行"
}

log "被测检查点 $CK；分片 $SHARDS × 引擎 $ENVS；时限 ${MAXH:-2} 小时"
# J0 贪心基线（同一检查点、同一引擎，用来和搜索逐题配对，也核对与复测的贪心一致）
if [ ! -s $O/greedy.jsonl ]; then
  log "开始 greedy"
  $PY -I tools/boss3_exam.py run --ckpt $CK --exam $EX --out $O/greedy.jsonl --envs 32 --device cuda > $O/greedy.log 2>&1
  log "结束 greedy：$(tail -1 $O/greedy.log)"
fi
# 先等已经在跑的分片结束（重启驱动时它们可能还没做完）；search_a 已做完的题会跳过
while pgrep -f "[t]ools/search_diag.py" > /dev/null; do sleep 10; done
phase search_a    search --leaf 12 --sims0 4  --rounds 2 --paired --z 2
phase value_c     value  --ids-mod 6/0 --every 8 --worlds 8
phase search_b16  search --ids-mod 6/0 --leaf 12 --sims0 16 --rounds 2 --paired --z 2
log "全部阶段结束"
