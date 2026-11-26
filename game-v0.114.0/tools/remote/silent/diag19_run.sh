#!/usr/bin/env bash
# 猎人决策记录第 19 条：选牌能力陷阱诊断的驱动（第二版：参数用数组传，通配符原样交给 Python；每个分片查退出码与报错）
# 第 1 步（最终模型 A、B）→ 第 2b 步（最终模型 C，S 与 Y 各自的 β）→ 第 2a 步（v20320、v22320 的 A、B）；底局已有（nodes_done）
#   SH=12 ENVS=12 setsid nohup bash /root/diag19/run_all2.sh > /root/diag19/out/run_all2.out 2>&1 < /dev/null &
set -u
cd /root/diag19
PY=/root/silent-deploy/sls2/.venv/bin/python
DLL=/root/diag19/engine_bin/Sts2Headless.dll
O=out/P1
SH=${SH:-12}; ENVS=${ENVS:-12}
log() { echo "$(date '+%F %T') $*" | tee -a $O/run_all.log; }
[ -f $O/nodes_done ] || { log "缺底局（nodes_done）"; exit 1; }
RUN=(run --nodes "$O/nodes_s*.jsonl" --base "$O/base_s*.jsonl" --classes classes.json --out $O)
par() {   # $1 = 步骤名，其余为 branch_force.py 的参数；全部分片退出码为 0 且日志里没有 error 才算完成
  local step=$1; shift; local pids=() bad=0
  for k in $(seq 0 $((SH - 1))); do
    $PY -I tools/branch_force.py "$@" --shard $k/$SH --envs $ENVS --device cuda --dll $DLL > $O/log_${step}_s$k.txt 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait $p || bad=$((bad + 1)); done
  if [ $bad -gt 0 ] || grep -lE "Traceback|error:" $O/log_${step}_s*.txt >/dev/null 2>&1; then
    log "失败 $step：$bad 个分片退出码非 0；$(grep -lE 'Traceback|error:' $O/log_${step}_s*.txt | tr '\n' ' ')"; exit 1
  fi
  touch $O/done_$step; log "完成 $step：$(cat $O/log_${step}_s*.txt | grep -h '^完成' | awk '{s+=$2} END {print s}') 枝"
}
for spec in "A m24721" "B m24721"; do
  set -- $spec; step=${1}_${2}; [ -f $O/done_$step ] && continue
  V=(); [ "$1" = A ] && [ "$2" = m24721 ] && V=(--verify 20)
  log "开始 $step"; par $step "${RUN[@]}" --cont-ckpt ck/$2.pt --tag $2 --arm $1 "${V[@]}"
done
for c in S Y; do
  step=C_${c}; [ -f $O/done_$step ] && continue
  B=$($PY -I tools/branch_force.py beta --dir $O --classes classes.json --cls $c)
  log "开始 $step（β = $B）"; par $step "${RUN[@]}" --cont-ckpt ck/m24721.pt --tag m24721 --arm C --bias $B --only $c
done
for spec in "A v20320" "B v20320" "A v22320" "B v22320"; do
  set -- $spec; step=${1}_${2}; [ -f $O/done_$step ] && continue
  log "开始 $step"; par $step "${RUN[@]}" --cont-ckpt ck/$2.pt --tag $2 --arm $1
done
log "全部完成"
