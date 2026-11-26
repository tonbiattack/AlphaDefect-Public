#!/usr/bin/env bash
# 铁甲战士千种池测试：当前权重、滑动平均权重、快照集成（2026-11-19，铁甲战士决策记录第 12 条；照 tools/remote/fool/finish/ens_test.sh）
# 在 C 上、训练与常规评测都停下以后跑：各组在同一批种子（W1 前 RUNS 个，同故障机器人、愚者的千种池）上纯贪心打进阶 10，
# 截断同本线常规评测（局面重复截断、20 回合无进展判输，eval_stall_turns 20），组与组依次跑，同种子逐局配对比较。
# 每组的成员列在 $T/arms/<组名>.txt（一行一个检查点，第一行作 --ckpt、其余作集成成员；只有一行就是单个网络），
# 成员由 ext_members.py 从完整检查点抽出。集成为每步概率平均（tools/ability_test.py 的 EnsembleNet，prob）。
# 引擎在 /root/ironclad-deploy 下，引擎看门狗（engine_watchdog.sh）照样管。
#   ARMS="V M E" setsid nohup bash /root/ens_ironclad/ens_test.sh > /root/ens_ironclad/run.out 2>&1 < /dev/null &
# 已打完的局会跳过（ability_test 按 runs_s*.jsonl 去重），加大 RUNS 重跑即在后面补种子。
set -u
T=${T:-/root/ens_ironclad}
D=/root/ironclad-deploy/sls2
SEEDS=${SEEDS:-$T/seeds_w1.json}
RUNS=${RUNS:-1000}; SHARDS=${SHARDS:-6}; ENVS=${ENVS:-16}; ARMS=${ARMS:-"V M E"}
EXTRA='{"loop_cut": true, "stall_turns": 20}'
log() { echo "$(date '+%F %T') $*" | tee -a "$T/log.txt"; }
[ -f "$SEEDS" ] || { log "缺 $SEEDS"; exit 1; }
cd $D || exit 1
for arm in $ARMS; do
  L=$T/arms/$arm.txt
  [ -f $L ] || { log "缺 $L，跳过 $arm"; continue; }
  first=$(head -n1 $L); O=$T/$arm; mkdir -p $O; tail -n +2 $L > $O/members.txt
  ENS=""; [ -s $O/members.txt ] && ENS="--ensemble-file $O/members.txt"
  log "开始 $arm（$(tr '\n' ' ' < $L)）"
  for k in $(seq 0 $((SHARDS - 1))); do
    .venv/bin/python -I tools/ability_test.py --ckpt "$first" $ENS --out $O --asc 10-10 --runs $RUNS --shard $k/$SHARDS \
      --envs $ENVS --device cuda --seeds-file "$SEEDS" --character IRONCLAD --time-cap 1200 --req-extra "$EXTRA" \
      > $O/log_s$k.txt 2>&1 < /dev/null &
  done
  wait
  log "完成 $arm：$(cat $O/runs_s*.jsonl | wc -l) 局，通关 $(cat $O/runs_s*.jsonl | grep -c '"win": true')"
done
log "全部完成"
