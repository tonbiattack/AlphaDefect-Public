#!/usr/bin/env bash
# 铁甲战士恐惧模式测试（2026-11-20，铁甲战士决策记录第 16 条）：现场随机生成的种子上纯贪心打恐惧模式（进阶固定 10），
# 截断同本线常规评测（局面重复截断、20 回合无进展判输）。引擎用测试目录里另编的一份（只多了 reset 的 "game_mode"，见 HeadlessRun.Mode），
# 训练部署 /root/ironclad-deploy 不动；测试目录路径里带 ironclad-deploy/sls2/engine，引擎看门狗（engine_watchdog.sh）照样管。
# 训练可以照常跑，只要先停常规评测腾出显卡（PARTS=eval stop.sh，测完 PARTS=eval start.sh）。
#   CKPT=/root/ironclad-dread-test/ck/m22660.pt setsid nohup bash /root/ironclad-dread-test/dread_test.sh > /root/ironclad-dread-test/run.out 2>&1 < /dev/null &
# GAME_MODE=standard 在同一批种子上打标准模式作配对对照（输出目录加 _standard）。已打完的局会跳过（ability_test 按 runs_s*.jsonl 去重）。
set -u
T=${T:-/root/ironclad-dread-test}
D=$T/ironclad-deploy/sls2
SEEDS=${SEEDS:-$T/seeds_dread300.json}
CKPT=${CKPT:?要给 CKPT}
GAME_MODE=${GAME_MODE:-dread}
O=${O:-$T/out_$(basename "$CKPT" .pt)$([ "$GAME_MODE" = dread ] || echo _$GAME_MODE)}
RUNS=${RUNS:-300}; SHARDS=${SHARDS:-4}; ENVS=${ENVS:-8}
EXTRA='{"loop_cut": true, "stall_turns": 20, "game_mode": "'$GAME_MODE'"}'
log() { echo "$(date '+%F %T') $*" | tee -a "$T/log.txt"; }
[ -f "$SEEDS" ] || { log "缺 $SEEDS"; exit 1; }
cd $D || exit 1
mkdir -p $O
log "开始 $CKPT → $O（$GAME_MODE，$RUNS 局，$SHARDS × $ENVS）"
for k in $(seq 0 $((SHARDS - 1))); do
  nice -n 10 .venv/bin/python -I tools/ability_test.py --ckpt "$CKPT" --out $O --asc 10-10 --runs $RUNS --shard $k/$SHARDS \
    --envs $ENVS --device cuda --seeds-file "$SEEDS" --character IRONCLAD --time-cap 1200 --req-extra "$EXTRA" \
    > $O/log_s$k.txt 2>&1 < /dev/null &
done
wait
log "完成：$(cat $O/runs_s*.jsonl | wc -l) 局，通关 $(cat $O/runs_s*.jsonl | grep -c '"win": true')"
