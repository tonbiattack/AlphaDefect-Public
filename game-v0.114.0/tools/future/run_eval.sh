#!/bin/bash
# 整局评测一个设置（run_search.py：局外贪心，ROOMS 里的战斗用树搜索；SIMS=0 为整局贪心）。打完的局按 runs_s*.jsonl 跳过，可随时停、重跑接着打。
#   OUT=/root/i2a/runs/b32 SIMS=32 ROOMS=Boss NS=16 RUNS=1000 bash tools/future/run_eval.sh
# 默认 P2-future 种子池前 RUNS 个（同 /root/future-c/c_games.sh 的整局，L1000 贪心在这 3,000 局里通关 60.5%）。KS 同 tree_eval.sh。
D=/root/i2a/sls2
SEEDS=${SEEDS:-/root/i2a/data/seeds_p2.json}; CK=${CK:-/root/i2a/data/L1000.pt}; RUNS=${RUNS:-1000}
NS=${NS:-8}; SIMS=${SIMS:-32}; LAM=${LAM:-0.06}; ROOMS=${ROOMS:-Boss}
mkdir -p ${OUT:?}; cd $D
echo "$(date '+%F %T')  $(cat $D/DEPLOYED)  SIMS=$SIMS LAM=$LAM ROOMS=$ROOMS RUNS=$RUNS NS=$NS KS=${KS:-全部} CK=$CK EXTRA=${EXTRA:-}" >> $OUT/RUN.txt
for k in ${KS:-$(seq 0 $((NS - 1)))}; do
  .venv/bin/python -I tools/future/run_search.py --seeds $SEEDS --runs $RUNS --ckpt $CK --out $OUT --shard $k/$NS \
    --sims $SIMS --lam $LAM --rooms $ROOMS ${EXTRA:-} >> $OUT/log_s$k.txt 2>&1 < /dev/null &
done
if [ -n "${WAIT:-}" ]; then wait; fi
