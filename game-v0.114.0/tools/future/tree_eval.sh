#!/bin/bash
# 树搜索在第三幕双 Boss 题上评测一个设置（逐题配对用；题目自己的种子）。打完的题按 res_s*.jsonl 跳过，可随时停、重跑接着打。
#   OUT=/root/i2a/tree/t128_kl03 SIMS=128 LAM=0.03 NS=4 bash tools/future/tree_eval.sh
# 默认 P2 题里 id % 4 == 0 的 618 道留出题；ITEM_MOD= 置空为全部 2,472 题。EXTRA 为额外参数（如 "--decide gumbel"、"--roll 12 --decide z"）。
# KS 只起其中几个分片（如 NS=16 KS="0 1 2 3 4 5"），同一组可以分几批或分到两台机器上起（各机目录分开，打完合并 res_s*.jsonl）。
D=/root/i2a/sls2
ITEMS=${ITEMS:-/root/i2a/data/items_p2.json}; ITEM_MOD=${ITEM_MOD-4:0}; CK=${CK:-/root/i2a/data/L1000.pt}
NS=${NS:-4}; SIMS=${SIMS:-128}; LAM=${LAM:-0.03}; DECIDE=${DECIDE:-kl}
mkdir -p ${OUT:?}; cd $D
echo "$(date '+%F %T')  $(cat $D/DEPLOYED)  SIMS=$SIMS DECIDE=$DECIDE LAM=$LAM NS=$NS KS=${KS:-全部} ITEM_MOD=$ITEM_MOD EXTRA=${EXTRA:-}" >> $OUT/RUN.txt
for k in ${KS:-$(seq 0 $((NS - 1)))}; do
  .venv/bin/python -I tools/future/tree_search.py run --items $ITEMS ${ITEM_MOD:+--item-mod "$ITEM_MOD"} --ckpt $CK --out $OUT --shard $k/$NS \
    --sims $SIMS --decide $DECIDE --lam $LAM ${EXTRA:-} >> $OUT/log_s$k.txt 2>&1 < /dev/null &
done
if [ -n "${WAIT:-}" ]; then wait; fi
