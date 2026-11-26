#!/usr/bin/env bash
# 猎人蒸馏学生带第三幕双 Boss 搜索的大样本评测（2026-11-11，猎人决策记录第 15 条；照储君第 29 条的做法）
# 在 B 上、训练停下以后跑：tools/sim_pimc_eval.py，参数同故障机器人第 290 条（PIMC tau 0.9、k 4、sims0 4、rounds 2、z 2、leaf 12，
# 推演与估值用学生、显卡，每个分片 2 个 sts2-sim 进程），只在第三幕两个 Boss 出题；--sync 同猎人适配冒烟（engine/Sts2SimServe/猎人适配.md）。
# sts2-sim 用本机 ~/.local/share/sls2/live/sim_engine_v23 的构建（拷到 /root/silent-sim/，校验和见记录），观测表为仓库的 sim_tables.json。
#   CKPT=/root/distill_silent/student.pt setsid nohup bash /root/pimc_silent/pimc_eval.sh > /root/pimc_silent/run.out 2>&1 < /dev/null &
# 种子：默认 W1 前 1,000 个（与集成测试同一批），SEEDS 可换别的种子表（胜率小模型的第二批用 W1 第 1,000–1,999 个）。按下标分片；已打完的种子跳过，重跑即续。
set -u
P=${P:-/root/pimc_silent}; D=/root/silent-deploy/sls2; SIM=/root/silent-sim
CKPT=${CKPT:?要给 CKPT}; OUT=${OUT:-$P/main}; SHARDS=${SHARDS:-20}; SEEDS=${SEEDS:-/root/ens_silent/seeds_w1.json}
mkdir -p $OUT
cd $D || exit 1
for k in $(seq 0 $((SHARDS - 1))); do
  .venv/bin/python -I tools/sim_pimc_eval.py run --ckpt $CKPT --real-dll $D/engine/Sts2Headless/bin/Release/net9.0/Sts2Headless.dll \
    --sim-dll $SIM/sim_engine_v23/Sts2SimServe.dll --sim-tables $SIM/sim_tables.json --character SILENT --out $OUT \
    --seeds-file $SEEDS --shard $k/$SHARDS --acts 2 --rooms boss --evaluator model --device cuda --sim-procs 2 \
    --sync --records --pause-file $P/PAUSE > $OUT/run_s$k.log 2>&1 < /dev/null &
done
wait
echo "$(date '+%F %T') 全部分片结束"
