#!/usr/bin/env bash
# 本机运行：启动铁甲战士训练线（2026-11-17，铁甲战士决策记录第 1 条；照 tools/remote/fool/start.sh 与恐惧线的 tools/remote/dread-defect/start.sh）。先 deploy.sh 部署并编译。
# 2026-11-20 起默认在 B（C 撤离，铁甲战士决策记录第 24 条）；HOST=c 仍按 C 的做法。
#   学习进程 + 本机采集（RUN_DIR=/root/.local/share/sls2/runs/ironclad，配置 tools/remote/ironclad/train_config.json，不开桥接）；
#   常规评测另起一个进程（remote_eval，显卡推理 4 个进程 × 6 个引擎、nice 10，种子池 P1 评测部分全部 384 个种子；前 192 个与愚者线的评测同种子），
#   B 的 cgroup 是 v2，评测按 --cores 限 8 核；C 的是 v1，限不了核，只靠 nice 让路（恐惧模式决策记录第 1 条）；每轮逐局结果全部保留（--keep-runs 100000）并留整局回放；
#   检查点：学习端每 20 版存 latest.pt（带滑动平均 ema），crontab 每 30 分钟备份到 backups/（检查点没变就跳过），
#   keep_backups 把每份备份硬链接进 keep/ 长期留存；评测每 500 版留一份权重到 ironclad-eval/keep_ens/。
# 定时任务只换本线的条目（按部署目录 ironclad-deploy 与评测权重链接 sts2rl_ironclad_eval.pt 匹配）；别的线（B 上亡灵契约师、愚者等，C 上恐惧线）的条目一行不动。
# C 的驱动是 570：triton 自带的 ptxas 比驱动新，编出的核载入不了，学习端与守护都带上虚拟环境里 nvidia-cuda-nvcc 的 ptxas 12.8（恐惧模式决策记录第 4 条）；
# B 的驱动是 580，虚拟环境里没有也不需要 ptxas，不带。
# 已在跑的部分不重复启动。
#   bash tools/remote/ironclad/start.sh             # 学习与评测
#   PARTS="train" bash tools/remote/ironclad/start.sh
set -euo pipefail
. "$(dirname "$0")/../hosts.sh"
H=${HOST:-b}
host_require "$H"
NEED_PTX=0; [ "$H" = c ] && NEED_PTX=1
D=/root/ironclad-deploy/sls2
RUN=/root/.local/share/sls2/runs/ironclad
EVAL=/root/.local/share/sls2/runs/ironclad-eval
CFG=$(dirname "$0")/train_config.json
PARTS=${PARTS:-train eval}
# 评测参数可用 EVAL_DEVICE、EVAL_WORKERS、EVAL_ENVS、EVAL_CORES、EVAL_SEEDS（默认 0，即 P1 评测部分全部 384 个；192 为只用前 192 个）覆盖
# EVAL_EXTRA 追加给 remote_eval.py 的参数：开训时评原始权重（同愚者从零那段）；以后换免调度优化器时用 EVAL_EXTRA=--ema 评平均点 x（愚者第 16 条）
SSH=$(host_ssh "$H")
$SSH "mkdir -p $RUN $EVAL"
rsync -a -e "$(host_rsync_e "$H")" "$CFG" "$(host_addr "$H"):$RUN/train_config.json"
rsync -a -e "$(host_rsync_e "$H")" "$CFG" "$(host_addr "$H"):/root/ironclad_train_config.json"
if [[ " $PARTS " == *" train "* ]]; then
  echo "== $H：铁甲战士学习端"
  $SSH "set -e; cd $RUN
    [ -f STOP ] && { echo '有 STOP 标记（rm $RUN/STOP 后再启动）'; exit 1; }
    PTX=\$(ls $D/.venv/lib/python3*/site-packages/nvidia/cuda_nvcc/bin/ptxas 2>/dev/null | head -1)
    [ -n \"\$PTX\" ] || [ $NEED_PTX = 0 ] || { echo 虚拟环境里没有 ptxas; exit 1; }
    TP=; [ -n \"\$PTX\" ] && TP=\"TRITON_PTXAS_PATH=\$PTX \"
    ( crontab -l 2>/dev/null | grep -v 'ironclad-deploy/sls2/tools/remote/' | grep -v 'sts2rl_ironclad_eval.pt' || true ;
      echo \"*/2 * * * * \${TP}RUN_DIR=$RUN bash $D/tools/remote/train_watchdog.sh\" ;
      echo '*/30 * * * * RUN_DIR=$RUN bash $D/tools/remote/ckpt_backup.sh' ;
      echo '15,45 * * * * RUN_DIR=$RUN KEEP_DIR=$RUN/keep FROM=0 bash $D/tools/remote/regent/keep_backups_b.sh' ;
      echo '*/10 * * * * EVAL_DIR=$EVAL bash $D/tools/remote/regent/keep_snapshots.sh' ;
      echo '* * * * * bash $D/tools/remote/ironclad/engine_watchdog.sh' ;
      echo '* * * * * P=\$(cat $RUN/learner.pid 2>/dev/null) && [ -f /dev/shm/sts2rl_\$P.pt ] && ln -sfn /dev/shm/sts2rl_\$P.pt /dev/shm/sts2rl_ironclad_eval.pt' ) | crontab -
    echo 'crontab（本线）：'; crontab -l | grep -e ironclad
    cp $D/DEPLOYED $RUN/DEPLOYED.\$(date +%Y%m%d-%H%M)
    env \${TP}RUN_DIR=$RUN bash $D/tools/remote/start_train.sh"
fi
if [[ " $PARTS " == *" eval "* ]]; then
  echo "== $H：铁甲战士常规评测"
  $SSH "set -e; cd $D
    P=\$(cat $RUN/learner.pid 2>/dev/null || true); [ -n \"\$P\" ] && [ -f /dev/shm/sts2rl_\$P.pt ] && ln -sfn /dev/shm/sts2rl_\$P.pt /dev/shm/sts2rl_ironclad_eval.pt || true
    if pgrep -a python | grep -q 'remote_eval.py --config /root/ironclad_train_config.json'; then echo 评测已在跑
    else
      setsid nohup .venv/bin/python -I $D/tools/remote/remote_eval.py --config /root/ironclad_train_config.json --run-dir $EVAL \
        --weights /dev/shm/sts2rl_ironclad_eval.pt --seeds-file $D/tools/remote/seed_pool.json --seeds-limit ${EVAL_SEEDS:-0} --workers ${EVAL_WORKERS:-4} --envs ${EVAL_ENVS:-6} --device ${EVAL_DEVICE:-cuda} --cores ${EVAL_CORES:-8} --nice 10 \
        --keep-runs 100000 ${EVAL_EXTRA:-} >> $EVAL/remote_eval.out 2>&1 < /dev/null &
      echo 评测已起：种子池 P1 评测部分（前 ${EVAL_SEEDS:-0} 个，0 为全部 384 个），${EVAL_DEVICE:-cuda} 推理 ${EVAL_WORKERS:-4} × ${EVAL_ENVS:-6}，逐局结果与回放全部保留
    fi"
fi
