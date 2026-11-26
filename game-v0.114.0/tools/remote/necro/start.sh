#!/usr/bin/env bash
# 本机运行：在 B 上启动亡灵契约师（NECROBINDER）训练线（2026-11-17，照 tools/remote/fool/start.sh 复制，只改目录、配置文件名与 crontab 条目的匹配）。先 deploy.sh 部署并编译。
#   学习进程 + 本机采集（RUN_DIR=/root/.local/share/sls2/runs/necro，配置 tools/remote/necro/train_config.json，不开桥接）；
#   常规评测另起一个进程（remote_eval，显卡推理、4 个进程 × 8 个引擎、限 8 核、nice 10，种子池 P1 评测部分的前 192 个种子），
#   每轮逐局结果全部保留（--keep-runs 100000）并留整局回放；
#   检查点：学习端每 20 版存 latest.pt（带滑动平均 ema），crontab 每 30 分钟备份到 backups/（检查点没变就跳过），
#   keep_backups 把每份备份硬链接进 keep/ 长期留存（不随 96 份轮换删掉）；评测每 500 版留一份权重到 necro-eval/keep_ens/。
# crontab 只换本线的条目（匹配 necro-deploy/sls2/tools/remote/ 与 sts2rl_necro_eval.pt），愚者等别的线的条目不动。已在跑的部分不重复启动。
#   bash tools/remote/necro/start.sh             # 学习与评测
#   PARTS="train" bash tools/remote/necro/start.sh
set -euo pipefail
. "$(dirname "$0")/../hosts.sh"
H=${HOST:-b}
D=/root/necro-deploy/sls2
RUN=/root/.local/share/sls2/runs/necro
EVAL=/root/.local/share/sls2/runs/necro-eval
CFG=$(dirname "$0")/train_config.json
PARTS=${PARTS:-train eval}
# 评测参数（默认显卡推理 4 × 8、cgroup 限 8 核、种子池 P1 评测部分的前 192 个种子）可用 EVAL_DEVICE、EVAL_WORKERS、EVAL_ENVS、EVAL_CORES、EVAL_SEEDS 覆盖
# EVAL_EXTRA 追加给 remote_eval.py 的参数（以后接免调度优化器时用 EVAL_EXTRA=--ema 评平均点 x，同愚者第 16 条）
SSH=$(host_ssh "$H")
$SSH "mkdir -p $RUN $EVAL"
rsync -a -e "$(host_rsync_e "$H")" "$CFG" "$(host_addr "$H"):$RUN/train_config.json"
rsync -a -e "$(host_rsync_e "$H")" "$CFG" "$(host_addr "$H"):/root/necro_train_config.json"
if [[ " $PARTS " == *" train "* ]]; then
  echo "== $H：亡灵契约师学习端"
  $SSH "set -e; cd $RUN
    [ -f STOP ] && { echo '有 STOP 标记（rm $RUN/STOP 后再启动）'; exit 1; }
    ( crontab -l 2>/dev/null | grep -v 'necro-deploy/sls2/tools/remote/' | grep -v 'sts2rl_necro_eval.pt' || true ;
      echo '*/2 * * * * RUN_DIR=$RUN bash $D/tools/remote/train_watchdog.sh' ;
      echo '*/30 * * * * RUN_DIR=$RUN bash $D/tools/remote/ckpt_backup.sh' ;
      echo '15,45 * * * * RUN_DIR=$RUN KEEP_DIR=$RUN/keep FROM=0 bash $D/tools/remote/regent/keep_backups_b.sh' ;
      echo '*/10 * * * * EVAL_DIR=$EVAL bash $D/tools/remote/regent/keep_snapshots.sh' ;
      echo '* * * * * bash $D/tools/remote/necro/engine_watchdog.sh' ;
      echo '* * * * * P=\$(cat $RUN/learner.pid 2>/dev/null) && [ -f /dev/shm/sts2rl_\$P.pt ] && ln -sfn /dev/shm/sts2rl_\$P.pt /dev/shm/sts2rl_necro_eval.pt' ) | crontab -
    echo 'crontab：'; crontab -l | grep necro
    cp $D/DEPLOYED $RUN/DEPLOYED.\$(date +%Y%m%d-%H%M)
    RUN_DIR=$RUN bash $D/tools/remote/start_train.sh"
fi
if [[ " $PARTS " == *" eval "* ]]; then
  echo "== $H：亡灵契约师常规评测"
  $SSH "set -e; cd $D
    P=\$(cat $RUN/learner.pid 2>/dev/null || true); [ -n \"\$P\" ] && [ -f /dev/shm/sts2rl_\$P.pt ] && ln -sfn /dev/shm/sts2rl_\$P.pt /dev/shm/sts2rl_necro_eval.pt || true
    if pgrep -a python | grep -q 'remote_eval.py --config /root/necro_train_config.json'; then echo 评测已在跑
    else
      setsid nohup .venv/bin/python -I $D/tools/remote/remote_eval.py --config /root/necro_train_config.json --run-dir $EVAL \
        --weights /dev/shm/sts2rl_necro_eval.pt --seeds-file $D/tools/remote/seed_pool.json --seeds-limit ${EVAL_SEEDS:-192} --workers ${EVAL_WORKERS:-4} --envs ${EVAL_ENVS:-8} --device ${EVAL_DEVICE:-cuda} --cores ${EVAL_CORES:-8} --nice 10 \
        --keep-runs 100000 ${EVAL_EXTRA:-} >> $EVAL/remote_eval.out 2>&1 < /dev/null &
      echo 评测已起：种子池 P1 评测部分的前 ${EVAL_SEEDS:-192} 个种子，${EVAL_DEVICE:-cuda} 推理 ${EVAL_WORKERS:-4} × ${EVAL_ENVS:-8}、限 ${EVAL_CORES:-8} 核，逐局结果与回放全部保留
    fi"
fi
