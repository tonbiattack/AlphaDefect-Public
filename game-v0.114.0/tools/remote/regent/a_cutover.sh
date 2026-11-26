#!/usr/bin/env bash
# A 到期前的第二次迁移与切换（2026-11-09，储君决策记录第 26 条；用户：“我续费了B机器，A即将在6小时后过期，做好准备，把所有工作和数据迁移到B机器，
# 先做一次迁移，维持A上面的工作，五个小时后再做一次迁移准备最终A撤离。”）。A 约 11-09 21:50（B 机时间，CST）到期。
# 在 B 上跑，只走机房内网（ssh sls2-a，B 的密钥在 A 上只认 B 的内网地址），不经本机与中转：
#   1. 停 A 上储君的远程采集与常规评测（同 stop.sh 的 A 部分），撤掉 A 的 keep_snapshots 定时；
#   2. 增量同步 A 的 /root 与 /tmp 到 B 的 /root/from_a/（与第一次迁移同样的排除项：.cache、.dotnet、游戏本体、.venv、__pycache__）；
#   3. A 的储君评测目录拷成 B 的 runs/regent-eval，在 B 上起常规评测（规模见 EVAL_*），B 的 crontab 加 keep_snapshots 与评测权重链接；
#      B 上的评测把每轮的逐局结果全部留在 eval_runs/（--keep-runs 100000，每轮约 2.3 MB；原来只留最近 20 轮，同行审查要能逐局核对，2026-11-09）；
#   4. 写 /root/from_a/cutover.done。已有这个标记时第 1、3 步跳过，只做第 2 步的增量同步。
# 会话在场时手动先跑；B 的 crontab 另有两条一次性兜底（11-09 20:50 与 21:20，B 机时间），会话不在时自动执行。
#   bash /root/regent-deploy/sls2/tools/remote/regent/a_cutover.sh
#   STEP=sync bash /root/regent-deploy/sls2/tools/remote/regent/a_cutover.sh     # 只做增量同步，可反复跑
set -u
D=/root/regent-deploy/sls2
RUN=/root/.local/share/sls2/runs/regent
EVAL=/root/.local/share/sls2/runs/regent-eval
F=/root/from_a
EVAL_DEVICE=${EVAL_DEVICE:-cpu}; EVAL_WORKERS=${EVAL_WORKERS:-4}; EVAL_ENVS=${EVAL_ENVS:-8}; EVAL_CORES=${EVAL_CORES:-8}
mkdir -p $F
log() { echo "$(date '+%F %T') $*" | tee -a $F/cutover.log; }
exec 9> $F/.cutover.lock
flock -n 9 || { log "另一次切换还在跑，退出"; exit 0; }
A="ssh -o BatchMode=yes -o ConnectTimeout=20 sls2-a"
SYNC_OPTS=(-aH --numeric-ids --partial --bwlimit=80m --rsync-path="nice -n 10 ionice -c3 rsync"
           --exclude=/.cache/ --exclude=/.dotnet/ --exclude=/.local/share/sls2/game/ --exclude=.venv/ --exclude=__pycache__/)
sync_a() {
  log "增量同步 A 的 /root 开始"
  nice -n 10 ionice -c3 rsync "${SYNC_OPTS[@]}" --stats sls2-a:/root/ $F/root/ > $F/sync_last.log 2>&1
  local rc=$?
  log "增量同步 A 的 /root 结束（rsync 退出码 $rc：$(grep -E 'Number of regular files transferred|Total transferred file size' $F/sync_last.log | tr '\n' ' ')）"
  nice -n 10 ionice -c3 rsync -a --partial --bwlimit=80m --rsync-path="nice -n 10 ionice -c3 rsync" sls2-a:/tmp/ $F/tmp/ > /dev/null 2>&1
  log "增量同步 A 的 /tmp 结束（退出码 $?）"
  $A "crontab -l" > $F/a_crontab.txt 2>/dev/null
  return $rc
}
if [ "${STEP:-}" = sync ] || [ -f $F/cutover.done ]; then
  [ -f $F/cutover.done ] && log "已切换过（$(cat $F/cutover.done)），只做增量同步"
  sync_a; exit $?
fi
log "== 开始切换：停 A 的储君工作 → 增量同步 → B 接管评测"
# 1. 停 A 上储君的远程采集与评测（只按储君的配置文件名匹配，[r] 写法免得匹配到这条命令自己）
$A "pkill -TERM -f '[r]emote_actors.py --config /root/regent_train_config.json' || true
    pkill -TERM -f '[r]emote_eval.py --config /root/regent_train_config.json' || true
    for i in \$(seq 30); do pgrep -f '[r]emote_(actors|eval).py --config /root/regent_train_config.json' >/dev/null || break; sleep 2; done
    desc() { for c in \$(pgrep -P \"\$1\"); do echo \"\$c\"; desc \"\$c\"; done; }
    for p in \$(pgrep -f '[r]emote_(actors|eval).py --config /root/regent_train_config.json'); do kill -9 \$p \$(desc \$p) 2>/dev/null; done
    crontab -l 2>/dev/null | grep -v 'regent/keep_snapshots.sh' | crontab -
    sleep 2; pgrep -fa '[r]emote_(actors|eval).py --config /root/regent_train_config.json' && echo 还有进程没退出 || echo 远程采集与评测已停
    echo A 的 crontab：; crontab -l" 2>&1 | tee -a $F/cutover.log
# 2. 增量同步（A 的工作停了，这一遍之后 A 上基本不再有新数据）
sync_a
# 3. B 接管评测：评测目录从迁来的副本拷一份（不用硬链接：评测会原地追加 eval.jsonl），配置副本与 A 上同名，stop.sh/start.sh 按名字认进程
if [ -d $EVAL ] && [ -s $EVAL/eval.jsonl ]; then
  log "B 上已有 $EVAL，只补齐迁来的新文件"
  rsync -a --ignore-existing $F/root/.local/share/sls2/runs/regent-eval/ $EVAL/
else
  rsync -a $F/root/.local/share/sls2/runs/regent-eval/ $EVAL/
fi
rm -f $EVAL/PAUSE
# 2026-11-09 用户：储君“之后不再恢复训练”，B 改练静默猎手：评测目录照样拷过来留档，但不在 B 上起储君评测、不加它的定时（REGENT_EVAL=1 时照原计划起）
if [ "${REGENT_EVAL:-0}" = 1 ]; then
cp $RUN/train_config.json /root/regent_train_config.json
P=$(cat $RUN/learner.pid 2>/dev/null); [ -n "$P" ] && [ -f /dev/shm/sts2rl_$P.pt ] && ln -sfn /dev/shm/sts2rl_$P.pt /dev/shm/sts2rl_regent_eval.pt
( crontab -l 2>/dev/null | grep -v 'regent/keep_snapshots.sh' | grep -v 'sts2rl_regent_eval.pt' ;
  echo '*/10 * * * * bash /root/regent-deploy/sls2/tools/remote/regent/keep_snapshots.sh' ;
  echo '* * * * * P=$(cat /root/.local/share/sls2/runs/regent/learner.pid 2>/dev/null) && [ -f /dev/shm/sts2rl_$P.pt ] && ln -sfn /dev/shm/sts2rl_$P.pt /dev/shm/sts2rl_regent_eval.pt' ) | crontab -
cd $D || exit 1
if pgrep -a python | grep -q 'remote_eval.py --config /root/regent_train_config.json'; then log "B 上评测已在跑"
else
  setsid nohup .venv/bin/python -I $D/tools/remote/remote_eval.py --config /root/regent_train_config.json --run-dir $EVAL \
    --weights /dev/shm/sts2rl_regent_eval.pt --seeds-file $D/tools/remote/seed_pool.json --workers $EVAL_WORKERS --envs $EVAL_ENVS \
    --device $EVAL_DEVICE --cores $EVAL_CORES --nice 10 --keep-runs 100000 --no-replay-all >> $EVAL/remote_eval.out 2>&1 < /dev/null &
  log "B 上评测已起：$EVAL_WORKERS 个进程 × $EVAL_ENVS 个引擎，$EVAL_DEVICE"
fi
else
  log "储君不再恢复训练：评测目录已拷到 $EVAL 留档，不起储君评测"
fi
date '+%F %T' > $F/cutover.done
log "== 切换完成"
