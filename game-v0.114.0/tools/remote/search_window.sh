#!/usr/bin/env bash
# 本机运行：一小时搜索测试的窗口（决策记录 217）。两台机器停训，全部算力用于 tools/search_test.py 的测试，结束（含出错、被杀）时恢复训练。
#   A、B 上事先放好 /root/sls2-search/{sls2（代码，引擎链到部署目录）, model.pt, exam_g1.json, exam_hold.json}
#   nohup bash tools/remote/search_window.sh > ~/.local/share/sls2/search_test/window.log 2>&1 &
# 阶段（时刻相对停训完成）：
#   0      B 贪心打全部题（整局口径的配对基线）与留出题；两台各 12 个分片 × 4 个引擎做带标签的搜索，SEARCH_MIN 分钟后不再开新题
#   ~30    标签经本机汇到 B，soft、hard 两种目标各微调一份
#   ~36    留出题：B 测 soft、A 测 hard；之后整局：soft 在 RT0000–0383 上打（两台各一半）
#   结束   B 删 STOP 拉起训练；A 起采集、恢复评测
# 远端的 pgrep/pkill 一律用 [t]ools 这种写法，免得匹配到 ssh 自己那一层 shell 的命令行把自己杀掉。
# 两台各有一个“到点自动恢复”的保险（SAFETY_MIN 分钟后若测试的暂停标记还在，就自己恢复训练），本机脚本中途断了也不会一直停着。
set -u
. "$(dirname "$0")/hosts.sh"
SEARCH_MIN=${SEARCH_MIN:-24}; SAFETY_MIN=${SAFETY_MIN:-80}; FULL_RUNS=${FULL_RUNS:-384}
T=/root/sls2-search; PY=/root/sls2-deploy/sls2/.venv/bin/python
STAMP=$(date +%Y%m%d-%H%M); O=$T/out/$STAMP
L=$HOME/.local/share/sls2/search_test/$STAMP; mkdir -p "$L"
SA=$(host_ssh a); SB=$(host_ssh b); RB=$(host_run b); DB=$(host_deploy b)
ENVSH='export DOTNET_ROOT=$HOME/.dotnet PATH=$HOME/.dotnet:$PATH DOTNET_CLI_TELEMETRY_OPTOUT=1'
log() { echo "$(date '+%F %T') $*"; }

resume() {
  log "恢复训练"
  $SB "pkill -f '[t]ools/search_test.py|[t]ools/boss3_exam.py|[t]ools/ability_test.py' 2>/dev/null; sleep 3; cd \$HOME/$RB && rm -f STOP STOP.search && RUN_DIR=\$HOME/$RB bash $DB/tools/remote/start_train.sh" 2>&1 | tail -2
  $SA "pkill -f '[t]ools/search_test.py|[t]ools/boss3_exam.py|[t]ools/ability_test.py' 2>/dev/null; sleep 3; rm -f $T/PAUSED \$HOME/.local/share/sls2/runs/defect-eval/PAUSE; bash /root/collector_restart.sh; bash /root/a_eval_restart.sh" 2>&1 | tail -2
  log "已恢复"
}

log "停训：B 学习进程存检查点后退出，A 停采集与评测"
$SB "set -e; cd \$HOME/$RB
$REMOTE_ALIVE
$REMOTE_STOP
touch STOP.search
sleep 5; pkill -9 -f '[s]ls2-deploy/sls2/engine/Sts2Headless' || true; sleep 2; echo B 剩余训练引擎 \$(pgrep -fc '[s]ls2-deploy/sls2/engine/Sts2Headless' || true)
TAG=搜索测试前 RUN_DIR=\$HOME/$RB bash $DB/tools/remote/ckpt_backup.sh && echo 已备份 \$(tail -1 backups/backup.log)
setsid nohup bash -c 'sleep $((SAFETY_MIN * 60)); cd \$HOME/$RB; [ -f STOP.search ] && { pkill -f \"[t]ools/search_test.py|[t]ools/boss3_exam.py|[t]ools/ability_test.py\"; sleep 3; rm -f STOP STOP.search; RUN_DIR=\$HOME/$RB bash $DB/tools/remote/start_train.sh; }' > /dev/null 2>&1 < /dev/null &" || { log "B 停训失败，不做测试"; exit 1; }
trap resume EXIT
$SA "touch $T/PAUSED \$HOME/.local/share/sls2/runs/defect-eval/PAUSE
pkill -f '[t]ools/remote/remote_actors.py --config'; desc() { for c in \$(pgrep -P \$1); do echo \$c; desc \$c; done; }
for p in \$(pgrep -f '^\.venv/bin/python -I tools/remote/remote_eval\.py'); do kill -9 \$p \$(desc \$p) 2>/dev/null; done
sleep 8; pkill -9 -f '[t]ools/remote/remote_actors.py --config'; pkill -9 -f '[s]ls2-deploy/sls2/engine/Sts2Headless' ; true
setsid nohup bash -c 'sleep $((SAFETY_MIN * 60)); [ -f $T/PAUSED ] && { pkill -f \"[t]ools/search_test.py|[t]ools/boss3_exam.py|[t]ools/ability_test.py\"; sleep 3; rm -f $T/PAUSED \$HOME/.local/share/sls2/runs/defect-eval/PAUSE; bash /root/collector_restart.sh; bash /root/a_eval_restart.sh; }' > /dev/null 2>&1 < /dev/null &
free -g | sed -n 2p"
sleep 5
T0=$(date +%s); DL=$((T0 + SEARCH_MIN * 60)); echo "$T0" > "$L/t0"
log "停训完成；搜索截止 $(date -d @$DL '+%T')（不再开新题）"

shards() {   # 主机 起始分片号
  local h=$1 k0=$2 S; S=$(host_ssh "$1")
  $S "mkdir -p $O; cd $T/sls2 && $ENVSH; for k in \$(seq $k0 $((k0 + 11))); do setsid nohup $PY -I tools/search_test.py search --ckpt $T/model.pt --exam $T/exam_g1.json \
    --out $O/search_s\$k.jsonl --labels $O/labels_s\$k.pkl --shard \$k/24 --envs 4 --live 2 --device cuda --deadline $DL > $O/search_s\$k.log 2>&1 < /dev/null & done; echo $h 分片已起"
}
shards b 0; shards a 12
$SB "cd $T/sls2 && $ENVSH; $PY -I tools/boss3_exam.py run --ckpt $T/model.pt --exam $T/exam_g1.json --out $O/greedy_g1.jsonl --envs 16 --device cuda > $O/greedy_g1.log 2>&1;
  $PY -I tools/boss3_exam.py run --ckpt $T/model.pt --exam $T/exam_hold.json --out $O/hold_orig.jsonl --envs 16 --device cuda > $O/hold_orig.log 2>&1; tail -1 $O/greedy_g1.log; tail -1 $O/hold_orig.log"
log "贪心基线完成"

# 等两台的分片都结束（截止后手上的题做完）；最晚截止后 12 分钟强停
while :; do
  na=$($SA "pgrep -fc '[t]ools/search_test.py search' || true" 2>/dev/null); nb=$($SB "pgrep -fc '[t]ools/search_test.py search' || true" 2>/dev/null)
  [ "${na:-0}" = 0 ] && [ "${nb:-0}" = 0 ] && break
  if [ "$(date +%s)" -gt $((DL + 720)) ]; then log "强停剩下的分片（A $na、B $nb）"; $SA "pkill -f '[t]ools/search_test.py search'"; $SB "pkill -f '[t]ools/search_test.py search'"; sleep 5; break; fi
  sleep 30
done
log "搜索结束：A $($SA "cat $O/search_s*.jsonl | wc -l") 题，B $($SB "cat $O/search_s*.jsonl | wc -l") 题"

log "标签汇到 B，微调"
mkdir -p "$L/a"
rsync -a -e "$(host_rsync_e a)" "$(host_addr a):$O/" "$L/a/" && rsync -a -e "$(host_rsync_e b)" "$L/a/labels_s"*.pkl "$(host_addr b):$O/"
$SB "mkdir -p $O/lab && mv $O/labels_s*.pkl $O/lab/ && cd $T/sls2 && \
  ($PY -I tools/search_test.py distill --ckpt $T/model.pt --labels $O/lab --out $O/distill_soft.pt --target soft > $O/distill_soft.log 2>&1 &
   $PY -I tools/search_test.py distill --ckpt $T/model.pt --labels $O/lab --out $O/distill_hard.pt --target hard > $O/distill_hard.log 2>&1; wait); tail -2 $O/distill_soft.log; tail -2 $O/distill_hard.log"
log "微调完成；hard 传到 A"
rsync -a -e "$(host_rsync_e b)" "$(host_addr b):$O/distill_hard.pt" "$(host_addr b):$O/distill_soft.pt" "$L/" && rsync -a -e "$(host_rsync_e a)" "$L/distill_hard.pt" "$L/distill_soft.pt" "$(host_addr a):$O/"

log "留出题：B 测 soft，A 测 hard"
$SB "cd $T/sls2 && $ENVSH; $PY -I tools/boss3_exam.py run --ckpt $O/distill_soft.pt --exam $T/exam_hold.json --out $O/hold_soft.jsonl --envs 32 --device cuda > $O/hold_soft.log 2>&1; tail -1 $O/hold_soft.log" &
$SA "cd $T/sls2 && $ENVSH; $PY -I tools/boss3_exam.py run --ckpt $O/distill_hard.pt --exam $T/exam_hold.json --out $O/hold_hard.jsonl --envs 32 --device cuda > $O/hold_hard.log 2>&1; tail -1 $O/hold_hard.log" &
wait

log "整局：soft 在 RT0000–$(printf %04d $((FULL_RUNS - 1))) 上（两台各一半）"
$SB "cd $T/sls2 && $ENVSH; for k in 0 1; do $PY -I tools/ability_test.py --ckpt $O/distill_soft.pt --out $O/full_soft --asc 10-10 --runs $FULL_RUNS --shard \$k/4 --envs 24 --device cuda --seed-prefix RT > $O/full_soft_s\$k.log 2>&1 & done; wait; tail -1 $O/full_soft_s0.log" &
$SA "cd $T/sls2 && $ENVSH; for k in 2 3; do $PY -I tools/ability_test.py --ckpt $O/distill_soft.pt --out $O/full_soft --asc 10-10 --runs $FULL_RUNS --shard \$k/4 --envs 24 --device cuda --seed-prefix RT > $O/full_soft_s\$k.log 2>&1 & done; wait; tail -1 $O/full_soft_s2.log" &
wait
log "测试全部完成，用时 $(( ($(date +%s) - T0) / 60 )) 分钟"
resume
trap - EXIT
log "拉回结果到 $L"
rsync -a --exclude '*.pkl' --exclude 'lab/' -e "$(host_rsync_e b)" "$(host_addr b):$O/" "$L/b/"
rsync -a --exclude '*.pkl' -e "$(host_rsync_e a)" "$(host_addr a):$O/" "$L/a/"
log "完成"
