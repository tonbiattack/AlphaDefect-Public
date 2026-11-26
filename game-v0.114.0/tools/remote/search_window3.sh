#!/usr/bin/env bash
# 搜索测试的收尾段（决策记录 217）：等两台的搜索都做完（最晚到 HARD 时刻强停），标签经本机汇到 B，soft、hard 各微调一份；
# 留出题 B 测 soft、A 测 hard；soft 在 RT0000–0383 上打整局（两台各一半）；之后恢复训练，结果（含标签）拉回本机。出错或被杀也恢复训练。
#   nohup bash tools/remote/search_window3.sh <结果目录名> <强停时刻 Unix 秒> > 日志 2>&1 &
set -u
. "$(dirname "$0")/hosts.sh"
STAMP=$1; HARD=$2; FULL_RUNS=${FULL_RUNS:-384}
T=/root/sls2-search; O=$T/out/$STAMP; PY=/root/sls2-deploy/sls2/.venv/bin/python
L=$HOME/.local/share/sls2/search_test/$STAMP; mkdir -p "$L/a" "$L/b"
SA=$(host_ssh a); SB=$(host_ssh b); RB=$(host_run b); DB=$(host_deploy b)
ENVSH='export DOTNET_ROOT=$HOME/.dotnet PATH=$HOME/.dotnet:$PATH DOTNET_CLI_TELEMETRY_OPTOUT=1'
log() { echo "$(date '+%F %T') $*"; }
resume() {
  log "恢复训练"
  $SB "pkill -f '[t]ools/search_test.py|[t]ools/boss3_exam.py|[t]ools/ability_test.py|[t]ools/meta_probe.py' 2>/dev/null; sleep 3; cd \$HOME/$RB && rm -f STOP STOP.search && RUN_DIR=\$HOME/$RB bash $DB/tools/remote/start_train.sh" 2>&1 | tail -2
  $SA "pkill -f '[t]ools/search_test.py|[t]ools/boss3_exam.py|[t]ools/ability_test.py|[t]ools/meta_probe.py' 2>/dev/null; sleep 3; rm -f $T/PAUSED \$HOME/.local/share/sls2/runs/defect-eval/PAUSE; bash /root/collector_restart.sh; bash /root/a_eval_restart.sh" 2>&1 | tail -2
  log "已恢复"
}
trap resume EXIT

while :; do
  na=$($SA "pgrep -fc '[t]ools/search_test.py search|[t]ools/meta_probe.py --ckpt' || true" 2>/dev/null); nb=$($SB "pgrep -fc '[t]ools/search_test.py search' || true" 2>/dev/null)
  [ "${na:-0}" = 0 ] && [ "${nb:-0}" = 0 ] && break
  if [ "$(date +%s)" -gt "$HARD" ]; then log "强停剩下的进程（A $na、B $nb）"; $SA "pkill -f '[t]ools/search_test.py search|[t]ools/meta_probe.py --ckpt'"; $SB "pkill -f '[t]ools/search_test.py search'"; sleep 5; break; fi
  sleep 20
done
log "搜索结束：A $($SA "cat $O/search_s*.jsonl $O/search2_s*.jsonl 2>/dev/null | wc -l") 题，B $($SB "cat $O/search_s*.jsonl $O/search2_s*.jsonl 2>/dev/null | wc -l") 题，探针 $($SA "cat $O/meta_s*.jsonl 2>/dev/null | wc -l") 个决策"

log "标签汇到 B，微调"
rsync -a -e "$(host_rsync_e a)" "$(host_addr a):$O/" "$L/a/"
rsync -a -e "$(host_rsync_e b)" "$L/a/"labels*_s*.pkl "$(host_addr b):$O/lab_a/" 2>/dev/null || { $SB "mkdir -p $O/lab_a"; rsync -a -e "$(host_rsync_e b)" "$L/a/"labels*_s*.pkl "$(host_addr b):$O/lab_a/"; }
$SB "mkdir -p $O/lab && cp $O/labels*_s*.pkl $O/lab/ && for f in $O/lab_a/*.pkl; do cp \$f $O/lab/a_\$(basename \$f); done && ls $O/lab | wc -l && cd $T/sls2 && \
  ($PY -I tools/search_test.py distill --ckpt $T/model.pt --labels $O/lab --out $O/distill_soft.pt --target soft > $O/distill_soft.log 2>&1 &
   $PY -I tools/search_test.py distill --ckpt $T/model.pt --labels $O/lab --out $O/distill_hard.pt --target hard > $O/distill_hard.log 2>&1; wait); grep -h '^标签\|^微调后' $O/distill_soft.log $O/distill_hard.log | cut -c1-400"
log "微调完成"
rsync -a -e "$(host_rsync_e b)" "$(host_addr b):$O/distill_hard.pt" "$(host_addr b):$O/distill_soft.pt" "$L/" && rsync -a -e "$(host_rsync_e a)" "$L/distill_hard.pt" "$L/distill_soft.pt" "$(host_addr a):$O/" || { log "检查点传输失败，跳过蒸馏评测"; exit 1; }

log "留出题：B 测 soft，A 测 hard"
$SB "cd $T/sls2 && $ENVSH; $PY -I tools/boss3_exam.py run --ckpt $O/distill_soft.pt --exam $T/exam_hold.json --out $O/hold_soft.jsonl --envs 32 --device cuda > $O/hold_soft.log 2>&1; tail -1 $O/hold_soft.log" &
$SA "cd $T/sls2 && $ENVSH; $PY -I tools/boss3_exam.py run --ckpt $O/distill_hard.pt --exam $T/exam_hold.json --out $O/hold_hard.jsonl --envs 32 --device cuda > $O/hold_hard.log 2>&1; tail -1 $O/hold_hard.log" &
wait

log "整局：soft 在 RT0000–$(printf %04d $((FULL_RUNS - 1))) 上（两台各一半）"
$SB "cd $T/sls2 && $ENVSH; for k in 0 1; do $PY -I tools/ability_test.py --ckpt $O/distill_soft.pt --out $O/full_soft --asc 10-10 --runs $FULL_RUNS --shard \$k/4 --envs 24 --device cuda --seed-prefix RT > $O/full_soft_s\$k.log 2>&1 & done; wait; tail -1 $O/full_soft_s0.log" &
$SA "cd $T/sls2 && $ENVSH; for k in 2 3; do $PY -I tools/ability_test.py --ckpt $O/distill_soft.pt --out $O/full_soft --asc 10-10 --runs $FULL_RUNS --shard \$k/4 --envs 24 --device cuda --seed-prefix RT > $O/full_soft_s\$k.log 2>&1 & done; wait; tail -1 $O/full_soft_s2.log" &
wait
log "测试全部完成"
resume
trap - EXIT
log "拉回结果（含标签）到 $L"
rsync -a -e "$(host_rsync_e b)" "$(host_addr b):$O/" "$L/b/"
rsync -a -e "$(host_rsync_e a)" "$(host_addr a):$O/" "$L/a/"
log "完成"
