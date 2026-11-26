#!/usr/bin/env bash
# 胜率模型数据：A 空闲时用最新复测的模型在新随机种子上打贪心整局（决策记录 235；2026-11-01 用户：
# “只有当A空闲的时候才这么做，然后不要干扰任何其他的训练和测试进程，作为低优先级存在。另外的话，你的产生的记录需要迭代，过旧的版本要自动删除淘汰。”）
# - 模型：/root/retest/ 下版本号最大的 v<版本>/model.pt（定期复测取的主线备份）；/root/wpdata/CKPT 里写一个 .pt 路径可以改用指定模型。
# - 空闲：评测没有 PAUSE、没有别的测试或训练任务（复测、能力测试、Boss 卷、搜索测试、配对检验、胜率模型训练等）、
#   可用内存 ≥ 6 GB、CPU 空闲 ≥ 10% 才开局；常规评测与远程采集是常态，不算忙。
# - 低优先级：SCHED_IDLE（只用没人要的 CPU）+ ionice 空闲档 + CPU 推理（不碰显卡）；一个进程、4 个引擎（进程树约 1.4 GB）。
#   开着的时候一旦出现别的任务、PAUSE，或可用内存 < 3 GB，立刻强杀（暂停的进程照样占内存），冷却 10 分钟再看；ability_test 可续打，打完的局不重打。
# - 迭代：每个模型版本一个目录 /root/wpdata/v<版本>/，只留当前版本和它之前最新一个有局的版本，更旧的整个删掉；每个版本打满 TARGET 局就停，等下一个版本。
#   种子前缀按版本取（W + 两个字母，相邻版本不重复）。
#   setsid nohup bash /root/wpdata/wp_collect.sh > /dev/null 2>&1 < /dev/null &
#   停：kill $(cat /root/wpdata/collect.pid)，再强杀 ability_test --ckpt /root/wpdata/ 的进程树（若在打）
set -u
W=/root/wpdata; DEP=/root/sls2-deploy/sls2; EVD=/root/.local/share/sls2/runs/defect-eval
TARGET=${TARGET:-12000}; ENVS=${ENVS:-4}; KEEP=2
LOG=$W/collect.log
log() { echo "$(date '+%m-%d %T') $*" >> $LOG; [ $(wc -l < $LOG) -gt 3000 ] && tail -n 2000 $LOG > $LOG.tmp && mv $LOG.tmp $LOG; }
PIDF=$W/collect.pid; [ -s $PIDF ] && kill -0 $(cat $PIDF) 2>/dev/null && exit 0; echo $$ > $PIDF   # 只跑一份（不用 flock：子进程会继承锁）
BUSY='tools/(ability_test|boss3_exam|search_test|meta_probe|pimc_rust_eval|engine_bench|search_diag|xchar)|code[0-9]*/(train|cmp|recal|build_cache)|retest\.sh|sprt\.sh|search_window'
mine() { pgrep -f "^\.venv/bin/python -I tools/ability_test\.py --ckpt $W/"; }
busy() {   # 有别的任务时输出原因
  [ -e $EVD/PAUSE ] && { echo "评测暂停中（有测试）"; return; }
  local o; o=$(pgrep -af "$BUSY" | grep -v -- "--ckpt $W/" | grep -v "pgrep" | head -1 | cut -c1-80)
  [ -n "$o" ] && echo "别的任务：$o"
}
mem() { awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo; }
idle_pct() { vmstat 30 2 | tail -1 | awk '{print $15}'; }
stop_mine() { for p in $(mine); do pkill -9 -P $p; kill -9 $p; done 2>/dev/null; }
ckpt() {
  [ -s $W/CKPT ] && { cat $W/CKPT; return; }
  local v; v=$(ls /root/retest | grep -E '^v[0-9]+$' | sort -V | tail -1); [ -n "$v" ] && echo /root/retest/$v/model.pt
}
prefix() { local L=ABCDEFGHIJKLMNOPQRSTUVWXYZ n=$(( ($1 / 100) % 676 )); echo "W${L:$((n / 26)):1}${L:$((n % 26)):1}"; }
prune() {  # 只留有局的最新 KEEP 个版本目录（当前版本不论有没有局都留），更旧的删掉；没打过局的旧目录直接删
  local cur=$1 v
  ls -d $W/v[0-9]* 2>/dev/null | sed 's|.*/v||' | sort -n | while read v; do
    [ "$v" = "$cur" ] && continue
    ls $W/v$v/runs_s*.jsonl > /dev/null 2>&1 || { rm -rf "$W/v$v"; continue; }
    echo $v
  done | head -n -$((KEEP - 1)) | while read v; do
    [ -n "$v" ] && [ -d "$W/v$v" ] && rm -rf "$W/v$v" && log "删掉旧版本 v$v 的局"
  done
}
log "启动（每版 $TARGET 局，$ENVS 个引擎）"
while true; do
  src=$(ckpt); [ -f "$src" ] || { sleep 600; continue; }
  v=$(basename $(dirname $src)); v=${v#v}; [[ $v =~ ^[0-9]+$ ]] || v=$(stat -c %Y $src)
  O=$W/v$v; mkdir -p $O; [ -f $O/model.pt ] || cp $src $O/model.pt
  prune $v
  done_n=$(cat $O/runs_s*.jsonl 2>/dev/null | wc -l)
  if [ $done_n -ge $TARGET ]; then sleep 1800; continue; fi
  why=$(busy); av=$(mem)
  if [ -n "$why" ] || [ $av -lt 6144 ]; then sleep 300; continue; fi
  id=$(idle_pct); [ "${id:-0}" -lt 10 ] && { sleep 300; continue; }
  log "开打 v$v（已有 $done_n 局，可用内存 ${av} MB，CPU 空闲 ${id}%）"
  ( cd $DEP && OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 exec chrt -i 0 ionice -c3 .venv/bin/python -I tools/ability_test.py --ckpt $O/model.pt --out $O \
      --asc 10-10 --runs $TARGET --shard 0/1 --envs $ENVS --device cpu --seed-prefix $(prefix $v) --temp 0 >> $O/log.txt 2>&1 ) &
  sleep 20
  while mine > /dev/null; do
    why=$(busy); av=$(mem)
    if [ -n "$why" ] || [ $av -lt 3072 ]; then
      stop_mine; log "让位：${why:-可用内存只剩 ${av} MB}；已有 $(cat $O/runs_s*.jsonl 2>/dev/null | wc -l) 局"; sleep 600; break
    fi
    [ "$(ckpt)" != "$src" ] && { stop_mine; log "有新版本模型，换过去"; break; }
    sleep 5
  done
  wait 2>/dev/null
done
