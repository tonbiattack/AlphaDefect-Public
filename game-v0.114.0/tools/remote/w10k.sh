#!/usr/bin/env bash
# 本机运行：SL 快照池万种级联测试（2026-11-02，决策记录 251；选题与记账见 tools/w10k.py）。停训，A、B 全部算力用于测试，结束（含出错、被杀）时恢复训练。
#   nohup bash tools/remote/w10k.sh > ~/.local/share/sls2/ability_test/w10k/run.log 2>&1 &
# 每个快照：还没通关的种子多于 BIG 个时两台一起打（A 8 个分片、B 12 个分片，每片 20 个引擎），否则只用 A。
# 只停训练与 A 的评测、采集，本机的直播服务不动；本机的首通探测在测试期间关掉（PROBE_OFF），结束时恢复。
# 两台各有“到点自动恢复”的保险（SAFETY_MIN 分钟后若测试标记还在，就自己恢复），本机脚本中途断了也不会一直停着。
# 远端的 pgrep/pkill 一律用 [t]ools 这种写法，免得匹配到 ssh 自己那一层 shell 的命令行。
set -u
cd "$(dirname "$0")/../.." && . tools/remote/hosts.sh
SAFETY_MIN=${SAFETY_MIN:-300}; BIG=${BIG:-1500}; A_SH=${A_SH:-8}; B_SH=${B_SH:-12}; ENVS=${ENVS:-20}
W=$HOME/.local/share/sls2/ability_test/w10k; mkdir -p "$W"
ST=$HOME/.local/share/sls2/ability_test/retest
SA=$(host_ssh a); SB=$(host_ssh b); EA=$(host_rsync_e a); EB=$(host_rsync_e b); AA=$(host_addr a); AB=$(host_addr b)
DA=$(host_deploy a); DB=$(host_deploy b); RB=$(host_run b)
R=/root/w10k
ENVSH='export DOTNET_ROOT=$HOME/.dotnet PATH=$HOME/.dotnet:$PATH DOTNET_CLI_TELEMETRY_OPTOUT=1'
log() { echo "$(date '+%F %T') $*"; }

[ -f "$W/seeds.json" ] || python3 tools/seed_pool.py --name W1 --n 10000 --eval 0 --out "$W/seeds.json" || { log "生成种子失败"; exit 1; }
python3 tools/w10k.py order > "$W/order.txt"
log "种子 $(python3 -c "import json;print(len(json.load(open('$W/seeds.json'))['seeds']))") 个；快照 $(wc -l < "$W/order.txt") 个，从第 $(head -1 "$W/order.txt" | cut -d' ' -f1) 版往前"

resume() {
  log "恢复训练"
  $SB "pkill -f '[t]ools/ability_test.py --ckpt $R/' 2>/dev/null; sleep 3; cd \$HOME/$RB && rm -f STOP STOP.w10k && RUN_DIR=\$HOME/$RB bash $DB/tools/remote/start_train.sh" 2>&1 | tail -2
  $SA "pkill -f '[t]ools/ability_test.py --ckpt $R/' 2>/dev/null; sleep 3; rm -f $R/PAUSED \$HOME/.local/share/sls2/runs/defect-eval/PAUSE; bash /root/collector_restart.sh; bash /root/a_eval_restart.sh" 2>&1 | tail -2
  rm -f "$ST/PROBE_OFF"
  log "已恢复"
}

log "停训：B 学习进程存检查点后退出，A 停采集与评测，本机停首通探测"
touch "$ST/PROBE_OFF"
$SB "set -e; cd \$HOME/$RB
$REMOTE_ALIVE
$REMOTE_STOP
touch STOP.w10k
sleep 5; pkill -9 -f '[s]ls2-deploy/sls2/engine/Sts2Headless' || true; sleep 2; echo B 剩余训练引擎 \$(pgrep -fc '[s]ls2-deploy/sls2/engine/Sts2Headless' || true)
TAG=万种测试前 RUN_DIR=\$HOME/$RB bash $DB/tools/remote/ckpt_backup.sh && echo 已备份 \$(tail -1 backups/backup.log)
mkdir -p $R/m
setsid nohup bash -c 'sleep $((SAFETY_MIN * 60)); cd \$HOME/$RB; [ -f STOP.w10k ] && { pkill -f \"[t]ools/ability_test.py --ckpt $R/\"; sleep 3; rm -f STOP STOP.w10k; RUN_DIR=\$HOME/$RB bash $DB/tools/remote/start_train.sh; }' > /dev/null 2>&1 < /dev/null &" \
  || { log "B 停训失败，不做测试"; rm -f "$ST/PROBE_OFF"; exit 1; }
trap resume EXIT
$SA "mkdir -p $R/m; touch $R/PAUSED \$HOME/.local/share/sls2/runs/defect-eval/PAUSE
pkill -f '[t]ools/remote/remote_actors.py --config'; desc() { for c in \$(pgrep -P \$1); do echo \$c; desc \$c; done; }
for p in \$(pgrep -f '^\.venv/bin/python -I tools/remote/remote_eval\.py'); do kill -9 \$p \$(desc \$p) 2>/dev/null; done
sleep 8; pkill -9 -f '[t]ools/remote/remote_actors.py --config'; pkill -9 -f '[s]ls2-deploy/sls2/engine/Sts2Headless'; true
setsid nohup bash -c 'sleep $((SAFETY_MIN * 60)); [ -f $R/PAUSED ] && { pkill -f \"[t]ools/ability_test.py --ckpt $R/\"; sleep 3; rm -f $R/PAUSED \$HOME/.local/share/sls2/runs/defect-eval/PAUSE; bash /root/collector_restart.sh; bash /root/a_eval_restart.sh; }' > /dev/null 2>&1 < /dev/null &
free -g | sed -n 2p"
rsync -a -e "$EA" tools/ability_test.py "$AA:$DA/tools/" && rsync -a -e "$EB" tools/ability_test.py "$AB:$DB/tools/" || { log "同步测试脚本失败"; exit 1; }

k=0
# 快照列表从 3 号描述符读：循环里的 ssh、rsync 会读标准输入，不能让它们把列表吃掉
while read -r v name ck <&3; do
  k=$((k + 1))
  n=$(python3 tools/w10k.py todo "$W" --out "$W/todo.json")
  [ "$n" -gt 0 ] || { log "已全部通关，停止"; break; }
  O=$R/s$k-$name; L=$W/s$k-$name; mkdir -p "$L"
  if [ "$n" -gt "$BIG" ]; then N=$((A_SH + B_SH)); hosts="a b"; else N=$(( (n + ENVS - 1) / ENVS )); [ "$N" -gt "$A_SH" ] && N=$A_SH; hosts="a"; fi
  log "第 $k 个快照 $name（第 $v 版）：打 $n 个还没通关的种子，$N 个分片（$hosts）"
  ok=1
  for h in $hosts; do
    S=$(host_ssh $h); E=$(host_rsync_e $h); AD=$(host_addr $h)
    timeout 120 $S "mkdir -p $O" && rsync -a -e "$E" "$ck" "$AD:$R/m/$name.pt" && rsync -a -e "$E" "$W/todo.json" "$AD:$O/seeds.json" || { log "上传到 $h 失败"; ok=0; }
  done
  [ "$ok" = 1 ] || break
  # 分片编号：A 取前 A_SH 个（只用 A 时取全部），B 取其余
  for h in $hosts; do
    D=$(host_deploy $h); S=$(host_ssh $h)
    if [ "$h" = a ]; then lo=0; if [ "$hosts" = a ]; then hi=$N; else hi=$A_SH; fi; else lo=$A_SH; hi=$N; fi
    timeout 120 $S "cd $D && $ENVSH && for i in \$(seq $lo $((hi - 1))); do setsid nohup .venv/bin/python -I tools/ability_test.py --ckpt $R/m/$name.pt --out $O --asc 10-10 --runs $n --shard \$i/$N --envs $ENVS --device cuda --seeds-file $O/seeds.json > $O/log_s\$i.txt 2>&1 < /dev/null & done" \
      || { log "在 $h 上启动失败"; ok=0; }
  done
  [ "$ok" = 1 ] || break
  t0=$(date +%s)
  while :; do
    sleep 30
    busy=0
    for h in $hosts; do c=$(timeout 60 $(host_ssh $h) "ps -eo args | grep -c '^.venv/bin/python -I tools/[a]bility_test.py --ckpt $R/m/$name.pt' || true" 2>/dev/null) || c=1; busy=$((busy + ${c:-1})); done
    [ "$busy" = 0 ] && break
    [ $(( $(date +%s) - t0 )) -gt 5400 ] && { log "一个半小时还没打完，强停"; for h in $hosts; do timeout 60 $(host_ssh $h) "pkill -f '[t]ools/ability_test.py --ckpt $R/m/$name.pt'"; done; break; }
  done
  for h in $hosts; do rsync -a --include='runs_s*.jsonl' --exclude='*' -e "$(host_rsync_e $h)" "$(host_addr $h):$O/" "$L/" 2>/dev/null; done
  python3 tools/w10k.py merge "$W" "$L" "$v" "$name"
done 3< "$W/order.txt"
log "测试结束：$(tail -1 "$W/cascade.jsonl" 2>/dev/null)"
