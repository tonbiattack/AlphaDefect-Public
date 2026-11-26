#!/usr/bin/env bash
# 本机运行：万种级联测试的续接驱动（决策记录 251 补记三）。用户：“跑完以后，把最近跑过首通测试的，不在模型池里的模型，再跑一遍首通测试。
# 用万种测试测出来的无模型通关的种子作为测试目标。”
#   1. 强停第三版驱动 w10k3.sh（SIGKILL，不触发恢复），它在两台上起的级联工作进程照常跑；先把附加快照传到 B；
#   2. 等两台上的级联工作进程都结束，取回结果，pipe-merge 写进 cascade.jsonl（同 w10k3.sh 的收尾）；
#   3. 附加一轮：首通探测测过、但不在 SL 快照池里的快照（tools/w10k.py extra-models，按探测时间从近到远），
#      去打万种测试里所有快照都没通关的种子，同样逐种子补位（tools/cascade_worker.py），结果写 cascade_extra.jsonl；
#   4. 恢复训练、采集、评测与首通探测。
#   nohup bash tools/remote/w10k4.sh >> ~/.local/share/sls2/ability_test/w10k/run.log 2>&1 &
set -u
cd "$(dirname "$0")/../.." && . tools/remote/hosts.sh
NA=${NA:-6}; NB=${NB:-7}; ENVS=${ENVS:-35}
W=$HOME/.local/share/sls2/ability_test/w10k
ST=$HOME/.local/share/sls2/ability_test/retest
R=/root/w10k; RB=$(host_run b); DB=$(host_deploy b)
ENVSH='export DOTNET_ROOT=$HOME/.dotnet PATH=$HOME/.dotnet:$PATH DOTNET_CLI_TELEMETRY_OPTOUT=1'
log() { echo "$(date '+%F %T') $*"; }
resume() {
  log "恢复训练"
  $(host_ssh b) "pkill -f '[t]ools/ability_test.py --ckpt $R/|[t]ools/cascade_worker.py' 2>/dev/null; sleep 3; cd \$HOME/$RB && rm -f STOP STOP.w10k && RUN_DIR=\$HOME/$RB bash $DB/tools/remote/start_train.sh" 2>&1 < /dev/null | tail -2
  $(host_ssh a) "pkill -f '[t]ools/ability_test.py --ckpt $R/|[t]ools/cascade_worker.py' 2>/dev/null; sleep 3; rm -f $R/PAUSED \$HOME/.local/share/sls2/runs/defect-eval/PAUSE; bash /root/collector_restart.sh; bash /root/a_eval_restart.sh" 2>&1 < /dev/null | tail -2
  rm -f "$ST/PROBE_OFF"
  log "已恢复"
}
running() { local c=0 x; for h in a b; do x=$(timeout 60 $(host_ssh $h) "ps -eo args | grep -c '[t]ools/cascade_worker.py' || true" < /dev/null 2>/dev/null) || x=1; c=$((c + ${x:-1})); done; echo "$c"; }

log "续接：强停第三版驱动（不触发恢复），两台上的级联工作进程照常跑"
for p in $(pgrep -f "^bash tools/remote/[w]10k3\.sh"); do pkill -9 -P "$p" 2>/dev/null; kill -9 "$p" 2>/dev/null; done
trap resume EXIT

# 附加快照：A 上用首通探测时留下的，B 上缺的从本机上传（趁级联还在跑）
mkdir -p "$W/pipe2/res"
python3 tools/w10k.py extra-models > "$W/pipe2/extra.txt"
log "附加快照 $(wc -l < "$W/pipe2/extra.txt") 个，先传到 B"
timeout 60 $(host_ssh b) "mkdir -p $R/m $R/pipe2" < /dev/null; timeout 60 $(host_ssh a) "mkdir -p $R/pipe2" < /dev/null
awk '{print $1, $3}' "$W/pipe2/extra.txt" > "$W/pipe2/models_a.txt"
awk -v r="$R" '{print $1, r "/m/" $1 ".pt"}' "$W/pipe2/extra.txt" > "$W/pipe2/models_b.txt"
EB=$(host_rsync_e b); AB=$(host_addr b)
awk -v a="$AB" -v r="$R" '{print $2, a ":" r "/m/" $1 ".pt"}' "$W/pipe2/extra.txt" | xargs -P 4 -n 2 sh -c 'rsync -a -e "$0" "$1" "$2"' "$EB" 2>&1 | tail -3
for h in a b; do rsync -a -e "$(host_rsync_e $h)" "$W/pipe2/models_$h.txt" "$(host_addr $h):$R/pipe2/models.txt" < /dev/null; done
missing=$(awk '{print $2}' "$W/pipe2/models_a.txt" | timeout 60 $(host_ssh a) 'while read -r p; do [ -f "$p" ] || echo "$p"; done' | wc -l)
[ "$missing" = 0 ] || log "A 上缺 $missing 个附加快照（首通探测目录被清了？），这几个只在 B 上能打"

log "等级联工作进程结束"
t0=$(date +%s)
while [ "$(running)" != 0 ]; do sleep 30; [ $(( $(date +%s) - t0 )) -gt 10800 ] && { log "三个小时还没打完，强停"; break; }; done
for h in a b; do rsync -a --include='out_*.jsonl' --exclude='*' -e "$(host_rsync_e $h)" "$(host_addr $h):$R/pipe/" "$W/pipe/res/" < /dev/null 2>/dev/null; done
python3 tools/w10k.py pipe-merge "$W" "$W/pipe/res"
log "33 个快照的级联结束：$(tail -1 "$W/cascade.jsonl")"

n=$(python3 tools/w10k.py extra-tasks "$W" --out "$W/pipe2/tasks.json")
log "附加一轮：$n 个所有快照都没通关的种子"
if [ "$n" -gt 0 ]; then
  python3 - "$W/pipe2/tasks.json" "$W/pipe2" "$NA" "$NB" <<'PY'
import json, random, sys
src, d, na, nb = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
t = json.load(open(src))["tasks"]; random.Random(2512).shuffle(t)
m = int(round(len(t) * na / (na + nb * 1.6))); parts = {"a": t[:m], "b": t[m:]}
for h, n in (("a", na), ("b", nb)):
    for i in range(n):
        json.dump({"tasks": parts[h][i::n]}, open(f"{d}/tasks_{h}{i}.json", "w"))
PY
  for h in a b; do
    k=$NA; [ "$h" = b ] && k=$NB
    for i in $(seq 0 $((k - 1))); do rsync -a -e "$(host_rsync_e $h)" "$W/pipe2/tasks_$h$i.json" "$(host_addr $h):$R/pipe2/" < /dev/null; done
    rsync -a -e "$(host_rsync_e $h)" tools/cascade_worker.py "$(host_addr $h):$(host_deploy $h)/tools/" < /dev/null
    timeout 120 $(host_ssh $h) "cd $(host_deploy $h) && $ENVSH && for i in \$(seq 0 $((k - 1))); do [ \$(python3 -c \"import json;print(len(json.load(open('$R/pipe2/tasks_$h\$i.json'))['tasks']))\") -gt 0 ] && setsid nohup .venv/bin/python -I tools/cascade_worker.py --models $R/pipe2/models.txt --tasks $R/pipe2/tasks_$h\$i.json --out $R/pipe2/out_$h\$i.jsonl --envs $ENVS --device cuda > $R/pipe2/log_$h\$i.txt 2>&1 < /dev/null & done; true" < /dev/null
  done
  sleep 30; t0=$(date +%s)
  while [ "$(running)" != 0 ]; do sleep 30; [ $(( $(date +%s) - t0 )) -gt 7200 ] && { log "两个小时还没打完，强停"; break; }; done
  for h in a b; do rsync -a --include='out_*.jsonl' --exclude='*' -e "$(host_rsync_e $h)" "$(host_addr $h):$R/pipe2/" "$W/pipe2/res/" < /dev/null 2>/dev/null; done
  python3 tools/w10k.py extra-merge "$W" "$W/pipe2/res" "$W/pipe2/models_a.txt"
fi
log "测试结束：$(tail -1 "$W/cascade_extra.jsonl" 2>/dev/null || tail -1 "$W/cascade.jsonl")"
