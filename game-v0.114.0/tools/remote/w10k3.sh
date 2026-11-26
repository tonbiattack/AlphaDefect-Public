#!/usr/bin/env bash
# 本机运行：SL 快照池万种级联测试的第三版驱动——逐种子级联（流水线，决策记录 251 补记二）。
# 阶段式（w10k.sh、w10k2.sh）每个快照都要上传、重起引擎、等最慢的几局，后面种子少于引擎数时大部分引擎闲着。这一版：
#   - 剩下的种子分给各个工作进程（tools/cascade_worker.py，A NA 个、B NB 个，每个 ENVS 个引擎），每个进程装着全部快照，
#     每个种子输了就立刻换下一个快照接着打，直到通关或快照用完；种子之间互不依赖，不用跨机器协调；
#   - 起点由 tools/w10k.py pipe-tasks 定：已测完的快照跳过，正在打的那个快照里已有结果的直接用；
#   - 全部进程结束后取回结果，pipe-merge 按快照汇总，接着写进 cascade.jsonl。
# 先强停阶段式驱动（SIGKILL，不触发恢复）与两台上的阶段式进程；结束（含出错、被杀）时恢复训练、采集、评测与首通探测。
#   nohup bash tools/remote/w10k3.sh >> ~/.local/share/sls2/ability_test/w10k/run.log 2>&1 &
set -u
cd "$(dirname "$0")/../.." && . tools/remote/hosts.sh
NA=${NA:-6}; NB=${NB:-7}; ENVS=${ENVS:-35}; FA=${FA:-0.35}
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

log "换流水线：强停阶段式驱动与两台上的阶段式进程"
for p in $(pgrep -f "^bash tools/remote/[w]10k2?\.sh"); do pkill -9 -P "$p" 2>/dev/null; kill -9 "$p" 2>/dev/null; done
for h in a b; do timeout 60 $(host_ssh $h) "pkill -f '[t]ools/ability_test.py --ckpt $R/'" < /dev/null; done
trap resume EXIT
sleep 5
# 正在打的那个快照的部分结果先取回（pipe-tasks、pipe-merge 都要用）
k=$(cat "$W/cascade.jsonl" | wc -l); cur=$(sed -n "$((k + 1))p" "$W/order.txt" | cut -d' ' -f2)
if [ -n "$cur" ]; then
  for h in a b; do mkdir -p "$W/s$((k + 1))-$cur/$h"; rsync -a --include='*/' --include='runs_s*.jsonl' --exclude='*' -e "$(host_rsync_e $h)" "$(host_addr $h):$R/s$((k + 1))-$cur/" "$W/s$((k + 1))-$cur/$h/" < /dev/null 2>/dev/null; done
fi
read -r ntask base part < <(python3 tools/w10k.py pipe-tasks "$W" --out "$W/pipe_tasks.json")
log "流水线起点：$ntask 个种子，从第 $((base + 1)) 个快照起（它已出结果里通关 $part 个）"
[ "$ntask" -gt 0 ] || { python3 tools/w10k.py pipe-merge "$W" "$W/pipe"; exit 0; }

# 每台一份快照表（路径为该机上的）：A 优先用 /root/w10k/m/ 或复测目录里已有的，没有的、以及 B 上没有的，都从本机上传
mkdir -p "$W/pipe/res"; rm -f "$W"/pipe/upload_*.txt
for h in a b; do
  python3 - "$W/order.txt" "$h" > "$W/pipe/need_$h.txt" <<'PY'
import sys, os
order, h = sys.argv[1], sys.argv[2]
for line in open(order):
    v, name, path = line.split()
    cands = [f"/root/w10k/m/{name}.pt"] + ([f"/root/retest/{os.path.basename(os.path.dirname(path))}/model.pt"] if h == "a" else [])
    print(name, path, " ".join(cands))
PY
  # 远端查哪些候选路径已经存在：每行打印 “名字 远端路径”（没有的打印 “名字 -”）
  awk '{printf "%s", $1; for (i = 3; i <= NF; i++) printf " %s", $i; print ""}' "$W/pipe/need_$h.txt" \
    | timeout 120 $(host_ssh $h) 'while read -r n rest; do p=-; for c in $rest; do [ -f "$c" ] && { p=$c; break; }; done; echo "$n $p"; done' > "$W/pipe/have_$h.txt"
  : > "$W/pipe/models_$h.txt"
  while read -r name local rest; do
    rp=$(awk -v n="$name" '$1 == n {print $2}' "$W/pipe/have_$h.txt")
    if [ -z "$rp" ] || [ "$rp" = - ]; then echo "$local $(host_addr $h):$R/m/$name.pt" >> "$W/pipe/upload_$h.txt"; rp=$R/m/$name.pt; fi
    echo "$name $rp" >> "$W/pipe/models_$h.txt"
  done < "$W/pipe/need_$h.txt"
  if [ -s "$W/pipe/upload_$h.txt" ]; then
    log "上传 $(wc -l < "$W/pipe/upload_$h.txt") 个快照到 $h"
    timeout 60 $(host_ssh $h) "mkdir -p $R/m $R/pipe" < /dev/null
    E=$(host_rsync_e $h)
    while read -r src dst; do echo "$src $dst"; done < "$W/pipe/upload_$h.txt" | xargs -P 4 -n 2 sh -c 'rsync -a -e "$0" "$1" "$2"' "$E" 2>&1 | tail -3
  fi
  timeout 60 $(host_ssh $h) "mkdir -p $R/pipe" < /dev/null
  rsync -a -e "$(host_rsync_e $h)" "$W/pipe/models_$h.txt" "$(host_addr $h):$R/pipe/models.txt" < /dev/null
  rsync -a -e "$(host_rsync_e $h)" tools/cascade_worker.py "$(host_addr $h):$(host_deploy $h)/tools/" < /dev/null
done

# 分任务：打乱后 A 占 FA，其余给 B，各自再平分给工作进程
python3 - "$W/pipe_tasks.json" "$W/pipe" "$NA" "$NB" "$FA" <<'PY'
import json, random, sys
src, d, na, nb, fa = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])
t = json.load(open(src))["tasks"]; random.Random(251).shuffle(t)
m = int(round(len(t) * fa)); parts = {"a": t[:m], "b": t[m:]}
for h, n in (("a", na), ("b", nb)):
    for i in range(n):
        json.dump({"tasks": parts[h][i::n]}, open(f"{d}/tasks_{h}{i}.json", "w"))
PY
for h in a b; do
  n=$NA; [ "$h" = b ] && n=$NB
  for i in $(seq 0 $((n - 1))); do rsync -a -e "$(host_rsync_e $h)" "$W/pipe/tasks_$h$i.json" "$(host_addr $h):$R/pipe/" < /dev/null; done
  timeout 120 $(host_ssh $h) "cd $(host_deploy $h) && $ENVSH && for i in \$(seq 0 $((n - 1))); do setsid nohup .venv/bin/python -I tools/cascade_worker.py --models $R/pipe/models.txt --tasks $R/pipe/tasks_$h\$i.json --out $R/pipe/out_$h\$i.jsonl --envs $ENVS --device cuda > $R/pipe/log_$h\$i.txt 2>&1 < /dev/null & done" < /dev/null
  log "  $h 起了 $n 个工作进程"
done

t0=$(date +%s)
while :; do
  sleep 30
  c=0; for h in a b; do x=$(timeout 60 $(host_ssh $h) "ps -eo args | grep -c '[t]ools/cascade_worker.py' || true" < /dev/null 2>/dev/null) || x=1; c=$((c + ${x:-1})); done
  [ "$c" = 0 ] && break
  [ $(( $(date +%s) - t0 )) -gt 10800 ] && { log "三个小时还没打完，强停"; break; }
done
for h in a b; do rsync -a --include='out_*.jsonl' --exclude='*' -e "$(host_rsync_e $h)" "$(host_addr $h):$R/pipe/" "$W/pipe/res/" < /dev/null 2>/dev/null; done
python3 tools/w10k.py pipe-merge "$W" "$W/pipe/res"
log "测试结束：$(tail -1 "$W/cascade.jsonl")"
