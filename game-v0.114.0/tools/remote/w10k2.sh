#!/usr/bin/env bash
# 本机运行：SL 快照池万种级联测试的第二版驱动（决策记录 251 补记）。第一版（w10k.sh）按分片平均分种子，B 每个分片的速度约是 A 的 2.4 倍，
# 大阶段里 B 打完自己那份就空等，小阶段只用 A 更慢。这一版：
#   - 大阶段按实测速度给 A、B 分种子（A 占 A_FRAC），小阶段（不多于 SMALL 个）两台都打全部，谁先打完用谁的；
#   - 哪台空下来、还有没打完的种子，就把剩下的全部接过去打（同一快照打同一种子结果确定，两台重复打到同一个种子没关系）；
#   - 全部种子都有结果就停掉两台上这个快照的进程，进下一个快照。
# 接管模式（ADOPT=1）：训练已由第一版停好，第一版被强杀（不触发恢复）后用这一版接着跑，正在跑的那个快照（cascade.jsonl 之后的第一个）直接接管。
# 结束（含出错、被杀）时照第一版恢复训练、采集、评测与首通探测；两台的到点自动恢复保险是第一版放的，仍然有效。
#   ADOPT=1 nohup bash tools/remote/w10k2.sh >> ~/.local/share/sls2/ability_test/w10k/run.log 2>&1 &
set -u
cd "$(dirname "$0")/../.." && . tools/remote/hosts.sh
A_SH=${A_SH:-12}; B_SH=${B_SH:-12}; ENVS=${ENVS:-20}; A_FRAC=${A_FRAC:-0.3}; SMALL=${SMALL:-300}
W=$HOME/.local/share/sls2/ability_test/w10k
ST=$HOME/.local/share/sls2/ability_test/retest
R=/root/w10k; RB=$(host_run b); DB=$(host_deploy b)
ENVSH='export DOTNET_ROOT=$HOME/.dotnet PATH=$HOME/.dotnet:$PATH DOTNET_CLI_TELEMETRY_OPTOUT=1'
log() { echo "$(date '+%F %T') $*"; }

resume() {
  log "恢复训练"
  $(host_ssh b) "pkill -f '[t]ools/ability_test.py --ckpt $R/' 2>/dev/null; sleep 3; cd \$HOME/$RB && rm -f STOP STOP.w10k && RUN_DIR=\$HOME/$RB bash $DB/tools/remote/start_train.sh" 2>&1 < /dev/null | tail -2
  $(host_ssh a) "pkill -f '[t]ools/ability_test.py --ckpt $R/' 2>/dev/null; sleep 3; rm -f $R/PAUSED \$HOME/.local/share/sls2/runs/defect-eval/PAUSE; bash /root/collector_restart.sh; bash /root/a_eval_restart.sh" 2>&1 < /dev/null | tail -2
  rm -f "$ST/PROBE_OFF"
  log "已恢复"
}
[ -n "${ADOPT:-}" ] || { log "这一版只用于接管（ADOPT=1）：停训由 w10k.sh 做"; exit 1; }
trap resume EXIT

busy() {   # 某台上这个快照还在跑的进程数（连不上时当作 1，免得误判空闲）
  local c; c=$(timeout 60 $(host_ssh "$1") "ps -eo args | grep -c '^.venv/bin/python -I tools/[a]bility_test.py --ckpt $R/m/$2.pt' || true" 2>/dev/null < /dev/null) || c=1
  echo "${c:-1}"
}
launch() {  # launch 主机 快照名 远端目录 本轮名 种子表 最多分片
  local h=$1 name=$2 O=$3 rn=$4 sf=$5 mx=$6 n N
  n=$(python3 -c "import json,sys;print(len(json.load(open(sys.argv[1]))['seeds']))" "$sf")
  [ "$n" -gt 0 ] || return 0
  N=$(( (n + ENVS - 1) / ENVS )); [ "$N" -gt "$mx" ] && N=$mx
  timeout 120 $(host_ssh "$h") "mkdir -p $O/$rn" < /dev/null && rsync -a -e "$(host_rsync_e "$h")" "$sf" "$(host_addr "$h"):$O/$rn/seeds.json" < /dev/null || { log "上传种子表到 $h 失败"; return 1; }
  timeout 120 $(host_ssh "$h") "cd $(host_deploy "$h") && $ENVSH && for i in \$(seq 0 $((N - 1))); do setsid nohup .venv/bin/python -I tools/ability_test.py --ckpt $R/m/$name.pt --out $O/$rn --asc 10-10 --runs $n --shard \$i/$N --envs $ENVS --device cuda --seeds-file $O/$rn/seeds.json > $O/$rn/log_s\$i.txt 2>&1 < /dev/null & done" < /dev/null \
    || { log "在 $h 上启动失败"; return 1; }
  log "  $h 开打 $rn：$n 个种子，$N 个分片"
}
fetch() {   # 两台的结果（含各轮子目录）取回本机 L/<主机>/
  for h in a b; do mkdir -p "$2/$h"; rsync -a --include='*/' --include='runs_s*.jsonl' --exclude='*' -e "$(host_rsync_e "$h")" "$(host_addr "$h"):$1/" "$2/$h/" < /dev/null 2>/dev/null; done
}
remaining() {   # remaining 阶段种子表 本机结果目录 输出种子表：打印还没结果的种子数
  python3 - "$1" "$2" "$3" <<'PY'
import json, sys, glob
src, d, out = sys.argv[1:]
seeds = json.load(open(src))["seeds"]
done = set()
for f in glob.glob(f"{d}/**/runs_s*.jsonl", recursive=True):
    for line in open(f, encoding="utf-8"):
        if line.strip():
            done.add(json.loads(line)["seed"])
rest = [s for s in seeds if s not in done]
json.dump({"name": "W1", "seeds": rest}, open(out, "w"))
print(len(rest))
PY
}

k=$(cat "$W/cascade.jsonl" 2>/dev/null | wc -l)
first=1
while read -r v name ck <&3; do
  k=$((k + 1))
  O=$R/s$k-$name; L=$W/s$k-$name; mkdir -p "$L"
  if [ -n "$first" ] && [ -n "${ADOPT:-}" ] && [ -f "$W/todo.json" ] && [ "$(busy a "$name")$(busy b "$name")" != "00" ]; then
    cp "$W/todo.json" "$L/stage.json"; log "接管第 $k 个快照 $name（第 $v 版），两台上已在跑"
    first=""
  else
    first=""
    n=$(python3 tools/w10k.py todo "$W" --out "$L/stage.json")
    [ "$n" -gt 0 ] || { log "已全部通关，停止"; break; }
    cp "$L/stage.json" "$W/todo.json"
    log "第 $k 个快照 $name（第 $v 版）：打 $n 个还没通关的种子"
    ok=1
    for h in a b; do timeout 120 $(host_ssh $h) "mkdir -p $R/m $O" < /dev/null && rsync -a -e "$(host_rsync_e $h)" "$ck" "$(host_addr $h):$R/m/$name.pt" < /dev/null || { log "上传快照到 $h 失败"; ok=0; }; done
    [ "$ok" = 1 ] || break
    if [ "$n" -le "$SMALL" ]; then
      launch a "$name" "$O" r0 "$L/stage.json" "$A_SH"; launch b "$name" "$O" r0 "$L/stage.json" "$B_SH"
    else
      python3 - "$L/stage.json" "$A_FRAC" "$L/a0.json" "$L/b0.json" <<'PY'
import json, sys
src, frac, fa, fb = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4]
s = json.load(open(src))["seeds"]; m = int(round(len(s) * frac))
json.dump({"name": "W1", "seeds": s[:m]}, open(fa, "w")); json.dump({"name": "W1", "seeds": s[m:]}, open(fb, "w"))
PY
      launch a "$name" "$O" r0 "$L/a0.json" "$A_SH"; launch b "$name" "$O" r0 "$L/b0.json" "$B_SH"
    fi
  fi
  t0=$(date +%s); rnd=0
  while :; do
    sleep 20
    fetch "$O" "$L"
    rest=$(remaining "$L/stage.json" "$L" "$L/rest.json")
    [ "$rest" = 0 ] && break
    ba=$(busy a "$name"); bb=$(busy b "$name")
    if [ "$ba" = 0 ] || [ "$bb" = 0 ]; then
      if [ "$ba" = 0 ] && [ "$bb" = 0 ]; then
        rnd=$((rnd + 1)); [ "$rnd" -gt 3 ] && { log "两台都停了还剩 $rest 个种子没有结果（可能卡死），按没通关算"; break; }
        log "  两台都空了还剩 $rest 个，补打"; launch a "$name" "$O" "r$rnd" "$L/rest.json" "$A_SH"; launch b "$name" "$O" "r$rnd" "$L/rest.json" "$B_SH"
      else
        idle=a; [ "$bb" = 0 ] && idle=b
        rnd=$((rnd + 1)); mx=$A_SH; [ "$idle" = b ] && mx=$B_SH
        launch "$idle" "$name" "$O" "r$rnd" "$L/rest.json" "$mx"
      fi
    fi
    [ $(( $(date +%s) - t0 )) -gt 5400 ] && { log "一个半小时还没打完，强停"; break; }
  done
  for h in a b; do timeout 60 $(host_ssh $h) "pkill -f '[t]ools/ability_test.py --ckpt $R/m/$name.pt'" < /dev/null; done
  fetch "$O" "$L"
  python3 - "$L" <<'PY'
import glob, os, sys
d = sys.argv[1]
os.makedirs(f"{d}/merged", exist_ok=True)
with open(f"{d}/merged/runs_s0.jsonl", "w", encoding="utf-8") as out:
    for f in glob.glob(f"{d}/[ab]/**/runs_s*.jsonl", recursive=True):
        out.write(open(f, encoding="utf-8").read())
PY
  python3 tools/w10k.py merge "$W" "$L/merged" "$v" "$name"
done 3< <(tail -n +$((k + 1)) "$W/order.txt")
log "测试结束：$(tail -1 "$W/cascade.jsonl" 2>/dev/null)"
