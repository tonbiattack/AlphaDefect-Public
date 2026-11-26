#!/usr/bin/env bash
# 首通探测（2026-11-02 起，决策记录 250）：拿一个还没探测过的快照（新的评测留存优先，没有新的就往前测从未测过的留档，含 B 的定时备份），在 A 上只打种子池 P1 里全局未通关的那几个种子（tools/unsolved_probe.py 选题、判定），
# 有首通就把快照链进 SL 直播快照池（~/.local/share/sls2/live/sl_pool/）并记录，看板复测区块显示。取代并集补测（union_backfill.sh）。
# 本机守护（local_watch.sh）每分钟调一次；每次探测一个快照（约 2–3 分钟），所以基本一直在测（用户：“保持一直有加测”）。
# 只有几局，不让常规评测让路（不放 PAUSE）：评测每轮留存的权重正是新快照的来源。以下情况直接退出：
#   - 定时复测（retest.sh）或并集补测在跑，或定时复测已经到点（让定时复测先测）；
#   - 已经没有全局未通关的种子，或没有可探测的快照；
#   - ~/.local/share/sls2/ability_test/retest/PROBE_OFF 存在（手动关掉）。
#   bash tools/remote/unsolved_probe.sh
set -u
name=""; id=""; link=""
cd "$(dirname "$0")/../.." && . tools/remote/hosts.sh
AT=$HOME/.local/share/sls2/ability_test; ST=$AT/retest
exec 7> "$ST/probe.lock"; flock -n 7 || exit 0
[ -f "$ST/PROBE_OFF" ] && exit 0
# 守护每分钟先拉起一次 retest.sh（没到时间一两秒就退出），先等它退出再看
sleep 20
pgrep -f "sls2/tools/remote/retest.sh" > /dev/null && exit 0
pgrep -f "sls2/tools/remote/union_backfill.sh" > /dev/null && exit 0
H=$(main_host); LOC=$(host_local "$H")
ver=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['version'])" "$LOC/status.json" 2>/dev/null) || exit 0
read -r last next step < <(python3 - "$ST/state.json" <<'PY'
import json, re, sys
s = json.load(open(sys.argv[1]))
step = int(re.search(r"STEP=\$\{STEP:-(\d+)\}", open("tools/remote/retest.sh", encoding="utf-8").read()).group(1))
print(s["runs"][-1]["version"] if s["runs"] else 0, s.get("next_at") or 0, step)
PY
) || exit 0
thr=$(( next > 0 ? next : last + step ))
[ "$ver" -lt "$thr" ] || exit 0
log() { echo "$(date '+%F %T') $*" >> "$ST/retest.log"; }
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
n=$(python3 tools/unsolved_probe.py seeds --out "$T/seeds.json") || exit 0
[ "$n" -gt 0 ] || exit 0
SSH=$(host_ssh a); E=$(host_rsync_e a); AD=$(host_addr a); TD=$(host_deploy a)
EVR=.local/share/sls2/runs/defect-eval
timeout 60 $SSH "ls \$HOME/$EVR/eval_runs/" > "$T/er.txt" 2>/dev/null
timeout 60 $SSH "cat \$HOME/$EVR/best/index.json" > "$T/best.json" 2>/dev/null
timeout 60 $SSH "cat \$HOME/$EVR/high/index.json" > "$T/high.json" 2>/dev/null
BS=$(host_ssh b); BE=$(host_rsync_e b); BA=$(host_addr b); BD=$(host_deploy b); BR=$(host_run b)
timeout 60 $BS "ls \$HOME/$BR/backups/" > "$T/bk.txt" 2>/dev/null
pick=$(python3 tools/unsolved_probe.py next --eval-runs "$T/er.txt" --best "$T/best.json" --high "$T/high.json" --backups "$T/bk.txt")
[ -n "$pick" ] || exit 0
set -- $pick
if [ "$1" = EVAL ]; then v=$2; L=$AT/v$v-首通探测; mkdir -p "$L"; ck=$L/model.pt
  # 评测留存的权重补上主线配置，存成带 config 的检查点（同 retest.sh）
  rsync -a -e "$E" "$AD:$EVR/$3/v$v.pt" "$T/state.pt" 2>/dev/null && .venv/bin/python -I - "$T/state.pt" "$LOC/config.json" "$v" "$ck" <<'PY' \
    || { log "首通探测：第 $v 版的评测留存权重取不到（$3/，可能刚被轮换掉）"; python3 tools/unsolved_probe.py judge "$v" "$L" "$ck" --tested "$T/seeds.json" >> "$ST/retest.log" 2>&1; exit 0; }
import json, re, sys, torch
st, cfgp, v, out = sys.argv[1:]
fmt = int(re.search(r"^FORMAT_VERSION = (\d+)", open("python/sts2env/protocol.py", encoding="utf-8").read(), re.M).group(1))
vn = json.load(open("engine/catalog/v0.111.0/catalog.json", encoding="utf-8"))["vocab_size"]
torch.save({"version": int(v), "state": torch.load(st, map_location="cpu", weights_only=False), "config": json.load(open(cfgp)),
            "obs_format": fmt, "vocab_n": vn, "note": "常规评测留存的权重，配置取主线的 config.json（首通探测）"}, out)
PY
elif [ "$1" = BACKUP ]; then name=$2; v=$3; stem=${name%.pt}; L=$AT/$stem-首通探测; mkdir -p "$L"; ck=$L/model.pt; id="B:$name"; link="$stem.pt"
  # 在 B 上只抽出滑动平均权重（ema，约 72 MB；整个备份含优化器约 276 MB），最低优先级、几秒钟，再经本机中转到 A
  timeout 300 $BS "mkdir -p /root/sls2-probe && cd $BD && nice -n 19 .venv/bin/python -I - \$HOME/$BR/backups/$name /root/sls2-probe/$stem.pt" <<'PY' \
    && rsync -a -e "$BE" "$BA:/root/sls2-probe/$stem.pt" "$ck" \
    || { log "首通探测：B 的备份 $name 取不到"; timeout 60 $BS "rm -f /root/sls2-probe/$stem.pt"; python3 tools/unsolved_probe.py judge "$v" "$L" "$ck" --tested "$T/seeds.json" --id "$id" >> "$ST/retest.log" 2>&1; exit 0; }
import sys, torch
src, out = sys.argv[1:]
d = torch.load(src, map_location="cpu", weights_only=False)
torch.save({"version": int(d["version"]), "state": d.get("ema") or d["state"], "config": d["config"], "obs_format": d.get("obs_format"),
            "vocab_n": d.get("vocab_n"), "note": f"B 的定时备份 {src.rsplit('/', 1)[-1]} 的滑动平均权重（首通探测）"}, out)
PY
  timeout 60 $BS "rm -f /root/sls2-probe/$stem.pt"
else ck=$2; v=$3; L=$AT/v$v-首通探测; mkdir -p "$L"; fi
log "首通探测：第 $v 版${name:+（B 备份 $name）}打 $n 个全局未通关的种子（$1），主线第 $ver 版"
O=/root/retest/$(basename "$L")
timeout 120 $SSH "mkdir -p $O" && rsync -a -e "$E" "$ck" "$AD:$O/model.pt" && rsync -a -e "$E" "$T/seeds.json" "$AD:$O/seeds.json" \
  && rsync -a -e "$E" tools/ability_test.py "$AD:$TD/tools/" || { log "首通探测：上传第 $v 版失败"; exit 0; }
timeout 1800 $SSH "cd $TD && export DOTNET_ROOT=\$HOME/.dotnet PATH=\$HOME/.dotnet:\$PATH DOTNET_CLI_TELEMETRY_OPTOUT=1 &&
  nice -n 10 .venv/bin/python -I tools/ability_test.py --ckpt $O/model.pt --out $O --asc 10-10 --runs $n --shard 0/1 --envs $n --device cuda --seeds-file $O/seeds.json > $O/log.txt 2>&1"
rsync -a --exclude 'A10/' -e "$E" "$AD:$O/runs_s0.jsonl" "$L/" 2>/dev/null
echo "{\"ckpt\": \"$ck\", \"probe\": true, \"seeds\": $n, \"host\": \"a\"}" > "$L/meta.json"
python3 tools/unsolved_probe.py judge "$v" "$L" "$ck" --tested "$T/seeds.json" ${id:+--id "$id"} ${link:+--link-name "$link"} >> "$ST/retest.log" 2>&1
python3 tools/retest_union.py >> "$ST/retest.log" 2>&1
