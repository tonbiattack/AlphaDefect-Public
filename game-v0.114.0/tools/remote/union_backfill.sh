#!/usr/bin/env bash
# 并集补测（2026-11-02 起，决策记录 248）：在定时复测的空档里，把一个快照在种子池上补打一遍（tools/remote/retest.sh 的 BACKFILL 模式）。
# 默认测最近的快照：评测最新留存、还没测过的那一版（11-02 用户：“A 上加测不要再测过去版本，而是有加测窗口时测最近快照”）；
# MODE=past 为原来的从近往前补过往强快照。
# 打完用 tools/union_backfill.py 判定：对快照通关并集有提升就纳入记录，tools/retest_union.py 的累计并集把它算进去。
# 本机守护（local_watch.sh）每分钟调一次，不满足条件就直接退出：
#   - 定时复测（或别的复测）正在跑：不开；
#   - 离下一次定时复测不到 MARGIN 版：不开（默认 0，即不看余量）。原来默认 150，可定时复测测的是常规评测的留存版（比主线落后几十版），
#     复测加双 Boss 卷又要约 20 分钟，测完时离下一次通常已不到 150 版，补测一直轮不上（11-02 改）。现在每次定时复测之后紧接着补 1 个，
#     下一次定时复测最多推迟一次补测的时间（约 12 分钟）；
#   - 这个空档（上一次定时复测之后）已经补了 PER_GAP 个（默认 1）：不开，免得 A 的常规评测让路太久；
#   - ~/.local/share/sls2/ability_test/retest/BACKFILL_OFF 存在：不开（手动关掉）。
#   bash tools/remote/union_backfill.sh
set -u
cd "$(dirname "$0")/../.." && . tools/remote/hosts.sh
ST=$HOME/.local/share/sls2/ability_test/retest
exec 8> "$ST/backfill.lock"; flock -n 8 || exit 0
[ -f "$ST/BACKFILL_OFF" ] && exit 0
# 守护每分钟先拉起一次 retest.sh（没到时间一两秒就退出），紧接着才拉起这里：先等 20 秒，免得每次都把那个马上要退出的当成在测（11-02 查出补测一直没开过）
sleep 20
pgrep -f "sls2/tools/remote/retest.sh" > /dev/null && exit 0
MARGIN=${MARGIN:-0}; PER_GAP=${PER_GAP:-1}
H=$(main_host); LOC=$(host_local "$H")
ver=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['version'])" "$LOC/status.json" 2>/dev/null) || exit 0
read -r last next step < <(python3 - "$ST/state.json" <<'PY'
import json, re, sys
s = json.load(open(sys.argv[1]))
src = open("tools/remote/retest.sh", encoding="utf-8").read()
step = int(re.search(r"STEP=\$\{STEP:-(\d+)\}", src).group(1))
print(s["runs"][-1]["version"] if s["runs"] else 0, s.get("next_at") or 0, step)
PY
) || exit 0
thr=$(( next > 0 ? next : last + step ))
[ $(( thr - ver )) -ge "$MARGIN" ] || exit 0
[ "$(python3 tools/union_backfill.py gapcount "$last")" -lt "$PER_GAP" ] || exit 0
log() { echo "$(date '+%F %T') $*" >> "$ST/retest.log"; }
EVR=.local/share/sls2/runs/defect-eval
T=$(mktemp -d)
timeout 60 $(host_ssh a) "cat \$HOME/$EVR/best/index.json" > "$T/best.json" 2>/dev/null
timeout 60 $(host_ssh a) "cat \$HOME/$EVR/high/index.json" > "$T/high.json" 2>/dev/null
timeout 60 $(host_ssh a) "ls \$HOME/$EVR/eval_runs/" > "$T/eval_runs.txt" 2>/dev/null
# 默认测最近的快照（评测最新留存、还没测过的那一版，11-02 用户要求）；MODE=past 为原来的从近往前补过往强快照
pick=$(python3 tools/union_backfill.py next --mode "${MODE:-recent}" --best "$T/best.json" --high "$T/high.json" --eval-runs "$T/eval_runs.txt"); rm -rf "$T"
[ -n "$pick" ] || exit 0
set -- $pick
if [ "$1" = LOCAL ]; then ck=$2; v=$3; log "并集补测：第 $v 版（本机 $ck），主线第 $ver 版、离下一次定时复测还有 $(( thr - ver )) 版"
  FORCE=1 BACKFILL=1 LOCAL_CK=$ck V=$v bash "$PWD/tools/remote/retest.sh"
else v=$2; log "并集补测：第 $v 版（评测留档），主线第 $ver 版、离下一次定时复测还有 $(( thr - ver )) 版"
  FORCE=1 BACKFILL=1 EVAL_V=$v bash "$PWD/tools/remote/retest.sh"
fi
python3 tools/union_backfill.py judge "$v" "$HOME/.local/share/sls2/ability_test/v$v-并集补测" --gap "$last" >> "$ST/retest.log" 2>&1
python3 tools/retest_union.py >> "$ST/retest.log" 2>&1
