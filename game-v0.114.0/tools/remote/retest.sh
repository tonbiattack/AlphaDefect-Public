#!/usr/bin/env bash
# 定期复测（2026-10-28 用户要求，决策记录 171）：主线每多训约 STEP 版（默认 500；决策记录 247 起复测加倍、原为 1,000；决策记录 222 起每版数据翻倍，原为 2,000），取主线机最新的定时备份，在主线机上用固定的新种子
# RT0000–RT0999 打 1,000 局（进阶 10、纯贪心，tools/ability_test.py，2 个分片 × 20 个引擎，约 7 分钟；
# 原来 8 × 5，观测格式 15 上线后学习端占显存约 17.8 GB，剩下的只够两三个进程各开一份 CUDA 上下文，决策记录 196），取回逐局结果，
# 与上一次复测逐局配对比较（tools/ability_compare.py），生成报告放到原目录。本机守护（local_watch.sh）每分钟调一次，没到时间就直接退出。
# 结果记在 ~/.local/share/sls2/ability_test/retest/state.json（每次一条：版本、通关、95% 区间、对上一次的配对结果）与 retest.log。
# 复测护栏：比上一次显著下降（只有上一次通关的局多于只有这一次通关的局，且 p < 0.05）时写 ALERT 文件，由主会话巡检时报告（不自动回退）。
#   bash tools/remote/retest.sh            # 到时间才测
#   FORCE=1 bash tools/remote/retest.sh    # 立即测一次
#   FORCE=1 CK=v66500-20261029-0500.pt bash tools/remote/retest.sh   # 指定备份
#   FORCE=1 NORECORD=1 LOCAL_CK=~/.local/share/sls2/keep/v84667.pt V=84667 bash tools/remote/retest.sh
#   FORCE=1 EVAL_V=117152 bash tools/remote/retest.sh   # 计划外复测评测留档的某一版（eval_runs/、best/、high/，决策记录 208；隐含 NORECORD）
#       计划外复测本机的检查点（决策记录 197）：结果放 ability_test/v<版本>-计划外，照样与上一次复测配对比较、出报告，但不记进 state.json
#       （不影响定时复测的节奏），也不写 ALERT
#   其他计划外的测法（都隐含 NORECORD，结果目录带标签）：SEEDS=HO 换验收种子 HO0000–HO0999（平时不测，只在切直播模型、大改之后测；
#   与上一次同种子的验收配对，没有就只出报告）；TEMP=1 按温度采样、EXPLORE=1 局外照训练的办法强制改选（采样拆分评测，与同版本的贪心复测配对）
set -u
cd "$(dirname "$0")/../.." && . tools/remote/hosts.sh
STEP=${STEP:-500}; RUNS=${RUNS:-1000}; SHARDS=${SHARDS:-2}; ENVS=${ENVS:-20}
W=/mnt/c/主存储文件/脚本开发/sls2
ST=$HOME/.local/share/sls2/ability_test/retest; mkdir -p "$ST"
exec 9> "$ST/lock"; flock -n 9 || exit 0          # 上一次还没测完
H=$(main_host); LOC=$(host_local "$H")
ver=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['version'])" "$LOC/status.json" 2>/dev/null) || exit 0
last=$(python3 -c "import json,sys; s=json.load(open(sys.argv[1])); print(s['runs'][-1]['version'] if s['runs'] else 0)" "$ST/state.json" 2>/dev/null || echo 0)
# state.json 的 next_at：下一次在第几版测（回退后版本号倒回去时手动设，测完即清掉，之后照旧每 STEP 版）；没有时为上一次 + STEP
next=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('next_at') or '')" "$ST/state.json" 2>/dev/null || echo)
thr=${next:-$((last + STEP))}
[ -n "${FORCE:-}" ] || [ "$ver" -ge "$thr" ] || exit 0
log() { echo "$(date '+%F %T') $*" >> "$ST/retest.log"; }
SSH=$(host_ssh "$H"); E=$(host_rsync_e "$H"); AD=$(host_addr "$H"); D=$(host_deploy "$H"); R=$(host_run "$H")
# CK=备份文件名 可指定测哪一份（补测用），默认取时间上最新的定时备份（不带“重启前”“回退前”标签）。
# 不按版本号排：回退之后，作废那条线上的备份版本号反而更大
# 种子（决策记录 240 起）：默认用种子池 tools/remote/seed_pool.json（tools/seed_pool.py，密码学随机源抽的 1,000 个，常规评测用其中 384 个），
# SEEDS 为池名（如 P1）；SEEDS=RT、HO 等仍按前缀起名（RT0000…，旧的序号种子）。换池后与上一次复测没有共同种子时不配对（只出报告）
POOL=tools/remote/seed_pool.json; PN=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['name'])" "$POOL" 2>/dev/null || echo RT)
SEEDS=${SEEDS:-$PN}; TEMP=${TEMP:-0}; EXPLORE=${EXPLORE:-}
SEEDARG="--seed-prefix $SEEDS"; [ "$SEEDS" = "$PN" ] && [ "$PN" != RT ] && SEEDARG="--seeds-file $POOL"
# 两个结果目录有没有共同的种子（换种子池之后，与旧池的复测、直播基线都没有）
overlap() { python3 - "$1" "$2" <<'PY'
import glob, json, sys
def ss(d):
    out = set()
    for f in glob.glob(f"{d}/runs_s*.jsonl"):
        out |= {json.loads(l)["seed"] for l in open(f, encoding="utf-8") if l.strip()}
    return out
sys.exit(0 if ss(sys.argv[1]) & ss(sys.argv[2]) else 1)
PY
}
# 贪心按等价动作合并：决策记录 206 起默认打开，10-31 用户决定取消（决策记录 207：同版本同种子配对，合并贪心反而少赢），默认关、与 A 上评测一致；MERGE=1 打开
MERGE=${MERGE-}
# 沿用常规评测（决策记录 203）：评测（A 上的 remote_eval.py --keep-runs）每轮把 384 个 RT 种子的逐局结果与权重留在 eval_runs/，
# 同一份权重的评测与复测逐局完全相同（决策记录 199 补记）。贪心 RT 复测时：定时复测（没给 CK、LOCAL_CK）改测评测留存的最新一版，
# 计划外复测若这一版有评测留存的逐局结果也沿用；这 384 局放进结果目录，ability_test 跳过，只打其余 616 个种子。没有留存时照旧打满
EH=${EVAL_HOST:-a}; EVR=.local/share/sls2/runs/defect-eval/eval_runs
GREEDY_RT=""; [ "$SEEDS" = "$PN" ] && [ "$TEMP" = 0 ] && [ -z "$EXPLORE" ] && host_exists "$EH" && GREEDY_RT=1
EVAL_CK=""
[ -z "${LOCAL_CK:-}${CK:-}" ] && [ -n "$GREEDY_RT" ] && EVAL_CK=$(timeout 60 $(host_ssh "$EH") "ls -t \$HOME/$EVR/ 2>/dev/null | grep -E '^v[0-9]+\.pt\$' | head -1" 2>/dev/null)
# 计划外复测评测留档的某一版（决策记录 208）：EVAL_V=版本号，依次在 eval_runs/（最近 2 轮）、best/（历来前几名）、high/（评测偏高的）里找
#   FORCE=1 EVAL_V=117152 bash tools/remote/retest.sh
if [ -n "${EVAL_V:-}" ]; then
  [ -n "$GREEDY_RT" ] || { log "EVAL_V 只用于贪心 RT 复测"; exit 1; }
  EVAL_CK=$(timeout 60 $(host_ssh "$EH") "cd \$HOME/${EVR%/eval_runs} && for d in eval_runs best high; do [ -f \$d/v$EVAL_V.pt ] && { echo ../\$d/v$EVAL_V.pt; break; }; done" 2>/dev/null)
  [ -n "$EVAL_CK" ] || { log "评测留档里没有第 $EVAL_V 版的权重"; exit 1; }
  NORECORD=1
fi
if [ -n "${LOCAL_CK:-}" ]; then ck=$(basename "$LOCAL_CK"); v=${V:?LOCAL_CK 要同时给 V=版本号}
elif [ -n "$EVAL_CK" ]; then ck="评测留存 $EVAL_CK"; v=$(basename "$EVAL_CK" .pt); v=${v#v}
else
ck=${CK:-$(timeout 60 $SSH "ls -t \$HOME/$R/backups/ | grep -E '^v[0-9]+-[0-9]{8}-[0-9]{4}\.pt\$' | head -1")} || { log "取备份列表失败"; exit 1; }
v=${ck%%-*}; v=${v#v}; fi
TAG=""; [ -n "${NORECORD:-}" ] && TAG="-计划外"
# 并集补测（决策记录 248，tools/remote/union_backfill.sh 调）：过往快照在种子池上补打一遍，只为算快照并集；不记 state.json、不写 ALERT，
# 结果目录带“-并集补测”，不出报告、不打双 Boss 卷、不与直播基线配对
[ -n "${BACKFILL:-}" ] && { NORECORD=1; TAG="-并集补测"; }
[ "$SEEDS" != "$PN" ] && { NORECORD=1; TAG="-验收$SEEDS"; }
[ "$TEMP" != 0 ] || [ -n "$EXPLORE" ] && { NORECORD=1; TAG="$TAG-采样T$TEMP${EXPLORE:+-改选}"; }
if [ -n "$next" ]; then min=$next; else min=$((last + 1)); fi
[ -n "$v" ] && { [ -n "${CK:-}${LOCAL_CK:-}${EVAL_V:-}" ] || [ "$v" -ge "$min" ]; } || { log "最新的定时备份（第 ${v:-?} 版）还没到第 $min 版，等下一份"; exit 0; }
# 在哪台机测（决策记录 196、198）：每个分片一份 CUDA 上下文约 600 MB。先找主线机以外显存够的机器，没有才用主线机
# （观测格式 15 上线后 4090 机学习端占约 17.8 GB；刚重启时学习端还没涨上去，显存看着够，测到一半学习端涨回来就可能挤爆）；
# RT_HOST=名字 可指定。都不够就下一分钟再看
gpu_free() { timeout 60 $(host_ssh "$1") "nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1" 2>/dev/null | tr -dc 0-9; }
TH=${RT_HOST:-}
if [ -z "$TH" ]; then
  for h in $(host_active | grep -vx "$H") "$H"; do f=$(gpu_free "$h"); [ -n "$f" ] && [ "$f" -ge $((SHARDS * 800)) ] && { TH=$h; break; }; done
fi
[ -n "$TH" ] || { log "各机显存都不够 $SHARDS 个分片，等下一分钟"; exit 0; }
TSSH=$(host_ssh "$TH"); TE=$(host_rsync_e "$TH"); TAD=$(host_addr "$TH"); TD=$(host_deploy "$TH")
log "开始复测第 $v 版（$ck），主线现在第 $ver 版，在 $TH 上测"
# 测试机上跑着常规评测（tools/remote/remote_eval.py）时，测完之前不让它开新一轮（决策记录 199）：两者抢同一台机器的 CPU，
# 2026-10-30 第 85,620 版的复测和同时的一轮 384 局评测都慢了一倍多
EVD=.local/share/sls2/runs/defect-eval
timeout 60 $TSSH "[ -d $EVD ] && touch $EVD/PAUSE" || true
trap 'timeout 60 $TSSH "rm -f $EVD/PAUSE" || true' EXIT
O=/root/retest/v$v$TAG
L=$HOME/.local/share/sls2/ability_test/v$v$TAG; mkdir -p "$L"
rsync -a -e "$TE" tools/ability_test.py tools/ability_report.py "$TAD:$TD/tools/" && rsync -a -e "$TE" "$POOL" "$TAD:$TD/tools/remote/" || { log "同步测试脚本失败"; exit 1; }
# 沿用评测：留存的权重补上主线的配置，存成本机检查点按 LOCAL_CK 上传；同一版本的逐局结果存成结果目录里的 runs_seval.jsonl
if [ -n "$GREEDY_RT" ]; then
  EE=$(host_rsync_e "$EH"); EA=$(host_addr "$EH")
  if [ -n "$EVAL_CK" ]; then
    rsync -a -e "$EE" "$EA:$EVR/$EVAL_CK" "$L/eval_state.pt" && .venv/bin/python -I - "$L/eval_state.pt" "$LOC/config.json" "$v" "$L/model.pt" <<'PY' \
      || { log "评测留存的权重补配置失败"; exit 1; }
import json, re, sys, torch
st, cfgp, v, out = sys.argv[1:]
# 补上观测格式与词表大小（2026-10-31 直播会话指出：play_live 等按缺省值当成格式 8、词表 1,873，会清零部分嵌入行），取本机代码的当前值
fmt = int(re.search(r"^FORMAT_VERSION = (\d+)", open("python/sts2env/protocol.py", encoding="utf-8").read(), re.M).group(1))
vn = json.load(open("engine/catalog/v0.111.0/catalog.json", encoding="utf-8"))["vocab_size"]
torch.save({"version": int(v), "state": torch.load(st, map_location="cpu", weights_only=False), "config": json.load(open(cfgp)),
            "obs_format": fmt, "vocab_n": vn, "note": "常规评测留存的权重（defect-eval/eval_runs），配置取主线的 config.json"}, out)
PY
    rm -f "$L/eval_state.pt"; LOCAL_CK=$L/model.pt
  fi
  rsync -a -e "$EE" "$EA:$EVR/v$v.jsonl" "$L/runs_seval.jsonl" 2>/dev/null || true
  # 评测留存的逐局结果只留这次复测种子里的（换池那几轮，评测可能还是旧种子打的），一局都不剩就不沿用
  [ -f "$L/runs_seval.jsonl" ] && python3 - "$L/runs_seval.jsonl" "$POOL" "$SEEDS" "$RUNS" <<'PY'
import json, os, sys
f, pool, name, runs = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
p = json.load(open(pool, encoding="utf-8"))
keep = set(p["seeds"][:runs]) if p.get("name") == name else {f"{name}{j:04d}" for j in range(runs)}
rows = [l for l in open(f, encoding="utf-8") if l.strip() and json.loads(l)["seed"] in keep]
if rows:
    open(f, "w", encoding="utf-8").writelines(rows)
else:
    os.remove(f)
PY
  if [ -f "$L/runs_seval.jsonl" ]; then
    log "第 $v 版沿用评测留存的 $(wc -l < "$L/runs_seval.jsonl") 局，只打其余的种子"
  fi
fi
if [ -n "${LOCAL_CK:-}" ]; then
  timeout 60 $TSSH "mkdir -p $O" && rsync -a -e "$TE" "$LOCAL_CK" "$TAD:$O/model.pt" || { log "上传本机检查点失败"; exit 1; }
elif [ "$TH" = "$H" ]; then
  timeout 120 $SSH "mkdir -p $O && cp \$HOME/$R/backups/$ck $O/model.pt" || { log "复制检查点失败"; exit 1; }
else    # 租用机之间只经本机中转：主线机 → 本机 → 测试机
  rsync -a -e "$E" "$AD:$R/backups/$ck" "$L/model.pt" && timeout 60 $TSSH "mkdir -p $O" && rsync -a -e "$TE" "$L/model.pt" "$TAD:$O/model.pt" \
    || { log "中转检查点失败"; exit 1; }
fi
# 补测时本机已取回的逐局结果一并放上去，已打完的局跳过（tools/ability_test.py 按全部 runs_s*.jsonl 去重）
ls "$L"/runs_s*.jsonl > /dev/null 2>&1 && { rsync -a -e "$TE" "$L"/runs_s*.jsonl "$TAD:$O/" || { log "上传已有结果失败"; exit 1; }; }
timeout 120 $TSSH "cd $TD && export DOTNET_ROOT=\$HOME/.dotnet PATH=\$HOME/.dotnet:\$PATH DOTNET_CLI_TELEMETRY_OPTOUT=1 &&
  for k in \$(seq 0 $((SHARDS - 1))); do setsid nohup .venv/bin/python -I tools/ability_test.py --ckpt $O/model.pt --out $O --asc 10-10 --runs $RUNS --shard \$k/$SHARDS --envs $ENVS --device cuda $SEEDARG --temp $TEMP ${EXPLORE:+--explore} ${MERGE:+--merge-equiv} > $O/log_s\$k.txt 2>&1 < /dev/null & disown; done" \
  || { log "启动失败"; exit 1; }
t0=$(date +%s)
while :; do
  sleep 30
  # grep -c 没有匹配时退出码为 1，远端补 || true，只有连接失败才重试
  n=$(timeout 60 $TSSH "ps -eo args | grep -c '^.venv/bin/python -I tools/ability_test.py --ckpt $O/' || true" 2>/dev/null) || continue
  [ -n "$n" ] || continue
  [ "$n" = 0 ] && break
  [ $(( $(date +%s) - t0 )) -gt 3600 ] && { log "一小时还没打完，停下"; timeout 60 $TSSH "pkill -f '^.venv/bin/python -I tools/ability_test.py --ckpt $O/'"; break; }
done
rsync -a --exclude 'A10/' -e "$TE" "$TAD:$O/" "$L/" || { log "取回结果失败"; exit 1; }
got=$(cat "$L"/runs_s*.jsonl 2>/dev/null | wc -l)
[ "$got" -ge "$RUNS" ] || { log "只打完 $got / $RUNS 局，不记这一次"; exit 1; }
prev=$(python3 -c "import json,sys; s=json.load(open(sys.argv[1])); print(s['runs'][-1]['dir'] if s['runs'] else '')" "$ST/state.json" 2>/dev/null)
[ -n "$prev" ] || prev=$HOME/.local/share/sls2/ability_test/v66040      # 第一次复测与 v66040（决策记录 170）比
PREV_EXAM=$prev       # 双 Boss 诊断卷与种子无关，换池后照旧与上一次复测按题配对
# 验收种子与上一次同种子的验收比；采样拆分与同一版本的贪心复测比（没有就跳过配对）
[ "$SEEDS" != "$PN" ] && prev=$(ls -dt "$HOME"/.local/share/sls2/ability_test/v*-验收$SEEDS 2>/dev/null | grep -vx "$L" | head -1)
[ "$TEMP" != 0 ] || [ -n "$EXPLORE" ] && { prev=$HOME/.local/share/sls2/ability_test/v$v; [ -d "$prev" ] || prev=""; }
[ -n "$prev" ] && ! overlap "$L" "$prev" && { log "第 $v 版与上一次复测（$(basename "$prev")）没有共同的种子（换了种子池），这一次不配对"; prev=""; }
if [ -n "$prev" ]; then python3 tools/ability_compare.py "$L" "$prev" --json "$L/compare.json" >> "$ST/retest.log" 2>&1
else python3 tools/ability_compare.py "$L" --json "$L/compare.json" >> "$ST/retest.log" 2>&1; fi
if [ -z "${BACKFILL:-}" ]; then
python3 tools/ability_report.py "$L" "$L/report.html" --target "$RUNS" --asc 10-10 > /dev/null 2>&1 && cp "$L/report.html" "$W/模型能力测试报告_v$v$TAG.html"
cp "$L/compare.json" "$W/模型能力测试_v${v}${TAG}_配对比较.json" 2>/dev/null
# 双 Boss 诊断卷（tools/boss3_exam.py，决策记录 197）：还没有考卷时，用第一份记了第三幕 Boss 快照的定时复测建卷；之后每次复测
# 同一个检查点再连打一遍全部题，与上一次复测的成绩按题配对
EX=$HOME/.local/share/sls2/boss3_exam.json
if [ ! -f "$EX" ] && [ -z "${NORECORD:-}" ] && grep -q '"boss3"' "$L"/runs_s*.jsonl 2>/dev/null; then
  python3 tools/boss3_exam.py build "$L" --out "$EX" >> "$ST/retest.log" 2>&1 && cp "$EX" "$W/双Boss诊断卷.json"
fi
if [ -f "$EX" ]; then
  rsync -a -e "$TE" tools/boss3_exam.py "$TAD:$TD/tools/" && rsync -a -e "$TE" "$EX" "$TAD:/root/retest/boss3_exam.json" &&
  timeout 1800 $TSSH "cd $TD && export DOTNET_ROOT=\$HOME/.dotnet PATH=\$HOME/.dotnet:\$PATH DOTNET_CLI_TELEMETRY_OPTOUT=1 &&
    .venv/bin/python -I tools/boss3_exam.py run --ckpt $O/model.pt --exam /root/retest/boss3_exam.json --out $O/boss3.jsonl --envs $((SHARDS * ENVS)) ${MERGE:+--merge-equiv}" >> "$ST/retest.log" 2>&1 &&
  rsync -a -e "$TE" "$TAD:$O/boss3.jsonl" "$L/" || log "双 Boss 诊断卷没打完"
  [ -f "$L/boss3.jsonl" ] && [ -n "$PREV_EXAM" ] && [ -f "$PREV_EXAM/boss3.jsonl" ] && python3 tools/boss3_exam.py compare "$L/boss3.jsonl" "$PREV_EXAM/boss3.jsonl" --exam "$EX" --json "$L/boss3_compare.json" >> "$ST/retest.log" 2>&1
fi
# 再与直播在用的版本配对（直播会话要求以它作基线：10-30 00:21 切到第 84,667 版，08:30 切到第 89,500 版，22:37 切到第 103,049 版）；直播换版本时改 LIVE_BASE
# 11-01 23:04 切到第 141,229 版；决策记录 240 换种子池后，基线是这一版在种子池 P1 上的计划外复测
LIVE_BASE=${LIVE_BASE:-$HOME/.local/share/sls2/ability_test/v141229-计划外}
if [ "$SEEDS" = "$PN" ] && [ "$TEMP" = 0 ] && [ -z "$EXPLORE" ] && [ -d "$LIVE_BASE" ] && [ "$LIVE_BASE" != "$prev" ] && [ "$LIVE_BASE" != "$L" ] && ! overlap "$L" "$LIVE_BASE"; then
  echo "  直播在用版本（$(basename "$LIVE_BASE")）没有打过种子池 $SEEDS，不配对" >> "$ST/retest.log"
elif [ "$SEEDS" = "$PN" ] && [ "$TEMP" = 0 ] && [ -z "$EXPLORE" ] && [ -d "$LIVE_BASE" ] && [ "$LIVE_BASE" != "$prev" ] && [ "$LIVE_BASE" != "$L" ]; then
  echo "  对直播在用版本（$(basename "$LIVE_BASE")）：" >> "$ST/retest.log"
  python3 tools/ability_compare.py "$L" "$LIVE_BASE" --json "$L/compare_live.json" >> "$ST/retest.log" 2>&1
  [ -f "$L/boss3.jsonl" ] && [ -f "$LIVE_BASE/boss3.jsonl" ] && python3 tools/boss3_exam.py compare "$L/boss3.jsonl" "$LIVE_BASE/boss3.jsonl" --exam "$EX" --json "$L/boss3_compare_live.json" >> "$ST/retest.log" 2>&1
fi
fi
# 复现信息：检查点与引擎程序集的哈希、词表大小、测试机与显卡、分片与引擎数
eng=$(timeout 60 $TSSH "sha256sum $TD/engine/Sts2Headless/bin/Release/net9.0/Sts2Headless.dll | cut -c1-16; wc -l < $TD/engine/catalog/vocab.txt; nvidia-smi --query-gpu=name --format=csv,noheader | head -1" 2>/dev/null | paste -sd'|')
META=$(python3 -c "import json,sys; e=(sys.argv[2].split('|') + ['', '', ''])[:3]; print(json.dumps({'ckpt': sys.argv[1], 'ckpt_sha256': sys.argv[3], 'engine_sha256': e[0], 'vocab': e[1], 'gpu': e[2], 'host': sys.argv[4], 'shards': int(sys.argv[5]), 'envs': int(sys.argv[6])}, ensure_ascii=False))" \
  "$ck" "$eng" "$(sha256sum "$L/model.pt" 2>/dev/null | cut -c1-16)" "$TH" "$SHARDS" "$ENVS")
echo "$META" > "$L/meta.json"
if [ -n "${NORECORD:-}" ]; then log "完成计划外复测第 $v 版（不记入 state.json）：$(python3 -c "import json,sys; c=json.load(open(sys.argv[1])); m=c['models'][0]; p=(c['pairs'] or [None])[0]; print(f\"通关 {m['win_rate']:.1%}\", p and {'vs': p['b'], 'only_this': p['only_a'], 'only_prev': p['only_b'], 'p': round(p['p'], 4)})" "$L/compare.json")"; exit 0; fi
SEEDS=$SEEDS python3 - "$ST/state.json" "$L" "$v" "$ST" "$META" <<'EOF'
import json, math, os, sys
path, d, v, st, meta = sys.argv[1:]
s = json.load(open(path)) if os.path.exists(path) else {"runs": []}
c = json.load(open(f"{d}/compare.json"))
m = c["models"][0]; p = c["pairs"][0] if c["pairs"] else None
n = c["seeds"]; w = m["win_rate"]; half = 1.96 * math.sqrt(w * (1 - w) / max(n, 1))
rec = {"version": int(v), "dir": d, "seeds": n, "win_rate": round(w, 4), "pm": round(half, 4), "floor_mean": round(m["floor_mean"], 2),
       "win_given_boss2": round(m["win_given_boss2"], 4), "meta": json.loads(meta), "seed_pool": os.environ.get("SEEDS", "RT")}
if p:
    rec["vs"] = {"version": p["b"], "only_this": p["only_a"], "only_prev": p["only_b"], "p": round(p["p"], 4)}
    if p["only_b"] > p["only_a"] and p["p"] < 0.05:
        open(f"{st}/ALERT", "a").write(json.dumps(rec, ensure_ascii=False) + "\n")
s["runs"].append(rec)
s.pop("next_at", None)
json.dump(s, open(path, "w"), ensure_ascii=False, indent=1)
print(json.dumps(rec, ensure_ascii=False))
EOF
log "完成第 $v 版：$(tail -1 "$ST/state.json" >/dev/null; python3 -c "import json,sys; r=json.load(open(sys.argv[1]))['runs'][-1]; print(f\"通关 {r['win_rate']:.1%} ± {r['pm']:.1%}\", r.get('vs', ''))" "$ST/state.json")"
# 复测快照的通关并集（决策记录 247）：同一种子池的历次定时复测里，至少一个快照通关、全部通关、都没通关的种子数
python3 tools/retest_union.py >> "$ST/retest.log" 2>&1 || log "并集统计失败"
