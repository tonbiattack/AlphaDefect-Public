#!/usr/bin/env bash
# 本机运行：两个检查点在确认卷上做配对序贯检验（后续计划第九节、决策记录 219）。确认卷 CF0000–CF9999 从不用来挑版本，
# 只用于接受训练改动、换直播模型、对外报数。每批两版各打 BATCH 个新种子（tools/ability_test.py 按已有结果去重，--runs 逐批加大），
# 打完一批取回、用 tools/sprt.py 算对数似然比，越界即停；最多 MAX 个种子。测试期间 A 的常规评测停掉（腾出内存，
# A 的余量只够评测与测试二选一），结束时用 /root/a_eval_restart.sh 拉起；采集照跑。
# 只比较同一观测格式下训练的检查点（现在是第 120,904 版以后的格式 18）：旧格式的检查点在新引擎上会看到没训练过的记录。
#   nohup bash tools/remote/sprt.sh 新版.pt 旧版.pt [标签] > 日志 2>&1 &      # 检查点为本机路径（带 config 的包装格式）
# 环境变量：SEEDS（默认 CF）、BATCH（400）、MAX（8000）、DELTA（0.02）、SHARDS（每版的分片数，2）、ENVS（每片引擎数，20）、TH（主机，a）
set -u
cd "$(dirname "$0")/../.." && . tools/remote/hosts.sh
NEW=$1; OLD=$2; TAG=${3:-$(basename "$NEW" .pt)-对-$(basename "$OLD" .pt)}
SEEDS=${SEEDS:-CF}; BATCH=${BATCH:-400}; MAX=${MAX:-8000}; DELTA=${DELTA:-0.02}; SHARDS=${SHARDS:-2}; ENVS=${ENVS:-20}; TH=${TH:-a}
S=$(host_ssh "$TH"); E=$(host_rsync_e "$TH"); AD=$(host_addr "$TH"); D=$(host_deploy "$TH")
O=/root/sprt/$TAG-$SEEDS; L=$HOME/.local/share/sls2/ability_test/sprt/$TAG-$SEEDS; mkdir -p "$L/new" "$L/old"
EVD=.local/share/sls2/runs/defect-eval
log() { echo "$(date '+%F %T') $*" | tee -a "$L/sprt.log"; }
$S "mkdir -p $O/new $O/old && touch $EVD/PAUSE; desc() { for c in \$(pgrep -P \$1); do echo \$c; desc \$c; done; }
  for p in \$(pgrep -f '^\.venv/bin/python -I tools/remote/remote_eval\.py'); do kill -9 \$p \$(desc \$p) 2>/dev/null; done; true" || { log "连不上 $TH"; exit 1; }
trap '$S "rm -f $EVD/PAUSE; bash /root/a_eval_restart.sh" > /dev/null 2>&1 || true; log "常规评测已恢复"' EXIT
rsync -a -e "$E" "$NEW" "$AD:$O/new/model.pt" && rsync -a -e "$E" "$OLD" "$AD:$O/old/model.pt" && rsync -a -e "$E" tools/ability_test.py "$AD:$D/tools/" || { log "上传失败"; exit 1; }
log "开始：$TAG，种子 $SEEDS，每批 $BATCH，上限 $MAX，δ = $DELTA，$TH 上每版 $SHARDS × $ENVS 个引擎"
N=0
while [ "$N" -lt "$MAX" ]; do
  N=$((N + BATCH))
  $S "cd $D && export DOTNET_ROOT=\$HOME/.dotnet PATH=\$HOME/.dotnet:\$PATH DOTNET_CLI_TELEMETRY_OPTOUT=1 &&
    for m in new old; do for k in \$(seq 0 $((SHARDS - 1))); do setsid nohup .venv/bin/python -I tools/ability_test.py --ckpt $O/\$m/model.pt --out $O/\$m --asc 10-10 --runs $N --shard \$k/$SHARDS --envs $ENVS --device cuda --seed-prefix $SEEDS >> $O/\$m/log_s\$k.txt 2>&1 < /dev/null & done; done" || { log "启动失败"; exit 1; }
  t0=$(date +%s)
  while :; do
    sleep 20
    n=$(timeout 60 $S "ps -eo args | grep -c '^.venv/bin/python -I tools/ability_test.py --ckpt $O/' || true" 2>/dev/null) || continue
    [ -n "$n" ] && [ "$n" = 0 ] && break
    [ $(( $(date +%s) - t0 )) -gt 2400 ] && { log "一批 40 分钟没打完，停下"; $S "pkill -f '^.venv/bin/python -I tools/ability_test.py --ckpt $O/'"; exit 1; }
  done
  rsync -a --exclude 'A10/' --exclude 'model.pt' -e "$E" "$AD:$O/new/" "$L/new/" && rsync -a --exclude 'A10/' --exclude 'model.pt' -e "$E" "$AD:$O/old/" "$L/old/" || { log "取回失败"; exit 1; }
  out=$(python3 tools/sprt.py "$L/new" "$L/old" --delta "$DELTA" --json "$L/sprt.json")
  log "第 $N 个种子（$(( ($(date +%s) - t0) / 60 )) 分钟一批）：$out"
  case "$out" in *接受*) break ;; esac
done
log "结束"
