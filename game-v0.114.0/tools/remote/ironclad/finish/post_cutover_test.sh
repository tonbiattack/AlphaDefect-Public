#!/usr/bin/env bash
# 撤离后千种池复测（2026-11-20，铁甲战士决策记录第 17 条；用户：“撤离后测一下，数据你对话里告诉我就行”）。
# B 上的 c_cutover.sh 在 11-20 13:00（C、B 机时间）停掉 C 上的训练与常规评测后，由 C 上的一次性定时任务（13:15，13:45 兜底）跑本脚本：
#   V3：最终检查点（停机时存的 latest.pt）的原始权重；M3：它的滑动平均；
#   E3：keep/ 里第 20,500 版到最终版之间按版本均匀挑的 16 个备份的原始权重集成（pick_span.py；第 18 条测出跨整个平台的集成比单个网络高约 5 个百分点，
#       只取最近 600 版的 8 个则没有增益）；
#   种子、截断、分片同第 12、14 条（W1 前 1,000 个，ens_test.sh 6 × 16），再与 V/M/E、V2/M2/E2 同种子逐局配对比较（cmp_arms.py → cmp3.txt）。
# 只在训练确实停了（运行目录有 STOP、学习进程与常规评测都不在）时跑；跑完写 post_cutover.done，不重跑。
# 撤离会撤掉本线的定时任务（引擎看门狗也在内），测试期间本脚本每分钟自己调一次 engine_watchdog.sh。
#   15 13 20 11 * bash /root/ens_ironclad/post_cutover_test.sh >> /root/ens_ironclad/post_cutover.cron.out 2>&1
set -u
T=/root/ens_ironclad
RUN=/root/.local/share/sls2/runs/ironclad
D=/root/ironclad-deploy/sls2
log() { echo "$(date '+%F %T') $*" | tee -a "$T/log.txt"; }
exec 8>"$T/.post_cutover.lock"; flock -n 8 || exit 0
[ -f "$T/post_cutover.done" ] && exit 0
[ -f "$RUN/STOP" ] || { log "撤离后复测：运行目录没有 STOP（训练没停），这次不测"; exit 0; }
P=$(cat "$RUN/learner.pid" 2>/dev/null || true)
[ -n "$P" ] && kill -0 "$P" 2>/dev/null && { log "撤离后复测：学习进程 $P 还在，这次不测"; exit 0; }
pgrep -f '[r]emote_eval.py --config /root/ironclad_train_config.json' >/dev/null && { log "撤离后复测：常规评测还在，这次不测"; exit 0; }
cd "$D" || exit 1
PY="nice -n 5 $D/.venv/bin/python -I"
F=$RUN/latest.pt
VF=$($PY -c "import sys, torch; print(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['version'])" "$F") || { log "撤离后复测：读不出 $F 的版本"; exit 1; }
mapfile -t K < <(python3 -I "$T/pick_span.py" 20500 "$VF" 16 2>> "$T/log.txt")
log "撤离后复测：最终检查点第 $VF 版；集成成员 ${K[*]##*/}"
$PY "$T/ext_members.py" "$T/ck" "$F" "${K[@]}" 2>&1 | tee -a "$T/log.txt"
echo "$T/ck/v$VF.pt" > "$T/arms/V3.txt"
echo "$T/ck/m$VF.pt" > "$T/arms/M3.txt"
: > "$T/arms/E3.txt"
for k in "${K[@]}"; do echo "$T/ck/v$(basename "$k" | sed 's/^v\([0-9]*\)-.*/\1/').pt" >> "$T/arms/E3.txt"; done
( while true; do bash "$D/tools/remote/ironclad/engine_watchdog.sh"; sleep 60; done ) & WD=$!
ARMS="V3 M3 E3" bash "$T/ens_test.sh"
kill "$WD" 2>/dev/null
python3 -I "$T/cmp_arms.py" "$T" V M E V2 M2 E2 V3 M3 E3 > "$T/cmp3.txt" 2>&1
log "撤离后复测完成：$(grep -E '^(V3|M3|E3):' "$T/cmp3.txt" | tr '\n' ' ')"
date '+%F %T' > "$T/post_cutover.done"
