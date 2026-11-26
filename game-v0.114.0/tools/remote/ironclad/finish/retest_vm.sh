#!/usr/bin/env bash
# B 上停训后的千种池复测，只测最新检查点的原始权重与滑动平均两组（2026-11-20，铁甲战士决策记录第 26 条满 500 版的主判据）。
# 组名 V<N>、M<N>（N 由第一个参数给），种子、截断、分片同第 12、14、17 条（W1 前 1,000 个，ens_test.sh 6 × 16），
# 与之前各组同种子逐局配对（cmp_arms.py → cmp<N>.txt）。只在训练与常规评测都停了时跑。
#   setsid nohup bash /root/ironclad-deploy/sls2/tools/remote/ironclad/finish/retest_vm.sh 6 > /root/ens_ironclad/run_vm6.out 2>&1 < /dev/null &
set -u
N=${1:?组号}
T=/root/ens_ironclad
RUN=/root/.local/share/sls2/runs/ironclad
D=/root/ironclad-deploy/sls2
F=$D/tools/remote/ironclad/finish
log() { echo "$(date '+%F %T') $*" | tee -a "$T/log.txt"; }
[ -f "$T/retest_vm$N.done" ] && exit 0
[ -f "$RUN/STOP" ] || { log "复测 $N：运行目录没有 STOP，不测"; exit 0; }
P=$(cat "$RUN/learner.pid" 2>/dev/null || true)
[ -n "$P" ] && kill -0 "$P" 2>/dev/null && { log "复测 $N：学习进程 $P 还在，不测"; exit 0; }
pgrep -f '[r]emote_eval.py --config /root/ironclad_train_config.json' >/dev/null && { log "复测 $N：常规评测还在，不测"; exit 0; }
cd "$D" || exit 1
PY="nice -n 5 $D/.venv/bin/python -I"
VF=$($PY -c "import sys, torch; print(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['version'])" "$RUN/latest.pt") || exit 1
log "复测 $N：最新检查点第 $VF 版（V$N 原始权重、M$N 滑动平均）"
$PY "$F/ext_members.py" "$T/ck" "$RUN/latest.pt" 2>&1 | tee -a "$T/log.txt"
echo "$T/ck/v$VF.pt" > "$T/arms/V$N.txt"; echo "$T/ck/m$VF.pt" > "$T/arms/M$N.txt"
ARMS="V$N M$N" T=$T bash "$F/ens_test.sh"
python3 -I "$F/cmp_arms.py" "$T" V M V2 M2 V3 M3 E3 V5 M5 E5 "V$N" "M$N" > "$T/cmp$N.txt" 2>&1
log "复测 $N 完成：$(grep -E "^(V$N|M$N):" "$T/cmp$N.txt" | tr '\n' ' ')"
date '+%F %T' > "$T/retest_vm$N.done"
