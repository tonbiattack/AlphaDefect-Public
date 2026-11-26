#!/usr/bin/env bash
# B 上暂停训练后的千种池复测（2026-11-20，铁甲战士决策记录第 24 条补记；用户：“暂停训练，测一下滑动和集成胜率”）。
# 在 B 上、训练与常规评测都停了以后跑（运行目录有 STOP、学习进程与常规评测都不在，否则不测）：
#   V5：最新检查点（停训时存的 latest.pt）的原始权重；M5：它的滑动平均；
#   E5：keep/ 里第 20,500 版到最新版之间按版本均匀挑的 16 个备份的原始权重集成（同 E3 的挑法，pick_span.py）；
#   R5：加血后（第 24,281 版起，第 24 条）的备份全部（最多 8 个，按版本均匀挑）加最新检查点的原始权重集成。
# 种子、截断、分片同第 12、14、17 条（W1 前 1,000 个，ens_test.sh 6 × 16），再与 V/M/E、V2/M2/E2、V3/M3/E3 同种子逐局配对（cmp_arms.py → cmp6.txt）。
# 测试目录 /root/ens_ironclad 由撤离存档 /root/from_c/root/ens_ironclad 建（检查点 ck/ 硬链接，其余实拷），旧组的结果不动。
#   setsid nohup bash /root/ironclad-deploy/sls2/tools/remote/ironclad/finish/retest_b.sh > /root/ens_ironclad/run6.out 2>&1 < /dev/null &
set -u
T=/root/ens_ironclad
RUN=/root/.local/share/sls2/runs/ironclad
D=/root/ironclad-deploy/sls2
F=$D/tools/remote/ironclad/finish
log() { echo "$(date '+%F %T') $*" | tee -a "$T/log.txt"; }
exec 8>"$T/.retest6.lock"; flock -n 8 || exit 0
[ -f "$T/retest6.done" ] && exit 0
[ -f "$RUN/STOP" ] || { log "复测：运行目录没有 STOP（训练没停），不测"; exit 0; }
P=$(cat "$RUN/learner.pid" 2>/dev/null || true)
[ -n "$P" ] && kill -0 "$P" 2>/dev/null && { log "复测：学习进程 $P 还在，不测"; exit 0; }
pgrep -f '[r]emote_eval.py --config /root/ironclad_train_config.json' >/dev/null && { log "复测：常规评测还在，不测"; exit 0; }
cd "$D" || exit 1
PY="nice -n 5 $D/.venv/bin/python -I"
VF=$($PY -c "import sys, torch; print(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['version'])" "$RUN/latest.pt") || { log "复测：读不出 latest.pt 的版本"; exit 1; }
mapfile -t K < <(python3 -I "$F/pick_span.py" 20500 "$VF" 16 2>> "$T/log.txt")
mapfile -t KR < <(python3 -I "$F/pick_span.py" 24281 "$VF" 8 2>> "$T/log.txt")
log "复测：最新检查点第 $VF 版；E5 成员 ${K[*]##*/}；R5 成员 ${KR[*]##*/} 加最新"
$PY "$F/ext_members.py" "$T/ck" "$RUN/latest.pt" "${K[@]}" "${KR[@]}" 2>&1 | tee -a "$T/log.txt"
vof() { echo "$T/ck/v$(basename "$1" | sed 's/^v\([0-9]*\)-.*/\1/').pt"; }
echo "$T/ck/v$VF.pt" > "$T/arms/V5.txt"
echo "$T/ck/m$VF.pt" > "$T/arms/M5.txt"
: > "$T/arms/E5.txt"; for k in "${K[@]}"; do vof "$k" >> "$T/arms/E5.txt"; done
echo "$T/ck/v$VF.pt" > "$T/arms/R5.txt"; for k in "${KR[@]}"; do vof "$k" >> "$T/arms/R5.txt"; done
awk '!s[$0]++' "$T/arms/R5.txt" > "$T/arms/R5.tmp" && mv "$T/arms/R5.tmp" "$T/arms/R5.txt"
ARMS="V5 M5 E5 R5" T=$T bash "$F/ens_test.sh"
python3 -I "$F/cmp_arms.py" "$T" V M E V2 M2 E2 V3 M3 E3 V5 M5 E5 R5 > "$T/cmp6.txt" 2>&1
log "复测完成：$(grep -E '^(V5|M5|E5|R5):' "$T/cmp6.txt" | tr '\n' ' ')"
date '+%F %T' > "$T/retest6.done"
