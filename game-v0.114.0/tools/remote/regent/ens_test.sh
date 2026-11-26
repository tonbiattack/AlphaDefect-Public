#!/usr/bin/env bash
# 储君快照集成测试（2026-11-09，储君决策记录第 24 条补记；用户：“先测一下集成胜率吧。”）
# 在 A 上跑：各组在同一批种子（W1 前 RUNS 个，储君没用过）上纯贪心打进阶 10，带常规评测同样的截断开关
# （局面重复截断、20 回合无进展判输），组与组依次跑，同种子逐局配对比较。成员是 B 的检查点抽出的原始权重，放在 $T/ck/：
#   S   单个第 53,840 版（原始权重）
#   E4  4 个、约每 500 版：53,840 / 52,320 / 52,840 / 53,340（照故障机器人第 266 条补记二的 4 个相近快照）
#   E8  8 个、第 23 条以来约每 1,000 版：53,840 / 47,300 / 48,320 / 49,320 / 50,300 / 51,320 / 52,320 / 53,080
#   M   第 53,840 版的学习端滑动平均权重（衰减 0.999）
#   EMA4 4 份滑动平均权重的集成，约每 2,000 版：54,603 / 52,580 / 50,540 / 48,560（第 24 条补记“滑动平均后集成”，2026-11-09 加；成员文件 e<版本>.pt）
# 集成为每步概率平均（tools/ability_test.py 的 EnsembleNet，prob）。常规评测先放 PAUSE 让路：一轮打完后它的引擎都退出，
# 8 个空闲工作进程只占约 8 GB，A 可用约 16 GB，够 4 个分片 × 16 个引擎（同时 64 局，与常规评测一样）。全部测完后本脚本删掉 PAUSE，
# 常规评测接着跑；它若已不在，照 start.sh 重新拉起。
#   setsid nohup bash /root/regent-deploy/sls2/tools/remote/regent/ens_test.sh > /root/ens_test/run.out 2>&1 < /dev/null &
# 已打完的局会跳过（ability_test 按 runs_s*.jsonl 去重），加大 RUNS 重跑即在后面补种子。
set -u
T=${T:-/root/ens_test}; CK=$T/ck
D=/root/regent-deploy/sls2
EVAL=/root/.local/share/sls2/runs/regent-eval
RUNS=${RUNS:-1000}; SHARDS=${SHARDS:-4}; ENVS=${ENVS:-16}; ARMS=${ARMS:-"S E4 E8 M"}
EXTRA='{"loop_cut": true, "stall_turns": 20}'
log() { echo "$(date '+%F %T') $*" | tee -a "$T/log.txt"; }
# 与常规评测一样限在 20 个核以内（cgroup），其余留给远程采集
G=/sys/fs/cgroup/sls2-enstest
{ mkdir -p $G && echo "2000000 100000" > $G/cpu.max && echo $$ > $G/cgroup.procs; } || log "cgroup 限核失败，照常跑"
python3 - "$D/tools/remote/seed_pool_w1.json" "$T/seeds_w1.json" "$RUNS" <<'EOF'
import json, sys
s = json.load(open(sys.argv[1]))["seeds"][:int(sys.argv[3])]
json.dump({"name": "W1", "seeds": s}, open(sys.argv[2], "w"))
EOF
members() {
  case $1 in
    S) echo v53840 ;;
    M) echo v53840ema ;;
    E4) echo v53840 v52320 v52840 v53340 ;;
    E8) echo v53840 v47300 v48320 v49320 v50300 v51320 v52320 v53080 ;;
    EMA4) echo e54603 e52580 e50540 e48560 ;;
  esac
}
cd $D || exit 1
for arm in $ARMS; do
  set -- $(members $arm); first=$1; shift
  O=$T/$arm; mkdir -p $O; : > $O/members.txt
  for m in "$@"; do echo "$CK/$m.pt" >> $O/members.txt; done
  ENS=""; [ $# -gt 0 ] && ENS="--ensemble-file $O/members.txt"
  log "开始 $arm（$first $*）"
  for k in $(seq 0 $((SHARDS - 1))); do
    nice -n 5 .venv/bin/python -I tools/ability_test.py --ckpt "$CK/$first.pt" $ENS --out $O --asc 10-10 --runs $RUNS --shard $k/$SHARDS \
      --envs $ENVS --device cuda --seeds-file "$T/seeds_w1.json" --character REGENT --time-cap 1200 --req-extra "$EXTRA" \
      > $O/log_s$k.txt 2>&1 < /dev/null &
  done
  wait
  log "完成 $arm：$(cat $O/runs_s*.jsonl | wc -l) 局，通关 $(cat $O/runs_s*.jsonl | grep -c '"win": true')"
done
# 重新拉起常规评测（与 tools/remote/regent/start.sh 相同的命令）
rm -f $EVAL/PAUSE
if pgrep -f '[r]emote_eval.py --config /root/regent_train_config.json' > /dev/null; then log "常规评测已在跑"
else
  setsid nohup .venv/bin/python -I $D/tools/remote/remote_eval.py --config /root/regent_train_config.json --run-dir $EVAL \
    --weights /dev/shm/sts2rl_regent.pt --seeds-file $D/tools/remote/seed_pool.json --workers 8 --envs 8 --device cuda --cores 20 --nice 5 \
    --keep-runs 20 --no-replay-all >> $EVAL/remote_eval.out 2>&1 < /dev/null &
  log "常规评测已重新拉起"
fi
