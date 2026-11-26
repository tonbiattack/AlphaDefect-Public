#!/usr/bin/env bash
# 战斗结果模型方案（reports/战斗结果模型方案.md）停训期间在 B 上一次跑完：攒数据 → 训两种模型 → 复测数据 → 出预测。
# 训练与常规评测须先停（stop.sh），本脚本不碰训练、配置与部署目录，输出都在 /root/cmbpred。
#   cd /root/cmbpred && setsid nohup bash cmbpred_pause.sh > pause_run.out 2>&1 < /dev/null &
set -u
cd /root/cmbpred
PY=/root/silent-deploy/sls2/.venv/bin/python
log() { echo "$(date '+%F %T') $*" | tee -a pause_run.log; }
Q='{"0:Monster": 30, "0:Elite": 30, "0:Boss": 30, "1:Monster": 30, "1:Elite": 30, "1:Boss": 30, "2:Monster": 30, "2:Elite": 30, "2:Boss": 30, "2:Boss:second": 30}'
if pgrep -f '[t]rain_defect.py --config /root/.local/share/sls2/runs/silent' >/dev/null || pgrep -f '[r]emote_eval.py --config /root/silent_train_config.json' >/dev/null; then
  log "猎人训练或评测还在跑，先用 stop.sh 停"; exit 1
fi
log "1 D1 真打 10,000 局（存开战帧与前后快照）＋ T2 来源整局 600 局"
for s in 11 12 13 14; do
  $PY -I cmbpred_check.py runs --frames --runs 2500 --envs 40 --threads 2 --device cuda --seed $s --out d1_$s.jsonl.gz > d1_$s.out 2>&1 &
done
$PY -I cmbpred_check.py runs --runs 600 --envs 24 --threads 2 --device cuda --seed 21 --out t2src.jsonl > t2src.out 2>&1 &
wait
log "1 完成：$(for s in 11 12 13 14; do tail -n1 d1_$s.out; done | tr '\n' ' ')"
D1=d1_11.jsonl.gz,d1_12.jsonl.gz,d1_13.jsonl.gz,d1_14.jsonl.gz
log "2 D2 改牌组单打 100,000 场"
for s in 0 1 2 3; do
  $PY -I cmbpred_check.py perturb --src $D1 --n 100000 --hard-share 0.7 --envs 48 --threads 2 --device cuda --seed 31 --shard $s/4 --out d2_$s.jsonl.gz > d2_$s.out 2>&1 &
done
wait
log "2 完成：$(for s in 0 1 2 3; do tail -n1 d2_$s.out; done | tr '\n' ' ')"
D2=d2_0.jsonl.gz,d2_1.jsonl.gz,d2_2.jsonl.gz,d2_3.jsonl.gz
log "3 训练两种模型 ＋ T1 开战帧 ＋ T2 留出集"
$PY -I combat_model.py train --d1 $D1 --d2 $D2 --kind frozen --device cuda --batch 512 --epochs 6 --patience 2 --lr-head 1e-3 --workers 6 --out m_frozen.pt > m_frozen.out 2>&1 &
$PY -I combat_model.py train --d1 $D1 --d2 $D2 --kind full --device cuda --batch 512 --epochs 6 --patience 2 --lr-head 1e-3 --lr-base 1e-4 --workers 6 --out m_full.pt > m_full.out 2>&1 &
for s in 0 1; do
  $PY -I cmbpred_check.py cf --runs-file runs.jsonl,runs_2.jsonl,runs_3.jsonl,runs_4.jsonl,runs_5.jsonl --quota "$Q" --k 24 --frames-only 8 \
    --envs 32 --threads 2 --device cuda --seed 7 --shard $s/2 --out t1f_$s.jsonl > t1f_$s.out 2>&1 &
done
for s in 0 1 2 3; do
  $PY -I cmbpred_check.py cf --runs-file t2src.jsonl --quota "$Q" --k 24 --save-frames \
    --envs 32 --threads 2 --device cuda --seed 41 --shard $s/4 --out t2_$s.jsonl > t2_$s.out 2>&1 &
done
wait
log "3 完成：$(tail -n1 m_frozen.out) | $(tail -n1 m_full.out) | $(tail -qn1 t1f_0.out t1f_1.out t2_0.out | tr '\n' ' ')"
log "4 出预测"
for M in frozen full; do
  $PY -I combat_model.py predict --model m_$M.pt --frames t1f_0.jsonl,t1f_1.jsonl --device cuda --batch 512 --out p_t1_$M.jsonl > p_$M.out 2>&1
  $PY -I combat_model.py predict --model m_$M.pt --frames t2_0.jsonl,t2_1.jsonl,t2_2.jsonl,t2_3.jsonl --device cuda --batch 512 --out p_t2_$M.jsonl >> p_$M.out 2>&1
  $PY -I combat_model.py predict --model m_$M.pt --frames $D1 --val-only --device cuda --batch 512 --out p_d1_$M.jsonl >> p_$M.out 2>&1
done
log "4 完成：$(cat p_frozen.out p_full.out | tr '\n' ' ')"
log "全部完成，可以恢复训练"
