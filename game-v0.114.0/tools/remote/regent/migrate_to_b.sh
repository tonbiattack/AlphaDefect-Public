#!/usr/bin/env bash
# 本机运行：储君训练从 A 单机迁回原结构——B 学习 + 本机采集，A 远程采集与常规评测（2026-11-06，用户：“后面 B 机器也会空出来。
# 我之后还是希望之前的那样的训练结构。所以说 A 作为主要机器只是暂时的。”）。
# 步骤：B 空闲检查 → 代码同步到 B 并编译 → 停 A 的学习进程（先存检查点）、撤掉 A 上储君的 crontab、备份一次（TAG=迁移前）
#       → B 直接从 A 取检查点与记录（B 能连 sls2-a，不经本机中转）→ A 的评测记录挪到 regent-eval（评测据 eval.jsonl 跳过已评过的版本，看板评测历史接上）
#       → start.sh 原结构（B 学习、A 采集与评测、看板目标改回 B 学习 A 评测）。
#   bash tools/remote/regent/migrate_to_b.sh            # B 上有训练、评测、引擎进程时拒绝
#   FORCE=1 bash tools/remote/regent/migrate_to_b.sh    # 不查 B 是否空闲
# B 的运行目录已有内容时改名为 regent.before-migrate-<时刻> 保留，不删除。A 的运行目录留着（远程采集也写在这里），STOP 留在里面防误启动。
set -euo pipefail
. "$(dirname "$0")/../hosts.sh"
D=/root/regent-deploy/sls2
RUN=/root/.local/share/sls2/runs/regent
EVAL=/root/.local/share/sls2/runs/regent-eval
HERE=$(cd "$(dirname "$0")" && pwd)

echo "== 检查 B"
n=$($(host_ssh b) 'pgrep -fc "[s]im_pimc_eval|[t]rain_defect.py|[r]emote_actors.py|[r]emote_eval.py|Sts2SimServe.dll [s]erve|Sts2Headless.dll [s]erve"' || true)
if [ "${n:-0}" != 0 ] && [ -z "${FORCE:-}" ]; then die "B 上还有 $n 个训练、评测或引擎进程，不迁（确认可以挤时 FORCE=1）"; fi

echo "== 代码同步到 B 并编译"
bash "$HERE/deploy.sh" b

echo "== 停 A 的学习进程"
$(host_ssh a) "set -e; cd $RUN; touch STOP
  if [ -f learner.pid ] && P=\$(cat learner.pid) && kill -0 \$P 2>/dev/null; then
    kill -TERM \$P; for i in \$(seq 90); do kill -0 \$P 2>/dev/null || break; sleep 2; done
    kill -0 \$P 2>/dev/null && { echo 学习进程停不下来; exit 1; }
  fi
  ( crontab -l 2>/dev/null | grep -v 'regent-deploy/sls2/tools/remote/' || true ) | crontab -
  TAG=迁移前 RUN_DIR=$RUN bash $D/tools/remote/ckpt_backup.sh
  python3 -c 'import json; print(\"检查点版本\", json.load(open(\"status.json\")).get(\"version\"))' 2>/dev/null || true"

echo "== B 从 A 取检查点与记录"
$(host_ssh b) "set -e
  if [ -d $RUN ] && [ -n \"\$(ls -A $RUN 2>/dev/null)\" ]; then mv $RUN $RUN.before-migrate-\$(date +%s); fi
  mkdir -p $RUN
  rsync -a --ignore-missing-args sls2-a:$RUN/latest.pt sls2-a:$RUN/status.json sls2-a:$RUN/log.jsonl sls2-a:$RUN/replays sls2-a:$RUN/solver $RUN/
  ls -la $RUN"

echo "== A：评测记录挪到 regent-eval"
$(host_ssh a) "set -e; mkdir -p $EVAL; cd $RUN
  for f in eval.jsonl best high eval_runs; do [ -e \$f ] && [ ! -e $EVAL/\$f ] && mv \$f $EVAL/; done; ls $EVAL"

echo "== 按原结构启动"
LAYOUT=b bash "$HERE/start.sh"
