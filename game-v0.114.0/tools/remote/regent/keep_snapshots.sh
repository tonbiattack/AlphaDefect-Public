#!/usr/bin/env bash
# 储君快照留存（2026-11-09，储君决策记录第 24 条；用户：“后面可能就是快照集成蒸馏，再配上搜索形成最后的模型。”）
# 常规评测只在 eval_runs/ 留最近两轮的权重（v<版本>.pt，原始权重、只含模型），B 的定时备份只留约 48 小时。
# 这里每隔 GAP 版（默认 500，约 1 小时）把一份评测留存的权重复制到 KEEP 目录，不参与轮换，供日后的快照集成与蒸馏挑选。
# 只复制写完至少 1 分钟的文件（评测在每轮开头写权重），先写 .part 再改名。A 的 crontab 每 10 分钟跑一次：
#   */10 * * * * bash /root/regent-deploy/sls2/tools/remote/regent/keep_snapshots.sh
set -euo pipefail
E=${EVAL_DIR:-/root/.local/share/sls2/runs/regent-eval}
K=${KEEP_DIR:-$E/keep_ens}
GAP=${GAP:-500}
mkdir -p "$K"
last=$(ls "$K" | sed -n 's/^v\([0-9]*\)\.pt$/\1/p' | sort -n | tail -1)
for v in $(find "$E/eval_runs" -maxdepth 1 -name 'v*.pt' -mmin +1 -printf '%f\n' | sed -n 's/^v\([0-9]*\)\.pt$/\1/p' | sort -n); do
  if [ -z "$last" ] || [ "$v" -ge $((last + GAP)) ]; then
    cp "$E/eval_runs/v$v.pt" "$K/v$v.pt.part" && mv "$K/v$v.pt.part" "$K/v$v.pt"
    echo "$(date '+%F %T') 留存 v$v" >> "$K/keep.log"
    last=$v
  fi
done
