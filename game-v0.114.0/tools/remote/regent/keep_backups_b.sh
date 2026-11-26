#!/usr/bin/env bash
# 储君：B 定时备份的长期留存（2026-11-09，储君决策记录第 24 条补记；为快照集成、蒸馏与权重平均备料）。
# ckpt_backup.sh 每 30 分钟把 latest.pt 复制进 backups/（约每 250 版一份，含原始权重 state、学习端滑动平均 ema、优化器 opt），
# 但只留最近 96 份（约 48 小时），第 21 条（第 43,999 版，学习率 2e-4）以来的检查点会陆续被轮换删掉。
# 这里把第 FROM 版起的备份硬链接进 KEEP 目录：同一文件系统，不占额外空间；轮换删掉 backups/ 那一份后，KEEP 里这份留下。
# 备份先写 .part 再改名，这里只匹配改名后的 v*.pt。B 的 crontab 每 30 分钟跑一次（与备份错开 15 分钟）：
#   15,45 * * * * bash /root/regent-deploy/sls2/tools/remote/regent/keep_backups_b.sh
set -euo pipefail
R=${RUN_DIR:-/root/.local/share/sls2/runs/regent}
K=${KEEP_DIR:-$R/keep_ens_b}
FROM=${FROM:-43998}
mkdir -p "$K"
for f in "$R"/backups/v*.pt; do
  [ -e "$f" ] || continue
  b=$(basename "$f"); v=${b#v}; v=${v%%-*}
  [[ "$v" =~ ^[0-9]+$ ]] && [ "$v" -ge "$FROM" ] || continue
  [ -e "$K/$b" ] && continue
  ln "$f" "$K/$b" 2>/dev/null || continue
  c=${f%.pt}.train_config.json
  [ -e "$c" ] && ln -f "$c" "$K/${b%.pt}.train_config.json" 2>/dev/null || true
  echo "$(date '+%F %T') 留存 $b" >> "$K/keep.log"
done
