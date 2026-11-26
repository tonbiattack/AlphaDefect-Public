#!/usr/bin/env bash
# 每 30 分钟由 cron 调用：把训练检查点和当时的配置复制到 backups/，文件名带版本号与时刻，保留最近 96 份（约 48 小时）。
# 检查点由训练进程先写临时文件再改名，复制时不会读到写了一半的文件。不改动训练本身。
#   */30 * * * * bash /root/sls2-deploy/sls2/tools/remote/ckpt_backup.sh
# 重启训练前（训练进程已停、检查点已存好）手动调用时传 TAG，文件名加上它；此时若定时备份正在跑，等它结束而不是跳过：
#   TAG=重启前 bash /root/sls2-deploy/sls2/tools/remote/ckpt_backup.sh
set -euo pipefail
R=${RUN_DIR:-$HOME/.local/share/sls2/runs/defect}
B=$R/backups
KEEP=${KEEP:-96}
mkdir -p "$B"
TAG=${TAG:-}
exec 9>"$B/.lock"
if [ -n "$TAG" ]; then flock -w 600 9; else flock -n 9 || exit 0; fi
[ -f "$R/latest.pt" ] || { [ -z "$TAG" ] || { echo "没有检查点可备份" >&2; exit 1; }; exit 0; }
# 检查点自上次备份以来没变（训练停着）就不再复制（2026-11-09：故障机器人停训后 96 份备份全是同一个 v169666；储君停训后 keep_backups_b
# 会把每份重复的都硬链接留下）。带 TAG 的手动备份照做
if [ -z "$TAG" ]; then
  newest=$(ls -1t "$B"/v*.pt 2>/dev/null | head -1 || true)
  [ -n "$newest" ] && [ "$newest" -nt "$R/latest.pt" ] && exit 0
fi
T=$(date +%Y%m%d-%H%M)${TAG:+-$TAG}
cp "$R/latest.pt" "$B/backup-$T.pt.part"
# 版本号从检查点本身读：检查点每 save_every 次更新才存一次，比日志里最新的版本旧
PY=$(cd "$(dirname "$0")/../.." && pwd)/.venv/bin/python
V=$("$PY" -c 'import sys, torch; print(torch.load(sys.argv[1], map_location="cpu", weights_only=False)["version"])' "$B/backup-$T.pt.part" 2>/dev/null || echo unknown)
mv "$B/backup-$T.pt.part" "$B/v$V-$T.pt"
cp "$R/train_config.json" "$B/v$V-$T.train_config.json"
ls -1t "$B"/v*.pt | tail -n +$((KEEP + 1)) | while read -r f; do rm -f "$f" "${f%.pt}.train_config.json"; done
echo "$(date '+%F %T') 备份 v$V-$T" >> "$B/backup.log"
