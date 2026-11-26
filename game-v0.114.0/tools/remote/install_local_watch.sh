#!/usr/bin/env bash
# 安装本机守护的 systemd 用户定时器（每分钟一次）。卸载：systemctl --user disable --now sls2-local-watch.timer
set -e
SRC=$(cd "$(dirname "$0")/../.." && pwd)
D=$HOME/.config/systemd/user
mkdir -p "$D"
cat > "$D/sls2-local-watch.service" <<UNIT
[Unit]
Description=杀戮尖塔2 训练看板：保持拉取与看板生成在运行
[Service]
Type=oneshot
ExecStart=/bin/bash $SRC/tools/remote/local_watch.sh
KillMode=process
UNIT
cat > "$D/sls2-local-watch.timer" <<UNIT
[Unit]
Description=每分钟检查一次杀戮尖塔2 看板的常驻进程
[Timer]
OnBootSec=1min
OnUnitActiveSec=1min
[Install]
WantedBy=timers.target
UNIT
systemctl --user daemon-reload
systemctl --user enable --now sls2-local-watch.timer
systemctl --user list-timers | grep sls2 || true
