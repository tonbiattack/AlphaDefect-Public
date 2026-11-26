#!/usr/bin/env bash
# A 上：从 B 直连拉房间结果记录（B 的 run_dir/room_log，决策记录 146）到 ~/roomlog/。
# 用 A 上的专用密钥 ~/.ssh/KEY_AB，B 的 authorized_keys 里只允许它从 A 的内网地址只读拉这个目录（rrsync -ro）。
# 由 A 的 crontab 每 10 分钟跑一次；当前小时的文件还在追加，下一次拉取会补齐。
B=${B:-root@B_LAN_IP}
mkdir -p ~/roomlog
exec 9>~/roomlog/.pull.lock; flock -n 9 || exit 0
timeout 600 rsync -a -e "ssh -i $HOME/.ssh/KEY_AB -o IdentitiesOnly=yes -o BatchMode=yes" "$B:" ~/roomlog/ \
  && echo "$(date '+%F %T') 拉取完成，$(ls ~/roomlog | grep -c '\.gz$') 个文件，$(du -sh ~/roomlog | cut -f1)" >> ~/roomlog/pull.log \
  || echo "$(date '+%F %T') 拉取失败" >> ~/roomlog/pull.log
