#!/usr/bin/env bash
# 本机运行：重启某台租用机上的训练（取代原来只认 A 的临时重启脚本）。
# 同步 python/ 和守护、备份脚本（ckpt_backup、train_watchdog、start_train、stoploss）到该主机，停掉训练，先备份一次
# （停止时训练进程会存好检查点，备份的是改动前那一刻的模型和配置），再按参数改 train_config.json（JSON 对象，可省略）并重启，
# 打印重启时刻。备份失败就不重启，训练保持停止，等人处理。
# ENGINE=1 时同时同步引擎源码与 catalog，在训练停下之后重新编译（运行中的引擎还在用旧程序集，不能边跑边覆盖）；编译失败同样不重启。
#   bash tools/remote/restart.sh '{"actors": 10}'        # 主线所在的主机（lines.json 的 main，默认 a）
#   bash tools/remote/restart.sh b '{"actors": 10}'      # 指定主机（hosts.json 里的名字）
#   ENGINE=1 bash tools/remote/restart.sh a
set -euo pipefail
. "$(dirname "$0")/hosts.sh"
if [ $# -gt 0 ] && host_exists "$1"; then H=$1; shift; else H=$(main_host); fi
host_require "$H"
UPD=${1:-}; [ -n "$UPD" ] || UPD="{}"; UPD=$(check_json_obj "$UPD")
SSH=$(host_ssh "$H"); R=$(host_run "$H"); D=$(host_deploy "$H")
echo "重启 $H（$(host_label "$H")）上的训练，配置改动：$UPD"
sync_code "$H"
BUILD=$(remote_build "$H")
$SSH "set -e; R=\$HOME/$R; cd \$R
$REMOTE_ALIVE
$REMOTE_STOP
TAG=重启前 RUN_DIR=\$R bash $D/tools/remote/ckpt_backup.sh || { echo 备份失败，训练未重启（STOP 仍在）; exit 1; }
echo 已备份 \$(tail -1 backups/backup.log)
$BUILD
python3 -c 'import json,sys; c=json.load(open(\"train_config.json\")); u=json.loads(sys.argv[1]); c.update(u); json.dump(c, open(\"train_config.json\",\"w\"), ensure_ascii=False, indent=1); print(\"改动:\", u)' $(printf %q "$UPD")
rm STOP; RUN_DIR=\$R bash $D/tools/remote/start_train.sh; date +%s"
