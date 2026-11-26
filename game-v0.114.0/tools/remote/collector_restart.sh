#!/usr/bin/env bash
# 在协同采集机（A）上运行：停掉远程采集进程（tools/remote/remote_actors.py）后按参数重新起（决策记录 201）。
# 配置读 /root/b_train_config.json（B 的 train_config.json 副本，改 B 的课程份额时同步改它）；进程数、每进程引擎数用环境变量给。
#   ACTORS=8 ENVS=20 bash /root/collector_restart.sh
# 按进程名匹配停止（本脚本自己的命令行里没有 remote_actors.py，不会误杀自己）；10 秒内没退出的强杀。
set -u
ACTORS=${ACTORS:-8}; ENVS=${ENVS:-20}
pkill -f 'tools/remote/remote_actors.py --config' || true
for _ in $(seq 10); do pgrep -f 'tools/remote/remote_actors.py --config' > /dev/null || break; sleep 1; done
pkill -9 -f 'tools/remote/remote_actors.py --config' || true
cd /root/sls2-deploy/sls2 && export DOTNET_ROOT=$HOME/.dotnet PATH=$HOME/.dotnet:$PATH DOTNET_CLI_TELEMETRY_OPTOUT=1
setsid nohup .venv/bin/python -I tools/remote/remote_actors.py --config /root/b_train_config.json --run-dir $HOME/.local/share/sls2/runs/defect \
  --host B_LAN_IP --port 47321 --token-file /root/.sls2_bridge_token --actors "$ACTORS" --envs "$ENVS" \
  >> $HOME/.local/share/sls2/runs/defect/collector.out 2>&1 < /dev/null &
sleep 2
echo "$(date '+%F %T') 远程采集已起：$ACTORS 个进程 × $ENVS 个引擎"
