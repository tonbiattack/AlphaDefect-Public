#!/usr/bin/env bash
# 在本机打包：代码（工作区当前内容，含未提交改动）与游戏程序集、本地化文件，生成一个 tar.gz，供上传到租用机。
# 游戏文件来自你已购买的本机安装，只用于自己的租用机，不要公开分发。
#   bash tools/remote/pack.sh                 # 只打包，输出到仓库上一级目录
#   bash tools/remote/pack.sh 输出目录         # 只打包到指定目录
#   bash tools/remote/pack.sh b               # 打包并上传到 hosts.json 里的主机 b，解压到其 deploy 的上一级（如 /root/sls2-deploy）
#   SETUP=1 bash tools/remote/pack.sh b       # 上传解压后，在该主机后台运行 setup.sh（日志 <deploy 上一级>/setup.out）
# 只用于新机器上机：目标机上已有部署（有 .venv 或已编译的引擎）时拒绝覆盖，FORCE=1 可强制；目标机上训练正在跑时一律拒绝
# （覆盖正在运行的代码会让采集进程、评测进程加载到新旧混杂的模块）。已上机的机器更新代码用 restart.sh。
# 环境变量：STS2_VERSION（默认 engine/game_version.txt 里的版本）、STS2_GAME_DIR（默认 ~/.local/share/sls2/game/<版本>）。
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
V=${STS2_VERSION:-$(tr -d '[:space:]' < "$REPO/engine/game_version.txt" 2>/dev/null || echo v0.111.0)}
GAME=${STS2_GAME_DIR:-$HOME/.local/share/sls2/game/$V}
[ -f "$GAME/sts2.dll" ] && [ -d "$GAME/res/localization/eng" ] || { echo "本机游戏文件不全：$GAME（需要 sts2.dll 与 res/localization）"; exit 1; }

HOST=
. "$REPO/tools/remote/hosts.sh"
if [ $# -gt 0 ] && host_exists "$1"; then HOST=$1; host_require "$HOST"; OUT=$(mktemp -d); else OUT=${1:-$REPO/..}; fi
STAMP=$(date +%Y%m%d-%H%M)
NAME=sls2-deploy-$STAMP.tar.gz
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"; [ -n "$HOST" ] && rm -rf "$OUT"' EXIT
mkdir -p "$STAGE/sls2" "$STAGE/game/$V"
rsync -a --exclude .git/ --exclude bin/ --exclude obj/ --exclude .venv/ --exclude __pycache__/ --exclude runs/ --exclude .hotdev-source \
  --exclude node_modules/ --exclude '回放/' --exclude '回放样例/' --exclude '训练看板*' "$REPO/" "$STAGE/sls2/"
# 记下打包时的代码版本，事后可对照
{ echo "packed_at=$(date '+%F %T')"; echo "game=$V"
  echo "commit=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo 未知)"
  echo "dirty_files=$(git -C "$REPO" status --porcelain 2>/dev/null | wc -l)"; } > "$STAGE/sls2/DEPLOY_INFO"
rsync -a --exclude '*.pdb' "$GAME/" "$STAGE/game/$V/"
tar -C "$STAGE" -czf "$OUT/$NAME" sls2 game
ls -la "$OUT/$NAME"

if [ -z "$HOST" ]; then
  echo "上传：rsync 或 scp $OUT/$NAME 到租用机的 /root/sls2-deploy/，然后在那里：tar xzf $NAME && bash sls2/tools/remote/setup.sh"
  echo "（或者先在 hosts.json 登记主机，再 bash tools/remote/pack.sh <主机名>，自动上传解压）"
  exit 0
fi

SSH=$(host_ssh "$HOST"); E=$(host_rsync_e "$HOST"); A=$(host_addr "$HOST")
D=$(host_deploy "$HOST"); P=${D%/*}; R=$(host_run "$HOST")
echo "目标：$HOST（$(host_label "$HOST")）$A:$P"
# 只读检查：训练是否在跑、是否已有部署
state=$($SSH "$REMOTE_ALIVE; cd \$HOME/$R 2>/dev/null && alive && echo ALIVE; [ -e $D/.venv ] || [ -e $D/engine/Sts2Headless/bin ] && echo DEPLOYED; true") \
  || { echo "连不上 $HOST"; exit 1; }
case $state in *ALIVE*) echo "$HOST 上训练正在运行，拒绝覆盖部署目录（更新代码请用 restart.sh）"; exit 1 ;; esac
case $state in *DEPLOYED*) [ "${FORCE:-0}" = 1 ] || { echo "$HOST 上 $D 已有部署（.venv 或已编译的引擎）。确需覆盖：FORCE=1"; exit 1; } ;; esac
$SSH "mkdir -p $P"
rsync -a --partial -e "$E" "$OUT/$NAME" "$A:$P/"
$SSH "cd $P && tar xzf $NAME && echo 已解压到 $P && cat $D/DEPLOY_INFO"
if [ "${SETUP:-0}" = 1 ]; then
  $SSH "cd $P && setsid nohup bash sls2/tools/remote/setup.sh > setup.out 2>&1 < /dev/null & echo 已在后台运行 setup.sh，日志 $P/setup.out"
  echo "查看进度：$SSH 'tail -f $P/setup.out'（最后一行出现“下一步：基准测试”即完成）"
else
  echo "安装：$SSH 'cd $P && setsid nohup bash sls2/tools/remote/setup.sh > setup.out 2>&1 < /dev/null &'"
fi
