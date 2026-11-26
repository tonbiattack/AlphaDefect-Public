#!/usr/bin/env bash
# 本机运行：把储君训练线要用的代码同步到租用机上单独的部署目录 /root/regent-deploy/sls2，并在那里编译引擎（2026-11-06）。
# 故障机器人的部署目录 /root/sls2-deploy/sls2 不动；虚拟环境与游戏程序集借用它（.venv 链接过去，游戏在 ~/.local/share/sls2/game/）。
#   bash tools/remote/regent/deploy.sh          # A、B 都同步并编译
#   bash tools/remote/regent/deploy.sh b        # 只弄 B
#   NOBUILD=1 bash tools/remote/regent/deploy.sh   # 只同步文件、不编译
# 编译之后核对程序集的修改时间确实变了（AGENTS.md：踩过多次用旧程序集验证新代码的坑）。运行中的训练进程不会换到新程序集，换代码要先停（stop.sh）。
set -euo pipefail
. "$(dirname "$0")/../hosts.sh"
D=/root/regent-deploy/sls2
HOSTS=("$@"); [ ${#HOSTS[@]} -gt 0 ] || HOSTS=(a b)
REV=$(git -C "$REPO" rev-parse --short HEAD)
DIRTY=$(git -C "$REPO" status --porcelain -- python engine/Sts2Headless engine/catalog tools/remote | grep -c . || true)
for h in "${HOSTS[@]}"; do
  host_require "$h"
  SSH=$(host_ssh "$h"); E=$(host_rsync_e "$h"); A=$(host_addr "$h")
  echo "== $h：同步到 $D"
  $SSH "mkdir -p $D/engine $D/tools && { [ -e $D/.venv ] || ln -s /root/sls2-deploy/sls2/.venv $D/.venv; }"
  rsync -a --exclude __pycache__ -e "$E" "$REPO/python/" "$A:$D/python/"
  rsync -a --exclude __pycache__ --exclude stream/ -e "$E" "$REPO/tools/" "$A:$D/tools/"
  rsync -a --exclude bin/ --exclude obj/ -e "$E" "$REPO/engine/Sts2Headless/" "$A:$D/engine/Sts2Headless/"
  rsync -a -e "$E" "$REPO/engine/catalog/" "$A:$D/engine/catalog/"
  rsync -a -e "$E" "$REPO/engine/card_stats/" "$A:$D/engine/card_stats/"
  $SSH "echo '$REV（本机未提交的改动 $DIRTY 处）$(date '+%F %T')' > $D/DEPLOYED"
  if [ -z "${NOBUILD:-}" ]; then
    $SSH "set -e; export DOTNET_ROOT=\$HOME/.dotnet PATH=\$HOME/.dotnet:\$PATH DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1 DOTNET_CLI_UI_LANGUAGE=en
      DLL=$D/engine/Sts2Headless/bin/Release/net9.0/Sts2Headless.dll
      before=\$(stat -c %Y \$DLL 2>/dev/null || echo 0)
      cd $D/engine/Sts2Headless && nice -n 19 dotnet build -c Release -p:Sts2GameDir=\$HOME/.local/share/sls2/game/$GAME_VER > $D/../build.log 2>&1 || { tail -20 $D/../build.log; exit 1; }
      after=\$(stat -c %Y \$DLL)
      # 源码没变时增量编译不改程序集：要求程序集不早于最新的源码（.cs、.csproj），否则算编译没生效
      src=\$(find $D/engine/Sts2Headless -path '*/bin' -prune -o -path '*/obj' -prune -o \( -name '*.cs' -o -name '*.csproj' \) -printf '%T@\\n' | sort -n | tail -1 | cut -d. -f1)
      [ \$after -ge \$src ] || { echo 程序集（\$(date -d @\$after '+%F %T')）比源码（\$(date -d @\$src '+%F %T')）旧，编译可能没生效; exit 1; }
      [ \$after -gt \$before ] && echo 引擎已编译：\$(date -d @\$after '+%F %T') || echo 引擎源码没变，程序集已是最新：\$(date -d @\$after '+%F %T')"
  fi
done
