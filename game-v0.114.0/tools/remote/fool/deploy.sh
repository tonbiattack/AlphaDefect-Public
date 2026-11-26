#!/usr/bin/env bash
# 本机运行：把 Fool 训练线要用的代码部署到 B 上单独的部署目录 /root/fool-deploy/sls2，并在那里编译引擎（2026-11-12，照 tools/remote/silent/ 复制，游戏 v0.113.0 的新角色 Fool）。
# 只发已提交的代码（git archive HEAD 的 python、tools、engine/Sts2Headless、engine/catalog、engine/card_stats、engine/game_version.txt），工作区里别的会话未提交的改动不跟过去；
# 部署的提交号写进 DEPLOYED（同行审查要能对上每一版用的代码）。虚拟环境借用故障机器人的部署目录（.venv 链接过去），游戏在 ~/.local/share/sls2/game/。
#   bash tools/remote/fool/deploy.sh            # 同步并编译
#   NOBUILD=1 bash tools/remote/fool/deploy.sh  # 只同步文件
# 编译之后核对程序集不早于源码（AGENTS.md：踩过多次用旧程序集验证新代码的坑）。运行中的训练进程不会换到新代码，换代码要先停（stop.sh）。
set -euo pipefail
. "$(dirname "$0")/../hosts.sh"
D=/root/fool-deploy/sls2
H=${HOST:-b}
REV=$(git -C "$REPO" rev-parse --short HEAD)
SSH=$(host_ssh "$H"); E=$(host_rsync_e "$H"); A=$(host_addr "$H")
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
git -C "$REPO" archive --format=tar HEAD python tools engine/Sts2Headless engine/catalog engine/card_stats engine/game_version.txt | tar -x -C "$TMP"
echo "== $H：部署提交 $REV 到 $D"
$SSH "mkdir -p $D && { [ -e $D/.venv ] || ln -s /root/sls2-deploy/sls2/.venv $D/.venv; }"
rsync -a --delete --exclude bin/ --exclude obj/ --exclude .venv --exclude DEPLOYED -e "$E" "$TMP/" "$A:$D/"
# 游戏文件（只放本机与租用机，不入库；引擎模块与游戏更新适配.md 第七节）：租用机上还没有这个版本（engine/game_version.txt）时从本机同步，约 100 MB
G=$HOME/.local/share/sls2/game/$GAME_VER
if ! $SSH "[ -f /root/.local/share/sls2/game/$GAME_VER/sts2.dll ]"; then
  [ -f "$G/sts2.dll" ] || { echo "本机没有 $G（先照适配文档第六节第 1 步摆好）"; exit 1; }
  echo "== $H：同步游戏文件 $GAME_VER"
  $SSH "mkdir -p /root/.local/share/sls2/game/$GAME_VER"
  rsync -a -e "$E" "$G/" "$A:/root/.local/share/sls2/game/$GAME_VER/"
  $SSH "cd /root/.local/share/sls2/game/$GAME_VER && sha256sum -c --quiet SHA256SUMS && echo 游戏文件校验通过"
fi
$SSH "echo '$REV $(date '+%F %T')（git archive HEAD，只含已提交的代码）' > $D/DEPLOYED; cat $D/DEPLOYED"
# 核对租用机上 Python 认的游戏版本与内容目录（2026-11-13：game_version.txt 没跟着发，Python 退回 v0.111.0 的内容目录，Fool 的课程取不到卡池、内容先验也按旧目录建）
PYVER=$($SSH "cd $D && .venv/bin/python -I -c 'import sys; sys.path.insert(0, \"python\"); from sts2env.gamever import GAME_VERSION, catalog_path; assert catalog_path().is_file(); print(GAME_VERSION)'") \
  || { echo "$H 上读不出游戏版本或内容目录不在"; exit 1; }
[ "$PYVER" = "$GAME_VER" ] || { echo "$H 上 Python 认的游戏版本是 $PYVER，本机是 $GAME_VER"; exit 1; }
echo "游戏版本 $PYVER（Python 与引擎一致）"
if [ -z "${NOBUILD:-}" ]; then
  $SSH "set -e; export DOTNET_ROOT=\$HOME/.dotnet PATH=\$HOME/.dotnet:\$PATH DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1 DOTNET_CLI_UI_LANGUAGE=en
    DLL=$D/engine/Sts2Headless/bin/Release/net9.0/Sts2Headless.dll
    cd $D/engine/Sts2Headless && nice -n 19 dotnet build -c Release -p:Sts2GameDir=\$HOME/.local/share/sls2/game/$GAME_VER > $D/../build.log 2>&1 || { tail -20 $D/../build.log; exit 1; }
    after=\$(stat -c %Y \$DLL)
    src=\$(find $D/engine/Sts2Headless -path '*/bin' -prune -o -path '*/obj' -prune -o \( -name '*.cs' -o -name '*.csproj' \) -printf '%T@\\n' | sort -n | tail -1 | cut -d. -f1)
    # 源码的修改时间是 git archive 写的提交时刻（本机时钟），B 的时钟慢约 8 秒（2026-11-13 实测），留 60 秒余量
    [ \$after -ge \$((src - 60)) ] || { echo 程序集（\$(date -d @\$after '+%F %T')）比源码（\$(date -d @\$src '+%F %T')）旧，编译可能没生效; exit 1; }
    echo 引擎已编译：\$(date -d @\$after '+%F %T')，md5 \$(md5sum < \$DLL | cut -c1-8)"
fi
