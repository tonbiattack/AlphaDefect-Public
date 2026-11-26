#!/usr/bin/env bash
# 本机运行：把铁甲战士训练线要用的代码部署到租用机上单独的部署目录 /root/ironclad-deploy/sls2，并在那里编译引擎
# （2026-11-17，铁甲战士决策记录第 1 条；照 tools/remote/fool/deploy.sh 与恐惧线的 tools/remote/dread/deploy.sh）。
# 2026-11-20 起默认部署到 B（C 撤离，铁甲战士决策记录第 24 条）；HOST=c 仍按 C 的做法。
# 只发已提交的代码（git archive HEAD 的 python、tools、engine/Sts2Headless、engine/catalog、engine/card_stats、engine/game_version.txt），
# 工作区里未提交的改动不跟过去；部署的提交号写进 DEPLOYED。游戏在 C 的 ~/.local/share/sls2/game/<版本>/，没有时从本机同步并校验。
# 虚拟环境借现成的，.venv 链接过去，只读不改，本脚本不往里装任何包：
#   B：/root/sls2-deploy/sls2/.venv（亡灵契约师、愚者线也链到这里：PyTorch 2.14.1 cu130，驱动 580，不用另配 ptxas）；
#   C：恐惧线的 venv-cu126（PyTorch 2.14.1 cu126、ptxas 12.8），缺 ptxas 时报错退出（驱动 570 下 triton 自带的 ptxas 编出的核载入不了，恐惧模式决策记录第 4 条）。
# 恐惧线的部署目录、运行目录、定时任务都不碰。
#   bash tools/remote/ironclad/deploy.sh            # 同步并编译
#   NOBUILD=1 bash tools/remote/ironclad/deploy.sh  # 只同步文件
# 编译之后核对程序集不早于源码（AGENTS.md）。运行中的训练进程不会换到新代码，换代码要先停（stop.sh）。
set -euo pipefail
. "$(dirname "$0")/../hosts.sh"
D=/root/ironclad-deploy/sls2
H=${HOST:-b}
case "$H" in
  c) VENV=${VENV:-/root/dread-deploy/venv-cu126}; NEED_PTX=1 ;;
  *) VENV=${VENV:-/root/sls2-deploy/sls2/.venv}; NEED_PTX=0 ;;
esac
host_require "$H"
REV=$(git -C "$REPO" rev-parse --short HEAD)
SSH=$(host_ssh "$H"); E=$(host_rsync_e "$H"); A=$(host_addr "$H")
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
git -C "$REPO" archive --format=tar HEAD python tools engine/Sts2Headless engine/catalog engine/card_stats engine/game_version.txt | tar -x -C "$TMP"
echo "== $H：部署提交 $REV 到 $D"
$SSH "set -e; mkdir -p $D; [ -x $VENV/bin/python ] || { echo 没有虚拟环境 $VENV; exit 1; }
  if [ ! -e $D/.venv ]; then ln -s $VENV $D/.venv; fi
  [ \"\$(readlink -f $D/.venv)\" = \"\$(readlink -f $VENV)\" ] || { echo $D/.venv 没指向 $VENV; exit 1; }
  [ $NEED_PTX = 0 ] || ls $D/.venv/lib/python3*/site-packages/nvidia/cuda_nvcc/bin/ptxas >/dev/null || { echo 虚拟环境里没有 ptxas; exit 1; }"
rsync -a --delete --exclude bin/ --exclude obj/ --exclude .venv --exclude DEPLOYED -e "$E" "$TMP/" "$A:$D/"
G=$HOME/.local/share/sls2/game/$GAME_VER
if ! $SSH "[ -f /root/.local/share/sls2/game/$GAME_VER/sts2.dll ]"; then
  [ -f "$G/sts2.dll" ] || { echo "本机没有 $G（先照适配文档第六节第 1 步摆好）"; exit 1; }
  echo "== $H：同步游戏文件 $GAME_VER"
  $SSH "mkdir -p /root/.local/share/sls2/game/$GAME_VER"
  rsync -a --partial -e "$E" "$G/" "$A:/root/.local/share/sls2/game/$GAME_VER/"
fi
$SSH "cd /root/.local/share/sls2/game/$GAME_VER && sha256sum -c --quiet SHA256SUMS && echo 游戏文件校验通过"
$SSH "echo '$REV $(date '+%F %T')（git archive HEAD，只含已提交的代码）' > $D/DEPLOYED; cat $D/DEPLOYED"
# 核对远端 Python 认的游戏版本与内容目录（愚者第 1 条：game_version.txt 没跟着发时 Python 会悄悄退回旧内容目录）
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
    # 源码的修改时间是 git archive 写的提交时刻（本机时钟），留 60 秒余量
    [ \$after -ge \$((src - 60)) ] || { echo 程序集（\$(date -d @\$after '+%F %T')）比源码（\$(date -d @\$src '+%F %T')）旧，编译可能没生效; exit 1; }
    echo 引擎已编译：\$(date -d @\$after '+%F %T')，md5 \$(md5sum < \$DLL | cut -c1-8)"
fi
