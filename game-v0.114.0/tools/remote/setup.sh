#!/usr/bin/env bash
# 在租用机（Linux x64，一般为 Ubuntu 22.04/24.04、root）上安装环境、编译引擎、冒烟测试、装守护定时任务。
# 在解压目录（含 sls2/ 与 game/，即 /root/sls2-deploy）运行；可重复运行，已装好的步骤会跳过：
#   cd /root/sls2-deploy && bash sls2/tools/remote/setup.sh
# 一般用 nohup 在后台跑（装 PyTorch 要下载约 3 GB），见 tools/remote/新机器上机.md。
# 环境变量：
#   STS2_VERSION   游戏版本（默认 engine/game_version.txt 里的版本，与 pack.sh 一致）
#   APT=0          不自动 apt 安装缺的系统包（libicu、python3-venv、rsync、cron 等；默认在 root 且有 apt-get 时安装）
#   DOTNET_SDK_VERSION  .NET SDK 版本（默认 9.0.319，与 A 和本机一致；装不上时退回 9.0 频道的最新版）
#   TORCH_VERSION  默认 2.14.1（与 A 一致）；NUMPY_VERSION 默认 2.2.6（与 A 一致）
#   TORCH_INDEX    默认按显卡驱动自动选：驱动 ≥ 580 用 pypi（经 PIP_INDEX_URL 镜像，Linux 上即 CUDA 13.0 版，与 A 相同）；
#                  更旧的驱动用 cu128（官方源）。也可手动设 pypi / cu130 / cu128 / cu126 / cpu
#   PIP_INDEX_URL  默认清华镜像
#   PYTHON         默认 python3；有 uv 时也可设 UV_PYTHON=3.12
#   CRON=0         不安装守护定时任务（train_watchdog 每 2 分钟、ckpt_backup 每 30 分钟）
#   RUN_DIR        训练运行目录（默认 ~/.local/share/sls2/runs/defect，与 hosts.json 的 run_dir 一致）
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
REPO=$ROOT/sls2
V=${STS2_VERSION:-$(tr -d '[:space:]' < "$REPO/engine/game_version.txt" 2>/dev/null || echo v0.111.0)}
GAME_SRC=$ROOT/game/$V
GAME=$HOME/.local/share/sls2/game/$V
RUN_DIR=${RUN_DIR:-$HOME/.local/share/sls2/runs/defect}
DOTNET_SDK_VERSION=${DOTNET_SDK_VERSION:-9.0.319}
TORCH_VERSION=${TORCH_VERSION:-2.14.1}
NUMPY_VERSION=${NUMPY_VERSION:-2.2.6}
PIP_INDEX_URL=${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}
PYTHON=${PYTHON:-python3}
WARN=()
warn() { echo "警告：$*"; WARN+=("$*"); }
T0=$(date +%s)

echo "== 0. 机器检查"
. /etc/os-release 2>/dev/null && echo "系统：${PRETTY_NAME:-未知}，内核 $(uname -r)"
echo "CPU：$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//')，$(nproc) 个逻辑核"
if [ -r /sys/devices/cpu_core/cpus ] && [ -r /sys/devices/cpu_atom/cpus ]; then
  echo "大小核：性能核 $(cat /sys/devices/cpu_core/cpus)，能效核 $(cat /sys/devices/cpu_atom/cpus)"
fi
echo "内存：$(free -g | awk '/^Mem|^内存/ {print $2}') GB；/dev/shm $(df -BG --output=size /dev/shm 2>/dev/null | tail -1 | tr -d ' ')；家目录剩余 $(df -BG --output=avail "$HOME" | tail -1 | tr -d ' ')"
[ "$(df -BG --output=avail "$HOME" | tail -1 | tr -dc 0-9)" -ge 40 ] || warn "家目录所在磁盘剩余不足 40 GB（运行目录与备份约 10–20 GB，PyTorch 约 5 GB）"
DRV=
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=name,driver_version,memory.total,compute_mode --format=csv,noheader || true
  DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
  command -v nvidia-cuda-mps-control >/dev/null || warn "没有 nvidia-cuda-mps-control：无法开 CUDA MPS（学习会慢约两成）"
else
  warn "没有 nvidia-smi：显卡驱动没装好，训练会退回 CPU，不能用"
fi
# 评测用 cgroup v2 限核（evaluator._cpu_cap 往 /sys/fs/cgroup 写）；不可写时评测只靠 SCHED_IDLE，会因超线程争用拖慢训练（决策记录第 50 条）
if [ "$(stat -fc %T /sys/fs/cgroup 2>/dev/null)" = cgroup2fs ] && [ -w /sys/fs/cgroup/cgroup.subtree_control ]; then
  grep -qw cpu /sys/fs/cgroup/cgroup.controllers && echo "cgroup v2 可用（评测限核）" || warn "cgroup v2 没有 cpu 控制器，评测无法限核"
else
  warn "cgroup v2 不可写（不是 root，或在容器里），评测无法限核"
fi

echo "== 1. 系统包"
need=()
ldconfig -p 2>/dev/null | grep libicuuc >/dev/null || need+=(libicu-dev)        # 无头引擎要 ICU（本地化格式化用到区域设置）
"$PYTHON" -c 'import venv, ensurepip' 2>/dev/null || command -v uv >/dev/null || need+=(python3-venv)
for c in rsync curl crontab flock; do
  command -v $c >/dev/null || case $c in crontab) need+=(cron) ;; flock) need+=(util-linux) ;; *) need+=($c) ;; esac
done
if [ ${#need[@]} -gt 0 ]; then
  if [ "${APT:-1}" = 1 ] && [ "$(id -u)" = 0 ] && command -v apt-get >/dev/null; then
    # Ubuntu 的 python3-venv 要装与系统 Python 同版本的那个包（如 python3.10-venv），两个都装
    pyv=$("$PYTHON" -c 'import sys; print(f"python{sys.version_info[0]}.{sys.version_info[1]}-venv")' 2>/dev/null || true)
    pk=("${need[@]}"); [[ " ${need[*]} " == *" python3-venv "* ]] && [ -n "$pyv" ] && pk+=("$pyv")
    echo "安装：${pk[*]}"
    DEBIAN_FRONTEND=noninteractive apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends "${pk[@]}" ca-certificates
  else
    warn "缺少系统包：${need[*]}（APT=0、不是 root 或没有 apt-get，未自动安装）"
  fi
else
  echo "齐全"
fi
pgrep -x cron >/dev/null || pgrep -x crond >/dev/null || { service cron start >/dev/null 2>&1 || true; pgrep -x cron >/dev/null || warn "cron 服务没在运行：守护和定时备份不会执行（可 service cron start）"; }

echo "== 2. 游戏程序集 → $GAME"
[ -d "$GAME_SRC" ] || { [ -f "$GAME/sts2.dll" ] || { echo "缺少 $GAME_SRC（pack.sh 打的包里应有 game/$V）"; exit 1; }; }
mkdir -p "$GAME"; [ -d "$GAME_SRC" ] && rsync -a "$GAME_SRC/" "$GAME/"
test -f "$GAME/sts2.dll" && test -d "$GAME/res/localization/eng" || { echo "缺少 sts2.dll 或本地化文件"; exit 1; }

echo "== 3. .NET 9 SDK（装在 ~/.dotnet）"
export DOTNET_ROOT=$HOME/.dotnet PATH=$HOME/.dotnet:$PATH DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1 DOTNET_CLI_UI_LANGUAGE=en
if ! "$HOME/.dotnet/dotnet" --list-sdks 2>/dev/null | grep -q '^9\.'; then
  curl -fsSL --retry 3 https://dot.net/v1/dotnet-install.sh -o /tmp/dotnet-install.sh
  bash /tmp/dotnet-install.sh --version "$DOTNET_SDK_VERSION" --install-dir "$HOME/.dotnet" \
    || bash /tmp/dotnet-install.sh --channel 9.0 --install-dir "$HOME/.dotnet"
fi
dotnet --list-sdks

echo "== 4. 编译引擎"
(cd "$REPO/engine/Sts2Headless" && dotnet build -c Release -p:Sts2GameDir="$GAME" > "$ROOT/build.log" 2>&1) || { tail -30 "$ROOT/build.log"; exit 1; }
echo "编译完成"

echo "== 5. Python 环境"
if [ ! -x "$REPO/.venv/bin/python" ]; then
  if command -v uv >/dev/null && [ -n "${UV_PYTHON:-}" ]; then
    (cd "$REPO" && uv venv -q --python "$UV_PYTHON" .venv)
  else
    "$PYTHON" -m venv "$REPO/.venv" || { echo "python3 -m venv 不可用（apt install python3-venv，或装 uv 后设 UV_PYTHON）"; exit 1; }
  fi
fi
VPY=$REPO/.venv/bin/python
"$VPY" -c 'import sys; assert sys.version_info >= (3, 10), sys.version' || { echo "需要 Python 3.10 以上"; exit 1; }
if [ -z "${TORCH_INDEX:-}" ]; then
  if [ -n "$DRV" ] && [ "$DRV" -ge 580 ]; then TORCH_INDEX=pypi
  elif [ -n "$DRV" ]; then TORCH_INDEX=cu128; warn "显卡驱动 $DRV 低于 580，不能用 CUDA 13.0 版 PyTorch，改装 cu128 版（官方源，国内可能慢；升级驱动到 580 以上可与 A 完全一致）"
  else TORCH_INDEX=cpu; fi
fi
echo "PyTorch $TORCH_VERSION / $TORCH_INDEX，numpy $NUMPY_VERSION"
if command -v uv >/dev/null; then PIP=(uv pip install -q --python "$VPY"); else "$VPY" -m pip install -q -U pip -i "$PIP_INDEX_URL"; PIP=("$VPY" -m pip install -q); fi
"$VPY" -c "import numpy; assert numpy.__version__ == '$NUMPY_VERSION'" 2>/dev/null || "${PIP[@]}" "numpy==$NUMPY_VERSION" --index-url "$PIP_INDEX_URL" \
  || { warn "装不上 numpy $NUMPY_VERSION（Python 版本太新？），改装最新版"; "${PIP[@]}" numpy --index-url "$PIP_INDEX_URL"; }
if [ "$TORCH_INDEX" = pypi ]; then TORCH_URL=$PIP_INDEX_URL; else TORCH_URL=https://download.pytorch.org/whl/$TORCH_INDEX; fi
"$VPY" -c "import torch; assert torch.__version__.startswith('$TORCH_VERSION')" 2>/dev/null || "${PIP[@]}" "torch==$TORCH_VERSION" --index-url "$TORCH_URL"
"$VPY" - <<'EOF'
import torch
print("torch", torch.__version__, "CUDA", torch.version.cuda, "可用", torch.cuda.is_available(), "显卡数", torch.cuda.device_count())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"显卡 {p.name}，{p.total_memory / 2**30:.1f} GiB，算力 {p.major}.{p.minor}，bf16 {torch.cuda.is_bf16_supported()}")
    x = torch.randn(2048, 2048, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = (x @ x).float().sum().item()
    print("显卡计算正常")
EOF
if [ "$TORCH_INDEX" != cpu ] && ! "$VPY" -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)'; then
  echo "PyTorch 用不了显卡（驱动与 CUDA 版本不匹配？）。可设 TORCH_INDEX=cu128 或 cu126 后重新运行（先删 $REPO/.venv）"; exit 1
fi

echo "== 6. 冒烟测试"
DLL=$REPO/engine/Sts2Headless/bin/Release/net9.0/Sts2Headless.dll
dotnet "$DLL" runs 5 --god | grep -E "局数|游戏报错"
dotnet "$DLL" determinism --god | grep DEFECT
(cd "$REPO/python" && ../.venv/bin/python -I bench_async.py 4 10 combat)
(cd "$REPO/python" && ../.venv/bin/python -I -c 'import sys; sys.path.insert(0, "."); from sts2rl import train, actor, evaluator; print("训练代码可导入")')

echo "== 7. 守护与备份的定时任务"
mkdir -p "$RUN_DIR"
if [ ! -f "$RUN_DIR/train_config.json" ] && [ ! -f "$RUN_DIR/latest.pt" ]; then
  # 新机器：运行目录是空的。先放 STOP，守护就不会在迁移只做了一半时从零开始训练；migrate 放好检查点和配置后删掉 STOP 再启动
  touch "$RUN_DIR/STOP"; echo "运行目录为空，已放 STOP（迁移完成后由 migrate.sh 删除）"
fi
if [ "${CRON:-1}" = 1 ] && command -v crontab >/dev/null; then
  W="*/2 * * * * bash $REPO/tools/remote/train_watchdog.sh"
  B="*/30 * * * * bash $REPO/tools/remote/ckpt_backup.sh"
  { crontab -l 2>/dev/null | grep -v -F "$REPO/tools/remote/train_watchdog.sh" | grep -v -F "$REPO/tools/remote/ckpt_backup.sh" || true; echo "$W"; echo "$B"; } | crontab -
  echo "crontab："; crontab -l | grep -F "$REPO/tools/remote/"
fi

echo "== 完成（用时 $(( $(date +%s) - T0 )) 秒）"
for w in "${WARN[@]}"; do echo "警告：$w"; done
echo "下一步：基准测试  bash $REPO/tools/remote/bench.sh   （约 10–12 分钟；结果里的 recommend_hw.json 用于迁移）"
