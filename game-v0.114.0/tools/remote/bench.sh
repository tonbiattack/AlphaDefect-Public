#!/usr/bin/env bash
# 租用机基准测试（在租用机上运行；默认约 10–12 分钟，--quick 约 5 分钟）：
#   引擎单进程吞吐；不同“采集进程 × 每进程引擎”组合的采集吞吐（用检查点推理、不训练，评测照常开着）；
#   采集与学习同时跑时学习一次的耗时；大小核 CPU 上比较几种亲和性放法；最后给出推荐的硬件配置。细节见 bench.py 开头。
#   bash /root/sls2-deploy/sls2/tools/remote/bench.sh [--quick] [--ckpt 检查点.pt] [--grid 12x16,16x16,12x20,16x20] [bench.py 的其他参数]
# 默认取：配置 $RUN_DIR/train_config.json（没有时用 tools/remote/train_config.4090.json）；
#         检查点 $RUN_DIR/latest.pt → $RUN_DIR/backups 里最新的一份 → ~/bench_ckpt.pt（上机清单里从 A 的备份复制来的），都没有就随机初始化。
# 结果在 ~/sls2-bench/<时刻>/：bench.log（屏幕输出）、bench.json（全部读数）、recommend_hw.json（推荐的硬件项，迁移时给 --hw @该文件）。
# 训练正在这台机器上跑、或守护随时可能拉起训练（运行目录有配置却没有 STOP）时拒绝运行，FORCE=1 可强制（会和训练抢资源）。
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
RUN_DIR=${RUN_DIR:-$HOME/.local/share/sls2/runs/defect}
OUT=${OUT:-$HOME/sls2-bench/$(date +%Y%m%d-%H%M%S)}
PY=$REPO/.venv/bin/python
[ -x "$PY" ] || { echo "没有 $PY，先运行 setup.sh"; exit 1; }
[ -f "$REPO/engine/Sts2Headless/bin/Release/net9.0/Sts2Headless.dll" ] || { echo "引擎还没编译，先运行 setup.sh"; exit 1; }
export DOTNET_ROOT=$HOME/.dotnet PATH=$HOME/.dotnet:$PATH DOTNET_CLI_TELEMETRY_OPTOUT=1

if [ "${FORCE:-0}" != 1 ]; then
  if [ -f "$RUN_DIR/learner.pid" ] && P=$(cat "$RUN_DIR/learner.pid") && kill -0 "$P" 2>/dev/null && grep -qa train_defect "/proc/$P/cmdline"; then
    echo "训练正在运行（pid $P），基准测试会和它抢 CPU 与显卡。确需运行：FORCE=1 bash $0 $*"; exit 1
  fi
  if [ -f "$RUN_DIR/train_config.json" ] && [ ! -f "$RUN_DIR/STOP" ] && crontab -l 2>/dev/null | grep train_watchdog.sh >/dev/null; then
    echo "运行目录有 train_config.json 而没有 STOP，守护随时可能拉起训练。先 touch $RUN_DIR/STOP，测完再删；或 FORCE=1"; exit 1
  fi
fi

ARGS=()
case " $* " in *" --config "*) ;; *)
  if [ -f "$RUN_DIR/train_config.json" ]; then CFG=$RUN_DIR/train_config.json; else CFG=$REPO/tools/remote/train_config.4090.json; fi
  ARGS+=(--config "$CFG"); echo "配置：$CFG" ;;
esac
case " $* " in *" --ckpt "*) ;; *)
  CK=
  if [ -f "$RUN_DIR/latest.pt" ]; then CK=$RUN_DIR/latest.pt
  elif ls "$RUN_DIR"/backups/v*.pt >/dev/null 2>&1; then CK=$(ls -1t "$RUN_DIR"/backups/v*.pt | head -1)
  elif [ -f "$HOME/bench_ckpt.pt" ]; then CK=$HOME/bench_ckpt.pt
  fi
  if [ -n "$CK" ]; then ARGS+=(--ckpt "$CK"); else echo "警告：没有检查点，用随机初始化的网络（对局分布与真实训练不同，读数只作参考）"; fi ;;
esac

# CUDA MPS：与训练时一样（start_train.sh 也会启动它；已在运行就不重复）
if [ "${MPS:-1}" = 1 ] && command -v nvidia-cuda-mps-control >/dev/null && ! pgrep -x nvidia-cuda-mps >/dev/null && ! pgrep -f "nvidia-cuda-mps-control -d" >/dev/null; then
  nvidia-cuda-mps-control -d && echo "已启动 CUDA MPS"
fi

mkdir -p "$OUT"
echo "输出目录：$OUT"
cd "$REPO/python"
"$PY" -I "$REPO/tools/remote/bench.py" --out "$OUT" "${ARGS[@]}" "$@" 2>&1 | tee "$OUT/bench.log"
