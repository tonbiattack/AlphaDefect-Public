#!/usr/bin/env bash
# 本机守护（systemd 用户定时器每分钟）：确保精简拉取与轻量看板（pull_lite.sh，2026-11-03 起）、公网推送（publish.sh）
# 求解器标注（tools/solver/label_loop.py，本机装了 Combat Solver 模组时）和搜索蒸馏中转（tools/distill/relay.py，打开时）几个常驻进程在跑。
#   安装：bash tools/remote/install_local_watch.sh
SRC=$(cd "$(dirname "$0")/../.." && pwd)
LOGD=$HOME/.local/share/sls2/runs/remote
mkdir -p "$LOGD"
# 拉取（2026-11-03 起，用户：“精简一下看板……其他的维持统计存在训练机，不要自动回传了”）：只取两台租用机的精简统计并生成轻量看板
# （tools/remote/pull_lite.sh → tools/dashboard_lite.py），日志 pull-lite.log。原来的 pull.sh（拉 log.jsonl、eval.jsonl 等完整统计）、
# 旧看板 dashboard.py --loop、价值探针档案回传 value_probe_pull.sh、每小时的回放拉取与推送都不再起；先停掉还在跑的旧进程。
pkill -f "sls2/tools/remote/[p]ull.sh" 2>/dev/null; pkill -f "sls2/tools/[d]ashboard.py --loop" 2>/dev/null; pkill -f "sls2/tools/remote/[v]alue_probe_pull.sh" 2>/dev/null
pgrep -f "sls2/tools/remote/pull_lite.sh" >/dev/null || LOOP=1 setsid nohup bash "$SRC/tools/remote/pull_lite.sh" >> "$LOGD/pull-lite.log" 2>&1 < /dev/null &
pgrep -f "sls2/tools/remote/publish.sh" >/dev/null || LOOP=1 setsid nohup bash "$SRC/tools/remote/publish.sh" >> "$LOGD/publish.log" 2>&1 < /dev/null &
SOLVERD=$HOME/.local/share/sls2/solver
# 2026-10-27 起暂停（用户：给开发 Rust 引擎腾出本机性能）：$SOLVERD/PAUSED 存在时不拉起，删掉即恢复
if [ -f "$HOME/.local/share/sls2/mods/CombatSolver/CombatSolver.dll" ] && [ ! -f "$SOLVERD/PAUSED" ]; then
  mkdir -p "$SOLVERD"
  pgrep -f "sls2/tools/solver/label_loop.py" >/dev/null || setsid nohup python3 "$SRC/tools/solver/label_loop.py" >> "$SOLVERD/label_loop.log" 2>&1 < /dev/null &
fi
# 搜索蒸馏的中转（tools/distill/relay.py，2026-10-27 起）：~/.local/share/sls2/distill/ENABLED 存在时保证在跑
DISTD=$HOME/.local/share/sls2/distill
if [ -f "$DISTD/ENABLED" ]; then
  pgrep -f "sls2/tools/distill/relay.py" >/dev/null || setsid nohup python3 "$SRC/tools/distill/relay.py" >> "$DISTD/relay.log" 2>&1 < /dev/null &
fi
# 定期复测（tools/remote/retest.sh，2026-10-28 起，决策记录 171）：主线每多训约 2,000 版自动测一次，没到时间时立即退出
pgrep -f "sls2/tools/remote/retest.sh" >/dev/null || setsid nohup bash "$SRC/tools/remote/retest.sh" >> "$LOGD/retest.out" 2>&1 < /dev/null &
# 首通探测（tools/remote/unsolved_probe.sh，2026-11-02 起，决策记录 250）：只打种子池 P1 里全局未通关的种子，一个个快照轮着测，首通的立即进 SL 直播快照池；
# 取代原来的并集补测（union_backfill.sh，决策记录 248，脚本留着、不再调）。定时复测在跑或到点时立即退出
pgrep -f "sls2/tools/remote/unsolved_probe.sh" >/dev/null || setsid nohup bash "$SRC/tools/remote/unsolved_probe.sh" >> "$LOGD/retest.out" 2>&1 < /dev/null &
exit 0
