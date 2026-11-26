#!/usr/bin/env bash
# 用法：bash tools/solver/solve_all.sh 配置.jsonl [每场限时秒数]；模组文件放在 ~/.local/share/sls2/mods/{CombatSolver,RitsuLib}（从创意工坊目录复制，不进仓库）
# 逐场试解：每场一个进程，求解限时 $2 秒；输出每场一行 JSON（含整个进程的峰值内存 max_rss_mb）
S=$(dirname "$0"); F=$1; T=${2:-600}
export DOTNET_ROOT=$HOME/.dotnet PATH=$HOME/.dotnet:$PATH DOTNET_CLI_TELEMETRY_OPTOUT=1
export COMBAT_SOLVER_DIR=$HOME/.local/share/sls2/mods/CombatSolver RITSULIB_DIR=$HOME/.local/share/sls2/mods/RitsuLib
D=$HOME/dev/sls2/engine/Sts2Headless/bin/Release/net9.0/Sts2Headless.dll
n=$(wc -l < "$F")
for ((i=0; i<n; i++)); do
  out=$( { /usr/bin/time -f "MAXRSS %M" timeout $((T + 120)) dotnet "$D" solve-test "$F" $i --timeout-s $T 2>&1; } 2>&1 )
  json=$(echo "$out" | grep '^{"label"' | tail -1)
  rss=$(echo "$out" | grep -o 'MAXRSS [0-9]*' | awk '{print int($2/1024)}')
  if [ -n "$json" ]; then echo "${json%\}},\"i\":$i,\"max_rss_mb\":${rss:-null}}"; else echo "{\"i\":$i,\"crash\":$(echo "$out" | grep -vE 'Sentry|^\s+at ' | tail -3 | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()[-400:]))'),\"max_rss_mb\":${rss:-null}}"; fi
done
