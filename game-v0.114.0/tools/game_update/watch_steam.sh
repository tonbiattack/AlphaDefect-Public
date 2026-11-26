#!/usr/bin/env bash
# 每 20 分钟匿名查一次《杀戮尖塔2》（appid 2868840）各分支的 buildid 和 Windows 仓库（2868841）的清单号；与基线不同就打印并退出
# （放在会话的后台任务里，退出时会话醒来），到 UNTIL（默认明早 8 点）也退出。用 steamcmd 的独立副本 W，不碰登录用的那份和 Windows 的 Steam。
#   BASE="$(tail -1 ~/.local/share/sls2/game-dl/watch/log.txt | cut -d" " -f3-)" FIRST_SLEEP=1200 bash tools/game_update/watch_steam.sh
set -u
W=${W:-/mnt/c/steamcmd-watch}      # steamcmd（Windows 版，Valve 官方 steamcmd.zip）的独立副本，只做匿名查询
OUT=${OUT:-$HOME/.local/share/sls2/game-dl/watch}
UNTIL=${UNTIL:-$(date -d 'tomorrow 08:00' +%s)}
BASE=${BASE:?要给基线，形如 "public=23811903 public-beta=26625880 | win:public=… win:public-beta=…"（上一次的 log.txt 末行）}
FIRST_SLEEP=${FIRST_SLEEP:-0}
mkdir -p "$OUT"
sleep "$FIRST_SLEEP"
FAIL=0
while :; do
  rm -f "$W/appcache/appinfo.vdf"
  (cd "$W" && timeout 300 ./steamcmd.exe +login anonymous +app_info_update 1 +app_info_print 2868840 +quit) 2>&1 | tr -d '\r' > "$OUT/last.txt"
  cur=$(python3 -I - "$OUT/last.txt" <<'EOF'
import re, sys
t = open(sys.argv[1], encoding="utf-8", errors="replace").read()
def block(s, key):
    i = s.find(f'"{key}"')
    if i < 0: return ""
    j = s.index("{", i); d = 0
    for k in range(j, len(s)):
        if s[k] == "{": d += 1
        elif s[k] == "}":
            d -= 1
            if d == 0: return s[j:k + 1]
    return ""
dep = block(t, "depots")
br = re.findall(r'"([^"]+)"\s*\{[^{}]*?"buildid"\s*"(\d+)"', block(dep, "branches"))
win = re.findall(r'"([^"]+)"\s*\{[^{}]*?"gid"\s*"(\d+)"', block(block(dep, "2868841"), "manifests"))
if br:
    print(" ".join(f"{a}={b}" for a, b in sorted(br)) + " | " + " ".join(f"win:{a}={b}" for a, b in sorted(win)))
EOF
)
  echo "$(date '+%F %T') ${cur:-（没取到）}" >> "$OUT/log.txt"
  if [ -z "$cur" ]; then
    FAIL=$((FAIL + 1))
    [ $FAIL -ge 3 ] && { echo "连续 3 次没取到版本信息"; tail -20 "$OUT/last.txt"; exit 0; }
  else
    FAIL=0
    if [ "$cur" != "$BASE" ]; then
      cp "$OUT/last.txt" "$OUT/appinfo_new_$(date +%Y%m%d-%H%M).txt"
      echo "发现新版本：$cur"; echo "原来：$BASE"; exit 0
    fi
  fi
  [ "$(date +%s)" -ge "$UNTIL" ] && { echo "今晚检查到点结束：$cur"; exit 0; }
  sleep 1200
done
