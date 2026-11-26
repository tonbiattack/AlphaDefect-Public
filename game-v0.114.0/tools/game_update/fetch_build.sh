#!/usr/bin/env bash
# 拿到《杀戮尖塔2》某个分支的当前版本（Windows 仓库 2868841），照“引擎模块与游戏更新适配.md”第六节第 1 步摆好：
#   ~/.local/share/sls2/game-dl/<分支>-<清单号>/        DepotDownloader 下的完整游戏（约 3 GB）
#   ~/.local/share/sls2/game/<版本>/                     data_sts2_windows_x86_64 + res/localization（eng、zhs、completion.json）+ SHA256SUMS
#   ~/.local/share/sls2/decompiled/<版本>/               ilspycmd 9.1 -p 反编译 sts2.dll（按命名空间分目录，同旧版）
# 再写一份初步对比（文件清单、release_info、反编译源码）到 OUT。不碰直播用的 Steam 安装，不改仓库。
#   BRANCH=public-beta OLD=v0.113.0 OLD_MANIFEST_DIR=<上一版清单目录名> bash tools/game_update/fetch_build.sh
# 第一次用（2026-11-12 升 v0.113.0 时这样做的）：
#   - DepotDownloader（SteamRE 开源，3.4.0 linux-x64）解到 ~/tools/DepotDownloader/；用户在终端里扫码登录一次：
#     DepotDownloader -app 2868840 -depot 2868841 -branch public-beta -os windows -qr -remember-password -manifest-only -dir ~/.local/share/sls2/game-dl/manifest-<buildid>
#     之后用 -username <账号> -remember-password 不用再扫。账号名写在 ~/.config/sls2/steam_user（只放本机，不入库）。
#   - ilspycmd 9.1（~/.dotnet/tools，要 DOTNET_ROLL_FORWARD=Major 才能在 .NET 9 上跑）。
# 版本检查（不用登录）见 watch_steam.sh。
set -euo pipefail
BRANCH=${BRANCH:-public-beta}
OLD=${OLD:-v0.111.0}
CONF=${CONF:-$HOME/.config/sls2}
OUT=${OUT:-$HOME/.local/share/sls2/game-dl/report-$(date +%Y%m%d-%H%M)}
REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
S=$HOME/.local/share/sls2
DD=${DD:-$HOME/tools/DepotDownloader/DepotDownloader}
U=$(cat "$CONF/steam_user")      # Steam 账号名（只放本机，不入库）；登录令牌由 DepotDownloader 自己保存（-remember-password）
mkdir -p "$OUT" "$S/game-dl"
log() { echo "$(date '+%F %T') $*" | tee -a "$OUT/fetch.log"; }

# 1. 先只取清单，拿到清单号
M=$S/game-dl/manifest-$BRANCH-$(date +%Y%m%d-%H%M)
"$DD" -app 2868840 -depot 2868841 -branch "$BRANCH" -os windows -username "$U" -remember-password -manifest-only -dir "$M" < /dev/null > "$OUT/manifest_only.log" 2>&1
MF=$(ls "$M"/manifest_2868841_*.txt | head -1)
GID=$(basename "$MF" .txt | sed 's/manifest_2868841_//')
log "分支 $BRANCH 当前清单 $GID"
D=$S/game-dl/$BRANCH-$GID
# 2. 下完整游戏（已下过就只校验）
log "下载到 $D"
"$DD" -app 2868840 -depot 2868841 -branch "$BRANCH" -os windows -username "$U" -remember-password -dir "$D" -validate < /dev/null > "$OUT/download.log" 2>&1
tail -3 "$OUT/download.log" | tee -a "$OUT/fetch.log"
cp "$MF" "$OUT/"
python3 -I -c "import json,sys; print(json.dumps(json.load(open(sys.argv[1])), ensure_ascii=False))" "$D/release_info.json" | tee "$OUT/release_info_new.json"
V=$(python3 -I -c "import json,sys; print(json.load(open(sys.argv[1]))['version'])" "$D/release_info.json")
C=$(python3 -I -c "import json,sys; print(json.load(open(sys.argv[1]))['commit'])" "$D/release_info.json")
# 3. 版本目录：同名目录已在而 sts2.dll 不同（同版本号的热修），改用 <版本>-<提交>
if [ -d "$S/game/$V" ] && ! cmp -s "$S/game/$V/sts2.dll" "$D/data_sts2_windows_x86_64/sts2.dll"; then V=$V-$C; fi
echo "$V" > "$OUT/new_version.txt"
log "新版本目录名 $V（提交 $C）"
if [ ! -d "$S/game/$V" ]; then
  mkdir -p "$S/game/$V"
  rsync -a "$D/data_sts2_windows_x86_64/" "$S/game/$V/"
  python3 -I "$REPO/tools/extract_pck.py" "$D/SlayTheSpire2.pck" "$S/game/$V/res" localization/eng/ localization/zhs/ localization/completion.json > "$OUT/extract_pck.log" 2>&1
  (cd "$S/game/$V" && find . -maxdepth 1 -type f ! -name SHA256SUMS | sort | xargs sha256sum > SHA256SUMS)
fi
# 4. 反编译
if [ ! -d "$S/decompiled/$V" ]; then
  log "反编译 sts2.dll"
  DOTNET_ROOT=$HOME/.dotnet DOTNET_ROLL_FORWARD=Major nice -n 10 "$HOME/.dotnet/tools/ilspycmd" -p -o "$S/decompiled/$V" "$S/game/$V/sts2.dll" > "$S/decompiled/$V.log" 2>&1
fi
# 5. 初步对比
OM=$(ls "$S"/game-dl/manifest-${OLD_MANIFEST_DIR:-24724944}/manifest_2868841_*.txt | head -1)      # 旧版的文件清单（第一次扫码登录时取的）
python3 -I - "$OM" "$MF" > "$OUT/manifest_diff.txt" <<'EOF'
import sys
def load(p):
    d = {}
    for l in open(p, encoding="utf-8", errors="replace"):
        f = l.split()
        if len(f) >= 5 and f[0].isdigit() and len(f[2]) == 40:
            d[" ".join(f[4:])] = (int(f[0]), f[2])
    return d
a, b = load(sys.argv[1]), load(sys.argv[2])
print(f"旧 {len(a)} 个文件，新 {len(b)} 个")
for k in sorted(set(a) | set(b)):
    if k not in a: print("新增", k, b[k][0])
    elif k not in b: print("删掉", k, a[k][0])
    elif a[k][1] != b[k][1]: print("改了", k, a[k][0], "→", b[k][0])
EOF
(cd "$S/decompiled" && diff -rq "$OLD" "$V" > "$OUT/decompiled_diffq.txt" 2>&1 || true)
(cd "$S/decompiled" && diff -ru "$OLD" "$V" > "$OUT/decompiled.diff" 2>&1 || true)
python3 -I - "$OUT/decompiled_diffq.txt" > "$OUT/decompiled_by_ns.txt" <<'EOF'
import sys, collections
c = collections.Counter()
for l in open(sys.argv[1], encoding="utf-8", errors="replace"):
    if l.startswith("Files "):
        p = l.split()[1].split("/", 1)[1]; c[("改了", p.split("/")[0])] += 1
    elif l.startswith("Only in "):
        side, rest = l[8:].split(": ", 1); top = side.split("/", 1)[0]
        ns = side.split("/")[1] if "/" in side else rest.strip()
        c[("只在" + top, ns)] += 1
for (k, ns), n in sorted(c.items(), key=lambda x: (-x[1], x[0])): print(f"{n:5d} {k} {ns}")
EOF
log "对比写好：$OUT（manifest_diff.txt、decompiled_by_ns.txt、decompiled.diff）"
echo "$V"
