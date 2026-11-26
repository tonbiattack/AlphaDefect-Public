#!/usr/bin/env bash
# 本机运行：采纳对照线——主线改从对照线的检查点继续训练（对照线整体胜出，连同它分叉后学到的进度一起合入，
# 而不是只把配置改动合进主线）。主线仍在原来的主机上跑；主线自己分叉后的那段进度弃用（记录存档，训练量照样计入看板）。
#   bash tools/remote/adopt.sh "原因" --dry-run      # 只读检查，打印计划
#   ENGINE=1 bash tools/remote/adopt.sh "原因"       # 执行；ENGINE=1 时同时同步引擎并在主线主机上编译
# 步骤：
#   1. 停对照线（放 STOP、存检查点、TAG=采纳 备份），取它的 latest.pt 与 train_config.json 到本机；
#      对照线的 log / eval / exam / calib 与通关回放存档到本机 archive/<对照主机>-<时刻>/
#   2. 停主线（TAG=采纳前 备份），主线的记录同样存档到 archive/<主线主机>-<时刻>-discarded/
#   3. 拼主线的新记录：分叉点之前（version <= branch_version）用主线的，分叉之后（time >= branch_time）用对照线的
#   4. 配置用对照线的，与机器相关的参数换成主线主机原来的；同步代码，上传检查点、配置、记录、回放，启动
#   5. lines.json：对照线记进 history（adopted、archive），主线弃用的那段记成 role=discarded（archive），看板按它们算各机器的训练量
set -euo pipefail
. "$(dirname "$0")/hosts.sh"
DRY= POS=()
for x in "$@"; do case $x in --dry-run) DRY=1 ;; *) POS+=("$x") ;; esac; done
REASON=${POS[0]:-采纳对照线}
M=$(main_host); C=$(challenger_host)
[ -n "$C" ] || die "lines.json 里没有对照线"
host_require "$M"; host_require "$C"
read -r BV BT < <(lines_show | python3 -c 'import json,sys; c=json.load(sys.stdin)["challenger"]; print(c["branch_version"], c["branch_time"])')
RM=$(host_run "$M"); RC=$(host_run "$C"); DM=$(host_deploy "$M"); DC=$(host_deploy "$C")
TS=$(date +%s); ARC_C=archive/$C-$TS; ARC_M=archive/$M-$TS-discarded
STAGE=$REMOTE_LOCAL/transfer/$(date +%Y%m%d-%H%M%S)-adopt-$C-$M
echo "== 采纳对照线：$C（$(host_label "$C")）的检查点成为主线，在 $M（$(host_label "$M")）上继续；分叉点第 $BV 版${DRY:+（预演）}"
echo "  原因：$REASON"
if [ -n "$DRY" ]; then
  $(host_ssh "$C") "ls -la $RC/latest.pt; tail -1 $RC/train.out | grep -o 'version=[0-9]*'" | sed 's/^/  对照：/'
  $(host_ssh "$M") "ls -la $RM/latest.pt; tail -1 $RM/train.out | grep -o 'version=[0-9]*'" | sed 's/^/  主线：/'
  echo "  [预演] 停 $C、$M，存档记录到 $REMOTE_LOCAL/$ARC_C 与 $ARC_M，拼记录，上传到 $M:$RM 并启动${ENGINE:+（同步引擎并编译）}"
  exit 0
fi
FILES="log.jsonl eval.jsonl exam.jsonl calib.jsonl"

echo "-- 1. 停对照线并取检查点"
$(host_ssh "$C") "set -e; R=\$HOME/$RC; cd \$R
$REMOTE_ALIVE
$REMOTE_STOP
TAG=采纳 RUN_DIR=\$R bash $DC/tools/remote/ckpt_backup.sh && echo 已备份 \$(tail -1 backups/backup.log)"
fetch_ckpt "$C" latest "$STAGE" 采纳
mkdir -p "$REMOTE_LOCAL/$ARC_C" "$REMOTE_LOCAL/$ARC_M"
EC=$(host_rsync_e "$C"); AC=$(host_addr "$C"); EM=$(host_rsync_e "$M"); AM=$(host_addr "$M")
rsync -a --ignore-missing-args -e "$EC" $(for f in $FILES; do echo "$AC:$RC/$f"; done) "$REMOTE_LOCAL/$ARC_C/"
rsync -a --ignore-missing-args -e "$EC" "$AC:$RC/replays/" "$REMOTE_LOCAL/$ARC_C/replays/" || true
echo "  对照线记录已存档：$REMOTE_LOCAL/$ARC_C"

echo "-- 2. 停主线并存档它的记录"
$(host_ssh "$M") "set -e; R=\$HOME/$RM; cd \$R
$REMOTE_ALIVE
$REMOTE_STOP
TAG=采纳前 RUN_DIR=\$R bash $DM/tools/remote/ckpt_backup.sh && echo 已备份 \$(tail -1 backups/backup.log)"
rsync -a --ignore-missing-args -e "$EM" $(for f in $FILES; do echo "$AM:$RM/$f"; done) "$REMOTE_LOCAL/$ARC_M/"
OLDCFG=$($(host_ssh "$M") "cat $RM/train_config.json")
echo "  主线记录已存档：$REMOTE_LOCAL/$ARC_M"

echo "-- 3. 拼主线的新记录（分叉点之前用主线的，之后用对照线的）"
mkdir -p "$STAGE/records"
python3 - "$REMOTE_LOCAL/$ARC_M" "$REMOTE_LOCAL/$ARC_C" "$STAGE/records" "$BV" "$BT" $FILES <<'EOF'
import json, os, sys
m_dir, c_dir, out, bv, bt, *files = sys.argv[1:]
bv, bt = int(bv), float(bt)
def read(p):
    rs = []
    if os.path.exists(p):
        for line in open(p, encoding="utf-8"):
            try: rs.append(json.loads(line))
            except json.JSONDecodeError: pass
    return rs
ver = lambda r: r.get("version", r.get("v", 0)) or 0
for f in files:
    a = [r for r in read(os.path.join(m_dir, f)) if ver(r) <= bv]
    b = [r for r in read(os.path.join(c_dir, f)) if r.get("time", bt) >= bt and ver(r) > bv]
    with open(os.path.join(out, f), "w", encoding="utf-8") as fh:
        for r in a + b:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  {f}：主线分叉前 {len(a)} 行 + 对照线分叉后 {len(b)} 行")
EOF

echo "-- 4. 配置、代码、上传、启动"
echo "$OLDCFG" > "$STAGE/.main_old_config.json"
HW=$(read_hw "@$STAGE/.main_old_config.json")
edit_config "$STAGE/train_config.json" '{}' "$HW" > /dev/null
echo "  配置取对照线的，与机器相关的参数用 $M 原来的：$HW"
sync_code "$M"
BUILD=$(remote_build "$M")
rsync -a --partial -e "$EM" "$STAGE/latest.pt" "$STAGE/train_config.json" "$AM:$RM/"
rsync -a -e "$EM" "$STAGE/records/" "$AM:$RM/"
[ -d "$REMOTE_LOCAL/$ARC_C/replays" ] && rsync -a -e "$EM" "$REMOTE_LOCAL/$ARC_C/replays/" "$AM:$RM/replays/" || true
$(host_ssh "$M") "set -e; R=\$HOME/$RM; cd \$R
$BUILD
rm -f STOP; RUN_DIR=\$R bash $DM/tools/remote/start_train.sh; date +%s"

echo "-- 5. 更新 lines.json"
lines_set_challenger null "$REASON"
python3 - "$LINES_JSON" "$ARC_C" "$ARC_M" "$M" "$BV" "$BT" "$REASON" <<'EOF'
import json, os, sys, time
p, arc_c, arc_m, m, bv, bt, reason = sys.argv[1:]
d = json.load(open(p, encoding="utf-8"))
d["history"][-1].update(archive=arc_c, adopted=True)
d["history"].append({"role": "discarded", "host": m, "branch_version": int(bv), "branch_time": float(bt), "archive": arc_m,
                     "ended": time.time(), "end_reason": "主线采纳对照线，主线分叉后的这段弃用：" + reason})
json.dump(d, open(p + ".tmp", "w", encoding="utf-8"), ensure_ascii=False, indent=1); os.replace(p + ".tmp", p)
EOF
echo "== 完成：主线在 $M 上从对照线的第 $CKPT_VERSION 版继续；$C 上 STOP 留着，等下一个分叉"
