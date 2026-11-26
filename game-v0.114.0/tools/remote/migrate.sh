#!/usr/bin/env bash
# 本机运行：把主线训练从主机 FROM 迁到主机 TO（经本机中转，两台之间不直接连），在 TO 上启动，并把 lines.json 的主线改为 TO。
# FROM 上的训练不停（之后要在 FROM 上跑对照线时，用 fork.sh 分叉过去，那时才会停 FROM 上原来的训练）。
#   bash tools/remote/migrate.sh a b --dry-run         # 只做只读查询，打印计划（目标机待填时也能预演）
#   bash tools/remote/migrate.sh a b                   # 取 FROM 最近一次定时备份（ckpt_backup，最多旧 30 分钟）
#   bash tools/remote/migrate.sh a b --src fresh       # 先在 FROM 上备份一次（TAG=迁移，不停训练）再取：最新，但会在 FROM 的 backups/ 里多一份
#   bash tools/remote/migrate.sh a b --src latest      # 直接取 FROM 的 latest.pt（训练进程先写临时文件再改名，取到的总是完整的）
# 其他选项：
#   --hw '<JSON>' 或 --hw @文件   与机器相关的参数（只取 hosts.sh 里 HW_KEYS 列的键，例如 @tools/remote/train_config.4090.json）；
#                               默认沿用 TO 原有 train_config.json 里的这些键，TO 没有旧配置时沿用 FROM 的
#   --no-start                  复制完不启动（之后 RUN_DIR=~/运行目录 bash 部署目录/tools/remote/start_train.sh）
#   ENGINE=1                    同时同步引擎源码与 catalog，并在 TO 上编译
# 本机代码（python/ 与远端脚本）会同步到 TO；本机与 FROM 的代码不一致时只提示（FROM 跑的是它自己的那份）。
#
# 搬过去的文件（运行目录里需要随训练延续的）：
#   latest.pt             检查点（权重、优化器状态、版本、课程进度、打分头步数）
#   train_config.json     检查点配套的配置（备份时的那份）
#   solver/labeled.jsonl  求解器标签：采集进程启动时从头读，决定 Boss 练习题；不搬就要从零攒
#   status.json           只为沿用 started_at（看板的已训练时长），启动后即被重写
#   log.jsonl、eval.jsonl 训练与评测记录，截到检查点的版本为止（比检查点新的记录删掉，免得续训后同一版本出现两次）；
#                         评测进程据 eval.jsonl 跳过已评过的版本
#   best/                 评测最好的几版权重与索引（训练跑偏时从这里重来）
#   replays/              通关回放（回放页从主线主机拉）
#   exam.json             固定考卷（评测每 500 版打一遍；求解器练习池也据此剔除这些局面）
#   exam.jsonl、calib.jsonl  考卷成绩与标定记录，同样截到检查点的版本
# 不搬：backups/（几 GB，目标机重新开始定时备份）、stoploss/（基线是来源机上那次改动的）、solver/pending/（待标注快照，标注只服务主线，
#   迁移后 FROM 上剩下的不再标注）、resource.jsonl（换机器后不可比）、config.json（启动时重写）、engine.log、train.out、watchdog.*、
#   learner.pid、STOP、archive/、mainline-*.pt、batch_dump.pkl。
# TO 的运行目录已有内容时，整个改名为 <运行目录>.before-migrate-<时刻> 保留，不删除；TO 上训练正在跑时拒绝执行。
# 中转文件留在本机 ~/.local/share/sls2/runs/remote/transfer/ 下（含检查点，约 0.4 GB/次，可手动删）。
set -euo pipefail
. "$(dirname "$0")/hosts.sh"
DRY= SRC=backup HW_ARG= START=1 POS=()
while [ $# -gt 0 ]; do
  case $1 in
    --dry-run) DRY=1 ;;
    --src) SRC=$2; shift ;;
    --hw) HW_ARG=$2; shift ;;
    --no-start) START= ;;
    -h|--help) sed -n '2,/^set -euo/p' "$0" | sed '$d'; exit 0 ;;
    *) POS+=("$1") ;;
  esac
  shift
done
[ ${#POS[@]} = 2 ] || die "用法：migrate.sh FROM TO [--dry-run] [--src backup|fresh|latest] [--hw JSON|@文件] [--no-start]"
FROM=${POS[0]} TO=${POS[1]}
[ "$FROM" != "$TO" ] || die "FROM 与 TO 相同"
host_require "$FROM"
host_exists "$TO" || die "hosts.json 里没有主机 $TO"
TO_PENDING=; host_pending "$TO" && TO_PENDING=1
[ -n "$TO_PENDING" ] && [ -z "$DRY" ] && host_require "$TO"
[ "$(challenger_host)" != "$TO" ] || die "$TO 上正跑着对照线，先 bash tools/remote/fork.sh --end"
[ "$(main_host)" != "$TO" ] || die "主线已经在 $TO 上"
[ -n "$HW_ARG" ] && HW_ARG=$(read_hw "$HW_ARG")
[ "$(main_host)" = "$FROM" ] || echo "注意：lines.json 里的主线在 $(main_host)，不在 $FROM；迁移后主线改为 $TO"

echo "== 迁移主线：$FROM（$(host_label "$FROM")）→ $TO（$(host_label "$TO")）${DRY:+（预演，不做任何改动）}"
echo "-- 目标机检查"
if [ -n "$TO_PENDING" ]; then
  echo "  $TO 还没登记地址（pending），预演跳过目标机检查；正式执行会报错退出"
else
  PF=$(preflight "$TO") || die "连不上 $TO"
  echo "$PF" | sed '$d' | sed 's/^/  /'
  [ "$(echo "$PF" | tail -1)" = ALIVE=0 ] || die "$TO 上训练正在跑，不覆盖（要换掉它先停：touch 运行目录/STOP 后 kill 学习进程）"
  if echo "$PF" | grep -q '^缺少' && [ -z "$DRY" ]; then die "$TO 的部署不完整（见上），先运行 setup.sh"; fi
fi
echo "-- 代码一致性（本机 → $FROM，只读比较）"
DIFF=$(code_diff "$FROM"); [ -z "$DIFF" ] && echo "  一致" || { echo "$DIFF" | head -20 | sed 's/^/  /'; echo "  共 $(echo "$DIFF" | wc -l) 处不同：python/ 以本机为准同步到 $TO；引擎与 catalog 只在 ENGINE=1 时同步"; }

TS=$(date +%Y%m%d-%H%M%S)
STAGE=$REMOTE_LOCAL/transfer/$TS-migrate-$FROM-$TO
echo "-- 取检查点（$SRC）"
fetch_ckpt "$FROM" "$SRC" "$STAGE" 迁移 "$DRY"
RF=$(host_run "$FROM")
[ -n "$DRY" ] || [ -n "$CKPT_VERSION" ] || die "不知道检查点的版本（本机没有 .venv 读不了检查点），停止"
if [ -n "$DRY" ]; then
  RT=$(host_run "$TO"); DT=$(host_deploy "$TO")
  cat <<EOF
-- 计划（正式执行时）
  1. 从 $FROM 复制 solver/labeled.jsonl、status.json、log.jsonl、eval.jsonl、best/、replays/、exam.json、exam.jsonl、calib.jsonl
     到本机 $STAGE，各记录截到检查点版本为止
  2. 与机器相关的参数：${HW_ARG:-沿用 $TO 原有配置（没有则沿用 $FROM）}
  3. 同步本机 python/ 与远端脚本到 $TO:$DT${ENGINE:+，同步引擎并编译}
  4. $TO 上 ~/$RT 若有内容，改名为 ~/$RT.before-migrate-$TS；放 STOP 后上传，核对检查点版本
  5. $( [ -n "$START" ] && echo "删 STOP，RUN_DIR=~/$RT bash $DT/tools/remote/start_train.sh 启动" || echo "不启动（--no-start），STOP 留着")
  6. lines.json 主线改为 $TO（原记录进 history）；$FROM 上的训练照常运行
EOF
  echo "  当前 lines.json："; lines_show | sed 's/^/    /'
  exit 0
fi

echo "-- 复制随训练延续的文件"
E=$(host_rsync_e "$FROM"); A=$(host_addr "$FROM")
mkdir -p "$STAGE/solver"
rsync -a --ignore-missing-args -e "$E" "$A:$RF/solver/labeled.jsonl" "$STAGE/solver/"
rsync -a --ignore-missing-args -e "$E" "$A:$RF/status.json" "$A:$RF/log.jsonl" "$A:$RF/eval.jsonl" "$A:$RF/best" "$A:$RF/replays" \
  "$A:$RF/exam.json" "$A:$RF/exam.jsonl" "$A:$RF/calib.jsonl" "$STAGE/"
python3 - "$STAGE" "$CKPT_VERSION" <<'EOF'
import json, os, sys
stage, v = sys.argv[1], int(sys.argv[2])
for name in ("log.jsonl", "eval.jsonl", "exam.jsonl", "calib.jsonl"):
    p = os.path.join(stage, name)
    if not os.path.exists(p):
        continue
    keep, drop = [], 0
    for line in open(p, encoding="utf-8"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            drop += 1; continue          # 复制时正好写到一半的最后一行
        if (r.get("version") or r.get("v") or 0) <= v:
            keep.append(line if line.endswith("\n") else line + "\n")
        else:
            drop += 1
    open(p + ".tmp", "w", encoding="utf-8").writelines(keep); os.replace(p + ".tmp", p)
    print(f"  {name}：保留 {len(keep)} 行，删去版本 {v} 之后的 {drop} 行")
EOF
du -sh "$STAGE" | sed 's/^/  本机中转 /'

echo "-- 配置"
HW=$HW_ARG
if [ -z "$HW" ]; then
  OLD=$($(host_ssh "$TO") "cat $(host_run "$TO")/train_config.json 2>/dev/null" || true)
  if [ -n "$OLD" ]; then echo "$OLD" > "$STAGE/.to_old_config.json"; HW=$(read_hw "@$STAGE/.to_old_config.json"); else HW='{}'; fi
fi
edit_config "$STAGE/train_config.json" '{}' "$HW" > /dev/null
echo "  与机器相关的参数：$HW"
printf '{"from": "%s", "to": "%s", "version": %s, "ckpt": "%s", "time": %s}\n' "$FROM" "$TO" "$CKPT_VERSION" "$CKPT_SRC" "$(date +%s)" > "$STAGE/migrate.json"

echo "-- 同步代码到 $TO"
sync_code "$TO"
RT=$(host_run "$TO"); DT=$(host_deploy "$TO")
echo "-- 准备 $TO 的运行目录"
$(host_ssh "$TO") "set -e; R=\$HOME/$RT
if [ -d \$R ] && [ -n \"\$(ls -A \$R)\" ]; then mv \$R \$R.before-migrate-$TS; echo \"  原运行目录已改名为 \$R.before-migrate-$TS\"; fi
mkdir -p \$R; touch \$R/STOP"
echo "-- 上传到 $TO:~/$RT"
rsync -a --partial -e "$(host_rsync_e "$TO")" --exclude .to_old_config.json "$STAGE/" "$(host_addr "$TO"):$RT/"
BUILD=$(remote_build "$TO")
$(host_ssh "$TO") "set -e; R=\$HOME/$RT; cd \$R
v=\$($DT/.venv/bin/python -c 'import sys, torch; print(torch.load(sys.argv[1], map_location=\"cpu\", weights_only=False)[\"version\"])' latest.pt)
[ \"\$v\" = $CKPT_VERSION ] || { echo \"$TO 上的检查点版本 \$v 与本机 $CKPT_VERSION 不符，未启动（STOP 仍在）\"; exit 1; }
echo \"  $TO 上检查点核对无误：版本 \$v\"
$BUILD
if [ -n '$START' ]; then rm STOP; RUN_DIR=\$R bash $DT/tools/remote/start_train.sh; date +%s; else echo '  未启动（--no-start），STOP 留着'; fi"

lines_set_main "$(printf '{"host": "%s", "since": %s, "note": "主线（%s 版本 %s 迁来）", "from_host": "%s", "from_version": %s, "from_ckpt": "%s"}' \
  "$TO" "$(date +%s)" "$FROM" "$CKPT_VERSION" "$FROM" "$CKPT_VERSION" "$CKPT_SRC")" "迁到 $TO"
echo "== 完成：主线在 $TO（版本 $CKPT_VERSION 起）。$FROM 上原来的训练仍在跑；要在它上面跑对照线：bash tools/remote/fork.sh $FROM '<JSON 改动>' \"说明\""
echo "   中转文件：$STAGE"
