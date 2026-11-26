#!/usr/bin/env bash
# 本机运行：从主线（lines.json 的 main）最新的检查点分叉一条对照训练线到主机 HOST（经本机中转），按 JSON 改动配置后启动，
# 登记到 lines.json 的 challenger（branch_version、branch_time、main_host、change、note）。主线的训练不停、不改。
#   bash tools/remote/fork.sh a '{"ent_coef_macro": 0.02}' "对照：局外熵 0.03→0.02" --dry-run
#   bash tools/remote/fork.sh a '{"ent_coef_macro": 0.02}' "对照：局外熵 0.03→0.02"
#   bash tools/remote/fork.sh --end ["结束原因"] [--dry-run]     # 结束对照：停 HOST 上的对照训练（放 STOP、存检查点、TAG=对照结束 备份），清空 challenger
# 选项：
#   --src latest|backup|fresh   检查点来源（默认 latest：主线运行目录的 latest.pt，训练进程先写临时文件再改名，正在训练时取也完整；
#                               backup 取最近一次定时备份；fresh 先在主线主机上备份一次再取）
#   --hw '<JSON>' 或 --hw @文件  与机器相关的参数（只取 hosts.sh 的 HW_KEYS），默认沿用 HOST 原有配置里的这些键
#   --allow-code-diff           本机代码与主线主机上的不一致时仍继续（默认拒绝：对照线必须和主线跑同一份代码，只差配置）
#   ENGINE=1                    同时同步引擎源码与 catalog 并在 HOST 上编译
# 带过去的文件：latest.pt、train_config.json（按改动修改）、solver/labeled.jsonl（让两条线从同一批 Boss 练习题出发）、exam.json（考同一套固定考卷）；
#   另写 fork.json 记录分叉信息。不带 log.jsonl、eval.jsonl、best/、replays/、status.json：对照线的记录从分叉点开始，
#   版本号接着检查点的版本往下数，按“分叉后同样多的版本”和主线比。
# HOST 上原来的训练会被停掉：先放 STOP 并等学习进程存好检查点，再备份一次（TAG=分叉前），然后把整个运行目录改名为
#   <运行目录>.before-fork-<时刻> 保留，不删除。
# 注意：求解器标注（tools/solver/label_loop.py）只服务主线，对照线的标签停留在分叉时那一份，Boss 练习题不再更新。
set -euo pipefail
. "$(dirname "$0")/hosts.sh"
DRY= SRC=latest HW_ARG= END= ALLOW_DIFF= POS=()
while [ $# -gt 0 ]; do
  case $1 in
    --dry-run) DRY=1 ;;
    --end) END=1 ;;
    --src) SRC=$2; shift ;;
    --hw) HW_ARG=$2; shift ;;
    --allow-code-diff) ALLOW_DIFF=1 ;;
    -h|--help) sed -n '2,/^set -euo/p' "$0" | sed '$d'; exit 0 ;;
    *) POS+=("$1") ;;
  esac
  shift
done

# ---------- 结束对照 ----------
if [ -n "$END" ]; then
  C=$(challenger_host)
  [ -n "$C" ] || die "lines.json 里没有对照线"
  REASON=${POS[0]:-手动结束}
  host_require "$C"
  RC=$(host_run "$C"); DC=$(host_deploy "$C")
  echo "== 结束对照线：$C（$(host_label "$C")）${DRY:+（预演）}"
  lines_show | python3 -c 'import json,sys; c=json.load(sys.stdin)["challenger"]; print("  ", json.dumps(c, ensure_ascii=False))'
  if [ -n "$DRY" ]; then
    echo "  [预演] 将在 $C 上：放 STOP、TERM 学习进程（存检查点）、TAG=对照结束 备份；STOP 留着，守护不再拉起"
    echo "  [预演] lines.json 的 challenger 置空（原记录进 history，原因：$REASON）"
    exit 0
  fi
  $(host_ssh "$C") "set -e; R=\$HOME/$RC; cd \$R
$REMOTE_ALIVE
$REMOTE_STOP
if [ -f latest.pt ]; then TAG=对照结束 RUN_DIR=\$R bash $DC/tools/remote/ckpt_backup.sh && echo 已备份 \$(tail -1 backups/backup.log); fi"
  # 对照线的训练与评测记录存档到本机：之后分叉会把这台机器的运行目录换新，看板算各机器、各线的累计训练量要用这段记录
  ARC=archive/$C-$(date +%s)
  mkdir -p "$REMOTE_LOCAL/$ARC"
  rsync -a --ignore-missing-args -e "$(host_rsync_e "$C")" "$(host_addr "$C"):$RC/log.jsonl" "$(host_addr "$C"):$RC/eval.jsonl" "$REMOTE_LOCAL/$ARC/" \
    && echo "  对照线记录已存档：$REMOTE_LOCAL/$ARC" || { echo "  存档对照线记录失败（看板的累计训练量会少算这一段）"; ARC=; }
  lines_set_challenger null "$REASON"
  [ -z "$ARC" ] || python3 - "$LINES_JSON" "$ARC" <<'PY'
import json, os, sys
p, arc = sys.argv[1:]
d = json.load(open(p, encoding="utf-8"))
d["history"][-1]["archive"] = arc
json.dump(d, open(p + ".tmp", "w", encoding="utf-8"), ensure_ascii=False, indent=1); os.replace(p + ".tmp", p)
PY
  echo "== 完成：$C 上的对照训练已停（STOP 留着），运行目录 ~/$RC 原样保留"
  exit 0
fi

# ---------- 分叉 ----------
[ ${#POS[@]} -ge 2 ] || die "用法：fork.sh HOST '<JSON 改动>' \"说明\" [--dry-run] [--src latest|backup|fresh] [--hw JSON|@文件]；结束：fork.sh --end"
H=${POS[0]} CHANGE=$(check_json_obj "${POS[1]}") NOTE=${POS[2]:-}
[ "$CHANGE" != '{}' ] || die "改动不能为空"
M=$(main_host)
host_require "$M"
host_exists "$H" || die "hosts.json 里没有主机 $H"
[ "$H" != "$M" ] || die "$H 是主线所在的主机，不能把对照线分叉到它上面"
[ -z "$(challenger_host)" ] || die "已有对照线（在 $(challenger_host) 上），先 bash tools/remote/fork.sh --end"
H_PENDING=; host_pending "$H" && H_PENDING=1
[ -n "$H_PENDING" ] && [ -z "$DRY" ] && host_require "$H"
[ -n "$HW_ARG" ] && HW_ARG=$(read_hw "$HW_ARG")
# 改动的键必须是训练配置里有的参数（防止拼错后悄悄不起作用）
PYV=$REPO/.venv/bin/python
if [ -x "$PYV" ]; then
  (cd "$REPO/python" && "$PYV" -c '
import dataclasses, json, sys
from sts2rl.config import TrainConfig
bad = [k for k in json.loads(sys.argv[1]) if k not in {f.name for f in dataclasses.fields(TrainConfig)}]
sys.exit(f"错误：训练配置里没有这些参数：{bad}" if bad else 0)' "$CHANGE")
fi

echo "== 分叉对照线：主线 $M（$(host_label "$M")）→ $H（$(host_label "$H")），改动 $CHANGE${NOTE:+，$NOTE}${DRY:+（预演，不做任何改动）}"
echo "-- 目标机检查"
H_ALIVE=0
if [ -n "$H_PENDING" ]; then
  echo "  $H 还没登记地址（pending），预演跳过目标机检查；正式执行会报错退出"
else
  PF=$(preflight "$H") || die "连不上 $H"
  echo "$PF" | sed '$d' | sed 's/^/  /'
  [ "$(echo "$PF" | tail -1)" = ALIVE=1 ] && H_ALIVE=1
  echo "  $H 上训练$([ $H_ALIVE = 1 ] && echo 正在跑，将被停掉 || echo 没在跑)"
  if echo "$PF" | grep -q '^缺少' && [ -z "$DRY" ]; then die "$H 的部署不完整（见上），先运行 setup.sh"; fi
fi
echo "-- 代码一致性（本机 → 主线 $M，只读比较）"
DIFF=$(code_diff "$M")
if [ -z "$DIFF" ]; then echo "  一致：$H 将运行与主线相同的代码"
else
  echo "$DIFF" | head -20 | sed 's/^/  /'; echo "  共 $(echo "$DIFF" | wc -l) 处不同"
  [ -n "$ALLOW_DIFF" ] || [ -n "$DRY" ] || die "本机代码与主线不一致，对照线会跑另一份代码；先用 restart.sh 同步主线，或加 --allow-code-diff"
fi

TS=$(date +%Y%m%d-%H%M%S)
STAGE=$REMOTE_LOCAL/transfer/$TS-fork-$M-$H
echo "-- 取主线检查点（$SRC）"
fetch_ckpt "$M" "$SRC" "$STAGE" 分叉 "$DRY"
RM=$(host_run "$M"); RH=$(host_run "$H"); DH=$(host_deploy "$H")
if [ -n "$DRY" ]; then
  cat <<EOF
-- 计划（正式执行时）
  1. 从 $M 复制 solver/labeled.jsonl、exam.json 到本机 $STAGE；配置按 $CHANGE 修改，与机器相关的参数：${HW_ARG:-沿用 $H 原有配置}
  2. $H 上：放 STOP，停训练（等检查点存好），TAG=分叉前 备份，运行目录 ~/$RH 改名为 ~/$RH.before-fork-$TS
  3. 同步本机 python/ 与远端脚本到 $H:$DH${ENGINE:+，同步引擎并编译}
  4. 新建 ~/$RH（先放 STOP），上传检查点、配置、标签与 fork.json，核对版本后删 STOP，用 start_train.sh 启动
  5. lines.json 的 challenger 写为 {host: $H, branch_version, branch_time, main_host: $M, change: $CHANGE, note}
EOF
  echo "  当前 lines.json："; lines_show | sed 's/^/    /'
  exit 0
fi
[ -n "$CKPT_VERSION" ] || die "不知道检查点的版本（本机没有 .venv 读不了检查点），停止"

echo "-- 复制求解器标签"
mkdir -p "$STAGE/solver"
rsync -a --ignore-missing-args -e "$(host_rsync_e "$M")" "$(host_addr "$M"):$RM/solver/labeled.jsonl" "$STAGE/solver/"
# 固定考卷：两条线考同一套题，按题配对比较
rsync -a --ignore-missing-args -e "$(host_rsync_e "$M")" "$(host_addr "$M"):$RM/exam.json" "$STAGE/"
echo "-- 配置"
HW=$HW_ARG
if [ -z "$HW" ]; then
  OLD=$($(host_ssh "$H") "cat $RH/train_config.json 2>/dev/null" || true)
  if [ -n "$OLD" ]; then echo "$OLD" > "$STAGE/.host_old_config.json"; HW=$(read_hw "@$STAGE/.host_old_config.json"); else HW='{}'; fi
fi
BEFORE=$(edit_config "$STAGE/train_config.json" "$CHANGE" "$HW")
echo "  改动 $BEFORE → $CHANGE；与机器相关的参数：$HW"
NOW=$(date +%s)
REC=$(python3 -c '
import json, sys
h, v, t, m, ch, be, note, ck, now = sys.argv[1:]
print(json.dumps({"host": h, "branch_version": int(v), "branch_time": float(t), "main_host": m, "change": json.loads(ch),
                  "before": json.loads(be), "note": note, "ckpt": ck, "started": float(now)}, ensure_ascii=False))' \
  "$H" "$CKPT_VERSION" "$CKPT_TIME" "$M" "$CHANGE" "$BEFORE" "$NOTE" "$CKPT_SRC" "$NOW")
echo "$REC" > "$STAGE/fork.json"

echo "-- 停 $H 上原来的训练并保留其运行目录"
$(host_ssh "$H") "set -e; R=\$HOME/$RH
if [ -d \$R ]; then
  cd \$R
  $REMOTE_ALIVE
  $REMOTE_STOP
  if [ -f latest.pt ]; then TAG=分叉前 RUN_DIR=\$R bash $DH/tools/remote/ckpt_backup.sh || { echo 备份失败，未分叉（STOP 仍在）; exit 1; }; echo 已备份 \$(tail -1 backups/backup.log); fi
  cd \$HOME; mv \$R \$R.before-fork-$TS; echo \"原运行目录已改名为 \$R.before-fork-$TS\"
fi
mkdir -p \$R; touch \$R/STOP"
echo "-- 同步代码到 $H"
sync_code "$H"
echo "-- 上传到 $H:~/$RH"
rsync -a --partial -e "$(host_rsync_e "$H")" --exclude .host_old_config.json "$STAGE/" "$(host_addr "$H"):$RH/"
BUILD=$(remote_build "$H")
$(host_ssh "$H") "set -e; R=\$HOME/$RH; cd \$R
v=\$($DH/.venv/bin/python -c 'import sys, torch; print(torch.load(sys.argv[1], map_location=\"cpu\", weights_only=False)[\"version\"])' latest.pt)
[ \"\$v\" = $CKPT_VERSION ] || { echo \"$H 上的检查点版本 \$v 与本机 $CKPT_VERSION 不符，未启动（STOP 仍在）\"; exit 1; }
echo \"$H 上检查点核对无误：版本 \$v\"
$BUILD
rm STOP; RUN_DIR=\$R bash $DH/tools/remote/start_train.sh; date +%s"
lines_set_challenger "$REC" "新对照线替换"
echo "== 完成：对照线在 $H 上从版本 $CKPT_VERSION 起训练，改动 $CHANGE。结束：bash tools/remote/fork.sh --end"
echo "   中转文件：$STAGE"
