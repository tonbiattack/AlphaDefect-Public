#!/usr/bin/env bash
# 本机常驻：每 EVERY 秒（默认 30；A 每 A_EVERY 轮一次，默认 2）从两台租用机各取一份精简统计（tools/remote/lite_stats.py，gzip 压缩后十几 KB），
# 再生成轻量看板（tools/dashboard_lite.py）。2026-11-03 起取代 pull.sh（用户：“其他的维持统计存在训练机，不要自动回传了”）：
# 不再拉 log.jsonl、eval.jsonl、resource.jsonl 等完整统计。
#   bash tools/remote/pull_lite.sh          # 拉一次
#   LOOP=1 bash tools/remote/pull_lite.sh   # 常驻（本机守护 local_watch.sh 起）
# 兼容：学习端的 status.json、watchdog.json、config.json 从精简统计里抽出来写到原来的本机目录（巡检脚本、复测脚本读它们）；
# 成功时刻照旧写 pull.ok（A）/ pull-b.ok（B）。评测留档的权重（best/、high/）照旧备份，但改为每 60 分钟一次、只传新文件。
set -u
. "$(dirname "$0")/hosts.sh"
EVERY=${EVERY:-30}        # 60 → 30 秒（2026-11-04，用户：“看板的更新频率……再增加一下，当然不要增加太多流量消耗”；同时把学习端统计里的走势只留两个字段，单次约 25 → 13 KB）
A_EVERY=${A_EVERY:-2}     # A（评测机）每几轮拉一次（每 60 秒）：评测约 6 分钟才出一次结果
N=0
SRC=$(cd "$(dirname "$0")/../.." && pwd)
# 看板目标（2026-11-06 起，储君训练）：$REMOTE_LOCAL/lite_target.json 的 deploy（远端部署目录）、run / eval（学习端、评测的运行目录名）、
# name（本机存放兼容文件与评测留档的目录名前缀）；没有这个文件时是故障机器人主线（部署目录取 hosts.json，运行目录 defect、defect-eval），与原来相同。
# 每轮重读，改了文件下一轮生效
tget() { python3 -c 'import json, os, sys; p = sys.argv[1]; d = json.load(open(p)) if os.path.exists(p) else {}; print(d.get(sys.argv[2]) or sys.argv[3])' "$REMOTE_LOCAL/lite_target.json" "$1" "$2"; }
once() {
  local h role tmp dep run ev name
  N=$((N + 1))
  run=$(tget run defect); ev=$(tget eval defect-eval); name=$(tget name defect)
  # 学习端、评测各在哪台机器（目标文件的 train_host、eval_host；默认 B 学习、A 评测）；本机文件按角色命名：lite_b.json 学习端、lite_a.json 评测
  local th eh f
  th=$(tget train_host b); eh=$(tget eval_host a)
  for role in train eval; do
    if [ $role = train ]; then h=$th; f=b; else h=$eh; f=a; fi
    [ $role = eval ] && [ -s "$REMOTE_LOCAL/lite_a.json" ] && [ $(( (N - 1) % A_EVERY )) -ne 0 ] && continue
    tmp=$REMOTE_LOCAL/lite_$f.json.tmp
    dep=$(tget deploy "$(host_deploy $h)")
    if timeout 120 $(host_ssh $h) "cd $dep && LITE_RUN=$run LITE_EVAL=$ev .venv/bin/python -I tools/remote/lite_stats.py $role | gzip -c" < /dev/null 2>> "$REMOTE_LOCAL/pull$(host_pull_tag $h).err" \
        | gunzip > "$tmp" 2>/dev/null && python3 -c "import json,sys; json.load(open(sys.argv[1]))" "$tmp" 2>/dev/null; then
      mv -f "$tmp" "$REMOTE_LOCAL/lite_$f.json"; date +%s > "$REMOTE_LOCAL/pull$(host_pull_tag $h).ok"
    else
      rm -f "$tmp"
    fi
  done
  local cdir; cdir=$(host_local b); [ "$name" = defect ] || cdir=$REMOTE_LOCAL/$name-b
  python3 - "$REMOTE_LOCAL/lite_b.json" "$cdir" <<'PY'
import json, os, sys
b = json.load(open(sys.argv[1])); d = sys.argv[2]; os.makedirs(d, exist_ok=True)
for k in ("status", "watchdog", "config"):
    if b.get(k) is not None:
        t = os.path.join(d, k + ".json.tmp"); json.dump(b[k], open(t, "w"), ensure_ascii=False); os.replace(t, os.path.join(d, k + ".json"))
PY
  python3 "$SRC/tools/dashboard_lite.py" 2>> "$REMOTE_LOCAL/dashboard.log"
  mkdir -p "$REMOTE_LOCAL/$ev"
  K=$REMOTE_LOCAL/$ev/keep-a.ts
  if [ ! -e "$K" ] || [ $(( $(date +%s) - $(stat -c %Y "$K") )) -ge 3600 ]; then
    touch "$K"
    # 评测留档（best、high）从评测所在的机器拉（eval_host；2026-11-09 A 到期、评测迁到 B 之前写死为 A）
    ( timeout 1800 rsync -a --partial --ignore-existing --ignore-missing-args -e "$(host_rsync_e $eh)" \
        "$(host_addr $eh):.local/share/sls2/runs/$ev/best" "$(host_addr $eh):.local/share/sls2/runs/$ev/high" \
        "$REMOTE_LOCAL/$ev/" 2>/dev/null ) &
  fi
}
if [ -z "${LOOP:-}" ]; then once; exit; fi
while true; do once; sleep "$EVERY"; done
