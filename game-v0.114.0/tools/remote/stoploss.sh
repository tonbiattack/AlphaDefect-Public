#!/usr/bin/env bash
# 训练调整的自动止损：由 cron 每 10 分钟调用，按 stoploss/state.json 里记的改动前基线，拿改动后的评测合计和它比，
# 明显变差就停训练、换回改动前的检查点和配置、重启，不依赖有人在线。
#   7-59/10 * * * * bash /root/sls2-deploy/sls2/tools/remote/stoploss.sh
# 布置一次改动的止损（重启训练之后立刻运行；BASE 为改动前那一刻的备份，即 restart.sh 产生的“重启前”备份，不带扩展名）：
#   bash stoploss.sh arm backups/v19600-20261026-0330-重启前 '{"floor": 29.94, "r2": 0.844, "r3": 0.201}' 说明文字
# 判定（评测按时间取改动之后的，先跳过 warmup 次）：
#   观察期：攒够 window 次评测后，平均层数低于基线 0.6 层、打过第一幕 Boss 低于基线 3 个百分点、到第三幕低于基线 3.5 个百分点，任一条成立就回退；都不成立则通过；
#   任何时候：最近 6 次评测平均层数低于基线 2 层，立即回退（练崩）；
#   通过后 24 小时内：最近 12 次评测平均层数低于基线 1.2 层，回退。之后不再检查。
set -euo pipefail
R=${RUN_DIR:-$HOME/.local/share/sls2/runs/defect}
S=$R/stoploss
mkdir -p "$S"
exec 9>"$S/.lock"; flock -n 9 || exit 0
if [ "${1:-}" = arm ]; then
  cp "$R/$2.pt" "$S/base.pt"; cp "$R/$2.train_config.json" "$S/base.train_config.json"
  python3 - "$S/state.json" "$3" "${4:-}" <<'EOF'
import json, sys, time
json.dump({"phase": "watch", "since": time.time(), "baseline": json.loads(sys.argv[2]), "note": sys.argv[3],
           "warmup": 4, "window": 24}, open(sys.argv[1], "w"), ensure_ascii=False, indent=1)
EOF
  echo "$(date '+%F %T') 布置止损：$4，基线 $3，退路 $2" >> "$S/log.txt"; exit 0
fi
[ -f "$S/state.json" ] || exit 0
ACT=$(python3 - "$S/state.json" "$R/eval.jsonl" <<'EOF'
import json, sys, time
st = json.load(open(sys.argv[1])); b = st["baseline"]
if st["phase"] not in ("watch", "passed"):
    print("none"); sys.exit()
ev = [json.loads(l) for l in open(sys.argv[2]) if l.strip()]
post = [e for e in ev if e["time"] > st["since"]][st["warmup"]:]
mean = lambda xs, k: sum(e[k] for e in xs) / len(xs)
msg = f"改动后评测 {len(post)} 次" + (f"，平均层数 {mean(post, 'floor_mean'):.2f}" if post else "")
act = "none"
if len(post) >= 6 and mean(post[-6:], "floor_mean") < b["floor"] - 2.0:
    act, msg = "revert", msg + f"，最近 6 次平均层数 {mean(post[-6:], 'floor_mean'):.2f}，比基线低 2 层以上，判定练崩"
elif st["phase"] == "watch" and len(post) >= st["window"]:
    f, r2, r3 = mean(post, "floor_mean"), mean(post, "reach_act2"), mean(post, "reach_act3")
    msg += f"，平均层数 {f:.2f}（基线 {b['floor']}）、打过第一幕 Boss {r2:.3f}（{b['r2']}）、到第三幕 {r3:.3f}（{b['r3']}）"
    if f < b["floor"] - 0.6 or r2 < b["r2"] - 0.03 or r3 < b["r3"] - 0.035:
        act, msg = "revert", msg + "，低于止损线"
    else:
        st["phase"], st["passed"] = "passed", time.time(); msg += "，通过"
elif st["phase"] == "passed":
    if time.time() - st["passed"] > 86400:
        st["phase"] = "done"; msg += "，通过后已满 24 小时，停止检查"
    elif len(post) >= 12 and mean(post[-12:], "floor_mean") < b["floor"] - 1.2:
        act, msg = "revert", msg + f"，通过后最近 12 次平均层数 {mean(post[-12:], 'floor_mean'):.2f}，比基线低 1.2 层以上"
if act == "revert":
    st["phase"], st["reverted"] = "reverted", time.time()
json.dump(st, open(sys.argv[1], "w"), ensure_ascii=False, indent=1)
print(act, msg)
EOF
)
[ "$ACT" = none ] && exit 0
echo "$(date '+%F %T') $ACT" >> "$S/log.txt"
[ "${ACT%% *}" = revert ] || exit 0
[ -n "${DRY:-}" ] && exit 0   # DRY=1：只判定、记日志，不动训练
touch "$R/STOP"; P=$(cat "$R/learner.pid"); kill -TERM "$P" 2>/dev/null || true
for i in $(seq 90); do kill -0 "$P" 2>/dev/null || break; sleep 2; done
kill -0 "$P" 2>/dev/null && { echo "$(date '+%F %T') 学习进程停不下来，未回退（STOP 仍在）" >> "$S/log.txt"; exit 1; }
cp "$R/latest.pt" "$S/reverted-from.pt"; cp "$R/train_config.json" "$S/reverted-from.train_config.json"
cp "$S/base.pt" "$R/latest.pt"; cp "$S/base.train_config.json" "$R/train_config.json"
rm -f "$R/STOP"; bash "$(dirname "$0")/start_train.sh" >> "$S/log.txt" 2>&1
echo "$(date '+%F %T') 已回退到改动前的检查点和配置并重启；改动后的检查点留在 stoploss/reverted-from.pt" >> "$S/log.txt"
