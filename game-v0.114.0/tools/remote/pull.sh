#!/usr/bin/env bash
# 本机常驻：每 FAST 秒（默认 60；2026-11-03 为省中转流量从 10 改为 60，B 的 log.jsonl 每拉一次要上传约 0.26 MB 块校验）从租用机拉回训练运行目录里的小文件，供看板使用（检查点不拉）。
# 主机取自 tools/remote/hosts.json，待填（pending）的跳过；各主机拉到各自的本机目录 ~/.local/share/sls2/runs/remote/<local>
# （A 仍是 .../remote/defect）。
#   bash tools/remote/pull.sh              # 每台已登记的主机各拉一次
#   bash tools/remote/pull.sh a            # 只拉 A 一次
#   LOOP=1 bash tools/remote/pull.sh a     # 常驻拉 A（本机守护 local_watch.sh 为每台主机各起一个）
#   LOOP=1 bash tools/remote/pull.sh       # 常驻，依次拉所有主机（一台慢会拖慢其他台，守护不用这种方式）
# 旧考卷（exam.jsonl）与标定记录（calib.jsonl）决策记录 197 起停写，不再拉；本机的旧副本留作存档。
# 拉取状态：成功时刻写 pull<后缀>.ok，错误追加到 pull<后缀>.err（都在 ~/.local/share/sls2/runs/remote/）。
#   A 的后缀为空，沿用原来的 pull.ok / pull.err（看板改版前读的就是它们）；其他主机为 -<名>，如 B 为 pull-b.ok / pull-b.err。
# 连接参数见 hosts.sh；A 经 ~/.ssh/config 里 rented-host.example 的跳转（伦敦 relay-host）与连接复用，所以 hosts.json 里主机名必须写 rented-host.example。
set -u
. "$(dirname "$0")/hosts.sh"
FAST=${FAST:-60}
if [ $# -gt 0 ]; then HOSTS=("$@"); else mapfile -t HOSTS < <(host_active); fi
for h in "${HOSTS[@]}"; do host_require "$h"; done

# 各主机的连接参数在启动时读一次（hosts.json 改了之后，守护重启本进程才生效）
declare -A RR LL EE AA TT
for h in "${HOSTS[@]}"; do
  RR[$h]=$(host_run "$h"); LL[$h]=$(host_local "$h"); EE[$h]=$(host_rsync_e "$h"); AA[$h]=$(host_addr "$h"); TT[$h]=$(host_pull_tag "$h")
  mkdir -p "${LL[$h]}"
done
pull() {   # pull 主机名
  local h=$1 R=${RR[$1]} A=${AA[$1]}
  timeout 120 rsync -az --ignore-missing-args -e "${EE[$h]}" \
    "$A:$R/log.jsonl" "$A:$R/eval.jsonl" "$A:$R/resource.jsonl" "$A:$R/status.json" \
    "$A:$R/config.json" "$A:$R/watchdog.json" "$A:$R/eval_error.txt" "$A:$R/eval_progress.json" "$A:$R/fork.json" \
    "$A:$R/collector.json" "${LL[$h]}/" \
    2>> "$REMOTE_LOCAL/pull${TT[$h]}.err" \
    && date +%s > "$REMOTE_LOCAL/pull${TT[$h]}.ok"
  # 常规评测改在协同采集机上跑的（tools/remote/remote_eval.py，决策记录 198）：评测目录 defect-eval 拉到本机同名目录，没有就跳过
  mkdir -p "$REMOTE_LOCAL/defect-eval"
  timeout 120 rsync -az --ignore-missing-args -e "${EE[$h]}" "$A:.local/share/sls2/runs/defect-eval/eval.jsonl" "$A:.local/share/sls2/runs/defect-eval/eval_progress.json" \
    "$A:.local/share/sls2/runs/defect-eval/eval_error.txt" "$REMOTE_LOCAL/defect-eval/" 2>/dev/null || true
  # 评测留档的权重（best/ 历来前几名、high/ 评测偏高的，决策记录 208）：每 10 分钟后台同步一次到本机，不删本机已有的，
  # 租用机上被挤掉的权重本机仍留着（2026-10-31 用户：“以后评测偏高的，也要自动备份，别给我再弄丢了”）
  K="$REMOTE_LOCAL/defect-eval/keep${TT[$h]}.ts"
  if [ ! -e "$K" ] || [ $(( $(date +%s) - $(stat -c %Y "$K") )) -ge 600 ]; then
    touch "$K"
    ( timeout 1800 rsync -a --partial --ignore-missing-args -e "${EE[$h]}" "$A:.local/share/sls2/runs/defect-eval/best" "$A:.local/share/sls2/runs/defect-eval/high" \
        "$REMOTE_LOCAL/defect-eval/" 2>/dev/null ) &
  fi
  # Boss 战搜索试验（tools/mcts_eval.py，2026-10-27 起在 A 上）：主目录 ~/mcts 下的结果与日志，没有这个目录的主机不报错
  timeout 60 rsync -az --exclude='exam.json' --include='*.jsonl' --include='*.log' --include='*.json' --exclude='*' -e "${EE[$h]}" \
    "$A:mcts/" "${LL[$h]}/mcts/" 2>/dev/null || true
}
if [ -z "${LOOP:-}" ]; then
  rc=0; for h in "${HOSTS[@]}"; do pull "$h" || rc=1; done; exit $rc
fi
while true; do
  for h in "${HOSTS[@]}"; do pull "$h"; done
  sleep "$FAST"
done
