#!/usr/bin/env bash
# 搜索测试延长的一小时（决策记录 217，用户：“允许你的测试窗口增加1个小时”）：第一段截止后，把还没做的题按两台的速度分开继续搜。
# A 的瓶颈在每个分片的 Python 进程（单核约 90%），所以 A 开 24 个分片 × 2 个引擎，B 仍是 12 × 4；剩下的题按随机顺序交替分，B 拿三分之二。
#   bash tools/remote/search_window2.sh <结果目录名，如 20261031-1426> <截止时刻 Unix 秒>
set -eu
. "$(dirname "$0")/hosts.sh"
STAMP=$1; DL=$2
T=/root/sls2-search; O=$T/out/$STAMP; PY=/root/sls2-deploy/sls2/.venv/bin/python
L=$HOME/.local/share/sls2/search_test/$STAMP; mkdir -p "$L/p1a" "$L/p1b"
ENVSH='export DOTNET_ROOT=$HOME/.dotnet PATH=$HOME/.dotnet:$PATH DOTNET_CLI_TELEMETRY_OPTOUT=1'
rsync -a -e "$(host_rsync_e a)" --include 'search_s*.jsonl' --exclude '*' "$(host_addr a):$O/" "$L/p1a/"
rsync -a -e "$(host_rsync_e b)" --include 'search_s*.jsonl' --exclude '*' "$(host_addr b):$O/" "$L/p1b/"
# 第一段还没收尾的分片：它们手上的题（各自题目序列里没做完的前 2 道）不再分出去，免得重复搜
RUN=$( { $(host_ssh a) "pgrep -af '[t]ools/search_test.py search' || true"; $(host_ssh b) "pgrep -af '[t]ools/search_test.py search' || true"; } | grep -o 'shard [0-9]*/24' | cut -d' ' -f2 | cut -d/ -f1 | tr '\n' ' ')
echo "第一段仍在收尾的分片：${RUN:-无}"
python3 - "$L" "$RUN" <<'EOF'
import json, sys, glob, os
L, running = sys.argv[1], {int(x) for x in sys.argv[2].split()}
exam = json.load(open(os.path.expanduser("~/.local/share/sls2/search_test/exam_g1.json"), encoding="utf-8"))
done = {json.loads(l)["id"] for f in glob.glob(f"{L}/p1*/search_s*.jsonl") for l in open(f, encoding="utf-8") if l.strip()}
busy = set()
for k in running:
    seq = [it["id"] for j, it in enumerate(exam["items"]) if j % 24 == k and it["id"] not in done]
    busy |= set(seq[:2])
json.dump(sorted(busy), open(f"{L}/p1_inflight.json", "w"))
rest = [it for it in exam["items"] if it["id"] not in done and it["id"] not in busy]
for name, part in (("a", rest[0::3]), ("b", [it for j, it in enumerate(rest) if j % 3])):
    json.dump({"built_from": exam["built_from"], "items": part}, open(f"{L}/rest_{name}.json", "w", encoding="utf-8"), ensure_ascii=False)
print(f"第一段做完 {len(done)} 题、在做 {len(busy)} 题；剩 {len(rest)} 题：A {len(rest[0::3])}、B {len(rest) - len(rest[0::3])}")
EOF
rsync -a -e "$(host_rsync_e a)" "$L/rest_a.json" "$(host_addr a):$T/"
rsync -a -e "$(host_rsync_e b)" "$L/rest_b.json" "$(host_addr b):$T/"
start() {   # 主机 分片数 每片引擎数
  $(host_ssh "$1") "cd $T/sls2 && $ENVSH; for k in \$(seq 0 $(($2 - 1))); do setsid nohup $PY -I tools/search_test.py search --ckpt $T/model.pt --exam $T/rest_$1.json \
    --out $O/search2_s\$k.jsonl --labels $O/labels2_s\$k.pkl --shard \$k/$2 --envs $3 --live 2 --device cuda --deadline $DL > $O/search2_s\$k.log 2>&1 < /dev/null & done; echo $1 第二段 $2 个分片已起"
}
start b 12 4
start a 24 2
echo "截止 $(date -d @$DL '+%T')（不再开新题）"
