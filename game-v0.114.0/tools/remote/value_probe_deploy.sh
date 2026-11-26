#!/usr/bin/env bash
# 只发布价值诊断涉及的文件。独立目录先测；核对旧文件哈希；正常停训、备份、续训。
# 不用 restart.sh 的全目录同步，避免发布工作区中其他任务的改动。
set -euo pipefail
. "$(dirname "$0")/hosts.sh"
H=${1:-b}; host_require "$H"
VP_VALIDATE_ONLY=${2:-}
S=$(host_ssh "$H"); E=$(host_rsync_e "$H"); A=$(host_addr "$H"); D=$(host_deploy "$H"); R=$(host_run "$H")
VP_TAG=$(date -u +%Y%m%dT%H%M%SZ)
VP_STAGE=$D/.cache/value-probe-$VP_TAG
VP_TMP=$(mktemp -d /tmp/sls2-value-probe-deploy.XXXXXX)
trap 'rm -rf "$VP_TMP"' EXIT
cp "$REPO/tools/remote/value_probe_baseline.json" "$VP_TMP/expected.json"
$S "set -e; mkdir -p '$VP_STAGE/python' '$VP_STAGE/tools'; cp -a '$D/python/.' '$VP_STAGE/python/'; ln -s '$D/engine' '$VP_STAGE/engine'; ln -s '$D/.venv' '$VP_STAGE/.venv'"
rsync -a -e "$E" "$REPO/python/sts2rl/actor.py" "$REPO/python/sts2rl/train.py" "$REPO/python/sts2rl/config.py" "$REPO/python/sts2rl/value_probe.py" "$A:$VP_STAGE/python/sts2rl/"
rsync -a -e "$E" "$REPO/tools/test_value_probe.py" "$REPO/tools/value_probe_summary.py" "$REPO/tools/value_probe_analysis.py" "$REPO/tools/value_probe_cards.py" "$VP_TMP/expected.json" "$A:$VP_STAGE/tools/"
$S "cd '$VP_STAGE'; OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VALUE_PROBE_TEST_DEVICE=cuda .venv/bin/python -I tools/test_value_probe.py"
if [ "$VP_VALIDATE_ONLY" = --validate-only ]; then
  echo "VALIDATION_ONLY_OK stage=$VP_STAGE（未停止训练、未替换线上文件或配置）"
  exit 0
fi
cat > "$VP_TMP/apply.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
VP_STAGE=$1; D=$2; R=$HOME/$3; H=$4
cd "$R"
python3 - "$VP_STAGE" "$D" <<'PY'
import hashlib,json,sys
from pathlib import Path
stage,deploy=map(Path,sys.argv[1:]);expected=json.loads((stage/'tools/expected.json').read_text())
for name,sha in expected.items():
 actual=hashlib.sha256((deploy/'python/sts2rl'/name).read_bytes()).hexdigest()
 if actual!=sha:raise SystemExit('线上文件已变，停止发布：'+name)
PY
[ ! -e STOP ] || { echo '已有 STOP，保持原状态，不发布'; exit 1; }
P=$(cat learner.pid)
kill -0 "$P"
grep -qa train_defect "/proc/$P/cmdline"
B=$R/value_probe_deploy/$(basename "$VP_STAGE");mkdir -p "$B/code"
for f in actor.py train.py config.py; do cp "$D/python/sts2rl/$f" "$B/code/$f"; done
cp train_config.json "$B/train_config.json"
restore() {
  echo '发布失败，恢复原代码与配置'
  for f in actor.py train.py config.py; do cp "$B/code/$f" "$D/python/sts2rl/$f"; done
  cp "$B/train_config.json" train_config.json
  rm -f STOP
  RUN_DIR="$R" bash "$D/tools/remote/start_train.sh"
}
trap restore ERR
touch STOP;kill -TERM "$P"
for _ in $(seq 90); do kill -0 "$P" 2>/dev/null || break; sleep 2; done
if kill -0 "$P" 2>/dev/null; then echo '旧学习进程尚未退出，恢复守护与原文件'; restore; trap - ERR; exit 1; fi
cp latest.pt "$B/before.pt"
for f in actor.py train.py config.py value_probe.py; do cp "$VP_STAGE/python/sts2rl/$f" "$D/python/sts2rl/$f"; done
python3 - <<'PY'
import json
from pathlib import Path
p=Path('train_config.json');old=json.loads(p.read_text());new=dict(old)
new.update(value_probe=True,value_probe_every=20,value_probe_samples=128,value_probe_grad_every=200,value_probe_episode_frac=0.02,value_probe_max_gb=8.0)
assert {k:v for k,v in old.items() if not k.startswith('value_probe')}=={k:v for k,v in new.items() if not k.startswith('value_probe')}
p.write_text(json.dumps(new,ensure_ascii=False,indent=1)+'\n')
PY
rm STOP;RUN_DIR="$R" bash "$D/tools/remote/start_train.sh"
trap - ERR
echo "VALUE_PROBE_DEPLOYED backup=$B learner=$(cat learner.pid)"
SH
rsync -a -e "$E" "$VP_TMP/apply.sh" "$A:$VP_STAGE/apply.sh"
$S "setsid nohup bash '$VP_STAGE/apply.sh' '$VP_STAGE' '$D' '$R' '$H' > '$VP_STAGE/deploy.log' 2>&1 < /dev/null &"
echo "STAGE=$VP_STAGE"
echo "读取发布结果：部署目录内 deploy.log"
