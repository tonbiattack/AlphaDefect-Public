#!/usr/bin/env bash
# 本机运行：启动储君训练线（2026-11-06）。先 deploy.sh 同步并编译。
#   B（4090）：学习进程 + 本机采集（RUN_DIR=/root/.local/share/sls2/runs/regent，配置 tools/remote/regent/train_config.json），
#              守护与定时备份各加一条 crontab（RUN_DIR 指向储君的运行目录，脚本取自储君的部署目录）；
#   A（3090）：远程采集（桥接端口 47322，权重文件 /dev/shm/sts2rl_regent.pt）+ 常规评测（种子池 P1 的评测部分 384 局，运行目录 regent-eval）；
#   本机：看板目标改成储君（~/.local/share/sls2/runs/remote/lite_target.json），重启精简拉取让新目标生效。
# 已在跑的部分不重复启动。故障机器人那边的进程、crontab、部署目录都不动。
#   bash tools/remote/regent/start.sh            # 全部
#   PARTS="b" bash tools/remote/regent/start.sh  # 只起 B（a、board 同理，可以组合："b a board"）
# A 到期后（2026-11-09，储君决策记录第 26 条；hosts.json 里 a 标为 pending）：常规评测改在 B 上跑（beval：CPU 推理、4 个进程 × 8 个引擎、nice 10，
# 权重经 /dev/shm/sts2rl_regent_eval.pt 链接到学习进程发布的文件，crontab 每分钟更新链接），默认 PARTS 变为 "b beval board"。
# 暂时单机（2026-11-06，B 被别的任务占着时）：LAYOUT=a 在 A 上起学习 + 本机采集 + 本进程评测（train_config.json 再覆盖 hw_a_main.json），
# 看板的训练端与评测端都指向 A；B 空出来后用 migrate_to_b.sh 回到 B 学习 + A 采集与评测。
#   LAYOUT=a bash tools/remote/regent/start.sh
set -euo pipefail
. "$(dirname "$0")/../hosts.sh"
D=/root/regent-deploy/sls2
RUN=/root/.local/share/sls2/runs/regent
EVAL=/root/.local/share/sls2/runs/regent-eval
CFG=$(dirname "$0")/train_config.json
PORT=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["bridge_port"])' "$CFG")
BIND=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["bridge_bind"])' "$CFG")
LAYOUT=${LAYOUT:-b}
A_GONE=""; host_pending a && A_GONE=1
if [ "$LAYOUT" = a ]; then PARTS=${PARTS:-amain board}; elif [ -n "$A_GONE" ]; then PARTS=${PARTS:-b beval board}; else PARTS=${PARTS:-b a board}; fi
A_ACTORS=${A_ACTORS:-8}; A_ENVS=${A_ENVS:-20}
# A 的评测规模：原结构时 A 上还有 8×20 的远程采集，A 只有 62 GB 内存（单机时 250 个引擎就只剩约 4 GB 可用），评测默认 8 个进程 × 8 个引擎
EVAL_WORKERS=${EVAL_WORKERS:-8}; EVAL_ENVS=${EVAL_ENVS:-8}

if [[ " $PARTS " == *" b "* ]]; then
  echo "== B：学习端"
  SSH=$(host_ssh b)
  $SSH "mkdir -p $RUN"
  rsync -a -e "$(host_rsync_e b)" "$CFG" "$(host_addr b):$RUN/train_config.json"
  $SSH "set -e; cd $RUN
    [ -f STOP ] && { echo '有 STOP 标记（rm $RUN/STOP 后再启动）'; exit 1; }
    ( crontab -l 2>/dev/null | grep -v 'regent-deploy/sls2/tools/remote/' || true ;
      echo '*/2 * * * * RUN_DIR=$RUN bash $D/tools/remote/train_watchdog.sh' ;
      echo '*/30 * * * * RUN_DIR=$RUN bash $D/tools/remote/ckpt_backup.sh' ;
      echo '15,45 * * * * bash $D/tools/remote/regent/keep_backups_b.sh' ;
      [ -n '$A_GONE' ] && echo '*/10 * * * * bash $D/tools/remote/regent/keep_snapshots.sh' ) | crontab -
    echo 'crontab：'; crontab -l
    RUN_DIR=$RUN bash $D/tools/remote/start_train.sh"
fi

if [[ " $PARTS " == *" a "* ]]; then
  echo "== A：远程采集与评测"
  SSH=$(host_ssh a)
  $SSH "mkdir -p $RUN $EVAL"
  rsync -a -e "$(host_rsync_e a)" "$CFG" "$(host_addr a):/root/regent_train_config.json"
  $SSH "set -e; cd $D; export DOTNET_ROOT=\$HOME/.dotnet PATH=\$HOME/.dotnet:\$PATH DOTNET_CLI_TELEMETRY_OPTOUT=1
    # 只看名字是 python 的进程：这条远程命令所在的 shell 的命令行里也有下面的启动命令，pgrep -f 会命中它自己
    if pgrep -a python | grep -q 'remote_actors.py --config /root/regent_train_config.json'; then echo 远程采集已在跑
    else
      setsid nohup .venv/bin/python -I $D/tools/remote/remote_actors.py --config /root/regent_train_config.json --run-dir $RUN \
        --host $BIND --port $PORT --token-file /root/.sls2_bridge_token --actors $A_ACTORS --envs $A_ENVS --weights /dev/shm/sts2rl_regent.pt \
        >> $RUN/collector.out 2>&1 < /dev/null &
      echo \"远程采集已起：$A_ACTORS 个进程 × $A_ENVS 个引擎\"
    fi
    if pgrep -a python | grep -q 'remote_eval.py --config /root/regent_train_config.json'; then echo 评测已在跑
    else
      setsid nohup .venv/bin/python -I $D/tools/remote/remote_eval.py --config /root/regent_train_config.json --run-dir $EVAL \
        --weights /dev/shm/sts2rl_regent.pt --seeds-file $D/tools/remote/seed_pool.json --workers $EVAL_WORKERS --envs $EVAL_ENVS --device cuda --cores 20 --nice 5 \
        --keep-runs 20 --no-replay-all >> $EVAL/remote_eval.out 2>&1 < /dev/null &
      echo 评测已起：种子池 P1 的评测部分（384 局）
    fi"
fi

if [[ " $PARTS " == *" beval "* ]]; then
  echo "== B：常规评测（A 到期后）"
  SSH=$(host_ssh b)
  rsync -a -e "$(host_rsync_e b)" "$CFG" "$(host_addr b):/root/regent_train_config.json"
  $SSH "set -e; mkdir -p $EVAL; cd $D
    P=\$(cat $RUN/learner.pid 2>/dev/null || true); [ -n \"\$P\" ] && [ -f /dev/shm/sts2rl_\$P.pt ] && ln -sfn /dev/shm/sts2rl_\$P.pt /dev/shm/sts2rl_regent_eval.pt
    crontab -l 2>/dev/null | grep -q 'sts2rl_regent_eval.pt' || ( crontab -l 2>/dev/null ;
      echo '* * * * * P=\$(cat $RUN/learner.pid 2>/dev/null) && [ -f /dev/shm/sts2rl_\$P.pt ] && ln -sfn /dev/shm/sts2rl_\$P.pt /dev/shm/sts2rl_regent_eval.pt' ) | crontab -
    if pgrep -a python | grep -q 'remote_eval.py --config /root/regent_train_config.json'; then echo 评测已在跑
    else
      setsid nohup .venv/bin/python -I $D/tools/remote/remote_eval.py --config /root/regent_train_config.json --run-dir $EVAL \
        --weights /dev/shm/sts2rl_regent_eval.pt --seeds-file $D/tools/remote/seed_pool.json --workers 4 --envs 8 --device cpu --cores 8 --nice 10 \
        --keep-runs 100000 --no-replay-all >> $EVAL/remote_eval.out 2>&1 < /dev/null &
      echo 评测已起：种子池 P1 的评测部分（384 局），CPU 推理
    fi"
fi

if [[ " $PARTS " == *" amain "* ]]; then
  echo "== A：单机训练（学习 + 本机采集 + 本进程评测）"
  SSH=$(host_ssh a)
  $SSH "mkdir -p $RUN"
  TMPC=$(mktemp); python3 - "$CFG" "$(dirname "$0")/hw_a_main.json" "$TMPC" <<'PY'
import json, sys
c = json.load(open(sys.argv[1], encoding="utf-8")); hw = json.load(open(sys.argv[2], encoding="utf-8"))
c.update({k: v for k, v in hw.items() if not k.startswith("_")})
json.dump(c, open(sys.argv[3], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
PY
  rsync -a -e "$(host_rsync_e a)" "$TMPC" "$(host_addr a):$RUN/train_config.json"; rm -f "$TMPC"
  # MPS=0：A 上 MPS 守护已在跑时 start_train.sh 本来也不会再起；写进 crontab 免得哪天守护停了被重新拉起、影响同机别的显卡任务
  $SSH "set -e; cd $RUN
    [ -f STOP ] && { echo '有 STOP 标记（rm $RUN/STOP 后再启动）'; exit 1; }
    ( crontab -l 2>/dev/null | grep -v 'regent-deploy/sls2/tools/remote/' || true ;
      echo '*/2 * * * * MPS=0 RUN_DIR=$RUN bash $D/tools/remote/train_watchdog.sh' ;
      echo '*/30 * * * * RUN_DIR=$RUN bash $D/tools/remote/ckpt_backup.sh' ) | crontab -
    echo 'crontab：'; crontab -l
    MPS=0 RUN_DIR=$RUN bash $D/tools/remote/start_train.sh"
fi

if [[ " $PARTS " == *" board "* ]]; then
  echo "== 本机：看板目标改为储君"
  L=$REMOTE_LOCAL
  python3 - "$L" "$LAYOUT" "${A_GONE:-0}" <<'PY'
import json, os, sys, time
L, layout, a_gone = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
tgt = {"name": "regent", "label": "储君", "line": "从零训练", "deploy": "/root/regent-deploy/sls2", "run": "regent",
       # 单机（A）时评测在学习进程里跑、写在同一个运行目录；原结构时评测在 A 的 regent-eval
       "eval": "regent" if layout == "a" else "regent-eval", "train_host": "a" if layout == "a" else "b", "eval_host": "b" if a_gone else "a",
       "markers": "markers-regent.json", "card_stats": "codex_regent_solo_a10_v0.111.0.json", "eval_desc": "评测（贪心出手，P1 的 384 局）",
       "note": "储君从零训练（同故障机器人主线的网络结构，观测格式 20），评测为种子池 P1 的评测部分 384 局、贪心、原始权重。"
               + ("暂时在 A 上单机训练。" if layout == "a" else ""),
       "since": time.time()}
json.dump(tgt, open(os.path.join(L, "lite_target.json.tmp"), "w"), ensure_ascii=False, indent=1)
os.replace(os.path.join(L, "lite_target.json.tmp"), os.path.join(L, "lite_target.json"))
mk = os.path.join(L, "markers-regent.json")
if not os.path.exists(mk):
    json.dump({"marks": [], "cols": [{"label": "最近 12 次", "from": 0, "last": 12}]}, open(mk, "w"), ensure_ascii=False, indent=1)
PY
  # 精简拉取常驻进程按启动时读到的脚本跑，重启它（本机守护每分钟拉起）才会按新目标去取统计；看板生成每轮现读目标文件
  pkill -f "sls2/tools/remote/[p]ull_lite.sh" || true
  echo "看板目标已写入 $L/lite_target.json；精简拉取一分钟内由本机守护重新拉起"
fi
