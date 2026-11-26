#!/usr/bin/env bash
# 决策记录 287：专项分支的定期验收。每 EVERY 版存一份快照（snaps/v<版>.pt），用留出验收卷第 0 组随机数打一遍，与学生 e 同组配对，追加到 exam_log.txt。
#   RUN=/root/b3s/warm EVERY=150 ENVS=40 bash exam_loop.sh
set -u
RUN=${RUN:?}; EVERY=${EVERY:-150}; ENVS=${ENVS:-40}
cd /root/b3s/repo && mkdir -p $RUN/snaps $RUN/exam
last=-1
while [ ! -f $RUN/STOP ]; do
  v=$(tail -1 $RUN/log.jsonl 2>/dev/null | python3 -c "import json,sys;print(json.loads(sys.stdin.read())['version'])" 2>/dev/null || echo -1)
  if [ "$v" -ge 0 ] && { [ $last -lt 0 ] || [ $((v - last)) -ge $EVERY ]; }; then
    cp $RUN/latest.pt $RUN/snaps/tmp.pt && sv=$(.venv/bin/python -I -c "import torch;print(torch.load('$RUN/snaps/tmp.pt',map_location='cpu',weights_only=False)['version'])") \
      && mv $RUN/snaps/tmp.pt $RUN/snaps/v$sv.pt || { sleep 60; continue; }
    nice -n 5 .venv/bin/python -I tools/boss3_exam.py run --ckpt $RUN/snaps/v$sv.pt --exam /root/b3s/exam.json --out $RUN/exam/v${sv}_r0.jsonl --envs $ENVS --reseed 0 > /dev/null 2>&1
    echo "$(date '+%F %T') 第 $sv 版：$(.venv/bin/python -I tools/boss3_exam.py compare /root/b3s/exam_res/e_r0.jsonl $RUN/exam/v${sv}_r0.jsonl | head -1)" >> $RUN/exam_log.txt
    last=$sv
  fi
  sleep 120
done
