#!/usr/bin/env bash
# 猎人集成蒸馏用的局面：每隔 GAP 秒让学习端存一批（DUMP 标记，train.py），等那一版学完再挪走，共 N 批
# setsid nohup bash /root/distill_silent/take_dumps.sh > /root/distill_silent/take.out 2>&1 < /dev/null &
set -u
R=/root/.local/share/sls2/runs/silent; O=/root/distill_silent
N=${N:-8}; GAP=${GAP:-270}
lastv() { tail -n1 $R/log.jsonl | python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["version"])'; }
for i in $(seq 1 $N); do
  [ -f $R/STOP ] && { echo "$(date +%T) 有 STOP，停"; break; }
  v0=$(lastv); touch $R/DUMP
  for t in $(seq 1 120); do [ -f $R/DUMP ] || break; sleep 2; done
  [ -f $R/DUMP ] && { echo "$(date +%T) 4 分钟没取走 DUMP，停"; rm -f $R/DUMP; break; }
  for t in $(seq 1 60); do v=$(lastv); [ "$v" -gt "$v0" ] && break; sleep 2; done
  sleep 3
  f=$O/d$(printf %02d $i)_v$((v0 + 1)).pkl; mv $R/batch_dump.pkl $f
  echo "$(date +%T) 第 $i 批：学习第 $((v0 + 1)) 版的批，$(stat -c %s $f) 字节 → $f"
  [ $i -lt $N ] && sleep $GAP
done
echo "$(date +%T) 完成"
