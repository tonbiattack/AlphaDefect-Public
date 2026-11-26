"""战斗模式连打题与整局模式的 Boss 帧头部对比。"""
import json, sys, glob
from pathlib import Path
sys.path.insert(0, "python"); sys.path.insert(0, "tools")
import numpy as np
from sts2env.env import _Process
from sts2env.protocol import parse
# 1) 战斗模式连打题（vpub 数据，题 id 1、2、3）的第一场与第二场的第一帧
res = [json.loads(l) for l in open("/root/i2a/vpub/data/res_s0.jsonl")][:3]
fb = open("/root/i2a/vpub/data/frames_s0.bin", "rb")
for r in res:
    tr = r["traj"]; seen = set()
    for x in tr:
        st = x[6]
        if st in seen: continue
        seen.add(st); fb.seek(x[0]); o = parse(fb.read(x[1]))
        print("战斗模式 题", r["id"], "第", st + 1, "场", "header", list(map(int, o.header[:30])))
# 2) 整局模式（P2 第一个种子）各 Boss 的第一帧
import torch
from sts2rl.model import ModelConfig, PolicyValueNet, load_state
import sim_pimc as X
b = torch.load("/root/i2a/data/L1000.pt", map_location="cpu", weights_only=False)
net = PolicyValueNet(ModelConfig(**b["config"]["model"])).eval().to("cuda"); load_state(net, b["state"])
ev = X.make_evaluator(None, net, "cuda", policy_mode=1, value_mode=1, strip_priv=True)
P = _Process(Path("engine/Sts2Headless/bin/Release/net9.0/Sts2Headless.dll"), open("/tmp/hdr_probe.log", "ab"))
seed = json.load(open("/root/i2a/data/seeds_p2.json"))["seeds"][0]
P.send({"op": "reset", "mode": "run", "character": "FOOL", "ascension": 10, "seed": seed, "labels": False, "snapshots": True, "route_summary": True, "loop_cut": True, "stall_turns": 20})
o = parse(P.recv())
while not (o.done or o.stuck or o.num_actions == 0):
    if o.extra and "combat" in o.extra and o.extra["combat"].get("room") == "Boss":
        print("整局模式 act", o.act, "层", o.floor, "header", list(map(int, o.header[:30])))
    lg, _ = ev.forward([o], [1]); P.send({"op": "step", "action": int(lg[0, :o.num_actions].argmax())}); o = parse(P.recv())
P.close()
