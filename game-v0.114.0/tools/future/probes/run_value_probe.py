"""整局价值头在 Boss 前后的读数；去特权特征对整局模式首选的影响；victory_reward。"""
import json, sys, glob, time
from pathlib import Path
sys.path.insert(0, "python"); sys.path.insert(0, "tools")
import numpy as np, torch
from sts2env.env import _Process
from sts2env.protocol import parse
from sts2rl.model import ModelConfig, PolicyValueNet, load_state
import sim_pimc as X
dll = Path("engine/Sts2Headless/bin/Release/net9.0/Sts2Headless.dll")
b = torch.load("/root/i2a/data/L1000.pt", map_location="cpu", weights_only=False)
print("victory_reward", b["config"].get("victory_reward"), "run_hp_shaping", b["config"].get("run_hp_shaping"), "prog_floor", b["config"].get("prog_floor"), "prog_boss", b["config"].get("prog_boss"), "bv_shaping", b["config"].get("bv_shaping"))
net = PolicyValueNet(ModelConfig(**b["config"]["model"])).eval().to("cuda"); load_state(net, b["state"])
evn = X.make_evaluator(None, net, "cuda", policy_mode=1, value_mode=1, strip_priv=False)
evs = X.make_evaluator(None, net, "cuda", policy_mode=1, value_mode=1, strip_priv=True)
seeds = json.load(open("/root/i2a/data/seeds_p2.json"))["seeds"][: int(sys.argv[1])]
P = _Process(dll, open("/tmp/run_probe_engine.log", "ab"))
diff = tot = 0
for s in seeds:
    P.send({"op": "reset", "mode": "run", "character": "FOOL", "ascension": 10, "seed": s, "labels": False, "snapshots": True,
            "route_summary": True, "loop_cut": True, "stall_turns": 20})
    o = parse(P.recv()); n = 0; room = None; out = []; prev_ic = False
    while not (o.done or o.stuck or o.num_actions == 0):
        lg, vn = evn.forward([o], [1]); lg2, vs = evs.forward([o], [1])
        a = int(lg[0, : o.num_actions].argmax()); tot += 1; diff += int(a != int(lg2[0, : o.num_actions].argmax()))
        if o.extra and "combat" in o.extra:
            room = o.extra["combat"].get("room")
            if room in ("Boss", "Elite"):
                out.append(f"开打 {room} act{o.act} 层{o.floor} 血{o.hp} V_run 不去/去特权 {float(vn[0]):.3f}/{float(vs[0]):.3f}")
        if prev_ic and not o.in_combat and room in ("Boss", "Elite"):
            out.append(f"  打完 {room} 层{o.floor} 血{o.hp} 决策 {o.decision.name} 候选 {o.num_actions} V_run {float(vs[0]):.3f}")
        prev_ic = o.in_combat
        P.send({"op": "step", "action": a}); o = parse(P.recv()); n += 1
    print(s, "win", bool(o.victory) and not o.stuck, "层", o.floor, "终局帧 V_run", round(float(evs.forward([o], [1])[1][0]), 3) if o.num_actions else "无候选")
    for x in out: print("  ", x)
print(f"去特权特征改变整局首选：{diff} / {tot}")
P.close()
