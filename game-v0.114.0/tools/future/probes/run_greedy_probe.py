"""整局贪心复现与第三幕 Boss 交接处的帧（探针）。"""
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
net = PolicyValueNet(ModelConfig(**b["config"]["model"])).eval().to("cuda"); load_state(net, b["state"])
ev = X.make_evaluator(None, net, "cuda", policy_mode=1, value_mode=0, strip_priv=False)
ref = {}
for f in glob.glob("/root/i2a/games_p2b/runs_s*.jsonl"):
    for l in open(f):
        r = json.loads(l); ref[r["seed"]] = r
seeds = json.load(open("/root/i2a/data/seeds_p2.json"))["seeds"][: int(sys.argv[1])]
P = _Process(dll, open("/tmp/run_probe_engine.log", "ab"))
for s in seeds:
    P.send({"op": "reset", "mode": "run", "character": "FOOL", "ascension": 10, "seed": s, "labels": False, "snapshots": True,
            "route_summary": True, "loop_cut": True, "stall_turns": 20})
    o = parse(P.recv()); n = 0; t0 = time.time(); log = []
    prev_ic = False
    while not (o.done or o.stuck or o.num_actions == 0):
        lg, _ = ev.forward([o], [1]); a = int(lg[0, : o.num_actions].argmax())
        if o.extra and "combat" in o.extra:
            c = o.extra["combat"]; log.append(f"  步{n} 开打 {c.get('room')} act{c.get('act')} 层{c.get('floor')} {c.get('encounter')} 血{c.get('hp')}")
        if o.act == 2 and (o.in_combat != prev_ic or (not o.in_combat and o.floor >= 48)):
            log.append(f"  步{n} act{o.act} 层{o.floor} in_combat {o.in_combat} h23 {int(o.header[23])} h17 {int(o.header[17])} 决策 {o.decision.name} 候选 {o.num_actions} 血 {o.hp}")
        prev_ic = o.in_combat
        P.send({"op": "step", "action": a}); o = parse(P.recv()); n += 1
    r = ref.get(s, {})
    print(f"{s}: 复现 win {bool(o.victory) and not o.stuck} 层 {o.floor}（原 win {r.get('win')} 层 {r.get('floor')}）{n} 步 {time.time() - t0:.0f} 秒；h23 {int(o.header[23])}", flush=True)
    for x in log[-12:]: print(x)
P.close()
