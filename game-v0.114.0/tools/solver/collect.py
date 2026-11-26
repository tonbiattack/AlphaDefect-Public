"""用训练模型（贪心）打整局，收集 Boss 战开打前的真实快照：python collect.py 代码目录 模型.pt 局数 输出.jsonl
每行 {"label": {...}, "spec": CombatSpec}，label 里记下这一局模型实际打这个 Boss 的结果。"""
import json, sys
import numpy as np
import torch

sys.path.insert(0, sys.argv[1])
from sts2env import VecEnv
from sts2rl.actor import MODE_RUN
from sts2rl.features import Ragged, collate_ragged
from sts2rl.model import ModelConfig, PolicyValueNet, load_state

torch.set_num_threads(4)
net = PolicyValueNet(ModelConfig(d=256, layers=4)).eval()
load_state(net, torch.load(sys.argv[2], map_location="cpu", weights_only=False)["state"])
n_runs, out_path = int(sys.argv[3]), sys.argv[4]
envs = 16
out = []
with VecEnv(envs) as env:
    nxt, live, snap, pending_boss = 0, {}, {}, {}

    def launch(i):
        global nxt
        if nxt >= n_runs:
            return
        seed = f"SOLV{nxt:03d}"; nxt += 1
        live[i] = seed; snap[i] = None; pending_boss[i] = None
        env.send(i, {"op": "reset", "mode": "run", "character": "DEFECT", "ascension": 10, "seed": seed, "snapshots": True})

    for i in range(envs):
        launch(i)
    obs = {}
    while env.pending or obs:
        for i, o in env.recv_ready(timeout=1.0):
            pb = pending_boss.get(i)
            ended = o.done or o.num_actions == 0
            if pb is not None and (ended or not o.in_combat):
                # Boss 战结束：对局还在继续（进奖励界面）或通关 = 打赢；对局结束（死亡后的结束界面没有候选动作）= 打输
                pb["label"]["model_won_in_run"] = bool(not ended or o.victory)
                pb["label"]["end_decision"] = o.decision.name
                pb["label"]["model_hp_after"] = int(o.hp)
                out.append(pb); pending_boss[i] = None
            if o.extra and "snapshot" in o.extra:
                snap[i] = o.extra["snapshot"]
            elif o.extra and "combat" in o.extra:
                c = o.extra["combat"]
                if c["room"] == "Boss" and snap[i] is not None and c["act"] < 2:
                    pending_boss[i] = {"label": {"seed": live[i], "act": c["act"], "floor": c["floor"], "encounter": c["encounter"], "deck": c["deck"],
                                                 "hp": c["hp"], "max_hp": c["max_hp"]},
                                       "spec": {"character": "DEFECT", "ascension": 10, "seed": f"{live[i]}_B{c['act']}", "act": c["act"],
                                                "encounter": c["encounter"], "player_snapshot": snap[i]}}
            if o.done or o.num_actions == 0:
                launch(i)
            else:
                obs[i] = o
        if not obs:
            continue
        ids = list(obs)
        with torch.no_grad():
            logits, _ = net(collate_ragged(Ragged.from_obs([obs[i] for i in ids], np.full(len(ids), MODE_RUN))).to("cpu"))
        for i, a in zip(ids, logits.argmax(-1).tolist()):
            env.send_step(i, a)
        obs.clear()
with open(out_path, "w") as f:
    for x in out:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")
print(f"{n_runs} 局，Boss 快照 {len(out)} 份：", {a: (sum(1 for x in out if x['label']['act'] == a), sum(x['label']['model_won_in_run'] for x in out if x['label']['act'] == a)) for a in (0, 1)})
