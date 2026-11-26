"""训练健康诊断（后续计划第九节、决策记录 219）：用一个检查点贪心打若干整局，每隔几步存一个真实观测（sts2env.Obs），连同局面信息与这一局的结局。
给 tools/health/activations.py 在本机对各检查点做前向（休眠单元、有效秩、价值头与结局的关系）。

  .venv/bin/python -I tools/health/capture_obs.py --ckpt 检查点.pt --out 观测.pkl --runs 40 --envs 8 --every 3 --seed-prefix HC
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--runs", type=int, default=40); ap.add_argument("--envs", type=int, default=8); ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--seed-prefix", default="HC"); ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    import numpy as np
    import torch
    from sts2env import EngineError, VecEnv
    from sts2rl.actor import MODE_RUN
    from sts2rl.features import Ragged, collate_ragged
    from sts2rl.model import ModelConfig, PolicyValueNet, load_state
    torch.set_num_threads(1)
    blob = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    net = PolicyValueNet(ModelConfig(**blob["config"]["model"])).eval().to(a.device)
    load_state(net, blob["state"])
    queue = [f"{a.seed_prefix}{j:04d}" for j in range(a.runs)][::-1]
    base = {"mode": "run", "character": "DEFECT", "ascension": 10, "route_summary": True}
    recs, result, steps, seed_of = [], {}, {}, {}
    t0 = time.time()
    with VecEnv(a.envs) as env:
        def launch(i):
            if queue:
                s = queue.pop(); seed_of[i] = s; steps[i] = 0
                env.send(i, dict(base, op="reset", seed=s))
        for i in range(len(env)):
            launch(i)
        obs = {}
        while env.pending or obs:
            for i, o in env.recv_ready(timeout=0.5):
                if isinstance(o, EngineError) or o.done or o.stuck or o.num_actions == 0 or steps[i] > 6000:
                    ok = not isinstance(o, EngineError)
                    result[seed_of[i]] = {"win": bool(ok and o.victory and not o.stuck), "floor": int(o.floor) if ok else -1, "steps": steps[i]}
                    print(f"  {seed_of[i]}：{'通关' if result[seed_of[i]]['win'] else '止于'} 第 {result[seed_of[i]]['floor']} 层（{time.time() - t0:.0f} 秒）", flush=True)
                    launch(i)
                    continue
                obs[i] = o
            if not obs:
                continue
            ids = list(obs)
            with torch.no_grad():
                logits, _ = net(collate_ragged(Ragged.from_obs([obs[i] for i in ids], np.full(len(ids), MODE_RUN))).to(a.device))
            for j, i in enumerate(ids):
                o = obs.pop(i)
                if steps[i] % a.every == 0:
                    recs.append({"obs": o, "seed": seed_of[i], "step": steps[i], "act": int(o.act), "floor": int(o.floor), "in_combat": bool(o.in_combat),
                                 "decision": int(o.header[1])})
                steps[i] += 1
                env.send(i, {"op": "step", "action": int(logits[j, : o.num_actions].argmax())})
    for r in recs:
        r.update(result.get(r["seed"], {}))
    pickle.dump({"version": blob.get("version"), "results": result, "recs": recs}, open(a.out, "wb"))
    print(f"{len(result)} 局、{len(recs)} 个观测 → {a.out}（{time.time() - t0:.0f} 秒）")


if __name__ == "__main__":
    main()
