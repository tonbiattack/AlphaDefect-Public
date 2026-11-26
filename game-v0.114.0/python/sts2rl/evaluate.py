"""固定评测集：按固定种子生成一组战斗，用策略（贪心或采样）或随机策略打完，统计胜率与掉血。

  python -I evaluate_defect.py --checkpoint runs/x/latest.pt --n 400 --progress 0.0
  python -I evaluate_defect.py --random --n 400
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np
import torch

from sts2env import VecEnv

from .catalog import Catalog
from .config import TrainConfig
from .curriculum import CombatCurriculum
from .features import Ragged, collate_ragged
from .model import ModelConfig, PolicyValueNet, load_state


def evaluate(net: PolicyValueNet | None, n: int, progress: float, envs: int = 8, greedy: bool = True, seed: int = 12345) -> dict:
    cur = CombatCurriculum(Catalog(), seed=seed)
    specs = [cur.sample(progress) for _ in range(n)]
    rng = np.random.default_rng(seed)
    results = []
    with VecEnv(envs) as env:
        nxt = 0
        slot_spec: dict[int, dict] = {}
        start_hp: dict[int, tuple[int, int]] = {}
        obs: dict[int, object] = {}
        for i in range(min(envs, n)):
            slot_spec[i] = specs[nxt]; env.send(i, {"op": "reset", "mode": "combat", "spec": specs[nxt]}); nxt += 1
        while env.pending or obs:
            for i, o in env.recv_ready(timeout=1.0):
                if i not in start_hp:
                    start_hp[i] = (o.hp, max(1, o.max_hp))
                if o.done or o.stuck or o.num_actions == 0:
                    hp0, mhp = start_hp.pop(i)
                    sp = slot_spec[i]
                    results.append({"win": bool(o.victory), "hp_delta": (o.hp - hp0) / mhp, "act": sp["act"], "enc": sp["encounter"], "stuck": bool(o.stuck)})
                    if nxt < n:
                        slot_spec[i] = specs[nxt]; env.send(i, {"op": "reset", "mode": "combat", "spec": specs[nxt]}); nxt += 1
                else:
                    obs[i] = o
            if not obs:
                continue
            ids = list(obs)
            if net is None:
                acts = [int(rng.integers(obs[i].num_actions)) for i in ids]
            else:
                with torch.no_grad():
                    logits, _ = net(collate_ragged(Ragged.from_obs([obs[i] for i in ids])).to("cpu"))
                acts = (logits.argmax(-1) if greedy else torch.distributions.Categorical(logits=logits).sample()).tolist()
            for i, a in zip(ids, acts):
                env.send_step(i, a)
            obs.clear()
    out = {"n": len(results), "win_rate": float(np.mean([r["win"] for r in results])),
           "hp_delta": float(np.mean([r["hp_delta"] for r in results])), "stuck": int(sum(r["stuck"] for r in results))}
    by = defaultdict(list)
    for r in results:
        by[f"幕{r['act'] + 1}-{r['enc']}"].append(r["win"])
    out["by_encounter"] = {k: f"{np.mean(v):.2f}（{len(v)}）" for k, v in sorted(by.items())}
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint"); p.add_argument("--random", action="store_true")
    p.add_argument("--n", type=int, default=400); p.add_argument("--progress", type=float, default=0.0)
    p.add_argument("--envs", type=int, default=8); p.add_argument("--sample", action="store_true", help="按策略分布采样（默认贪心）")
    a = p.parse_args(argv)
    net = None
    if not a.random:
        blob = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
        mc = ModelConfig(**blob["config"]["model"])
        net = PolicyValueNet(mc); load_state(net, blob["state"], blob.get("content_names")); net.eval()
    print(json.dumps(evaluate(net, a.n, a.progress, a.envs, greedy=not a.sample), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
