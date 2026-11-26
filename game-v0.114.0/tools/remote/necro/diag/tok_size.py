"""亡灵契约师每帧 token 数的分布（2026-11-17，决策记录第 4 条补记：学习端显存不足的超大帧从哪来）。
用当前检查点按训练的开局请求（cut_flags(evaluating=False)）打整局，按策略采样出手（同采集进程），记每帧 token 数；
超过 --big 的帧按（记录类型、区域）统计构成，并记楼层、是否战斗、这场战斗第几回合。几个进程并行，CPU 推理。

    REPO=/root/necro-deploy/sls2 nice -n 19 $REPO/.venv/bin/python -I tok_size.py --dll 引擎.dll --ckpt latest.pt \\
        --config /root/necro_train_config.json --runs 32 --procs 4
"""
from __future__ import annotations

import argparse
import collections
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

REPO = Path(os.environ.get("REPO") or Path(__file__).resolve().parents[4])
sys.path.insert(0, str(REPO / "python"))


def work(args: tuple) -> dict:
    import numpy as np
    import torch
    from sts2env.env import Sts2Env
    from sts2env.protocol import parse
    from sts2rl.actor import MODE_RUN
    from sts2rl.config import TrainConfig
    from sts2rl.features import Ragged, collate_ragged
    from sts2rl.model import PolicyValueNet, load_state

    a, seeds = args
    torch.set_num_threads(2)
    cfg = TrainConfig.from_args(["--config", a["config"], "--run-dir", "/tmp/tok_size"])
    blob = torch.load(a["ckpt"], map_location="cpu", weights_only=False)
    net = PolicyValueNet(cfg.model).eval()
    load_state(net, blob["state"], blob.get("content_names"))
    base = {"op": "reset", "mode": "run", "character": cfg.character, "ascension": 10, "route_summary": cfg.route_summary,
            **cfg.cut_flags(evaluating=False)}
    sizes: list[int] = []
    big = collections.Counter()      # (类型, 区域) → token 数，只算超大帧
    big_frames = []
    per_run = []
    rng = np.random.default_rng(abs(hash(seeds[0])) % 2**32)
    with Sts2Env(Path(a["dll"])) as e:
        for seed in seeds:
            e._proc.send({**base, "seed": seed}); o = parse(e._proc.recv())
            mx = 0; steps = 0
            while not o.done and steps < 20000:
                n = int(o.tok_ints.shape[0]); sizes.append(n); mx = max(mx, n); steps += 1
                if n > a["big"]:
                    c = collections.Counter(zip(o.tok_ints[:, 0].tolist(), o.tok_ints[:, 2].tolist()))
                    big.update(c)
                    if len(big_frames) < 30:
                        big_frames.append({"seed": seed, "n": n, "floor": int(o.floor), "combat": bool(o.in_combat),
                                           "top": [[int(k), int(z), int(v)] for (k, z), v in c.most_common(6)]})
                with torch.no_grad():
                    lg, _ = net(collate_ragged(Ragged.from_obs([o], np.full(1, MODE_RUN))).to("cpu"))
                k = o.num_actions
                p = torch.softmax(lg[0, :k].float(), -1).numpy().astype(np.float64); p /= p.sum()
                o = e.step(int(rng.choice(k, p=p)))
            per_run.append({"seed": seed, "max": mx, "steps": steps, "floor": int(o.floor)})
    return {"sizes": sizes, "big": [[k, z, v] for (k, z), v in big.items()], "big_frames": big_frames, "per_run": per_run}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dll", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--runs", type=int, default=32)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--big", type=int, default=1500)
    a = ap.parse_args()
    seeds = [f"TOKSIZE{i:04d}" for i in range(a.runs)]
    jobs = [(vars(a), seeds[i::a.procs]) for i in range(a.procs)]
    with mp.get_context("spawn").Pool(a.procs) as pool:
        res = pool.map(work, jobs)
    import numpy as np
    sizes = np.array([s for r in res for s in r["sizes"]])
    big = collections.Counter()
    for r in res:
        for k, z, v in r["big"]:
            big[(k, z)] += v
    per_run = sorted((x for r in res for x in r["per_run"]), key=lambda x: -x["max"])
    q = [50, 90, 99, 99.9, 100]
    out = {"帧数": int(sizes.size), "分位": {str(p): float(np.percentile(sizes, p)) for p in q},
           "平均": round(float(sizes.mean()), 1), f"超过{a.big}的帧": int((sizes > a.big).sum()),
           "超大帧的构成（类型,区域→token 数）": [[k, z, v] for (k, z), v in big.most_common(12)],
           "单局最大前 8": per_run[:8], "超大帧样例": [f for r in res for f in r["big_frames"]][:12]}
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
