"""几个快照两两之间的首选一致率（决策记录 274 补记三）：判断续训攒下的快照拉开了多少差异、够不够拿来做下一轮“集成 → 蒸馏”。
SL 快照池里单个快照与 62 个的集成首选不同约 11%，不加锚定的主线相隔 160 版约 8%；两两不同到 6%–8% 时集成才平均得掉东西。

    .venv/bin/python -I tools/snap_agree.py --dump 局面.pkl 快照1.pt 快照2.pt ... [--n 8192]
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--n", type=int, default=8192)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("ckpts", nargs="+")
    a = ap.parse_args()
    import torch
    from sts2rl.features import collate_ragged
    from sts2rl.model import ModelConfig, PolicyValueNet, load_state, packed_meta
    from sts2env.protocol import DecisionKind
    r = pickle.load(open(a.dump, "rb"))
    multi = np.flatnonzero(np.diff(r.cand_off) > 1)
    data = r.select(np.sort(np.random.default_rng(0).choice(multi, min(a.n, len(multi)), replace=False)))
    combat = (data.header[:, 11] != 0) | (data.header[:, 1] == int(DecisionKind.COMBAT))
    dev = torch.device(a.device)
    acts, names = [], []
    for p in a.ckpts:
        blob = torch.load(p, map_location="cpu", weights_only=False)
        mc = blob["config"]["model"]
        net = PolicyValueNet(ModelConfig(**mc)).to(dev).eval(); load_state(net, blob["state"]); net.fast = dev.type == "cuda"
        out = []
        with torch.no_grad():
            for j in range(0, len(data), 512):
                bt = collate_ragged(data.select(np.arange(j, min(j + 512, len(data)))), attend=bool(mc.get("macro_attend"))); b = bt.to(dev)
                if net.fast:
                    b.update({k: (torch.from_numpy(v).to(dev) if isinstance(v, np.ndarray) else v) for k, v in packed_meta(bt.core_idx, bt.core_mask).items()})
                out.append(net(b)[0].float().argmax(-1).cpu().numpy())
        acts.append(np.concatenate(out)); names.append(f"v{blob.get('version')}")
        del net
    print(f"局面 {len(data)}（战斗 {int(combat.sum())}、局外 {int((~combat).sum())}），首选不同的比例（全部 / 战斗 / 局外）：", flush=True)
    for i in range(len(acts)):
        for j in range(i + 1, len(acts)):
            d = acts[i] != acts[j]
            print(f"  {names[i]} 对 {names[j]}：{d.mean():.2%} / {d[combat].mean():.2%} / {d[~combat].mean():.2%}", flush=True)


if __name__ == "__main__":
    main()
