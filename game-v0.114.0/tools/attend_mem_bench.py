"""局外帧扩展注意力（决策记录 262）对学习端显存与速度的影响：按学习端的做法（打包变长注意力、bf16 混合精度）在真实学习数据上
做前向加反向，比较扩展前后不同小批大小的峰值显存与每个小批的耗时。

    .venv/bin/python -I tools/attend_mem_bench.py --ckpt 检查点.pt --dump 学习数据.pkl [--sizes 4096 2048] [--reps 6]
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--sizes", type=int, nargs="+", default=[4096, 2048])
    ap.add_argument("--reps", type=int, default=6)
    a = ap.parse_args()
    import torch
    from sts2rl.features import collate_ragged
    from sts2rl.model import ModelConfig, PolicyValueNet, load_state, packed_meta, use_tf32
    use_tf32()
    dev = torch.device("cuda")
    blob = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    r = pickle.load(open(a.dump, "rb"))
    mc = dict(blob["config"]["model"]); mc.pop("macro_attend", None)
    rng = np.random.default_rng(0)
    for attend in (False, True):
        net = PolicyValueNet(ModelConfig(**mc, macro_attend=attend)).to(dev)
        load_state(net, blob["state"]); net.fast = True; net.train()
        for size in a.sizes:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            ts, toks = [], []
            for k in range(a.reps):
                idx = np.sort(rng.choice(len(r), size, replace=False))
                bt = collate_ragged(r.select(idx), attend=attend)
                b = bt.to(dev); b.update({x: (torch.from_numpy(v).to(dev) if isinstance(v, np.ndarray) else v) for x, v in packed_meta(bt.core_idx, bt.core_mask).items()})
                toks.append(int(bt.core_mask.sum()) + 5 * size)
                torch.cuda.synchronize(); t0 = time.time()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lg, v, _, ex = net.forward_all(b)
                loss = torch.nan_to_num(lg.float(), neginf=0.0).mean() + v.float().mean() + sum(x.float().mean() for x in ex.values())
                loss.backward(); net.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                if k:
                    ts.append(time.time() - t0)
            print(f"{'扩展' if attend else '原来'} 小批 {size}：注意力序列总长平均 {np.mean(toks):,.0f}，峰值显存 {torch.cuda.max_memory_allocated() / 2**30:.2f} GB，"
                  f"每个小批 {np.mean(ts) * 1000:.0f} 毫秒", flush=True)
        del net; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
