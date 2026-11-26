"""可塑性诊断（决策记录 266）：同一批真实局面喂给不同时期的检查点，看网络有没有“可塑性丧失”的迹象。

每个检查点：
  - 休眠比例（ReDo，Sokar 等 2023）：神经元在这批局面上的平均激活绝对值 ÷ 本层所有神经元的平均，≤ τ 记为休眠（τ = 0 即从不激活，τ = 0.1 为文中常用）。
    看编码器各层前馈的 ReLU、四个头（战斗/局外的打分头与价值头）隐层的 GELU；
  - 有效秩（Kumar 等 2021 的 srank，δ = 0.01）：全局表示 g（每步一个 d 维向量）与核心记录上下文表示的奇异值，累计占 99% 所需的个数；
  - 各层权重范数与全局表示的平均范数（权重一路长大、表示范数飙升也是常见征兆）。

    .venv/bin/python -I tools/plasticity_probe.py --dump 学习数据.pkl --ckpt 标签=路径 [--ckpt ...] [--n 4096] [--device cuda]
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))


def srank(x, delta: float = 0.01) -> int:
    import torch
    s = torch.linalg.svdvals(x.float())
    c = torch.cumsum(s, 0) / s.sum().clamp(min=1e-12)
    return int((c < 1 - delta).sum()) + 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--ckpt", action="append", required=True, help="标签=路径")
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--mb", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    import torch
    import torch.nn.functional as F
    from sts2rl.features import collate_ragged
    from sts2rl.model import ModelConfig, PolicyValueNet, load_state, packed_meta
    dev = torch.device(a.device)
    r = pickle.load(open(a.dump, "rb"))
    idx = np.sort(np.random.default_rng(0).choice(len(r), min(a.n, len(r)), replace=False))
    data = r.select(idx)
    res = {}
    for spec in a.ckpt:
        label, path = spec.split("=", 1)
        blob = torch.load(path, map_location="cpu", weights_only=False)
        mc = blob.get("config", {}).get("model") or {}
        net = PolicyValueNet(ModelConfig(**mc)).to(dev).eval()
        load_state(net, blob["state"])
        net.fast = dev.type == "cuda"
        acc: dict[str, list] = {}

        def hook(name, relu):
            def f(_m, _i, out):
                h = (F.relu(out) if relu else out).float().abs()
                s = h.reshape(-1, h.shape[-1])
                a_ = acc.setdefault(name, [torch.zeros(s.shape[-1], device=s.device), 0])
                a_[0] += s.sum(0); a_[1] += s.shape[0]
            return f
        hs = []
        for i, layer in enumerate(net.encoder.layers):
            hs.append(layer.linear1.register_forward_hook(hook(f"编码器第{i}层前馈", True)))
        for nm in ("cand_mlp", "cand_mlp_m", "value_mlp", "value_mlp_m"):
            hs.append(getattr(net, nm)[1].register_forward_hook(hook(nm, False)))
        gs, cs = [], []
        attend = bool(mc.get("macro_attend"))
        with torch.no_grad():
            for j in range(0, len(data), a.mb):
                bt = collate_ragged(data.select(np.arange(j, min(j + a.mb, len(data)))), attend=attend)
                b = bt.to(dev)
                if net.fast:
                    b.update({k: (torch.from_numpy(v).to(dev) if isinstance(v, np.ndarray) else v) for k, v in packed_meta(bt.core_idx, bt.core_mask).items()})
                net.forward_all(b)
                e, ctx, g = net._encode(b)
                gs.append(g.float().cpu())
                core = b["core_idx"][b["core_mask"]]
                cs.append(ctx[core].float().cpu())
        for h in hs:
            h.remove()
        G = torch.cat(gs); C = torch.cat(cs)
        C = C[torch.randperm(len(C), generator=torch.Generator().manual_seed(0))[:20000]]
        rec = {"version": blob.get("version"), "dormant": {}, "srank_g": srank(G), "srank_ctx": srank(C), "g_norm": round(float(G.norm(dim=1).mean()), 3)}
        for name, (s, n) in acc.items():
            m = s / max(n, 1)
            score = m / m.mean().clamp(min=1e-12)
            rec["dormant"][name] = {"τ=0": round(float((m == 0).float().mean()), 4), "τ=0.1": round(float((score <= 0.1).float().mean()), 4), "单元数": int(len(m))}
        st = blob["state"]
        rec["weight_norm"] = {f"L{i}": round(float(sum(st[k].float().norm() ** 2 for k in st if k.startswith(f"encoder.layers.{i}.") and k.endswith("weight")) ** 0.5), 2)
                              for i in range(len(net.encoder.layers))}
        res[label] = rec
        d = rec["dormant"]
        print(f"{label}（第 {rec['version']} 版）：全局表示有效秩 {rec['srank_g']}/{G.shape[1]}，上下文有效秩 {rec['srank_ctx']}/{C.shape[1]}，全局表示范数 {rec['g_norm']}；"
              + "；".join(f"{k} 休眠 {v['τ=0.1']:.1%}（从不激活 {v['τ=0']:.1%}）" for k, v in d.items()), flush=True)
        del net
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    if a.out:
        Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
