"""候选未来读数头的残差实际改了多少战斗首选（2026-11-17，铁甲战士决策记录第 2 条补记；口径同愚者决策记录第 25 条补记）。
同一检查点算两遍打分：残差照常 / 残差输出层 fh_res[2] 置零，在学习端存下的一批学习数据上比较。
残差只加在战斗候选上，所以两遍打分有差别的步就是战斗步；只统计有两个以上合法候选的决策。
数据：在运行目录放 DUMP 标记，学习端存下下一批到 batch_dump.pkl（train.py，存完删标记）。
  cd <部署目录> && nice -n 10 .venv/bin/python -I tools/remote/ironclad/diag/fh_res_effect.py --ckpt latest.pt --batch batch_dump.pkl [--n 8192] [--threads 4]
用 CPU 算，不占学习端与评测的显存。
"""
from __future__ import annotations

import argparse
import copy
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path.cwd() / "python"))
from sts2rl.model import ModelConfig, PolicyValueNet, load_state  # noqa: E402
from sts2rl.train import minibatches  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--batch", required=True)
    ap.add_argument("--n", type=int, default=8192); ap.add_argument("--minibatch", type=int, default=256)
    ap.add_argument("--threads", type=int, default=4); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    dev = torch.device("cpu")
    blob = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    net = PolicyValueNet(ModelConfig(**blob["config"]["model"])).eval()
    load_state(net, blob["state"], blob.get("content_names"))
    if not net.cfg.future_head:
        print("检查点没有读数头"); return 1
    off = copy.deepcopy(net)
    with torch.no_grad():
        off.fh_res[2].weight.zero_(); off.fh_res[2].bias.zero_()
    w = float(net.fh_res[2].weight.detach().norm())
    data = pickle.load(open(a.batch, "rb"))
    if isinstance(data, dict):
        data = data["data"]
    perm = np.random.default_rng(a.seed).permutation(len(data.mode))[:a.n]
    tops, kls, rngs, gaps, pmax = [], [], [], [], []
    n_multi = n_diff = 0
    with torch.no_grad():
        for _idx, b in minibatches(data, perm, a.minibatch, dev, None, None, packed=False, workers=1):
            l1 = net.forward_all(b)[0].float(); l0 = off.forward_all(b)[0].float()
            ok = torch.isfinite(l0) & (l0 > -1e8)
            diff = torch.where(ok, (l1 - l0).abs(), torch.zeros_like(l0)).amax(1) > 1e-7   # 战斗步（残差只加在战斗候选上）；-inf 减 -inf 是 NaN，先用 where 去掉
            multi = ok.sum(1) >= 2
            m = diff & multi
            n_multi += int(multi.sum()); n_diff += int(diff.sum())
            if not m.any():
                continue
            L1, L0, K = l1[m], l0[m], ok[m]
            neg = torch.finfo(torch.float32).min
            L1 = torch.where(K, L1, torch.full_like(L1, neg)); L0 = torch.where(K, L0, torch.full_like(L0, neg))
            p1, p0 = torch.softmax(L1, 1), torch.softmax(L0, 1)
            tops.append((L1.argmax(1) != L0.argmax(1)).numpy())
            kls.append((p1 * (torch.log(p1.clamp_min(1e-30)) - torch.log(p0.clamp_min(1e-30))) * K).sum(1).numpy())
            r = torch.where(K, L1 - L0, torch.zeros_like(L1))
            rngs.append((torch.where(K, r, torch.full_like(r, -1e9)).amax(1) - torch.where(K, r, torch.full_like(r, 1e9)).amin(1)).numpy())
            s = torch.sort(L0, 1, descending=True).values
            gaps.append((s[:, 0] - s[:, 1]).numpy()); pmax.append(p0.amax(1).numpy())
    top, kl, rg, gp, pm = (np.concatenate(x) for x in (tops, kls, rngs, gaps, pmax))
    print(f"检查点 v{blob.get('version')}，残差输出层权重范数 {w:.4f}；抽 {len(perm)} 步，两遍打分有差别（战斗）的 {n_diff} 步，两个以上候选 {n_multi} 步，两者都满足的决策 {len(top)} 个")
    print(f"首选不同 {top.mean():.4%}（{int(top.sum())} 个）；KL(开‖关) 均值 {kl.mean():.2e}、最大 {kl.max():.2e}")
    for lo, hi in ((0, 0.5), (0.5, 0.9), (0.9, 1.01)):
        s = (pm >= lo) & (pm < hi)
        if s.any():
            print(f"  原首选概率 [{lo}, {min(hi, 1)}) {int(s.sum())} 个：首选不同 {top[s].mean():.4%}")
    print(f"同一决策内残差的极差：中位 {np.median(rg):.4f}、最大 {rg.max():.4f} 个对数几率单位；原首选与次选的对数几率差中位 {np.median(gp):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
