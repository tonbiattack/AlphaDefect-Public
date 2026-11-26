"""离线查学习端非有限梯度的来源（决策记录 173 留档的 nan_batches/v版本.pkl）。不碰训练进程，在别的机器上跑。

1. 数据本身：观测里的浮点数组、优势、回报、旧对数概率有没有非有限值或极端值，候选数为 0 的步；
2. 用给定检查点按学习端同样的算法（bf16 自动混合精度、打包注意力、PPO 损失）逐个小批重算前向与反向，
   找出梯度不是有限值的小批，再逐步定位：哪些步的对数几率、对数概率、概率比、价值、熵不是有限值。
留档时的权重没有存，只能用前后最近的备份近似；若是数据本身的问题（如候选全被屏蔽、旧对数概率为 −inf），换哪份权重都会复现。

    python -I tools/nan_probe.py --batch v73740.pkl --ckpt v73880.pt [--ckpt v73400.pt] --device cuda
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from sts2env.protocol import DecisionKind  # noqa: E402
from sts2rl.config import ModelConfig  # noqa: E402
from sts2rl.model import PolicyValueNet, load_state  # noqa: E402
from sts2rl.train import minibatches  # noqa: E402


def scan_arrays(name: str, x) -> None:
    if isinstance(x, np.ndarray) and x.dtype.kind == "f" and x.size:
        bad = ~np.isfinite(x)
        print(f"  {name:16s} {str(x.shape):18s} 非有限 {int(bad.sum()):6d}  最小 {np.nanmin(x):.4g}  最大 {np.nanmax(x):.4g}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    ap.add_argument("--ckpt", action="append", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--minibatch", type=int, default=4096)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--bf16", type=int, default=1)
    ap.add_argument("--compile", type=int, default=0, help="与学习端一样用 torch.compile(dynamic=True) 的前向")
    ap.add_argument("--perms", type=int, default=0, help="另按几种随机顺序切小批（学习端每遍都随机打乱）")
    a = ap.parse_args()
    d = pickle.load(open(a.batch, "rb"))
    data = d["data"]
    print(f"第 {d['version']} 版留档：{len(data)} 步，学习时跳过 {d['skipped']} 个小批")
    print("数据：")
    for k, v in vars(data).items():
        scan_arrays(k, v)
    for k in ("ppo_olp", "ppo_adv", "ppo_ret", "ppo_w", "adv_raw"):
        scan_arrays(k, d[k])
    olp = d["ppo_olp"]
    print(f"  旧对数概率 < −20 的步 {int((olp < -20).sum())}，< −50 的 {int((olp < -50).sum())}；|优势(标准化)| > 20 的 {int((np.abs(d['ppo_adv']) > 20).sum())}；"
          f"|回报| > 20 的 {int((np.abs(d['ppo_ret']) > 20).sum())}")
    per_step = {k: d[k] for k in ("ppo_a", "ppo_olp", "ppo_adv", "ppo_ret", "ppo_w")}
    dev = torch.device(a.device)
    for path in a.ckpt:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        net = PolicyValueNet(ModelConfig(**ck["config"]["model"])).to(dev)
        load_state(net, ck["state"])
        net.fast = dev.type == "cuda"
        net.train()
        fwd = torch.compile(net, dynamic=True) if a.compile else net
        orders = [("按顺序", np.arange(len(data)))] + [(f"随机顺序 {k + 1}", np.random.default_rng(k).permutation(len(data))) for k in range(a.perms)]
        print(f"\n权重 {Path(path).name}（第 {ck['version']} 版）：{a.minibatch} 步一批重算，{'编译' if a.compile else '不编译'}，{len(orders)} 种切法")
        n_bad = n_all = 0
        batches = ((name, idx, b) for name, perm in orders for idx, b in minibatches(data, perm, a.minibatch, dev, None, per_step, packed=net.fast, workers=1))
        for name, idx, b in batches:
            n_all += 1
            act, olp_t, A, R, W = b["ppo_a"], b["ppo_olp"], b["ppo_adv"], b["ppo_ret"], b["ppo_w"]
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=bool(a.bf16) and dev.type == "cuda"):
                logits, v = fwd(b)
            logits, v = logits.float(), v.float()
            logits.retain_grad()
            dist = torch.distributions.Categorical(logits=logits, validate_args=False)
            lp = dist.log_prob(act); ratio = torch.exp(lp - olp_t)
            l_pi = -(W * torch.min(ratio * A, ratio.clamp(1 - a.clip, 1 + a.clip) * A)).sum() / W.sum()
            l_v = torch.nn.functional.mse_loss(v, R)
            ent_all = dist.entropy()
            loss = l_pi + a.vf_coef * l_v - 0.01 * ent_all.mean()
            net.zero_grad(set_to_none=True); loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(net.parameters(), float("inf"))
            if torch.isfinite(gn):
                continue
            n_bad += 1
            n_valid = torch.isfinite(logits).sum(1)
            row_bad = {"候选数为 0": n_valid == 0, "对数概率非有限": ~torch.isfinite(lp), "概率比非有限": ~torch.isfinite(ratio),
                       "价值非有限": ~torch.isfinite(v), "熵非有限": ~torch.isfinite(ent_all),
                       "对数几率梯度非有限": ~torch.isfinite(logits.grad).all(1) if logits.grad is not None else torch.zeros_like(v, dtype=torch.bool)}
            print(f"  {name}、含第 {idx[0]} 步的这一批：梯度范数 {gn.item()}，损失 策略 {l_pi.item():.4g} 价值 {l_v.item():.4g}")
            for k, m in row_bad.items():
                if m.any():
                    rows = torch.nonzero(m).flatten().tolist()
                    print(f"    {k}：{len(rows)} 步，如 {[int(idx[r]) for r in rows[:8]]}")
            for r in torch.nonzero(~torch.isfinite(lp) | ~torch.isfinite(ratio) | ~torch.isfinite(v)).flatten().tolist()[:5]:
                g = int(idx[r])
                print(f"    步 {g}：决策 {DecisionKind(int(b['decision'][r])).name}，候选 {int(n_valid[r])}，动作 {int(act[r])}，"
                      f"旧对数概率 {olp[g]:.4g}，新对数概率 {lp[r].item():.4g}，价值 {v[r].item():.4g}，优势 {d['ppo_adv'][g]:.4g}")
        print(f"  共 {n_all} 个小批，{n_bad} 个梯度不是有限值")


if __name__ == "__main__":
    main()
