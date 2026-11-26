"""模型扩容（不损失已学到的东西）：加深 Transformer 层数、加宽前馈层，扩容后的网络与原网络输出完全相同，之后再慢慢学会用新容量。

    .venv/bin/python -I tools/grow_model.py --in 检查点.pt --out 扩容后.pt --layers 6 --ff-mult 4

做法（编码器是先归一化的残差结构 norm_first）：
  - 原有的层：注意力原样拷贝；前馈层第一层多出的行随机初始化、偏置为 0，第二层多出的列为 0，新增的隐藏单元对输出没有贡献；
  - 新增的层：随机初始化，但注意力输出投影和前馈第二层置 0，整层等于直接跳过；
  - 其余（嵌入、各个头、归一化）原样拷贝。
扩容后用本机引擎采一批真实局面（整局与单场战斗），比对新旧网络的动作打分与局面估值，差异超过 1e-4 就报错不写文件。
输出的检查点不带优化器状态（参数的个数和形状变了）：训练载入时重新建立优化器，并在前若干次更新里逐步升学习率。
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from sts2env import VecEnv  # noqa: E402
from sts2rl.features import Ragged, collate_ragged  # noqa: E402
from sts2rl.model import ModelConfig, PolicyValueNet, load_state  # noqa: E402


def grow(old: PolicyValueNet, new: PolicyValueNet, old_layers: int, old_ff: int) -> None:
    so, sn = old.state_dict(), new.state_dict()
    out = {}
    for k, v in sn.items():
        if k.startswith("encoder.layers."):
            i = int(k.split(".")[2])
            if i >= old_layers:                       # 新增的层：输出置 0，整层等于跳过
                out[k] = torch.zeros_like(v) if (".self_attn.out_proj." in k or ".linear2." in k) else v.clone()
                continue
        ov = so.get(k)
        if ov is None:
            raise RuntimeError(f"原网络里没有 {k}")
        if ov.shape == v.shape:
            out[k] = ov.clone()
        elif k.endswith("linear1.weight"):          # [ff, d]：多出的行保留随机初始化
            w = v.clone(); w[:old_ff] = ov; out[k] = w
        elif k.endswith("linear1.bias"):
            b = torch.zeros_like(v); b[:old_ff] = ov; out[k] = b
        elif k.endswith("linear2.weight"):          # [d, ff]：多出的列为 0
            w = torch.zeros_like(v); w[:, :old_ff] = ov; out[k] = w
        else:
            raise RuntimeError(f"{k} 形状 {tuple(ov.shape)} → {tuple(v.shape)}，不知道怎么扩")
    new.load_state_dict(out)


def sample_obs(n_env: int = 6, steps: int = 60) -> list[tuple[object, int]]:
    """本机引擎随机走几步，收一批真实局面（整局与单场战斗各一半）。"""
    rng = np.random.default_rng(0)
    obs = []
    with VecEnv(n_env) as env:
        for i in range(n_env):
            env.send(i, {"op": "reset", "mode": "run", "character": "DEFECT", "ascension": 10, "seed": f"GROW{i}"} if i % 2 == 0 else
                     {"op": "reset", "mode": "combat", "spec": {"character": "DEFECT", "ascension": 10, "seed": f"GROW{i}", "act": i % 3, "encounter": "elite"}})
        live = set(range(n_env))
        for _ in range(steps):
            for i, o in env.recv_ready(timeout=60):
                if isinstance(o, Exception) or o.done or o.stuck or o.num_actions == 0:
                    live.discard(i); continue
                obs.append((o, 1 if i % 2 == 0 else 0))
                env.send(i, {"op": "step", "action": int(rng.integers(o.num_actions))})
            if not live:
                break
    return obs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--ff-mult", type=int, default=4)
    a = ap.parse_args()
    blob = torch.load(a.inp, map_location="cpu", weights_only=False)
    mc_old = dict((blob.get("config") or {}).get("model") or {})
    old = PolicyValueNet(ModelConfig(**mc_old)).eval()
    load_state(old, blob["state"])
    mc_new = dict(mc_old, layers=a.layers, ff_mult=a.ff_mult)
    torch.manual_seed(0)
    new = PolicyValueNet(ModelConfig(**mc_new)).eval()
    grow(old, new, mc_old["layers"], mc_old["ff_mult"] * mc_old["d"])
    n_old = sum(p.numel() for p in old.encoder.parameters()); n_new = sum(p.numel() for p in new.encoder.parameters())
    print(f"编码器 {mc_old['layers']} 层 × 前馈 {mc_old['ff_mult'] * mc_old['d']} → {a.layers} 层 × 前馈 {a.ff_mult * mc_old['d']}："
          f"参数 {n_old / 1e6:.2f} M → {n_new / 1e6:.2f} M")

    obs = sample_obs()
    worst = [0.0, 0.0]
    with torch.no_grad():
        for k in range(0, len(obs), 32):
            chunk = obs[k:k + 32]
            b = collate_ragged(Ragged.from_obs([o for o, _ in chunk], np.array([m for _, m in chunk]))).to("cpu")
            lo, vo = old(b); ln, vn = new(b)
            fin = torch.isfinite(lo)
            worst[0] = max(worst[0], float((lo[fin] - ln[fin]).abs().max()))
            worst[1] = max(worst[1], float((vo - vn).abs().max()))
    print(f"核对 {len(obs)} 个真实局面：动作打分最大差 {worst[0]:.2e}，局面估值最大差 {worst[1]:.2e}")
    if max(worst) > 1e-4:
        sys.exit("扩容后与原网络输出不一致，不写文件")
    out = {k: v for k, v in blob.items() if k != "opt"}
    out["state"] = new.state_dict()
    out["config"] = copy.deepcopy(blob.get("config") or {}); out["config"]["model"] = mc_new
    out["grown"] = {"from": mc_old, "to": mc_new, "version": blob.get("version")}
    torch.save(out, a.out)
    print("已写出", a.out, "（不带优化器状态）")


if __name__ == "__main__":
    main()
