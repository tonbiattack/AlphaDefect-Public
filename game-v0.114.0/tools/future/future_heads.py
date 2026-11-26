"""未来推演方案初测的公共部分：从冻结的策略网络取候选表示，以及接到候选打分上的小残差头（2026-11-16）。

候选表示 z = 候选头 cand_mlp 的输入 [ha, hb, act, g]（4d = 1,024 维，见 model.py forward_all），报告第 6.3 节的接法：
新头读公开的 g 与候选引用，不读价值一侧的特权通路。残差头最后一层清零，接上时原 logits 不变。

只处理战斗内的候选（cand_mlp；局外的 cand_mlp_m 不动）：第三幕双 Boss 题里的决策都是战斗决策。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class ResidualHead(nn.Module):
    """z → 残差 logit；kind=pred 时输出 m 维未来读数，残差 = alpha · 读数[0]（第 0 维是与推演得分同一尺度的估值）。"""

    def __init__(self, d_in: int = 1024, hidden: int = 128, out: int = 1, kind: str = "residual", alpha: float = 1.0, zero: bool = True):
        super().__init__()
        self.kind, self.alpha = kind, float(alpha)
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, out))
        if zero:
            nn.init.zeros_(self.net[2].weight); nn.init.zeros_(self.net[2].bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z.float())

    def logit(self, z: torch.Tensor) -> torch.Tensor:
        y = self.forward(z)
        return y[:, :1] if self.kind == "residual" else self.alpha * y[:, :1]

    def save(self, path: str, **meta) -> None:
        torch.save({"kind": self.kind, "alpha": self.alpha, "shape": [self.net[0].in_features, self.net[0].out_features, self.net[2].out_features],
                    "state": self.state_dict(), **meta}, path)

    @staticmethod
    def load(path: str, device="cpu") -> "ResidualHead":
        b = torch.load(path, map_location="cpu", weights_only=False)
        d_in, h, out = b["shape"]
        m = ResidualHead(d_in, h, out, kind=b["kind"], alpha=b.get("alpha", 1.0), zero=False)
        m.load_state_dict(b["state"])
        return m.to(device).eval()


def attach_residual(net, path: str, device, scale: float = 1.0):
    """把残差头挂到 net.cand_mlp 的输出上（前向钩子），之后 net 的战斗候选打分 = 原打分 + scale × 残差。返回头（钩子句柄在 head._hook）。"""
    head = ResidualHead.load(path, device)

    def hook(mod, inp, out):
        with torch.no_grad():
            return out + (scale * head.logit(inp[0])).to(out.dtype)
    head._hook = net.cand_mlp.register_forward_hook(hook)
    return head


class Capture:
    """前向时记下战斗候选的 z（cand_mlp 的输入）。要求一批里全是战斗决策（宏观决策的候选走 cand_mlp_m，不在这里）。"""

    def __init__(self, net):
        self.z = None
        self.h = net.cand_mlp.register_forward_hook(self._hook)

    def _hook(self, mod, inp, out):
        self.z = inp[0].detach()

    def close(self):
        self.h.remove()


@torch.no_grad()
def extract(net, frames, device, mode: int = 0, chunk: int = 256):
    """frames: Obs 列表（全为战斗内决策）。返回 [(z: float16[n_i, 1024], logits: float32[n_i]), …] 与价值 float32[N]。
    逐块前向；核对钩子取到的行数与各帧候选数之和一致（顺序同 Ragged：按帧、帧内按候选）。"""
    from sts2rl.features import Ragged, collate_ragged
    import sim_pimc as X
    cap = Capture(net)
    out, vals = [], []
    try:
        for i in range(0, len(frames), chunk):
            fs = [X._no_priv(o) for o in frames[i:i + chunk]]
            b = collate_ragged(Ragged.from_obs(fs, np.full(len(fs), mode))).to(device)
            logits, values = net(b)
            z = cap.z
            ns = [int(o.num_actions) for o in fs]
            if z.shape[0] != sum(ns):
                raise RuntimeError(f"钩子取到 {z.shape[0]} 行，候选共 {sum(ns)} 个（批里有局外决策？）")
            zz = z.half().cpu().numpy()
            lg = logits.float().cpu().numpy()
            s = 0
            for j, n in enumerate(ns):
                out.append((zz[s:s + n], lg[j, :n].copy()))
                s += n
            vals.append(values.float().cpu().numpy())
    finally:
        cap.close()
    return out, np.concatenate(vals) if vals else np.zeros(0, np.float32)
