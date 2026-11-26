"""Muon 优化器（储君决策记录第 25 条，2026-11-09）：二维权重的更新取动量的正交化。

Jordan 等 2024（modded-nanogpt）：动量 → Newton–Schulz 五次迭代近似 U·Vᵀ（把更新的奇异值都推到 1 附近）→ 更新。
Liu 等 2025（Moonlight，“Muon is Scalable for LLM Training”）：再乘 0.2·√max(行, 列)，每步更新的均方根约为 0.2 × 学习率，与 AdamW 同一量纲。
动量用插值式 buf ← μ·buf + (1 − μ)·g，与 Adam 的一阶矩 exp_avg 是同一个量，从 Adam 切过来时直接接着它（见 init_from_adam）。

免调度平均（sf=True，2026-11-15，愚者决策记录第 16 条；用户：“muon 也改为平均”）：照 schedulefree.AdamScheduleFree 的办法，参数里存 y（算梯度的点），
z 用 Muon 的正交化方向按固定学习率走，平均点 x = (y − (1−β)·z)/β 只在导出时算；每步 y ← (1−c)·y + c·z，y ← y + lr·(β·(1−c) − 1)·d，z ← z − lr·d，
d 为正交化后的方向（已乘 rms·√max(行, 列)），c = lr_max^power·k^r / Σ。Muon 自己的动量照留（正交化作用在动量上），与免调度的插值叠在一起，
没有现成的理论或先例，靠试验看。

只管隐藏层的二维权重（select）：编码器的注意力（in_proj_weight 按 Q、K、V 三块各自正交化）与前馈、各头的隐藏层、中间投影。
嵌入、输入维很小的特征投影、各头的输出层、偏置、LayerNorm 与门控标量仍由原来的 Adam 管：guarded_step 先走 Muon，
再把这些参数的梯度清掉，融合 Adam 跳过没有梯度的参数（与只训价值一侧时同一机制）。Adam 里这些参数的旧状态原样留着。
"""
from __future__ import annotations

from collections import defaultdict

import torch
from torch import nn


def select(net: nn.Module, min_dim: int = 64) -> list[tuple[str, nn.Parameter, int]]:
    """交给 Muon 的参数：[(名字, 参数, 按第 0 维切成几块)]。nn.Linear（含注意力的 out_proj）的权重，两维都不小于 min_dim，
    且不是某个 nn.Sequential 头的最后一层（输出层）；注意力的 in_proj_weight 切成 3 块。
    候选未来读数头（fh_ 开头，model.future_head）有自己的 Adam，不交给 Muon。"""
    last = set()
    for mn, m in net.named_modules():
        if isinstance(m, nn.Sequential):
            lins = [f"{mn}.{cn}" if mn else cn for cn, c in m.named_children() if isinstance(c, nn.Linear)]
            if lins:
                last.add(lins[-1])
    out = []
    for mn, m in net.named_modules():
        if isinstance(m, nn.MultiheadAttention) and m.in_proj_weight is not None and m.in_proj_weight.requires_grad:
            out.append((f"{mn}.in_proj_weight", m.in_proj_weight, 3))
        elif isinstance(m, nn.Linear) and mn not in last and min(m.weight.shape) >= min_dim and m.weight.requires_grad and not mn.startswith("fh_") and not mn.endswith("_v29"):
            out.append((f"{mn}.weight", m.weight, 1))
    return out


def orthogonalize(g: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """[B, m, n] 逐个矩阵做 Newton–Schulz 迭代（Jordan 等 2024 的五次多项式系数，bfloat16），近似 U·Vᵀ。"""
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.bfloat16()
    tall = x.size(-2) > x.size(-1)
    if tall:
        x = x.mT
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        s = x @ x.mT
        x = a * x + (b * s + c * s @ s) @ x
    return x.mT if tall else x


class Muon:
    def __init__(self, items: list[tuple[str, nn.Parameter, int]], lr: float, momentum: float = 0.95, nesterov: bool = True,
                 ns_steps: int = 5, rms: float = 0.2, sf: bool = False, sf_beta: float = 0.9, sf_warmup: int = 0, sf_power: float = 2.0,
                 sf_r: float = 0.0):
        self.items = items
        self.params = [p for _, p, _ in items]
        self.lr, self.momentum, self.nesterov, self.ns_steps, self.rms = lr, momentum, nesterov, ns_steps, rms
        self.buf = {n: torch.zeros_like(p, dtype=torch.float32) for n, p, _ in items}
        self.sf, self.sf_beta, self.sf_warmup, self.sf_power, self.sf_r = sf, sf_beta, int(sf_warmup), sf_power, sf_r
        self.sf_k, self.sf_weight_sum, self.sf_lr_max = 0, 0.0, -1.0
        self.z = {n: p.detach().float().clone() for n, p, _ in items} if sf else {}
        self.by_shape: dict[tuple[int, int], list[tuple[str, nn.Parameter, int]]] = defaultdict(list)
        for n, p, k in items:
            self.by_shape[(p.shape[0] // k, p.shape[1])].append((n, p, k))

    def init_from_adam(self, opt: torch.optim.Optimizer) -> int:
        """动量接着 Adam 的一阶矩（同为 μ = 0.9 左右的梯度滑动平均）；返回接上的个数，没有状态的从零开始。"""
        n_ok = 0
        for n, p, _ in self.items:
            st = opt.state.get(p) or {}
            m = st.get("exp_avg")
            if m is not None and m.shape == p.shape:
                self.buf[n].copy_(m.float()); n_ok += 1
        return n_ok

    @torch.no_grad()
    def step(self, ok: torch.Tensor) -> None:
        """ok：显卡上的 0 维张量，1 为正常更新，0 为这一步跳过（梯度不是有限值）——动量与参数都不变，主机不用等显卡。
        梯度为 None 的参数（只训价值一侧时）不动。"""
        mu = self.momentum
        okf = ok.float()
        if self.sf:     # 免调度：这一步的学习率与平均系数（预热按步数线性升）
            k = self.sf_k
            lr = self.lr * ((k + 1) / self.sf_warmup if k < self.sf_warmup else 1.0)
            self.sf_lr_max = max(lr, self.sf_lr_max)
            w = ((k + 1) ** self.sf_r) * (self.sf_lr_max ** self.sf_power)
            self.sf_weight_sum += w
            ckp1 = w / self.sf_weight_sum if self.sf_weight_sum > 0 else 0.0
            self.sf_k = k + 1
        for (m, n), its in self.by_shape.items():
            its = [x for x in its if x[1].grad is not None]
            if not its:
                continue
            us = []
            for name, p, k in its:
                g = torch.nan_to_num(p.grad.float(), nan=0.0, posinf=0.0, neginf=0.0)
                buf = self.buf[name]
                buf.lerp_(g, okf * (1.0 - mu))
                us.append((g.lerp(buf, mu) if self.nesterov else buf).view(k, m, n))
            o = orthogonalize(torch.cat(us), self.ns_steps)
            off = 0
            if self.sf:
                scale = self.rms * max(m, n) ** 0.5
                for name, p, kk in its:
                    d = o[off:off + kk].reshape(p.shape).float() * scale
                    z = self.z[name]
                    p.lerp_(z.to(p.dtype), (okf * ckp1).to(p.dtype))
                    p.add_((d * (okf * lr * (self.sf_beta * (1.0 - ckp1) - 1.0))).to(p.dtype))
                    z.sub_(d * (okf * lr))
                    off += kk
                continue
            step = okf * (-self.lr * self.rms * max(m, n) ** 0.5)
            for name, p, k in its:
                p.add_(o[off:off + k].reshape(p.shape).to(p.dtype) * step)
                off += k

    @torch.no_grad()
    def x_of(self, name: str, p: torch.Tensor) -> torch.Tensor:
        """免调度的平均点 x；没开免调度或还没走过一步时就是 p 本身。"""
        if not self.sf or not self.sf_k or name not in self.z:
            return p.detach()
        return ((p.detach().float() - (1.0 - self.sf_beta) * self.z[name]) / self.sf_beta).to(p.dtype)

    @torch.no_grad()
    def restart_average(self) -> int:
        """从当前平均点重新开始免调度平均（config.sf_restart，愚者决策记录第 19 条），同 AdamScheduleFree.restart_average：
        参数（y）与 z 都换成 x，累计权重清零、lr_max 重记；步数 sf_k 照留（不再预热），动量照留。没开免调度时什么都不做。返回处理的参数个数。"""
        if not self.sf:
            return 0
        n = 0
        for name, p, _ in self.items:
            if name not in self.z:
                continue
            if self.sf_k:
                p.copy_(self.x_of(name, p))
            self.z[name].copy_(p.detach().float())
            n += 1
        self.sf_weight_sum, self.sf_lr_max = 0.0, -1.0
        return n

    def state_dict(self) -> dict:
        sd = {"buf": {n: b.detach().cpu() for n, b in self.buf.items()}, "lr": self.lr, "momentum": self.momentum}
        if self.sf:
            sd["sf"] = {"z": {n: z.detach().cpu() for n, z in self.z.items()}, "k": self.sf_k, "weight_sum": self.sf_weight_sum,
                        "lr_max": self.sf_lr_max}
        return sd

    def load_state_dict(self, sd: dict) -> int:
        """返回接上动量的个数；免调度时检查点里有 sf 就接着它（z 与平均的进度），没有就从当前参数起（构造时已设好）。"""
        n_ok = 0
        for n, b in (sd.get("buf") or {}).items():
            if n in self.buf and self.buf[n].shape == b.shape:
                self.buf[n].copy_(b.to(self.buf[n].device)); n_ok += 1
        sf = sd.get("sf")
        if self.sf and sf:
            for n, z in (sf.get("z") or {}).items():
                if n in self.z and self.z[n].shape == z.shape:
                    self.z[n].copy_(z.to(self.z[n].device))
            self.sf_k, self.sf_weight_sum, self.sf_lr_max = int(sf.get("k", 0)), float(sf.get("weight_sum", 0.0)), float(sf.get("lr_max", -1.0))
        return n_ok
