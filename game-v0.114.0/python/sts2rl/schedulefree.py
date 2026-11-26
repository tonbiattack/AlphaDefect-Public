"""免调度 Adam（Schedule-Free，Defazio 等 2024，“The Road Less Scheduled”；照 facebookresearch/schedule_free 的
AdamWScheduleFreeReference 写，不做权重衰减）。2026-11-15 用户：“关于 muon 的平均，既然没人验证过，那我们来验证，不过先验证 adam 那一块”。

三组参数：z 是用固定学习率更新的基础迭代（state["z"]），x 是 z 的加权平均（评测、发布用），y = (1−β1)·z + β1·x 是算梯度的点。
模型参数里存的始终是 y，学习端的前向、反向都在 y 上做；x = (y − (1−β1)·z) / β1 只在导出时算（x_state），不来回切换参数。
每步（k 从 0 起）：
    lr_k = lr × 预热系数；权重 c_k = lr_max^weight_lr_power × (k+1)^r，ckp1 = c_k / Σc
    v ← β2·v + (1−β2)·g²；ĝ = g / (√(v / (1 − β2^n)) + eps)，n 是这个参数自己的更新次数（从 Adam 接过来的二阶矩 n 记为很大，不再修正；
    新建的参数从 0 起照常修正，免得头几步 v 太小、步子过大）
    y ← (1−ckp1)·y + ckp1·z；y ← y + lr_k·(β1·(1−ckp1) − 1)·ĝ；z ← z − lr_k·ĝ
没有梯度的参数跳过、不建状态（Muon 管的二维权重：guarded_step 先由 Muon 更新，再把它们的 grad 清掉），导出时照原值。
"""
from __future__ import annotations

import torch


class AdamScheduleFree(torch.optim.Optimizer):
    def __init__(self, params, lr: float, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-5, warmup_steps: int = 0,
                 r: float = 0.0, weight_lr_power: float = 2.0):
        defaults = dict(lr=lr, betas=tuple(betas), eps=eps, warmup_steps=int(warmup_steps), r=r, weight_lr_power=weight_lr_power,
                        k=0, weight_sum=0.0, lr_max=-1.0)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for g in self.param_groups:
            beta1, beta2 = g["betas"]
            k, warm = g["k"], g["warmup_steps"]
            sched = (k + 1) / warm if k < warm else 1.0
            lr = g["lr"] * sched
            g["lr_max"] = lr_max = max(lr, g["lr_max"])
            weight = ((k + 1) ** g["r"]) * (lr_max ** g["weight_lr_power"])
            g["weight_sum"] = ws = g["weight_sum"] + weight
            ckp1 = weight / ws if ws > 0 else 0.0
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "z" not in st:
                    st["z"] = p.detach().clone()
                    st["exp_avg_sq"] = torch.zeros_like(p)
                    st["n"] = 0
                z, v = st["z"], st["exp_avg_sq"]
                st["n"] += 1
                st["sf_steps"] = st.get("sf_steps", 0) + 1
                v.mul_(beta2).addcmul_(p.grad, p.grad, value=1.0 - beta2)
                bc2 = 1.0 - beta2 ** min(st["n"], 100_000)
                gh = p.grad / (v / bc2).sqrt().add_(g["eps"])
                p.lerp_(z, ckp1)
                p.add_(gh, alpha=lr * (beta1 * (1.0 - ckp1) - 1.0))
                z.sub_(gh, alpha=lr)
            g["k"] = k + 1
        return loss

    # ---- 导出与互转

    @torch.no_grad()
    def x_of(self, p: torch.Tensor) -> torch.Tensor:
        """参数 p 的平均点 x；免调度没更新过的（Muon 管的、切过来之后还没有梯度的）就是 p 本身。"""
        st = self.state.get(p)
        if not st or "z" not in st or not st.get("sf_steps"):
            return p.detach()
        beta1 = next(g["betas"][0] for g in self.param_groups if any(q is p for q in g["params"]))
        return (p.detach() - (1.0 - beta1) * st["z"]) / beta1

    @torch.no_grad()
    def x_state(self, net: torch.nn.Module) -> dict[str, torch.Tensor]:
        """整个网络的状态字典，参数换成平均点 x（缓冲区照原值），放在 CPU 上：存检查点、做测试用。"""
        named = dict(net.named_parameters())
        return {k: (self.x_of(named[k]).to(v.dtype) if k in named else v).detach().cpu() for k, v in net.state_dict().items()}

    @torch.no_grad()
    def restart_average(self) -> int:
        """从当前平均点重新开始平均（config.sf_restart，愚者决策记录第 19 条）：参数（y）与 z 都换成 x，累计权重清零、lr_max 重记，
        下一步的平均系数是 1（x 就是新的 z），之后 x 只平均这以后的 z。步数 k 与各参数的 sf_steps、n 照留（不再预热、不再做偏差修正），二阶矩照留。
        返回处理的参数个数。"""
        n = 0
        for g in self.param_groups:
            beta1 = g["betas"][0]
            for p in g["params"]:
                st = self.state.get(p)
                if not st or "z" not in st:
                    continue
                if st.get("sf_steps"):
                    p.copy_((p - (1.0 - beta1) * st["z"]) / beta1)
                st["z"].copy_(p)
                n += 1
            g["weight_sum"], g["lr_max"] = 0.0, -1.0
        return n

    def adam_state_dict(self) -> dict:
        """换回普通 Adam 时能直接载入的状态（一阶矩为 0、二阶矩照搬、步数为 k）：免调度分支的检查点也能交回主线续训。"""
        sd = self.state_dict()
        state = {}
        for i, st in sd["state"].items():
            if "exp_avg_sq" in st:
                state[i] = {"step": torch.tensor(float(max(1, self.param_groups[0]["k"]))), "exp_avg": torch.zeros_like(st["exp_avg_sq"]),
                            "exp_avg_sq": st["exp_avg_sq"]}
        groups = [{"params": g["params"], "lr": g["lr"], "betas": g["betas"], "eps": g["eps"], "weight_decay": 0, "amsgrad": False,
                   "maximize": False, "foreach": None, "capturable": False, "differentiable": False, "fused": None} for g in sd["param_groups"]]
        return {"state": state, "param_groups": groups}

    @classmethod
    def from_adam(cls, adam: torch.optim.Optimizer, lr: float, betas=(0.9, 0.999), warmup_steps: int = 0, r: float = 0.0,
                  weight_lr_power: float = 2.0, sf_state: dict | None = None, exclude: set[int] | None = None) -> tuple["AdamScheduleFree", str]:
        """按 Adam 的参数组建免调度 Adam。sf_state（免调度分支自己的检查点）在就接着它；否则 z 从当前参数起，二阶矩接着 Adam 的
        （已经收敛，不再做偏差修正），从零累计平均。exclude 为另由别的优化器更新的参数编号（Muon 管的二维权重），不建状态。返回 (优化器, 说明)。"""
        eps = adam.param_groups[0].get("eps", 1e-5)
        sf = cls([{"params": g["params"]} for g in adam.param_groups], lr=lr, betas=betas, eps=eps, warmup_steps=warmup_steps, r=r,
                 weight_lr_power=weight_lr_power)
        if sf_state:
            sf.load_state_dict(sf_state)
            for g in sf.param_groups:
                g["lr"], g["warmup_steps"] = lr, int(warmup_steps)
            n = sum(1 for st in sf.state.values() if "z" in st)
            return sf, f"接着检查点里的免调度状态（{n} 个参数，已走 {sf.param_groups[0]['k']} 步）"
        n = 0
        for g in sf.param_groups:
            for p in g["params"]:
                v = (adam.state.get(p) or {}).get("exp_avg_sq")
                if v is not None and v.shape == p.shape and id(p) not in (exclude or set()):
                    sf.state[p] = {"z": p.detach().clone(), "exp_avg_sq": v.detach().clone(), "n": 100_000}
                    n += 1
        return sf, f"从 Adam 切过来：z 从当前参数起，二阶矩接着 Adam 的 {n} 个，平均从零累计"
