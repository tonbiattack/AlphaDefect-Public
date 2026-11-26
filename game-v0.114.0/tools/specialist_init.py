"""第三幕双 Boss 专项模型的起点（决策记录 287）：从学生 e 生成新运行目录的 latest.pt（第 0 版、不带优化器状态，训练端会重建优化器并预热学习率）。

  warm：学生 e 的全部权重，二维及以上的参数做一次 shrink-and-perturb：θ ← λθ + (1 − λ)θ_new（θ_new 为同结构的新初始化），
        偏置与 LayerNorm 不动（Juliani & Ash 2024：soft shrink-and-perturb 加 LayerNorm 在 PPO 的可塑性测试里最好）；
  reset：新初始化的网络，只保留学生 e 的词表嵌入表（各类 *_emb 与 4096 行的编号表：牌、遗物、怪物等的语义），编码器与全部头重新初始化。

    python -I tools/specialist_init.py --ckpt student_e.pt --mode warm --lam 0.9 --out 运行目录/latest.pt [--seed 0]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

from sts2rl.model import ModelConfig, PolicyValueNet, load_state  # noqa: E402


def is_vocab_table(name: str, p: torch.Tensor, vocab: int) -> bool:
    return name.split(".")[0].endswith("_emb") or (p.dim() == 2 and p.shape[0] == vocab)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--mode", choices=("warm", "reset"), required=True)
    ap.add_argument("--lam", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    blob = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    mc = dict(blob["config"]["model"]); mc.pop("oracle", None)
    cfg = ModelConfig(**mc)
    src = PolicyValueNet(cfg); load_state(src, blob["state"])
    torch.manual_seed(a.seed)
    fresh = PolicyValueNet(cfg)
    fs = dict(fresh.named_parameters())
    n_mix = n_keep = n_new = 0
    with torch.no_grad():
        for name, p in src.named_parameters():
            q = fs[name]
            if a.mode == "warm":
                if p.dim() >= 2:
                    p.mul_(a.lam).add_(q, alpha=1.0 - a.lam); n_mix += p.numel()
                else:
                    n_keep += p.numel()
            else:
                if is_vocab_table(name, p, cfg.vocab_size):
                    n_keep += p.numel()
                else:
                    p.copy_(q); n_new += p.numel()
    out = {k: v for k, v in blob.items() if k not in ("opt", "ema", "state")}
    out.update(state=src.state_dict(), version=0, total_steps=0, specialist={"mode": a.mode, "lam": a.lam, "seed": a.seed, "from": str(a.ckpt),
                                                                            "from_version": blob.get("version")})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(a.out) + ".tmp"); torch.save(out, tmp); tmp.replace(a.out)
    print(f"{a.mode}：混合 {n_mix:,}、保留 {n_keep:,}、重新初始化 {n_new:,} 个参数 → {a.out}（第 0 版，不带优化器状态）")


if __name__ == "__main__":
    main()
