"""ReDo（Sokar 等 2023）式的死单元重置（决策记录 266，准备好、未上线）：找出各个头隐层里在一批真实局面上从不激活的神经元，
把它们的输入权重（第一层 Linear 的那一行与偏置）按默认初始化重新随机、输出权重（第二层 Linear 的那一列）清零；Adam 状态不动（死单元的矩本来接近 0）。
重置的单元输出权重为 0，所以重置当时网络的输出逐位不变（脚本会核对），之后这些单元重新参与学习。编码器不动（诊断里编码器几乎没有死单元）。

    .venv/bin/python -I tools/redo_reset.py --ckpt latest.pt --dump 学习数据.pkl --out 重置后.pt [--heads value_mlp,value_mlp_m,cand_mlp,cand_mlp_m] [--n 4096]
"""
from __future__ import annotations

import argparse
import math
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--heads", default="value_mlp,value_mlp_m,cand_mlp,cand_mlp_m")
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--mb", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import torch
    from sts2rl.features import collate_ragged
    from sts2rl.model import ModelConfig, PolicyValueNet, load_state, packed_meta
    torch.manual_seed(a.seed)
    dev = torch.device(a.device)
    blob = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    mc = blob["config"]["model"]
    net = PolicyValueNet(ModelConfig(**mc)).to(dev).eval(); load_state(net, blob["state"]); net.fast = dev.type == "cuda"
    r = pickle.load(open(a.dump, "rb"))
    data = r.select(np.sort(np.random.default_rng(0).choice(len(r), min(a.n, len(r)), replace=False)))
    heads = a.heads.split(",")
    act = {h: None for h in heads}
    hooks = []
    for h in heads:
        def f(_m, _i, out, h=h):
            s = out.float().abs().sum(0)
            act[h] = s if act[h] is None else act[h] + s
        hooks.append(getattr(net, h)[1].register_forward_hook(f))
    attend = bool(mc.get("macro_attend"))

    def batches():
        for j in range(0, len(data), a.mb):
            bt = collate_ragged(data.select(np.arange(j, min(j + a.mb, len(data)))), attend=attend)
            b = bt.to(dev)
            if net.fast:
                b.update({k: (torch.from_numpy(v).to(dev) if isinstance(v, np.ndarray) else v) for k, v in packed_meta(bt.core_idx, bt.core_mask).items()})
            yield b
    before = []
    with torch.no_grad():
        for b in batches():
            lg, v = net(b); before.append((lg.float().cpu(), v.float().cpu()))
    for x in hooks:
        x.remove()
    sd = net.state_dict()
    dead_idx = {}
    for h in heads:
        dead = (act[h] == 0).nonzero().squeeze(-1)
        dead_idx[h] = dead
        w1, b1, w2 = sd[f"{h}.0.weight"], sd[f"{h}.0.bias"], sd[f"{h}.2.weight"]
        fan_in = w1.shape[1]; bound = 1 / math.sqrt(fan_in)
        with torch.no_grad():
            w1[dead] = torch.empty(len(dead), fan_in, device=w1.device).uniform_(-bound, bound)     # 同 nn.Linear 的默认初始化
            b1[dead] = torch.empty(len(dead), device=b1.device).uniform_(-bound, bound)
            w2[:, dead] = 0
        print(f"{h}：隐层 {w1.shape[0]} 个单元，从不激活 {len(dead)} 个（{len(dead) / w1.shape[0]:.1%}），已重置", flush=True)
    # 核对：同一批局面重置前后的输出。显卡上 index_add 的累加顺序不固定，同一网络算两遍也有约 1e-5 量级的差，所以另算一遍“不重置的基准噪声”作比较
    def outs():
        res = []
        with torch.no_grad():
            for b in batches():
                lg, v = net(b); res.append((lg.float().cpu(), v.float().cpu()))
        return res
    fin = lambda x: torch.nan_to_num(x, neginf=0.0, posinf=0.0)
    after = outs()
    dl = max(float((fin(l0) - fin(l1)).abs().max()) for (l0, _), (l1, _) in zip(before, after))
    dv = max(float((v0 - v1).abs().max()) for (_, v0), (_, v1) in zip(before, after))
    agree = np.mean(np.concatenate([(l0.argmax(-1) == l1.argmax(-1)).numpy() for (l0, _), (l1, _) in zip(before, after)]))
    print(f"重置前后：策略对数几率最大差 {dl:.2e}，价值最大差 {dv:.2e}，首选一致 {agree:.4%}", flush=True)
    after_ok = dl < 1e-3 and dv < 1e-3 and agree > 0.999
    print("重置前后输出" + ("一致（在显卡数值误差内）" if after_ok else "不一致！不保存"), flush=True)
    if not after_ok:
        sys.exit(1)
    out = dict(blob); out["state"] = {k: v.detach().cpu() for k, v in net.state_dict().items()}
    # 滑动平均权重里这几个头也换成重置后的值，否则评测用的滑动平均是“新输入权重 + 旧输出权重”的混合
    if isinstance(out.get("ema"), dict):
        out["ema"] = dict(out["ema"])
        for h in heads:
            for k in (f"{h}.0.weight", f"{h}.0.bias", f"{h}.2.weight", f"{h}.2.bias"):
                if k in out["ema"]:
                    out["ema"][k] = out["state"][k].float().clone()
    # Adam 状态不动：死单元本来几乎收不到梯度，一、二阶矩接近 0；优化器的参数编号按学习端参数组的顺序，不是 net.parameters() 的顺序，这里不去改它
    out["redo"] = {"decision": 266, "heads": {h: int(len(d)) for h, d in dead_idx.items()}}
    torch.save(out, a.out)
    print(f"已存 {a.out}", flush=True)


if __name__ == "__main__":
    main()
