"""搜索蒸馏的规模曲线（2026-11-16，愚者决策记录第 26 条；用户：“在B上直接换引擎，然后在C上做搜索蒸馏的规模曲线”）。

问题：第 25 条测试二在 1.45 万个根上，冻结表示的小头（S1 蒸馏老师选择、S2 拟合老师配对差）对老师候选差的相关都在 0 附近。
是数据太少，还是这个信号本来就学不进来？局面复制引擎让老师标签便宜了约 2.4 倍，在 C 上多出约一个数量级的标签，看学生随数据量的走势：
随数据涨 → 在线老师标签值得做（C 持续出标签、B 的训练加蒸馏损失）；平 → 关掉这条路。

输入：heads.py feats 产出的根表（--roots，可给多个，按出现顺序拼接；题号加偏移避免撞号）。
分组（按题号，一道题的所有根同组）：题号 % 5 == 0 为固定留出组（报告用）；% 5 == 1 为验证组（选系数 α 与早停）；
其余为训练池，按题号的固定随机顺序取前 f（嵌套子集，f 小的是 f 大的子集）。每档训 S1、S2 各 --seeds 个种子。
评价同 heads.py：留出组的候选差相关（corr_diff）、显著对的符号一致率、老师改选时命中、没改选时保持原首选、按老师配对差折算的收益/根。

  scale_curve.py --roots roots.pt --out 目录 [--fracs 1/32,1/16,1/8,1/4,1/2,1] [--seeds 2]
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import heads as Hd  # noqa: E402
from future_heads import ResidualHead  # noqa: E402


def load_roots(paths: list[str]) -> dict:
    """拼接多个根表：Z、BL 接在一起，off 平移，题号加 100,000 × 文件序号（不同题集的题号不撞）。"""
    Ds = [torch.load(p, weights_only=False) for p in paths]
    if len(Ds) == 1:
        return Ds[0]
    roots, Z, BL, off = [], [], [], [0]
    for j, D in enumerate(Ds):
        base = off[-1]
        for r in D["roots"]:
            roots.append({**r, "id": int(r["id"]) + 100_000 * j})
        Z.append(D["Z"]); BL.append(D["BL"])
        off.extend(base + x for x in D["off"][1:])
    return {"roots": roots, "Z": torch.cat(Z), "BL": torch.cat(BL), "off": off}


def train_s1(rows, va, D, dev, a, seed):
    """S1：被搜候选里交叉熵拟合老师的选择（同 heads.py），每轮在验证组上按收益/根挑最好的一轮。"""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    Zd, BLd = D["Zd"], D["BLd"]
    head = ResidualHead(kind="residual").to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=1e-4)
    fn = lambda z: head(z)[:, 0]
    best = (-1e9, None, 0)
    for ep in range(a.epochs):
        perm = rng.permutation(len(rows))
        for s in range(0, len(perm), a.bs):
            seg = [rows[i] for i in perm[s:s + a.bs]]
            idx = torch.tensor([a0 + c for a0, cs, _ in seg for c in cs], device=dev)
            lens = [len(cs) for _, cs, _ in seg]
            r = head(Zd[idx]).reshape(-1)
            lg = (BLd[idx] + r).split(lens)
            L = torch.full((len(seg), max(lens)), float("-inf"), device=dev)
            for j, x in enumerate(lg):
                L[j, :len(x)] = x
            tgt = torch.tensor([cs.index(b) for _, cs, b in seg], device=dev)
            loss = F.cross_entropy(L, tgt) + a.l2 * (r ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % a.eval_every == a.eval_every - 1 or ep == a.epochs - 1:
            g = Hd.evaluate(va, D, fn, 1.0, dev)["gain_per_root"]
            if g > best[0]:
                best = (g, copy.deepcopy(head.state_dict()), ep + 1)
    head.load_state_dict(best[1])
    return head, best[2]


def train_s2(pairs, va, D, dev, a, seed):
    """S2：F(z_c) − F(z_原首选) 拟合老师配对差（5 个读数，按标准差归一，权重 = 世界数）；每轮在验证组上看候选差相关，挑最好的一轮。"""
    torch.manual_seed(seed)
    Zd = D["Zd"]
    h = ResidualHead(out=Hd.M, kind="pred", zero=False).to(dev)
    o = torch.optim.AdamW(h.parameters(), lr=a.lr, weight_decay=1e-4)
    ic = torch.tensor([p[0] for p in pairs], device=dev); ig = torch.tensor([p[1] for p in pairs], device=dev)
    Y = torch.tensor(np.stack([p[2] for p in pairs]), device=dev); W = torch.tensor([p[3] for p in pairs], device=dev, dtype=torch.float32)
    sd = Y.std(0) + 1e-3
    fn = lambda z: h(z)[:, 0]
    best = (-1e9, None, 0)
    for ep in range(a.epochs):
        perm = torch.randperm(len(Y), device=dev)
        for s in range(0, len(perm), a.bs * 4):
            k = perm[s:s + a.bs * 4]
            d = h(Zd[ic[k]]) - h(Zd[ig[k]])
            loss = ((((d - Y[k]) / sd) ** 2).mean(1) * W[k]).sum() / W[k].sum()
            o.zero_grad(); loss.backward(); o.step()
        if ep % a.eval_every == a.eval_every - 1 or ep == a.epochs - 1:
            c = Hd.diff_fit(fn, va, D, dev)["corr_diff"]
            c = -1e9 if c != c else c
            if c > best[0]:
                best = (c, copy.deepcopy(h.state_dict()), ep + 1)
    h.load_state_dict(best[1])
    return h, best[2]


def pick_alpha(fn, va, D, dev, alphas):
    rows = [(al, Hd.evaluate(va, D, fn, al, dev)["gain_per_root"]) for al in alphas]
    return max(rows, key=lambda x: x[1])[0], rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", required=True, nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fracs", default="1/32,1/16,1/8,1/4,1/2,1")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--eval-every", type=int, default=3)
    ap.add_argument("--bs", type=int, default=256); ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--l2", type=float, default=0.01)
    a = ap.parse_args()
    dev = a.device
    t0 = time.time()
    D = load_roots(a.roots)
    D["Zd"], D["BLd"] = D["Z"].to(dev), D["BL"].to(dev).float()
    D["Z"], D["BL"] = D["Zd"], D["BLd"]      # heads.evaluate / diff_fit 按根切片再 .to(dev)：放在显卡上切片免得逐根搬运
    R = Hd.root_tables(D)
    te = [t for t in R if t["id"] % 5 == 0]
    va = [t for t in R if t["id"] % 5 == 1]
    pool = [t for t in R if t["id"] % 5 >= 2]
    ids = sorted({t["id"] for t in pool})
    rank = {i: r for r, i in enumerate(np.random.default_rng(12345).permutation(ids))}
    tg = lambda S: round(sum(t["gain"].get(t["best"], 0.0) for t in S) / max(1, len(S)), 5)
    rep = {"roots_files": a.roots, "n_roots": len(R), "n_test": len(te), "n_val": len(va), "n_pool": len(pool), "items_pool": len(ids),
           "test_items": len({t["id"] for t in te}), "teacher_changed_test": round(float(np.mean([t["best"] != t["greedy"] for t in te])), 4),
           "teacher_gain_test": tg(te), "reliability_test": Hd.reliability(te), "S0_test": Hd.evaluate(te, D, None, 0.0, dev), "curve": []}
    print(json.dumps({k: v for k, v in rep.items() if k != "curve"}, ensure_ascii=False), flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for fs in a.fracs.split(","):
        f = float(Fraction(fs))
        m = max(1, int(round(len(ids) * f)))
        tr = [t for t in pool if rank[t["id"]] < m]
        rows = [(t["a0"], [t["greedy"]] + t["cand"], t["best"]) for t in tr if t["cand"]]
        pairs = [(t["a0"] + c, t["a0"] + t["greedy"], y, n) for t in tr for c, y, n in zip(t["cand"], t["dy"], t["nc"])]
        for seed in range(a.seeds):
            row = {"frac": fs, "items": m, "roots": len(tr), "pairs": len(pairs), "seed": seed}
            h1, e1 = train_s1(rows, va, D, dev, a, seed)
            fn1 = lambda z: h1(z)[:, 0]
            al1, sw1 = pick_alpha(fn1, va, D, dev, (0.25, 0.5, 1.0, 2.0))
            row["S1"] = {"epochs": e1, "alpha": al1, "fit_test": Hd.diff_fit(fn1, te, D, dev), "test": Hd.evaluate(te, D, fn1, al1, dev),
                         "val": Hd.evaluate(va, D, fn1, al1, dev)}
            h2, e2 = train_s2(pairs, va, D, dev, a, seed)
            fn2 = lambda z: h2(z)[:, 0]
            al2, sw2 = pick_alpha(fn2, va, D, dev, (0.5, 1.0, 2.0, 4.0, 8.0, 16.0))
            row["S2"] = {"epochs": e2, "alpha": al2, "fit_test": Hd.diff_fit(fn2, te, D, dev), "test": Hd.evaluate(te, D, fn2, al2, dev),
                         "val": Hd.evaluate(va, D, fn2, al2, dev)}
            if fs == a.fracs.split(",")[-1] and seed == 0:
                h1.save(f"{a.out}/s1_full.pt", what="S1 规模曲线最大档", scale=al1)
                h2.alpha = al2; h2.save(f"{a.out}/s2_full.pt", what="S2 规模曲线最大档")
            rep["curve"].append(row)
            print(f"f={fs}（{m} 题、{len(tr)} 根）种子 {seed}：S1 相关 {row['S1']['fit_test']['corr_diff']} 收益/根 {row['S1']['test']['gain_per_root']} "
                  f"命中 {row['S1']['test']['hit_changed']} 保持 {row['S1']['test']['keep_greedy']}｜S2 相关 {row['S2']['fit_test']['corr_diff']} "
                  f"收益/根 {row['S2']['test']['gain_per_root']} 命中 {row['S2']['test']['hit_changed']}（{time.time() - t0:.0f} 秒）", flush=True)
            Path(f"{a.out}/curve.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print("完", flush=True)


if __name__ == "__main__":
    main()
