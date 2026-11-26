"""胜率模型核对：在特征缓存（train.py --cache-dir 写的 *.pkl）上比较几个模型（决策记录 254）。
每个头给对数损失、Brier、AUC、平均预测与实际（每局每层等权），按幕的平均预测与实际，校准十档（预测区间、平均预测、实际、步数）。

    python tools/winprob/check.py --cache 缓存目录 [--split test|all] 模型1.pt 模型2.pt …
--split test 只看按种子留出的测试种子（bucket < 0.2，同 train.py / recal.py），all 用全部（拿别的模型、别的种子打的局核对时用）。"""
from __future__ import annotations

import argparse
import glob
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import bucket, calib, concat, scores  # noqa: E402
from winprob import HEADS, N_SCALAR, WinProb, select, to_torch  # noqa: E402


def load_net(path: str) -> WinProb:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    d = ck["args"].get("d", 16)
    net = WinProb(**{**ck["args"], "amounts": ck["model"]["mlp.0.weight"].shape[1] == 9 * d + N_SCALAR}).eval()
    net.load_state_dict(ck["model"])
    return net


@torch.no_grad()
def probs(net, D: dict, bs: int = 65536) -> np.ndarray:
    out = [torch.sigmoid(net(to_torch(select(D, np.arange(i, min(i + bs, D["n"]))), "cpu"))).numpy() for i in range(0, D["n"], bs)]
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--split", choices=("test", "all"), default="test")
    ap.add_argument("models", nargs="+")
    a = ap.parse_args()
    torch.set_num_threads(2)
    d = concat([pickle.load(open(f, "rb")) for f in sorted(glob.glob(f"{a.cache}/*.pkl"))])
    seeds = d.pop("seed"); n_games = d.pop("games")
    if a.split == "test":
        useed, inv = np.unique(seeds, return_inverse=True)
        d = select(d, np.flatnonzero(np.asarray([bucket(str(x)) for x in useed])[inv] < 0.2))
    print(f"{a.cache}（{a.split}）：{d['n']} 步、{len(set(d['game']))} 局（缓存共 {n_games} 局）", flush=True)
    for path in a.models:
        p = probs(load_net(path), d)
        print(f"== {Path(path).name}")
        for h, head in enumerate(HEADS):
            m = d["y"][:, h] >= 0
            y, w, ph, act = d["y"][m, h].astype(float), d["w"][m], p[m, h], d["act"][m]
            s = scores(ph, y, w)
            print(f"  {head}：对数损失 {s['logloss']}  Brier {s['brier']}  AUC {s['auc']}  平均预测 {np.average(ph, weights=w):.3f}  实际 {s['rate']}")
            print("    按幕（预测/实际）：" + "  ".join(f"第{x + 1}幕 {np.average(ph[act == x], weights=w[act == x]):.3f}/{np.average(y[act == x], weights=w[act == x]):.3f}"
                                             for x in range(3) if (act == x).any()))
            print("    校准：", [tuple(c) for c in calib(ph, y, w)])


if __name__ == "__main__":
    main()
