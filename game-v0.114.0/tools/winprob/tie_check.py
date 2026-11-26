"""胜率小模型在最后一个 Boss 前后三个数的检验（储君决策记录第 30 条）：在按种子留出的测试局上逐步跑 Estimator（不统一），
看 FinalStage 为 1（下一个 Boss 是最后一个，打过它与通关是同一件事）、为 2（正在打最后一个 Boss，三件事相同）的步上，
三个头原本差多少，以及统一时取哪个值（平均或某一个头）对数损失最低。也报这些局整体的通关校准（平均预测对实际）。

    python tools/winprob/tie_check.py 模型.pt 回放目录 [--test-frac 0.2]
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import bucket  # noqa: E402
from winprob import Estimator, labels, read_replay  # noqa: E402


def ll(p: float, y: int) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return -math.log(p if y else 1 - p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("data")
    ap.add_argument("--test-frac", type=float, default=0.2)
    a = ap.parse_args()
    est = Estimator(a.model)
    est.tie = False                       # 看原始的三个头
    s1 = {"diff": [], "ll": {"boss": [], "win": [], "mean": []}}
    s2 = {"diff": [], "ll": {"combat": [], "boss": [], "win": [], "mean": []}}
    first, wins, games = [], 0, 0
    for f in sorted(Path(a.data).rglob("*.jsonl.gz")):
        r = read_replay(f)
        if r is None:
            continue
        head, steps = r
        if bucket(str(head.get("seed"))) >= a.test_frac:
            continue
        games += 1; wins += bool(head.get("win"))
        y = labels(head, steps)
        est.deck = []; est.stage.reset()
        for t, s in enumerate(steps):
            p = est(s); st = est.stage.done3 >= 1 and s["g"][0] == 2
            if t == 0:
                first.append(p["win"])
            if not st:
                continue
            yw = int(y[t, 2])
            in_boss = s["g"][0] == 2 and s["g"][8] and s["g"][9] == 3
            if in_boss and p["combat"] is not None:
                vals = [p["combat"], p["boss"], p["win"]]
                s2["diff"].append(max(vals) - min(vals))
                for k, v in (("combat", p["combat"]), ("boss", p["boss"]), ("win", p["win"]), ("mean", sum(vals) / 3)):
                    s2["ll"][k].append(ll(v, yw))
            else:
                s1["diff"].append(abs(p["boss"] - p["win"]))
                for k, v in (("boss", p["boss"]), ("win", p["win"]), ("mean", (p["boss"] + p["win"]) / 2)):
                    s1["ll"][k].append(ll(v, yw))
    print(f"测试局 {games} 局，实际通关 {wins / max(1, games):.3f}，开局第一步预测通关平均 {np.mean(first):.3f}")
    for name, s in (("下一个 Boss 是最后一个（战斗外）", s1), ("正在打最后一个 Boss", s2)):
        if not s["diff"]:
            print(f"{name}：没有样本"); continue
        d = np.asarray(s["diff"])
        print(f"{name}：{len(d)} 步，三个头原本最大差 平均 {d.mean():.3f}、中位 {np.median(d):.3f}、九成分位 {np.quantile(d, 0.9):.3f}、最大 {d.max():.3f}；"
              "对数损失 " + "、".join(f"{k} {np.mean(v):.4f}" for k, v in s["ll"].items()))


if __name__ == "__main__":
    main()
