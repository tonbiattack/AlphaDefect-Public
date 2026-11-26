"""常规评测的种子集（2026-10-29 起，决策记录 198）：从千种子复测的 RT0000–RT0999 里按难度分层抽 n 个，让评测的通关率大体代表
千种子复测（≈ 真实胜率）。

原来的常规评测用 EVAL000–EVAL191（局数 384 时到 EVAL383），这组种子没有按难度挑过：EVAL192–383 明显偏难（第 85,782、85,845 版
前 192 个种子通关 73、71 局，后 192 个只有 61、63 局），整组的通关率因此比千种子复测低好几个百分点，而且偏多少随种子组而定。

做法：每个 RT 种子的难度 = 最近几次千种子复测里它被通关的次数（--recent 个版本，0…k）；按难度排序（同难度内随机打乱）后
等间隔取 n 个（系统抽样），抽出的种子集与整池的难度分布一致。只分层、不在历史版本上挑“误差最小”的一组：试过挑选，
在没参与挑选的版本上误差反而更大（过拟合）。--check 用没参与分层的旧版本核对抽出的种子集与整池通关率的差。

    python3 tools/eval_seeds.py --n 384 --out tools/remote/eval_seeds.json
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

AT = Path.home() / ".local/share/sls2/ability_test"


def load(d: Path) -> dict[str, int]:
    w = {}
    for f in d.glob("runs_s*.jsonl"):
        for line in open(f, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                if r["seed"].startswith("RT"):
                    w[r["seed"]] = int(r["win"])
    return w


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=384)
    ap.add_argument("--recent", type=int, default=5, help="按最近几次定时千种子复测算难度")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="tools/remote/eval_seeds.json")
    a = ap.parse_args()
    runs = json.loads((AT / "retest/state.json").read_text(encoding="utf-8"))["runs"]
    dirs = [Path(r["dir"]) for r in runs if r.get("seeds") == 1000]
    W = {d.name: w for d in dirs if len(w := load(d)) == 1000}
    names = list(W)
    recent, older = names[-a.recent:], names[:-a.recent]
    seeds = sorted(W[names[-1]])
    diff = {s: sum(W[v][s] for v in recent) for s in seeds}
    rng = random.Random(a.seed)
    order = sorted(seeds, key=lambda s: (diff[s], rng.random()))
    step = len(order) / a.n
    off = rng.random() * step
    pick = sorted(order[int(off + i * step)] for i in range(a.n))
    rate = lambda v, ss: sum(W[v][s] for s in ss) / len(ss)
    check = {v: round(100 * (rate(v, pick) - rate(v, seeds)), 2) for v in older + recent}
    out = {"seeds": pick, "built_from": recent, "difficulty": {str(k): sum(1 for s in pick if diff[s] == k) for k in range(len(recent) + 1)},
           "check_pp": check, "note": "按最近几次千种子复测的通关次数分层系统抽样（tools/eval_seeds.py，决策记录 198）"}
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    old = [abs(check[v]) for v in older]
    print(f"{a.n} 个种子 → {a.out}；难度分布 {out['difficulty']}")
    print(f"与整池通关率之差（百分点）：分层用的版本 {[check[v] for v in recent]}；没参与分层的旧版本平均 {statistics.mean(old):.2f}、最大 {max(old):.2f}" if old else "")


if __name__ == "__main__":
    main()
