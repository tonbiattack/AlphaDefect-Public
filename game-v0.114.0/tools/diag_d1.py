"""平台期诊断 D1/D3 分析（2026-11-03）：把诊断各组（/root/diag/<名字>/）的逐种子胜负，和 P1 上 15 个定时复测快照的贪心结果比较。
    python3 tools/diag_d1.py 组结果.json
组结果.json：{名字: {种子: 0/1}}（远端用一行 python 汇总后拉回，免得传整份逐局记录）。
输出：每组胜率、相对 v149402 贪心的翻面率，K 组并集 vs K 个快照并集，逐种子胜率的相关。"""
import itertools
import json
import os
import random
import sys
from pathlib import Path

AT = Path(os.path.expanduser("~/.local/share/sls2/ability_test"))
RT = ["v141229-计划外", "v142196", "v143133", "v144081", "v144507", "v145449", "v146055", "v146509", "v146963", "v147404",
      "v147859", "v148300", "v148538", "v148980", "v149402"]


def load_dir(d: Path) -> dict:
    out = {}
    for f in d.glob("runs_s*.jsonl"):      # runs_seval.jsonl：复测沿用的同版本评测（P1 里的 384 个评测种子）
        for l in open(f, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                out[r["seed"]] = int(r["win"])
    return out


def union_curve(groups: list, seeds, kmax, trials=200):
    rng = random.Random(0)
    res = {}
    for k in range(1, min(kmax, len(groups)) + 1):
        tot = 0.0
        combos = list(itertools.combinations(range(len(groups)), k))
        pick = combos if len(combos) <= trials else rng.sample(combos, trials)
        for c in pick:
            tot += sum(any(groups[i][s] for i in c) for s in seeds) / len(seeds)
        res[k] = tot / len(pick)
    return res


def main():
    G = json.load(open(sys.argv[1], encoding="utf-8"))
    base = {v: load_dir(AT / v) for v in RT}
    seeds = sorted(set.intersection(*(set(b) for b in base.values())))
    ref = base["v149402"]
    print(f"种子 {len(seeds)} 个；v149402 贪心 {sum(ref[s] for s in seeds) / len(seeds):.1%}")
    flip_adj = [sum(base[RT[i]][s] != base[RT[i + 1]][s] for s in seeds) / len(seeds) for i in range(len(RT) - 1)]
    print(f"相邻定时快照（约 450 版）翻面率 平均 {sum(flip_adj) / len(flip_adj):.1%}（{min(flip_adj):.1%}–{max(flip_adj):.1%}）")
    fams = {}
    for name, d in sorted(G.items()):
        d = {s: int(v) for s, v in d.items()}
        ok = all(s in d for s in seeds)
        wr = sum(d.get(s, 0) for s in seeds) / len(seeds)
        fl = sum(d.get(s, 0) != ref[s] for s in seeds) / len(seeds)
        print(f"{name:14s} {'' if ok else '（未打完）'}胜率 {wr:.1%}  与 v149402 贪心翻面 {fl:.1%}")
        if ok:
            fams.setdefault(name.rsplit("_", 1)[0], []).append(d)
    ck = [base[v] for v in RT]
    cu = union_curve(ck, seeds, 8)
    print("K 个定时快照（贪心）并集：", {k: f"{v:.1%}" for k, v in cu.items()})
    for fam, gs in fams.items():
        if len(gs) >= 2:
            print(f"K 组 {fam} 并集：", {k: f"{v:.1%}" for k, v in union_curve(gs, seeds, 8).items()})
    # 逐种子胜率：快照间（15 个） vs 同一快照采样
    pc = {s: sum(b[s] for b in ck) / len(ck) for s in seeds}
    for fam, gs in fams.items():
        if len(gs) >= 3:
            ps = {s: sum(g[s] for g in gs) / len(gs) for s in seeds}
            mx, my = sum(pc.values()) / len(seeds), sum(ps.values()) / len(seeds)
            cov = sum((pc[s] - mx) * (ps[s] - my) for s in seeds)
            vx = sum((pc[s] - mx) ** 2 for s in seeds); vy = sum((ps[s] - my) ** 2 for s in seeds)
            print(f"逐种子胜率相关（15 个快照 vs {fam} {len(gs)} 组）：{cov / (vx * vy) ** .5:.3f}")


if __name__ == "__main__":
    main()
