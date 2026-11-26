"""集成的决策一致率扫描（决策记录 272）：在一批真实局面上，比较不同集成方法、快照数、选法给出的首选，与“全部快照”的首选有多大比例相同。
不打整局，几分钟就能扫完全部组合；一致率很高的两种组合，整局结果也几乎一样，不必再花时间打局区分。

每个快照只前向一次，各候选的对数概率存下来，再离线组合：
  方法 prob 概率平均、logit 对数概率平均、vote 多数投票（同票按平均概率）、median 概率中位数、trim 去掉最高最低各 20% 后平均（同 tools/ability_test.py 的 EnsembleNet）；
  快照数 K；选法 recent 最近 K 个、spread 按版本均匀间隔取 K 个。
只看候选不少于 2 个的局面，战斗与局外分开报。

    .venv/bin/python -I tools/ens_agree.py --dump 局面.pkl [--dump ...] --list 快照列表.txt --n 16384 --out 结果.json
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))


def combine(lp: np.ndarray, method: str) -> np.ndarray:
    """lp：[K, N, A] 对数概率（不合法为 -inf）。返回 [N] 首选。"""
    valid = np.isfinite(lp[0])
    p = np.exp(lp)
    if method == "prob":
        s = p.mean(0)
    elif method == "logit":
        s = np.where(valid, np.where(valid[None], lp, 0.0).mean(0), -np.inf)
    elif method == "vote":
        top = lp.argmax(-1)                                   # [K, N]
        cnt = np.zeros(p.shape[1:], np.float64)
        np.add.at(cnt, (np.arange(p.shape[1])[None].repeat(len(top), 0), top), 1.0)
        s = cnt + 1e-3 * p.mean(0)
    elif method == "median":
        s = np.median(p, 0) + 1e-6 * p.mean(0)
    elif method == "trim":
        k = len(lp); cut = int(k * 0.2)
        s = np.sort(p, 0)[cut:k - cut].mean(0) if k - 2 * cut > 0 else p.mean(0)
    else:
        raise ValueError(method)
    return np.where(valid, s, -np.inf).argmax(-1)


def pick(n_total: int, k: int, how: str) -> list[int]:
    if k >= n_total:
        return list(range(n_total))
    if how == "recent":
        return list(range(n_total - k, n_total))
    return sorted({int(round(x)) for x in np.linspace(0, n_total - 1, k)})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", action="append", required=True)
    ap.add_argument("--list", required=True, help="快照路径，一行一个，按版本从旧到新")
    ap.add_argument("--n", type=int, default=16384)
    ap.add_argument("--mb", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    import torch
    from sts2rl.features import Ragged, collate_ragged
    from sts2rl.model import ModelConfig, PolicyValueNet, load_state
    from sts2env.protocol import DecisionKind
    torch.set_num_threads(4)
    parts = [pickle.load(open(d, "rb")) for d in a.dump]
    r = Ragged.concat(parts) if len(parts) > 1 else parts[0]
    multi = np.flatnonzero(np.diff(r.cand_off) > 1)
    rows = np.sort(np.random.default_rng(0).choice(multi, min(a.n, len(multi)), replace=False))
    data = r.select(rows)
    ncand = np.diff(data.cand_off).astype(np.int64)
    W = int(ncand.max())
    combat = (data.header[:, 11] != 0) | (data.header[:, 1] == int(DecisionKind.COMBAT))
    paths = [l.strip() for l in open(a.list) if l.strip()]
    dev = torch.device(a.device)
    lps, vers = [], []
    for path in paths:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        mc = blob["config"]["model"]
        net = PolicyValueNet(ModelConfig(**mc)).to(dev).eval(); load_state(net, blob["state"])
        out = np.full((len(data), W), -np.inf, np.float32)
        with torch.no_grad():
            for j in range(0, len(data), a.mb):
                idx = np.arange(j, min(j + a.mb, len(data)))
                b = collate_ragged(data.select(idx), attend=bool(mc.get("macro_attend"))).to(dev)
                lg = torch.log_softmax(net(b)[0].float(), -1).cpu().numpy()
                out[idx, :lg.shape[1]] = lg[:, :W]
        lps.append(out); vers.append(int(blob["version"]))
        del net
        print(f"{len(vers)}/{len(paths)} 第 {vers[-1]} 版", flush=True)
    LP = np.stack(lps)                                       # [M, N, W]
    M = len(LP)
    methods = ["prob", "logit", "vote", "median", "trim"]
    ks = [k for k in (1, 2, 4, 8, 16, 32, M) if k <= M]
    refs = {m: combine(LP, m) for m in methods}
    single = LP[-1].argmax(-1)
    res = {"versions": vers, "n": int(len(data)), "n_combat": int(combat.sum()), "n_macro": int((~combat).sum()), "rows": []}
    def agree(x, y, mask):
        return round(float((x[mask] == y[mask]).mean()), 4) if mask.any() else None
    allm = np.ones(len(data), bool)
    print(f"局面 {len(data)}（战斗 {int(combat.sum())}、局外 {int((~combat).sum())}），快照 {M} 个", flush=True)
    print("方法 选法 K | 与全部快照同方法一致（全部/战斗/局外） | 与全部快照中位数一致 | 与单个最新快照一致", flush=True)
    for m in methods:
        for how in ("recent", "spread"):
            for k in ks:
                if how == "spread" and k in (1, M):
                    continue
                d = combine(LP[pick(M, k, how)], m) if k > 1 else LP[-1].argmax(-1)
                row = {"method": m, "how": how, "k": k,
                       "vs_full_same": [agree(d, refs[m], allm), agree(d, refs[m], combat), agree(d, refs[m], ~combat)],
                       "vs_full_median": agree(d, refs["median"], allm), "vs_single": agree(d, single, allm)}
                res["rows"].append(row)
                print(f"{m} {how} {k} | {row['vs_full_same']} | {row['vs_full_median']} | {row['vs_single']}", flush=True)
    res["cross"] = {f"{x}-{y}": agree(refs[x], refs[y], allm) for i, x in enumerate(methods) for y in methods[i + 1:]}
    print("全部快照时各方法之间的一致率：" + json.dumps(res["cross"], ensure_ascii=False), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
