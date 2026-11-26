"""学习端（B）上运行：用 snap_drift.py 缓存的各快照首选（~/.local/share/sls2/runs/defect/snap_drift/v*.npy），
挑出按版本均匀间隔的至多 K 份，算两两首选不同比例的平均，判断够不够做下一轮“集成 → 蒸馏”（决策记录 275）。不跑网络。

    python3 tools/remote/snap_pairs.py --since 165163 [--k 16]
输出一行 JSON：可用快照数、挑出的版本、两两平均差异、最小与最大差异。
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

CACHE = Path.home() / ".local/share/sls2/runs/defect/snap_drift"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", type=int, required=True, help="只用这一版及以后的快照（不含锚定放松前的）")
    ap.add_argument("--k", type=int, default=16)
    a = ap.parse_args()
    vs = sorted(int(m.group(1)) for f in CACHE.glob("v*.npy") if (m := re.match(r"v(\d+)\.npy$", f.name)) and int(m.group(1)) >= a.since)
    if len(vs) < 2:
        print(json.dumps({"n": len(vs), "versions": vs}, ensure_ascii=False)); return
    idx = sorted({int(round(x)) for x in np.linspace(0, len(vs) - 1, min(a.k, len(vs)))})
    pick = [vs[i] for i in idx]
    X = np.stack([np.load(CACHE / f"v{v}.npy") for v in pick])
    D = (X[:, None, :] != X[None, :, :]).mean(-1)
    iu = np.triu_indices(len(pick), 1)
    print(json.dumps({"n": len(vs), "picked": pick, "mean_pair": round(float(D[iu].mean()), 4),
                      "min_pair": round(float(D[iu].min()), 4), "max_pair": round(float(D[iu].max()), 4)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
