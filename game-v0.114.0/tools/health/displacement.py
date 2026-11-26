"""训练健康诊断（决策记录 220–221）：两个检查点之间各组参数的“净位移”与这段时间“逐版更新量之和”的比值。

净位移 = ‖θ末 − θ初‖ / ‖θ初‖；逐版更新量之和 = Σ 每版 health.upd（log.jsonl，每版更新量占权重的比例）。
两者之比是更新的“方向一致程度”：每版方向完全一致时接近 1，纯随机游走时约 1/√版数（1,000 版约 0.03）。

  python3 tools/health/displacement.py 初.pt 末.pt log.jsonl
"""
from __future__ import annotations

import json
import math
import sys


def group(name: str) -> str:
    if name.startswith("encoder.layers."):
        return "L" + name.split(".")[2]
    return next((h for h in ("cand_mlp_m", "cand_mlp", "value_mlp_m", "value_mlp", "id_emb", "aux_emb", "cand_id_emb") if name.startswith(h + ".")), "other")


def main() -> None:
    import torch
    a, b, logp = sys.argv[1:4]
    A = torch.load(a, map_location="cpu", weights_only=False); B = torch.load(b, map_location="cpu", weights_only=False)
    va, vb = int(A["version"]), int(B["version"])
    sa, sb = A["state"], B["state"]
    num, den = {}, {}
    for k, t in sa.items():
        if not torch.is_floating_point(t) or k not in sb or sb[k].shape != t.shape:
            continue
        g = group(k)
        num[g] = num.get(g, 0.0) + float(((sb[k].float() - t.float()) ** 2).sum()); den[g] = den.get(g, 0.0) + float((t.float() ** 2).sum())
    path, n = {}, 0
    for line in open(logp, encoding="utf-8"):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if va < r.get("version", 0) <= vb and r.get("health"):
            n += 1
            for g, x in r["health"]["upd"].items():
                path[g] = path.get(g, 0.0) + x
    print(f"第 {va} → {vb} 版（{vb - va} 版，其中 {n} 版有逐版统计）；纯随机游走时比值约 {1 / math.sqrt(max(1, n)):.3f}")
    out = {}
    for g in sorted(num):
        disp = math.sqrt(num[g]) / max(1e-12, math.sqrt(den[g]))
        p = path.get(g)
        out[g] = {"disp": round(disp, 5), "path": round(p, 5) if p else None, "ratio": round(disp / p, 4) if p else None}
        print(f"  {g:12s} 净位移 {disp:.4f}  逐版之和 {p if p is None else round(p, 4)}  比值 {out[g]['ratio']}")
    print(json.dumps({"from": va, "to": vb, "n_logged": n, "groups": out}, ensure_ascii=False))


if __name__ == "__main__":
    main()
