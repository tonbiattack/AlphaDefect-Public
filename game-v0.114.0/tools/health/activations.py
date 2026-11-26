"""训练健康诊断（后续计划第九节、决策记录 219）：各检查点在同一批真实观测上前向，看可塑性与价值头的几个指标（本机 CPU）。

  - 休眠单元：每层前馈的隐藏单元（ReLU 之后）在全部记录上的平均激活，不到本层平均的 τ（0.1，ReDo 的定义）算休眠，恒为 0 算死
  - 有效秩：编码后归一化（norm）的记录表示，奇异值累计到 99% 所需的个数（srank，Kumar 等）
  - 价值头与结局：价值输出与这一局最终通关与否的 AUC，按幕、战斗内外分开
观测来自 tools/health/capture_obs.py（现模型自己打的整局）；格式 18 之前训练的检查点也喂格式 18 的观测（选牌界面多出的记录它没见过），
比较趋势时注意这一点。

  .venv/bin/python -I tools/health/activations.py --obs 观测.pkl --ckpts 检查点1.pt 检查点2.pt ... --out 结果.json [--n 3000]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))


def auc(pos, neg):
    if not pos or not neg:
        return None
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    rank, s = 0, 0.0
    i = 0
    while i < len(allv):
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]:
            j += 1
        r = (i + j + 1) / 2
        s += r * sum(1 for k in range(i, j) if allv[k][1])
        i = j
    return round((s - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)), 4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", required=True); ap.add_argument("--ckpts", nargs="+", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=3000); ap.add_argument("--tau", type=float, default=0.1); ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()
    import pickle
    import numpy as np
    import torch
    from sts2rl.actor import MODE_RUN
    from sts2rl.features import Ragged, collate_ragged
    from sts2rl.model import ModelConfig, PolicyValueNet, load_state
    torch.set_num_threads(8)
    data = pickle.load(open(a.obs, "rb"))
    recs = data["recs"]
    random.Random(0).shuffle(recs)
    recs = recs[: a.n]
    out = {"obs_version": data.get("version"), "n_obs": len(recs), "ckpts": []}
    for path in a.ckpts:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        mc = dict(blob["config"]["model"])
        if "draw_obs" in mc:            # 格式 16 上线前的名字（决策记录 205 改名为 obs_v16）
            mc["obs_v16"] = mc.pop("draw_obs")
        net = PolicyValueNet(ModelConfig(**{k: v for k, v in mc.items() if k in ModelConfig.__dataclass_fields__})).eval()
        load_state(net, blob["state"])
        L = len(net.encoder.layers)
        sums = [None] * L; cnt = [0] * L; mx = [None] * L
        reps = []

        def ff_hook(k):
            def h(_m, _i, o):
                r = torch.relu(o.detach().float().reshape(-1, o.shape[-1]))
                sums[k] = r.sum(0) if sums[k] is None else sums[k] + r.sum(0)
                mx[k] = r.amax(0) if mx[k] is None else torch.maximum(mx[k], r.amax(0))
                cnt[k] += r.shape[0]
            return h
        hs = [net.encoder.layers[k].linear1.register_forward_hook(ff_hook(k)) for k in range(L)]
        hs.append(net.norm.register_forward_hook(lambda _m, _i, o: reps.append(o.detach().float().reshape(-1, o.shape[-1])[::7])))
        vals = []
        with torch.no_grad():
            for j in range(0, len(recs), a.batch):
                b = recs[j: j + a.batch]
                _, v = net(collate_ragged(Ragged.from_obs([r["obs"] for r in b], np.full(len(b), MODE_RUN))).to("cpu"))
                vals += [float(x) for x in v.float()]
        for h in hs:
            h.remove()
        layers = []
        for k in range(L):
            m = sums[k] / max(1, cnt[k])
            layers.append({"dormant": round(float((m <= a.tau * m.mean()).float().mean()), 4), "dead": round(float((mx[k] <= 0).float().mean()), 4),
                           "mean_act": round(float(m.mean()), 4)})
        R = torch.cat(reps)[:20000]
        sv = torch.linalg.svdvals(R - R.mean(0))
        cs = torch.cumsum(sv, 0) / sv.sum()
        srank = int((cs < 0.99).sum()) + 1
        groups = {}
        for r, v in zip(recs, vals):
            key = f"幕{r['act'] + 1}{'战斗' if r['in_combat'] else '局外'}"
            groups.setdefault(key, ([], []))[0 if r.get("win") else 1].append(v)
        ver = blob.get("version")
        row = {"version": ver, "layers": layers, "srank": srank, "auc": {k: auc(p, n) for k, (p, n) in sorted(groups.items())},
               "value_mean": round(float(np.mean(vals)), 4)}
        out["ckpts"].append(row)
        print(f"第 {ver} 版：休眠 {[l['dormant'] for l in layers]}，死 {[l['dead'] for l in layers]}，有效秩 {srank}，AUC {row['auc']}", flush=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
