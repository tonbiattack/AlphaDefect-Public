"""战斗网络试点：把树搜索的结论蒸馏进一份 L1000 的副本（2026-11-17）。

用户：“我们考虑结合搜索另练一个单独的战斗网络……局外决策用主模型，局内决策参考战斗网络并使用搜索工具来增强我们的胜率。”
为什么现在重试：以前用老师（PIMC）的标签蒸馏学不进网络，根因是标签噪声（候选差复测可靠性 0.43）；树搜索（32 次模拟）同类读数 0.888
（tools/future/label_reliability.py，reports/树搜索试验记录.md 第六节）。

数据：tree_search.py --labels 在非留出题（id % 4 != 0）上记下的每个搜索根：根帧、网络概率 p、候选与各自的 Q、模拟次数。
目标（与搜索的最终选法同一个改进算子，KL 正则化）：模拟次数 ≥ --min-visits 的候选组成集合 C（同一语义键的候选算一组），
  组 k 的新质量 ∝ P_k · exp(Q_k / λ)，C 的总质量保持不变，组内按原概率分；C 以外的动作保持原概率。λ 默认 0.03（与搜索相同）。
损失：交叉熵 H(π′, π_θ)（战斗模式、去掉特权特征，同搜索与实机贪心的输入）+ --v-coef ×（价值头 − 原价值头读数）²（拴住价值头，搜索的叶子还要用它）。
按题号留出 id % 5 == 0 的题做验证：与搜索最终选法一致的比例、相对原贪心改动的比例、改选命中率。
存成与 L1000 同格式的检查点（state 换成微调后的权重），可直接给 tree_search.py / boss_pimc.py 的 --ckpt。

  .venv/bin/python -I tools/future/tree_distill.py --labels '/root/i2a/tree/lab_m*/labels_s*.jsonl' --ckpt L1000.pt --out 目录
"""
from __future__ import annotations

import argparse
import base64
import glob
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def load_roots(pattern: str, min_visits: int, lam: float):
    """每个搜索根 → (题号, 原始帧字节, 目标分布 π′, 原价值, 搜索选的下标, 原贪心下标)。没有至少两个合格候选的根跳过。"""
    from sts2env.protocol import parse
    import sim_pimc as X
    roots, skipped = [], 0
    files = sorted(f for pat in pattern.split(",") for f in glob.glob(pat))      # 逗号分隔的几个通配
    for f in files:
        for l in open(f, encoding="utf-8"):
            if not l.strip():
                continue
            r = json.loads(l)
            raw = base64.b64decode(r["frame"])
            o = parse(raw)
            n = int(o.num_actions)
            p = np.asarray(r["p"], np.float64)
            if len(p) != n:
                skipped += 1
                continue
            keys = [k[:-1] for k in X.cand_keys(o)]
            q = {int(c): (float(v[0]), int(v[1])) for c, v in (r.get("q") or {}).items()}
            C = [c for c in q if q[c][1] >= min_visits]
            # 贪心项可能是同名牌里下标较大的那张（候选按键合并后取下标最小的），按语义键比
            if len(C) < 2 or keys[int(r["greedy"])] not in {keys[c] for c in C}:
                skipped += 1
                continue
            groups = {}
            for i, k in enumerate(keys):
                groups.setdefault(k, []).append(i)
            gk = [keys[c] for c in C]
            Pk = np.array([p[groups[k]].sum() for k in gk])
            s = np.log(np.maximum(Pk, 1e-12)) + np.array([q[c][0] for c in C]) / lam
            w = np.exp(s - s.max()); w /= w.sum()
            tgt = p.copy()
            M = Pk.sum()
            for k, Pk_, wk in zip(gk, Pk, w):
                idx = groups[k]
                tgt[idx] = M * wk * (p[idx] / max(Pk_, 1e-12))
            tgt /= tgt.sum()
            roots.append({"id": int(r["id"]), "raw": raw, "tgt": tgt.astype(np.float32), "v0": float(r.get("v0", 0.0)),
                          "best": int(r["best"]), "greedy": int(r["greedy"]), "keys": keys})
    return roots, skipped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True); ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--epochs", type=int, default=3); ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-5); ap.add_argument("--lam", type=float, default=0.03); ap.add_argument("--min-visits", type=int, default=4)
    ap.add_argument("--v-coef", type=float, default=1.0); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bf16", action="store_true", help="训练与验证的前向用 bf16 自动混合精度（默认 fp32）")
    a = ap.parse_args()
    from sts2env.protocol import parse
    from sts2rl.features import Ragged, collate_ragged
    from sts2rl.model import ModelConfig, PolicyValueNet, load_state
    import sim_pimc as X
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = a.device
    blob = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    net = PolicyValueNet(ModelConfig(**blob["config"]["model"])).to(dev)
    load_state(net, blob["state"])
    t0 = time.time()
    roots, skipped = load_roots(a.labels, a.min_visits, a.lam)
    tr = [r for r in roots if r["id"] % 5 != 0]
    te = [r for r in roots if r["id"] % 5 == 0]
    print(f"{len(roots)} 个根（跳过 {skipped}），训练 {len(tr)}、验证 {len(te)}，读入 {time.time() - t0:.0f} 秒", flush=True)

    def fwd(rs, grad: bool):
        fs = [X._no_priv(parse(r["raw"])) for r in rs]
        b = collate_ragged(Ragged.from_obs(fs, np.zeros(len(fs), np.int64))).to(dev)
        # fp32（同搜索与实机贪心的前向）：bf16 会让起点的首选与原贪心有约 3% 不同（几乎打平的两项翻转），验证读数不干净
        with torch.set_grad_enabled(grad), torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.bf16):
            logits, values = net(b)
        return logits.float(), values.float().reshape(-1)

    @torch.no_grad()
    def evaluate(S) -> dict:
        net.eval()
        agree = moved = hit = nchg = keep = nkeep = 0
        ce = 0.0
        for i in range(0, len(S), 256):
            rs = S[i:i + 256]
            lg, _ = fwd(rs, False)
            for j, r in enumerate(rs):
                n = len(r["tgt"])
                lp = torch.log_softmax(lg[j, :n], -1).cpu().numpy()
                ce += float(-(r["tgt"] * lp).sum())
                pk = int(np.argmax(lp))
                # 同一语义键的候选算同一个动作
                same = lambda x, y: r["keys"][x] == r["keys"][y]
                agree += int(same(pk, r["best"])); moved += int(not same(pk, r["greedy"]))
                if r["best"] != r["greedy"]:
                    hit += int(same(pk, r["best"])); nchg += 1
                else:
                    keep += int(same(pk, r["greedy"])); nkeep += 1
        net.train()
        return {"ce": round(ce / max(1, len(S)), 4), "agree": round(agree / max(1, len(S)), 4), "moved": round(moved / max(1, len(S)), 4),
                "hit_changed": round(hit / max(1, nchg), 4), "keep": round(keep / max(1, nkeep), 4), "n_changed": nchg}

    print("起点 验证", evaluate(te), flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.0)
    net.train()
    for ep in range(a.epochs):
        perm = np.random.permutation(len(tr))
        tot = [0.0, 0.0, 0]
        for s in range(0, len(perm), a.bs):
            rs = [tr[i] for i in perm[s:s + a.bs]]
            lg, v = fwd(rs, True)
            l_pi = torch.zeros((), device=dev)
            for j, r in enumerate(rs):
                n = len(r["tgt"])
                l_pi = l_pi - (torch.from_numpy(r["tgt"]).to(dev) * torch.log_softmax(lg[j, :n], -1)).sum()
            l_pi = l_pi / len(rs)
            v0 = torch.tensor([r["v0"] for r in rs], device=dev)
            l_v = ((v - v0) ** 2).mean()
            loss = l_pi + a.v_coef * l_v
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot[0] += float(l_pi.detach()); tot[1] += float(l_v.detach()); tot[2] += 1
        print(f"第 {ep + 1} 遍：策略交叉熵 {tot[0] / tot[2]:.4f}，价值偏移 {tot[1] / tot[2]:.5f}，{time.time() - t0:.0f} 秒；验证 {evaluate(te)}", flush=True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    blob2 = {k: v for k, v in blob.items() if k != "state"}
    blob2["state"] = {k: v.detach().cpu() for k, v in net.state_dict().items()}
    blob2["tree_distill"] = {"from": a.ckpt, "labels": a.labels, "epochs": a.epochs, "lr": a.lr, "lam": a.lam, "min_visits": a.min_visits,
                             "v_coef": a.v_coef, "n_train": len(tr), "n_val": len(te)}
    torch.save(blob2, out / "distill.pt")
    (out / "report.json").write_text(json.dumps({"val": evaluate(te), "args": vars(a)}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("存", out / "distill.pt", flush=True)


if __name__ == "__main__":
    main()
