"""未来推演方案初测之二：冻结 L1000 的候选表示上，比较三种“不在部署时推演”的小头（2026-11-16）。

报告第 8 节阶段 2 的缩小版（共用数据、分组与预算）：
  S1 直接蒸馏：残差头 r(z) 加到原 logits 上，交叉熵拟合 PIMC 老师的选择（老师没改选时目标就是原首选）。
  S2 反事实未来预测：头 F(z) 预测“走这个候选、之后贪心 12 步”的读数（推演得分/叶子估值、血量、敌人剩余生命比例、
     12 步内赢、12 步内输），标签来自老师各世界的推演（每个被搜的候选都有）；决策用 原 logits + α·F 的得分维。
  S3 只用真实所选动作的未来标签（DFP 式、不需要推演）：同一组读数，但标签取自同策略采样对局里实际走的那一步之后 12 步的真实轨迹
     （第 12 步帧的价值头估值或提前分出的胜负）；决策同 S2。
评价按题号分组（题号 % 5 == 0 为留出组）：老师改选的根上是否选到老师的候选、老师没改选的根上是否保持原首选、
按老师各世界配对差折算的选择收益（报告：看候选差值与后悔值，不只看总体拟合）。

  feats：  heads.py feats --labels '/root/future/t1/pimc/labels_s*.jsonl' --ckpt L1000.pt --out roots.pt
  ftraj：  heads.py ftraj --dir /root/future/t2/samp --ckpt L1000.pt --out traj.pt
  train：  heads.py train --roots roots.pt [--traj traj.pt] --out 目录
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
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from future_heads import ResidualHead, extract  # noqa: E402

H = 12          # 读数的步数（同 PIMC 的 leaf）
M = 5           # 读数：得分、血量/100、敌人剩余比例、12 步内赢、12 步内输


def load_net(path: str, device: str):
    from sts2rl.model import ModelConfig, PolicyValueNet, load_state
    blob = torch.load(path, map_location="cpu", weights_only=False)
    net = PolicyValueNet(ModelConfig(**blob["config"]["model"])).eval().to(device)
    load_state(net, blob["state"])
    return net


def terminal(win: bool, stage: int, left: float) -> float:
    return 1.5 if win else -1.0 + 0.25 * (stage + 1.0 - left)


def feats(a) -> None:
    from sts2env.protocol import parse
    net = load_net(a.ckpt, a.device)
    roots, frames = [], []
    for f in sorted(glob.glob(a.labels)):
        for l in open(f, encoding="utf-8"):
            if not l.strip():
                continue
            r = json.loads(l)
            if r.get("errors"):
                continue
            o = parse(base64.b64decode(r.pop("frame")))
            # 每个被搜的候选：各世界读数的平均与世界数
            per = {}
            for c, s, sc, hp, left, stage, term, steps in r["meas"]:
                per.setdefault(c, []).append((s, sc, hp / 100.0, left, float(term == 1), float(term == 2)))
            r["per"] = per
            del r["meas"]
            roots.append(r); frames.append(o)
    print(f"{len(roots)} 个根，前向取候选表示…", flush=True)
    t0 = time.time()
    zs, _ = extract(net, frames, a.device)
    Z, off, BL = [], [0], []
    for (z, lg) in zs:
        Z.append(z); BL.append(lg); off.append(off[-1] + len(lg))
    torch.save({"roots": roots, "Z": torch.from_numpy(np.concatenate(Z)), "off": off, "BL": torch.from_numpy(np.concatenate(BL))}, a.out)
    print(f"→ {a.out}（{off[-1]} 个候选，{time.time() - t0:.0f} 秒）", flush=True)


def ftraj(a) -> None:
    """同策略采样对局：每步存了原始帧（frames_s*.bin）与 [偏移, 长度, 动作, 候选数, 血量, 敌人剩余比例, 已赢场数, 价值, 动作概率]。
    标签：第 t 步动作之后第 12 步的读数（对局提前结束的取终局）。只存实际所选候选的 z。"""
    from sts2env.protocol import parse
    net = load_net(a.ckpt, a.device)
    zc, Y, IDS = [], [], []
    for res in sorted(glob.glob(f"{a.dir}/res_s*.jsonl")):
        k = res.split("res_s")[1].split(".")[0]
        fb = open(f"{a.dir}/frames_s{k}.bin", "rb")
        batch_f, batch_meta = [], []

        def flush():
            if not batch_f:
                return
            zs, _ = extract(net, batch_f, a.device)
            for (z, lg), (act, y, pid) in zip(zs, batch_meta):
                zc.append(z[act]); Y.append(y); IDS.append(pid)
            batch_f.clear(); batch_meta.clear()
        for l in open(res, encoding="utf-8"):
            if not l.strip():
                continue
            r = json.loads(l)
            tr = r.get("traj") or []
            if r.get("stuck") or not tr:
                continue
            T = len(tr)
            for t, (off, ln, act, n, hp, left, stage, v, pa) in enumerate(tr):
                if n <= 1:
                    continue
                j = t + H
                if j < T:
                    _, _, _, _, hp2, left2, st2, v2, _ = tr[j]
                    y = [v2 - v, (hp2 - hp) / 100.0, left2 - left, 0.0, 0.0]
                else:   # 12 步内结束
                    win = bool(r["win"])
                    st_end = tr[-1][6]
                    left_end = 0.0 if win else tr[-1][5]
                    y = [terminal(win, st_end, left_end) - v, ((r.get("hp") or 0) - hp) / 100.0, left_end - left, float(win), float(not win)]
                fb.seek(off)
                batch_f.append(parse(fb.read(ln))); batch_meta.append((act, y, r["id"]))
                if len(batch_f) >= 2048:
                    flush()
        flush()
        print(f"{res}：累计 {len(Y)} 步", flush=True)
    torch.save({"Z": torch.from_numpy(np.stack(zc)), "Y": torch.tensor(Y, dtype=torch.float32), "ids": torch.tensor(IDS)}, a.out)
    print(f"→ {a.out}", flush=True)


# ---------------------------------------------------------------- 训练与评价

def root_tables(D: dict):
    """每个根：候选下标范围、被搜候选、老师对原首选的配对差（均值、标准误、世界数）、分半估计（可靠性诊断）、各读数的配对差。"""
    out = []
    for i, r in enumerate(D["roots"]):
        a0, a1 = D["off"][i], D["off"][i + 1]
        g = r["greedy"]
        per = {int(c): v for c, v in r["per"].items()}
        if g not in per:
            continue
        gw = {x[0]: np.array(x[1:], np.float32) for x in per[g]}      # 种子 → 读数 [得分, 血量, 敌人剩余, 赢, 输]
        cand, gain, se, nc, half, dy = [], [], [], [], [], []
        for c, v in per.items():
            if c == g:
                continue
            d = np.array([np.array(x[1:], np.float32) - gw[x[0]] for x in v if x[0] in gw])
            if len(d) < 2:
                continue
            cand.append(c); nc.append(len(d)); gain.append(float(d[:, 0].mean())); se.append(float(d[:, 0].std(ddof=1) / math.sqrt(len(d))))
            dy.append(d.mean(0))
            half.append((float(d[0::2, 0].mean()), float(d[1::2, 0].mean())) if len(d) >= 4 else None)
        out.append({"i": i, "id": r["id"], "a0": a0, "a1": a1, "greedy": g, "best": int(r["best"]), "cand": cand,
                    "gain": dict(zip(cand, gain)), "se": se, "nc": nc, "half": half, "dy": np.stack(dy) if dy else np.zeros((0, M), np.float32)})
    return out


def evaluate(R: list, D: dict, score_fn, alpha: float, dev) -> dict:
    """决策 = argmax(原 logits + alpha·score_fn(z))。老师改选的根上选中老师候选的比例、没改选的根上保持原首选的比例、
    按老师配对差折算的收益/根（选到没推演过的候选按 0 计，另报比例）。"""
    ch_ok = ch_n = keep_ok = keep_n = moved = outside = 0
    gain_sum = 0.0
    for t in R:
        z = D["Z"][t["a0"]:t["a1"]].to(dev)
        bl = D["BL"][t["a0"]:t["a1"]].to(dev).float()
        with torch.no_grad():
            s = bl + alpha * score_fn(z).reshape(-1) if alpha else bl
        pick = int(s.argmax())
        if pick != t["greedy"]:
            moved += 1
            if pick in t["gain"]:
                gain_sum += t["gain"][pick]
            else:
                outside += 1
        if t["best"] != t["greedy"]:
            ch_n += 1; ch_ok += int(pick == t["best"])
        else:
            keep_n += 1; keep_ok += int(pick == t["greedy"])
    n = max(1, len(R))
    return {"n": len(R), "teacher_changed": ch_n, "hit_changed": round(ch_ok / max(1, ch_n), 4), "keep_greedy": round(keep_ok / max(1, keep_n), 4),
            "moved": round(moved / n, 4), "outside": round(outside / n, 4), "gain_per_root": round(gain_sum / n, 5)}


def diff_fit(fn, S, D, dev) -> dict:
    """留出根上：预测的候选差（候选 − 原首选）与老师配对差的相关、符号一致率（只算老师差值超过 1 个标准误的对）。"""
    p, y, sig = [], [], []
    for t in S:
        if not t["cand"]:
            continue
        z = D["Z"][t["a0"]:t["a1"]].to(dev)
        with torch.no_grad():
            pr = fn(z).reshape(-1).cpu().numpy()
        for c, s_ in zip(t["cand"], t["se"]):
            p.append(pr[c] - pr[t["greedy"]]); y.append(t["gain"][c]); sig.append(abs(t["gain"][c]) > s_)
    p, y, sig = map(np.array, (p, y, sig))
    cor = lambda u, v: float(np.corrcoef(u, v)[0, 1]) if len(u) > 2 and u.std() > 0 and v.std() > 0 else float("nan")
    return {"corr_diff": round(cor(p, y), 4), "sign_agree_sig": round(float(np.mean(np.sign(p[sig]) == np.sign(y[sig]))), 4) if sig.any() else None,
            "n_pairs": int(len(p)), "n_sig": int(sig.sum())}


def reliability(S) -> dict:
    """老师标签自身的可靠性：同一对候选，用奇数号世界与偶数号世界各算一次配对差，两半的相关（世界数 ≥ 4 的对）。
    按 Spearman–Brown 折算成全部世界的可靠性：任何学生对老师配对差的相关都不会超过 sqrt(可靠性)。"""
    h = np.array([x for t in S for x in t["half"] if x is not None])
    if len(h) < 10:
        return {}
    r = float(np.corrcoef(h[:, 0], h[:, 1])[0, 1])
    full = 2 * r / (1 + r) if r > -1 else float("nan")
    return {"pairs": int(len(h)), "split_half_r": round(r, 4), "reliability_full": round(full, 4), "max_corr": round(math.sqrt(max(full, 0)), 4)}


def train(a) -> None:
    dev = a.device
    D = torch.load(a.roots, weights_only=False)
    R = root_tables(D)
    tr = [t for t in R if t["id"] % 5 != 0]
    te = [t for t in R if t["id"] % 5 == 0]
    tg = lambda S: round(sum(t["gain"].get(t["best"], 0.0) for t in S) / max(1, len(S)), 5)
    report = {"n_train": len(tr), "n_test": len(te), "changed_frac_test": round(float(np.mean([t["best"] != t["greedy"] for t in te])), 4),
              "teacher_gain_test": tg(te), "reliability": reliability(R)}
    print(f"根：训练 {len(tr)}、留出 {len(te)}；老师改选 {report['changed_frac_test']:.3f}；老师自评收益/根（同一批世界，偏乐观）{report['teacher_gain_test']}；"
          f"老师标签可靠性 {report['reliability']}", flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0); np.random.seed(0)
    Zd, BLd = D["Z"].to(dev), D["BL"].to(dev).float()
    report["S0"] = {"test": evaluate(te, D, None, 0.0, dev)}

    # ---------- S1 直接蒸馏：被搜候选里交叉熵拟合老师的选择（硬标签，ExIt 式）
    head = ResidualHead(kind="residual").to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=1e-4)
    rows = [(t["a0"], [t["greedy"]] + t["cand"], t["best"]) for t in tr if t["cand"]]
    for ep in range(a.epochs):
        perm = np.random.permutation(len(rows))
        for s in range(0, len(perm), a.bs):
            seg = [rows[i] for i in perm[s:s + a.bs]]
            idx = torch.tensor([a0 + c for a0, cs, _ in seg for c in cs], device=dev)
            lens = [len(cs) for _, cs, _ in seg]
            lg = (BLd[idx] + head(Zd[idx]).reshape(-1)).split(lens)
            L = torch.full((len(seg), max(lens)), float("-inf"), device=dev)
            for j, x in enumerate(lg):
                L[j, :len(x)] = x
            tgt = torch.tensor([cs.index(b) for _, cs, b in seg], device=dev)
            r = head(Zd[idx]).reshape(-1)
            loss = F.cross_entropy(L, tgt) + a.l2 * (r ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    report["S1"] = sweep(head, tr, te, D, dev, "S1", alphas=(0.25, 0.5, 1.0))
    head.save(f"{a.out}/s1.pt", what="S1 直接蒸馏 PIMC 选择", scale=report["S1"]["alpha"])

    # ---------- S2 反事实候选差：F(z_c) − F(z_原首选) 拟合老师配对差（得分与各读数），权重 = 世界数
    def fit_adv(pairs, tag):
        h = ResidualHead(out=M, kind="pred", zero=False).to(dev)
        o = torch.optim.AdamW(h.parameters(), lr=a.lr, weight_decay=1e-4)
        ic = torch.tensor([p[0] for p in pairs], device=dev); ig = torch.tensor([p[1] for p in pairs], device=dev)
        Y = torch.tensor(np.stack([p[2] for p in pairs]), device=dev); W = torch.tensor([p[3] for p in pairs], device=dev, dtype=torch.float32)
        sd = Y.std(0) + 1e-3
        for ep in range(a.epochs):
            perm = torch.randperm(len(Y), device=dev)
            for s in range(0, len(perm), a.bs * 4):
                k = perm[s:s + a.bs * 4]
                d = h(Zd[ic[k]]) - h(Zd[ig[k]])
                loss = ((((d - Y[k]) / sd) ** 2).mean(1) * W[k]).sum() / W[k].sum()
                o.zero_grad(); loss.backward(); o.step()
        return h
    pairs = [(t["a0"] + c, t["a0"] + t["greedy"], y, n) for t in tr for c, y, n in zip(t["cand"], t["dy"], t["nc"])]
    h2 = fit_adv(pairs, "S2")
    report["S2"] = sweep(h2, tr, te, D, dev, "S2")
    h2.save(f"{a.out}/s2.pt", what="S2 反事实候选差（老师各世界的配对差）")

    # ---------- S3 真实所选动作：F(z_a) 拟合 12 步后的得分 − 这一步的价值（优势式；同策略采样，不推演）
    if a.traj:
        T = torch.load(a.traj, weights_only=False)
        keep = (T["ids"] % 5 != 0)
        Z3, Y3 = T["Z"][keep].to(dev), T["Y"][keep].to(dev)
        h3 = ResidualHead(out=M, kind="pred", zero=False).to(dev)
        o = torch.optim.AdamW(h3.parameters(), lr=a.lr, weight_decay=1e-4)
        mu, sd = Y3.mean(0), Y3.std(0) + 1e-3
        for ep in range(a.epochs):
            perm = torch.randperm(len(Y3), device=dev)
            for s in range(0, len(perm), a.bs * 4):
                k = perm[s:s + a.bs * 4]
                loss = (((h3(Z3[k]) - (Y3[k] - mu) / sd)) ** 2).mean()
                o.zero_grad(); loss.backward(); o.step()
        with torch.no_grad():
            h3.net[2].weight.mul_(sd[:, None]); h3.net[2].bias.mul_(sd).add_(mu)
        report["S3"] = sweep(h3, tr, te, D, dev, "S3")
        h3.save(f"{a.out}/s3.pt", what="S3 真实所选动作的优势式读数（同策略采样）")
        report["S3"]["n_steps"] = int(keep.sum())
    Path(f"{a.out}/report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "sweep_train"}) for k, v in report.items()}, ensure_ascii=False, indent=1))


def sweep(h, tr, te, D, dev, tag, alphas=(0.5, 1.0, 2.0, 4.0, 8.0, 16.0)) -> dict:
    """系数按训练组的收益/根挑；另报留出组的候选差拟合。"""
    fn = (lambda z: h(z)[:, 0]) if h.kind == "pred" else (lambda z: h(z)[:, 0])
    rows = [(al, evaluate(tr, D, fn, al, dev)["gain_per_root"]) for al in alphas]
    al = max(rows, key=lambda x: x[1])[0]
    out = {"alpha": al, "sweep_train": rows, "fit_test": diff_fit(fn, te, D, dev), "train": evaluate(tr, D, fn, al, dev), "test": evaluate(te, D, fn, al, dev),
           "test_all_alpha": {str(x): evaluate(te, D, fn, x, dev) for x in alphas}}
    h.alpha = al if h.kind == "pred" else 1.0
    print(f"{tag}：α={al}；留出 {out['test']}；候选差拟合 {out['fit_test']}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("feats"); f.add_argument("--labels", required=True); f.add_argument("--ckpt", required=True); f.add_argument("--out", required=True)
    f.add_argument("--device", default="cuda")
    g = sub.add_parser("ftraj"); g.add_argument("--dir", required=True); g.add_argument("--ckpt", required=True); g.add_argument("--out", required=True)
    g.add_argument("--device", default="cuda")
    t = sub.add_parser("train"); t.add_argument("--roots", required=True); t.add_argument("--traj", default=""); t.add_argument("--out", required=True)
    t.add_argument("--device", default="cuda"); t.add_argument("--epochs", type=int, default=30); t.add_argument("--bs", type=int, default=256)
    t.add_argument("--lr", type=float, default=1e-3); t.add_argument("--l2", type=float, default=0.01)
    a = ap.parse_args()
    {"feats": feats, "ftraj": ftraj, "train": train}[a.cmd](a)


if __name__ == "__main__":
    main()
