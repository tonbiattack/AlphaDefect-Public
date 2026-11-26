"""未来推演方案初测之三：整网微调对照（2026-11-16）。冻结表示上的小头学不到老师的候选差时，区分“表示不够”还是“标签不够”（报告第 8 节阶段 2）。

在 PIMC 老师的根上（tools/future/boss_pimc.py 的 labels），整网可训：
  S2f：候选头输入 z 上接一个 1,024→128→1 的差值头 q，(q(候选) − q(原首选)) 拟合老师配对差（世界数加权）；
  S1f：策略本身在被搜候选里交叉熵拟合老师的选择；
两者都加 KL(原策略 ‖ 新策略) 保持原来的出手分布（权重 --kl）。按题号分组留出（题号 % 5 == 0），报告留出组的候选差拟合与按老师配对差折算的选择收益。
存下微调后的整网（S1f 可直接拿去上机对打；S2f 另存差值头，上机时按 α 接到候选打分上）。

  .venv/bin/python -I tools/future/finetune.py --labels '/root/future/t1/pimc/labels_s*.jsonl' --ckpt L1000.pt --out 目录 --mode s2f|s1f
"""
from __future__ import annotations

import argparse
import base64
import copy
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

from future_heads import ResidualHead  # noqa: E402


def load_roots(pattern: str):
    from sts2env.protocol import parse
    roots = []
    for f in sorted(glob.glob(pattern)):
        for l in open(f, encoding="utf-8"):
            if not l.strip():
                continue
            r = json.loads(l)
            if r.get("errors"):
                continue
            g = int(r["greedy"])
            gw = {}
            per = {}
            for c, s, sc, *_ in r["meas"]:
                per.setdefault(int(c), {})[s] = sc
            if g not in per:
                continue
            cand, gain, nc = [], [], []
            for c, d in per.items():
                if c == g:
                    continue
                common = [d[s] - per[g][s] for s in d if s in per[g]]
                if len(common) >= 2:
                    cand.append(c); gain.append(float(np.mean(common))); nc.append(len(common))
            roots.append({"id": r["id"], "o": parse(base64.b64decode(r["frame"])), "greedy": g, "best": int(r["best"]),
                          "cand": cand, "gain": gain, "nc": nc})
    return roots


def load_traj(d: str, H: int = 12):
    """同策略采样对局（boss_pimc.py --temp 1 --save-frames）：每个多候选的步 → (帧, 所选动作, 读数, 题号)。读数同 heads.ftraj：
    [12 步后价值 − 这一步价值, 血量变化/100, 敌人剩余比例变化, 12 步内赢, 12 步内输]（提前结束取终局）。"""
    from sts2env.protocol import parse
    out = []
    for res in sorted(glob.glob(f"{d}/res_s*.jsonl")):
        k = res.split("res_s")[1].split(".")[0]
        fb = open(f"{d}/frames_s{k}.bin", "rb")
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
                else:
                    win = bool(r["win"]); left_end = 0.0 if win else tr[-1][5]
                    term = 1.5 if win else -1.0 + 0.25 * (tr[-1][6] + 1.0 - left_end)
                    y = [term - v, ((r.get("hp") or 0) - hp) / 100.0, left_end - left, float(win), float(not win)]
                fb.seek(off)
                out.append({"id": r["id"], "o": parse(fb.read(ln)), "act": int(act), "y": y})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True); ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="s2f", choices=["s2f", "s1f", "s3f"]); ap.add_argument("--traj-dir", default=""); ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=4); ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-5); ap.add_argument("--head-lr", type=float, default=1e-3); ap.add_argument("--kl", type=float, default=1.0)
    a = ap.parse_args()
    from sts2rl.features import Ragged, collate_ragged
    from sts2rl.model import ModelConfig, PolicyValueNet, load_state
    import sim_pimc as X
    dev = a.device
    torch.manual_seed(0); np.random.seed(0)
    blob = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    net = PolicyValueNet(ModelConfig(**blob["config"]["model"])).to(dev)
    load_state(net, blob["state"])
    roots = load_roots(a.labels)
    tr = [r for r in roots if r["id"] % 5 != 0]
    te = [r for r in roots if r["id"] % 5 == 0]
    allf = roots
    if a.mode == "s3f":     # 训练用同策略采样的真实所选动作（留出题号的步不用），评价仍在老师的留出根上
        tr = [r for r in load_traj(a.traj_dir) if r["id"] % 5 != 0]
        allf = roots + tr
        ysd = torch.tensor(np.std([r["y"] for r in tr], 0) + 1e-3, device=dev, dtype=torch.float32)
        ymu = torch.tensor(np.mean([r["y"] for r in tr], 0), device=dev, dtype=torch.float32)
    print(f"{len(roots)} 个老师根，训练样本 {len(tr)}、留出根 {len(te)}", flush=True)
    cap = {}
    hk = net.cand_mlp.register_forward_hook(lambda m, i, o: cap.__setitem__("z", i[0]))

    def fwd(rs, grad: bool):
        fs = [X._no_priv(r["o"]) for r in rs]
        b = collate_ragged(Ragged.from_obs(fs, np.zeros(len(fs), np.int64))).to(dev)
        with torch.set_grad_enabled(grad), torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = net(b)
        ns = [int(r["o"].num_actions) for r in rs]
        off = np.concatenate([[0], np.cumsum(ns)])
        if cap["z"].shape[0] != off[-1]:
            raise RuntimeError(f"候选表示 {tuple(cap['z'].shape)} 与候选数 {off[-1]} 不一致（grad={grad}，{len(rs)} 帧，logits {tuple(logits.shape)}，macro {int(b['macro'].sum())}，cand_step {tuple(b['cand_step'].shape)}，训练态 {net.training}）")
        return logits.float(), cap["z"], off

    # 原策略的 logits（KL 的参照），先算好
    net.eval()
    base = {}
    for i in range(0, len(allf), 512):
        rs = allf[i:i + 512]
        lg, _, _ = fwd(rs, False)
        for j, r in enumerate(rs):
            base[id(r)] = lg[j, : int(r["o"].num_actions)].detach().cpu()
    head = ResidualHead(out=5 if a.mode == "s3f" else 1, kind="pred", zero=False).to(dev)
    params = [{"params": net.parameters(), "lr": a.lr}, {"params": head.parameters(), "lr": a.head_lr}]
    opt = torch.optim.AdamW(params, weight_decay=0.0)
    gsd = float(np.std([g for r in tr for g in r["gain"]]) + 1e-3) if a.mode != "s3f" else 1.0
    t0 = time.time()
    for ep in range(a.epochs):
        net.train()
        perm = np.random.permutation(len(tr))
        tot = [0.0, 0.0, 0]
        for s in range(0, len(perm), a.bs):
            rs = [tr[i] for i in perm[s:s + a.bs]]
            lg, z, off = fwd(rs, True)
            l_task = torch.zeros((), device=dev); l_kl = torch.zeros((), device=dev)
            if a.mode == "s3f":
                rows = torch.tensor([off[j] + r["act"] for j, r in enumerate(rs)], device=dev)
                y = torch.tensor([r["y"] for r in rs], device=dev, dtype=torch.float32)
                l_task = (((head(z[rows].float()) - (y - ymu)) / ysd) ** 2).mean()
            elif a.mode == "s2f":
                q = head(z.float()).reshape(-1)
                ic, ig, y, w = [], [], [], []
                for j, r in enumerate(rs):
                    for c, g_, n in zip(r["cand"], r["gain"], r["nc"]):
                        ic.append(off[j] + c); ig.append(off[j] + r["greedy"]); y.append(g_); w.append(n)
                if ic:
                    ic, ig = torch.tensor(ic, device=dev), torch.tensor(ig, device=dev)
                    y, w = torch.tensor(y, device=dev), torch.tensor(w, device=dev, dtype=torch.float32)
                    l_task = (((q[ic] - q[ig] - y) / gsd) ** 2 * w).sum() / w.sum()
            else:
                ce = []
                for j, r in enumerate(rs):
                    cs = [r["greedy"]] + r["cand"]
                    if len(cs) > 1:
                        ce.append(F.cross_entropy(lg[j, cs][None], torch.tensor([cs.index(r["best"])], device=dev)))
                if ce:
                    l_task = torch.stack(ce).mean()
            for j, r in enumerate(rs):
                n = int(r["o"].num_actions)
                pb = torch.softmax(base[id(r)].to(dev), -1)
                l_kl = l_kl + (pb * (torch.log(pb + 1e-12) - torch.log_softmax(lg[j, :n], -1))).sum()
            l_kl = l_kl / len(rs)
            loss = l_task + a.kl * l_kl
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot[0] += float(l_task); tot[1] += float(l_kl); tot[2] += 1
        res = evaluate(net, head, te, fwd, base, a.mode)
        print(f"第 {ep + 1} 遍：任务损失 {tot[0] / tot[2]:.4f}，KL {tot[1] / tot[2]:.4f}，{time.time() - t0:.0f} 秒；留出 {res}", flush=True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    final = {"mode": a.mode, "test": evaluate(net, head, te, fwd, base, a.mode),
             "train": evaluate(net, head, [r for r in roots if r["id"] % 5 != 0][:3000], fwd, base, a.mode)}
    hk.remove()
    blob2 = {k: v for k, v in blob.items() if k not in ("state",)}
    blob2["state"] = {k: v.detach().cpu() for k, v in net.state_dict().items()}
    blob2["finetune"] = {"mode": a.mode, "from": a.ckpt, "labels": a.labels, "epochs": a.epochs, "kl": a.kl}
    torch.save(blob2, out / f"{a.mode}_net.pt")
    if a.mode == "s2f":
        head.save(str(out / "s2f_head.pt"), what="S2f 整网微调的候选差值头")
    (out / f"{a.mode}_report.json").write_text(json.dumps(final, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(final, ensure_ascii=False))


@torch.no_grad()
def evaluate(net, head, S, fwd, base, mode) -> dict:
    net.eval()
    p, y, sig_n = [], [], 0
    res = {}
    picks = {al: [0, 0, 0, 0, 0.0, 0] for al in (0.0, 1.0, 2.0, 4.0, 8.0)}      # 改选命中、改选数、保持、保持数、收益、移动
    for i in range(0, len(S), 256):
        rs = S[i:i + 256]
        lg, z, off = fwd(rs, False)
        q = head(z.float())[:, 0]
        for j, r in enumerate(rs):
            n = int(r["o"].num_actions)
            qj = q[off[j]:off[j] + n]
            for c, g_ in zip(r["cand"], r["gain"]):
                p.append(float(qj[c] - qj[r["greedy"]])); y.append(g_)
            gain = dict(zip(r["cand"], r["gain"]))
            for al, st in picks.items():
                s_ = (lg[j, :n] if mode == "s1f" else base[id(r)].to(lg.device)) + (al * qj if mode in ("s2f", "s3f") else 0.0)
                pk = int(s_.argmax())
                if r["best"] != r["greedy"]:
                    st[0] += int(pk == r["best"]); st[1] += 1
                else:
                    st[2] += int(pk == r["greedy"]); st[3] += 1
                if pk != r["greedy"]:
                    st[4] += gain.get(pk, 0.0); st[5] += 1
    p, y = np.array(p), np.array(y)
    res["corr_diff"] = round(float(np.corrcoef(p, y)[0, 1]), 4) if len(p) > 2 and p.std() > 0 else None
    for al, st in picks.items():
        if mode == "s1f" and al != 0.0:
            continue
        res[f"a{al:g}"] = {"hit_changed": round(st[0] / max(1, st[1]), 4), "keep": round(st[2] / max(1, st[3]), 4),
                           "gain_per_root": round(st[4] / max(1, len(S)), 5), "moved": round(st[5] / max(1, len(S)), 4)}
    return res


if __name__ == "__main__":
    main()
