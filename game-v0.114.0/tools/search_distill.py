"""搜索蒸馏（决策记录 274）：把第三幕 Boss 战 PIMC 搜索的改进蒸馏进网络，按调研报告（reports/搜索与模型池蒸馏路线.md）改正第 217 条的做法。

目标（每个搜索过的决策）：logit 目标 = log π₀(c) + max(0, d_c − z·se_c) / T，再 softmax。
  π₀ 为起步网络的出手概率；d_c、se_c 为候选 c 相对贪心项的平均差与标准误（tools/sim_pimc_eval.py --labels 的 diff），贪心项与没搜到的候选按 0。
  只奖励“超过 z 个标准误之后还剩的优势”：没有显著更好的候选时目标就是 π₀ 本身（确认过的决策不被改动）；温度 T 很低，显著更好的候选会成为目标首选。
  第 217 条的 π·exp(ΔQ/0.1) 只把 13% 的改选排成首选，而且 544 次更新、学习率 2e-5 远远不够（欠拟合，训练集上也只学进 5%）。
损失：标签决策上 KL(目标 ‖ 学生)；另从学习端的批次存档抽“锚点”局面，KL(π₀ ‖ 学生) × w_anchor，压住标签以外的决策不漂；价值拴在起步网络上。
留存：按文件（整场战斗）分出一部分标签不参与训练，报“显著改选学进的比例”“确认决策被改的比例”。

    .venv/bin/python -I tools/search_distill.py --student 起步.pt --labels 标签目录 [--labels ...] --anchor-dumps d1.pkl ... --out 学生.pt
        [--z 2 --temp 0.005 --steps 4000 --lr 1e-4 --w-anchor 1.0 --scan]
--scan 只统计目标（不训练）：不同温度下，显著改选成为目标首选的比例、确认决策目标首选变了的比例。
"""
from __future__ import annotations

import argparse
import glob
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))


def load_labels(dirs: list[str]) -> tuple[list, list, list]:
    """返回 (Ragged 列表, 逐行 meta, 逐行文件号)。"""
    rs, metas, fid = [], [], []
    files = sorted({f for d in dirs for f in glob.glob(str(Path(d) / "*.pkl"))})
    for i, f in enumerate(files):
        d = pickle.load(open(f, "rb"))
        if not d["meta"]:
            continue
        rs.append(d["ragged"]); metas += d["meta"]; fid += [i] * len(d["meta"])
    return rs, metas, fid


def boost(meta: dict, n: int, z: float) -> np.ndarray:
    """各候选的显著优势 max(0, d − z·se)（相对贪心项），长度 n。"""
    b = np.zeros(n, np.float64)
    for c, (d, se, _k) in meta["diff"].items():
        c = int(c)
        if 0 <= c < n:
            b[c] = max(0.0, d - z * se)
    return b


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--labels", action="append", required=True)
    ap.add_argument("--anchor-dumps", nargs="*", default=[])
    ap.add_argument("--out", default=None)
    ap.add_argument("--z", type=float, default=2.0)
    ap.add_argument("--temp", type=float, default=0.005)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--mb", type=int, default=512, help="每步标签决策数")
    ap.add_argument("--mb-anchor", type=int, default=512, help="每步锚点局面数")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--w-anchor", type=float, default=1.0)
    ap.add_argument("--w-v", type=float, default=0.5)
    ap.add_argument("--hold-frac", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scan", action="store_true")
    a = ap.parse_args()
    import torch
    import torch.nn.functional as F
    from sts2rl.features import Ragged, collate_ragged
    from sts2rl.model import ModelConfig, PolicyValueNet, load_state, packed_meta, use_tf32
    use_tf32(); torch.manual_seed(a.seed)
    dev = torch.device(a.device)

    rs, metas, fid = load_labels(a.labels)
    lab = Ragged.concat(rs) if len(rs) > 1 else rs[0]
    fid = np.array(fid)
    ncand = np.diff(lab.cand_off).astype(np.int64)
    B = [boost(m, int(n), a.z) for m, n in zip(metas, ncand)]
    sig = np.array([b.max() > 0 for b in B])
    best = np.array([int(m["best"]) for m in metas]); greedy = np.array([int(m["greedy"]) for m in metas])
    print(f"标签：{len(metas)} 个决策，{len(set(fid.tolist()))} 场战斗；搜索改选 {int((best != greedy).sum())}，"
          f"显著（d − {a.z}·se > 0）{int(sig.sum())}", flush=True)

    blob = torch.load(a.student, map_location="cpu", weights_only=False)
    mc = blob["config"]["model"]; attend = bool(mc.get("macro_attend"))

    def build(train):
        net = PolicyValueNet(ModelConfig(**mc)).to(dev)
        load_state(net, blob["state"]); net.fast = dev.type == "cuda"; net.train(train)
        if not train:
            for p in net.parameters():
                p.requires_grad_(False)
        return net
    origin = build(False)

    def batch(r, idx):
        bt = collate_ragged(r.select(idx), attend=attend)
        d = bt.to(dev)
        if dev.type == "cuda":
            d.update({k: (torch.from_numpy(v).to(dev) if isinstance(v, np.ndarray) else v) for k, v in packed_meta(bt.core_idx, bt.core_mask).items()})
        return d

    # 起步网络在标签决策上的对数概率（各决策候选数不同，按行存）
    lp0 = []
    with torch.no_grad():
        for j in range(0, len(lab), 512):
            idx = np.arange(j, min(j + 512, len(lab)))
            lg = torch.log_softmax(origin(batch(lab, idx))[0].float(), -1).cpu().numpy()
            for t, i in enumerate(idx):
                lp0.append(lg[t, :ncand[i]].astype(np.float64))

    def target(i, T):
        x = lp0[i] + B[i] / T
        x = x - x.max(); p = np.exp(x); return p / p.sum()

    def scan_report(T, rows):
        rows = np.asarray(rows)
        s = rows[sig[rows]]; c = rows[~sig[rows]]
        hit = np.mean([target(i, T).argmax() == int(np.argmax(B[i])) for i in s]) if len(s) else float("nan")
        moved = np.mean([target(i, T).argmax() != int(np.argmax(lp0[i])) for i in c]) if len(c) else float("nan")
        return hit, moved
    allr = np.arange(len(metas))
    for T in (0.1, 0.03, 0.01, 0.005, 0.003, 0.001):
        h, mv = scan_report(T, allr)
        print(f"温度 {T}：显著改选成为目标首选 {h:.1%}；不显著的决策目标首选变了 {mv:.2%}", flush=True)
    if a.scan:
        return

    # 按场分留存
    rng = np.random.default_rng(a.seed)
    files = np.unique(fid); hold_f = set(rng.choice(files, max(1, int(len(files) * a.hold_frac)), replace=False).tolist())
    hold = np.array([i for i in allr if fid[i] in hold_f]); train = np.array([i for i in allr if fid[i] not in hold_f])
    tgt = [target(i, a.temp) for i in allr]
    print(f"训练标签 {len(train)}（显著 {int(sig[train].sum())}），留存 {len(hold)}（显著 {int(sig[hold].sum())}）", flush=True)

    anc = None
    if a.anchor_dumps:
        parts = [pickle.load(open(p, "rb")) for p in a.anchor_dumps]
        anc = Ragged.concat(parts) if len(parts) > 1 else parts[0]
        anc_multi = np.flatnonzero(np.diff(anc.cand_off) > 1)
        print(f"锚点局面 {len(anc)}（多选 {len(anc_multi)}）", flush=True)

    student = build(True)

    def pad_target(idx, W):
        t = np.zeros((len(idx), W), np.float32)
        for r, i in enumerate(idx):
            t[r, :ncand[i]] = tgt[i]
        return torch.from_numpy(t).to(dev)

    @torch.no_grad()
    def evaluate(step):
        student.eval()
        res = {}
        for name, rows in (("训练", train[rng.permutation(len(train))[:4000]] if len(train) > 4000 else train), ("留存", hold)):
            learned, kept = [], []
            for j in range(0, len(rows), 512):
                idx = np.sort(rows[j:j + 512]); b = batch(lab, idx)
                with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    am = student(b)[0].float().argmax(-1).cpu().numpy()
                for r, i in enumerate(idx):
                    if sig[i]:
                        learned.append(am[r] == int(np.argmax(B[i])))
                    else:
                        kept.append(am[r] == int(np.argmax(lp0[i])))
            res[name] = {"显著改选学进": round(float(np.mean(learned)), 4) if learned else None,
                         "不显著决策保持": round(float(np.mean(kept)), 4) if kept else None, "显著数": len(learned)}
        if anc is not None:
            idx = np.sort(np.random.default_rng(1).choice(anc_multi, 4096, replace=False)); same = []
            for j in range(0, len(idx), 512):
                b = batch(anc, idx[j:j + 512])
                with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    same.append((student(b)[0].float().argmax(-1) == origin(b)[0].float().argmax(-1)).float().cpu())
            res["锚点首选与起步一致"] = round(float(torch.cat(same).mean()), 4)
        student.train()
        print(f"第 {step} 步：{json.dumps(res, ensure_ascii=False)}", flush=True)
        return res

    hist = [evaluate(0)]
    opt = torch.optim.Adam(student.parameters(), lr=a.lr, eps=1e-5)
    t0 = time.time()
    for step in range(1, a.steps + 1):
        idx = np.sort(rng.choice(train, min(a.mb, len(train)), replace=False)); b = batch(lab, idx)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            slog, sv = student(b)
        sl = F.log_softmax(slog.float(), -1); T_ = pad_target(idx, sl.shape[1])
        l_lab = (T_ * (torch.log(T_.clamp(min=1e-12)) - sl)).nan_to_num(0.0, 0.0, 0.0).sum(-1).mean()
        with torch.no_grad():
            v0 = origin(b)[1].float()
        loss = l_lab + a.w_v * ((sv.float() - v0) ** 2).mean()
        l_anc = torch.zeros(())
        if anc is not None and a.w_anchor > 0:
            ia = np.sort(rng.choice(anc_multi, a.mb_anchor, replace=False)); ba = batch(anc, ia)
            with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                pl = torch.log_softmax(origin(ba)[0].float(), -1)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                al = torch.log_softmax(student(ba)[0].float(), -1)
            l_anc = (pl.exp() * (pl - al)).nan_to_num(0.0, 0.0, 0.0).sum(-1).mean()
            loss = loss + a.w_anchor * l_anc
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0); opt.step()
        if step % 100 == 0:
            print(f"第 {step} 步：标签 KL {float(l_lab):.4f}，锚点 KL {float(l_anc):.4f}，{time.time() - t0:.0f} 秒", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            hist.append(evaluate(step))
    if a.out:
        out = {k: v for k, v in blob.items() if k not in ("opt", "ema", "ema_state")}
        out["state"] = {k: v.detach().cpu() for k, v in student.state_dict().items()}
        out["search_distill"] = {"decision": 274, "labels": a.labels, "z": a.z, "temp": a.temp, "steps": a.steps, "lr": a.lr, "w_anchor": a.w_anchor, "eval": hist}
        torch.save(out, a.out)
        print(f"已存 {a.out}", flush=True)


if __name__ == "__main__":
    main()
