"""战斗结果模型（2026-11-11，方案见 reports/战斗结果模型方案.md）：从开战第一帧预测本场输赢、赢了之后的战后血量分布、每瓶药水用掉的概率。

两种结构：
  frozen  冻结策略网络的主干（编码器），只训新头：看现有表示的上限；
  full    拷贝一份策略网络整网微调，与训练中的网络完全分开。
输入只用编码器的全局表示 g（编码器不读特权特征，帧带不带特权对它一样）。模式嵌入沿用出手时的模式：
整局里的战斗为 MODE_RUN，按快照开的单场战斗为 MODE_COMBAT。

子命令：
  train    --d1 collect 的输出 --d2 perturb 的输出 --kind frozen|full --out 模型.pt
  predict  --model 模型.pt --frames 带 frame 字段的记录文件 --out 预测.jsonl（每行原记录的键 + p、ef、h、pot）
在 B 上跑（显卡）：/root/silent-deploy/sls2/.venv/bin/python -I combat_model.py train ...
"""
from __future__ import annotations

import argparse
import base64
import collections
import gzip
import hashlib
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

DEPLOY = os.environ.get("CMB_DEPLOY", "/root/silent-deploy/sls2")
sys.path.insert(0, os.path.join(DEPLOY, "python"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from sts2env.protocol import TokenKind, Zone, parse  # noqa: E402
from sts2rl.actor import MODE_COMBAT, MODE_RUN, load_weights  # noqa: E402
from sts2rl.config import TrainConfig  # noqa: E402
from sts2rl.features import Ragged, collate_ragged  # noqa: E402
from sts2rl.model import PolicyValueNet  # noqa: E402

BINS = 41
POT_SLOTS = 5


def read_lines(path: str):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def held_potions(o) -> list[int]:
    """帧里我方的药水（词表 ID），按帧里的先后。"""
    ti = o.tok_ints
    m = (ti[:, 0] == TokenKind.POTION) & (ti[:, 2] == Zone.ALLY)
    return [int(x) for x in ti[m, 1]]


def snap_potions(snap: str | None, vi: dict) -> collections.Counter | None:
    if not snap:
        return None
    try:
        d = json.loads(snap)
    except Exception:
        return None
    return collections.Counter(vi.get(p["id"], -1) for p in (d.get("potions") or []))


def potion_labels(o, end: collections.Counter | None) -> list[int] | None:
    """每瓶开战时的药水打完后还在不在：按药水种类的数量差，前几瓶记为用掉。end 为战后快照里各种药水的瓶数，缺失（输了、取不到）时为 None。"""
    if end is None:
        return None
    start = held_potions(o)
    left = dict(end)
    out = []
    for vid in start:
        if left.get(vid, 0) > 0:
            left[vid] -= 1; out.append(0)
        else:
            out.append(1)
    return out


def load_records(d1: list[str], d2: list[str], vi: dict) -> list[dict]:
    """把 collect（D1）与 perturb（D2）的记录统一成：帧字节、模式、输赢、开战与战后血量、战后各种药水的瓶数、分组（按整局种子）、分层。
    战后快照读进来就只留药水瓶数，不留原文（省内存，组批时也不用再解析）。"""
    recs = []
    for fn in d1:
        for r in read_lines(fn):
            for ci, c in enumerate(r["combats"]):
                if not c.get("frame") or c.get("win") is None:
                    continue
                recs.append({"src": "D1", "group": r["seed"], "key": {"seed": r["seed"], "ci": ci}, "mode": MODE_RUN,
                             "frame": base64.b64decode(c["frame"]), "win": int(c["win"]), "hp0": c["hp0"], "max0": c["max0"],
                             "hp_end": c.get("hp_end", 0), "end_pots": snap_potions(c.get("snap1"), vi), "act": c["act"], "room": c["room"],
                             "encounter": c.get("encounter"), "second": bool(c.get("second")), "p_old": c.get("p"), "h_old": c.get("h")})
    for fn in d2:
        for r in read_lines(fn):
            if "error" in r or not r.get("frame") or r.get("win") is None:
                continue
            recs.append({"src": "D2", "group": r["src"], "key": {"src": r["src"], "ci": r["ci"], "n_edits": len(r.get("edits") or [])},
                         "mode": MODE_COMBAT, "frame": base64.b64decode(r["frame"]), "win": int(r["win"]), "hp0": r["hp0"], "max0": r["max0"],
                         "hp_end": r.get("hp_end", 0), "end_pots": snap_potions(r.get("snap1"), vi), "act": r["act"], "room": r["room"],
                         "encounter": r.get("encounter"), "second": bool(r.get("second")), "p_old": r.get("p"), "h_old": r.get("h")})
    return recs


def is_val(group: str) -> bool:
    return int(hashlib.md5(group.encode()).hexdigest(), 16) % 20 == 0


class CombatModel(nn.Module):
    def __init__(self, base: PolicyValueNet, frozen: bool):
        super().__init__()
        d = base.cfg.d
        self.base, self.frozen = base, frozen
        if frozen:
            base.requires_grad_(False)
        self.win = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, 1))
        self.hp = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, BINS))
        self.pot = nn.Sequential(nn.Linear(2 * d + POT_SLOTS, 2 * d), nn.GELU(), nn.Linear(2 * d, 1))
        edges = torch.linspace(0.0, 1.0, BINS + 1)
        self.register_buffer("edges", edges, persistent=False)
        self.register_buffer("centers", (edges[:-1] + edges[1:]) / 2, persistent=False)

    def forward(self, b: dict, pot_vid, pot_row, pot_slot):
        if self.frozen:
            with torch.no_grad():
                _, _, g = self.base._encode(b)
            pe = self.base.id_emb(pot_vid).detach()
        else:
            _, _, g = self.base._encode(b)
            pe = self.base.id_emb(pot_vid)
        g = g.float()
        wl = self.win(g).squeeze(-1)
        hl = self.hp(g)
        pl = self.pot(torch.cat([g[pot_row], pe.float(), F.one_hot(pot_slot.clamp(0, POT_SLOTS - 1), POT_SLOTS).float()], -1)).squeeze(-1)
        return wl.float(), hl.float(), pl.float()

    def set_train(self, on: bool) -> None:
        self.train(on)
        if self.frozen:
            self.base.eval()        # 冻结主干始终用推理模式（不开随机失活）


def hl_target(r: torch.Tensor, edges: torch.Tensor, sigma: float = 0.75) -> torch.Tensor:
    """同 train.hl_target：以目标为中心、标准差 sigma × 桶宽的正态分布落在各桶的概率。"""
    s = sigma * float(edges[1] - edges[0]) * 2 ** 0.5
    cdf = torch.special.erf((edges[None, :] - r.clamp(float(edges[0]), float(edges[-1]))[:, None]) / s)
    p = cdf[:, 1:] - cdf[:, :-1]
    return p / p.sum(-1, keepdim=True).clamp(min=1e-12)


def make_batch(recs: list[dict], idx: np.ndarray, vi: dict, device: str, labels: bool = True):
    obs = [parse(recs[i]["frame"]) for i in idx]
    b = collate_ragged(Ragged.from_obs(obs, np.array([recs[i]["mode"] for i in idx]))).to(device)
    pv, pr, ps, pl = [], [], [], []
    for row, (i, o) in enumerate(zip(idx, obs)):
        vids = held_potions(o)
        lab = potion_labels(o, recs[i].get("end_pots")) if labels and recs[i]["win"] else None
        for s, v in enumerate(vids):
            pv.append(v); pr.append(row); ps.append(s); pl.append(-1 if lab is None else lab[s])
    t = lambda x, dt: torch.tensor(x, dtype=dt, device=device)
    out = {"b": b, "pot_vid": t(pv, torch.long), "pot_row": t(pr, torch.long), "pot_slot": t(ps, torch.long), "n_pot": [len(held_potions(o)) for o in obs]}
    if labels:
        out["win"] = t([recs[i]["win"] for i in idx], torch.float32)
        out["frac"] = t([min(1.0, max(0.0, recs[i]["hp_end"] / max(1, recs[i]["max0"]))) for i in idx], torch.float32)
        out["pot_y"] = t(pl, torch.float32)
    return out


def build(cfg_path: str, weights: str, kind: str, device: str):
    cfg = TrainConfig.from_args(["--config", cfg_path, "--run-dir", "/root/cmbpred/rd"])
    base = PolicyValueNet(cfg.model)
    ver, _, _ = load_weights(base, weights, -1)
    if ver < 0:
        raise SystemExit(f"权重读不出来：{weights}")
    m = CombatModel(base, frozen=(kind == "frozen")).to(device)
    base.fast = device.startswith("cuda")       # 显卡上用打包变长注意力（与学习端相同，数学等价）
    return cfg, m, ver


def vocab_index() -> dict:
    vocab = [l.rstrip("\n") for l in open(os.path.join(DEPLOY, "engine", "catalog", "vocab.txt"), encoding="utf-8")]
    return {k: i for i, k in enumerate(vocab)}


def loss_terms(m: CombatModel, bt: dict):
    wl, hl, pl = m(bt["b"], bt["pot_vid"], bt["pot_row"], bt["pot_slot"])
    lw = F.binary_cross_entropy_with_logits(wl, bt["win"])
    wm = bt["win"] > 0.5
    lh = -(hl_target(bt["frac"][wm], m.edges) * F.log_softmax(hl[wm], -1)).sum(-1).mean() if wm.any() else wl.new_zeros(())
    pm = bt["pot_y"] >= 0
    lp = F.binary_cross_entropy_with_logits(pl[pm], bt["pot_y"][pm]) if pm.any() else wl.new_zeros(())
    return lw, lh, lp


def cmd_train(a) -> None:
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    t0 = time.time()
    vi = vocab_index()
    recs = load_records([x for x in a.d1.split(",") if x], [x for x in a.d2.split(",") if x], vi)
    tr = np.array([i for i, r in enumerate(recs) if not is_val(r["group"])])
    va = np.array([i for i, r in enumerate(recs) if is_val(r["group"])])
    print(f"读入 {len(recs)} 场（D1 {sum(r['src'] == 'D1' for r in recs)}，D2 {sum(r['src'] == 'D2' for r in recs)}），"
          f"训练 {len(tr)}，验证 {len(va)}，{time.time() - t0:.0f}s", flush=True)
    cfg, m, ver = build(a.config, a.weights, a.kind, a.device)
    heads = [p for n, p in m.named_parameters() if not n.startswith("base.")]
    groups = [{"params": heads, "lr": a.lr_head}]
    if a.kind == "full":
        groups.append({"params": [p for p in m.base.parameters() if p.requires_grad], "lr": a.lr_base})
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    rng = np.random.default_rng(a.seed)
    best, best_state, bad = float("inf"), None, 0
    pool = ThreadPoolExecutor(max_workers=a.workers)
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if a.device.startswith("cuda") else torch.autocast("cpu", enabled=False)

    def batches(ix: np.ndarray, shuffle: bool):
        ix = rng.permutation(ix) if shuffle else ix
        chunks = [ix[s:s + a.batch] for s in range(0, len(ix), a.batch)]
        # 组批线程里直接放到目标设备（Batch.to 会连 groups 里的张量一起搬）
        futs = [pool.submit(make_batch, recs, c, vi, a.device) for c in chunks[:a.workers * 2]]
        for k in range(len(chunks)):
            bt = futs[k].result()
            if k + a.workers * 2 < len(chunks):
                futs.append(pool.submit(make_batch, recs, chunks[k + a.workers * 2], vi, a.device))
            yield bt

    def evaluate() -> tuple[float, float, float]:
        m.set_train(False)
        s = np.zeros(3); n = 0
        with torch.no_grad(), amp:
            for bt in batches(va, False):
                lw, lh, lp = loss_terms(m, bt)
                k = len(bt["win"]); s += np.array([lw.item(), lh.item(), lp.item()]) * k; n += k
        m.set_train(True)
        return tuple(s / max(n, 1))

    m.set_train(True)
    step = 0
    for ep in range(a.epochs):
        te = time.time(); acc = np.zeros(3); n = 0
        for bt in batches(tr, True):
            with amp:
                lw, lh, lp = loss_terms(m, bt)
                loss = lw + lh + a.w_pot * lp
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
            opt.step()
            acc += np.array([lw.item(), lh.item(), lp.item()]); n += 1; step += 1
            if a.max_steps and step >= a.max_steps:
                break
        vw, vh, vp = evaluate()
        val = vw + vh + a.w_pot * vp
        print(f"第 {ep + 1} 遍 {time.time() - te:.0f}s 训练 输赢 {acc[0] / n:.4f} 血量 {acc[1] / n:.4f} 药水 {acc[2] / n:.4f} | "
              f"验证 输赢 {vw:.4f} 血量 {vh:.4f} 药水 {vp:.4f}", flush=True)
        if val < best - 1e-4:
            best, bad = val, 0
            best_state = {k: v.detach().cpu().clone() for k, v in m.state_dict().items() if a.kind == "full" or not k.startswith("base.")}
        else:
            bad += 1
            if bad >= a.patience:
                break
        if a.max_steps and step >= a.max_steps:
            break
    torch.save({"kind": a.kind, "weights": a.weights, "version": ver, "state": best_state, "best_val": best,
                "config": a.config}, a.out)
    print(f"存 {a.out}：最好验证损失 {best:.4f}，共 {time.time() - t0:.0f}s", flush=True)


def load_model(path: str, device: str) -> CombatModel:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    _, m, _ = build(blob["config"], blob["weights"], blob["kind"], device)
    missing, unexpected = m.load_state_dict(blob["state"], strict=False)
    bad = [k for k in missing if not (blob["kind"] == "frozen" and k.startswith("base."))]
    if bad or unexpected:
        raise SystemExit(f"模型权重对不上：缺 {bad[:5]}，多 {unexpected[:5]}")
    return m.eval()


def cmd_predict(a) -> None:
    m = load_model(a.model, a.device)
    vi = vocab_index()
    fh = open(a.out, "w", encoding="utf-8")
    n = 0
    for fn in a.frames.split(","):
        rows = []
        for r in read_lines(fn):
            if "combats" in r:          # collect 的整局记录：每场战斗出一行，模式为整局
                if a.val_only and not is_val(r["seed"]):
                    continue
                for ci, c in enumerate(r["combats"]):
                    if c.get("frame"):
                        rows.append({"seed": r["seed"], "ci": ci, "frame": c["frame"], "hp0": c["hp0"], "max0": c["max0"], "_mode": MODE_RUN})
            elif r.get("frame") and "error" not in r:     # 按快照开的单场战斗（cf、perturb、只取开战帧）
                rows.append(r)
        recs = [{"frame": base64.b64decode(r["frame"]), "mode": r.get("_mode", MODE_COMBAT), "win": 1, "end_pots": None} for r in rows]
        for s in range(0, len(recs), a.batch):
            idx = np.arange(s, min(len(recs), s + a.batch))
            bt = make_batch(recs, idx, vi, a.device, labels=False)
            with torch.no_grad(), (torch.autocast("cuda", dtype=torch.bfloat16) if a.device.startswith("cuda") else torch.autocast("cpu", enabled=False)):
                wl, hl, pl = m(bt["b"], bt["pot_vid"], bt["pot_row"], bt["pot_slot"])
            p = torch.sigmoid(wl.float()).cpu().numpy()
            ef = (F.softmax(hl.float(), -1) @ m.centers).cpu().numpy()
            pp = torch.sigmoid(pl.float()).cpu().numpy()
            off = 0
            for row, i in enumerate(idx):
                r = rows[i]
                k = bt["n_pot"][row]
                key = {kk: r[kk] for kk in ("sid", "var", "k", "seed", "ci", "src") if kk in r}
                h = r["hp0"] / max(1, r["max0"]) - float(p[row]) * float(ef[row])     # 与旧辅助头同口径：到本场结束要掉的血 / 最大生命（输了按掉光）
                fh.write(json.dumps({**key, "p": float(p[row]), "ef": float(ef[row]), "h": h, "pot": [round(float(x), 4) for x in pp[off:off + k]]}) + "\n")
                off += k
                n += 1
    fh.close()
    print(f"预测 {n} 帧 → {a.out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "predict"])
    ap.add_argument("--config", default="/root/silent_train_config.json")
    ap.add_argument("--weights", default="/root/cmbpred/w.pt")
    ap.add_argument("--kind", choices=["frozen", "full"], default="frozen")
    ap.add_argument("--d1", default="")
    ap.add_argument("--d2", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="")
    ap.add_argument("--frames", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-base", type=float, default=1e-4)
    ap.add_argument("--w-pot", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-steps", type=int, default=0, help="冒烟用：最多训几步")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-only", action="store_true", help="predict：collect 的整局记录只预测验证整局")
    a = ap.parse_args()
    {"train": cmd_train, "predict": cmd_predict}[a.cmd](a)


if __name__ == "__main__":
    main()
