"""实时胜率模型的训练与检验（决策记录 175）。在 A 上跑（显卡训练，读回放用一个低优先级进程）。

数据：若干目录下的整局回放（*.jsonl.gz，每局一个文件，输赢都在：定期复测的 RT 种子、评测的 eval_replays/）。
按种子分训练 / 测试（同一种子的各版本放在同一边，测试集的地图与遭遇训练时没见过），训练种子里再留 10% 做早停。
每步的权重 = 1 / 这一局这一层的步数：每局每层合计为 1，免得战斗里步数多的层、打得久的局占比过大。
检验：三个头各算加权的对数损失、Brier、AUC，与两个统计基线比——
  基线 0：按（层、房间类型、是否战斗中）统计的训练集平均；基线 1：再加血量比例十档。
另按幕分列，并给模型的校准表（预测十档：平均预测 vs 实际）。产物：<out>/model.pt、<out>/metrics.json。

    nice -n 19 .venv/bin/python -I tools/winprob/train.py --data /root/winprob/data/rt --out /root/winprob/run1 --device cuda
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from winprob import HEADS, WinProb, featurize_step, labels, pack, read_replay, select, to_torch  # noqa: E402


def load_game(head: dict, steps: list[dict], per_floor: int | None, rng) -> dict:
    """一局 → 数组。per_floor 给定时每层最多随机留这么多步（权重 1 / 留下的步数，每局每层合计仍为 1），只给留下的步算特征。"""
    y = labels(head, steps)
    g = np.asarray([s["g"] for s in steps], np.int64)
    by_floor = defaultdict(list)
    for t in range(len(steps)):
        by_floor[g[t, 1]].append(t)
    keep = sorted(t for ts in by_floor.values() for t in (ts if not per_floor or len(ts) <= per_floor else rng.choice(ts, per_floor, replace=False)))
    kept_n = defaultdict(int)
    for t in keep:
        kept_n[g[t, 1]] += 1
    rows, deck, j = [], [], 0
    for t, s in enumerate(steps):          # 牌组沿用最近一次战斗外的，所以要从头走一遍
        if "deck" in s:
            deck = s["deck"]
        if j < len(keep) and keep[j] == t:
            rows.append(featurize_step(s, deck)); j += 1
    k = np.asarray(keep, np.int64)
    p = pack(rows)
    p["y"] = y[k]
    p["w"] = np.asarray([1.0 / kept_n[g[t, 1]] for t in keep], np.float32)
    p["seed"] = np.full(len(k), str(head.get("seed")))
    p["game"] = np.zeros(len(k), np.int32)
    gk = g[k]
    p["floor"], p["room"], p["fight"], p["act"] = (gk[:, 1].astype(np.int16), gk[:, 9].astype(np.int8), gk[:, 8].astype(np.int8), gk[:, 0].astype(np.int8))
    p["hpb"] = np.minimum(9, 10 * gk[:, 2] // np.maximum(1, gk[:, 3])).astype(np.int8)
    p["games"] = 1
    return p


def load(dirs: list[str], per_floor: int | None = None, keep_prefix: str | None = None) -> dict:
    """读回放，逐局打包成数组再拼接（不攒逐步的 Python 对象）；每 500 局先拼成一块，少占内存。
    keep_prefix：种子以此开头的局不抽样（测试与早停从这些局里分，保持逐步完整）。"""
    files = sorted(f for d in dirs for f in Path(d).rglob("*.jsonl.gz"))
    chunks, parts = [], []
    for f in files:
        r = read_replay(f)
        if r is None:
            continue
        head, steps = r
        full = keep_prefix is not None and str(head.get("seed")).startswith(keep_prefix)
        rng = np.random.default_rng(int(hashlib.md5(f.name.encode()).hexdigest()[:8], 16))
        parts.append(load_game(head, steps, None if full else per_floor, rng))
        if len(parts) >= 500:
            chunks.append(concat(parts)); parts = []
    if parts:
        chunks.append(concat(parts))
    return concat(chunks) if len(chunks) > 1 else chunks[0]


def concat(ds: list[dict]) -> dict:
    """几份 load 的结果拼成一份（袋子的起止位置、局号依次平移）。"""
    out = {"n": sum(d["n"] for d in ds), "games": sum(d["games"] for d in ds)}
    for k in ds[0]:
        if k in ("n", "games") or k.endswith("_row"):
            continue
        if k.endswith("_off"):
            b = k[:-4]; base, parts = 0, [np.zeros(1, np.int64)]
            for d in ds:
                parts.append(d[k][1:] + base); base += len(d[f"{b}_id"])
            out[k] = np.concatenate(parts)
            continue
        if k == "game":
            gb, parts = 0, []
            for d in ds:
                parts.append(d[k] + gb); gb += d["games"]
            out[k] = np.concatenate(parts)
            continue
        out[k] = np.concatenate([d[k] for d in ds])
    return out


def load_cached(dirs: list[str], cache_dir: str | None, per_floor: int | None = None, keep_prefix: str | None = None) -> dict:
    """每个数据目录各缓存一份特征（<cache_dir>/<目录名>.pkl）：有就直接读，没有就读回放后写入。"""
    ds = []
    for x in dirs:
        c = Path(cache_dir) / f"{Path(x).name}.pkl" if cache_dir else None
        if c and c.exists():
            ds.append(pickle.load(open(c, "rb")))
        else:
            ds.append(load([x], per_floor, keep_prefix))
            if c:
                c.parent.mkdir(parents=True, exist_ok=True)
                pickle.dump(ds[-1], open(c, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    return ds[0] if len(ds) == 1 else concat(ds)


def bucket(seed: str) -> float:
    return int(hashlib.md5(seed.encode()).hexdigest()[:8], 16) / 2**32


def wauc(p: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """加权 AUC：正例权重 × 排在它下面的负例权重之和（并列算一半），除以总的正负权重之积。"""
    o = np.argsort(p, kind="mergesort"); p, y, w = p[o], y[o], w[o]
    wn = np.where(y == 0, w, 0.0); wp = np.where(y == 1, w, 0.0)
    _, start, inv = np.unique(p, return_index=True, return_inverse=True)
    neg_grp = np.bincount(inv, wn); neg_below = np.concatenate([[0], np.cumsum(neg_grp)[:-1]])
    num = (wp * (neg_below[inv] + 0.5 * neg_grp[inv])).sum()
    den = wp.sum() * wn.sum()
    return float(num / den) if den > 0 else float("nan")


def scores(p: np.ndarray, y: np.ndarray, w: np.ndarray) -> dict:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    ws = w.sum()
    return {"logloss": round(float(-(w * (y * np.log(p) + (1 - y) * np.log(1 - p))).sum() / ws), 4),
            "brier": round(float((w * (p - y) ** 2).sum() / ws), 4), "auc": round(wauc(p, y, w), 4), "rate": round(float((w * y).sum() / ws), 4), "n": int(len(y))}


def table_baseline(keys_tr, y_tr, w_tr, keys_te, k: float = 5.0) -> np.ndarray:
    """按键统计的加权平均，向全体平均收缩（相当于 k 份先验）。"""
    prior = (w_tr * y_tr).sum() / w_tr.sum()
    num, den = defaultdict(float), defaultdict(float)
    for kk, yy, ww in zip(map(tuple, keys_tr), y_tr, w_tr):
        num[kk] += ww * yy; den[kk] += ww
    return np.asarray([(num[kk] + k * prior) / (den[kk] + k) if kk in den else prior for kk in map(tuple, keys_te)])


def calib(p, y, w, bins: int = 10) -> list:
    out = []
    for b in range(bins):
        m = (p >= b / bins) & (p < (b + 1) / bins) if b < bins - 1 else p >= b / bins
        if w[m].sum() > 0:
            out.append([f"{b / bins:.1f}–{(b + 1) / bins:.1f}", round(float((w[m] * p[m]).sum() / w[m].sum()), 3), round(float((w[m] * y[m]).sum() / w[m].sum()), 3), int(m.sum())])
    return out


@torch.no_grad()
def predict(net, d: dict, device, bs: int = 65536) -> np.ndarray:
    net.eval(); out = []
    for i in range(0, d["n"], bs):
        b = to_torch(select(d, np.arange(i, min(d["n"], i + bs))), device)
        out.append(torch.sigmoid(net(b)).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(HEADS)), np.float32)


def loss_on(net, b: dict) -> torch.Tensor:
    logit = net(b)
    y = b["y"].float(); m = (b["y"] >= 0).float() * b["w"][:, None]
    l = F.binary_cross_entropy_with_logits(logit, y.clamp(min=0), reduction="none")
    return (l * m).sum() / m.sum().clamp(min=1e-6)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--dim", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--eval-every", type=int, default=25, help="每多少个小批测一次早停集")
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--cache", default=None, help="特征缓存（pickle，只有一个数据目录时用）：有就直接读，没有就读回放后写入")
    ap.add_argument("--cache-dir", default=None, help="每个数据目录各一份特征缓存（抽样设置变了要换目录）")
    ap.add_argument("--per-floor", type=int, default=None, help="每局每层最多随机留几步（控制内存；--holdout-prefix 的局不抽样）")
    ap.add_argument("--holdout-prefix", default=None,
                    help="只从种子以此开头的局里分出测试与早停集（如 RT），其余种子全部用来训练（评测种子 EVAL000–191 固定、反复出现）")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0); rng = np.random.default_rng(0)
    t0 = time.time()
    torch.set_num_threads(2)
    if a.cache_dir:
        d = load_cached(a.data, a.cache_dir, a.per_floor, a.holdout_prefix)
    elif a.cache and Path(a.cache).exists():
        d = pickle.load(open(a.cache, "rb"))
    else:
        d = load(a.data, a.per_floor, a.holdout_prefix)
        if a.cache:
            pickle.dump(d, open(a.cache, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    print(f"读入 {d['games']} 局、{d['n']} 步，用时 {time.time() - t0:.0f} 秒", flush=True)
    n_games, seeds, n_steps = d.pop("games"), d.pop("seed"), d["n"]
    useed, inv = np.unique(seeds, return_inverse=True)
    u = np.asarray([bucket(str(x)) for x in useed])[inv]
    hold = np.char.startswith(seeds.astype(str), a.holdout_prefix) if a.holdout_prefix else np.ones(len(seeds), bool)
    te = hold & (u < a.test_frac); va = hold & (u >= a.test_frac) & (u < a.test_frac + 0.1); tr = ~te & ~va
    # 训练集不另复制（占了大半内存），小批直接按下标从 d 里取；早停、测试集各复制一份
    tr_i = np.flatnonzero(tr)
    D = {k: select(d, np.flatnonzero(m)) for k, m in (("va", va), ("te", te))}
    D["tr"] = {k: d[k][tr_i] for k in ("y", "w", "floor", "room", "fight", "hpb", "act", "game")}
    D["tr"]["n"] = len(tr_i)
    dev = torch.device(a.device if torch.cuda.is_available() or a.device == "cpu" else "cpu")
    args = {"d": a.dim, "hidden": a.hidden, "dropout": a.dropout}
    net = WinProb(**args).to(dev)
    n_par = sum(p.numel() for p in net.parameters())
    print(f"参数 {n_par:,}；训练 {D['tr']['n']} 步（{len(set(D['tr']['game']))} 局），早停 {D['va']['n']}，测试 {D['te']['n']}（{len(set(D['te']['game']))} 局）", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    va_t = to_torch(D["va"], dev)
    best, best_state, bad, step, stop = 1e9, None, 0, 0, False
    for ep in range(a.epochs):
        perm = rng.permutation(D["tr"]["n"])
        for i in range(0, len(perm), a.bs):
            net.train()
            b = to_torch(select(d, tr_i[perm[i:i + a.bs]]), dev)
            opt.zero_grad(set_to_none=True); loss_on(net, b).backward(); opt.step()
            step += 1
            if step % a.eval_every:
                continue
            net.eval()
            with torch.no_grad():
                vl = float(loss_on(net, va_t))
            if vl < best - 1e-4:
                best, best_state, bad = vl, {k: v.detach().clone() for k, v in net.state_dict().items()}, 0
                print(f"第 {ep + 1} 遍第 {step} 批：早停集损失 {vl:.4f}（最好）", flush=True)
            else:
                bad += 1
                if bad >= a.patience:
                    stop = True
                    break
        if stop:
            break
    print(f"停在第 {step} 批，早停集最好 {best:.4f}", flush=True)
    del va_t
    net.load_state_dict(best_state)
    p = predict(net, D["te"], dev)
    T, R = D["te"], D["tr"]
    key0 = lambda X: np.stack([X["floor"], X["room"], X["fight"]], 1)
    key1 = lambda X: np.stack([X["floor"], X["room"], X["fight"], X["hpb"]], 1)
    res = {"args": vars(a), "best_val": best, "games": n_games, "steps": n_steps, "params": n_par, "test_games": len(set(T["game"])), "heads": {}}
    for h, name in enumerate(HEADS):
        mt, mr = T["y"][:, h] >= 0, R["y"][:, h] >= 0
        y, w = T["y"][mt, h].astype(float), T["w"][mt]
        b0 = table_baseline(key0(R)[mr], R["y"][mr, h], R["w"][mr], key0(T)[mt])
        b1 = table_baseline(key1(R)[mr], R["y"][mr, h], R["w"][mr], key1(T)[mt])
        r = {"模型": scores(p[mt, h], y, w), "基线0（层·房间·战斗中）": scores(b0, y, w), "基线1（再加血量十档）": scores(b1, y, w)}
        r["按幕"] = {f"第{x + 1}幕": {"模型": scores(p[mt, h][T["act"][mt] == x], y[T["act"][mt] == x], w[T["act"][mt] == x]),
                                  "基线1": scores(b1[T["act"][mt] == x], y[T["act"][mt] == x], w[T["act"][mt] == x])} for x in range(3) if (T["act"][mt] == x).any()}
        r["校准"] = calib(p[mt, h], y, w)
        res["heads"][name] = r
    torch.save({"model": net.state_dict(), "args": args, "data": a.data, "time": time.time()}, out / "model.pt")
    json.dump(res, open(out / "metrics.json", "w"), ensure_ascii=False, indent=1)
    for name, r in res["heads"].items():
        print(name, {k: v for k, v in r.items() if k not in ("按幕", "校准")}, flush=True)
    print(f"完成，用时 {time.time() - t0:.0f} 秒", flush=True)


if __name__ == "__main__":
    main()
