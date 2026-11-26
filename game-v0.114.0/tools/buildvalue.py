"""构筑价值模型（2026-11-03，决策记录 252，情况二）：每到战斗外的决策，用当时的构筑估计这一局最后通关的概率。
只看构筑相关的输入：牌组（每张卡分升级与否）、遗物、药水、血量与上限、金币、幕与层、决策类型；不看战斗里的局面。
用途：给局外决策做按势能的塑形，把“这次选择让构筑变好还是变坏”的信号从第三幕 Boss 提前到当场（上线另需用户批准）。

数据是 tools/ability_test.py 的整局回放（<目录>/A10/replays/v<版本>_<种子>.jsonl.gz：首行整局摘要，其后每步一行，
见 python/sts2rl/replay.py 的 describe）。按种子分训练 / 测试，同一种子的各局放在同一边。

    .venv/bin/python -I tools/buildvalue.py extract --dirs 目录1 目录2 ... --out 特征.npz [--workers 24]
    .venv/bin/python -I tools/buildvalue.py train --data 特征1.npz [特征2.npz ...] --out 输出目录 [--device cuda]
    .venv/bin/python -I tools/buildvalue.py check --ckpt 检查点.pt --model 构筑价值模型.pt [--games 8]   # 采集端与训练端的输入逐步核对
特征文件是压缩的整数数组（save_compact）；早先的 .pkl 格式照常能读。

train 同时训三个对照：按层的平均、按（层 × 血量十档）的平均、只用数值输入（层、血量、上限、金币、牌数、升级比例、遗物数、药水数，
不看具体是哪些牌和遗物）的同结构网络。完整模型比第三个多出来的部分，就是“具体构筑”带来的信息。
检验按全部局外决策、第二幕开头、第三幕开头、第三幕 Boss 进场前最后一个局外决策分列：对数损失、Brier、AUC 与校准表。
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import pickle
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from sts2rl.buildvalue_model import N_KIND, V, expand, make_model  # noqa: E402  编码与网络和采集端共用一份

CKPTS = ("act2_start", "act3_start", "boss3_entry")


def game_rows(path: str):
    """一局回放 → 这一局所有战斗外决策的行。返回 None 表示文件坏了或没打完。"""
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            head = json.loads(f.readline())
            steps = [json.loads(l) for l in f if l.strip()]
    except Exception:
        return None
    if "win" not in head or not steps:
        return None
    seed = head.get("seed") or Path(path).name.split("_", 1)[1][:-9]
    rows, first_boss3 = [], None
    for t, s in enumerate(steps):
        g = s["g"]
        if g[8] and g[0] == 2 and g[1] >= 48 and first_boss3 is None:
            first_boss3 = t
    marks = {}
    seen_act = set()
    last_macro_before_boss3 = None
    for t, s in enumerate(steps):
        g = s["g"]
        if g[8] or s["d"] in (0, 12):           # 战斗中、卡住、结束
            continue
        # 紧凑行：幕、层、血量、上限、金币、决策类型、牌组（编号 × 2 + 是否升级）、遗物、药水（展开见 expand）
        rows.append((g[0], g[1], g[2], g[3], g[4], s["d"], [min(c[0], V - 1) * 2 + (c[1] > 0) for c in s.get("deck") or []],
                     [min(r[0], V - 1) for r in s.get("rel") or []], [min(p[0], V - 1) for p in s.get("pot") or []]))
        act = g[0]
        if act not in seen_act:
            seen_act.add(act)
            if act == 1: marks["act2_start"] = len(rows) - 1
            if act == 2: marks["act3_start"] = len(rows) - 1
        if first_boss3 is not None and t < first_boss3:
            last_macro_before_boss3 = len(rows) - 1
    if last_macro_before_boss3 is not None:
        marks["boss3_entry"] = last_macro_before_boss3
    return {"seed": seed, "version": int(head.get("version", 0)), "win": int(bool(head["win"])), "rows": rows, "marks": marks}


def save_compact(games: list[dict], path: str) -> None:
    """特征存成整数数组再压缩（2026-11-03 用户：特征文件太大，跨机器传不划算）。"""
    G, deck, rel, pot, start = [], [], [], [], [0]
    for g in games:
        for r in g["rows"]:
            G.append(r[:6]); deck.append(r[6]); rel.append(r[7]); pot.append(r[8])
        start.append(len(G))
    flat = lambda ls: (np.fromiter((x for l in ls for x in l), np.int16), np.cumsum([0] + [len(l) for l in ls]).astype(np.int64))
    (df, do), (rf, ro), (pf, po) = flat(deck), flat(rel), flat(pot)
    mk = np.asarray([[g["marks"].get(c, -1) for c in CKPTS] for g in games], np.int32).reshape(-1, len(CKPTS))
    np.savez_compressed(path, seed=np.asarray([g["seed"] for g in games]), version=np.asarray([g["version"] for g in games], np.int32),
                        win=np.asarray([g["win"] for g in games], np.int8), start=np.asarray(start, np.int64), marks=mk,
                        g=np.asarray(G, np.int16).reshape(-1, 6), deck=df, deck_off=do, rel=rf, rel_off=ro, pot=pf, pot_off=po)


def load_games(path: str) -> list[dict]:
    """读特征文件，返回每局一个 {seed, version, win, rows（训练用的行）, marks}。.pkl 是早先未压缩的格式。"""
    if path.endswith(".pkl"):
        old = pickle.loads(Path(path).read_bytes())
        return [{**g, "rows": [expand(r) if len(r) == 9 else r for r in g["rows"]]} for g in old]
    z = dict(np.load(path))      # 先整份读进内存：NpzFile 每取一次 z[键] 都会重新解压整个数组（第一版卡了一个小时就是这个）
    G, st = z["g"].tolist(), z["start"]
    seg = lambda f, o, i: f[o[i]:o[i + 1]].tolist()
    out = []
    for k in range(len(z["seed"])):
        rows = [expand((*G[i], seg(z["deck"], z["deck_off"], i), seg(z["rel"], z["rel_off"], i), seg(z["pot"], z["pot_off"], i)))
                for i in range(st[k], st[k + 1])]
        out.append({"seed": str(z["seed"][k]), "version": int(z["version"][k]), "win": int(z["win"][k]), "rows": rows,
                    "marks": {c: int(m) for c, m in zip(CKPTS, z["marks"][k]) if m >= 0}})
    return out


def extract(a) -> None:
    files = []
    for d in a.dirs:
        files += [str(p) for p in Path(d).glob("**/replays/*.jsonl.gz")]
    print(f"{len(files)} 局回放", flush=True)
    t0 = time.time()
    with Pool(a.workers) as pool:
        games = [g for g in pool.imap_unordered(game_rows, files, chunksize=32) if g and g["rows"]]
    print(f"读完 {len(games)} 局，{sum(len(g['rows']) for g in games)} 个局外决策，用时 {time.time() - t0:.0f} 秒", flush=True)
    save_compact(games, a.out)
    print(f"写入 {a.out}（{Path(a.out).stat().st_size / 1e6:.1f} MB）", flush=True)


# ---------------- 训练与检验 ----------------

def flatten(games: list[dict]) -> dict:
    """按层加权（每局每层合计 1），同时记每行属于哪一局、是不是检查点。"""
    scal, deck, rel, pot, y, w, gid, floor, hpf, mark = [], [], [], [], [], [], [], [], [], []
    for gi, g in enumerate(games):
        per_floor = defaultdict(int)
        for r in g["rows"]:
            per_floor[r[2]] += 1
        inv = {v: k for k, v in g["marks"].items()}
        for j, r in enumerate(g["rows"]):
            scal.append(r[4]); deck.append(r[5]); rel.append(r[6]); pot.append(r[7])
            y.append(g["win"]); w.append(1.0 / per_floor[r[2]]); gid.append(gi); floor.append(r[2]); hpf.append(r[3]); mark.append(inv.get(j, ""))
    return {"scal": np.asarray(scal, np.float32), "deck": deck, "rel": rel, "pot": pot, "y": np.asarray(y, np.float32),
            "w": np.asarray(w, np.float32), "gid": np.asarray(gid), "floor": np.asarray(floor), "hpf": np.asarray(hpf, np.float32),
            "mark": np.asarray(mark)}


def bag(lists, idx, device, drop: float = 0.0, rng=None):
    import torch
    sel = [lists[i] for i in idx]
    if drop > 0:      # 训练时每件随机丢掉一部分（正则化：同一局的上百个决策高度相关，嵌入很容易把整局背下来）
        sel = [[x for x in l if rng.random() >= drop] for l in sel]
    off = np.zeros(len(sel), np.int64); n = 0
    for k, l in enumerate(sel):
        off[k] = n; n += len(l)
    flat = np.fromiter((x for l in sel for x in l), np.int64, count=n)
    cnt = np.asarray([len(l) for l in sel], np.float32)
    return torch.from_numpy(flat).to(device), torch.from_numpy(off).to(device), torch.from_numpy(cnt).to(device)


def metrics(p, y, w) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    ll = float(np.sum(w * -(y * np.log(p) + (1 - y) * np.log(1 - p))) / np.sum(w))
    br = float(np.sum(w * (p - y) ** 2) / np.sum(w))
    o = np.argsort(p); ys, ws = y[o], w[o]
    pos = np.cumsum(ws * (1 - ys))          # 按预测从小到大，累计负例权重
    auc = float(np.sum(ws * ys * pos) / max(1e-9, np.sum(ws * ys) * np.sum(ws * (1 - ys))))
    return {"logloss": round(ll, 4), "brier": round(br, 4), "auc": round(auc, 4), "n": int(len(y)), "base": round(float(np.sum(w * y) / np.sum(w)), 4)}


def calib(p, y, w, bins=10) -> list:
    out = []
    for k in range(bins):
        m = (p >= k / bins) & (p < (k + 1) / bins) if k < bins - 1 else (p >= k / bins)
        if m.sum():
            out.append([k / bins, round(float(np.sum(w[m] * p[m]) / np.sum(w[m])), 3), round(float(np.sum(w[m] * y[m]) / np.sum(w[m])), 3), int(m.sum())])
    return out


def train(a) -> None:
    import torch
    import torch.nn.functional as F
    games = []
    for p in a.data:
        games += load_games(p)
    seeds = sorted({g["seed"] for g in games})
    h = lambda s: int(hashlib.md5(s.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    test_s = {s for s in seeds if h(s) < a.test_frac}
    val_s = {s for s in seeds if a.test_frac <= h(s) < a.test_frac + 0.1}
    tr_s = sorted(s for s in seeds if s not in test_s and s not in val_s)
    if a.train_frac < 1:      # 学习曲线：训练种子只取一部分（按种子取，测试集不变）
        tr_s = [s for s in tr_s if h(s + "#tf") < a.train_frac]
    tr_s = set(tr_s)
    part = {"train": [g for g in games if g["seed"] in tr_s],
            "val": [g for g in games if g["seed"] in val_s], "test": [g for g in games if g["seed"] in test_s]}
    D = {k: flatten(v) for k, v in part.items()}
    print(f"种子 {len(seeds)}（测试 {len(test_s)}、早停 {len(val_s)}）；局 " + "、".join(f"{k} {len(v)}" for k, v in part.items())
          + "；局外决策 " + "、".join(f"{k} {len(D[k]['y'])}" for k in D), flush=True)
    dev = a.device
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    res = {"seeds": len(seeds), "test_seeds": len(test_s), "games": {k: len(v) for k, v in part.items()}, "models": {}}

    # 对照 0、1：查表
    tr = D["train"]
    def table(keyf):
        s, n = defaultdict(float), defaultdict(float)
        for k, y, w in zip(keyf(tr), tr["y"], tr["w"]):
            s[k] += w * y; n[k] += w
        glob = float(np.sum(tr["w"] * tr["y"]) / np.sum(tr["w"]))
        return lambda d: np.asarray([(s[k] + 20 * glob) / (n[k] + 20) for k in keyf(d)])
    preds = {"按层平均": table(lambda d: list(d["floor"])),
             "层×血量十档": table(lambda d: list(zip(d["floor"], np.minimum(9, (d["hpf"] * 10).astype(int)))))}
    # 额外检验集（--extra-test）：整份不参与训练，例如局外强制探索的对局——看模型在策略平时不做的构筑上准不准
    X = {Path(x).stem: flatten(load_games(x)) for x in (a.extra_test or [])}
    xpreds = {k: {n: f(d) for n, d in X.items()} for k, f in preds.items()}
    preds = {k: f(D["test"]) for k, f in preds.items()}

    num_cols = list(range(12)) + list(range(12, 12 + N_KIND))
    for name, use_bags in ((("构筑价值（完整）", True),) if a.only_full else (("只看数值", False), ("构筑价值（完整）", True))):
        torch.manual_seed(0)
        net = make_model(D["train"]["scal"].shape[1], use_bags, a.dim, a.hidden, a.dropout).to(dev)
        emb = [p for n_, p in net.named_parameters() if n_.split(".")[0] in ("deck", "rel", "pot")]
        rest = [p for n_, p in net.named_parameters() if n_.split(".")[0] not in ("deck", "rel", "pot")]
        opt = torch.optim.AdamW([{"params": rest, "weight_decay": a.wd}] + ([{"params": emb, "weight_decay": a.emb_wd}] if emb else []), lr=a.lr)
        brng = np.random.default_rng(1)

        def fwd(d, idx, train=False):
            sc = torch.from_numpy(d["scal"][idx][:, num_cols]).to(dev)
            if not use_bags:
                return net(sc)
            dr = a.bag_drop if train else 0.0
            return net(sc, bag(d["deck"], idx, dev, dr, brng), bag(d["rel"], idx, dev, dr, brng), bag(d["pot"], idx, dev, dr, brng))

        def predict(d):
            net.eval(); ps = []
            with torch.no_grad():
                for i in range(0, len(d["y"]), 65536):
                    idx = np.arange(i, min(len(d["y"]), i + 65536))
                    ps.append(torch.sigmoid(fwd(d, idx)).cpu().numpy())
            net.train()
            return np.concatenate(ps)
        best, bad, best_state, step = 9e9, 0, None, 0
        n = len(D["train"]["y"]); rng = np.random.default_rng(0)
        t0 = time.time()
        stop = False
        for ep in range(a.epochs):
            perm = rng.permutation(n)
            for i in range(0, n, a.bs):
                idx = np.sort(perm[i:i + a.bs])
                y = torch.from_numpy(D["train"]["y"][idx]).to(dev); w = torch.from_numpy(D["train"]["w"][idx]).to(dev)
                loss = (F.binary_cross_entropy_with_logits(fwd(D["train"], idx, True), y, reduction="none") * w).sum() / w.sum()
                opt.zero_grad(); loss.backward(); opt.step(); step += 1
                if step % a.eval_every:
                    continue
                v = metrics(predict(D["val"]), D["val"]["y"], D["val"]["w"])["logloss"]
                if v < best - 1e-4:
                    best, bad, best_state = v, 0, {k: x.detach().clone() for k, x in net.state_dict().items()}
                else:
                    bad += 1
                print(f"  {name} 第 {step} 批（第 {ep + 1} 轮）：早停集对数损失 {v:.4f}（最好 {best:.4f}），{time.time() - t0:.0f} 秒", flush=True)
                if bad >= a.patience:
                    stop = True; break
            if stop:
                break
        net.load_state_dict(best_state)
        preds[name] = predict(D["test"])
        xpreds[name] = {n: predict(d) for n, d in X.items()}
        if use_bags:
            torch.save({"state": best_state, "config": {"n_scal": D["train"]["scal"].shape[1], "dim": a.dim, "hidden": a.hidden, "V": V},
                        "note": "构筑价值模型（tools/buildvalue.py，决策记录 252）"}, out / "model.pt")
    def report(te, p):
        m = {"全部局外决策": metrics(p, te["y"], te["w"])}
        for c in CKPTS:
            k = te["mark"] == c
            if k.sum():
                m[c] = metrics(p[k], te["y"][k], np.ones(int(k.sum()), np.float32))
        m["校准"] = calib(p, te["y"], te["w"])
        return m
    for name, p in preds.items():
        res["models"][name] = report(D["test"], p)
        for xn, xd in X.items():
            res["models"][name][f"额外检验 {xn}"] = report(xd, xpreds[name][xn])
        print(f"{name}：" + "；".join(f"{k} 损失 {v['logloss']} AUC {v['auc']}（{v['n']}）" for k, v in res["models"][name].items()
                                    if k != "校准" and not k.startswith("额外")), flush=True)
        for xn in X:
            m = res["models"][name][f"额外检验 {xn}"]
            print(f"  额外检验 {xn}：" + "；".join(f"{k} 损失 {v['logloss']} AUC {v['auc']}（{v['n']}）" for k, v in m.items() if k != "校准")
                  + f"；校准 {m['校准']}", flush=True)
    (out / "metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


def check(a) -> None:
    """采集端取输入（buildvalue_model.obs_row，直接读观测）与离线训练取输入（game_rows，读回放 describe）逐步核对：
    用检查点贪心打几局，每个战斗外决策两边各取一次，比较紧凑行是否完全一致；再核对估计器与训练端的预测一致。"""
    import torch
    from sts2env.env import VecEnv
    from sts2rl.actor import MODE_RUN
    from sts2rl.buildvalue_model import Estimator, obs_row
    from sts2rl.features import Ragged, collate_ragged
    from sts2rl.model import ModelConfig, PolicyValueNet, load_state
    from sts2rl.replay import describe
    blob = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    net = PolicyValueNet(ModelConfig(**blob["config"]["model"])).eval().to(a.device)
    load_state(net, blob["state"])
    est = Estimator(a.model)
    env = VecEnv(a.games)
    seeds = [f"BVCHK{j:03d}" for j in range(a.games)]
    for i, s in enumerate(seeds):
        env.send(i, {"op": "reset", "mode": "run", "character": "DEFECT", "ascension": 10, "seed": s, "snapshots": False, "route_summary": True})
    live, n, bad, rows, floors = set(range(a.games)), 0, 0, [], []
    obs = {}
    steps = 0
    while live and steps < 200000:
        for i, o in env.recv_ready(timeout=1.0):
            if isinstance(o, Exception) or o.done or o.stuck:
                live.discard(i); continue
            obs[i] = o
        if not obs:
            continue
        ids = sorted(obs)
        with torch.no_grad():
            rg = Ragged.from_obs([obs[i] for i in ids], np.full(len(ids), MODE_RUN))
            logits = net(collate_ragged(rg).to(a.device))[0]
        for k, i in enumerate(ids):
            o = obs[i]; na = o.num_actions
            pr = torch.softmax(logits[k, :na].float(), -1).cpu().numpy()
            act = int(pr.argmax())
            r1 = obs_row(o)
            d = describe(o, act, pr)
            if not d["g"][8] and d["d"] not in (0, 12):
                g = d["g"]
                r2 = (g[0], g[1], g[2], g[3], g[4], d["d"], [min(c[0], V - 1) * 2 + (c[1] > 0) for c in d.get("deck") or []],
                      [min(r[0], V - 1) for r in d.get("rel") or []], [min(p[0], V - 1) for p in d.get("pot") or []])
                n += 1
                if r1 != r2:
                    bad += 1
                    if bad <= 5:
                        print("不一致：", r1[:6], r2[:6], r1[6] == r2[6], r1[7] == r2[7], r1[8] == r2[8], flush=True)
                rows.append(r2); floors.append(g[1])
            elif r1 is not None and d["g"][8]:
                bad += 1
            env.send_step(i, act)
            steps += 1
        obs.clear()
    p_est = est(rows)
    ex = [expand(r) for r in rows]
    with torch.no_grad():
        sc = torch.from_numpy(np.asarray([e[4] for e in ex], np.float32))
        p_tr = torch.sigmoid(est.net(sc, *[bag([e[j] for e in ex], np.arange(len(ex)), "cpu") for j in (5, 6, 7)])).numpy()
    print(f"{a.games} 局、{n} 个战斗外决策（最高到第 {max(floors) if floors else 0} 层）：输入不一致 {bad} 个；"
          f"估计器与训练端预测最大差 {float(np.abs(p_est - p_tr).max()) if rows else 0:.2e}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract"); e.add_argument("--dirs", nargs="+", required=True); e.add_argument("--out", required=True)
    e.add_argument("--workers", type=int, default=16)
    t = sub.add_parser("train"); t.add_argument("--data", nargs="+", required=True); t.add_argument("--out", required=True)
    t.add_argument("--test-frac", type=float, default=0.2); t.add_argument("--epochs", type=int, default=40); t.add_argument("--bs", type=int, default=4096)
    t.add_argument("--lr", type=float, default=5e-4); t.add_argument("--wd", type=float, default=1e-2); t.add_argument("--dim", type=int, default=8)
    t.add_argument("--hidden", type=int, default=128); t.add_argument("--dropout", type=float, default=0.3); t.add_argument("--patience", type=int, default=6)
    t.add_argument("--emb-wd", type=float, default=0.1, help="嵌入的权重衰减（单独设，防止把整局背下来）")
    t.add_argument("--bag-drop", type=float, default=0.15, help="训练时牌组、遗物、药水每件随机丢掉的比例")
    t.add_argument("--eval-every", type=int, default=60, help="每多少个小批查一次早停集")
    t.add_argument("--device", default="cuda")
    t.add_argument("--train-frac", type=float, default=1.0, help="学习曲线：只用这个比例的训练种子（测试集不变）")
    t.add_argument("--only-full", action="store_true", help="只训完整模型（不训“只看数值”的对照）")
    t.add_argument("--extra-test", nargs="*", default=None, help="额外检验集（特征文件），整份不参与训练、分开报告")
    c = sub.add_parser("check"); c.add_argument("--ckpt", required=True); c.add_argument("--model", required=True)
    c.add_argument("--games", type=int, default=8); c.add_argument("--device", default="cuda")
    a = ap.parse_args()
    {"extract": extract, "train": train, "check": check}[a.cmd](a)


if __name__ == "__main__":
    main()
