"""房间结果模型的增量训练：每小时一批，先测后训（决策记录 147）。在 A 上由 crontab 每小时跑一次。

~/roomlog/ 里的文件按 B 的时钟每小时一个（r<采集进程>_<年月日时>.jsonl.gz）。对每个已经写完、还没处理的小时（过了整点 --min-age 分钟，
A 每 10 分钟拉一次，留够余量），按时间顺序：
1. 先测：用当前模型预测这一小时的全部记录，和“按（遭遇、数据来源）统计的历史平均”这个基线比——死亡的对数损失、Brier、AUC，
   存活局掉血的平均绝对误差，整个分布的交叉熵；按数据来源 × 房间类型分开列；
2. 后训：这一小时的数据 + 从之前各小时按时间远近抽的回放（越近抽得越多，总量与这一小时相当），从上一次的检查点接着训
   --epochs 遍（学习率不衰减），存检查点。
第一个小时没有模型可测，留出最后 25%（按时间）当测试集先报一个数，再用整小时训练。
结果追加到 <root>/metrics.jsonl，检查点 <root>/model.pt，每小时特征缓存 <root>/cache/<小时>.pkl。

    python tools/roommodel/train.py --root ~/roommodel --log ~/roomlog --vocab engine/catalog/vocab.txt
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import pickle
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from roommodel import RoomModel, concat, featurize, load_vocab, predict, read_rows, select, to_torch  # noqa: E402

HOUR_RE = re.compile(r"_(\d{10})\.jsonl\.gz$")


def hours_on_disk(log: Path) -> dict[str, list[Path]]:
    out = defaultdict(list)
    for f in log.glob("*.jsonl.gz"):
        m = HOUR_RE.search(f.name)
        if m:
            out[m.group(1)].append(f)
    return dict(out)


def hour_end(h: str) -> float:
    return time.mktime(time.strptime(h, "%Y%m%d%H")) + 3600


def load_hour(root: Path, h: str, files: list[Path], vocab: dict) -> dict:
    c = root / "cache" / f"{h}.pkl"
    if c.exists():
        return pickle.load(open(c, "rb"))
    rows = read_rows(sorted(files))
    rows.sort(key=lambda r: r.get("t", 0))
    d = featurize(rows, vocab)
    d["t"] = np.asarray([r.get("t", 0) for r in rows], np.float64)
    d["enc_name"] = np.asarray([r["enc"] for r in rows])
    c.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(d, open(c, "wb"))
    return d


def batches(d: dict, size: int, shuffle: bool, rng):
    idx = rng.permutation(d["n"]) if shuffle else np.arange(d["n"])
    for s in range(0, d["n"], size):
        yield select(d, idx[s:s + size])


def auc(y: np.ndarray, p: np.ndarray) -> float:
    pos = y.sum(); neg = len(y) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    r = np.empty(len(p)); r[np.argsort(p, kind="mergesort")] = np.arange(1, len(p) + 1)
    return float((r[y].sum() - pos * (pos + 1) / 2) / (pos * neg))


def baseline(hist: list[dict]):
    """历史数据里按（遭遇、数据来源）统计的死亡率与存活掉血均值（向全局值收缩，样本少的遭遇不至于乱跳）。"""
    die = defaultdict(lambda: [0.0, 0.0]); dm = defaultdict(lambda: [0.0, 0.0])
    gd, gn, gm, gs = 0.0, 0.0, 0.0, 0.0
    for d in hist:
        for e, s, w, x in zip(d["enc_name"], d["src"], d["win"], d["dmg"]):
            k = (e, int(s))
            die[k][0] += (not w); die[k][1] += 1; gd += (not w); gn += 1
            if w:
                dm[k][0] += x; dm[k][1] += 1; gm += x; gs += 1
    g_die, g_dm = gd / max(gn, 1), gm / max(gs, 1)
    k_ = 20.0
    pd = {k: (v[0] + k_ * g_die) / (v[1] + k_) for k, v in die.items()}
    md = {k: (v[0] + k_ * g_dm) / (v[1] + k_) for k, v in dm.items()}
    return (lambda e, s: pd.get((e, s), g_die)), (lambda e, s: md.get((e, s), g_dm))


def evaluate(model, d: dict, hist: list[dict], device, rng) -> dict:
    model.eval()
    P, E, NLL = [], [], []
    with torch.no_grad():
        for b in batches(d, 8192, False, rng):
            tb = to_torch(b, device)
            lg = model(tb)
            NLL.append(F.cross_entropy(lg.float(), tb["y"], reduction="none").cpu().numpy())
            p, e = predict(lg); P.append(p); E.append(e)
    pdie, edmg, nll = np.concatenate(P), np.concatenate(E), np.concatenate(NLL)
    bdie_f, bdm_f = baseline(hist)
    bdie = np.array([bdie_f(e, int(s)) for e, s in zip(d["enc_name"], d["src"])])
    bdm = np.array([bdm_f(e, int(s)) for e, s in zip(d["enc_name"], d["src"])])
    y = ~d["win"]

    def part(m):
        if m.sum() == 0:
            return None
        yy, sv = y[m], d["win"][m]
        ll = lambda p: float(np.mean(-(yy * np.log(np.clip(p[m], 1e-6, 1)) + (~yy) * np.log(np.clip(1 - p[m], 1e-6, 1)))))
        out = {"n": int(m.sum()), "死亡率": round(float(yy.mean()), 4),
               "死亡对数损失": round(ll(pdie), 4), "基线对数损失": round(ll(bdie), 4),
               "死亡Brier": round(float(np.mean((pdie[m] - yy) ** 2)), 5), "基线Brier": round(float(np.mean((bdie[m] - yy) ** 2)), 5),
               "死亡AUC": round(auc(yy, pdie[m]), 4), "基线AUC": round(auc(yy, bdie[m]), 4), "分布交叉熵": round(float(nll[m].mean()), 4)}
        if sv.any():
            dm_ = d["dmg"][m][sv]
            out.update({"掉血MAE": round(float(np.abs(edmg[m][sv] - dm_).mean()), 2), "基线掉血MAE": round(float(np.abs(bdm[m][sv] - dm_).mean()), 2),
                        "平均掉血": round(float(dm_.mean()), 2)})
        return out

    res = {"全部": part(np.ones(d["n"], bool))}
    for sname, s in (("整局", 0), ("练习", 1)):
        for rname, r in (("普通", 0), ("精英", 1), ("Boss", 2)):
            v = part((d["src"] == s) & (d["room"] == r))
            if v:
                res[f"{sname}{rname}"] = v
    return res


def train(model, opt, d: dict, epochs: int, device, rng, bs: int = 4096) -> float:
    model.train()
    tot, nb = 0.0, 0
    for _ in range(epochs):
        for b in batches(d, bs, True, rng):
            tb = to_torch(b, device)
            loss = F.cross_entropy(model(tb), tb["y"])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
    return tot / max(nb, 1)


def replay(root: Path, done: list[str], h: str, n_target: int, rng, vocab: dict, files: dict) -> dict | None:
    """从之前各小时按时间远近抽样（权重 exp(−相隔小时 / 24)），总量约 n_target。"""
    prev = [x for x in done if x < h][-72:]
    if not prev:
        return None
    w = np.array([math.exp(-(hour_end(h) - hour_end(x)) / 3600 / 24) for x in prev]); w /= w.sum()
    parts = []
    for x, wx in zip(prev, w):
        dx = load_hour(root, x, files.get(x, []), vocab)
        k = min(dx["n"], int(round(n_target * wx)))
        if k > 0:
            parts.append(select(dx, rng.choice(dx["n"], k, replace=False)))
    return concat(parts) if parts else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path.home() / "roommodel"))
    ap.add_argument("--log", default=str(Path.home() / "roomlog"))
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--min-age", type=float, default=12, help="整点过后多少分钟才算这一小时写完")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--hours", default="", help="只处理这些小时（逗号分隔，调试用）；默认处理全部写完未处理的")
    a = ap.parse_args()
    root, log = Path(a.root), Path(a.log)
    root.mkdir(parents=True, exist_ok=True)
    lock = open(root / ".lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("上一轮还在跑", flush=True); return
    torch.set_num_threads(4)
    vocab = load_vocab(Path(a.vocab))
    rng = np.random.default_rng(int(time.time()))
    state_p = root / "state.json"
    state = json.loads(state_p.read_text()) if state_p.exists() else {"done": []}
    files = hours_on_disk(log)
    todo = [h for h in sorted(files) if h not in state["done"] and time.time() >= hour_end(h) + a.min_age * 60]
    if a.hours:
        todo = [h for h in a.hours.split(",") if h in files]
    model = RoomModel().to(a.device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=0.0)
    ck = root / "model.pt"
    if ck.exists():
        blob = torch.load(ck, map_location=a.device, weights_only=False)
        model.load_state_dict(blob["model"]); opt.load_state_dict(blob["opt"])
    for h in todo:
        t0 = time.time()
        d = load_hour(root, h, files[h], vocab)
        hist = [load_hour(root, x, files.get(x, []), vocab) for x in state["done"][-24:]]
        rec = {"hour": h, "n": d["n"], "time": round(time.time())}
        if not state["done"]:
            # 第一个小时：按时间留出最后 25% 先测一次
            cut = int(d["n"] * 0.75)
            tr, te = select(d, np.arange(cut)), select(d, np.arange(cut, d["n"]))
            tr["enc_name"], te["enc_name"] = d["enc_name"][:cut], d["enc_name"][cut:]
            rec["first_train_loss"] = round(train(model, opt, tr, 3, a.device, rng), 4)
            rec["test"] = evaluate(model, te, [tr], a.device, rng)
            rec["test_note"] = "第一个小时：前 75% 训练、后 25% 测试"
            rec["train_loss"] = round(train(model, opt, select(d, np.arange(cut, d["n"])), 1, a.device, rng), 4)
        else:
            rec["test"] = evaluate(model, d, hist, a.device, rng)
            rp = replay(root, state["done"], h, d["n"], rng, vocab, files)
            data = concat([d, rp]) if rp is not None else d
            rec["replay"] = 0 if rp is None else rp["n"]
            rec["train_loss"] = round(train(model, opt, data, a.epochs, a.device, rng), 4)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "hour": h}, ck)
        state["done"].append(h); state_p.write_text(json.dumps(state))
        rec["secs"] = round(time.time() - t0, 1)
        with open(root / "metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        a_ = rec["test"]["全部"]
        print(f"{h}：{d['n']} 场；测试 死亡对数损失 {a_['死亡对数损失']}（基线 {a_['基线对数损失']}）AUC {a_['死亡AUC']}（基线 {a_['基线AUC']}）"
              f" 掉血MAE {a_.get('掉血MAE')}（基线 {a_.get('基线掉血MAE')}）；训练损失 {rec['train_loss']}，用时 {rec['secs']} 秒", flush=True)


if __name__ == "__main__":
    main()
