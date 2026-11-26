"""一小时搜索测试（2026-10-31，决策记录 217）：两台机器停训，测搜索能让现模型在实际对局里多赢多少，以及搜索蒸馏对网络有没有用。

搜索沿用第 0 步诊断的设置（tools/search_diag.py search，推演式改选：概率不到 0.9 时取前 4 个候选、共同随机数、逐轮减半 2 轮、
首轮每候选 4 次、价值叶子 12 步、与贪心项逐世界配对超过 2 个标准误才改选），只在第三幕两个 Boss 上用。题取自千种子复测整局里
打到第三幕第一个 Boss 的全部快照（不分层抽样，保持实际对局的分布），连打两个 Boss，与贪心逐题配对。

  python3 tools/search_test.py exam 复测目录... --out 题.json [--prefix SX] [--shuffle 1]
  .venv/bin/python -I tools/search_test.py search --ckpt 检查点.pt --exam 题.json --out 结果_s0.jsonl --labels 标签_s0.pkl --shard 0/12 --envs 4 --device cuda
  .venv/bin/python -I tools/search_test.py distill --ckpt 检查点.pt --labels 标签目录 --out 新检查点.pt --target soft|hard [--device cuda]
  python3 tools/search_test.py combine --exam 题.json --greedy 贪心.jsonl --search 搜索结果目录 --retest 复测目录

标签（蒸馏用）：每个搜索过的决策一条——原始观测（sts2env.Obs）、网络概率、贪心项、搜索的选择、各候选的推演得分（与贪心项同一批世界）。
蒸馏只训策略输出：soft 目标 π′(a) ∝ π(a)·exp((Q̂(a) − Q̂(贪心))/λ)（只调整搜索过的候选，其余保持原概率）；hard 目标为搜索的选择，
另加 β·KL(π 原 ‖ π 新) 拴住原策略。每条标签同时按战斗练习与整局两种模式标记各训一次（整局里的 Boss 战，网络看到的是整局模式）。
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "tools"))


# ---------- exam：取全部第三幕 Boss 快照建题

def cmd_exam(a) -> None:
    items = []
    for d in a.dirs:
        for f in sorted(Path(d).glob("runs_s*.jsonl")):
            for line in open(f, encoding="utf-8"):
                if not line.strip():
                    continue
                r = json.loads(line)
                b = r.get("boss3")
                if b and b.get("snap") and b.get("next_encounter"):
                    items.append({"from_dir": Path(d).name, "from_seed": r["seed"], "from_win": int(bool(r.get("win"))), "version": r.get("version"),
                                  "encounter": b["encounter"], "next_encounter": b["next_encounter"],
                                  "hp_frac": (b.get("hp") or 0) / max(1, b.get("max_hp") or 1), "potions": b.get("potions"), "snap": b["snap"]})
    if a.shuffle:
        random.Random(a.shuffle).shuffle(items)      # 题号顺序随机：中途截止时做完的题仍是随机样本
    for k, it in enumerate(items):
        it["id"] = k
        it["spec"] = {"character": "DEFECT", "ascension": 10, "seed": f"{a.prefix}{k:04d}", "act": 2, "encounter": it["encounter"], "room": "Boss",
                      "next_encounter": it["next_encounter"], "player_snapshot": it.pop("snap")}
    Path(a.out).write_text(json.dumps({"built_from": [str(d) for d in a.dirs], "items": items}, ensure_ascii=False), encoding="utf-8")
    print(f"{len(items)} 道题 → {a.out}")


# ---------- search：带标签的推演式改选

class _Until(list):
    """过了截止时刻就不再交出新题（已经开打的题照常做完）。题号顺序是随机的，所以做完的题仍是随机样本，不偏向打得快的题。"""
    def __init__(self, xs, deadline):
        super().__init__(xs)
        self.deadline = deadline

    def __iter__(self):
        for x in list.__iter__(self):
            if self.deadline and time.time() > self.deadline:
                print("到截止时刻，不再开新题", flush=True)
                return
            yield x


def _search_item(M, np, item, a, seed_base, labels):
    """mcts_eval.search_item 的同一算法（配对、z 个标准误），另把各候选的推演得分记进标签。"""
    spec, prefix = item["spec"], []
    n_search = n_sims = n_changed = 0
    lose = {"win": False, "hp": 0, "max_hp": 1}
    while True:
        (o,) = yield ("tasks", [M.Task(spec, prefix, "obs")])
        if isinstance(o, M.EngineError) or o.done or o.stuck or o.num_actions == 0 or len(prefix) > M.MAX_STEPS:
            res = M.outcome(o) if not isinstance(o, M.EngineError) and len(prefix) <= M.MAX_STEPS else dict(lose)
            res.update(decisions=len(prefix), searched=n_search, sims=n_sims, changed=n_changed, root=None)
            return res
        p = yield ("policy", o)
        order = np.argsort(-p)
        act = int(order[0])
        if o.num_actions > 1 and p[act] < a.tau:
            cands = [int(c) for c in order[: a.k] if p[c] >= a.pmin] or [act]
            if len(cands) > 1:
                n_search += 1
                sc: dict = {c: {} for c in cands}
                alive, n_new, used = list(cands), a.sims0, 0
                for _ in range(a.rounds):
                    seeds = [seed_base + 7919 * (len(prefix) + 1) + used + j for j in range(n_new)]
                    used += n_new
                    tasks = [M.Task(spec, prefix, "rollout", seed=s, first=c, owner=(c, s)) for c in alive for s in seeds]
                    results = yield ("tasks", tasks)
                    n_sims += len(tasks)
                    for t, r in zip(tasks, results):
                        c, s_ = t.owner
                        sc[c][s_] = M.score_value(r if isinstance(r, dict) else lose, 0.0)
                    if len(alive) <= 2:
                        continue
                    mean = lambda c: sum(sc[c].values()) / len(sc[c])
                    rest = sorted((c for c in alive if c != act), key=lambda c: (-mean(c), -p[c]))
                    alive = [act] + rest[: max(1, math.ceil(len(alive) / 2) - 1)]
                    n_new *= 2
                best, best_m = act, 0.0
                diff = {}
                for c in sc:
                    if c == act:
                        continue
                    common = [s_ for s_ in sc[c] if s_ in sc[act]]
                    if len(common) < 2:
                        continue
                    d = [sc[c][s_] - sc[act][s_] for s_ in common]
                    m = sum(d) / len(d)
                    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1))
                    diff[c] = (m, sd / math.sqrt(len(d)), len(d))
                    if m > best_m and m > a.z * sd / math.sqrt(len(d)):
                        best, best_m = c, m
                n_changed += best != act
                labels.append({"obs": o, "p": p.astype(np.float32), "act": act, "best": best, "diff": diff,
                               "q": {c: (sum(v.values()) / len(v), len(v)) for c, v in sc.items()}})
                act = best
        prefix.append(act)


def cmd_search(a) -> None:
    import numpy as np
    import torch
    import mcts_eval as M
    import search_diag as SD
    torch.set_num_threads(a.threads)
    SD._install(M, np)
    M.SCORE, M.LEAF = "dmg", a.leaf
    M.search_item = lambda item, args, seed_base, labels=None: _search_item(M, np, item, args, seed_base, labels)
    net, blob = SD._load(a, torch)
    items = _Until([x for x in SD._items(a) if x[0] not in SD._done_ids(Path(a.out))], a.deadline)
    print(f"版本 {blob.get('version')}，{len(items)} 道题待做；leaf={a.leaf} sims0={a.sims0} rounds={a.rounds} k={a.k} tau={a.tau} z={a.z}", flush=True)
    labels: dict = {}
    out = open(a.out, "a", encoding="utf-8")
    lab = open(a.labels, "ab")
    t0 = time.time()

    def on_item(i, res):
        rows = labels.pop(i, [])
        pickle.dump({"id": i, "rows": rows}, lab); lab.flush()
        out.write(json.dumps({"id": i, "win": int(res["win"]), "stage": int(res.get("stage", 0)), "decisions": res["decisions"], "searched": res["searched"],
                              "sims": res["sims"], "changed": res["changed"], "t": round(time.time())}, ensure_ascii=False) + "\n"); out.flush()
        print(f"  题 {i}：{'赢' if res['win'] else '输'} 第{res.get('stage', 0) + 1}场 搜索 {res['searched']} 次、改选 {res['changed']} 次、推演 {res['sims']} 次"
              f"（{time.time() - t0:.0f} 秒）", flush=True)

    M.run(a, items, net, a.device, on_item=on_item, labels=labels)


# ---------- distill：用标签微调策略输出

def _load_labels(paths):
    rows = []
    for p in paths:
        with open(p, "rb") as f:
            while True:
                try:
                    rec = pickle.load(f)
                except EOFError:
                    break
                except Exception as e:      # 进程被强停时最后一条可能只写了一半
                    print(f"{p}：末尾一条读不出（{type(e).__name__}），跳过", flush=True)
                    break
                for r in rec["rows"]:
                    r["item"] = rec["id"]
                    rows.append(r)
    return rows


def _target(np, r, kind, lam):
    p = r["p"].astype(np.float64)
    if kind == "hard":
        t = np.zeros_like(p); t[r["best"]] = 1.0
        return t
    t = p.copy()
    q0 = r["q"][r["act"]][0]
    for c, (m, _) in r["q"].items():
        t[c] = p[c] * math.exp(max(-20.0, min(20.0, (m - q0) / lam)))
    return t / t.sum()


def cmd_distill(a) -> None:
    import numpy as np
    import torch
    from sts2rl.actor import MODE_COMBAT, MODE_RUN
    from sts2rl.features import Ragged, collate_ragged
    import search_diag as SD
    torch.manual_seed(0)
    paths = sorted(Path(a.labels).glob("*.pkl")) if Path(a.labels).is_dir() else [Path(a.labels)]
    rows = _load_labels(paths)
    val = [r for r in rows if a.val_mod and r["item"] % a.val_mod == 0]       # 按题留出（同一道题的决策不跨训练、留出）
    tr = [r for r in rows if not (a.val_mod and r["item"] % a.val_mod == 0)]
    print(f"标签 {len(rows)} 条（训练 {len(tr)}、留出 {len(val)}），改选的 {sum(r['best'] != r['act'] for r in rows)} 条；目标 {a.target}", flush=True)
    net, blob = SD._load(a, torch)
    dev = torch.device(a.device)
    net.eval()       # 不开 dropout 一类的训练期随机性；梯度照常
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.0)

    def forward(batch, mode):
        logits, _ = net(collate_ragged(Ragged.from_obs([r["obs"] for r in batch], np.full(len(batch), mode))).to(dev))
        return torch.log_softmax(logits.float(), -1)

    def metrics(rs):
        agree_best = agree_act = n_ch = n_un = 0
        kl = 0.0
        with torch.no_grad():
            for j in range(0, len(rs), 256):
                b = rs[j: j + 256]
                lp = forward(b, MODE_COMBAT).cpu().numpy()
                for r, l in zip(b, lp):
                    n = r["obs"].num_actions
                    arg = int(np.argmax(l[:n]))
                    if r["best"] != r["act"]:
                        n_ch += 1; agree_best += arg == r["best"]
                    else:
                        n_un += 1; agree_act += arg == r["act"]
                    p = r["p"][:n].astype(np.float64)
                    kl += float(np.sum(np.where(p > 0, p * (np.log(np.maximum(p, 1e-12)) - l[:n]), 0.0)))
        return {"changed": n_ch, "argmax_eq_search_on_changed": round(agree_best / max(1, n_ch), 4),
                "unchanged": n_un, "argmax_eq_greedy_on_unchanged": round(agree_act / max(1, n_un), 4), "kl_from_orig": round(kl / max(1, len(rs)), 5)}

    log = {"before": {"train": metrics(tr), "val": metrics(val)}}
    print("微调前", log["before"], flush=True)
    t0 = time.time()
    step = 0
    for ep in range(a.epochs):
        random.Random(ep).shuffle(tr)
        for j in range(0, len(tr), a.batch):
            b = tr[j: j + a.batch]
            loss = 0.0
            for mode in (MODE_COMBAT, MODE_RUN):
                lp = forward(b, mode)
                T = torch.zeros_like(lp); P = torch.zeros_like(lp)
                for k, r in enumerate(b):
                    n = r["obs"].num_actions
                    T[k, :n] = torch.from_numpy(_target(np, r, a.target, a.lam)).float()
                    P[k, :n] = torch.from_numpy(r["p"][:n]).float()
                ce = -torch.where(T > 0, T * lp, torch.zeros_like(lp)).sum(-1).mean()
                kl = torch.where(P > 0, P * (torch.log(P.clamp_min(1e-12)) - lp), torch.zeros_like(lp)).sum(-1).mean()
                loss = loss + ce + (a.beta * kl if a.target == "hard" else 0.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            step += 1
        print(f"第 {ep + 1} 遍完，{step} 次更新，{time.time() - t0:.0f} 秒", flush=True)
    log["after"] = {"train": metrics(tr), "val": metrics(val)}
    log["params"] = {k: getattr(a, k) for k in ("target", "lam", "beta", "lr", "epochs", "batch")}
    print("微调后", log["after"], flush=True)
    out = dict(blob)
    out["state"] = {k: v.detach().cpu() for k, v in net.state_dict().items()}
    out["note"] = f"搜索蒸馏 {a.target}（{len(tr)} 条标签，{step} 次更新），基于第 {blob.get('version')} 版"
    torch.save(out, a.out)
    Path(a.out).with_suffix(".json").write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")


# ---------- combine：合成“只在第三幕 Boss 用搜索”的整局通关率

def cmd_combine(a) -> None:
    exam = {it["id"]: it for it in json.loads(Path(a.exam).read_text(encoding="utf-8"))["items"]}
    greedy = {r["id"]: r for r in (json.loads(l) for l in open(a.greedy, encoding="utf-8") if l.strip())}
    search = {}
    for f in sorted(Path(a.search).glob("*.jsonl")):
        for l in open(f, encoding="utf-8"):
            if l.strip():
                r = json.loads(l); search[r["id"]] = r
    ids = sorted(i for i in search if i in greedy and not greedy[i].get("stuck"))
    n = len(ids)
    gw = sum(greedy[i]["win"] for i in ids); sw = sum(search[i]["win"] for i in ids)
    only_s = sum(search[i]["win"] and not greedy[i]["win"] for i in ids); only_g = sum(greedy[i]["win"] and not search[i]["win"] for i in ids)
    d = [search[i]["win"] - greedy[i]["win"] for i in ids]
    m = sum(d) / n; se = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1) / n)
    k = min(only_s, only_g); p = min(1.0, 2 * sum(math.comb(only_s + only_g, j) for j in range(k + 1)) / 2 ** (only_s + only_g)) if only_s + only_g else 1.0
    act_win = sum(exam[i]["from_win"] for i in ids)
    allg = [greedy[i]["win"] for i in exam if i in greedy and not greedy[i].get("stuck")]
    # 到达第三幕第一个 Boss 的比例：复测千种子（含评测复用的局）
    runs = [json.loads(l) for f in sorted(Path(a.retest).glob("runs_s*.jsonl")) for l in open(f, encoding="utf-8") if l.strip()]
    cmp_ = json.loads((Path(a.retest) / "compare.json").read_text(encoding="utf-8"))
    me = next(x for x in cmp_["models"] if str(x["version"]) == str(runs[0]["version"]))
    reach = me["reach_a3_boss1"]
    out = {"n": n, "greedy_cond": round(gw / n, 4), "search_cond": round(sw / n, 4), "diff_cond": round(m, 4), "se_cond": round(se, 4),
           "only_search": only_s, "only_greedy": only_g, "p": round(p, 4),
           "actual_run_cond_same_items": round(act_win / n, 4), "greedy_cond_all_items": round(sum(allg) / max(1, len(allg)), 4), "all_items": len(allg),
           "reach_a3_boss1": reach, "run_win_retest": me["win_rate"],
           "run_diff": round(reach * m, 4), "run_se": round(reach * se, 4), "run_win_search_est": round(me["win_rate"] + reach * m, 4),
           "searched": sum(search[i]["searched"] for i in ids), "changed": sum(search[i]["changed"] for i in ids), "sims": sum(search[i]["sims"] for i in ids)}
    print(json.dumps(out, ensure_ascii=False, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


def cmd_fullcmp(a) -> None:
    """整局逐种子配对：新检查点的复测式结果目录（ability_test 的 runs_s*.jsonl，可多个）对原版本的千种子复测（同一批种子）。"""
    def load(dirs):
        out = {}
        for d in dirs:
            for f in sorted(Path(d).glob("runs_s*.jsonl")):
                for l in open(f, encoding="utf-8"):
                    if l.strip():
                        r = json.loads(l); out[r["seed"]] = r
        return out
    new, old = load(a.new), load([a.base])
    seeds = sorted(s_ for s_ in new if s_ in old and not new[s_].get("stuck") and not old[s_].get("stuck"))
    n = len(seeds)
    wn = sum(new[s_]["win"] for s_ in seeds); wo = sum(old[s_]["win"] for s_ in seeds)
    x = sum(new[s_]["win"] and not old[s_]["win"] for s_ in seeds); y = sum(old[s_]["win"] and not new[s_]["win"] for s_ in seeds)
    k = min(x, y); p = min(1.0, 2 * sum(math.comb(x + y, j) for j in range(k + 1)) / 2 ** (x + y)) if x + y else 1.0
    same = sum(new[s_]["floor"] == old[s_]["floor"] and new[s_]["win"] == old[s_]["win"] for s_ in seeds)
    out = {"n": n, "win_new": wn, "win_base": wo, "rate_new": round(wn / max(1, n), 4), "rate_base": round(wo / max(1, n), 4),
           "only_new": x, "only_base": y, "p": round(p, 4), "same_outcome_and_floor": same}
    print(json.dumps(out, ensure_ascii=False))
    if a.json:
        Path(a.json).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("exam"); e.add_argument("dirs", nargs="+"); e.add_argument("--out", required=True)
    e.add_argument("--prefix", default="SX"); e.add_argument("--shuffle", type=int, default=1)
    s = sub.add_parser("search")
    s.add_argument("--ckpt", required=True); s.add_argument("--exam", required=True); s.add_argument("--out", required=True); s.add_argument("--labels", required=True)
    s.add_argument("--shard", default="0/1"); s.add_argument("--ids-mod", default=None); s.add_argument("--limit", type=int, default=0)
    s.add_argument("--envs", type=int, default=4); s.add_argument("--live", type=int, default=2)
    s.add_argument("--device", default="cpu"); s.add_argument("--threads", type=int, default=1)
    s.add_argument("--tau", type=float, default=0.9); s.add_argument("--k", type=int, default=4); s.add_argument("--pmin", type=float, default=0.02)
    s.add_argument("--sims0", type=int, default=4); s.add_argument("--rounds", type=int, default=2)
    s.add_argument("--paired", action="store_true", default=True); s.add_argument("--z", type=float, default=2.0); s.add_argument("--leaf", type=int, default=12)
    s.add_argument("--deadline", type=float, default=0, help="Unix 时刻；过了就不再开新题，手上的题做完为止")
    d = sub.add_parser("distill")
    d.add_argument("--ckpt", required=True); d.add_argument("--labels", required=True); d.add_argument("--out", required=True)
    d.add_argument("--target", choices=("soft", "hard"), default="soft"); d.add_argument("--lam", type=float, default=0.1); d.add_argument("--beta", type=float, default=0.5)
    d.add_argument("--lr", type=float, default=2e-5); d.add_argument("--epochs", type=int, default=2); d.add_argument("--batch", type=int, default=128)
    d.add_argument("--device", default="cuda"); d.add_argument("--val-mod", type=int, default=10, help="题号 %% 这个数 == 0 的题留出（0 为不留出）")
    c = sub.add_parser("combine")
    c.add_argument("--exam", required=True); c.add_argument("--greedy", required=True); c.add_argument("--search", required=True); c.add_argument("--retest", required=True)
    c.add_argument("--json", default=None)
    f = sub.add_parser("fullcmp"); f.add_argument("--new", nargs="+", required=True); f.add_argument("--base", required=True); f.add_argument("--json", default=None)
    a = ap.parse_args()
    {"fullcmp": cmd_fullcmp, "exam": cmd_exam, "search": cmd_search, "distill": cmd_distill, "combine": cmd_combine}[a.cmd](a)


if __name__ == "__main__":
    main()
