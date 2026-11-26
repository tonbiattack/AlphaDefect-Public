"""路线摘要（观测格式版本 7）的核对：本机引擎打开 route_summary，用模型打若干整局（地图上随机选路，其余决策贪心），
每次地图决策按观测里的地图连线暴力枚举全部路线，与引擎给出的路线摘要逐项比对；同时核对可达范围（地图节点特征 5、9–15）
和“前往地图节点”候选的对象 B。最后报告地图决策的引擎耗时（与关闭时对比）。

    .venv/bin/python -I tools/check_route_summary.py --ckpt 检查点.pt --runs 24
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from sts2env import VecEnv  # noqa: E402
from sts2env.protocol import ActionKind, DecisionKind, TokenKind  # noqa: E402
from sts2rl.actor import MODE_RUN  # noqa: E402
from sts2rl.catalog import Catalog  # noqa: E402
from sts2rl.features import Ragged, collate_ragged  # noqa: E402
from sts2rl.model import ModelConfig, PolicyValueNet, load_state  # noqa: E402

KINDS = ["Monster", "Elite", "Unknown", "Shop", "RestSite"]


def brute(o, vocab, fire0: int) -> tuple[dict, list[str]]:
    """按地图连线暴力枚举：返回 {(行, 列): 16 项期望特征}（每个现在可前往的节点）与可达性的错误。"""
    ti, tf = o.tok_ints, o.tok_feats
    nodes = {}
    for k in np.flatnonzero(ti[:, 0] == TokenKind.MAP_NODE):
        f = tf[k]
        nodes[(int(f[0]), int(f[1]))] = dict(k=k, type=vocab[int(ti[k, 1])].split(":")[1], f=f)
    def children(rc):
        r, _ = rc
        f = nodes[rc]["f"]
        if r == min(x[0] for x in nodes):       # 每幕的起点连着第一行的全部节点（观测里的连线最多列 3 个）
            return [x for x in nodes if x[0] == r + 1]
        out = [(r + 1, int(c) - 1) for c in f[6:9] if c > 0]
        # 连到 Boss：Boss 的行号不一定是 r + 1，按类型找
        res = []
        for x in out:
            if x in nodes:
                res.append(x)
            else:
                b = [rc2 for rc2, v in nodes.items() if v["type"] == "Boss" and rc2[0] > r]
                res.append(min(b))
        return res
    def desc(rc, seen=None):
        seen = set() if seen is None else seen
        if rc in seen:
            return seen
        seen.add(rc)
        for c in children(rc):
            desc(c, seen)
        return seen
    errs = []
    nxt = sorted([rc for rc, v in nodes.items() if v["f"][3] == 1], key=lambda x: (x[1], x[0]))[:7]
    reach = set().union(*[desc(t) for t in nxt]) if nxt else set()
    for rc, v in nodes.items():
        f = v["f"]
        if int(f[5]) != (rc in reach):
            errs.append(f"可达 {rc}")
        for k, t in enumerate(nxt):
            if int(f[9 + k]) != (rc in desc(t)):
                errs.append(f"从第 {k} 个出发可达 {rc}")
    bosses = {rc for rc, v in nodes.items() if v["type"] == "Boss"}
    first_boss = min(bosses) if bosses else None
    exp = {}
    for t in nxt:
        routes = []
        def dfs(rc, path):
            if rc == first_boss or rc in bosses or not children(rc):
                routes.append(list(path)); return
            path.append(nodes[rc]["type"])
            for c in children(rc):
                dfs(c, path)
            path.pop()
        dfs(t, [])
        cnt = [[r.count(x) for x in KINDS] for r in routes]
        tre = [r.count("Treasure") for r in routes]
        def cover(r, f):
            cov, allok = 0, True
            for x in r:
                if x == "RestSite": f = 1
                elif x == "Elite":
                    cov += f; allok &= bool(f); f = 0
            return cov, allok
        cv = [cover(r, fire0) for r in routes]
        rbe = [-1] * 4
        for c in cnt:
            e = min(c[1], 3); rbe[e] = max(rbe[e], c[4])
        allc = [c[1] for c, (_, ok) in zip(cnt, cv) if ok]
        exp[t] = dict(feat=[first_boss[0] - t[0], math.log2(len(routes) + 1)] + [max(c[i] for c in cnt) for i in range(5)]
                      + [min(c[0] for c in cnt), min(c[1] for c in cnt), min(c[4] for c in cnt)]
                      + [max(c for c, _ in cv), max(allc) if allc else -1] + rbe,
                      treasure=(min(tre), max(tre)), n=len(routes))
    return exp, errs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--runs", type=int, default=24)
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--off", action="store_true", help="关掉路线摘要只跑对局，用来对比引擎耗时（不核对）")
    ap.add_argument("--log", default="/dev/null", help="引擎标准错误写到这里（退出时有各步平均耗时 [timing]，地图决策的编码为 encode_map）")
    a = ap.parse_args()
    blob = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    net = PolicyValueNet(ModelConfig(**blob["config"]["model"])).eval()
    load_state(net, blob["state"])
    vocab = Catalog().vocab
    rng = np.random.default_rng(0)
    stats = dict(maps=0, records=0, bad=0, reach_err=0, b_err=0, treasure_var=0, max_routes=0)
    worst = []
    with VecEnv(a.envs, log=open(a.log, "ab")) as env:
        todo = list(range(a.runs))
        fire = {}
        def launch(i):
            if todo:
                k = todo.pop()
                fire[i] = (None, 0)
                env.send(i, {"op": "reset", "mode": "run", "character": "DEFECT", "ascension": 10, "seed": f"ROUTE{k:03d}", "route_summary": not a.off})
                return True
            return False
        live = {i for i in range(a.envs) if launch(i)}
        while live:
            batch = []
            for i, o in env.recv_ready(timeout=60):
                if isinstance(o, Exception) or o.done or o.stuck or o.num_actions == 0:
                    if not launch(i):
                        live.discard(i)
                    continue
                batch.append((i, o))
            if not batch:
                continue
            with torch.no_grad():
                logits, _ = net(collate_ragged(Ragged.from_obs([o for _, o in batch], np.array([MODE_RUN] * len(batch)))).to("cpu"))
            for (i, o), lg in zip(batch, logits):
                a_ = int(lg[: o.num_actions].argmax())
                if o.decision == DecisionKind.MAP and a.off:
                    moves = [c for c in range(o.num_actions) if o.cand_ints[c][0] == ActionKind.MAP_MOVE]
                    a_ = int(rng.choice(moves))
                elif o.decision == DecisionKind.MAP:
                    act, f0 = fire[i]
                    if act != o.act:
                        f0 = 0
                    stats["maps"] += 1
                    exp, errs = brute(o, vocab, f0)
                    stats["reach_err"] += len(errs)
                    rs = np.flatnonzero(o.tok_ints[:, 0] == TokenKind.ROUTE_SUMMARY)
                    got = {}
                    for c in range(o.num_actions):
                        ci = o.cand_ints[c]
                        if ci[0] == ActionKind.MAP_MOVE:
                            node = o.tok_feats[ci[1]]
                            rc = (int(node[0]), int(node[1]))
                            if ci[2] < 0 or o.tok_ints[ci[2], 0] != TokenKind.ROUTE_SUMMARY:
                                stats["b_err"] += 1; continue
                            got[rc] = o.tok_feats[ci[2]]
                    if len(rs) != len(exp):
                        stats["bad"] += 1; worst.append(f"记录数 {len(rs)} ≠ {len(exp)}")
                    for rc, e in exp.items():
                        stats["records"] += 1
                        stats["max_routes"] = max(stats["max_routes"], e["n"])
                        stats["treasure_var"] += e["treasure"][0] != e["treasure"][1]
                        g = got.get(rc)
                        if g is None or not np.allclose(g, e["feat"], atol=1e-4):
                            stats["bad"] += 1
                            if len(worst) < 5:
                                worst.append(f"{rc} 引擎 {None if g is None else g.round(2).tolist()} 枚举 {np.round(e['feat'], 2).tolist()}")
                    # 随机选路，并记下走过的是休息处还是精英（决定下一个精英前有没有火堆）
                    moves = [c for c in range(o.num_actions) if o.cand_ints[c][0] == ActionKind.MAP_MOVE]
                    a_ = int(rng.choice(moves))
                    t = vocab[int(o.tok_ints[o.cand_ints[a_][1], 1])].split(":")[1]
                    fire[i] = (o.act, 1 if t == "RestSite" else 0 if t == "Elite" else f0)
                env.send(i, {"op": "step", "action": a_})
    print(stats)
    for w in worst:
        print("  ", w)


if __name__ == "__main__":
    main()
