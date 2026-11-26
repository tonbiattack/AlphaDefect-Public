"""战斗搜索第 0 步诊断（2026-10-30，研究报告/2026-10-30_战斗搜索方案调查 第 4.0 节）。只评测，不碰训练。

在双 Boss 诊断卷（tools/boss3_exam.py 的 boss3_exam.json，第三幕连打两个 Boss）上，用现模型：
  search   推演式改选（tools/mcts_eval.py 的搜索，改成连打口径的打分），逐题记输赢、到达第二场、改选次数；
           每个搜索过的决策另记一份现场（观测摘要、贪心项与最终选择的动作键、概率），供 0d 分类和以后的触发门槛用
  value    价值头校准（0c）：沿贪心轨迹取若干局面，各用重洗后的贪心推演打到底，与价值头比较
           （不带特权 = 搜索里实际用的；带特权 = 训练时价值头看到的）
  compare  把 search 的结果与贪心结果（boss3_exam.py run 的 jsonl）逐题配对

  .venv/bin/python -I tools/search_diag.py search --ckpt 检查点.pt --exam boss3_exam.json --out 结果.jsonl \
        --shard 0/4 --envs 4 --leaf 12 --sims0 4 --rounds 2 --paired --z 2 [--limit 60] [--ids-mod 3/0]
  .venv/bin/python -I tools/search_diag.py value --ckpt 检查点.pt --exam boss3_exam.json --out 结果.jsonl --ids-mod 3/0 --every 8 --worlds 12
  .venv/bin/python -I tools/search_diag.py report 结果目录   （汇总 0a–0d：目录里有 greedy.jsonl、search_*_s*.jsonl、value_c_s*.jsonl）
  python3 tools/search_diag.py compare 贪心结果.jsonl 搜索结果1.jsonl [搜索结果2.jsonl ...]

打分与连打练习的结算同一尺度（决策记录 187；win_reward 1 + boss_win_bonus 0.5，boss_dmg_coef 0.5）：
两场都赢 1.5；输了 −1 + 0.25 × (打过的场数 + 输的那场打掉的敌人生命比例)。价值叶子用战斗价值头的输出，同一尺度。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "tools"))


def terminal(r: dict) -> float:
    if r["win"]:
        return 1.5
    return -1.0 + 0.25 * (r.get("stage", 0) + 1.0 - r.get("enemy_left", 1.0))


def _install(M, np):
    """把 mcts_eval 的几处换成连打口径：敌人生命用比例（每场各自的分母）、结果带上已打赢的场数、打分用 terminal。"""
    orig_enemy_hp = M.enemy_hp

    def enemy_hp(o):
        hp, mx = orig_enemy_hp(o)
        return (hp / mx, 1.0) if mx > 0 else (0.0, 0.0)

    def outcome(o, t=None):
        r = {"win": bool(o.victory) and not o.stuck, "hp": int(o.hp), "max_hp": max(1, int(o.max_hp)), "stage": int(o.header[23])}
        if t is not None and t.emax:
            r["enemy_left"] = round(min(1.0, t.ehp / t.emax), 4)
        return r

    M.enemy_hp = enemy_hp
    M.outcome = outcome
    M.score = terminal
    M.score_value = lambda r, hp0: r["value"] if "value" in r else terminal(r)


def _load(a, torch):
    from sts2rl.model import ModelConfig, PolicyValueNet, load_state
    blob = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    net = PolicyValueNet(ModelConfig(**blob["config"]["model"])).eval().to(a.device)
    load_state(net, blob["state"])
    return net, blob


def _items(a) -> list[tuple[int, dict]]:
    exam = json.loads(Path(a.exam).expanduser().read_text(encoding="utf-8"))["items"]
    items = [(it["id"], it) for it in exam]
    if a.ids_mod:
        m, r = map(int, a.ids_mod.split("/"))
        items = [x for x in items if x[0] % m == r]
    if a.limit:
        items = items[: a.limit]
    k, n = map(int, a.shard.split("/"))
    return [x for j, x in enumerate(items) if j % n == k]


def _done_ids(out: Path) -> set[int]:
    done = set()
    for f in out.parent.glob(out.stem.rsplit("_s", 1)[0] + "_s*.jsonl"):
        for l in open(f, encoding="utf-8"):
            if l.strip():
                done.add(json.loads(l)["id"])
    return done


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if hasattr(x, "item"):
        return x.item()
    return x


def _make_fast_search(M, np, azmcts):
    """mcts_eval.search_item 的省算力版：--dedupe 合并动作键相同的候选；--early skip/race 提前淘汰（规则同 _policy，离线对照见 simulate）。"""
    def search_item(item, args, seed_base, labels=None):
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
            if o.num_actions > 1 and p[act] < args.tau:
                if args.dedupe:
                    keys = [repr(k) for k in azmcts.action_keys(o)]
                    grp: dict = {}
                    for c in order:
                        g = grp.setdefault(keys[int(c)], [0.0, int(c)])
                        g[0] += float(p[c])
                    reps = [g[1] for g in sorted(grp.values(), key=lambda g: -g[0]) if g[0] >= args.pmin][:6]
                    if act not in reps:
                        reps = [act] + reps[:5]
                    reps = reps[: args.k]
                    cands = reps if act in reps else [act] + reps[: args.k - 1]
                else:
                    cands = [int(c) for c in order[: args.k] if p[c] >= args.pmin] or [act]
                if len(cands) > 1:
                    n_search += 1
                    sc: dict = {c: {} for c in cands}
                    alive, n_new, used = list(cands), args.sims0, 0
                    for _ in range(args.rounds):
                        seeds = [seed_base + 7919 * (len(prefix) + 1) + used + j for j in range(n_new)]
                        used += n_new
                        tasks = [M.Task(spec, prefix, "rollout", seed=s, first=c, owner=(c, s)) for c in alive for s in seeds]
                        results = yield ("tasks", tasks)
                        n_sims += len(tasks)
                        for t, r in zip(tasks, results):
                            c, s_ = t.owner
                            sc[c][s_] = M.score_value(r if isinstance(r, dict) else lose, 0.0)
                        if args.early:
                            keep = [act]
                            for c in alive:
                                if c == act:
                                    continue
                                common = [s_ for s_ in sc[c] if s_ in sc[act]]
                                d = [sc[c][s_] - sc[act][s_] for s_ in common]
                                m = sum(d) / len(d)
                                sd = math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1)) if len(d) > 1 else 0.0
                                if (args.early == "skip" and m > 0) or (args.early == "race" and m + args.zf * sd / math.sqrt(len(d)) >= 0):
                                    keep.append(c)
                            alive = keep
                            if len(alive) == 1:
                                break
                        if len(alive) <= 2:
                            continue
                        mean = lambda c: sum(sc[c].values()) / len(sc[c])
                        rest = sorted((c for c in alive if c != act), key=lambda c: (-mean(c), -p[c]))
                        alive = [act] + rest[: max(1, math.ceil(len(alive) / 2) - 1)]
                        n_new *= 2
                    best, best_m = act, 0.0
                    for c in sc:
                        if c == act:
                            continue
                        common = [s_ for s_ in sc[c] if s_ in sc[act]]
                        if len(common) < 2:
                            continue
                        d = [sc[c][s_] - sc[act][s_] for s_ in common]
                        m = sum(d) / len(d)
                        sd = math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1))
                        if m > best_m and m > args.z * sd / math.sqrt(len(d)):
                            best, best_m = c, m
                    n_changed += best != act
                    if labels is not None:
                        labels.append((o, act, best, float(p[act])))
                    act = best
            prefix.append(act)
    return search_item


def cmd_search(a) -> None:
    import numpy as np
    import torch
    import mcts_eval as M
    from sts2rl import azmcts
    from sts2rl.replay import describe
    torch.set_num_threads(a.threads)
    _install(M, np)
    M.SCORE, M.LEAF = "dmg", a.leaf
    if a.dedupe or a.early:
        M.search_item = _make_fast_search(M, np, azmcts)
    net, blob = _load(a, torch)
    items = [x for x in _items(a) if x[0] not in _done_ids(Path(a.out))]
    print(f"版本 {blob.get('version')}，{len(items)} 道题待做；leaf={a.leaf} sims0={a.sims0} rounds={a.rounds} k={a.k} tau={a.tau} paired={a.paired} z={a.z}", flush=True)
    meta = Path(a.out).with_suffix(".meta.json")
    if not meta.exists():
        meta.write_text(json.dumps({"version": blob.get("version"), "params": {k: getattr(a, k) for k in ("leaf", "sims0", "rounds", "k", "tau", "pmin", "paired", "z", "shard", "ids_mod", "limit", "dedupe", "early", "zf")},
                                    "start": round(time.time())}, ensure_ascii=False), encoding="utf-8")
    labels: dict = {}
    out = open(a.out, "a", encoding="utf-8")
    done_n = [0]

    def on_item(i, res):
        rows = []
        for (o, act, best, pact) in labels.pop(i, []):
            keys = azmcts.action_keys(o)
            pr = np.zeros(o.num_actions, np.float32); pr[act] = pact
            rows.append({"act": int(act), "best": int(best), "p": round(float(pact), 4), "n": int(o.num_actions),
                         "key_act": _jsonable(keys[act]), "key_best": _jsonable(keys[best]), "obs": _jsonable(describe(o, act, pr))})
        line = {"id": i, "win": int(res["win"]), "stage": int(res.get("stage", 0)), "hp": res.get("hp"), "decisions": res["decisions"], "searched": res["searched"],
                "sims": res["sims"], "changed": res["changed"], "root": res.get("root"), "rows": rows, "t": round(time.time())}
        out.write(json.dumps(line, ensure_ascii=False) + "\n"); out.flush()
        done_n[0] += 1
        print(f"  题 {i}：{'赢' if res['win'] else '输'} 第{res.get('stage', 0) + 1}场 搜索 {res['searched']} 次、改选 {res['changed']} 次、推演 {res['sims']} 次（已做 {done_n[0]}）", flush=True)

    M.run(a, items, net, a.device, on_item=on_item, labels=labels)


def cmd_value(a) -> None:
    import numpy as np
    import torch
    import mcts_eval as M
    from sts2env import VecEnv as _VE
    from sts2rl.actor import MODE_COMBAT
    from sts2rl.features import Ragged, collate_ragged
    torch.set_num_threads(a.threads)
    _install(M, np)
    M.LEAF = 0

    class PrivVec(_VE):
        """spec 里带 _priv 的请求加上 privileged（引擎忽略 spec 里的未知字段）。"""
        def send(self, i, req):
            if isinstance(req, dict) and req.get("op") == "reset" and req.get("spec", {}).get("_priv"):
                req = dict(req, privileged=True)
            return super().send(i, req)
    M.VecEnv = PrivVec
    net, blob = _load(a, torch)
    items = [x for x in _items(a) if x[0] not in _done_ids(Path(a.out))]
    print(f"版本 {blob.get('version')}，{len(items)} 道题；每 {a.every} 个决策取一个局面，{a.worlds} 个世界", flush=True)
    out = open(a.out, "a", encoding="utf-8")

    def values(obs):
        with torch.no_grad():
            _, v = net(collate_ragged(Ragged.from_obs(obs, np.array([MODE_COMBAT] * len(obs)))).to(a.device))
        return [float(x) for x in v.float().cpu()]

    def value_item(it):
        spec, prefix, step = it["spec"], [], 0
        recs = []
        while True:
            (o,) = yield ("tasks", [M.Task(spec, prefix, "obs")])
            if isinstance(o, M.EngineError) or o.done or o.stuck or o.num_actions == 0 or len(prefix) > M.MAX_STEPS:
                return recs
            p = yield ("policy", o)
            if step % a.every == 0 and o.num_actions > 1:
                seeds = [911 * (len(prefix) + 1) + j for j in range(a.worlds)]
                pspec = dict(spec, _priv=True)
                tasks = [M.Task(spec, prefix, "rollout", seed=s, owner=("r", s)) for s in seeds]
                tasks += [M.Task(spec, prefix, "obs", seed=s, owner=("o", s)) for s in seeds]
                tasks += [M.Task(pspec, prefix, "obs", seed=s, owner=("p", s)) for s in seeds]
                res = yield ("tasks", tasks)
                roll = [terminal(r) if isinstance(r, dict) else terminal({"win": False}) for r in res[: a.worlds]]
                obs_np = res[a.worlds: 2 * a.worlds]
                obs_p = res[2 * a.worlds:]
                vnp = values([x for x in obs_np if not isinstance(x, M.EngineError)])
                vp = values([x for x in obs_p if not isinstance(x, M.EngineError)])
                vroot = values([o])[0]
                g = o.tok_feats[np.flatnonzero(o.tok_ints[:, 0] == M.TokenKind.GLOBAL)[0]]
                recs.append({"step": step, "stage": int(o.header[23]), "hp": int(o.hp), "max_hp": int(o.max_hp),
                             "roll": [round(x, 4) for x in roll], "v_np": [round(x, 4) for x in vnp], "v_p": [round(x, 4) for x in vp], "v_root_np": round(vroot, 4)})
            step += 1
            prefix.append(int(np.argmax(p)))

    def on_item(i, recs):
        out.write(json.dumps({"id": i, "states": recs, "t": round(time.time())}, ensure_ascii=False) + "\n"); out.flush()
        print(f"  题 {i}：{len(recs)} 个局面", flush=True)

    # mcts_eval.run 的控制器是 search_item，这里换成 value_item
    M.search_item = lambda item, args, seed_base, labels=None: value_item(item)
    M.run(a, items, net, a.device, on_item=on_item, labels=None)


def cmd_table(a) -> None:
    """省算力优化用：在贪心轨迹上取若干个“会触发搜索”的局面，每个局面对一组候选各用同一批随机世界推演，
    记下 候选 × 世界 的得分表。之后离线（simulate）比较各种省算力规则，不用每改一条规则就重跑引擎。"""
    import numpy as np
    import torch
    import mcts_eval as M
    from sts2rl import azmcts
    torch.set_num_threads(a.threads)
    _install(M, np)
    M.LEAF = a.leaf
    net, blob = _load(a, torch)
    items = [x for x in _items(a) if x[0] not in _done_ids(Path(a.out))]
    print(f"版本 {blob.get('version')}，{len(items)} 道题；每 {a.every} 个会触发搜索的局面取一个，{a.seeds} 个世界", flush=True)
    out = open(a.out, "a", encoding="utf-8")

    def table_item(it, seed_base):
        spec, prefix, n_s, recs = it["spec"], [], 0, []
        while True:
            (o,) = yield ("tasks", [M.Task(spec, prefix, "obs")])
            if isinstance(o, M.EngineError) or o.done or o.stuck or o.num_actions == 0 or len(prefix) > M.MAX_STEPS:
                return recs
            p = yield ("policy", o)
            order = np.argsort(-p)
            act = int(order[0])
            if o.num_actions > 1 and p[act] < a.tau:
                keys = [json.dumps(_jsonable(k)) for k in azmcts.action_keys(o)]
                top4 = [int(c) for c in order[:4] if p[c] >= a.pmin] or [act]
                grp: dict[str, list] = {}
                for c in order:
                    g = grp.setdefault(keys[int(c)], [0.0, int(c)])
                    g[0] += float(p[c])
                reps = [g[1] for g in sorted(grp.values(), key=lambda g: -g[0]) if g[0] >= a.pmin][:6]
                if act not in reps:
                    reps = [act] + reps[:5]
                union = list(dict.fromkeys(top4 + reps))
                if len(top4) > 1 or len(reps) > 1:
                    n_s += 1
                    if n_s % a.every == 0:
                        seeds = [seed_base + 7919 * (len(prefix) + 1) + j for j in range(a.seeds)]
                        tasks = [M.Task(spec, prefix, "rollout", seed=s, first=c, owner=(c, j)) for c in union for j, s in enumerate(seeds)]
                        res = yield ("tasks", tasks)
                        sc: dict[int, list] = {c: [0.0] * a.seeds for c in union}
                        for t, r in zip(tasks, res):
                            c, j = t.owner
                            sc[c][j] = M.score_value(r if isinstance(r, dict) else {"win": False, "hp": 0, "max_hp": 1}, 0.0)
                        recs.append({"step": len(prefix), "act": act, "p": {str(int(c)): round(float(p[c]), 4) for c in union}, "key": {str(c): keys[c] for c in union},
                                     "top4": top4, "reps": reps, "sc": {str(c): [round(x, 4) for x in v] for c, v in sc.items()}})
            prefix.append(act)

    def on_item(i, recs):
        out.write(json.dumps({"id": i, "roots": recs}, ensure_ascii=False) + "\n"); out.flush()
        print(f"  题 {i}：{len(recs)} 个局面", flush=True)

    M.search_item = lambda item, args, seed_base, labels=None: table_item(item, seed_base)
    M.run(a, items, net, a.device, on_item=on_item, labels=None)


def _policy(R: dict, dedupe: bool, k: int, early: str, zf: float, sims0: int = 4, rounds: int = 2, z: float = 2.0, pmin: float = 0.02):
    """离线复现搜索的一次决策（与 mcts_eval.search_item 同一规则，随机世界取表里的前几列）。返回 (选的候选下标, 推演次数)。
    dedupe：动作键相同的候选只留一个；early：""（原做法）、"skip"（第一轮后没有候选的平均差为正就不做第二轮）、"race"（平均差 + zf 个标准误仍为负的候选淘汰）。"""
    p = {int(c): v for c, v in R["p"].items()}
    key = {int(c): v for c, v in R["key"].items()}
    sc = {int(c): v for c, v in R["sc"].items()}
    act = R["act"]
    if dedupe:
        reps = [c for c in R["reps"] if c in sc][:k]
        cands = reps if act in reps else [act] + reps[: k - 1]
    else:
        cands = [c for c in R["top4"] if c in sc][:k] or [act]
    if len(cands) <= 1:
        return act, 0
    seen: dict[int, dict[int, float]] = {c: {} for c in cands}
    alive, n_new, used, cost = list(cands), sims0, 0, 0
    for _ in range(rounds):
        for j in range(used, used + n_new):
            for c in alive:
                seen[c][j] = sc[c][j]
        cost += n_new * len(alive)
        used += n_new
        if early:
            keep = [act]
            for c in alive:
                if c == act:
                    continue
                common = [j for j in seen[c] if j in seen[act]]
                d = [seen[c][j] - seen[act][j] for j in common]
                m = sum(d) / len(d)
                sd = math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1)) if len(d) > 1 else 0.0
                se = sd / math.sqrt(len(d))
                if (early == "skip" and m > 0) or (early == "race" and m + zf * se >= 0):
                    keep.append(c)
            alive = keep
            if len(alive) == 1:
                break
        if len(alive) <= 2:
            continue
        mean = lambda c: sum(seen[c].values()) / len(seen[c])
        rest = sorted((c for c in alive if c != act), key=lambda c: (-mean(c), -p[c]))
        alive = [act] + rest[: max(1, math.ceil(len(alive) / 2) - 1)]
        n_new *= 2
    best, best_m = act, 0.0
    for c in seen:
        if c == act:
            continue
        common = [j for j in seen[c] if j in seen[act]]
        if len(common) < 2:
            continue
        d = [seen[c][j] - seen[act][j] for j in common]
        m = sum(d) / len(d)
        sd = math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1))
        if m > best_m and m > z * sd / math.sqrt(len(d)):
            best, best_m = c, m
    return best, cost


def cmd_simulate(a) -> None:
    roots = []
    for f in sorted(Path(a.dir).glob("table_s*.jsonl")):
        for l in open(f, encoding="utf-8"):
            if l.strip():
                for r in json.loads(l)["roots"]:
                    roots.append(r)
    n_seeds = len(next(iter(roots[0]["sc"].values())))
    print(f"{len(roots)} 个局面，每个候选 {n_seeds} 个世界；前 12 列给决策用，其余列只用来评估")
    held = range(12, n_seeds)
    variants = [("原做法（前 4 个下标、两轮）", dict(dedupe=False, k=4, early="", zf=0)),
                ("合并等价动作", dict(dedupe=True, k=4, early="", zf=0)),
                ("合并 + 第一轮没有正差就不做第二轮", dict(dedupe=True, k=4, early="skip", zf=0)),
                ("合并 + 淘汰（差 + 1 个标准误仍为负）", dict(dedupe=True, k=4, early="race", zf=1.0)),
                ("合并 + 淘汰（差 + 2 个标准误仍为负）", dict(dedupe=True, k=4, early="race", zf=2.0)),
                ("合并 + 只取 3 个候选 + 淘汰（1 个标准误）", dict(dedupe=True, k=3, early="race", zf=1.0)),
                ("合并 + 只取 3 个候选 + 第一轮没有正差就不做第二轮", dict(dedupe=True, k=3, early="skip", zf=0)),
                ("不合并 + 淘汰（1 个标准误）", dict(dedupe=False, k=4, early="race", zf=1.0))]
    base = [_policy(R, **variants[0][1]) for R in roots]
    kc = lambda R, c: R["key"][str(c)]
    print(f"{'规则':<44}{'每局面推演':>10}{'相对原做法':>10}{'改选率':>8}{'与原做法决策一致':>16}{'漏掉原做法的改选':>16}{'多出的改选':>10}{'留出世界上的平均增益':>20}")
    for name, kw in variants:
        res = [_policy(R, **kw) for R in roots]
        cost = sum(c for _, c in res) / len(roots)
        bcost = sum(c for _, c in base) / len(roots)
        ch = sum(1 for R, (c, _) in zip(roots, res) if kc(R, c) != kc(R, R["act"]))
        agree = sum(1 for R, (c, _), (b, _) in zip(roots, res, base) if kc(R, c) == kc(R, b))
        miss = sum(1 for R, (c, _), (b, _) in zip(roots, res, base) if kc(R, b) != kc(R, R["act"]) and kc(R, c) != kc(R, b))
        extra = sum(1 for R, (c, _), (b, _) in zip(roots, res, base) if kc(R, b) == kc(R, R["act"]) and kc(R, c) != kc(R, R["act"]))
        gain = sum(sum(R["sc"][str(c)][j] - R["sc"][str(R["act"])][j] for j in held) / len(held) for R, (c, _) in zip(roots, res)) / len(roots)
        print(f"{name:<40}{cost:>10.1f}{cost / bcost:>10.2f}{100 * ch / len(roots):>7.1f}%{100 * agree / len(roots):>15.1f}%{miss:>16}{extra:>10}{gain:>+20.4f}")
    # 动作键重复的情况
    dup = sum(1 for R in roots if len({kc(R, c) for c in R["top4"]}) < len(R["top4"]))
    one = sum(1 for R in roots if len(R["reps"]) <= 1)
    print(f"前 4 个候选里有动作键重复的局面 {dup}/{len(roots)}（{100 * dup / len(roots):.0f}%）；合并后只剩一个动作、不用搜索的 {one} 个")
    allk = sum(len(R["top4"]) for R in roots) / len(roots)
    print(f"原做法平均候选 {allk:.2f} 个；改选到的候选在概率排名里的位置（按下标、原做法）：" + str({r: sum(1 for R, (c, _) in zip(roots, base) if c != R['act'] and R['top4'].index(c) == r) for r in range(1, 4)}))


def cmd_compare(a) -> None:
    def load(p):
        d = {}
        for l in open(p, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                d[r["id"]] = r
        return d
    S = {}
    for p in a.search:
        S.update(load(p))
    G = load(a.greedy)
    ids = sorted(i for i in S if i in G and not G[i].get("stuck"))
    x = sum(S[i]["win"] and not G[i]["win"] for i in ids)
    y = sum(G[i]["win"] and not S[i]["win"] for i in ids)
    n = x + y
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, k) for k in range(min(x, y) + 1)) / 2 ** n)
    ws, wg = sum(S[i]["win"] for i in ids), sum(G[i]["win"] for i in ids)
    d = [S[i]["win"] - G[i]["win"] for i in ids]
    m = sum(d) / max(1, len(d))
    sd = math.sqrt(sum((v - m) ** 2 for v in d) / max(1, len(d) - 1))
    print(f"{len(ids)} 题：搜索 {ws}、贪心 {wg}，差 {100 * m:+.1f} ± {100 * sd / math.sqrt(max(1, len(d))):.1f} 个百分点；只搜索赢 {x}、只贪心赢 {y}，符号检验 p = {p:.3f}")
    s2s = sum(S[i]["stage"] >= 1 for i in ids); s2g = sum(G[i].get("stage2", 0) for i in ids)
    print(f"打到第二场：搜索 {s2s}、贪心 {s2g}；搜索 {sum(S[i]['searched'] for i in ids)} 次、改选 {sum(S[i]['changed'] for i in ids)} 次、推演 {sum(S[i]['sims'] for i in ids)} 次")


def _read(pattern: str, d: Path) -> dict[int, dict]:
    out = {}
    for f in sorted(d.glob(pattern)):
        for l in open(f, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                out[r["id"]] = r
    return out


def _sign_p(x: int, y: int) -> float:
    n = x + y
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, k) for k in range(min(x, y) + 1)) / 2 ** n)


def _pair(S: dict, G: dict, ids: list[int]) -> str:
    if not ids:
        return "无数据"
    d = [S[i]["win"] - G[i]["win"] for i in ids]
    m = sum(d) / len(d)
    sd = math.sqrt(sum((v - m) ** 2 for v in d) / max(1, len(d) - 1))
    x = sum(S[i]["win"] and not G[i]["win"] for i in ids); y = sum(G[i]["win"] and not S[i]["win"] for i in ids)
    return (f"{len(ids)} 题：搜索 {sum(S[i]['win'] for i in ids)}、贪心 {sum(G[i]['win'] for i in ids)}，差 {100 * m:+.1f} ± {100 * sd / math.sqrt(len(d)):.1f}，"
            f"只搜索赢 {x}、只贪心赢 {y}，p = {_sign_p(x, y):.3f}")


def _corr(a, b) -> float:
    n = len(a)
    if n < 3:
        return float("nan")
    ma, mb = sum(a) / n, sum(b) / n
    va = sum((x - ma) ** 2 for x in a); vb = sum((y - mb) ** 2 for y in b)
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / math.sqrt(va * vb) if va > 0 and vb > 0 else float("nan")


def _fit(x, y):
    n = len(x); mx, my = sum(x) / n, sum(y) / n
    vx = sum((v - mx) ** 2 for v in x)
    slope = sum((a - mx) * (b - my) for a, b in zip(x, y)) / vx if vx > 0 else float("nan")
    return slope, my - slope * mx


def cmd_report(a) -> None:
    from sts2rl.catalog import Catalog
    from sts2env.protocol import ActionKind
    vocab = Catalog().vocab
    d = Path(a.dir)
    G = _read("greedy.jsonl", d)
    print(f"== 贪心基线：{sum(r['win'] for r in G.values())}/{len(G)} 题两场都赢，{sum(r.get('stage2', 0) for r in G.values())} 题打到第二场")
    variants = {"search_a": "×1（每候选 4 次，价值叶子 12 步）", "search_b16": "×4（每候选 16 次）", "search_b64": "×16（每候选 64 次）", "search_end4": "推演打到底（每候选 4 次）"}
    SV = {k: _read(f"{k}_s*.jsonl", d) for k in variants}
    print("\n== 0a / 0b：搜索对贪心逐题配对")
    for k, name in variants.items():
        S = SV[k]
        if not S:
            continue
        ids = sorted(i for i in S if i in G and not G[i].get("stuck"))
        real = sum(1 for r in S.values() for x in r["rows"] if x["key_act"] != x["key_best"])
        print(f"- {name}：{_pair(S, G, ids)}")
        print(f"    搜索 {sum(r['searched'] for r in S.values())} 次，改选 {sum(r['changed'] for r in S.values())} 次（其中动作键真不同 {real} 次），推演 {sum(r['sims'] for r in S.values())} 次，平均每题 {sum(r['sims'] for r in S.values()) / len(S):.0f} 次")
    print("\n== 预算曲线（同一批题上）")
    for mod, keys in ((6, ("search_a", "search_b16")),):
        base = [i for i in G if i % mod == 0]
        for k in keys:
            S = SV[k]
            ids = [i for i in base if i in S and i in SV["search_a"]]
            if ids:
                print(f"  题号 % {mod} == 0，{variants[k]}：{_pair(S, G, ids)}")
    # 0d：改选分类（以 search_a 为准）
    S = SV["search_a"]
    if S:
        print("\n== 0d：改选分类（search_a，动作键真不同的改选才算）")
        rows = [x for r in S.values() for x in r["rows"]]
        kind = lambda key: ActionKind(key[0]).name
        cname = lambda key: vocab[key[1][1]] if key[1] else "-"
        cat: dict[str, int] = {}
        chosen_over_end: dict[str, int] = {}
        dropped_for_end: dict[str, int] = {}
        for x in rows:
            ka, kb = x["key_act"], x["key_best"]
            if ka == kb:
                continue
            a_k, b_k = kind(ka), kind(kb)
            if a_k == "END_TURN" and b_k == "PLAY_CARD":
                c = "贪心结束回合 → 搜索改打牌"; chosen_over_end[cname(kb)] = chosen_over_end.get(cname(kb), 0) + 1
            elif a_k == "PLAY_CARD" and b_k == "END_TURN":
                c = "贪心打牌 → 搜索改结束回合"; dropped_for_end[cname(ka)] = dropped_for_end.get(cname(ka), 0) + 1
            elif a_k == b_k == "PLAY_CARD":
                c = "同一张牌改目标" if ka[1] == kb[1] else "出牌顺序/换一张牌"
            elif "POTION" in a_k or "POTION" in b_k:
                c = "涉及药水"
            else:
                c = f"其他（{a_k} → {b_k}）"
            cat[c] = cat.get(c, 0) + 1
        tot = sum(cat.values())
        print(f"  搜索 {len(rows)} 次，真改选 {tot} 次（{100 * tot / max(1, len(rows)):.1f}%）")
        for c, n in sorted(cat.items(), key=lambda kv: -kv[1]):
            print(f"    {c}：{n}")
        top = lambda dct: "、".join(f"{k} {v}" for k, v in sorted(dct.items(), key=lambda kv: -kv[1])[:8])
        if chosen_over_end:
            print(f"  搜索改打的牌（贪心要结束回合时）：{top(chosen_over_end)}")
        if dropped_for_end:
            print(f"  搜索放弃的牌（改成结束回合）：{top(dropped_for_end)}")
        print("  按贪心项的概率分档的真改选率（触发门槛的依据）：")
        for lo, hi in ((0, .3), (.3, .5), (.5, .7), (.7, .9)):
            sel = [x for x in rows if lo <= x["p"] < hi]
            if sel:
                ch = sum(1 for x in sel if x["key_act"] != x["key_best"])
                print(f"    p ∈ [{lo}, {hi})：{len(sel)} 次搜索，真改选 {ch} 次（{100 * ch / len(sel):.1f}%）")
    # 0c：价值头校准
    V = _read("value_c_s*.jsonl", d)
    if V:
        print("\n== 0c：价值头校准（每个局面 = 若干重洗世界；MC = 贪心推演打到底的得分，同连打结算尺度）")
        st = [dict(s, id=i) for i, r in V.items() for s in r["states"] if s["roll"] and s["v_np"] and s["v_p"]]
        mc = [sum(s["roll"]) / len(s["roll"]) for s in st]
        for name, key in (("不带特权（搜索里实际用的）", "v_np"), ("带特权（训练时价值头看到的）", "v_p")):
            vm = [sum(s[key]) / len(s[key]) for s in st]
            sl, ic = _fit(vm, mc)
            rmse = math.sqrt(sum((v - m) ** 2 for v, m in zip(vm, mc)) / len(vm))
            pw_v = [v for s in st for v in s[key][: len(s["roll"])]]; pw_r = [r for s in st for r in s["roll"][: len(s[key])]]
            print(f"- {name}：{len(st)} 个局面，局面均值相关 {_corr(vm, mc):.3f}，MC ≈ {sl:.2f} × V + {ic:+.2f}，平均偏差 V − MC = {sum(v - m for v, m in zip(vm, mc)) / len(vm):+.3f}，RMSE {rmse:.3f}；逐世界相关 {_corr(pw_v, pw_r):.3f}")
        for stg in (0, 1):
            sub = [(sum(s["v_np"]) / len(s["v_np"]), m) for s, m in zip(st, mc) if s["stage"] == stg]
            if len(sub) > 3:
                print(f"    第 {stg + 1} 场：{len(sub)} 个局面，不带特权相关 {_corr([x for x, _ in sub], [y for _, y in sub]):.3f}，平均 V {sum(x for x, _ in sub) / len(sub):+.3f}，平均 MC {sum(y for _, y in sub) / len(sub):+.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("search", "value", "table"):
        s = sub.add_parser(name)
        s.add_argument("--ckpt", required=True); s.add_argument("--exam", required=True); s.add_argument("--out", required=True)
        s.add_argument("--shard", default="0/1"); s.add_argument("--ids-mod", default=None, help="m/r：只取题号 % m == r 的题（取固定子集）")
        s.add_argument("--limit", type=int, default=0); s.add_argument("--envs", type=int, default=4); s.add_argument("--live", type=int, default=2)
        s.add_argument("--device", default="cpu"); s.add_argument("--threads", type=int, default=2)
        s.add_argument("--tau", type=float, default=0.9); s.add_argument("--k", type=int, default=4); s.add_argument("--pmin", type=float, default=0.02)
        s.add_argument("--sims0", type=int, default=4); s.add_argument("--rounds", type=int, default=2)
        s.add_argument("--paired", action="store_true"); s.add_argument("--z", type=float, default=2.0); s.add_argument("--leaf", type=int, default=0)
        if name == "value":
            s.add_argument("--every", type=int, default=8); s.add_argument("--worlds", type=int, default=12)
        if name == "search":
            s.add_argument("--dedupe", action="store_true", help="合并动作键相同的候选")
            s.add_argument("--early", default="", choices=["", "skip", "race"], help="提前淘汰：skip 第一轮没有正差就不做第二轮；race 差 + zf 个标准误仍为负的淘汰")
            s.add_argument("--zf", type=float, default=1.0)
        if name == "table":
            s.add_argument("--every", type=int, default=3); s.add_argument("--seeds", type=int, default=20)
    c = sub.add_parser("compare"); c.add_argument("greedy"); c.add_argument("search", nargs="+")
    r = sub.add_parser("report"); r.add_argument("dir")
    sm = sub.add_parser("simulate"); sm.add_argument("dir")
    a = ap.parse_args()
    {"search": cmd_search, "value": cmd_value, "compare": cmd_compare, "report": cmd_report, "table": cmd_table, "simulate": cmd_simulate}[a.cmd](a)


if __name__ == "__main__":
    main()
