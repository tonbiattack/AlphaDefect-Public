"""Boss 战确定化搜索与贪心在固定考卷上的配对比较（2026-10-27 起，技术路线第七节 M4 的第一步，只评测、不训练）。

每道题（exam.json 的一个 Boss 局面，带种子）各打两遍：
- 贪心：每步选网络概率最高的候选（与评测、考卷的打法相同）；
- 搜索：网络最想选的那一项概率不到 --tau 时，取概率前 --k 的候选（各至少 --pmin），在若干个“可能世界”里推演：
  从这场战斗开头按已走的动作重放到当前局面（引擎是确定的），用 reseed 换掉以后要用的随机数、重洗抽牌堆（玩家看不到的部分），
  走这个候选，之后用贪心把这场战斗打完，记输赢和剩余血量。所有候选用同一组随机数（共同随机数，比较更准），
  逐轮减半：第一轮每个候选 --sims0 次，留下一半，第二轮再加一倍……直到剩一个或用完轮数，选平均得分最高的。
  得分 = 赢 + 0.1 × 剩余血量比例（只在赢时）。真实对局始终用原种子，搜索只决定走哪一步。

结果每题一行写到 --out（JSON：题号、幕、遭遇、贪心与搜索的输赢和剩余血量、搜索的决策数与推演数）。
多进程：同时开几个进程，各用 --shard k/n 取一部分题目。

    .venv/bin/python -I tools/mcts_eval.py --ckpt 检查点.pt --exam exam.json --out 结果.jsonl --envs 16 --shard 0/8 --device cuda
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

import numpy as np  # noqa: E402
import torch  # noqa: E402

from sts2env import VecEnv  # noqa: E402
from sts2env.env import EngineError  # noqa: E402
from sts2env.protocol import TokenKind  # noqa: E402
from sts2rl.actor import MODE_COMBAT  # noqa: E402
from sts2rl.features import Ragged, collate_ragged  # noqa: E402
from sts2rl.model import ModelConfig, PolicyValueNet, load_state  # noqa: E402

MAX_STEPS = 600     # 单场战斗的步数上限（同评测的 max_steps_combat 量级），超过按输


class Task:
    """在一个引擎上执行：开局并按 prefix 重放、（可选）reseed 后走 first（引擎一次往返完成）→ 返回观测（mode="obs"）或贪心打完（mode="rollout"）。"""
    __slots__ = ("spec", "prefix", "seed", "first", "mode", "pos", "stage", "steps", "result", "owner", "emax", "ehp")

    def __init__(self, spec, prefix, mode, seed=None, first=None, owner=None):
        self.spec, self.prefix, self.mode, self.seed, self.first, self.owner = spec, list(prefix), mode, seed, first, owner
        self.pos, self.stage, self.steps, self.result = 0, "reset", 0, None
        self.emax, self.ehp = None, None      # 推演开始时敌人的最大生命合计、最近一次看到的敌人剩余生命合计


def outcome(o, t: "Task | None" = None) -> dict:
    r = {"win": bool(o.victory) and not o.stuck, "hp": int(o.hp), "max_hp": max(1, int(o.max_hp))}
    if t is not None and t.emax:
        r["enemy_left"] = round(min(1.0, t.ehp / t.emax), 4)
    return r


def enemy_hp(o) -> tuple[float, float]:
    """观测里敌人的剩余生命与最大生命合计（战斗结束那一帧没有敌人记录，所以推演时每步记下）。"""
    k = np.flatnonzero((o.tok_ints[:, 0] == TokenKind.CREATURE) & (o.tok_feats[:, 5] == 0) & (o.tok_feats[:, 3] > 0))
    return float(o.tok_feats[k, 0].sum()), float(o.tok_feats[k, 1].sum())


SCORE = "win"
LEAF = 0        # 价值估值叶子：推演走这么多步后停下，用网络的价值头估这个局面（0 为一直打到底）


def score_value(r: dict, hp0: float) -> float:
    """价值估值叶子时的得分，与价值头同一尺度（战斗课程的 Boss 战回报，gamma = 1）：推演到叶子停下的用价值头的估计；
    提前分出胜负的按回报公式算——赢 1 + Boss 奖励 0.5，输 −1 + 0.5 × 打掉的敌人生命比例，再加 (结束血量 − 开战血量) / 最大生命。"""
    if "value" in r:
        return r["value"]
    hp = (r["hp"] - hp0) / r["max_hp"]
    if r["win"]:
        return 1.5 + hp
    return -1.0 + 0.5 * (1.0 - r.get("enemy_left", 1.0)) + hp


def score(r: dict) -> float:
    """推演结果的得分。win：赢 = 1 + 0.1 × 剩余血量比例，输 = 0。
    dmg（2026-10-27 起的第二版）：输了按打掉敌人生命的比例给 0.5 × 比例，贪心推演多数会输（第三幕 Boss 约八成），只看输赢时多数候选都是 0、分不出好坏。"""
    if r["win"]:
        return 1.0 + 0.1 * r["hp"] / r["max_hp"]
    if SCORE == "dmg" and "enemy_left" in r:
        return 0.5 * (1.0 - r["enemy_left"])
    return 0.0


def search_item(item: dict, a: argparse.Namespace, seed_base: int, labels: list | None = None):
    """一道题的搜索对局（生成器）：yield ("policy", obs) 取网络概率，yield ("tasks", [Task]) 等推演结果。
    labels 不为 None 时（搜索蒸馏，tools/distill/label_worker.py），每个搜索过的决策追加 (观测, 网络原本的选择, 搜索的选择, 原选择的概率)。"""
    spec, prefix = item["spec"], []
    hp0 = float(json.loads(spec["player_snapshot"]).get("current_hp", 0)) if spec.get("player_snapshot") else 0.0
    sc_of = (lambda r: score_value(r, hp0)) if LEAF else score
    n_search = n_sims = n_changed = 0
    root = None     # 第一次搜索时贪心那一项的推演胜率（核对推演是否贴近真实：全卷平均应接近贪心的实际胜率）
    while True:
        (o,) = yield ("tasks", [Task(spec, prefix, "obs")])
        if isinstance(o, EngineError) or o.done or o.stuck or o.num_actions == 0 or len(prefix) > MAX_STEPS:
            res = outcome(o) if not isinstance(o, EngineError) and len(prefix) <= MAX_STEPS else {"win": False, "hp": 0, "max_hp": 1}
            res.update(decisions=len(prefix), searched=n_search, sims=n_sims, changed=n_changed, root=root)
            return res
        p = yield ("policy", o)
        order = np.argsort(-p)
        act = int(order[0])
        if o.num_actions > 1 and p[order[0]] < a.tau:
            cands = [int(c) for c in order[: a.k] if p[c] >= a.pmin] or [act]
            if len(cands) > 1:
                n_search += 1
                sc: dict[int, dict[int, float]] = {c: {} for c in cands}     # 候选 → {推演种子: 得分}；各候选用同一组种子
                alive, n_new, used = list(cands), a.sims0, 0
                wins: list[int] = []
                for _ in range(a.rounds):
                    seeds = [seed_base + 7919 * (len(prefix) + 1) + used + j for j in range(n_new)]
                    used += n_new
                    tasks = [Task(spec, prefix, "rollout", seed=s, first=c, owner=(c, s)) for c in alive for s in seeds]
                    results = yield ("tasks", tasks)
                    n_sims += len(tasks)
                    for t, r in zip(tasks, results):
                        c, s_ = t.owner
                        sc[c][s_] = sc_of(r) if isinstance(r, dict) else sc_of({"win": False, "hp": 0, "max_hp": 1})
                        if root is None and c == act:
                            wins.append(int(isinstance(r, dict) and r["win"]))
                    if len(alive) <= 2:
                        continue
                    # 逐轮减半，贪心那一项始终留下（之后要和它逐个世界配对比较）
                    mean = lambda c: sum(sc[c].values()) / len(sc[c])
                    rest = sorted((c for c in alive if c != act), key=lambda c: (-mean(c), -p[c]))
                    alive = [act] + rest[: max(1, math.ceil(len(alive) / 2) - 1)]
                    n_new *= 2
                if root is None and wins:
                    root = round(sum(wins) / len(wins), 4)
                if a.paired:
                    # 第三版：与贪心那一项在同一批随机世界上逐个配对取差，平均差超过 z 个标准误才换（第一、二版直接比平均分，
                    # 推演次数少时一半的步都被噪声改掉了，诊断批 1,968 次搜索改了 1,004 次，结果与贪心打平）
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
                        if m > best_m and m > a.z * sd / math.sqrt(len(d)):
                            best, best_m = c, m
                else:
                    best = max(cands, key=lambda c: (sum(sc[c].values()) / max(1, len(sc[c])) + (1e-6 if c == act else 0.0), p[c]))
                n_changed += best != act
                if labels is not None:
                    labels.append((o, act, best, float(p[act])))
                act = best
        prefix.append(act)


def greedy_item(item: dict):
    (r,) = yield ("tasks", [Task(item["spec"], [], "rollout")])
    return r if isinstance(r, dict) else {"win": False, "hp": 0, "max_hp": 1}


def run(a: argparse.Namespace, items: list[tuple[int, dict]], net, device, on_item=None, labels: dict | None = None) -> None:
    """on_item 为 None 时（考卷比较）：每题贪心、搜索各打一遍，写一行到 a.out。
    搜索蒸馏：只打搜索那一遍，题做完时调用 on_item(题号, 搜索的结果)；labels[题号] 为该题的标签列表（见 search_item）。"""
    kinds = ("greedy", "search") if on_item is None else ("search",)
    if on_item is None:
        out = open(a.out, "a", encoding="utf-8")
        # 同一批（文件名 <批名>_s<分片>.jsonl）里任何分片做过的题都跳过：改了进程数接着跑时不重做
        done_ids = set()
        tag = Path(a.out).stem.rsplit("_s", 1)[0]
        for f in Path(a.out).parent.glob(f"{tag}_s*.jsonl"):
            for l in open(f, encoding="utf-8"):
                if l.strip():
                    done_ids.add(json.loads(l)["i"])
        items = [(i, it) for i, it in items if i not in done_ids]
        print(f"{len(items)} 道题待做（已做 {len(done_ids)}）", flush=True)
    # 每道题两个控制器（贪心、搜索），结果都齐了才写一行
    ctrls: dict = {}         # 控制器 id → [生成器, 题号, 种类, 待交付的结果]
    partial: dict[int, dict] = {}
    queue: list = []         # 待执行的 Task
    waiting: dict = {}       # 控制器 id → (任务列表, 已完成数)
    pol_req: list = []       # (控制器 id, 观测)
    env_task: dict[int, Task] = {}
    obs_of: dict[int, object] = {}
    t0 = time.time()
    next_items = iter(items)
    n_live = 0

    def advance(cid, value):
        nonlocal n_live
        gen = ctrls[cid][0]
        try:
            kind, payload = gen.send(value)
        except StopIteration as e:
            _, i, which, _ = ctrls.pop(cid)
            rec = partial.setdefault(i, {})
            rec[which] = e.value
            if on_item is not None:
                on_item(i, e.value)
                del partial[i]
                n_live -= 1
                start_more()
            elif len(rec) == 2:
                it = dict(items_by_id[i])
                line = {"i": i, "act": it["act"], "encounter": it["encounter"], "greedy": rec["greedy"], "search": rec["search"], "t": round(time.time())}
                out.write(json.dumps(line, ensure_ascii=False) + "\n"); out.flush()
                del partial[i]
                n_live -= 1
                start_more()
            return
        if kind == "policy":
            pol_req.append((cid, payload))
        else:
            for t in payload:
                t.owner = (cid, t.owner)
            waiting[cid] = (payload, 0)
            queue.extend(payload)

    items_by_id = dict(items)
    cid_seq = [0]

    def start_more():
        nonlocal n_live
        while n_live < a.live:
            try:
                i, it = next(next_items)
            except StopIteration:
                return
            n_live += 1
            gens = {"greedy": lambda: greedy_item(it), "search": lambda: search_item(it, a, 100003 * (i + 1), None if labels is None else labels.setdefault(i, []))}
            for which, gen in ((w, gens[w]()) for w in kinds):
                cid = cid_seq[0]; cid_seq[0] += 1
                ctrls[cid] = [gen, i, which, None]
                advance(cid, None)

    def task_done(t: Task, value):
        cid, owner = t.owner
        t.owner = owner
        t.result = value
        tasks, k = waiting[cid]
        k += 1
        if k == len(tasks):
            del waiting[cid]
            r = [x.result for x in tasks]
            advance(cid, r)
        else:
            waiting[cid] = (tasks, k)

    with VecEnv(a.envs) as env:
        free = list(range(len(env)))

        def feed(i: int, o) -> None:
            """引擎 i 返回观测 o 之后，推进它手上的任务。"""
            t = env_task[i]
            if isinstance(o, EngineError):
                del env_task[i]; free.append(i)
                task_done(t, o)
                return
            if t.mode == "obs":
                del env_task[i]; free.append(i)
                task_done(t, o)
                return
            if o.done or o.stuck or o.num_actions == 0 or t.steps > MAX_STEPS:
                del env_task[i]; free.append(i)
                task_done(t, outcome(o, t) if t.steps <= MAX_STEPS else {"win": False, "hp": 0, "max_hp": 1})
                return
            hp_, mx_ = enemy_hp(o)
            if t.emax is None:
                t.emax = mx_
            t.ehp = hp_
            obs_of[i] = o      # 等网络选动作

        start_more()
        while ctrls:
            while free and queue:
                i = free.pop(); t = queue.pop()
                env_task[i] = t
                req = {"op": "reset", "mode": "combat", "spec": t.spec, "replay": t.prefix}
                if t.seed is not None:      # 引擎在一次往返里完成：开局、重放、换随机数、走候选
                    req["reseed"] = int(t.seed) & 0xFFFFFFFF
                    if t.first is not None:
                        req["first"] = int(t.first)
                env.send(i, req)
            for i, o in env.recv_ready(timeout=0.5):
                feed(i, o)
            deadline = time.time() + 0.002
            while env.pending and time.time() < deadline:
                for i, o in env.recv_ready(timeout=max(0.0, deadline - time.time())):
                    feed(i, o)
            if not obs_of and not pol_req:
                continue
            ids = list(obs_of)
            reqs = list(pol_req); pol_req.clear()
            batch = [obs_of[i] for i in ids] + [o for _, o in reqs]
            with torch.no_grad():
                logits, values = net(collate_ragged(Ragged.from_obs(batch, np.array([MODE_COMBAT] * len(batch)))).to(device))
            logits, values = logits.float().cpu(), values.float().cpu()
            for k, i in enumerate(ids):
                o = obs_of.pop(i)
                t = env_task[i]
                if LEAF and t.mode == "rollout" and t.seed is not None and t.steps >= LEAF:
                    # 价值估值叶子：不再往下打，用价值头估这个局面
                    del env_task[i]; free.append(i)
                    task_done(t, {"win": False, "hp": int(o.hp), "max_hp": max(1, int(o.max_hp)), "value": float(values[k])})
                    continue
                t.steps += 1
                env.send(i, {"op": "step", "action": int(logits[k, : o.num_actions].argmax())})
            for k, (cid, o) in enumerate(reqs):
                p = torch.softmax(logits[len(ids) + k, : o.num_actions], -1).numpy()
                advance(cid, p)
    print(f"完成，用时 {time.time() - t0:.0f} 秒", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--exam", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--acts", default="1,2", help="考卷里哪几幕的题（1 第二幕 Boss，2 第三幕 Boss）")
    ap.add_argument("--limit", type=int, default=0, help="每幕最多几道（0 为全部）")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--live", type=int, default=6, help="同时在做的题数")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tau", type=float, default=0.9)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--pmin", type=float, default=0.02)
    ap.add_argument("--sims0", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--score", default="win", choices=["win", "dmg"], help="推演结果怎么打分，见 score()")
    ap.add_argument("--paired", action="store_true", help="第三版：与贪心逐个世界配对比较，显著更好才换")
    ap.add_argument("--z", type=float, default=2.0, help="配对比较时平均差要超过几个标准误")
    ap.add_argument("--leaf", type=int, default=0, help="价值估值叶子：推演走这么多步后用价值头估值（0 为打到底）")
    a = ap.parse_args()
    torch.set_num_threads(2)
    global SCORE, LEAF
    SCORE, LEAF = a.score, a.leaf
    blob = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    net = PolicyValueNet(ModelConfig(**blob["config"]["model"])).eval().to(a.device)
    load_state(net, blob["state"])
    exam = json.load(open(a.exam, encoding="utf-8"))["items"]
    acts = {int(x) for x in a.acts.split(",")}
    items, per = [], {}
    for i, it in enumerate(exam):
        if it["act"] in acts and (not a.limit or per.get(it["act"], 0) < a.limit):
            per[it["act"]] = per.get(it["act"], 0) + 1
            items.append((i, it))
    total = len(items)
    k, n = map(int, a.shard.split("/"))
    items = [x for j, x in enumerate(items) if j % n == k]
    print(f"版本 {blob.get('version')}，{len(items)} 道题；搜索参数 tau={a.tau} k={a.k} pmin={a.pmin} sims0={a.sims0} rounds={a.rounds} score={a.score}", flush=True)
    # 看板用的说明文件（与结果同名，后缀 .meta.json）：哪个版本、多少题、什么参数、何时开始
    meta = Path(a.out).with_suffix(".meta.json")
    if not meta.exists():
        meta.write_text(json.dumps({"version": blob.get("version"), "items": len(items), "total": total, "acts": sorted(acts), "shard": a.shard, "envs": a.envs,
                                    "live": a.live, "start": round(time.time()),
                                    "params": {"tau": a.tau, "k": a.k, "pmin": a.pmin, "sims0": a.sims0, "rounds": a.rounds, "score": a.score, "paired": a.paired, "z": a.z, "leaf": a.leaf}},
                                   ensure_ascii=False), encoding="utf-8")
    run(a, items, net, a.device)


if __name__ == "__main__":
    main()
