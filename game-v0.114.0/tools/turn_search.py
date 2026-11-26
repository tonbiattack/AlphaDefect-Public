"""回合级束搜索（engine/Sts2SimServe/方案.md）：把这一回合剩下的出牌当成一个整体规划，在 sts2-sim 上展开、用价值头给方案打分。

PIMC（sim_pimc.pimc_decide）只在“这一步出什么”上改选、之后交给贪心，看不到同一回合里几张牌的组合与先后。这里：
- 从镜像的当前局面出发（先 reseed：换掉以后的随机数，不碰真实随机数；同一次搜索里所有分支共用这一组随机数，即共同随机数），
  逐层展开：每个节点取策略最想选的 expand 个候选，外加结束回合；
- 子节点三种：战斗分出胜负（按 score_terminal 记分）、回合结束（下一回合开始的那一帧，用价值头估值，敌人的行动按这个世界的随机数已经发生）、
  还在本回合（进下一层）。选牌界面（打出某些牌后要选牌）在 sts2-sim 里不能存档：按策略贪心走完选牌、回到顶层决策再存。
- 同一层里看得到的局面相同的子节点只留一个（按观测帧去掉步数后的字节去重：不同出牌顺序打出同样的局面）；
- 每层按价值头留下最好的 beam 个节点往下走；贪心路线（每层都选策略最高项）一直留着，最后和它比；
- 整回合最好的方案比贪心路线的得分高出 margin 以上，才把这一步换成那个方案的第一步（价值头有误差，差一点不改）。
  worlds > 1 时在几个重抽的世界里各搜一遍，按各方案第一步的平均得分选（目前默认 1）。

只用镜像看得到的局面；所有请求经 SimMirror 的进程与槽位（槽位 0 为镜像锚点，这里用 1000 起的槽位，用完不清理，服务端按槽位号覆盖）。
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field

import numpy as np

import sim_pimc as S
from sts2env.env import EngineError
from sts2env.protocol import ActionKind, DecisionKind, Obs, TokenKind, parse

SLOT0 = 1000


@dataclass
class TurnParams:
    tau: float = 0.9           # 最想选的项概率不到它才搜索（同 PIMC 的门限）
    beam: int = 6              # 每层留几个节点
    expand: int = 4            # 每个节点展开策略最高的几个候选（另加结束回合）
    max_depth: int = 10        # 本回合最多再出几步
    max_nodes: int = 400       # 一次搜索最多展开多少个子节点
    margin: float = 0.03       # 最好的方案比贪心路线高出多少才改
    win_hp: float = 0.5        # 打赢的得分 = 1.5 + win_hp × 剩余生命比例（第一场省下的血留给第二场）
    worlds: int = 1            # 几个重抽的世界
    win_bonus_run: float = 0.1 # 价值头用整局模式时：打赢的得分 = 打出最后一击之前那个局面的整局价值 + 这个奖励
                               # （整局价值的尺度是通关 +1、失败 −1，与战斗课程的 1.5 不同；那个局面的价值已经含剩余生命、药水对整局的影响）


@dataclass
class TurnInfo:
    searched: bool = False
    greedy: int = -1
    best: int = -1
    changed: bool = False
    nodes: int = 0
    leaves: int = 0
    depth: int = 0
    dedup: int = 0
    errors: int = 0
    err_text: str | None = None
    greedy_score: float | None = None
    best_score: float | None = None
    total_s: float = 0.0
    sim_s: float = 0.0
    net_s: float = 0.0
    plan: list = field(default_factory=list)


@dataclass
class _Node:
    proc: int
    slot: int | None           # 存档的槽位；None 表示存不了（选牌途中），靠 restore 重建
    restore: list              # 重建这个节点的请求（load 某个槽位 + 若干步）
    frame: Obs
    raw: bytes
    path: list                 # 从根出发走过的镜像候选下标（根这一步为第一个）
    first: int                 # 根这一步的镜像候选下标
    greedy: bool               # 是否在贪心路线上
    value: float = 0.0
    left: float = 1.0          # 父节点上敌人剩余生命的比例（战斗以输结束时记分用）
    parent_value: float = 0.0  # 父节点的价值（整局模式下打赢时记分用）


def _turn(o: Obs) -> int:
    g = np.flatnonzero(o.tok_ints[:, 0] == TokenKind.GLOBAL)
    return int(o.tok_feats[g[0], 9]) if len(g) else -1


def _key(raw: bytes) -> bytes:
    """看得到的局面：帧字节去掉头部 13（整局步数，不同出牌顺序走的步数可能不同）。"""
    b = bytearray(raw)
    b[52:56] = b"\0\0\0\0"
    return hashlib.blake2b(bytes(b), digest_size=16).digest()


def score_terminal(o: Obs, left: float, P: TurnParams) -> float:
    r = S.outcome(o, left)
    if r["win"]:
        return 1.5 + P.win_hp * r["hp"] / max(1, r["max_hp"])
    return S.terminal(r)


def _top_level(o: Obs) -> bool:
    return int(o.decision) == int(DecisionKind.COMBAT) and o.num_actions > 0


def turn_decide(mirror: S.SimMirror, ev: S.ModelEvaluator, real: Obs, p: np.ndarray, cmap: list[int], P: TurnParams, seed: int) -> TurnInfo:
    """真实帧 real 上的这一步：门限到了就在镜像上做回合级束搜索，返回 TurnInfo（best 为真实候选下标）。cmap：真实候选 → 镜像候选。
    返回前各进程一律回到镜像的当前局面（镜像接着跟真实对局走）。"""
    info = None
    try:
        info = _turn_decide(mirror, ev, real, p, cmap, P, seed)
        return info
    finally:
        if info is None or info.searched or info.err_text:
            _restore(mirror, info)


def _restore(mirror: S.SimMirror, info: "TurnInfo | None") -> None:
    """各进程回到当前局面（load 锚点 + 重走 tape），核对回来的帧与镜像记的一致。"""
    base = mirror.root_ops()
    for c in mirror.clients:
        for q in base[:-1]:
            c.send({**q, "obs": False})
        c.send(base[-1])
    for c in mirror.clients:
        r = c.drain()[-1]
        if info is not None and (isinstance(r, EngineError) or S.compare(mirror.frame, parse(r)) is not None):
            info.errors += 1
            info.err_text = info.err_text or "搜索后回不到当前局面"


def _turn_decide(mirror: S.SimMirror, ev: S.ModelEvaluator, real: Obs, p: np.ndarray, cmap: list[int], P: TurnParams, seed: int) -> TurnInfo:
    t_all = time.perf_counter()
    info = TurnInfo()
    act = int(np.argmax(p))
    info.greedy = info.best = act
    if real.num_actions <= 1 or p[act] >= P.tau:
        return info
    if not mirror.anchor_ok:
        info.err_text = "镜像没有可用的锚点"
        return info
    inv = {int(m): r for r, m in enumerate(cmap)}            # 镜像候选 → 真实候选
    info.searched = True
    clients = mirror.clients
    n_proc = len(clients)
    base = mirror.root_ops()
    slot_next = [SLOT0] * n_proc
    root = mirror.frame
    turn0 = _turn(root)

    def new_slot(j: int) -> int:
        slot_next[j] += 1
        return slot_next[j]

    # 根：每个进程回到当前局面、reseed 成这次搜索的世界，存进根槽位
    roots = []
    for j, c in enumerate(clients):
        for q in base:
            c.send({**q, "obs": False} if q["op"] in ("load", "step") else q)
        c.send({"op": "reseed", "seed": seed & 0xFFFFFFFF, "obs": True})
        s0 = new_slot(j)
        c.send({"op": "save", "slot": s0})
    for j, c in enumerate(clients):
        r = c.drain()
        raw = r[-2]
        if isinstance(raw, EngineError) or isinstance(r[-1], EngineError):
            info.errors += 1
            info.err_text = f"根局面：{str(raw if isinstance(raw, EngineError) else r[-1])[:160]}"
            info.searched = False
            return info
        roots.append(_Node(j, slot_next[j], [{"op": "load", "slot": slot_next[j], "obs": False}], parse(raw), raw, [], -1, True))

    def evaluate(frames: list[Obs]):
        """策略（整局模式，同真实对局出手）与价值（value_mode）一起算：一批里两份。"""
        t0 = time.perf_counter()
        n = len(frames)
        logits, values = ev.forward(frames + frames, [ev.policy_mode] * n + [ev.value_mode] * n)
        info.net_s += time.perf_counter() - t0
        pol = [ev.torch.softmax(logits[i, : frames[i].num_actions], -1).numpy() for i in range(n)]
        return pol, [float(values[n + i]) for i in range(n)]

    pol, val = evaluate([roots[0].frame])
    roots[0].value = val[0]
    frontier = [roots[0]]
    root_pol = pol
    leaves: list[tuple[float, _Node]] = []
    greedy_leaf: tuple[float, _Node] | None = None
    seen: set[bytes] = {_key(roots[0].raw)}

    for depth in range(P.max_depth):
        if not frontier or info.nodes >= P.max_nodes:
            break
        info.depth = depth + 1
        if depth == 0:
            pols = root_pol
        else:
            pols, _ = evaluate([n.frame for n in frontier])
        # 每个节点要展开的候选：策略最高的 expand 个，加结束回合；根这一层的贪心项一定在里面
        plans: list[tuple[_Node, int, bool]] = []
        for k, (nd, pp) in enumerate(zip(frontier, pols)):
            o = nd.frame
            order = [int(x) for x in np.argsort(-pp)]
            pick = order[: P.expand]
            ends = [a for a in range(o.num_actions) if int(o.cand_ints[a, 0]) == int(ActionKind.END_TURN)]
            for a in ends:
                if a not in pick:
                    pick.append(a)
            if depth == 0:
                pick = [a for a in pick if a in inv]           # 根这一层只走真实对局里也有的候选
                ga = int(cmap[act]) if act < len(cmap) else None
                if ga is not None and ga not in pick:
                    pick.insert(0, ga)
            for a in pick:
                g = nd.greedy and (a == (int(cmap[act]) if depth == 0 else order[0]))
                plans.append((nd, a, g))
        # 根这一层的子节点轮流分到各个进程（各进程的根都一样）；之后子节点跟父节点在同一个进程
        children: list[_Node] = []
        t0 = time.perf_counter()
        by: dict[int, list] = {}
        for i, (nd, a, g) in enumerate(plans):
            j = (i % n_proc) if depth == 0 else nd.proc
            src = roots[j] if depth == 0 else nd
            by.setdefault(j, []).append((src, nd, a, g))
        for j, items in by.items():
            c = clients[j]
            for src, nd, a, g in items:
                for q in src.restore:
                    c.send(q)
                c.send({"op": "step", "action": int(a)})
                c.send({"op": "save", "slot": new_slot(j)})
        for j, items in by.items():
            replies = clients[j].drain()
            k = 0
            slot_base = slot_next[j] - len(items)
            for idx, (src, nd, a, g) in enumerate(items):
                k += len(src.restore)
                raw, sv = replies[k], replies[k + 1]
                k += 2
                if isinstance(raw, EngineError):
                    info.errors += 1
                    info.err_text = info.err_text or f"展开：{str(raw)[:160]}"
                    continue
                slot = slot_base + idx + 1
                ok = not isinstance(sv, EngineError)
                child = _Node(j, slot if ok else None,
                              [{"op": "load", "slot": slot, "obs": False}] if ok else src.restore + [{"op": "step", "action": int(a), "obs": False}],
                              parse(raw), raw, nd.path + [a], a if depth == 0 else nd.first, g, left=S.enemy_left(nd.frame),
                              parent_value=nd.value)
                children.append(child)
        info.sim_s += time.perf_counter() - t0
        info.nodes += len(children)
        # 选牌途中的子节点：按策略贪心走完选牌，回到顶层再存
        _resolve_selects(clients, children, evaluate, info, new_slot)
        # 分类、去重、估值
        nxt: list[_Node] = []
        to_eval: list[_Node] = []
        for ch in children:
            o = ch.frame
            if S.combat_over(o):
                if ev.value_mode == S.MODE_RUN:
                    sc = ch.parent_value + P.win_bonus_run if S.outcome(o)["win"] else -1.0
                else:
                    sc = score_terminal(o, ch.left, P)
                leaves.append((sc, ch))
                if ch.greedy:
                    greedy_leaf = (sc, ch)
                continue
            kk = _key(ch.raw)
            if kk in seen and not ch.greedy:
                info.dedup += 1
                continue
            seen.add(kk)
            to_eval.append(ch)
        if to_eval:
            _, vals = evaluate([c.frame for c in to_eval])
            for ch, v in zip(to_eval, vals):
                ch.value = v
                if _turn(ch.frame) != turn0 or not _top_level(ch.frame):
                    leaves.append((v, ch))         # 回合结束：下一回合开始的局面（或无法再展开），用价值头
                    if ch.greedy:
                        greedy_leaf = (v, ch)
                else:
                    nxt.append(ch)
        nxt.sort(key=lambda c: -c.value)
        keep = nxt[: P.beam]
        g = next((c for c in nxt if c.greedy), None)
        if g is not None and g not in keep:
            keep.append(g)
        frontier = keep
    # 束宽之外没走完的节点也按价值头算作叶子（到了深度或节点上限）
    for nd in frontier:
        leaves.append((nd.value, nd))
        if nd.greedy and greedy_leaf is None:
            greedy_leaf = (nd.value, nd)
    info.leaves = len(leaves)
    if not leaves or greedy_leaf is None:
        info.err_text = info.err_text or "没有贪心路线的叶子"
        info.total_s = time.perf_counter() - t_all
        return info
    best_sc, best = max(leaves, key=lambda x: x[0])
    info.greedy_score, info.best_score = round(greedy_leaf[0], 4), round(best_sc, 4)
    info.plan = best.path
    if best.first in inv and best_sc > greedy_leaf[0] + P.margin and inv[best.first] != act:
        info.best, info.changed = inv[best.first], True
    info.total_s = time.perf_counter() - t_all
    return info


def _resolve_selects(clients, children: list[_Node], evaluate, info: TurnInfo, new_slot) -> None:
    """选牌途中的子节点按策略贪心走完，回到顶层战斗决策（或战斗结束）再存档；每个这样的节点逐个处理（少见）。"""
    for ch in children:
        guard = 0
        while not S.combat_over(ch.frame) and not _top_level(ch.frame) and guard < 30:
            guard += 1
            pol, _ = evaluate([ch.frame])
            a = int(np.argmax(pol[0]))
            c = clients[ch.proc]
            for q in ch.restore:
                c.send(q)
            c.send({"op": "step", "action": a})
            s = new_slot(ch.proc)
            c.send({"op": "save", "slot": s})
            r = c.drain()
            raw, sv = r[-2], r[-1]
            if isinstance(raw, EngineError):
                info.errors += 1
                break
            ch.frame, ch.raw = parse(raw), raw
            if not isinstance(sv, EngineError):
                ch.slot, ch.restore = s, [{"op": "load", "slot": s, "obs": False}]
            else:
                ch.restore = ch.restore + [{"op": "step", "action": a, "obs": False}]
