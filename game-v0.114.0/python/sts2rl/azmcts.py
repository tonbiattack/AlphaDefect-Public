"""AlphaZero 式的战斗搜索（2026-10-27 起，方案见 AlphaZero式搜索方案.md，决策记录 145）。

单观察者信息集蒙特卡洛树搜索（SO-ISMCTS）＋ PUCT：
- 每次模拟从根局面开一个新的确定化世界（reseed：只换玩家看不到的随机数、重洗抽牌堆），沿一棵共用的树往下走；
- 子节点按“动作键”区分（不同世界里手牌不同，候选下标的含义会变；键由动作类型、牌或药水的身份、目标的身份组成，
  一模一样的两张牌合成一个动作）；
- 选择用 PUCT：Q + c · P · sqrt(可用次数) / (1 + 访问次数)，可用次数是这个动作在经过该节点的模拟里合法过几次；
  P 是策略头在“第一次见到这个动作时那个世界的观测”上的概率；Q 按树里见过的最小、最大估值归一化到 [0, 1]；
- 走到树里没有的动作就扩展一个节点，对新局面调一次网络（策略先验 + 价值），不推演；战斗结束直接按回报公式算分；
- 一棵树可以同时挂多次模拟，路过的边加“虚拟损失”，让并发的模拟分散到不同的路径上。

这里只有树和单次模拟的逻辑，与引擎无关。模拟写成生成器，向调度器要三样东西：
  ("reset", 种子)   → 根局面在一个新确定化世界里的观测
  ("step", 候选下标) → 走一步后的观测
  ("eval", 观测)     → (各候选的概率, 价值)
C# 重放后端与调度器见 tools/az_eval.py；Rust 引擎能复制状态以后换成 Rust 后端。
"""
from __future__ import annotations

import math

import numpy as np

from sts2env.protocol import ActionKind, TokenKind

# 卡牌记录里决定“这是不是同一张牌”的特征列（观测格式.md）：0 费用，1 X 费，2 星费，3 升级次数，4 能否打出，
# 9 虚无，10 保留，11 消耗，12 不能打出，13 附魔层数，14 苦痛层数，15 固有，18 奇巧，19 永恒
_CARD_FEATS = (0, 1, 2, 3, 4, 9, 10, 11, 12, 13, 14, 15, 18, 19)


def _tok_key(o, t: int):
    """一条记录的身份：卡牌按（ID、附魔、区域、上面那些特征），生物按（ID、站位），药水只按 ID（同名药水用哪瓶都一样），其余按（类型、ID、附加 ID）。"""
    if t < 0:
        return None
    kind, vid, zone, _owner, aux = (int(x) for x in o.tok_ints[t])
    if kind in (TokenKind.CARD, TokenKind.SELECTED_CARD):
        return (kind, vid, aux, zone) + tuple(round(float(o.tok_feats[t, j]), 3) for j in _CARD_FEATS)
    if kind == TokenKind.POTION:
        return (kind, vid)
    return (kind, vid, aux)


def action_keys(o) -> list:
    """每个候选的动作键（与候选下标一一对应；同一观测里两个候选的键相同，说明它们等价）。"""
    out = []
    for k in range(o.num_actions):
        ci = o.cand_ints[k]
        out.append((int(ci[0]), _tok_key(o, int(ci[1])), _tok_key(o, int(ci[2])), int(ci[3])))
    return out


def enemy_hp(o) -> tuple[float, float]:
    """敌人剩余生命与最大生命合计（同 tools/mcts_eval.py）。"""
    k = np.flatnonzero((o.tok_ints[:, 0] == TokenKind.CREATURE) & (o.tok_feats[:, 5] == 0) & (o.tok_feats[:, 3] > 0))
    return float(o.tok_feats[k, 0].sum()), float(o.tok_feats[k, 1].sum())


def terminal_value(o, hp0: float, enemy_left: float) -> float:
    """战斗结束时的得分，与战斗课程的 Boss 战回报、价值头同一尺度（同 tools/mcts_eval.py 的 score_value）：
    赢 1 + Boss 奖励 0.5，输 −1 + 0.5 × 打掉的敌人生命比例，再加 (结束血量 − 开战血量) / 最大生命。"""
    hp = (int(o.hp) - hp0) / max(1, int(o.max_hp))
    if bool(o.victory) and not o.stuck:
        return 1.5 + hp
    return -1.0 + 0.5 * (1.0 - enemy_left) + hp


def is_terminal(o) -> bool:
    return bool(o.done) or bool(o.stuck) or o.num_actions == 0


class Edge:
    __slots__ = ("P", "N", "W", "avail", "vl")

    def __init__(self, p: float):
        self.P, self.N, self.W, self.avail, self.vl = p, 0, 0.0, 0, 0


class Node:
    __slots__ = ("edges", "children", "value")

    def __init__(self, value: float | None = None):
        self.edges: dict = {}           # 动作键 → Edge
        self.children: dict = {}        # 动作键 → Node
        self.value = value              # 扩展时网络给的价值（FPU 用）


class Tree:
    """一个决策的搜索树。c_puct：探索系数；vl：每次虚拟损失加的访问数。"""

    def __init__(self, c_puct: float = 1.25, vl: int = 1):
        self.root = Node()
        self.c, self.vl = c_puct, vl
        self.vmin, self.vmax = math.inf, -math.inf
        self.sims = 0

    def _norm(self, v: float) -> float:
        if self.vmax <= self.vmin:
            return 0.5
        return min(1.0, max(0.0, (v - self.vmin) / (self.vmax - self.vmin)))

    def _seen(self, v: float) -> None:
        self.vmin = min(self.vmin, v); self.vmax = max(self.vmax, v)

    def missing_priors(self, node: Node, keys: list) -> bool:
        return any(k not in node.edges for k in keys)

    def add_priors(self, node: Node, keys: list, probs: np.ndarray) -> None:
        """记下这个世界里各动作的先验（键相同的候选概率相加）；已有先验的动作不改。"""
        acc: dict = {}
        for k, p in zip(keys, probs):
            acc[k] = acc.get(k, 0.0) + float(p)
        for k, p in acc.items():
            if k not in node.edges:
                node.edges[k] = Edge(p)

    def select(self, node: Node, keys: list):
        """在这个世界合法的动作里按 PUCT 选一个；顺带给它们的可用次数加一。"""
        avail = list(dict.fromkeys(keys))
        for k in avail:
            node.edges[k].avail += 1
        tot_p = sum(node.edges[k].P for k in avail) or 1.0
        fpu = self._norm(node.value) if node.value is not None else 0.5
        best, best_s = None, -math.inf
        for k in avail:
            e = node.edges[k]
            n = e.N + e.vl
            q = self._norm((e.W + e.vl * (self.vmin if self.vmin < math.inf else 0.0)) / n) if n > 0 else fpu
            u = self.c * (e.P / tot_p) * math.sqrt(e.avail) / (1 + n)
            s = q + u
            if s > best_s or (s == best_s and e.P > node.edges[best].P):
                best, best_s = k, s
        return best

    def add_vl(self, node: Node, k) -> None:
        node.edges[k].vl += self.vl

    def backup(self, path: list, v: float | None) -> None:
        """沿路径回传估值并撤掉虚拟损失；v 为 None（引擎出错）时只撤虚拟损失。"""
        if v is not None:
            self._seen(v)
            self.sims += 1
        for node, k in path:
            e = node.edges[k]
            e.vl -= self.vl
            if v is not None:
                e.N += 1; e.W += v

    def best(self, keys: list):
        """根节点访问最多的动作（并列时取先验大的）。"""
        cand = [k for k in dict.fromkeys(keys) if k in self.root.edges]
        return max(cand, key=lambda k: (self.root.edges[k].N, self.root.edges[k].P))

    def visit_dist(self) -> dict:
        tot = sum(e.N for e in self.root.edges.values()) or 1
        return {k: e.N / tot for k, e in self.root.edges.items() if e.N}


def simulate(tree: Tree, seed: int, hp0: float, root_keys: list, stats: dict):
    """一次模拟（生成器）。stats 累计诊断：根局面的键在确定化后对不上的次数、引擎出错次数、扩展次数、到终局次数。"""
    o = yield ("reset", seed)
    if o is None or is_terminal(o):
        stats["error"] = stats.get("error", 0) + 1
        return
    keys = action_keys(o)
    if keys != root_keys:               # 根局面玩家看得到的部分在确定化前后应完全相同
        stats["root_mismatch"] = stats.get("root_mismatch", 0) + 1
    node, path = tree.root, []
    eleft = 1.0
    hp_now, emax = enemy_hp(o)
    if emax > 0:
        eleft = hp_now / emax
    v = None
    while True:
        if tree.missing_priors(node, keys):
            p, val = yield ("eval", o)
            tree.add_priors(node, keys, p)
            if node.value is None:
                node.value = val
        k = tree.select(node, keys)
        path.append((node, k)); tree.add_vl(node, k)
        o = yield ("step", keys.index(k))
        if o is None:                   # 引擎出错：这次模拟作废
            stats["error"] = stats.get("error", 0) + 1
            break
        if is_terminal(o):
            v = terminal_value(o, hp0, eleft)
            stats["terminal"] = stats.get("terminal", 0) + 1
            break
        h, m = enemy_hp(o)
        if m > 0:
            eleft = h / m
        keys = action_keys(o)
        child = node.children.get(k)
        if child is None:
            p, v = yield ("eval", o)
            child = Node(v)
            tree.add_priors(child, keys, p)
            node.children[k] = child
            stats["expand"] = stats.get("expand", 0) + 1
            break
        node = child
    tree.backup(path, v)


__all__ = ["ActionKind", "Edge", "Node", "Tree", "action_keys", "enemy_hp", "is_terminal", "simulate", "terminal_value"]
