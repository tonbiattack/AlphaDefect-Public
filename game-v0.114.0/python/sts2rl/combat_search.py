"""战斗搜索的对外接口（engine/Sts2SimServe/方案.md）：任何一边的真实对局（评测引擎、复测、直播桥）都按同样的三步调用。

    s = CombatSearcher(net, sim_dll=..., vocab=..., tables=...)
    s.begin(save, next_room_id, target, room, root_frame)      # 每场战斗开打时：选路那一刻的整局存档 + 这场战斗第一帧
    a = s.decide(frame, p)                                       # 每个战斗决策：真实引擎的原始帧（bytes）与策略网络的概率，回真实候选下标
    s.observe(a)                                                 # 真实对局执行了动作 a 之后（不管是不是 decide 给的）
    s.end()                                                      # 这场战斗打完

推演在 sts2-sim 上（tools/sim_pimc.py 的 SimMirror + pimc_decide，决策记录 217 的 PIMC 改选规则），只读看得到的局面，推演前一律 reseed，
不碰真实随机数。镜像（sts2-sim 跟着真实对局走同样的动作）与真实局面不一致时，先按真实帧同步镜像（服务的 sync：生命、格挡、能量、各牌堆、
敌人生命），同步后还不一致、或者候选对不上时，这一步按贪心（返回网络最想选的项）；
真实动作对应不到镜像时镜像作废，这场余下全按贪心。所以任何情况下 decide 都会给出一个合法的真实候选下标，调用方不用处理异常。

存档：Sts2Headless 的只读 save 操作回的 {"save": 游戏自己的 SerializableRun, "next_room_id"}；target 是要去的格子 [列, 行]，room 为
monster / elite / boss。第三幕第二个 Boss 没有选路：用打完第一个 Boss 之后那一刻的存档、Boss 格子（存档地图里的 second_boss）。
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

import sim_pimc as S  # noqa: E402
import turn_search as T  # noqa: E402
from sts2env.env import EngineError  # noqa: E402
from sts2env.protocol import Obs, TokenKind, Zone, parse  # noqa: E402


@dataclass
class SearchStats:
    """一场战斗里的计数（调用方写日志用）。"""
    decisions: int = 0
    searched: int = 0
    changed: int = 0
    fallback: dict = field(default_factory=dict)     # 退回贪心的原因 → 次数
    lost_at: int | None = None                       # 镜像作废于第几个决策
    lost_why: str | None = None
    search_s: float = 0.0
    rollouts: int = 0
    synced: int = 0                                  # 镜像与真实局面不一致、按真实帧同步的次数
    sync_ok: int = 0                                 # 其中同步后一致的次数

    def fall(self, why: str) -> None:
        key = why.split(":")[0].split("：")[0]
        self.fallback[key] = self.fallback.get(key, 0) + 1


class CombatSearcher:
    def __init__(self, net, sim_dll=None, vocab=None, tables=None, procs: int = 4, device: str = "cpu",
                 params: S.PimcParams | None = None, policy_mode: int = S.MODE_RUN, value_mode: int = S.MODE_COMBAT,
                 seed: int = 0, log=None, algo: str = "pimc", turn_params: "T.TurnParams | None" = None):
        """net：sts2rl 的 PolicyValueNet（推演出手取策略最高项、叶子取价值头）；sim_dll / vocab / tables：sts2-sim 服务（Sts2SimServe.dll，
        须按 -p:WithObs=true 编译）、与真实引擎同一份的词表、sim_tables.json；procs：镜像用几个 sts2-sim 进程（各自推演一部分世界）。"""
        self.P = params or S.PimcParams()
        self.algo = algo                            # "pimc"：只改选这一步（决策记录 217）；"turn"：回合级束搜索（tools/turn_search.py）
        self.TP = turn_params or T.TurnParams(tau=self.P.tau)
        self.ev = S.ModelEvaluator(net, device, policy_mode=policy_mode, value_mode=value_mode)
        self.mirror = S.SimMirror(procs, sim_dll, log=log, vocab=vocab, tables=tables)
        self.vocab = Path(vocab).read_text(encoding="utf-8").split("\n") if vocab else []
        self.seed = seed
        self.m: Obs | None = None
        self.cmap: list[int] | None = None
        self.n = 0
        self.k = 0                                  # 第几场战斗（推演世界的种子按场、按决策分开）
        self.stats = SearchStats()

    # ---------------------------------------------------------------- 一场战斗

    def begin(self, save: dict, next_room_id: int, target, room: str, root_frame: bytes) -> bool:
        """进这场战斗：sts2-sim 导入存档、进同一场战斗，根帧给移植版编码器（整局层面、战斗中不变的字段从它沿用）。导入失败返回 False（整场按贪心）。"""
        self.stats, self.n, self.cmap = SearchStats(), 0, None
        self.k += 1
        try:
            self.m = self.mirror.reset(save, int(next_room_id), target, room, root_frame)
            return True
        except EngineError as e:
            self.m = None
            self.stats.lost_at, self.stats.lost_why = 0, f"导入失败：{str(e)[:300]}"
            return False

    def decide(self, frame: bytes | Obs, p: np.ndarray) -> int:
        """这个决策选哪个真实候选。p 为策略网络在这一帧上的概率（长度 = 候选数）。"""
        o = frame if isinstance(frame, Obs) else parse(frame)
        p = np.asarray(p, dtype=np.float64)
        greedy = int(np.argmax(p))
        self.stats.decisions += 1
        self.cmap = None
        order = np.argsort(-p)
        gate = o.num_actions > 1 and p[greedy] < self.P.tau and sum(1 for c in order[: self.P.k] if p[c] >= self.P.pmin) > 1
        if self.m is None:
            if gate:
                self.stats.fall(self.stats.lost_why or "镜像作废")
            return greedy
        if S.combat_over(self.m):
            if gate:
                self.stats.fall("镜像已结束")
            return greedy
        cmap, how = S.cand_map(o, self.m)
        why = S.compare(o, self.m) or (how if cmap is None else None)
        if why is not None and self.vocab:
            # 规则差异让镜像走岔了：按真实帧把镜像看得到的部分校正过来（服务的 sync 操作），再对一次；只让推演有偏差，不让搜索停掉
            try:
                self.m = self.mirror._all(self.sync_req(o))
                self.stats.synced += 1
                cmap, how = S.cand_map(o, self.m)
                why = S.compare(o, self.m) or (how if cmap is None else None)
                self.stats.sync_ok += why is None
            except EngineError as e:
                why = f"同步失败：{str(e)[:120]}"
        self.cmap = cmap
        if not gate:
            return greedy
        if why is not None:
            self.stats.fall(why)
            return greedy
        t0 = time.perf_counter()
        seed = 100003 * (self.seed * 4096 + self.k) + 7919 * (self.n + 1)
        if self.algo == "turn":
            info = T.turn_decide(self.mirror, self.ev, o, p, cmap, self.TP, seed)
            self.stats.search_s += time.perf_counter() - t0
            if not info.searched:
                self.stats.fall(info.err_text or "无锚点")
                return greedy
            self.stats.searched += 1
            self.stats.rollouts += info.nodes
            self.stats.changed += info.changed
            return int(info.best)
        info = S.pimc_decide(self.mirror, self.ev, o, p, cmap, self.P, 100003 * (self.seed * 4096 + self.k), self.n)
        self.stats.search_s += time.perf_counter() - t0
        if not info.searched:
            self.stats.fall(info.err_text or "无锚点")
            return greedy
        self.stats.searched += 1
        self.stats.rollouts += info.rollouts
        self.stats.changed += info.changed
        return int(info.best)

    def sync_req(self, o: Obs) -> dict:
        """按真实帧拼 sync 请求：玩家生命、格挡、能量，四个牌堆的牌（词表名 + 升级，按帧里的顺序），活着的敌人（词表名、生命、格挡）。"""
        name = lambda t: self.vocab[int(o.tok_ints[t, 1])]
        cards = lambda z: [[name(t), int(o.tok_feats[t, 3])] for t in
                           np.flatnonzero((o.tok_ints[:, 0] == TokenKind.CARD) & (o.tok_ints[:, 2] == z))]
        me = S._player_tok(o)
        g = np.flatnonzero(o.tok_ints[:, 0] == TokenKind.GLOBAL)
        return {"op": "sync", "obs": True,
                "player": {"hp": int(o.hp), "block": int(o.tok_feats[me, 2]) if me >= 0 else 0, "energy": int(o.tok_feats[g[0], 6]) if len(g) else 0},
                "piles": {"hand": cards(Zone.HAND), "discard": cards(Zone.DISCARD), "exhaust": cards(Zone.EXHAUST), "draw": cards(Zone.DRAW)},
                "enemies": [[name(t), int(o.tok_feats[t, 0]), int(o.tok_feats[t, 2])] for t in S._enemies(o)]}

    def observe(self, action: int) -> None:
        """真实对局执行了真实候选 action：镜像走对应的那一步（对应关系用 decide 时算好的；没调用 decide 的帧要先调 decide 或 sync）。"""
        self.n += 1
        if self.m is None:
            return
        if self.cmap is None:
            self.stats.lost_at, self.stats.lost_why = self.n, "动作对应不到镜像"
            self.m = None
            return
        try:
            self.m = self.mirror.step(self.cmap[int(action)])
        except EngineError as e:
            self.stats.lost_at, self.stats.lost_why = self.n, f"镜像走步报错：{str(e)[:200]}"
            self.m = None
        self.cmap = None

    def end(self) -> SearchStats:
        st, self.m, self.cmap = self.stats, None, None
        return st

    def close(self) -> None:
        self.mirror.close()
