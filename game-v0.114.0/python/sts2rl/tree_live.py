"""直播的 Boss 与精英战树搜索（2026-11-17；离线方法见 game-v0.113.0 分支 reports/整夜搜索探究总结.md 与 tools/future/run_search.py）。

用户（11-17）：“把 Boss 与精英战树搜索接进直播实机……只做好暂不上线等我指令”；
“新的搜索可以设置仅对第三幕boss开启/对所有boss开启/对所有boss和精英开启/对所有战斗开启”。

离线结论（整局实测，逐种子配对）：局外照常用主模型贪心，战斗里用同一个网络（整局模式）加 32 次模拟的闭环信息集树搜索，
L1000 在 3,000 个 P2-future 种子上 60.47% → 65.43%（Boss 与精英都搜，+4.97 ± 0.89）；只搜三幕 Boss +3.2 ± 1.4（1,000 个种子）。

这个文件把 run_search.py 的 RunTreeRunner（整局模式那一支）原样搬到直播策略端，配合 headless_search 的镜像：
  - 镜像：一个无头引擎进程跟着实机一步一步走，每个实机帧都和镜像帧比网络读的部分（头部除 12、各数组；候选文字不比，见 same_body）。
    战斗帧对得上才搜；局外帧对不上（第三幕审判、瓦库等事件的“事件动态数值”，引擎每次跑都不一样，只是显示）照样跟下去，
    战斗帧对不上则这一局余下不搜。直播录像 33 局 22,051 帧验收：对不上的 13 帧全在局外事件，285 场 Boss / 精英战全部逐位跟到打完。
  - 推演进程：带局面复制的引擎（Engine/Clone），每个进程槽位 0 存当前根：第一次 reset + 重放这一局实机走过的全部动作 + 快照，
    之后 restore + 只重放新走的几步 + 快照；每次模拟 restore 槽位 0 + reseed（换掉以后所有随机数、重洗抽牌堆）+ 走根动作。
    跟根的那个回复与镜像帧比对，不同就这次不搜（防止推演进程的局面与实机不同时还去搜）。
  - 搜索：同 run_search.RunTreeRunner（tree_search.TreeRunner 的整局版）。非根 Gumbel 补全 Q 选择、根上逐轮减半（同一组世界种子），
    最后在访问数 ≥ min_visits 的候选里取 argmax(log 先验 + Q/λ)；叶子只用价值头；回传 = 路上的奖励 + 末帧整局价值（死亡、卡住记 0）；
    战斗打完停在局外多候选帧时取这一帧的整局价值。先验、估值、根上的概率都用整局模式、清零特权特征（同离线 ModelEvaluator strip_priv）。
    决策种子 crc32("{run_seed}:{这一局已走步数}:{policy_seed}")，与离线相同，所以同一局面、同一网络时选的一样。
  - 搜索返回的键与贪心相同时用策略端原来的动作（0 次模拟时与贪心逐步相同）。

开法（play_live 的哨兵 sim_search.on，内容为 JSON，经 headless_search.HeadlessSearchAdvisor 转到这里，策略端不用重启）：
  {"method": "tree", "scope": "boss_elite"}          scope：act3_boss（只第三幕 Boss）、boss（三幕 Boss）、boss_elite（Boss 与精英，默认）、all（所有战斗）
  可选键：sims（32）、lam（0.06）、tree_procs（推演进程数，8 同离线）、budget_ms（一次搜索最多等多久，到点按贪心）、hang_ms、dll、policy_seed、log（每次搜索一行）
"""
from __future__ import annotations

import json
import math
import threading
import time
import zlib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np

from sts2rl import headless_search as H
from sts2env.env import EngineError
from sts2env.protocol import parse

import sim_pimc as X        # tools/（headless_search 已把它放进 sys.path）；cand_keys、check_items 与 game-v0.113.0 分支逐行相同

LIVE = Path.home() / ".local/share/sls2/live"
DEFAULT_DLL = LIVE / "tree_engine_4b4cae7" / "Sts2Headless.dll"      # game-v0.113.0 @ 4b4cae7 编的 v0.113.1 引擎（带局面复制）
DEFAULT_LOG = LIVE / "tree_search.jsonl"
MODE_RUN = 1
# 整局奖励（检查点配置里的 floor_reward 等；L1000 与第 24,181 版相同，2026-11-17 核对）。直播的网络对象不带配置，哨兵里可用 "rewards" 覆盖
DEFAULT_RW = {"floor_reward": 0.02, "act1_boss_reward": 0.1, "act2_boss_reward": 0.5, "act3_boss_reward": 1.5,
              "victory_reward": 2.0, "act3_boss2_reward": 2.0}
EXTRA = {"loop_cut": True, "stall_turns": 20}       # 推演进程同离线（run_search.EXTRA）：局面重复截断、20 回合无进展判输
# 搜索范围（用户 11-17）：房间类型与幕（从 0 起）
SCOPES = {
    "act3_boss": (("boss",), (2,)),
    "boss": (("boss",), ()),
    "boss_elite": (("boss", "elite"), ()),
    "all": (("monster", "elite", "boss"), ()),
}


@dataclass(frozen=True)
class TreeParams:
    """同 run_search.main 的默认值（--sims 32 --lam 0.06 --k 4 --tau 0.9，pmin 0.02，--min-visits 4，minmax，c_visit 50，c_scale 0.1）。"""
    sims: int = 32
    lam: float = 0.06
    k: int = 4
    tau: float = 0.9
    pmin: float = 0.02
    min_visits: int = 4
    c_visit: float = 50.0
    c_scale: float = 0.1
    qnorm: str = "minmax"
    value_range: float = 5.0
    max_depth: int = 200
    max_steps: int = 600        # sim_pimc.PimcParams.max_steps：一次模拟超过这么多步就在那一帧估值
    policy_seed: int = 0

    @classmethod
    def from_dict(cls, d: dict) -> "TreeParams":
        names = {f.name: f.type for f in fields(cls)}
        kw = {}
        for k, v in d.items():
            if k in names and v is not None:
                kw[k] = str(v) if k == "qnorm" else (int(v) if names[k] == "int" else float(v))
        return cls(**kw)


def tree_config(d: dict) -> dict:
    """哨兵内容 → HeadlessSearchConfig 认的字典：scope 换成 rooms / acts，默认推演进程 8 个、预算放宽。"""
    d = dict(d or {})
    scope = str(d.get("scope") or "boss_elite")
    if scope not in SCOPES:
        raise ValueError(f"scope 只能是 {list(SCOPES)}：{scope}")
    d["rooms"], d["acts"] = SCOPES[scope]
    # 推演进程数影响树里的并发虚拟访问，与离线（--procs 8）一致才能套用离线的数字；play_live 给无头搜索的默认 procs 是 6，所以单独用 tree_procs
    d["procs"] = int(d.get("tree_procs") or 8)
    d.setdefault("budget_ms", 30000)
    d.setdefault("hang_ms", 90000)
    if not d.get("dll"):
        d["dll"] = str(DEFAULT_DLL)
    d["scope"] = scope
    return d


def same_body(a: bytes | None, b: bytes | None) -> bool:
    """网络读的部分逐位相同：头部（跳过 12，同 headless_search.same_frame）与 tok_ints、tok_feats、tok_priv、cand_ints、cand_feats。
    候选文字不比（模组发游戏语言的文字，引擎有些是英文，如先古之民的选项），附加信息不比。"""
    if H.same_frame(a, b):
        return True
    if a is None or b is None:
        return False
    try:
        x, y = parse(a), parse(b)
    except Exception:
        return False
    hx, hy = x.header.copy(), y.header.copy()
    if hx.shape != hy.shape:
        return False
    hx[12] = hy[12] = 0
    if not np.array_equal(hx, hy):
        return False
    for n in ("tok_ints", "tok_feats", "tok_priv", "cand_ints", "cand_feats"):
        u, v = getattr(x, n), getattr(y, n)
        if u.shape != v.shape or not np.array_equal(u, v):
            return False
    return True


# ---------------------------------------------------------------- 树（同 tools/future/tree_search.py）

class Node:
    __slots__ = ("N", "Nq", "W", "out", "v0", "prior", "expanded")

    def __init__(self):
        self.N: dict = {}
        self.Nq: dict = {}
        self.W: dict = {}
        self.out: dict = {}
        self.v0 = 0.0
        self.prior: dict = {}
        self.expanded = False

    def q(self, k) -> float:
        n = self.Nq.get(k, 0)
        return self.W[k] / n if n else float("nan")


def frame_keys(o) -> tuple[list, dict]:
    keys = [k[:-1] for k in X.cand_keys(o)]
    first: dict = {}
    for i, k in enumerate(keys):
        first.setdefault(k, i)
    return keys, first


def signature(o, keys: list) -> tuple:
    c = X.check_items(o)
    return (int(o.decision), c["hp"], c["block"], c["energy"], tuple(c["hand"]), tuple(c["enemies"]), tuple(sorted(map(repr, keys))))


def merged_prior(p: np.ndarray, keys: list) -> dict:
    out: dict = {}
    for i, k in enumerate(keys):
        out[k] = out.get(k, 0.0) + float(p[i])
    return out


class _Timeout(Exception):
    pass


class TreeSearchAdvisor(H.HeadlessSearchAdvisor):
    """接口同 HeadlessSearchAdvisor（play_live 的胶水层不用改）：on_run_info、choose、observe、reset_run、engaged、gated、close。"""

    def __init__(self, config=None, get_net=None, *, net=None, clock=time.monotonic, logger=None):
        d = tree_config(config if isinstance(config, dict) else {})
        self.T = TreeParams.from_dict(d)
        self.scope = d["scope"]
        self.log_path = d.get("log", str(DEFAULT_LOG))
        self.mirror_extra = dict(d.get("mirror_extra") or {})     # 测试用：镜像的 reset 另加的键（直播不用）
        self._sum = {"searches": 0, "changed": 0, "ms": []}
        self._rw = {**DEFAULT_RW, **{k: float(v) for k, v in (d.get("rewards") or {}).items() if k in DEFAULT_RW}}
        self.wroot: list = []         # 先于父类起预热线程设好（线程里会改它）
        # 父类在引擎文件不在时提前返回（只设了 init_error），close() 与 stats 要先有
        self._workers, self._mirror, self._closed = [], None, False
        self.stats = {"frames": 0, "synced": 0, "resync": 0, "desync": 0, "searches": 0, "changed": 0}
        super().__init__(d, get_net, net=net, clock=clock, logger=logger)
        self._run_no = 0
        self._exact = False           # 这一帧与镜像逐位相同（只在这时搜）
        self._fight = None            # (幕, 层, 房间)
        self._act3_bosses = 0
        self._ctx: dict = {}
        self._root_raw: bytes | None = None
        self._root_bad = 0
        self.stats.update(display_mismatch=0, root_mismatch=0, timeouts=0)

    # ------------------------------------------------------------------ 进程

    def _start_workers(self) -> None:
        try:
            ws = [H._Worker(self._dll, self.config.engine_log) for _ in range(max(1, int(self.config.procs)))]
            # 预热（同 boss_pimc.Runner.warm）：.NET 第一次跑要 JIT，第一次拍快照要找一遍静态根，先各走一遍；不影响之后的结果（槽位 0 作废）
            warm = {"op": "reset", "mode": "run", "character": "FOOL", "ascension": 10, "seed": "TREEWARM", "labels": False,
                    "replay": [0] * 30, "reseed": 1, "first": 0, "snapshot": 0, **EXTRA}
            for w in ws:
                w.send(warm)
                w.send({"op": "restore", "slot": 0, "reseed": 2, "first": 0})
                w.send({"op": "drop"})
            for w in ws:
                for _ in range(3):
                    w.recv()
            self._workers = ws
            self.wroot = [None] * len(ws)
        except Exception as e:
            self.init_error = f"推演进程起不来：{type(e).__name__}: {e}"

    def _restart_workers(self) -> None:
        self.wroot = []
        super()._restart_workers()

    # ------------------------------------------------------------------ 胶水层接口

    def on_run_info(self, info: dict) -> None:
        if self._closed or self.init_error:
            return
        old = self._run
        super().on_run_info(info)
        if self._run is old:
            return          # 同一局的局信息重发（重连等）：照常
        self._run_no += 1
        self._fight, self._act3_bosses, self._ctx, self._exact = None, 0, {}, bool(self._tracked)
        if self._run is not None and self.mirror_extra and self._tracked:
            # 测试：镜像按另加的键重开（与“游戏”同一套 reset）
            self._run = {**self._run, **self.mirror_extra}
            try:
                self._mframe = self._mirror_req(self._run)
            except Exception as e:
                self._lose(f"镜像开局失败：{type(e).__name__}")

    def reset_run(self) -> None:
        super().reset_run()
        self._fight, self._act3_bosses, self._ctx, self._exact = None, 0, {}, False

    def prime(self, actions: list[int]) -> bytes | None:
        r = super().prime(actions)
        self._exact = r is not None
        return r

    def choose(self, obs, probs, raw: bytes | None = None, fallback_action: int | None = None) -> tuple[int, dict]:
        t0 = self._clock()
        n = int(obs.num_actions)
        p_in = np.asarray(list(probs)[:n], dtype=float)
        greedy = int(fallback_action) if fallback_action is not None else int(np.argmax(p_in))
        self._sync_tree(raw, obs)
        if not obs.in_combat:
            self._fight = None
            self._combat = None
            return greedy, self._info(greedy, greedy, t0, reason="not_combat")
        room, act, floor = H.room_of(obs), int(obs.act), int(obs.floor)
        key = (act, floor, room)
        if key != self._fight:          # 新的一场（同 run_search.play_run 在战斗开始帧上记的 fight_ctx）
            self._fight = key
            if room == "boss" and act == 2:
                self._act3_bosses += 1
            self._ctx = {"boss3_first": room == "boss" and act == 2 and self._act3_bosses == 1, "floor0": floor}
        if not self.config.wants(room, act):
            self._combat = None
            return greedy, self._info(greedy, greedy, t0, reason="room_filtered")
        self._combat = key
        if not self._tracked or not self._exact:
            return greedy, self._info(greedy, greedy, t0, reason="untracked")
        if self._disabled:
            return greedy, self._info(greedy, greedy, t0, reason="disabled")
        if self.init_error:
            return greedy, self._info(greedy, greedy, t0, reason=f"init_error: {self.init_error}")
        if n <= 1:
            return greedy, self._info(greedy, greedy, t0, reason="single_action")
        # 根上的概率：同离线（整局模式、清零特权、double softmax），与策略端算的只差浮点末位
        try:
            logits, _ = self._forward([obs])
        except Exception as e:
            return greedy, self._info(greedy, greedy, t0, reason=f"forward_error: {type(e).__name__}")
        import torch
        p = torch.softmax(logits[0, :n].double(), -1).numpy()
        if not self._gated(p):
            return greedy, self._info(greedy, greedy, t0, reason="confident")
        if self._warm.is_alive():
            self._warm.join(timeout=max(0.0, self.config.budget_ms / 1000.0))
            if self._warm.is_alive() or not self._workers:
                return greedy, self._info(greedy, greedy, t0, reason="workers_not_ready")
        self._dec += 1
        seed = str(self._run.get("seed"))
        dec_seed = zlib.crc32(f"{seed}:{len(self._hist)}:{self.T.policy_seed}".encode()) & 0x3FFFFFFF
        self._root_raw = self._mframe
        try:
            res = self._search_tree(obs, p, dec_seed, t0)
        except Exception as e:
            if isinstance(e, _Timeout):
                self.stats["timeouts"] += 1
            self._fail(f"{type(e).__name__}: {str(e)[:200]}")
            self._restart_workers()
            return greedy, self._info(greedy, greedy, t0, reason=f"search_error: {type(e).__name__}")
        self._failures = 0
        self.stats["searches"] += 1
        keys = [k[:-1] for k in X.cand_keys(obs)]
        g_mine = int(np.argmax(p))
        act_out = greedy
        if keys[int(res["best"])] != keys[g_mine]:
            act_out = int(res["best"])
            self.stats["changed"] += 1
        res["changed"] = act_out != greedy
        self._log_search(obs, room, res, greedy, act_out, t0)
        return act_out, self._info(act_out, greedy, t0, searched=True, extra=res)

    # ------------------------------------------------------------------ 同步（局外帧对不上照样跟）

    def _sync_tree(self, raw: bytes | None, obs) -> None:
        if not self._tracked or raw is None:
            self._exact = False
            return
        self.stats["frames"] += 1
        if same_body(raw, self._mframe):
            if self._pending is not None:
                self._hist.append(self._pending)
                self._pending = None
            self.stats["synced"] += 1
            self._exact = True
            return
        if self._pending is not None and same_body(raw, self._mprev):
            # 模组没执行那个动作（过时被丢弃，同一局面又发了一次）：镜像按历史重建到走之前
            self.stats["resync"] += 1
            self._pending = None
            try:
                self._mframe = self._mirror_req({**self._run, "replay": list(self._hist)})
                if same_body(self._mframe, raw):
                    self._exact = True
                    return
            except Exception:
                pass
        self._exact = False
        if not obs.in_combat:
            # 局外显示数值不同（事件动态数值等）：认这一步，接着跟；之后的战斗帧对得上才会搜
            if self._pending is not None:
                self._hist.append(self._pending)
                self._pending = None
            self.stats["display_mismatch"] += 1
            return
        self.stats["desync"] += 1
        self._lose(f"第 {len(self._hist)} 步战斗帧与镜像对不上")

    # ------------------------------------------------------------------ 前向（同离线 sim_pimc.ModelEvaluator，strip_priv）

    def _forward(self, frames: list, modes: list[int] | None = None):
        import torch
        from sts2rl.features import Ragged, collate_ragged
        net = self._get_net()
        r = Ragged.from_obs(frames, np.array(modes or [MODE_RUN] * len(frames)))
        r.tok_priv[:] = 0
        with torch.no_grad():
            logits, values = net(collate_ragged(r).to(self.config.device))
        return logits.float().cpu(), values.float().cpu()

    # ------------------------------------------------------------------ 搜索（同 tree_search.TreeRunner.search，整局版）

    def _gated(self, p: np.ndarray) -> bool:
        n = len(p)
        if n <= 1:
            return False
        order = np.argsort(-p)
        return bool(p[order[0]] < self.T.tau and sum(1 for k in order[: self.T.k] if p[k] >= self.T.pmin) > 1)

    def gated(self, probs, n: int) -> bool:
        try:
            return self._gated(np.asarray(list(probs)[:n], dtype=float))
        except Exception:
            return False

    def _search_tree(self, root, p: np.ndarray, dec_seed: int, t0: float) -> dict:
        T = self.T
        self._deadline = t0 + self.config.budget_ms / 1000.0
        hard = threading.Timer(self.config.hang_ms / 1000.0, self._kill_workers)
        hard.daemon = True
        hard.start()
        try:
            return self._search_inner(root, p, dec_seed)
        finally:
            hard.cancel()

    def _search_inner(self, root, p: np.ndarray, dec_seed: int) -> dict:
        T = self.T
        ts = time.perf_counter()
        keys, first = frame_keys(root)
        prior = merged_prior(p, keys)
        greedy_idx = int(np.argmax(p))
        gk = keys[greedy_idx]
        order = sorted(prior, key=lambda k: -prior[k])
        cands = [k for k in order[: T.k] if prior[k] >= T.pmin]
        if gk not in cands:
            cands = [gk] + cands[: T.k - 1]
        if len(cands) <= 1 or T.sims <= 0:
            return {"best": greedy_idx, "greedy": greedy_idx, "cands": [first[k] for k in cands], "sims": 0, "ms": 0.0}
        rootn = Node()
        rootn.prior = prior
        rootn.expanded = True
        self.tstat = {"sims": 0, "evals": 0, "term": 0, "depth": 0, "maxd": 0, "err": 0, "nodes": 0, "chance": 0, "post": 0}
        self._root_bad = 0
        ret: dict = {k: {} for k in cands}
        alive = list(cands)
        phases = max(1, math.ceil(math.log2(len(cands))))
        used = 0
        for ph in range(phases):
            per = max(1, (T.sims // phases) // len(alive))
            seeds = [dec_seed + used + j for j in range(per)]
            used += per
            self._simulate([{"seed": s, "key0": k} for s in seeds for k in alive], rootn, first, root, ret)
            if self._root_bad:
                raise RuntimeError("推演进程跟到的根局面与镜像不同")
            if len(alive) <= 2 or ph == phases - 1:
                continue
            rest = [k for k in alive if k != gk]
            sc = np.array([np.mean(list(ret[k].values())) if ret[k] else -9.0 for k in rest])
            keep = [rest[i] for i in np.argsort(-sc, kind="stable")[: max(1, math.ceil(len(alive) / 2) - 1)]]
            alive = [gk] + keep
        if self.tstat["sims"] == 0:
            raise RuntimeError(f"{self.tstat['err']} 次模拟全部出错")
        best = gk
        sc = {k: math.log(max(prior[k], 1e-6)) + rootn.q(k) / T.lam for k in cands if rootn.Nq.get(k, 0) >= T.min_visits}
        if sc:
            best = max(sc, key=lambda k: (sc[k], k == gk))
        st = self.tstat
        q = {first[k]: (round(rootn.q(k), 4), rootn.Nq.get(k, 0)) for k in cands if rootn.Nq.get(k, 0)}
        ms = (time.perf_counter() - ts) * 1000.0
        return {"best": int(first[best]), "greedy": greedy_idx, "cands": [first[k] for k in cands], "q": {str(k): v for k, v in q.items()},
                "sims": st["sims"], "rollouts": st["sims"], "evals": st["evals"], "term": st["term"], "err": st["err"], "nodes": st["nodes"],
                "post": st["post"], "depth": round(st["depth"] / max(1, st["sims"]), 2), "maxd": st["maxd"], "search_ms": round(ms, 1),
                "errors": st["err"], "cut": False}

    # ---- 推演进程：跟根与派世界（同 boss_pimc.Runner 的 --worlds clone、--auto-single）

    def _world_reqs(self, j: int, w: dict) -> list[dict]:
        base = {**self._run, "labels": False, "snapshots": True, **EXTRA}
        base.pop("replay", None)
        for k in self.mirror_extra:
            base.setdefault(k, self.mirror_extra[k])
        hist = self._hist
        key = (self._run_no, str(self._run.get("seed")))
        rs = {"reseed": int(w["seed"]) & 0xFFFFFFFF, "first": int(w["cand"]), "auto_single": True}
        pre = []
        st = self.wroot[j]
        if st is None or st[0] != key or st[1] > len(hist):
            pre.append({**base, "replay": list(hist), "snapshot": 0})
        elif st[1] < len(hist):
            pre.append({"op": "restore", "slot": 0, "replay": list(hist[st[1]:]), "snapshot": 0})
        self.wroot[j] = (key, len(hist))
        return pre + [{"op": "restore", "slot": 0, **rs}]

    def _dispatch(self, j: int, w: dict) -> None:
        w.update(left=1.0, steps=0, frame=None)
        reqs = self._world_reqs(j, w)
        w["skip"] = len(reqs) - 1
        for r in reqs:
            self._workers[j].send(r)

    def _take(self, j: int, w: dict) -> str:
        r = self._workers[j].recv()
        err = str(r)[:200] if isinstance(r, EngineError) else None
        if w["skip"] > 0:
            w["skip"] -= 1
            if err:
                w["error"] = err
                self.wroot[j] = None
            elif not same_body(r, self._root_raw):
                # 跟到的根与镜像（= 实机）不同：这个进程的槽位不可信，这次搜索作废
                w["error"] = "根局面不同"
                self.wroot[j] = None
                self._root_bad += 1
                self.stats["root_mismatch"] += 1
            return "skip"
        if w.get("error"):
            return "dead"
        if err:
            w["error"] = err
            self.wroot[j] = None
            return "dead"
        o = parse(r)
        w["frame"] = o
        n = 1
        if isinstance(o.extra, dict):
            n += int(o.extra.get("auto_steps", 0))
        w["steps"] += n
        return "frame"

    def _path_reward(self, w: dict, o) -> None:
        rw, ctx = self._rw, self._ctx
        if o.floor > w["floor"]:
            w["acc"] += rw["floor_reward"] * (o.floor - w["floor"])
        for act in range(w["act"] + 1, o.act + 1):
            w["acc"] += {1: rw["act1_boss_reward"], 2: rw["act2_boss_reward"]}.get(act, 0.0)
        if ctx.get("boss3_first") and not w.get("b3") and not o.done and (not o.in_combat or o.floor > ctx["floor0"]):
            w["acc"] += rw["act3_boss_reward"]; w["b3"] = True
        if o.done and o.victory and not o.stuck:
            w["acc"] += rw["victory_reward"] + rw["act3_boss2_reward"]
        w["floor"], w["act"] = max(w["floor"], int(o.floor)), max(w["act"], int(o.act))

    def _check_time(self) -> None:
        if self._clock() > self._deadline:
            raise _Timeout(f"超过 {self.config.budget_ms} 毫秒")

    def _simulate(self, sims, rootn, first, root, ret) -> None:
        """同 run_search.RunTreeRunner._simulate（整局模式）：锁步，每一轮把空闲进程派满，按进程顺序把忙着的都读一遍。"""
        import torch
        T = self.T
        queue = list(sims)
        busy: dict[int, dict] = {}
        while queue or busy:
            self._check_time()
            for j in range(len(self._workers)):
                if j in busy or not queue:
                    continue
                w = queue.pop(0)
                k0 = w["key0"]
                rootn.N[k0] = rootn.N.get(k0, 0) + 1
                w.update(cand=first[k0], edge=(rootn, k0), path=[(rootn, k0)], acc=0.0, floor=int(root.floor), act=int(root.act))
                self._dispatch(j, w)
                busy[j] = w
            if not busy:
                break
            leaves, posts = [], []
            for j in list(busy):
                w = busy[j]
                got = self._take(j, w)
                while got == "skip":
                    got = self._take(j, w)
                if got == "dead":
                    self._abort(w)
                    del busy[j]
                    continue
                o = w["frame"]
                self._path_reward(w, o)
                if o.done or o.stuck or o.num_actions == 0:
                    self._backup(w, w["acc"], ret)
                    self.tstat["term"] += 1
                    del busy[j]
                    continue
                if not o.in_combat or w["steps"] > T.max_steps:
                    posts.append(j)
                    continue
                keys, fk = frame_keys(o)
                parent, k = w["edge"]
                kids = parent.out.setdefault(k, {})
                sig = signature(o, keys)
                node = kids.get(sig)
                if node is None:
                    node = kids[sig] = Node()
                    self.tstat["nodes"] += 1
                    if len(kids) > 1:
                        self.tstat["chance"] += 1
                w["node"], w["keys"] = node, keys
                if not node.expanded or len(w["path"]) >= T.max_depth:
                    leaves.append(j)
                    continue
                k = self._select(node, list(fk))
                node.N[k] = node.N.get(k, 0) + 1
                w["edge"] = (node, k)
                w["path"].append((node, k))
                self._workers[j].send({"op": "step", "action": int(fk[k]), "auto_single": True})
            js = leaves + posts
            if js:
                frames = [busy[j]["frame"] for j in js]
                logits, values = self._forward(frames)
                self.tstat["evals"] += len(frames)
                for i, j in enumerate(js):
                    w = busy.pop(j)
                    v = float(values[i])
                    if i < len(leaves):
                        node, o = w["node"], w["frame"]
                        if not node.expanded:
                            n = int(o.num_actions)
                            pr = torch.softmax(logits[i, :n].double(), -1).numpy()
                            node.prior = merged_prior(pr, w["keys"])
                            node.v0 = w["acc"] + v
                            node.expanded = True
                    else:
                        self.tstat["post"] += 1
                    self._backup(w, w["acc"] + v, ret)

    def _completed(self, node: Node, legal: list):
        lp = np.array([math.log(max(node.prior.get(k, 1e-6), 1e-6)) for k in legal])
        Nk = np.array([node.N.get(k, 0) for k in legal], np.float64)
        Qk = np.array([node.q(k) for k in legal])
        vis = ~np.isnan(Qk)
        sum_n = float(sum(node.Nq.get(k, 0) for k in legal))
        if vis.any():
            pr = np.exp(lp[vis]); pr = pr / max(pr.sum(), 1e-12)
            vmix = (node.v0 + sum_n * float((pr * Qk[vis]).sum())) / (sum_n + 1.0)
        else:
            vmix = node.v0
        return lp, np.where(vis, Qk, vmix), Nk

    def _sigma(self, cq: np.ndarray, Nk: np.ndarray) -> np.ndarray:
        T = self.T
        if T.qnorm == "minmax":
            lo, hi = float(cq.min()), float(cq.max())
            qn = (cq - lo) / max(hi - lo, 1e-8)
        else:
            qn = cq / T.value_range
        return (T.c_visit + float(Nk.max() if len(Nk) else 0.0)) * T.c_scale * qn

    def _select(self, node: Node, legal: list):
        lp, cq, Nk = self._completed(node, legal)
        z = lp + self._sigma(cq, Nk)
        pi = np.exp(z - z.max()); pi /= pi.sum()
        score = pi - Nk / (1.0 + Nk.sum())
        return legal[int(np.argmax(score))]

    def _backup(self, w: dict, v: float, ret: dict) -> None:
        for node, k in w["path"]:
            node.Nq[k] = node.Nq.get(k, 0) + 1
            node.W[k] = node.W.get(k, 0.0) + v
        ret[w["key0"]][w["seed"]] = v
        d = len(w["path"])
        self.tstat["sims"] += 1
        self.tstat["depth"] += d
        self.tstat["maxd"] = max(self.tstat["maxd"], d)

    def _abort(self, w: dict) -> None:
        for node, k in w.get("path", []):
            node.N[k] -= 1
        self.tstat["err"] += 1

    # ------------------------------------------------------------------ 记录与侧栏

    def _log_search(self, obs, room: str, res: dict, greedy: int, act: int, t0: float) -> None:
        ms = (self._clock() - t0) * 1000.0
        self._sum["searches"] += 1
        self._sum["changed"] += int(act != greedy)
        self._sum["ms"].append(ms)
        if not self.log_path:
            return
        try:
            row = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), "seed": self._run.get("seed"), "step": len(self._hist), "act": int(obs.act),
                   "floor": int(obs.floor), "room": room, "scope": self.scope, "sims": res.get("sims"), "greedy": greedy, "action": act,
                   "changed": act != greedy, "cands": res.get("cands"), "q": res.get("q"), "ms": round(ms, 1), "search_ms": res.get("search_ms"),
                   "evals": res.get("evals"), "err": res.get("err")}
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _info(self, action: int, greedy: int, t0: float, *, reason: str | None = None, searched: bool = False, extra: dict | None = None) -> dict:
        d = super()._info(action, greedy, t0, reason=reason, searched=searched, extra=extra)
        d["backend"] = "tree"
        if searched:
            extra = extra or {}
            d["tier"] = "tree"
            d["summary"] = (f"树搜索 {extra.get('sims', 0)} 次模拟，{d['elapsed_ms']:.0f}ms，动作 {action}（贪心 {greedy}{'，改选' if d['changed'] else ''}）"
                            + (f"，{extra.get('err')} 次出错" if extra.get("err") else ""))
        d["synced"] = bool(self._tracked and self._exact)
        return d
