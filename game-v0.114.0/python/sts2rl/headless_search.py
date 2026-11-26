"""直播战斗搜索的无头引擎后端（2026-11-14，游戏 v0.113.0 起；sts2-sim 还没有新版本）。

用户：“目前还没有新版本的模拟器，我想知道搜索可不可以用我们的引擎来做？慢一点没事。”“那你现在就把它做好吧。”

做法：推演用我们自己的无头引擎（engine/Sts2Headless，跑的就是游戏本身的代码，规则与实机一致；实机帧与它逐字节相同，见 实机桥接.md）。
引擎不能复制战斗状态，所以每个“世界”都从这一局开头重放：
    reset（同一个角色、进阶、模式、种子）+ replay（这一局到现在实机走过的全部动作）→ 正好是实机的当前局面
    → reseed（换掉以后所有的随机数、重洗抽牌堆：只换玩家看不到的未来）→ first（走要比较的那个候选）→ 之后用模型推演
算法照 tools/sim_pimc.py 的 PIMC 改选（决策记录 217，直播 sts2-sim 搜索用的同一套门限、候选、逐轮减半、配对检验与叶子估值），
只把“镜像”换成重放：每个工作进程一次只推一个世界，推完再接下一个；各进程同时算，网络的前向合成一批。

公平性（2026-11-14 核对 v0.113.0 反编译源码）：战斗里的随机数都来自整局随机数组（抽牌、洗牌、生成卡牌与药水、敌人行动、目标、充能球……），
reseed 全部换掉；遭遇自带的随机数只在开战前生成敌人阵容时用；玩家自己的随机数组（奖励、商店、变化、战斗外生成）战斗里不用。
种子在游戏画面右上角本来就显示；重放用真实种子只是为了还原“已经发生的过去”，推演的未来一律是 reseed 之后的新随机数。
已知局限：被效果放到抽牌堆顶、玩家其实知道的牌，reseed 也一并打乱了（只会让搜索略保守）。

同步：一个“镜像”进程跟着实机一步一步走（同样的 reset 参数，带候选文字和路线摘要，与模组发的帧逐字节可比）。每个实机帧都和镜像的帧比原始字节：
相同才算对得上；动作被模组丢弃（同一帧再发一次）时退掉最后一个动作重建；别的不一致（游戏读档接上、中途连上、规则差异）这一局余下不搜，按贪心。

用法同 sim_search_backend.SimSearchAdvisor（play_live 的胶水层不用改），多一个 on_run_info：

    adv = HeadlessSearchAdvisor(cfg, get_net=lambda: net)
    adv.on_run_info(info)                          # 模组的局信息（character、ascension、run_seed、game_mode、loaded_floor）
    act, info = adv.choose(obs, probs, raw=m.raw, fallback_action=act)    # 每个观测帧都调
    link.act(seq, act); adv.observe(act)          # 真正发出去的动作
    adv.reset_run(); adv.close()
"""
from __future__ import annotations

import json
import logging
import math
import os
import sys
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))

from sts2env.env import EngineError, _Process  # noqa: E402
from sts2env.protocol import DecisionKind, TokenKind, parse  # noqa: E402

DEFAULT_DLL = REPO / "engine" / "Sts2Headless" / "bin" / "Release" / "net9.0" / "Sts2Headless.dll"
ROOM_KIND = {1: "monster", 2: "elite", 3: "boss"}       # 全局记录特征 14
COMBAT_ROOMS = ("monster", "elite", "boss")
QUIET_REASONS = ("not_combat", "confident", "single_action", "room_filtered", "untracked", "disabled")

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class HeadlessSearchConfig:
    """开关、房间、预算、进程与 PIMC 参数（pimc 里的键覆盖 sim_pimc.PimcParams：tau、k、pmin、sims0、rounds、z、leaf、max_steps）。"""
    enabled: bool = True
    rooms: tuple = ("boss",)          # 只在这些房间类型的战斗里搜索
    acts: tuple = (2,)                # 只在这几幕（从 0 起，2 为第三幕）；空为不限
    budget_ms: int = 30000            # 一次搜索最多花多久：到点不再开新的世界，按已推完的比较（推完的不够就回贪心）
    hang_ms: int = 90000              # 一次搜索超过它视为卡死：杀掉工作进程，这次回贪心，下次重建
    procs: int = 6                    # 推演用几个工作进程
    max_consecutive_failures: int = 3
    seed: int = 0
    device: str = "cpu"
    dll: str | None = None
    engine_log: str | None = None     # 引擎的标准错误追加到这里
    pimc: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict | None) -> "HeadlessSearchConfig":
        d = dict(d or {})
        rooms = d.get("rooms", cls.rooms)
        if isinstance(rooms, str):
            rooms = tuple(x.strip() for x in rooms.split(",") if x.strip())
        acts = d.get("acts", cls.acts)
        if isinstance(acts, str):
            acts = [x.strip() for x in acts.split(",") if x.strip()]
        kw = {k: d[k] for k in ("enabled", "budget_ms", "hang_ms", "procs", "max_consecutive_failures", "seed", "device", "dll", "engine_log")
              if d.get(k) is not None}
        if d.get("sim_log") and "engine_log" not in kw:
            kw["engine_log"] = d["sim_log"]
        bad = [r for r in rooms if r not in COMBAT_ROOMS]
        if bad:
            raise ValueError(f"rooms 只能是 {COMBAT_ROOMS}：{bad}")
        return cls(rooms=tuple(rooms), acts=tuple(int(x) for x in (acts or ())), pimc=dict(d.get("pimc") or {}), **kw)

    def wants(self, room: str, act: int) -> bool:
        return room in self.rooms and (not self.acts or act in self.acts)


def room_of(o) -> str:
    """这一帧所在房间的类型（全局记录特征 14）。"""
    k = np.flatnonzero(o.tok_ints[:, 0] == TokenKind.GLOBAL)
    return ROOM_KIND.get(int(o.tok_feats[k[0], 14]), "other") if len(k) else "other"


# 头部 12 是“本进程赢下的战斗数”（引擎 UiFlows.CombatsWon、模组 Shims.UiFlows.CombatsWon，训练端靠它的增量认战斗结束），跨局累计、各进程各算各的：
# 实机打过几局之后，模组发的这一项就比镜像（这一局才开）大。模型不读这一项（features.py 只用头部 0、1、11、16–34），比帧时跳过这 4 个字节，其余逐字节比
# （2026-11-14 离线配对评测查出：同一进程先打一局再开下一局，只有头部 12 不同）
_H12 = slice(12 * 4, 13 * 4)


def same_frame(a: bytes | None, b: bytes | None) -> bool:
    if a is None or b is None or len(a) != len(b):
        return False
    return a[:_H12.start] == b[:_H12.start] and a[_H12.stop:] == b[_H12.stop:]


class _Worker:
    """一个引擎进程：请求可以先写出、回复再读；read 带超时（卡死时由搜索的看门狗杀进程）。"""

    def __init__(self, dll: Path, log_path: str | None):
        self.log = open(log_path, "ab") if log_path else open(os.devnull, "wb")
        self.p = _Process(dll, self.log, "csharp")

    def send(self, req: dict) -> None:
        self.p.send(req)

    def recv(self) -> bytes | EngineError:
        try:
            return self.p.recv()
        except EngineError as e:
            return e

    def kill(self) -> None:
        try:
            self.p.p.kill()
        except Exception:
            pass

    def close(self) -> None:
        try:
            self.p.close()
        except Exception:
            self.kill()
        try:
            self.log.close()
        except Exception:
            pass


@dataclass
class _World:
    cand: int
    seed: int
    steps: int = -1
    left: float = 1.0
    frame: Any = None
    score: float | None = None
    error: str | None = None


class HeadlessSearchAdvisor:
    def __new__(cls, config=None, *args, **kwargs):
        # 哨兵内容带 "method": "tree" 时换成树搜索（sts2rl/tree_live.py，2026-11-17）。play_live 只认这个类名，
        # 这样不改 play_live、不重启策略端，写好哨兵就能用上
        if cls is HeadlessSearchAdvisor and isinstance(config, dict) and config.get("method") == "tree":
            from sts2rl.tree_live import TreeSearchAdvisor
            return super().__new__(TreeSearchAdvisor)
        return super().__new__(cls)

    def __init__(self, config=None, get_net: Callable[[], Any] | None = None, *, net=None, clock=time.monotonic, logger=None):
        self.config = config if isinstance(config, HeadlessSearchConfig) else HeadlessSearchConfig.from_dict(config)
        self._get_net = get_net or (lambda: net)
        self._clock = clock
        self._log = logger or log
        self.init_error: str | None = None
        self._closed = False
        self._dll = Path(self.config.dll) if self.config.dll else DEFAULT_DLL
        if not self._dll.exists():
            self.init_error = f"无头引擎不在：{self._dll}"
            return
        from sim_pimc import PimcParams, ModelEvaluator, combat_over, enemy_left, outcome, terminal  # noqa: E402
        self._pimc = dict(PimcParams=PimcParams, combat_over=combat_over, enemy_left=enemy_left, outcome=outcome, terminal=terminal)
        self._ModelEvaluator = ModelEvaluator
        self.P = replace(PimcParams(), **{k: v for k, v in self.config.pimc.items() if hasattr(PimcParams, k)})
        self._workers: list[_Worker] = []
        self._mirror: _Worker | None = None
        self._warm = threading.Thread(target=self._start_workers, daemon=True)
        self._warm.start()
        self._run: dict | None = None     # reset 的参数（不含 replay）
        self._hist: list[int] = []        # 这一局实机走过的全部动作（按模组执行的顺序）
        self._tracked = False             # 镜像与实机对得上
        self._mframe: bytes | None = None     # 镜像当前的帧（应等于下一个实机帧）
        self._mprev: bytes | None = None      # 走上一个动作之前的帧（动作被丢弃时实机会再发它）
        self._pending: int | None = None      # 已发给镜像、还没被下一个实机帧确认的动作
        self._failures = 0
        self._disabled = False
        self._combat = None               # 正在一场要搜的战斗里（sim_poll 据此不在战斗中途换顾问）
        self._dec = 0
        self.stats = {"frames": 0, "synced": 0, "resync": 0, "desync": 0, "searches": 0, "changed": 0}
        self.last_desync: str | None = None

    # ------------------------------------------------------------------ 进程

    def _start_workers(self) -> None:
        try:
            for _ in range(max(1, int(self.config.procs))):
                self._workers.append(_Worker(self._dll, self.config.engine_log))
            # 预热：.NET 第一次跑重放要好几秒（JIT），先让每个进程各开一局走几十步
            for w in self._workers:
                w.send({"op": "reset", "mode": "run", "character": "DEFECT", "ascension": 10, "seed": "HLWARM", "replay": [0] * 60, "reseed": 1, "first": 0})
            for w in self._workers:
                w.recv()
        except Exception as e:
            self.init_error = f"工作进程起不来：{type(e).__name__}: {e}"

    def _ensure_mirror(self) -> _Worker:
        if self._mirror is None:
            self._mirror = _Worker(self._dll, self.config.engine_log)
        return self._mirror

    def _mirror_req(self, req: dict) -> bytes:
        m = self._ensure_mirror()
        m.send(req)
        r = m.recv()
        if isinstance(r, EngineError):
            raise r
        return r

    # ------------------------------------------------------------------ 胶水层接口

    def on_run_info(self, info: dict) -> None:
        """模组的局信息。新的一局（loaded_floor 为 0、有 run_seed）：镜像从头开局；从存档接上的（loaded_floor > 0）：历史接不上，这一局不搜。"""
        if self._closed or self.init_error:
            return
        seed, ch = info.get("run_seed"), info.get("character")
        lf = int(info.get("loaded_floor") or 0)
        run = None
        if seed and ch and lf == 0:
            run = {"op": "reset", "mode": "run", "character": ch, "ascension": int(info.get("ascension") or 0), "seed": seed,
                   "route_summary": True, "labels": True}
            if str(info.get("game_mode") or "").lower() == "dread":
                run["game_mode"] = "dread"
        if run == self._run and self._tracked:
            return          # 同一局的局信息重发（重连等）：照常
        self._run, self._hist, self._pending, self._mprev = run, [], None, None
        self._disabled, self._failures, self._combat = False, 0, None
        self._tracked = False
        self._mframe = None
        if run is None:
            if lf:
                self.last_desync = f"游戏从存档第 {lf} 层接上，这一局不搜"
            return
        try:
            self._mframe = self._mirror_req(run)
            self._tracked = True
        except Exception as e:
            self.last_desync = f"镜像开局失败：{type(e).__name__}: {str(e)[:120]}"

    def prime(self, actions: list[int]) -> bytes | None:
        """离线评测用：这一局已经按 actions 走到当前局面（on_run_info 之后调），镜像直接重放过去。返回镜像的帧。"""
        if self._run is None:
            return None
        self._hist, self._pending, self._mprev = [int(a) for a in actions], None, None
        self._mframe = self._mirror_req({**self._run, "replay": list(self._hist)})
        self._tracked = True
        return self._mframe

    def on_save(self, info: dict, seq: int | None = None) -> None:
        """sts2-sim 用的整局存档；这里用不上。"""

    def reset_run(self) -> None:
        self._run, self._hist, self._tracked, self._pending, self._combat = None, [], False, None, None

    def engaged(self) -> bool:
        return self._combat is not None and self._tracked and not self._disabled and not self._closed

    def gated(self, probs, n: int) -> bool:
        try:
            p = np.asarray(probs[:n], dtype=float)
            if n <= 1:
                return False
            order = np.argsort(-p)
            return bool(p[order[0]] < self.P.tau and sum(1 for k in order[: self.P.k] if p[k] >= self.P.pmin) > 1)
        except Exception:
            return False

    def observe(self, action: int) -> None:
        """动作发回游戏后调：镜像也走这一步（先记成待确认，下一个实机帧对上了才算数）。"""
        if not self._tracked or self._mframe is None:
            return
        try:
            self._mprev = self._mframe
            self._mframe = self._mirror_req({"op": "step", "action": int(action)})
            self._pending = int(action)
        except Exception as e:
            self._lose(f"镜像走 {action} 出错：{type(e).__name__}: {str(e)[:120]}")

    def choose(self, obs, probs, raw: bytes | None = None, fallback_action: int | None = None) -> tuple[int, dict]:
        t0 = self._clock()
        n = int(obs.num_actions)
        p = np.asarray(list(probs)[:n], dtype=float)
        greedy = int(fallback_action) if fallback_action is not None else int(np.argmax(p))
        self._sync(raw)
        if obs.decision not in (DecisionKind.COMBAT, DecisionKind.CARD_SELECT) or not obs.in_combat:
            self._combat = None
            return greedy, self._info(greedy, greedy, t0, reason="not_combat")
        room, act = room_of(obs), int(obs.act)
        if not self.config.wants(room, act):
            self._combat = None
            return greedy, self._info(greedy, greedy, t0, reason="room_filtered")
        self._combat = (act, int(obs.floor), room)
        if not self._tracked:
            return greedy, self._info(greedy, greedy, t0, reason="untracked")
        if self._disabled:
            return greedy, self._info(greedy, greedy, t0, reason="disabled")
        if self.init_error:
            return greedy, self._info(greedy, greedy, t0, reason=f"init_error: {self.init_error}")
        if n <= 1:
            return greedy, self._info(greedy, greedy, t0, reason="single_action")
        if not self.gated(p, n):
            return greedy, self._info(greedy, greedy, t0, reason="confident")
        if self._warm.is_alive():
            self._warm.join(timeout=max(0.0, self.config.budget_ms / 1000.0))
            if self._warm.is_alive() or not self._workers:
                return greedy, self._info(greedy, greedy, t0, reason="workers_not_ready")
        self._dec += 1
        try:
            res = self._search(obs, p, greedy)
        except Exception as e:
            self._fail(f"{type(e).__name__}: {str(e)[:200]}")
            self._restart_workers()
            return greedy, self._info(greedy, greedy, t0, reason=f"search_error: {type(e).__name__}")
        self._failures = 0
        self.stats["searches"] += 1
        act = res["best"]
        if act != greedy:
            self.stats["changed"] += 1
        return act, self._info(act, greedy, t0, searched=True, extra=res)

    def close(self) -> None:
        self._closed = True
        for w in self._workers + ([self._mirror] if self._mirror else []):
            w.close()
        self._workers, self._mirror = [], None

    # ------------------------------------------------------------------ 同步

    def _sync(self, raw: bytes | None) -> None:
        """实机帧到了：和镜像的帧比原始字节。对上 → 确认上一个动作；等于走之前的帧 → 那个动作被模组丢了，退掉重建；都不是 → 这一局不再跟。"""
        if not self._tracked or raw is None:
            return
        self.stats["frames"] += 1
        if same_frame(raw, self._mframe):
            if self._pending is not None:
                self._hist.append(self._pending)
                self._pending = None
            self.stats["synced"] += 1
            return
        if self._pending is not None and same_frame(raw, self._mprev):
            # 模组没执行那个动作（过时被丢弃，同一局面又发了一次）：镜像按历史重建到走之前
            self.stats["resync"] += 1
            self._pending = None
            try:
                self._mframe = self._mirror_req({**self._run, "replay": list(self._hist)})
                if same_frame(self._mframe, raw):
                    return
            except Exception:
                pass
        self.stats["desync"] += 1
        self._lose(f"第 {len(self._hist)} 步实机帧与镜像对不上")

    def _lose(self, why: str) -> None:
        self._tracked = False
        self._pending = None
        self.last_desync = why
        self._log.warning("无头搜索：%s，这一局余下不搜", why)

    # ------------------------------------------------------------------ 搜索（照 sim_pimc.pimc_decide）

    def _search(self, real, p: np.ndarray, greedy: int) -> dict:
        P, X = self.P, self._pimc
        t_all = self._clock()
        order = np.argsort(-p)
        act = int(order[0])
        cands = [int(c) for c in order[: P.k] if p[c] >= P.pmin] or [act]
        net = self._get_net()
        ev = self._ModelEvaluator(net, device=self.config.device)
        deadline = t_all + self.config.budget_ms / 1000.0
        hard = threading.Timer(self.config.hang_ms / 1000.0, self._kill_workers)
        hard.daemon = True
        hard.start()
        sc: dict[int, dict[int, float]] = {c: {} for c in cands}
        alive, n_new, used, rollouts, errors, cut = list(cands), P.sims0, 0, 0, 0, False
        base_seed = (self.config.seed * 1000003 + 7919 * self._dec) & 0x7FFFFFFF
        try:
            for _ in range(P.rounds):
                if self._clock() > deadline:
                    cut = True
                    break
                seeds = [base_seed + used + j for j in range(n_new)]
                used += n_new
                worlds = [_World(cand=c, seed=s) for c in alive for s in seeds]
                stop = self._run_worlds(worlds, ev, real, deadline)
                cut = cut or stop
                rollouts += sum(1 for w in worlds if w.score is not None)
                for w in worlds:
                    if w.error:
                        errors += 1
                    elif w.score is not None:
                        sc[w.cand][w.seed] = w.score
                if len(alive) <= 2:
                    continue
                mean = lambda c: sum(sc[c].values()) / len(sc[c]) if sc[c] else -9.0
                rest = sorted((c for c in alive if c != act), key=lambda c: (-mean(c), -p[c]))
                alive = [act] + rest[: max(1, math.ceil(len(alive) / 2) - 1)]
                n_new *= 2
        finally:
            hard.cancel()
        best, best_m, diff = act, 0.0, {}
        for c in sc:
            if c == act:
                continue
            common = [s for s in sc[c] if s in sc[act]]
            if len(common) < 2:
                continue
            d = [sc[c][s] - sc[act][s] for s in common]
            m = sum(d) / len(d)
            sd = math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1))
            se = sd / math.sqrt(len(d))
            diff[c] = (round(m, 4), round(se, 4), len(d))
            if m > best_m and m > P.z * se:
                best, best_m = c, m
        if errors and not rollouts:
            raise RuntimeError(f"{errors} 个世界全部出错")
        return {"best": int(best), "changed": best != act, "rollouts": rollouts, "errors": errors, "cut": cut, "diff": diff,
                "search_ms": (self._clock() - t_all) * 1000.0, "net_s": ev.t_fwd, "cands": cands}

    def _run_worlds(self, worlds: list[_World], ev, root, deadline: float) -> bool:
        """各工作进程一次推一个世界：开局 + 重放 + reseed + 走候选一次往返，之后每步由网络（各进程的帧合成一批）定。返回是否因预算截断。"""
        P, X = self.P, self._pimc
        stage = int(root.header[23])
        left0 = X["enemy_left"](root)
        queue = list(worlds)
        busy: dict[int, _World] = {}
        cut = False
        while queue or busy:
            # 空闲的进程领新世界（预算到了就不领）
            for j, wk in enumerate(self._workers):
                if j in busy or not queue:
                    continue
                if self._clock() > deadline:
                    cut = True
                    queue.clear()
                    break
                w = queue.pop(0)
                w.left = left0
                # 推演用的帧不要候选文字（编码省一点）；局面、候选顺序与镜像相同
                wk.send({**self._run, "labels": False, "replay": list(self._hist), "reseed": int(w.seed) & 0xFFFFFFFF, "first": int(w.cand)})
                busy[j] = w
            if not busy:
                break
            # 读回各进程的帧
            for j in list(busy):
                w = busy[j]
                r = self._workers[j].recv()
                if isinstance(r, EngineError):
                    w.error = str(r)[:200]
                    del busy[j]
                    continue
                w.frame = parse(r)
                w.steps += 1
            # 分出胜负的记分、到叶子的估值，其余一起前向定下一步
            ask = []
            for j in list(busy):
                w = busy[j]
                o = w.frame
                if X["combat_over"](o):
                    w.score = X["terminal"](X["outcome"](o, w.left))
                elif w.steps > P.max_steps:
                    w.score = X["terminal"]({"win": False, "stage": stage})
                else:
                    w.left = X["enemy_left"](o)
                    ask.append(j)
                    continue
                del busy[j]
            if ask:
                leaf = [bool(P.leaf) and busy[j].steps >= P.leaf for j in ask]
                acts, vals = ev.rollout([busy[j].frame for j in ask], leaf, [(busy[j].seed, busy[j].steps) for j in ask])
                for j, l, a, v in zip(ask, leaf, acts, vals):
                    if l:
                        busy[j].score = float(v)
                        del busy[j]
                    else:
                        self._workers[j].send({"op": "step", "action": int(a)})
        return cut

    def _kill_workers(self) -> None:
        self._log.warning("无头搜索：一次搜索超过 %d 毫秒，杀掉工作进程", self.config.hang_ms)
        for w in self._workers:
            w.kill()

    def _restart_workers(self) -> None:
        for w in self._workers:
            w.close()
        self._workers = []
        self._warm = threading.Thread(target=self._start_workers, daemon=True)
        self._warm.start()

    def _fail(self, reason: str) -> None:
        self._failures += 1
        self._log.warning("无头搜索：失败（%s），连续 %d/%d", reason, self._failures, self.config.max_consecutive_failures)
        if self._failures >= self.config.max_consecutive_failures:
            self._disabled = True

    def _info(self, action: int, greedy: int, t0: float, *, reason: str | None = None, searched: bool = False, extra: dict | None = None) -> dict:
        elapsed = max(0.0, (self._clock() - t0) * 1000.0)
        extra = extra or {}
        rollouts = int(extra.get("rollouts", 0) or 0)
        changed = bool(searched and extra.get("changed") and int(action) != int(greedy))
        if searched:
            summary = (f"无头引擎 PIMC {rollouts} 个世界，{elapsed:.0f}ms，动作 {action}（贪心 {greedy}{'，改选' if changed else ''}）"
                       + (f"，{extra.get('errors')} 个出错" if extra.get("errors") else "") + ("，预算到点" if extra.get("cut") else ""))
        else:
            summary = f"退贪心（{reason}）" if reason else "未启用"
        quiet = bool(reason) and str(reason).split(":")[0] in QUIET_REASONS
        return {"backend": "headless", "action": int(action), "greedy_action": int(greedy), "searched": bool(searched), "fallback": not searched,
                "reason": reason, "quiet": quiet, "tier": "pimc" if searched else None, "sims": rollouts, "worlds": rollouts,
                "elapsed_ms": round(elapsed, 1), "search_ms": round(float(extra.get("search_ms", 0.0) or 0.0), 1),
                "budget_hit": bool(extra.get("cut")), "changed": changed, "synced": self._tracked, "diff": extra.get("diff"),
                "failures": int(self._failures), "disabled": bool(self._disabled), "summary": summary}
