"""PIMC 战斗改选，推演在 sts2-sim 上（engine/Sts2SimServe/方案.md 第 6 步）。

算法照决策记录 217 与 tools/search_test.py：网络最想选的项概率不到 tau 时取前 k 个候选（各至少 pmin），
在若干个重抽的世界里推演——各候选用同一组世界（共同随机数），逐轮减半 rounds 轮、首轮每候选 sims0 个世界、之后每轮加倍；
每个世界：走这个候选，之后贪心推演 leaf 步，用价值头估值（提前分出胜负的按 terminal() 记分）；
最后与贪心项逐世界配对取差，平均差超过 z 个标准误才改选。
和 search_test 的区别只在推演怎么到达根局面：原来每个世界都“从战斗开头按已走的动作重放”，这里镜像每走一步都把局面存进槽位 0，
每个世界 load 0 → reseed（换以后的随机数、重洗抽牌堆）→ 走候选；同一进程里的几个世界轮流 load / step / save 各自的槽位，
所以所有世界可以按步齐头推进、每一步合成一批做网络前向。推演前一律 reseed，搜索不碰真实随机数。
sts2-sim 只能在顶层战斗决策存档（选牌途中不行）：存不了时记下之后走的动作，需要时 load 再重走（SimMirror.tape、_World.tape）。

推演用的出手与估值是可替换的接口（Evaluator）：
- ModelEvaluator：sts2rl 的 PolicyValueNet，出手取策略最高项、叶子取价值头；
- HeuristicEvaluator：替身，不看网络——随机出牌（不用药水）、叶子按双方剩余生命估值。sts2-sim 的观测移植完之前，
  用它跑通流程、量耗时（sts2-sim 现在的帧只有头部、全局、生物、手牌和候选，网络在这种帧上的输出没有意义）。

镜像（SimMirror）是 1 个或几个 sts2-sim 进程，跟着真实对局走同样的动作；多个进程时各进程分担一部分世界。
真实帧与镜像帧的对应：compare() 比 Check.cs 的那几项（玩家生命格挡、能量、手牌、敌人生命格挡），
cand_map() 按候选的语义（动作类型、牌 ID 升级附魔费用、目标是第几个敌人）把真实候选下标对应到镜像的下标。

配对评测（真实引擎执行、sts2-sim 规划）见 tools/sim_pimc_eval.py。
"""
from __future__ import annotations

import math
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

from sts2env.env import EngineError  # noqa: E402
from sts2env.protocol import ActionKind, DecisionKind, Obs, TokenKind, Zone, parse  # noqa: E402
from sts2env.sim_client import SimClient, transplant_req  # noqa: E402

MODE_COMBAT, MODE_RUN = 0, 1      # 同 sts2rl.actor（这里不导入 actor，免得替身也要拉起 torch）


@dataclass
class PimcParams:
    tau: float = 0.9          # 最想选的项概率不到它才搜索
    k: int = 4                # 候选数上限
    pmin: float = 0.02        # 候选的最低概率
    sims0: int = 4            # 首轮每个候选的世界数
    rounds: int = 2           # 逐轮减半的轮数
    z: float = 2.0            # 配对平均差要超过几个标准误
    leaf: int = 12            # 走候选之后贪心推演几步再估值（0 为打到底）
    max_steps: int = 600      # 一次推演的步数上限，超过按输


# ---------- 帧上的读数

def combat_over(o: Obs) -> bool:
    """这一帧已不在战斗里（整局模式下战斗结束后是奖励或结局；sts2-sim 打完这一场为 done）。"""
    return o.done or o.stuck or o.num_actions == 0 or not o.in_combat


def enemy_left(o: Obs) -> float:
    """活着的敌人剩余生命占最大生命的比例（战斗结束那一帧没有敌人记录，推演时每步记下）。"""
    k = np.flatnonzero((o.tok_ints[:, 0] == TokenKind.CREATURE) & (o.tok_feats[:, 5] == 0) & (o.tok_feats[:, 3] > 0))
    hp, mx = float(o.tok_feats[k, 0].sum()), float(o.tok_feats[k, 1].sum())
    return hp / mx if mx > 0 else 0.0


def outcome(o: Obs, left: float | None = None) -> dict:
    """战斗结束帧的结果。整局模式下赢了是战斗外的下一个决策或通关；输了是结局帧（生命 0）；sts2-sim 打完这一场是结束帧。"""
    # 战斗只会以打赢或玩家死亡结束，所以看玩家还活着就行；不靠头部 5：移植版编码器的结束帧不置它（m1 里镜像结束帧全是 0）
    win = not o.stuck and o.hp > 0
    r = {"win": bool(win), "hp": int(o.hp), "max_hp": max(1, int(o.max_hp)), "stage": int(o.header[23])}
    if not win and left is not None:
        r["enemy_left"] = round(min(1.0, left), 4)
    return r


def terminal(r: dict) -> float:
    """提前分出胜负的推演得分（同 search_diag.terminal，与战斗价值头同一尺度）：赢 1.5；输 −1 + 0.25 ×（已打过的场数 + 这场打掉的敌人生命比例）。"""
    if r["win"]:
        return 1.5
    return -1.0 + 0.25 * (r.get("stage", 0) + 1.0 - r.get("enemy_left", 1.0))


# ---------- 真实帧与镜像帧的对应

def _player_tok(o: Obs) -> int:
    k = np.flatnonzero((o.tok_ints[:, 0] == TokenKind.CREATURE) & (o.tok_feats[:, 5] != 0))
    return int(k[0]) if len(k) else -1


def _enemies(o: Obs) -> list[int]:
    return [int(t) for t in np.flatnonzero((o.tok_ints[:, 0] == TokenKind.CREATURE) & (o.tok_ints[:, 2] == Zone.ENEMY) & (o.tok_feats[:, 3] != 0))]


def check_items(o: Obs) -> dict:
    """Check.cs 比的那几项：玩家生命与格挡、能量、手牌（牌 ID + 升级，排序）、活着的敌人（ID、生命、格挡，按顺序）。"""
    me = _player_tok(o)
    g = np.flatnonzero(o.tok_ints[:, 0] == TokenKind.GLOBAL)
    hand = np.flatnonzero((o.tok_ints[:, 0] == TokenKind.CARD) & (o.tok_ints[:, 2] == Zone.HAND))
    return {"hp": int(o.hp), "block": int(o.tok_feats[me, 2]) if me >= 0 else -1,
            "energy": int(o.tok_feats[g[0], 6]) if len(g) else -1,
            "hand": sorted((int(o.tok_ints[t, 1]), int(o.tok_feats[t, 3])) for t in hand),
            "enemies": [(int(o.tok_ints[t, 1]), int(o.tok_feats[t, 0]), int(o.tok_feats[t, 2])) for t in _enemies(o)]}


def compare(real: Obs, sim: Obs) -> str | None:
    """镜像与真实局面不一致时返回第一处不同的项名（含两边的值），一致返回 None。"""
    a, b = check_items(real), check_items(sim)
    for k in ("hp", "block", "energy", "hand", "enemies"):
        # 开战时的选牌（工具箱等）上回合开始的效果还没结算完，能量两边的时点不同（真实 5、sts2-sim 已是 6，进战斗决策后一致），
        # 只在战斗决策上比能量，同 Check.cs 的口径（2026-11-06，SIMY0018 第三幕两个 Boss）
        if k == "energy" and real.decision != DecisionKind.COMBAT:
            continue
        if a[k] != b[k]:
            return f"{k}: 真实 {a[k]} / 镜像 {b[k]}"
    if real.decision != sim.decision:
        return f"决策类型: 真实 {int(real.decision)} / 镜像 {int(sim.decision)}"
    return None


def sync_req(o: Obs, vocab: list[str], obs: bool = True) -> dict:
    """按真实帧拼服务的 sync 请求（把镜像看得到的部分校正成真实局面）：玩家生命、格挡、能量，
    四个牌堆的牌（词表名 + 升级，按帧里的顺序），活着的敌人（词表名、生命、格挡）。只在顶层战斗决策可用。
    观测格式 20（储君）：已知的抽牌堆顶部（区域 16–18，按顺序）放在抽牌堆最前面，draw_top 给张数，服务照这个顺序放、reseed 不打乱它们。"""
    name = lambda t: vocab[int(o.tok_ints[t, 1])]
    cards = lambda z: [[name(t), int(o.tok_feats[t, 3])] for t in
                       np.flatnonzero((o.tok_ints[:, 0] == TokenKind.CARD) & (o.tok_ints[:, 2] == z))]
    top = [[name(t), int(o.tok_feats[t, 3])] for t in
           np.flatnonzero((o.tok_ints[:, 0] == TokenKind.CARD) & (o.tok_ints[:, 2] >= Zone.DRAW_TOP1) & (o.tok_ints[:, 2] <= Zone.DRAW_TOP3))]
    me = _player_tok(o)
    g = np.flatnonzero(o.tok_ints[:, 0] == TokenKind.GLOBAL)
    return {"op": "sync", "obs": obs,
            "player": {"hp": int(o.hp), "block": int(o.tok_feats[me, 2]) if me >= 0 else 0, "energy": int(o.tok_feats[g[0], 6]) if len(g) else 0},
            "piles": {"hand": cards(Zone.HAND), "discard": cards(Zone.DISCARD), "exhaust": cards(Zone.EXHAUST), "draw": top + cards(Zone.DRAW), "draw_top": len(top)},
            "enemies": [[name(t), int(o.tok_feats[t, 0]), int(o.tok_feats[t, 2])] for t in _enemies(o)]}


def _tok_desc(o: Obs, t: int, enemies: list[int], loose: bool = False) -> tuple:
    """候选指向的记录的语义：牌按 ID、升级、附魔、当前费用、区域（同 Check.cs 的 Pick）；生物按玩家或第几个活着的敌人；药水按栏位。
    loose 时牌只按 ID、升级、区域（镜像的附加 ID 或费用修正和真实引擎不同时的退路）。"""
    if t < 0 or t >= len(o.tok_ints):
        return ()
    kind, vid, zone, _, aux = (int(x) for x in o.tok_ints[t])
    f = o.tok_feats[t]
    if kind == TokenKind.CARD:
        if loose:
            return (kind, vid, int(f[3]), zone)
        # 格式 20（储君）：手牌再按星费（特征 2，带修正）分开——发现、虚空形态给的免费副本与原价的同名牌能量费相同、星费不同；格式 19 的键不变
        star = (int(f[2]),) if int(o.header[0]) >= 20 and zone == Zone.HAND else ()
        return (kind, vid, int(f[3]), aux, int(f[0]), zone) + star
    if kind == TokenKind.CREATURE:
        return (kind, "P") if f[5] != 0 else (kind, "E", enemies.index(t) if t in enemies else -1, vid)
    if kind == TokenKind.POTION:
        return (kind, vid, int(f[0]))
    return (kind, vid, zone)


def cand_keys(o: Obs, loose: bool = False) -> list[tuple]:
    """每个候选的语义键；一模一样的候选（两张同样的牌）加序号区分，按出现先后对应。"""
    en = _enemies(o)
    keys, seen = [], {}
    for c in o.cand_ints:
        b = int(c[2])
        # 升级类选牌的对象 B 是“升级后的复制品”（区域 12），sts2-sim 现在不给（协议.md 第六节）；它不区分候选，不进键
        if 0 <= b < len(o.tok_ints) and int(o.tok_ints[b, 2]) == Zone.UPGRADE_PREVIEW:
            b = -1
        k = (int(c[0]), int(c[3]), _tok_desc(o, int(c[1]), en, loose), _tok_desc(o, b, en, loose))
        n = seen.get(k, 0); seen[k] = n + 1
        keys.append(k + (n,))
    return keys


def cand_map(real: Obs, sim: Obs) -> tuple[list[int] | None, str]:
    """真实候选下标 → 镜像候选下标，以及对应方式。真实候选的语义键在镜像里都找得到就按语义对应；
    否则退到宽松语义（牌只看 ID + 升级）；再不行，候选数与动作类型序列都相同就按下标对应（记为“按下标”）；都不行返回 None。"""
    ka, kb = cand_keys(real), cand_keys(sim)
    pos = {k: j for j, k in enumerate(kb)}
    if all(k in pos for k in ka):
        # 镜像可以多出真实引擎没有的候选（例如 sts2-sim 在某些选牌里多给一个“取消”）：真实候选都找得到就照样对应，另记一笔
        return [pos[k] for k in ka], "语义" if len(ka) == len(kb) else "镜像多出候选"
    la, lb = cand_keys(real, True), cand_keys(sim, True)
    lpos = {k: j for j, k in enumerate(lb)}
    if all(k in lpos for k in la):
        # 第二级：牌只按 ID + 升级对应（镜像的牌少了附加 ID，或费用修正没跟上），同样的牌按出现先后对应
        return [lpos[k] for k in la], "宽松语义"
    if len(ka) == len(kb) and np.array_equal(real.cand_ints[:, 0], sim.cand_ints[:, 0]):
        return list(range(len(ka))), "按下标"
    return None, f"候选不同: 真实 {len(ka)} 个 / 镜像 {len(kb)} 个"


# ---------- 推演用的出手与估值（可替换）

class Evaluator:
    """rollout(frames, leaf, keys) → (每帧的出手, 每帧的估值)。leaf[i] 为真的帧要估值、不用出手；否则要出手、估值不用。
    keys[i] 为 (世界种子, 推演步数)：替身靠它让同一个世界在各候选下抽到同样的随机数。"""

    def rollout(self, frames: list[Obs], leaf: list[bool], keys: list[tuple]) -> tuple[list[int], list[float]]:
        raise NotImplementedError


class ModelEvaluator(Evaluator):
    """策略价值网络：出手取 policy_mode 下的策略最高项，叶子取 value_mode 下的价值头（默认战斗课程，与 terminal() 同一尺度）。
    一批里各行的模式可以不同，所以出手与估值合成一次前向。strip_priv 清掉特权特征（搜索只用公开信息，决策记录 214）。
    infer 不给时照原样调 net(batch)（直播的 sim_search_backend 传进来的是只会 net(batch) 的代理）；
    给一个字典时走 sts2rl.infer.InferEngine（inference_mode + forward_infer，可选 tf32 / bf16 / fp16、CPU int8 与线程数，键见 InferEngine），见 make_evaluator。"""

    def __init__(self, net, device="cpu", policy_mode: int = MODE_RUN, value_mode: int = MODE_COMBAT, strip_priv: bool = True,
                 infer: dict | None = None):
        import torch
        from sts2rl.features import Ragged, collate_ragged
        self.torch, self.Ragged, self.collate = torch, Ragged, collate_ragged
        self.net, self.device = net, device
        self.policy_mode, self.value_mode, self.strip_priv = policy_mode, value_mode, strip_priv
        self.engine = None
        if infer is not None:
            from sts2rl.infer import InferEngine
            self.engine = InferEngine(net, device, **infer)
        self.n_fwd = self.n_rows = 0
        self.t_fwd = 0.0

    def _ragged(self, frames: list[Obs], modes: list[int]):
        r = self.Ragged.from_obs(frames, np.array(modes))
        if self.strip_priv:
            r.tok_priv[:] = 0       # from_obs 拼出来的是副本；与逐帧 _no_priv 后再拼逐位相同
        return r

    def _run(self, r):
        if self.engine is not None:
            lg, v = self.engine.run(r)
            return self.torch.from_numpy(lg), self.torch.from_numpy(v)
        with self.torch.no_grad():
            logits, values = self.net(self.collate(r).to(self.device))
        return logits.float().cpu(), values.float().cpu()

    def forward(self, frames: list[Obs], modes: list[int]):
        t0 = time.perf_counter()
        logits, values = self._run(self._ragged(frames, modes))
        self.n_fwd += 1; self.n_rows += len(frames); self.t_fwd += time.perf_counter() - t0
        return logits, values

    def policy(self, o: Obs, mode: int | None = None) -> np.ndarray:
        """根决策用：真实帧上的策略概率。"""
        logits, _ = self.forward([o], [self.policy_mode if mode is None else mode])
        return self.torch.softmax(logits[0, : o.num_actions], -1).numpy()

    def rollout(self, frames, leaf, keys):
        if not frames:
            return [], []
        logits, values = self.forward(frames, [self.value_mode if l else self.policy_mode for l in leaf])
        acts = [0 if l else int(logits[i, : o.num_actions].argmax()) for i, (o, l) in enumerate(zip(frames, leaf))]
        return acts, [float(values[i]) for i in range(len(frames))]


class RemoteEvaluator(ModelEvaluator):
    """同 ModelEvaluator，前向交给共用推理进程（sts2rl.infer.serve，经 Unix 套接字 path）：一台机器只载一份模型，
    各搜索分片的帧在推理进程里合成大批。strip_priv 时不发特权特征（推理进程补 0，等价于清零）。"""

    def __init__(self, path: str, policy_mode: int = MODE_RUN, value_mode: int = MODE_COMBAT, strip_priv: bool = True):
        import torch
        from sts2rl.features import Ragged
        from sts2rl.infer import InferClient
        self.torch, self.Ragged = torch, Ragged
        self.net, self.device, self.engine = None, "remote", None
        self.policy_mode, self.value_mode, self.strip_priv = policy_mode, value_mode, strip_priv
        self.client = InferClient(path, with_priv=not strip_priv)
        self.n_fwd = self.n_rows = 0
        self.t_fwd = 0.0

    def _ragged(self, frames, modes):
        return self.Ragged.from_obs(frames, np.array(modes))

    def _run(self, r):
        lg, v = self.client.run(r)
        return self.torch.from_numpy(lg.copy()), self.torch.from_numpy(v.copy())


def make_evaluator(spec: str | None, net, device="cpu", policy_mode: int = MODE_RUN, value_mode: int = MODE_COMBAT,
                   strip_priv: bool = True) -> ModelEvaluator:
    """按字符串选推理路径（tools/pimc_infer.py 的 --ev）：
      空 / "torch"：原来的 net(batch) + no_grad；
      "fp32" / "tf32" / "bf16" / "fp16"，后面可接 ",int8"、",threads=N"、",nopack"、",fast"、",compile"：InferEngine（例 "fp32,threads=4"）；
      "unix:路径"：共用推理进程（RemoteEvaluator，net 不用；推理进程用 tools/pimc_infer.py serve 起）。
    各项的偏差与速度见 reports/PIMC推理侧提速.md：搜索上用 fp32（原路径或 "fp32"），bf16、int8 偏差大。"""
    kw = dict(policy_mode=policy_mode, value_mode=value_mode, strip_priv=strip_priv)
    if not spec or spec == "torch":
        return ModelEvaluator(net, device, **kw)
    if spec.startswith("unix:"):
        return RemoteEvaluator(spec[5:], **kw)
    from sts2rl.infer import parse_spec
    return ModelEvaluator(net, device, infer=parse_spec(spec), **kw)


def _no_priv(o: Obs) -> Obs:
    if not o.tok_priv.any():
        return o
    import copy
    o = copy.copy(o)
    o.tok_priv = np.zeros_like(o.tok_priv)
    return o


class HeuristicEvaluator(Evaluator):
    """替身（观测移植完之前用）：出手在能打的牌里随机选（不用药水、不弃药水），选牌界面随机选或确认，没牌可打才结束回合；
    叶子估值 = −1 + 2.5 ×（1 − 敌人剩余生命比例）× 玩家剩余生命比例，大致落在 terminal() 的范围里。随机数由 (世界种子, 步数) 决定。"""

    def __init__(self, salt: int = 0):
        self.salt = salt
        self.n_fwd = self.n_rows = 0
        self.t_fwd = 0.0

    def rollout(self, frames, leaf, keys):
        acts, vals = [], []
        for o, l, key in zip(frames, leaf, keys):
            if l:
                acts.append(0)
                vals.append(-1.0 + 2.5 * (1.0 - enemy_left(o)) * (o.hp / max(1, o.max_hp)))
                continue
            rng = random.Random(hash((self.salt,) + tuple(key)))
            kinds = o.cand_ints[: o.num_actions, 0]
            good = [i for i, k in enumerate(kinds) if k in (ActionKind.PLAY_CARD, ActionKind.SELECT_CARD, ActionKind.CONFIRM_SELECTION)]
            end = [i for i, k in enumerate(kinds) if k == ActionKind.END_TURN]
            if good and (not end or rng.random() > 0.15):
                acts.append(rng.choice(good))
            elif end:
                acts.append(end[0])
            else:
                acts.append(0)
            vals.append(0.0)
        return acts, vals


# ---------- 镜像与推演

ANCHOR = 0      # 镜像槽位 0：当前局面或它最近一个能存档的祖先（sts2-sim 只能在顶层战斗决策存档，选牌途中不行）


def _is_select(o: Obs) -> bool:
    return int(o.header[1]) == 2       # DecisionKind.CARD_SELECT


class SimMirror:
    """真实对局的镜像：n 个 sts2-sim 进程走同样的动作，各自持有同一个当前局面；推演时各进程分担一部分世界。
    每走一步都把局面存进槽位 ANCHOR；在选牌界面里存不了，就把锚点之后走的动作记在 tape 里——
    load ANCHOR 再按 tape 重走，就回到当前局面（推演从这里出发，搜索完也靠它还原）。"""

    def __init__(self, n: int = 1, dll=None, log=None, vocab=None, tables=None):
        import subprocess
        self.clients = [SimClient(dll, log or subprocess.DEVNULL, vocab, tables) for _ in range(n)]
        self.frame: Obs | None = None
        self.tape: list[int] = []
        self.anchor_ok = False
        self.diverged = 0          # 各进程之间帧不同的次数（应为 0）

    def _all(self, req: dict) -> Obs:
        for c in self.clients:
            c.send(req)
            c.send({"op": "save", "slot": ANCHOR})
        replies = [c.drain() for c in self.clients]
        raws = [r[0] for r in replies]
        for r in raws:
            if isinstance(r, EngineError):
                raise r
        if any(r != raws[0] for r in raws[1:]):
            self.diverged += 1
        self.frame = parse(raws[0])
        saved = [len(r) > 1 and not isinstance(r[1], EngineError) for r in replies]
        if all(saved):
            self.tape, self.anchor_ok = [], True
        elif req["op"] == "step" and self.anchor_ok and not any(saved):
            self.tape.append(int(req["action"]))
        else:
            self.anchor_ok = False     # 开局就停在选牌，或各进程存档结果不同：这一段不能搜索，等下一次存档成功
        return self.frame

    def reset(self, save: dict, next_room_id: int, target, room: str, root_frame: bytes | None = None) -> Obs:
        self.tape, self.anchor_ok = [], False
        return self._all(transplant_req(save, next_room_id, target, room, root_frame))

    def step(self, action: int) -> Obs:
        return self._all({"op": "step", "action": int(action)})

    def root_ops(self) -> list[dict]:
        """回到当前局面的请求序列。"""
        return [{"op": "load", "slot": ANCHOR}] + [{"op": "step", "action": a} for a in self.tape]

    def close(self) -> None:
        for c in self.clients:
            c.close()


@dataclass
class _World:
    cand: int               # 真实候选下标
    seed: int
    proc: int
    slot: int
    pending: int | None = None     # 下一步要走的镜像候选下标
    restore: list = field(default_factory=list)    # 重建这个世界的请求：load 自己的槽位，或回根局面 + reseed
    tape: list = field(default_factory=list)       # restore 之后已走的动作（选牌途中存不了档时累积）
    steps: int = -1         # 走候选之后的推演步数（走候选那一步之后为 0）
    left: float = 1.0
    frame: Obs | None = None
    score: float | None = None
    error: str | None = None


@dataclass
class SearchInfo:
    searched: bool = False
    best: int = -1
    greedy: int = -1
    changed: bool = False
    rollouts: int = 0
    errors: int = 0
    restore_bad: int = 0
    sim_ops: int = 0
    sim_s: float = 0.0
    net_s: float = 0.0
    total_s: float = 0.0
    diff: dict = field(default_factory=dict)      # 候选 → (平均差, 标准误, 配对数)
    err_text: str | None = None


def run_worlds(mirror: SimMirror, ev: Evaluator, worlds: list[_World], cmap: list[int], P: PimcParams, root: Obs, info: SearchInfo) -> None:
    """在镜像上推演一批世界，各世界的得分写进 world.score（报错的写 error、不参与比较）；返回前各进程回到根局面。
    各世界按步齐头推进：每一轮所有活着的世界各走一步，帧合成一批交给 ev。同一进程上的几个世界轮流用：
    切到某个世界前先按它的 restore + tape 重建，走完存进它的槽位（选牌途中存不了就记进 tape，下次重走）。
    一个进程的请求先全部写出、再依次读回，各进程同时算。"""
    stage = int(root.header[23])
    left0 = enemy_left(root)
    base = mirror.root_ops()
    by: dict[int, list[_World]] = {}
    for w in worlds:
        w.left, w.pending = left0, int(cmap[w.cand])
        w.restore = base + [{"op": "reseed", "seed": w.seed & 0xFFFFFFFF}]
        by.setdefault(w.proc, []).append(w)
    live: dict[int, _World | None] = {j: None for j in by}
    single = {j: len(ws) == 1 for j, ws in by.items()}

    def flush(plan: dict[int, list]) -> None:
        """plan[进程] = 依次发出的 (世界, 种类, 请求)：种类 r 为重建（回复不用）、s 为这一轮的一步（回复是新帧）、v 为存档。"""
        t0 = time.perf_counter()
        for j, ops in plan.items():
            for _, _, req in ops:
                mirror.clients[j].send(req)
        for j, ops in plan.items():
            replies = mirror.clients[j].drain()
            info.sim_ops += len(ops)
            for (w, kind, req), r in zip(ops, replies):
                if w.error is not None:
                    continue
                if kind == "v":
                    if not isinstance(r, EngineError):
                        w.restore, w.tape = [{"op": "load", "slot": w.slot}], []
                    continue        # 存不了（选牌途中）：restore + tape 照样能重建，下次切回来时重走
                if isinstance(r, EngineError):
                    w.error = f"{req['op']}：{str(r)[:200]}"
                    continue
                if kind == "s":
                    w.frame = parse(r)
                    w.tape.append(w.pending)
                    w.pending = None
                    w.steps += 1
        info.sim_s += time.perf_counter() - t0

    info.rollouts += len(worlds)
    while True:
        # 给拿到新帧、还没定下一步的世界定下一步：分出胜负的记分，到叶子的估值，其余出手
        ask = []
        for w in worlds:
            if w.error is not None or w.score is not None or w.pending is not None:
                continue
            o = w.frame
            if combat_over(o):
                w.score = terminal(outcome(o, w.left))
            elif w.steps > P.max_steps:
                w.score = terminal({"win": False, "stage": stage})
            else:
                w.left = enemy_left(o)
                ask.append(w)
        if ask:
            leaf = [bool(P.leaf) and w.steps >= P.leaf for w in ask]
            t0 = time.perf_counter()
            acts, vals = ev.rollout([w.frame for w in ask], leaf, [(w.seed, w.steps) for w in ask])
            info.net_s += time.perf_counter() - t0
            for w, l, a, v in zip(ask, leaf, acts, vals):
                if l:
                    w.score = v
                else:
                    w.pending = int(a)
        plan: dict[int, list] = {}
        for j, ws in by.items():
            todo = [w for w in ws if w.error is None and w.score is None and w.pending is not None]
            todo.sort(key=lambda w: w is not live[j])          # 当前就在进程里的世界先走，省一次重建
            for w in todo:
                ops = plan.setdefault(j, [])
                if live[j] is not w:
                    # 重建用的请求回复都不看，让服务不编码观测（编一帧约是走一步的 4 倍）
                    ops += [(w, "r", dict(q, obs=False)) for q in w.restore] + [(w, "r", {"op": "step", "action": a, "obs": False}) for a in w.tape]
                ops.append((w, "s", {"op": "step", "action": w.pending}))
                if not single[j]:
                    ops.append((w, "v", {"op": "save", "slot": w.slot}))
                live[j] = w
        if not plan:
            break
        flush(plan)

    # 还原：各进程回到根局面，并核对回来的帧与根局面一致
    for c in mirror.clients:
        for i, q in enumerate(base):
            c.send(q if i == len(base) - 1 else dict(q, obs=False))     # 只有最后一条要帧（核对还原）
    for c in mirror.clients:
        r = c.drain()[-1]
        info.sim_ops += len(base)
        if isinstance(r, EngineError) or compare(root, parse(r)) is not None:
            info.restore_bad += 1


def pimc_decide(mirror: SimMirror, ev: Evaluator, real: Obs, p: np.ndarray, cmap: list[int], P: PimcParams,
                seed_base: int, dec_idx: int) -> SearchInfo:
    """一个决策的改选（真实帧 real 上的网络概率 p；cmap 为真实候选 → 镜像候选）。返回 SearchInfo，best 为最终选的真实候选下标。
    门限不到或只有一个候选时 searched 为假、best 为贪心项。"""
    t_all = time.perf_counter()
    info = SearchInfo()
    order = np.argsort(-p)
    act = int(order[0])
    info.greedy = info.best = act
    if real.num_actions <= 1 or p[act] >= P.tau:
        return info
    cands = [int(c) for c in order[: P.k] if p[c] >= P.pmin] or [act]
    if len(cands) <= 1:
        return info
    if not mirror.anchor_ok:
        info.err_text = "镜像没有可用的锚点（开局停在选牌或存档失败）"
        return info
    info.searched = True
    root = mirror.frame
    sc: dict[int, dict[int, float]] = {c: {} for c in cands}       # 候选 → {世界种子: 得分}；各候选同一组种子
    alive, n_new, used = list(cands), P.sims0, 0
    n_proc = len(mirror.clients)
    for _ in range(P.rounds):
        seeds = [seed_base + 7919 * (dec_idx + 1) + used + j for j in range(n_new)]
        used += n_new
        worlds = []
        for i, (c, s) in enumerate((c, s) for c in alive for s in seeds):
            worlds.append(_World(cand=c, seed=s, proc=i % n_proc, slot=1 + i // n_proc))
        run_worlds(mirror, ev, worlds, cmap, P, root, info)
        for w in worlds:
            if w.error is not None:
                info.errors += 1
                info.err_text = info.err_text or w.error
                continue
            sc[w.cand][w.seed] = w.score
        if len(alive) <= 2:
            continue
        # 逐轮减半，贪心那一项始终留下（最后要和它逐世界配对）
        mean = lambda c: sum(sc[c].values()) / len(sc[c]) if sc[c] else -9.0
        rest = sorted((c for c in alive if c != act), key=lambda c: (-mean(c), -p[c]))
        alive = [act] + rest[: max(1, math.ceil(len(alive) / 2) - 1)]
        n_new *= 2
    best, best_m = act, 0.0
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
        info.diff[c] = (round(m, 4), round(se, 4), len(d))
        if m > best_m and m > P.z * se:
            best, best_m = c, m
    info.best, info.changed = best, best != act
    info.total_s = time.perf_counter() - t_all
    return info


def cand_diff(real: Obs, sim: Obs, vocab: list[str] | None = None) -> dict:
    """两边候选语义键的差（诊断用）：只在真实里有的、只在镜像里有的，记录的词表 ID 换成名字。"""
    def name(k):
        def d(t):
            return [vocab[x] if vocab and isinstance(x, int) and j == 1 and 0 <= x < len(vocab) else x for j, x in enumerate(t)]
        return [k[0], d(k[2]), d(k[3])]
    ka, kb = cand_keys(real), cand_keys(sim)
    return {"real_only": [name(k) for k in ka if k not in kb][:6], "sim_only": [name(k) for k in kb if k not in ka][:6]}
