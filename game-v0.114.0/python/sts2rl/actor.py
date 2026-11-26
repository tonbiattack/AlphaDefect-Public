"""采集进程：管理一组引擎进程，用本地策略副本批量推理，把完整的对局轨迹送给学习进程。

两种对局：
- 战斗课程（mode 0）：按课程进度直接开一场战斗；终局奖励 = 胜负 + 血量变化 − 药水消耗，Boss 名额打赢再 +boss_win_bonus。
- 整局（mode 1）：A10 Defect 从 Neow 打到通关或死亡；每多到一层 +floor_reward，打过第一、二幕 Boss 各 +act1/act2_boss_reward，
  通关再 +victory_reward；血量势能（hp_potential，血量越低每点血越值钱）的变化记在导致变化的那一步（死时血量为 0，
  整局合计约为固定的 −run_hp_shaping，只是把“掉血不好、回血好”提前到发生的那一步）；
  非 Boss 战结束时按这场的净掉血再扣 combat_dmg_coef × 掉血 / 开打时最大生命，非 Boss 战里每用一瓶药水扣 run_potion_coef，
  遗物势能（每件 relic_coef）、牌组势能里的稀有卡（每张 rare_coef）的变化记在拿到（失去）遗物的那一步；
  deck_shaping > 0 时，同一幕内相邻两个非战斗决策之间牌组打分的变化 × deck_shaping 记在前一个决策上（现已关闭）。
每个采集进程中前 round(envs × run_fraction) 个引擎固定打整局，其余打战斗课程。
整局的引擎附带玩家快照（选路线时）和战斗信息（战斗开始时），攒进 SnapshotPools，战斗课程用来在真实牌组上练 Boss 和第二幕。
用真实快照练 Boss 的结果连同快照那一刻的局面视图一起送给学习进程，训练牌组打分头。
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import queue
import time
import traceback
from collections import Counter, deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from sts2env import EngineError, VecEnv
from sts2env.protocol import ActionKind, DecisionKind, TokenKind, Zone

from .catalog import Catalog
from .config import deck_size_ref
from .curriculum import CombatCurriculum, CurriculumConfig, SnapshotPools
from .features import Ragged, collate_ragged, deck_view
from .model import CO_POT_SLOTS, CombatOutcomeHeads, PolicyValueNet, held_potion_ids, load_state, use_tf32
from .replay import describe
from .room_log import RoomLog
from .solver_pool import SolverPool
from . import winprob_online
from .value_probe import ActorProbe, reward_event

MODE_COMBAT, MODE_RUN = 0, 1


class Trajectory:
    __slots__ = ("mode", "obs", "actions", "logp", "logp_mu", "values", "rewards", "version", "vers", "hp_start", "max_hp", "potions", "spec", "floor", "act", "t0",
                 "snap", "snap_n", "view", "label_view", "phi", "phi_prev", "shaped", "hp_last", "hp_shaped", "choices", "pot_held", "pot_pending", "pot_wasted", "deck_phi", "deck_shaped",
                 "room", "cb_hp", "cb_max", "dmg_shaped", "pot_cost", "skip_cost", "relic_phi", "relic_shaped", "cb_key", "cb_floor", "bosses",
                 "elite_fights", "elite_pots", "boss_pots", "ancient", "cards", "req", "cb_log", "forced", "anc", "seg0", "stage2",
                 "wp_last", "wp_shaped", "wp_rows", "wp_deck", "wp_up", "wp_key",
                 "pg_last", "pg_shaped", "bv_last", "bv_shaped", "pq_v", "pq_entry", "tape", "cx_key", "probe", "gid", "sig", "ms_full",
                 "co_real", "cb_first", "co_pred", "seg_fill", "det", "srch_ctx", "srch_b3", "fight0", "nv")

    def __init__(self, mode: int, version: int, spec: dict):
        self.mode, self.version, self.spec = mode, version, spec
        self.req = None     # 这一局的 reset 请求（卡住时记进 stuck.jsonl，照着重放可复现）
        self.cb_log = None  # 整局：当前战斗抽中要记房间结果时，开打时的那一半记录（room_log.py）
        self.obs, self.actions, self.logp, self.values, self.rewards = [], [], [], [], []
        self.logp_mu = []   # 每一步实际出手的概率（含强制探索，explore_logmu）的对数；logp 为策略自己的概率，只用来算 KL
        self.forced = []    # 每一步是否被强制探索改过（explore_probs）：这些步是离策略的，学习端特殊处理
        self.nv = []        # 运气校正（config.luck_cv）：这一步之后“平均运气下”的下一步价值 V̄，没算的为 nan；学习端算 δ 时代替真实下一步的价值
        self.det = []       # 每一步是不是照部署方式确定性出手的（search_collect_rooms 的战斗：不犹豫取最大、犹豫时搜索；学习端不进策略梯度）
        self.srch_ctx = None    # search_collect：当前战斗的上下文（第三幕第一个 Boss、开打层数），整局搜索的路上奖励要用
        self.srch_b3 = 0        # search_collect：这一局已开打的第三幕 Boss 场数
        self.anc = []       # 每一步是不是古神三选一的界面（只用于统计，决策记录 202）
        self.vers = []      # 每一步实际用的权重版本：长局跨好几个版本，学习端按步判断是否过旧
        self.tape = None    # 抽中另存回放的整局：每步的 replay.describe（config.run_replay_frac，决策记录 236）
        self.probe = None   # 只读价值诊断的整局标识与分段边界，不进动作或目标计算
        self.cx_key = None  # 战斗定向探索（决策记录 238）上一次抽签的（层数, 第几场, 回合）：每个回合只抽一次
        self.wp_last = None     # 胜率势能（决策记录 189）：上一步的势能 Φ；wp_shaped 为这一局势能塑形的合计
        self.wp_shaped = 0.0
        self.wp_rows = {}       # 胜率模型的训练样本：{层: [(数值特征, 袋子), ...]}，每层最多 wp_per_floor 步（蓄水池抽样）
        self.wp_deck, self.wp_up = [], 0    # 最近一次战斗外的牌组与升级张数（战斗中沿用）
        self.wp_key = None      # 上一次算势能的时刻（winprob_online.checkpoint）
        self.pg_last = None     # 进度势能（决策记录 191）：上一次的势能与这一局的塑形合计
        self.pg_shaped = 0.0
        self.gid = os.urandom(8).hex()     # 局编号：整局分段送出时学习端靠它把同一局的各段接起来（局外优势跨段计算，决策记录 255）
        self.ms_full = 0.0  # 里程碑奖励逐步撤掉（决策记录 271）：这一局按原系数本该拿到的里程碑合计（层数、各幕 Boss），通关时补回撤掉的部分
        self.sig = []       # 同种子成组采集（决策记录 267）：整局里局外决策那一帧的签名（frame_sig），其余步为 0；学习端据此找同组里走到同一局面的局
        self.bv_last = None     # 构筑价值势能（决策记录 252）：上一个战斗外决策的势能与这一局的塑形合计
        self.bv_shaped = 0.0
        self.pq_v = None        # 按学习潜力选题（决策记录 226）：练习用的快照入池时的价值估计
        self.pq_entry = None    # 整局里第三幕第一个 Boss 开打这一步：算完价值后记到池里的这一条上
        self.stage2 = None  # 连打练习（spec 带 next_encounter）：第二场开打后的第一步下标，没打到为 None
        self.seg0 = 0       # 整局分段提交（run_segment_steps）：还没送出的部分从这一步起；各列表仍存整局，下标不变
        self.hp_start = self.max_hp = None
        self.potions = 0
        self.floor = self.act = 0
        self.t0 = time.time()
        self.snap = None    # 整局：最近一次选路线时的玩家快照
        self.snap_n = -10   # 拿到这份快照时已有的步数（只收选路之后紧接着开打的战斗入练习池，决策记录 206）
        self.view = None    # 及那一刻的局面视图
        self.label_view = None   # 战斗课程：用真实快照练 Boss 时，快照那一刻的局面视图（结果是它的打分标签）
        self.phi = []       # 整局：每一步的牌组打分（战斗中的步为 nan）
        self.phi_prev = None     # 上一个非战斗决策的 (步下标, 幕, 打分)
        self.shaped = 0.0   # 牌组打分带来的奖励合计
        self.hp_last = None       # 整局：上一步的血量势能（hp_potential）
        self.hp_shaped = 0.0     # 血量塑形的奖励合计
        self.choices = Counter()  # 整局：局外决策选了什么（choice_key）
        self.pot_held = None      # 整局：上一步手上的药水瓶数
        self.pot_pending = []     # 丢了药水、还没领到新药水的那几步（步下标）
        self.pot_wasted = 0       # 白丢的药水瓶数（受了惩罚的）
        self.deck_phi = None      # 整局：上一个战斗外决策时的牌组势能（deck_potential）
        self.deck_shaped = 0.0    # 牌组规模与升级塑形的奖励合计
        self.room = None          # 整局：当前战斗的房间类型（Monster、Elite、Boss），战斗外为 None
        self.fight0 = 0           # 整局：当前战斗开打时已有的步数（单场战斗步数上限 max_steps_fight 用）
        self.cb_hp = self.cb_max = 0   # 整局：当前战斗开打时的血量、最大生命
        self.dmg_shaped = 0.0     # 非 Boss 战掉血扣分合计
        self.pot_cost = 0.0       # 非 Boss 战用药水扣分合计
        self.skip_cost = 0.0      # 放弃无负面遗物、金币的扣分合计（决策记录 160）
        self.elite_fights = 0     # 整局：打了几场精英战
        self.elite_pots = 0       # 整局：精英战里用了几瓶药水
        self.boss_pots = []       # 整局：每场 Boss 战开打时 (幕, 手上的药水瓶数)
        self.ancient = {}           # 古神三选一遗物：{(层数, 提供的遗物): (幕, 选中的遗物或 None)}
        self.cards = {}             # 选牌奖励：{(层数, 提供的牌): (幕, 选中的牌或 None)}，同评测的 card_offer
        self.relic_phi = None     # 整局：上一步的遗物势能
        self.relic_shaped = 0.0   # 遗物塑形的奖励合计
        self.cb_key = None        # 整局：最近一场战斗（"幕:房间类型:遭遇"，幕从 0 起）及其所在层，用来记死因
        self.cb_floor = -1
        self.bosses = []          # 整局：打过的每场 Boss ["幕:遭遇", 是否打赢]；卡住时正在打的那场不计
        # 跳过稳赢战斗（config.skip_easy，reports/跳过稳赢战斗方案.md）
        self.co_real = False      # 这一局完全真打（开局时以 skip_real_runs 的概率指定），不跳过任何战斗
        self.cb_first = False     # 整局：下一次推理的这一帧是一场战斗的第一帧
        self.co_pred = None       # 整局：这场战斗够条件但真打了（或在完全真打的局里），开战时的预测，打完算校准残差
        self.seg_fill = 0         # 整局分段：这一段里跳过的战斗折合的步数（按这类战斗的平均步数补上，分段长度与原来一样）


_STOP = None


def run_actor(rank: int, cfg, weights_path: str, out_queue, stop_event) -> None:
    global _STOP
    _STOP = stop_event
    # 采集进程（连同之后创建的线程和引擎子进程，nice 按线程继承）让学习进程优先用 CPU：
    # 4090 机上 CPU 被 256 个引擎占满时，学习一次从 3.05 秒降到 2.89 秒，训练吞吐 +6.5%（2026-10-26 实测）
    if cfg.actor_nice:
        try:
            os.nice(cfg.actor_nice)
        except OSError:
            pass
    try:
        _actor_loop(rank, cfg, weights_path, out_queue, stop_event)
    except Exception:
        out_queue.put(("error", rank, traceback.format_exc()))
    finally:
        # 退出时不等待队列把剩余数据写完：学习进程可能已经不在，等待会让本进程永远挂住
        out_queue.cancel_join_thread()


def card_offer(o, a: int, vocab: list[str]) -> tuple[tuple[str, ...], str | None]:
    """选牌奖励界面（评测与训练整局都记，2026-10-29 起训练也记）：提供的牌（词表名去掉 CARD. 前缀，按候选顺序）与选中的牌（跳过等其他选项为 None）。"""
    names, picked = [], None
    for k in range(o.num_actions):
        if int(o.cand_ints[k][0]) == ActionKind.PICK_CARD_REWARD:
            n = cand_name(o, k, vocab).removeprefix("CARD.")
            names.append(n)
            if k == a:
                picked = n
    return tuple(names), picked


def ancient_offer(o, a: int, vocab: list[str]) -> tuple[tuple[str, ...], str | None] | None:
    """古神（先古）三选一遗物（2026-10-27 起记录）：事件里至少两个选项各附带一件遗物时，返回 (这些遗物，去掉 RELIC. 前缀、按候选顺序,
    选中的那件；选了不带遗物的选项为 None)；不是这种界面返回 None。涅奥与各幕开头的古神都是这样的界面。"""
    if o.decision != DecisionKind.EVENT:
        return None
    rel = {}
    for k in range(o.num_actions):
        if int(o.cand_ints[k][0]) == ActionKind.EVENT_OPTION and int(o.cand_ints[k][1]) >= 0:
            r = int(o.tok_ints[int(o.cand_ints[k][1]), 4])
            if 0 < r < len(vocab):
                rel[k] = vocab[r].removeprefix("RELIC.")
    if len(rel) < 2:
        return None
    return tuple(rel.values()), rel.get(a)


def ancient_rows(obs, modes, vocab: list[str]) -> np.ndarray:
    """先古界面是观测属性：无论强制探索关闭、过期或未配置先古概率，都要记录。"""
    return np.array([mode == MODE_RUN and is_macro(o) and ancient_offer(o, 0, vocab) is not None
                     for o, mode in zip(obs, modes)], dtype=bool)


EXPLORE_KINDS = {DecisionKind.CARD_REWARD: "card_reward", DecisionKind.SHOP: "shop", DecisionKind.REST_SITE: "rest",
                 DecisionKind.EVENT: "event", DecisionKind.CARD_SELECT: "card_select", DecisionKind.MAP: "map"}


def explore_spec(o, vocab: list[str], probs: dict, temper: dict | None = None) -> tuple | None:
    """这个界面的强制探索规则：(类别, 改选概率, 古神三选一时带遗物的选项掩码否则 None, 温度改选参数 (T, ε) 否则 None)；不会改选时返回 None。
    古神三选一（ancient_offer）只在带遗物的其他选项里选；其余类别在除抽中动作以外的全部合法选项里选。
    temper（config.explore_temper，决策记录 202）里有这个类别时，替代项按 swap_dist 抽，否则均匀抽。"""
    if o.num_actions < 2:
        return None
    if ancient_offer(o, 0, vocab) is not None:
        kind = "ancient"
        rel = np.array([int(o.cand_ints[k][0]) == ActionKind.EVENT_OPTION and int(o.cand_ints[k][1]) >= 0
                        and 0 < int(o.tok_ints[int(o.cand_ints[k][1]), 4]) < len(vocab) for k in range(o.num_actions)])
    else:
        kind, rel = EXPLORE_KINDS.get(o.decision), None
    p = probs.get(kind, 0.0) if kind else 0.0
    tp = (temper or {}).get(kind)
    return (kind, p, rel, None if tp is None else (float(tp[0]), float(tp[1]))) if p > 0 else None


def swap_dist(pi: np.ndarray, pool: list[int], tp: tuple[float, float]) -> np.ndarray:
    """温度改选（决策记录 202）：替代项 pool 上的抽取概率 = (1 − ε)·π^(1/T) 归一 + ε / |pool|。π 在 pool 上全为 0 时退为均匀。"""
    t, eps = tp
    w = np.power(np.maximum(pi[pool], 0.0), 1.0 / t)
    s = w.sum()
    base = w / s if s > 0 else np.full(len(pool), 1.0 / len(pool))
    return (1 - eps) * base + eps / len(pool)


def explore_choice(o, a: int, vocab: list[str], probs: dict, rng, spec=None, pi: np.ndarray | None = None) -> tuple[str, int] | None:
    """强制探索（config.explore_probs）：这一步要改选时返回 (类别, 改选的动作)，否则 None。spec 为 explore_spec 的结果（不给就现算，
    这时不用温度改选）。温度改选的类别要给 pi（策略在全部合法动作上的概率，与 explore_logmu 用的同一份）。"""
    spec = spec if spec is not None else explore_spec(o, vocab, probs)
    if spec is None:
        return None
    kind, p, rel, tp = spec
    pool = [k for k in range(o.num_actions) if k != a and (rel is None or rel[k])]
    if not pool or rng.random() >= p:
        return None
    if tp is None:
        return kind, int(pool[int(rng.integers(len(pool)))])
    return kind, int(pool[int(rng.choice(len(pool), p=swap_dist(np.asarray(pi, np.float64), pool, tp)))])


def explore_logmu(pi: np.ndarray, b: int, spec) -> float:
    """强制探索下动作 b 实际被选的概率（行为分布 μ）的对数（决策记录 197）。pi 为策略在全部合法动作上的概率。
    策略抽中 a 后以概率 p 换成 pool(a) 里按 q_a 抽的一个：μ(b) = Σ_a π(a)·[(1−p)·1(b = a) + p·q_a(b)]。
    均匀改选 q_a = 1 / |pool(a)|：一般类别 pool(a) 为 a 以外的全部选项，μ(b) = (1−p)π(b) + p(1−π(b))/(n−1)；古神三选一 pool(a) 为 a 以外带遗物的选项。
    温度改选（决策记录 202）q_a 为 swap_dist(π, pool(a))，逐个 a 加总（局外界面选项少，n² 的开销可以忽略）。"""
    _, p, rel, tp = spec
    n = len(pi)
    if tp is not None:
        mu = 0.0
        for a in range(n):
            pool = [k for k in range(n) if k != a and (rel is None or rel[k])]
            if not pool:
                mu += pi[a] * (a == b)
                continue
            mu += pi[a] * (1 - p) * (a == b)
            if b in pool:
                mu += pi[a] * p * swap_dist(pi, pool, tp)[pool.index(b)]
    elif rel is None:
        mu = (1 - p) * pi[b] + p * (1 - pi[b]) / (n - 1)
    else:
        size = rel.sum() - rel.astype(np.int64)         # 抽中 a 时可换的选项数
        share = np.where(size > 0, pi / np.maximum(size, 1), 0.0)
        stay = np.where(size > 0, 1 - p, 1.0)            # 没得换的（理论上不会有）原样保留
        mu = stay[b] * pi[b] + (p * (share.sum() - share[b]) if rel[b] else 0.0)
    return float(np.log(max(mu, 1e-12)))


def combat_explore_spec(o, t: "Trajectory", vocab: list[str], table: dict, room: str | None, gate: dict) -> tuple | None:
    """战斗定向探索（config.combat_explore，决策记录 238）这一步的规则：(名单牌名, 各自的出牌候选, 各自的举手概率)；不抽签时返回 None。
    名单牌按 table[牌][room] 取概率，只收只有一个出牌候选的（有目标的牌、手里同名的有两张都不收）。这一回合第一次有名单牌可打时记下
    t.cx_key（层数、第几场 header[23]、回合），同一回合后面的决策不再抽签；门关（现在结束回合预计掉血不低于当前生命的 loss_frac，或本回合能杀光
    全部敌人）时这一回合也算抽过、不改，返回 None。"""
    if not table or room is None or not o.in_combat or o.decision != DecisionKind.COMBAT:
        return None
    g = np.flatnonzero(o.tok_ints[:, 0] == TokenKind.GLOBAL)
    key = (int(o.floor), int(o.header[23]), int(o.tok_feats[g[0], 9]) if len(g) else -1)     # 回合数是全局记录的特征 9（回放里的 g[7]）
    if t.cx_key == key:
        return None
    cand: dict[str, list[int]] = {}
    for k in range(o.num_actions):
        if int(o.cand_ints[k, 0]) != ActionKind.PLAY_CARD:
            continue
        tok = int(o.cand_ints[k, 1])
        vid = int(o.tok_ints[tok, 1]) if tok >= 0 else 0
        name = vocab[vid] if 0 < vid < len(vocab) else None
        if name is not None and float((table.get(name) or {}).get(room, 0.0)) > 0:
            cand.setdefault(name, []).append(k)
    picks = [(n, ks[0]) for n, ks in cand.items() if len(ks) == 1]
    if not picks:
        return None
    t.cx_key = key
    kind, tf = o.tok_ints[:, 0], o.tok_feats
    cre = np.flatnonzero(kind == TokenKind.CREATURE)
    me = [j for j in cre if tf[j, 5]]
    if me and gate.get("loss_frac") is not None and tf[me[0], 17] >= float(gate["loss_frac"]) * tf[me[0], 0]:
        return None
    foes = [j for j in cre if int(o.tok_ints[j, 2]) == Zone.ENEMY and tf[j, 3]]
    if gate.get("kill") and foes and all(tf[j, 19] >= 1 for j in foes):
        return None
    names = [n for n, _ in picks]
    return names, np.array([k for _, k in picks], np.int64), np.array([float(table[n][room]) for n in names], np.float64)


def combat_explore_q(ps: np.ndarray) -> tuple[np.ndarray, float]:
    """举手抽签：每张牌独立以 ps[i] 举手，有人举手就在举手的牌里均匀选一张。返回 (q：最后选到各张牌的概率, q_any：有人举手的概率)。
    对举手集合 S 枚举：q_c = Σ_{S∋c} Π_{i∈S} p_i · Π_{i∉S} (1 − p_i) / |S|。名单牌只有几张，2^n 可以忽略。"""
    n = len(ps)
    q = np.zeros(n)
    for m in range(1, 1 << n):
        s = [i for i in range(n) if m >> i & 1]
        w = float(np.prod([ps[i] if m >> i & 1 else 1 - ps[i] for i in range(n)]))
        q[s] += w / len(s)
    return q, float(1 - np.prod(1 - ps))


def combat_explore_choice(a: int, spec, rng) -> tuple[str, int] | None:
    """战斗定向探索：要改选时返回 (牌名, 改选的动作)，否则 None。策略抽中的 a 已经是名单牌时不改（也不抽签）。"""
    names, ks, ps = spec
    if a in ks:
        return None
    up = np.flatnonzero(rng.random(len(ps)) < ps)
    if not len(up):
        return None
    j = int(up[int(rng.integers(len(up)))])
    return names[j], int(ks[j])


def combat_explore_logmu(pi: np.ndarray, b: int, spec) -> float:
    """战斗定向探索下动作 b 的行为概率 μ 的对数。π(P) 为名单牌候选的策略概率之和，q、q_any 见 combat_explore_q：
    b 在 P 里时 μ(b) = π(b) + (1 − π(P))·q_b，否则 μ(b) = π(b)·(1 − q_any)。"""
    _, ks, ps = spec
    q, q_any = combat_explore_q(ps)
    pp = float(pi[ks].sum())
    hit = np.flatnonzero(ks == b)
    mu = pi[b] + (1 - pp) * q[hit[0]] if len(hit) else pi[b] * (1 - q_any)
    return float(np.log(max(mu, 1e-12)))


def offer_counts(offers: dict) -> dict[str, list[int]]:
    """一局的三选一记录 → {名字: [第1/2/3幕提供次数, 第1/2/3幕选中次数]}；"_SCREEN" 记各幕界面数与没选的次数。
    offers 以 (层数, 提供的东西) 为键，同一界面离开又回来再选，只按最后一次算。选牌奖励（评测的 cards）与古神遗物都用它。"""
    out: dict[str, list[int]] = {}
    for (_, names), (act, picked) in offers.items():
        if not names:
            continue
        act = min(max(act, 0), 2)
        scr = out.setdefault("_SCREEN", [0] * 6)
        scr[act] += 1
        scr[3 + act] += picked is None
        for n in names:
            out.setdefault(n, [0] * 6)[act] += 1
        if picked is not None:
            out[picked][3 + act] += 1
    return out


def merge_counts(items) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for c in items:
        for n, v in (c or {}).items():
            acc = out.setdefault(n, [0] * len(v))
            for j, x in enumerate(v):
                acc[j] += x
    return out


def frame_sig(o) -> int:
    """一帧的签名（决策记录 267）：记录、记录特征与候选三张表的 64 位摘要。引擎确定，同种子、同样的动作走到的同一局面签名相同；
    头部不算（第 12、14 项按进程累计）。0 留作“没有签名”。"""
    h = hashlib.blake2b(digest_size=8)
    for a in (o.tok_ints, o.tok_feats, o.cand_ints):
        h.update(np.ascontiguousarray(a).tobytes())
    return int.from_bytes(h.digest(), "little", signed=True) or 1


def is_macro(o) -> bool:
    """整局里战斗之外的决策：选路线、抓牌、事件、商店、休息处等。"""
    return not o.in_combat and o.decision not in (DecisionKind.COMBAT, DecisionKind.GAME_OVER, DecisionKind.STUCK)


# 牌组大小分档的上界：≤15、16–20、21–25、26–30、31–35、≥36（Boss 练习胜率、评测的节点分布都按这一档）
DECK_BINS = (15, 20, 25, 30, 35)


def deck_bin(n: int) -> int:
    return int(np.searchsorted(DECK_BINS, n, side="left"))


def potions_held(o) -> int:
    return int(((o.tok_ints[:, 0] == TokenKind.POTION) & (o.tok_ints[:, 2] == Zone.ALLY)).sum())


def settle_discards(cfg, t: "Trajectory", o, scene_over: bool) -> None:
    """白丢药水的惩罚：丢了之后药水瓶数回升（领到新药水）就抵掉一次；离开场景（下一个选路线决策或对局结束）时还没抵掉的，
    每瓶在丢药水的那一步记 −potion_discard_penalty。"""
    h = potions_held(o)
    if t.pot_held is not None and h > t.pot_held:
        del t.pot_pending[:h - t.pot_held]
    t.pot_held = h
    if scene_over and t.pot_pending:
        for k in t.pot_pending:
            t.rewards[k] -= cfg.potion_discard_penalty
            reward_event(t, "potion_discard", -cfg.potion_discard_penalty, k)
        t.pot_wasted += len(t.pot_pending)
        t.pot_pending.clear()


def hp_potential(cfg, hp: float, max_hp: float, max_hp0: float) -> float:
    """血量势能 = run_hp_shaping × (当前最大生命 / 开局最大生命) × V(血量比例)，V(x) = x + hp_concavity × x(1 − x)。
    V(0) = 0、V(1) = 1，斜率从空血时的 1 + k 降到满血时的 1 − k：同样回 1 点血，血量越低加得越多；掉血同理。"""
    x = min(1.0, max(0.0, hp / max(1.0, max_hp)))
    return cfg.run_hp_shaping * max_hp / max_hp0 * (x + cfg.hp_concavity * x * (1.0 - x))


def relics_held(o) -> int:
    return int(((o.tok_ints[:, 0] == TokenKind.RELIC) & (o.tok_ints[:, 2] == Zone.ALLY)).sum())


_RARE = None


def rare_mask() -> np.ndarray:
    """词表编号 → 是否为稀有（金卡，目录里 rarity 为 Rare 的卡牌，含无色稀有）。"""
    global _RARE
    if _RARE is None:
        cat = Catalog()
        rare = {e["key"] for e in cat.of("CARD") if e.get("rarity") == "Rare"}
        _RARE = np.array([k in rare for k in cat.vocab], bool)
    return _RARE


def deck_potential(cfg, o) -> float:
    """牌组势能：超出当前层上限的牌数受罚，已升级的卡、稀有卡加分（只在战斗外算：牌组区域只在战斗外输出）。"""
    g = o.tok_ints[:, 0] == TokenKind.GLOBAL
    size = float(o.tok_feats[g, 12][0]) if g.any() else 0.0
    ref = deck_size_ref(cfg.character)
    limit = ref[min(int(o.floor), len(ref) - 1)] + cfg.deck_size_margin
    deck = (o.tok_ints[:, 0] == TokenKind.CARD) & (o.tok_ints[:, 2] == Zone.DECK)
    upgraded = int((o.tok_feats[deck, 3] > 0).sum())
    ids, mask = o.tok_ints[deck, 1], rare_mask()
    rare = int(mask[ids[(ids >= 0) & (ids < len(mask))]].sum())
    return -cfg.deck_over_coef * max(0.0, size - limit) + cfg.upgrade_coef * upgraded + cfg.rare_coef * rare


def choice_key(o, a: int, vocab: list[str]) -> str:
    """局外决策的选择归类成“决策:动作:对象类别”，如 REST_SITE:REST_OPTION:REST:HEAL、CARD_REWARD:CARD_REWARD_ALT:ALT:SKIP、
    MAP:MAP_MOVE:MAPNODE:Elite；卡牌、遗物、药水、事件选项只保留类别（CARD、RELIC、POTION、EVENTOPT），不展开到具体哪一张。"""
    name = cand_name(o, a, vocab)
    if name.startswith("EVENTOPT:"):
        name = "EVENTOPT"
    elif "." in name and ":" not in name:
        name = name.split(".")[0]
    return f"{o.decision.name}:{ActionKind(int(o.cand_ints[a][0])).name}:{name}"


# 无负面效果的遗物（决策记录 160、162）：名单在 engine/catalog/no_downside_relics.txt（引擎也读它，在奖励界面与宝箱里自动拿这些遗物），
# 91 件：奖励界面与宝箱能给故障机器人的 93 件遗物去掉佛珠手链、石化蟾蜍两件有取舍的。
NO_DOWNSIDE_RELICS = frozenset(
    l.strip() for l in (Path(__file__).resolve().parents[2] / "engine/catalog/no_downside_relics.txt").read_text(encoding="utf-8").splitlines()
    if l.strip() and not l.startswith("#"))


def skipped_free(o, a: int, vocab: list[str]) -> tuple[int, int]:
    """这一步主动放弃的无负面奖励（决策记录 160）：（无负面遗物件数，金币奖励份数）。奖励界面点离开时数还剩的，宝箱选跳过时数可拿的遗物；
    其他情况为 (0, 0)。"""
    d, kind = o.decision, int(o.cand_ints[a][0])
    if d == DecisionKind.REWARDS and kind == ActionKind.LEAVE:
        take = ActionKind.TAKE_REWARD
    elif d == DecisionKind.TREASURE and kind == ActionKind.SKIP_RELIC:
        take = ActionKind.PICK_RELIC
    else:
        return 0, 0
    names = [cand_name(o, k, vocab) for k in range(o.num_actions) if int(o.cand_ints[k][0]) == take]
    return (sum(n.startswith("RELIC.") and n[6:] in NO_DOWNSIDE_RELICS for n in names), sum(n == "REWARD:GoldReward" for n in names))


def milestone_scale(cfg, version: int) -> float:
    """里程碑奖励的系数（决策记录 271）：milestone_from 为 None 时为 1；否则从那一版起 milestone_ramp 版内由 1 线性降到 0。"""
    return 1.0 if cfg.milestone_from is None else 1.0 - ramp(cfg.milestone_from, cfg.milestone_ramp, version)


def ramp(start: int | None, span: int, version: int) -> float:
    """从 start 版起、span 版内由 0 线性升到 1（start 为 None 时为 0）。"""
    if start is None or version < start:
        return 0.0
    return 1.0 if span <= 0 else min(1.0, (version - start) / span)


def relic_offer(o, a: int, vocab: list[str]) -> tuple[list[str], list[int], str | None, bool] | None:
    """跳过遗物的探针（2026-10-28，决策记录 159）：奖励界面（领奖励 TAKE_REWARD 里的遗物）和宝箱（PICK_RELIC）这一步可拿的遗物，
    返回（遗物名列表〔去掉 RELIC.〕、对应候选下标、这一步拿的是哪件〔没拿为 None〕、这一步是否主动跳过〔奖励界面点离开、宝箱选跳过〕）；
    这一步没有可拿的遗物时为 None。商店不买、事件与古神的遗物不在此列（古神另见 ancient_offer）。"""
    d = o.decision
    if d == DecisionKind.REWARDS:
        take, leave = ActionKind.TAKE_REWARD, ActionKind.LEAVE
    elif d == DecisionKind.TREASURE:
        take, leave = ActionKind.PICK_RELIC, ActionKind.SKIP_RELIC
    else:
        return None
    ks = [k for k in range(o.num_actions) if int(o.cand_ints[k][0]) == take and cand_name(o, k, vocab).startswith("RELIC.")]
    if not ks:
        return None
    names = [cand_name(o, k, vocab).removeprefix("RELIC.") for k in ks]
    return names, ks, (names[ks.index(a)] if a in ks else None), int(o.cand_ints[a][0]) == leave


def relic_skip_detail(o, a: int, vocab: list[str], names: list[str], ks: list[int], probs) -> dict:
    """主动跳过遗物时的现场：局面、可拿的遗物、模型给每件遗物和“跳过”的概率、奖励界面上同时还剩什么。"""
    ti = o.tok_ints
    others = [cand_name(o, k, vocab) for k in range(o.num_actions) if k not in ks and k != a]
    return {"dec": o.decision.name, "act": int(o.act), "floor": int(o.floor), "hp": int(o.hp), "max_hp": int(o.max_hp), "gold": o.gold,
            "relics_held": relics_held(o), "potions": potions_held(o), "deck": int(((ti[:, 0] == TokenKind.CARD) & (ti[:, 2] == Zone.DECK)).sum()),
            "offered": names, "p_take": [round(float(probs[k]), 4) for k in ks] if probs is not None else None,
            "p_skip": round(float(probs[a]), 4) if probs is not None else None, "others": others}


def cand_name(o, a: int, vocab: list[str]) -> str:
    """候选 a 涉及对象的词表名（如 CARD.ZAP）；候选本身没有编号时取它引用的记录的编号，都没有为 "-"。"""
    ci = o.cand_ints[a]
    vid = int(ci[3])
    if vid == 0 and 0 <= ci[1] < len(o.tok_ints):
        vid = int(o.tok_ints[ci[1], 1])
    return vocab[vid] if 0 < vid < len(vocab) else "-"


_weights_mtime: dict[str, int] = {}
_cur_version = [-1]     # 这个采集进程当前用的权重版本：随轨迹送出（v_send），学习端把落后版本数拆成“采集时已落后”和“排队等待”两段（决策记录 196）


def load_weights(net: PolicyValueNet, path: str, current: int, wp: dict | None = None, use_ema: bool = False,
                 co: dict | None = None) -> tuple[int, float, float]:
    """权重文件 = {"version", "progress", "shaping", "state"}；版本没变就不加载。shaping 为牌组打分奖励的系数（打分头还没训练好时为 0）。
    采集进程每秒查一次：文件修改时间没变就直接返回，不再每次整个读进来（原来这一项占采集进程约 13% 的 CPU）。"""
    try:
        mtime = os.stat(path).st_mtime_ns
        if current >= 0 and _weights_mtime.get(path) == mtime:
            return current, -1.0, -1.0
        blob = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        _weights_mtime[path] = mtime
    except Exception:
        return current, -1.0, -1.0    # 还没有或正在被替换，下次再试
    if blob["version"] != current:
        # use_ema：评测用学习端的滑动平均权重（决策记录 223），文件里没有就用当前参数
        load_state(net, blob["ema_state"] if use_ema and blob.get("ema_state") else blob["state"])
        # 胜率势能的模型与系数（决策记录 189）：wp 为 {"net", "coef", "device"}，随权重一起换
        if wp is not None and blob.get("winprob"):
            if wp.get("net") is None:
                wp["net"] = winprob_online.build(blob["winprob"]["args"]).to(wp["device"]).eval()
            wp["net"].load_state_dict(blob["winprob"]["model"])
            wp["coef"] = float(blob.get("wp_coef", 0.0))
        # 跳过稳赢战斗的结果头（model.CombatOutcomeHeads）：co 为 {"mod", "ok"}，随权重一起换；文件里没有时 ok 为假，不跳过
        if co is not None and blob.get("co_state"):
            co["mod"].load_state_dict(blob["co_state"]); co["ok"] = True
    return blob["version"], blob["progress"], blob.get("shaping", 0.0)


def _enemy_hp(o) -> dict:
    """敌方生物按怪物分组 → [（最大生命 + 后面形态的血量, 当前生命 + 后面形态的血量）, ...]（按出场顺序）；死了的当前算 0。
    多形态的怪（战斗计数记录子类型 1，所属记录为怪物）：特征 3 为后面形态的血量合计。"""
    ti, tf = o.tok_ints, o.tok_feats
    later = {int(ti[j, 3]): float(tf[j, 3]) for j in np.flatnonzero((ti[:, 0] == TokenKind.COMBAT_STAT) & (ti[:, 2] == Zone.ENEMY) & (tf[:, 0] == 1))}
    out: dict = {}
    for j in np.flatnonzero((ti[:, 0] == TokenKind.CREATURE) & (ti[:, 2] == Zone.ENEMY)):
        f = tf[j]; rest = later.get(int(j), 0.0)
        out.setdefault(int(ti[j, 1]), []).append((float(f[1]) + rest, (float(f[0]) if f[3] else 0.0) + rest))
    return out


def boss_score(t: "Trajectory", win: bool, dmg_weight: float, first: int = 0) -> float:
    """Boss 练习的成绩：打赢为 1，打输为 dmg_weight × 打掉的敌人血量比例。
    2026-10-29 起（决策记录 182）只算开场就在的敌人：每种怪最多算开场时的只数（按出场顺序；站位会因死亡、召唤错开，不用它认），
    中途召唤的其他怪分子分母都不算；多形态的怪按“当前生命 + 后面各形态的血量”算（原来打完第一形态后，
    第二形态的血量拿去和第一形态的最大生命比，部分分会失真）。first：从第几步算起（连打练习的第二场，决策记录 187）。"""
    if win:
        return 1.0
    start, end = _enemy_hp(t.obs[first]), _enemy_hp(t.obs[-1])
    total = sum(m for v in start.values() for m, _ in v)
    left = sum(c for k, v in start.items() for _, c in end.get(k, [])[:len(v)])
    return dmg_weight * float(np.clip(1 - left / total, 0, 1)) if total > 0 else 0.0


def luck_room(cfg, t: "Trajectory") -> bool:
    """运气校正（config.luck_cv）要不要管这一场战斗。luck_cv_rooms 为逗号分隔的房间类型（boss、elite、monster），或 "all" 为全部战斗。
    整局按当前战斗的房间类型；战斗练习按 spec：Boss 为 Boss 名额、Boss 房间、随机牌组的 boss 遭遇与连打（与结算时 is_boss 的判断相同），
    精英为精英房间与随机牌组的 elite 遭遇（同结算时的 is_elite），其余算普通战。"""
    kinds = {x.strip().lower() for x in str(cfg.luck_cv_rooms).split(",") if x.strip()}
    if "all" in kinds:
        return True
    if t.mode == MODE_RUN:
        return t.room is not None and t.room.lower() in kinds
    s = t.spec
    enc = str(s.get("encounter", "")).lower()
    if s.get("slot") == "boss" or s.get("room") == "Boss" or enc == "boss" or bool(s.get("next_encounter")):
        return "boss" in kinds
    if s.get("room") == "Elite" or enc == "elite":
        return "elite" in kinds
    return "monster" in kinds


def luck_ended(o) -> bool:
    """这一帧是不是已经打完了这场战斗（局结束、卡住、离开战斗或没有可选动作）：这样的推演或真实那一步不做运气校正。"""
    return bool(o.done or o.stuck or o.num_actions == 0 or not o.in_combat)


def _actor_loop(rank: int, cfg, weights_path: str, out_queue, stop_event) -> None:
    torch.set_num_threads(cfg.actor_threads); use_tf32()
    parent = os.getppid()
    device = torch.device(cfg.actor_device)
    wp = {"net": None, "coef": 0.0, "device": device}     # 胜率势能的模型与系数（随权重换，决策记录 189）
    prog_on = cfg.prog_floor > 0 or any(float(x) > 0 for x in cfg.prog_boss)     # 进度势能（决策记录 191）
    bv_est = None       # 构筑价值势能（决策记录 252）：冻结的模型，采集进程各自在 CPU 上算
    if cfg.bv_shaping > 0 and cfg.bv_model:
        from .buildvalue_model import Estimator, obs_row as bv_row
        bv_est = Estimator(cfg.bv_model)
    rng = np.random.default_rng(cfg.seed * 1000 + rank)
    cat = Catalog()
    vocab = cat.vocab
    free_pots = {i for i, k in enumerate(vocab) if k.startswith("POTION.") and k[len("POTION."):] in set(cfg.potion_free)}
    explore = {k: float(v) for k, v in (cfg.explore_probs or {}).items() if float(v) > 0}
    cexp = {k: dict(v) for k, v in (cfg.combat_explore or {}).items() if any(float(x) > 0 for x in (v or {}).values())}
    if cexp and not cfg.explore_offpolicy:
        raise ValueError("combat_explore 要求 explore_offpolicy 为开（改选的步要按行为分布 μ 记概率，决策记录 238）")
    cgate = dict(cfg.combat_explore_gate or {})
    temper = dict(cfg.explore_temper or {})
    cur = CombatCurriculum(cat, CurriculumConfig.for_character(cfg.character), seed=cfg.seed * 1000 + rank)
    w = np.asarray(cfg.boss_act_probs, dtype=float); cur.boss_act_probs = tuple(w / w.sum())
    cur.loss_power = cfg.boss_loss_power
    cur.pq_weight, cur.pq_power, cur.pq_drop, cur.pq_mix = cfg.boss_pq_weight, cfg.boss_pq_power, cfg.boss_pq_drop, cfg.boss_pq_mix
    pools = SnapshotPools(cfg.snapshot_pool)
    act_pools = {1: deque(maxlen=cfg.act_start_pool), 2: deque(maxlen=cfg.act_start_pool)}   # 开幕快照（中途开局用）
    start_p = np.asarray(cfg.run_start_probs, dtype=float); start_p = start_p / start_p.sum()
    solver = SolverPool(cfg, rank)
    # 房间结果记录（room_log.py）：单独的随机数，不改变课程抽样的随机数序列
    room_log = RoomLog(cfg.run_dir, rank, cfg.room_log, np.random.default_rng(cfg.seed * 1000 + rank + 7777))
    tape_rng = np.random.default_rng(cfg.seed * 1000 + rank + 9999)     # 整局回放的抽样，同样单独用一份随机数
    tape_dir = os.path.join(cfg.run_dir, "run_replays")
    value_probe = ActorProbe(cfg, rank) if cfg.value_probe else None
    net = PolicyValueNet(cfg.model).eval()
    # 跳过稳赢战斗（config.skip_easy，reports/跳过稳赢战斗方案.md）：结果头随权重换；判定、抽结果用单独的随机数，不改变课程抽样的随机数序列
    co = {"mod": CombatOutcomeHeads(cfg.model.d).eval(), "ok": False} if cfg.skip_easy else None
    skip_rng = np.random.default_rng(cfg.seed * 1000 + rank + 5555)
    skip_black = tuple(cfg.skip_blacklist or ())
    skip_off: dict = {}         # （幕, 房间）→ 掉血校准偏差：真打的够条件战斗里，实际战后血量比例减预测期望，按 skip_hp_ema 滚动
    skip_steps: dict = {}       # （幕, 房间）→ 真打这类战斗的平均步数（跳过时给整局分段补步），滚动估计；没数据时用下表（第 24,020 版实测）
    SKIP_STEPS0 = {(0, "Monster"): 26, (0, "Elite"): 34, (1, "Monster"): 34, (1, "Elite"): 37, (2, "Monster"): 36, (2, "Elite"): 49}
    _CO_LABELS[0] = bool(cfg.co_heads)
    _FH_LABELS[0] = int(cfg.fh_h) if cfg.model.future_head else 0
    version, progress, shaping = -1, cfg.progress_init, 0.0
    while version < 0 and not stop_event.is_set():
        version, p, sh = load_weights(net, weights_path, version, wp, co=co)
        _cur_version[0] = version
        if p >= 0:
            progress, shaping = p, sh
        if version < 0:
            time.sleep(0.2)
    net.to(device)
    if co is not None:
        co["mod"].to(device)
    det_rooms = {r for r in cfg.det_greedy_rooms.split(",") if r}     # 整局里按贪心确定性出手的房间（det），默认空
    srch = None     # 采集端用搜索出手（2026-11-17，局外适应搜索战斗，sts2rl/actor_search.py）；默认关
    if cfg.search_collect_rooms:
        from .actor_search import ActorSearch
        srch = ActorSearch(cfg, net, device, rank, version)
        print(f"采集进程 {rank}：{cfg.search_collect_rooms} 的战斗照部署方式出手（犹豫时 {cfg.search_collect_sims} 次模拟的树搜索）", flush=True)

    n = cfg.envs_per_actor
    n_run = int(round(n * cfg.run_fraction))
    # 第三幕双 Boss 专项（决策记录 287）：题目文件按修改时间重读
    chain = {"path": (cfg.chain_problems if os.path.isabs(cfg.chain_problems) else os.path.join(cfg.run_dir, cfg.chain_problems)) if cfg.chain_problems else "",
             "mtime": None, "items": [], "w": None}

    def chain_reload() -> None:
        try:
            mt = os.path.getmtime(chain["path"])
        except OSError:
            return
        if mt == chain["mtime"]:
            return
        items = [json.loads(l) for l in open(chain["path"], encoding="utf-8") if l.strip()]
        items = [x for x in items if x.get("next") and x.get("snap") and float(x.get("w", 1.0)) > 0]
        if items:
            w = np.array([float(x.get("w", 1.0)) for x in items])
            chain.update(mtime=mt, items=items, w=w / w.sum())
            print(f"采集进程 {rank} 读入专项题目 {len(items)} 道", flush=True)
    # 引擎的标准错误（卡住原因、异常堆栈）写到运行目录，便于事后查
    env = VecEnv(n, log=open(os.path.join(cfg.run_dir, "engine.log"), "ab"), engine=cfg.engine)
    trajs: list[Trajectory | None] = [None] * n
    ready_obs: dict[int, object] = {}
    # 运气校正（config.luck_cv）：在途的推演 {引擎下标: 状态}；推演都回来、真实那一步也回来的，等下一次凑批时一起算价值
    cv: dict[int, dict] = {}
    cv_ready: list[tuple] = []
    luck_rng = np.random.default_rng(cfg.seed * 1000 + rank + 7_000_003)      # 单独的随机数，不改变其余抽样的序列
    grp = {"seed": None, "left": 0}     # 同种子成组采集（决策记录 267）：当前在发的种子与还要发几局
    # 训练加血（config.train_bonus_hp，铁甲战士决策记录第 24 条）：只加在训练采集的 reset 上（整局与战斗练习），评测等别处不带
    bonus_flags = {"bonus_hp": int(cfg.train_bonus_hp)} if cfg.train_bonus_hp else {}

    def start(i: int):
        if i < n_run:
            spec = {"character": cfg.character, "ascension": 10, "seed": f"R{rank}_{int(rng.integers(1 << 40))}"}
            if cfg.run_group > 1:       # 同一个种子连续发给 run_group 局（接着空出来的引擎），学习端按种子成组
                if grp["left"] <= 0:
                    grp["seed"], grp["left"] = spec["seed"], cfg.run_group
                spec["seed"] = grp["seed"]; spec["grp"] = cfg.run_group; grp["left"] -= 1
            a = int(rng.choice(len(start_p), p=start_p)) if start_p[0] < 1 else 0
            # privileged：只有训练采集要特权特征（抽牌顺序、预抽的遭遇、怪物抽签推演），只给价值网络；评测、实机、搜索都不要（决策记录 161）
            req = {"op": "reset", "mode": "run", "snapshots": True, "route_summary": cfg.route_summary, "privileged": True,
                   **{k: v for k, v in spec.items() if k != "grp"}}
            req.update(cfg.cut_flags())         # 局面重复截断、跨回合无进展判输（引擎 RunSession.LoopCut、StallTurns）
            req.update(bonus_flags)             # 训练加血（铁甲战士决策记录第 24 条），关闭时为空
            if cfg.model.future_content:
                req["future_content"] = True    # 整局预生成内容（决策记录 259）：战斗外的帧多出类型 17 的记录，只进价值头
            if cfg.model.oracle:
                req["oracle"] = True            # 预知记录（决策记录 280）：战斗外的帧多出类型 20 的记录，只进价值头
            if cfg.manual_gold_start is not None and rng.random() < ramp(cfg.manual_gold_start, cfg.manual_gold_span, version):
                req["auto_gold"] = False        # 金币改由策略领（决策记录 160），按版本逐步提高这样开局的比例
            if a and act_pools.get(a):       # 中途开局：从这一幕的开幕快照里随机取一份；池子还空就从第一层开
                pool = act_pools[a]
                req["act_start"] = pool[int(rng.integers(len(pool)))]
                spec["start_act"] = a
            trajs[i] = Trajectory(MODE_RUN, version, spec)
            trajs[i].req = req
            if cfg.skip_easy and skip_rng.random() < cfg.skip_real_runs:
                trajs[i].co_real = True; trajs[i].choices["SKIP_REAL_RUN"] += 1    # 完全真打的局：不跳过任何战斗（对照与校准）
            if value_probe:
                value_probe.begin(trajs[i])
            if cfg.run_replay_frac > 0 and "start_act" not in spec and tape_rng.random() < cfg.run_replay_frac:
                trajs[i].tape = []
            env.send(i, req)
        elif chain["path"] and (chain_reload() or chain["items"]):
            spec = cur.chain_problem(chain["items"], chain["w"])
            spec.pop("_pid", None)
            trajs[i] = Trajectory(MODE_COMBAT, version, spec)
            trajs[i].label_view = None
            trajs[i].pq_v = None
            trajs[i].req = {"op": "reset", "mode": "combat", "spec": spec, "privileged": True, **cfg.cut_flags(), **bonus_flags}
            if value_probe:
                value_probe.begin(trajs[i])
            env.send(i, trajs[i].req)
        else:
            spec = cur.sample(progress, cfg.boss_fraction, pools, cfg.boss_real_share, (cfg.act1_fraction, cfg.act2_fraction, cfg.act3_fraction),
                              cfg.probe_share, (cfg.act1_elite_share, cfg.act2_elite_share, cfg.act3_elite_share),
                              solver, cfg.solver_share, cfg.mix_fraction, cfg.mix_elite_share,
                              cfg.second_boss_share, cfg.boss_chain_share)
            view = spec.pop("_view", None)
            pv = spec.pop("_pv", None)
            trajs[i] = Trajectory(MODE_COMBAT, version, spec)
            trajs[i].label_view = view
            trajs[i].pq_v = pv
            trajs[i].req = {"op": "reset", "mode": "combat", "spec": spec, "privileged": True, **cfg.cut_flags(), **bonus_flags}
            if value_probe:
                value_probe.begin(trajs[i])
            env.send(i, trajs[i].req)

    def log_stuck(t: Trajectory, reason: str, detail: str = "") -> None:
        """卡住的局记一行到 run_dir/stuck.jsonl（2026-10-27 起）：原因、哪一版、开局请求原文、依次走过的动作、最后几步的决策类型，
        照着重放（reset 请求带 "replay": actions）就能复现。引擎判卡住、引擎报错的，引擎日志里另有带游戏报错全文的 [stuck-detail] / [error-detail]。"""
        try:
            with open(os.path.join(cfg.run_dir, "stuck.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps({"time": round(time.time(), 1), "rank": rank, "version": t.version, "reason": reason, "detail": detail[:500],
                                    "mode": "run" if t.mode == MODE_RUN else "combat", "act": int(t.act), "floor": int(t.floor), "steps": len(t.actions),
                                    "recent": [o.decision.name for o in t.obs[-30:]], "req": t.req, "actions": [int(a) for a in t.actions]},
                                   ensure_ascii=False, separators=(",", ":")) + "\n")
        except Exception as e:      # 记不下来不影响采集
            print(f"采集进程 {rank} 卡住记录写入失败：{e}", flush=True)

    def end(t: Trajectory, final) -> None:
        win = _finish(cfg, t, final, out_queue, rank)
        if value_probe:
            value_probe.finish(t, final)
        if t.tape and not final.stuck:
            save_run_replay(tape_dir, cfg.run_replay_keep, t, win, final, tape_rng)
        if t.mode == MODE_COMBAT and t.spec.get("player_snapshot") and not final.stuck and t.hp_start is not None and room_log.sample():
            room_log.write({"src": "practice", "v": t.version, "slot": t.spec.get("slot"), "act": t.spec.get("act"), "room": t.spec.get("room"),
                            "enc": t.spec.get("encounter"), "probe": bool(t.spec.get("probe")), "snap": t.spec["player_snapshot"],
                            "hp0": int(t.hp_start), "max0": int(t.max_hp), "hp1": int(final.hp), "win": bool(win), "pots_used": int(t.potions),
                            "steps": len(t.obs)})
        if t.spec.get("cert"):
            solver.record(t.spec["cert"], win)
        if t.mode == MODE_COMBAT and not final.stuck:      # 第三幕 Boss 快照按输率加权（决策记录 187）
            if t.spec.get("next_encounter"):
                cur.record(("chain", t.spec["encounter"], t.spec["next_encounter"]), win)
                if cfg.boss_pq_weight and getattr(t, "pq_v", None) is not None:     # 按学习潜力选题的校准数据（决策记录 226）
                    try:
                        with open(os.path.join(cfg.run_dir, "boss_pq.jsonl"), "a", encoding="utf-8") as f:
                            f.write(json.dumps({"v": round(t.pq_v, 4), "win": bool(win), "ver": t.version}) + "\n")
                    except OSError:
                        pass
            elif t.spec.get("second"):
                cur.record(("b2", t.spec["encounter"]), win)

    def receive(i: int, o) -> None:
        t = trajs[i]
        if isinstance(o, EngineError):
            # 引擎报错：这一局按卡住结束（记为失败），开下一局；不让整个采集进程退出
            print(f"采集进程 {rank} 引擎 {i} 报错，本局按卡住结束：{str(o).splitlines()[0][:200]}", flush=True)
            log_stuck(t, "引擎报错", str(o))
            hp = t.hp_start if t.hp_start is not None else 0
            end(t, SimpleNamespace(victory=False, stuck=True, hp=hp, floor=t.floor, act=t.act))
            start(i)
            return
        if t.mode == MODE_RUN and o.extra and "combat" in o.extra:
            c = o.extra["combat"]
            t.room, t.cb_hp, t.cb_max = c["room"], o.hp, o.max_hp
            t.fight0 = len(t.obs)
            if srch is not None:
                b3 = c["room"] == "Boss" and int(c.get("act", -1)) == 2
                t.srch_b3 += int(b3)
                t.srch_ctx = {"boss3_first": b3 and t.srch_b3 == 1, "floor0": int(o.floor)}
            if t.room == "Elite":
                t.elite_fights += 1
            elif t.room == "Boss":
                t.boss_pots.append((int(o.act), potions_held(o)))
            t.cb_key, t.cb_floor = f"{c['act']}:{c['room']}:{c['encounter']}", o.floor
            # 跳过稳赢战斗：标记下一次推理的是开战第一帧；"map" 为选路之后直接开打（上一步是地图决策），其余（事件里开的战等）不跳
            if cfg.skip_easy:
                t.cb_first = "map" if (t.obs and t.obs[-1].decision == DecisionKind.MAP) else "other"
                if c["room"] == "Boss" and int(c["act"]) == 2 and not any(b[0].startswith("2:") for b in t.bosses):
                    # 进第三幕第一个 Boss 时的血量比例与药水瓶数，按局的类别记（护栏：允许跳过的局与完全真打的局差多少）
                    kind = "real" if t.co_real else "skip"
                    t.choices[f"B3_N:{kind}"] += 1; t.choices[f"B3_HP:{kind}"] += o.hp / max(1, o.max_hp)
                    t.choices[f"B3_POT:{kind}"] += potions_held(o)
            t.cb_log = ({"src": "run", "v": t.version, "act": int(c["act"]), "floor": int(o.floor), "room": c["room"], "enc": c["encounter"],
                         "snap": t.snap, "hp0": int(o.hp), "max0": int(o.max_hp), "pots0": potions_held(o), "step0": len(t.obs)}
                        if t.snap and room_log.sample() else None)
        if o.extra and "act_start" in o.extra:
            ast = o.extra["act_start"]
            if ast.get("Act") in act_pools:
                act_pools[ast["Act"]].append(ast)
        if o.extra:
            if "snapshot" in o.extra:
                t.snap, t.view, t.snap_n = o.extra["snapshot"], deck_view(o), len(t.obs)
                if rng.random() < cfg.probe_prob:
                    pools.add_probe(o.act, t.snap, t.view)
            elif "combat" in o.extra and t.snap and len(t.obs) != t.snap_n + 1:
                # 快照只在选路线时导出：选路之后先经过事件（问号房里的事件开战、事件里回血或加最大生命等）才开打的战斗，
                # 快照已经不是开战时的状态，不进练习池（第十三份外部评审，决策记录 206）；只收选路之后紧接着开打的
                t.choices["SNAP_STALE"] += 1
            elif "combat" in o.extra and t.snap:
                # 同一幕已经打过一个 Boss：这是 A10 最后一幕的第二个 Boss，带着第一场打完后的血量
                c = o.extra["combat"]
                second = c["room"] == "Boss" and any(b[0].startswith(f"{c['act']}:") for b in t.bosses)
                # 第三幕第一个 Boss 的快照记下紧接着的第二个 Boss（头部 17），连打练习用（决策记录 187）
                nxt = vocab[int(o.header[17])] if c["room"] == "Boss" and not second and 0 < int(o.header[17]) < len(vocab) else None
                # 第二个 Boss 的快照记下本幕已经打赢的第一个 Boss（单打时填进观测头部 16，与整局对齐，观测格式 15）
                prev = next((b[0].split(":", 1)[1] for b in t.bosses if b[0].startswith(f"{c['act']}:")), None) if second else None
                ent = pools.add(c, t.snap, t.view, second=second, next_encounter=nxt, prev_encounter=prev)
                # 第三幕第一个 Boss：这一步算完价值后记到池里的这一条上（按学习潜力选题，决策记录 226）
                t.pq_entry = ent if (cfg.boss_pq_weight and nxt) else None
                solver.maybe_export(o.extra["combat"], t.snap)
        if t.hp_start is None:
            t.hp_start, t.max_hp, t.floor = o.hp, max(1, o.max_hp), o.floor
            t.hp_last = hp_potential(cfg, o.hp, o.max_hp, t.max_hp)
            t.relic_phi = cfg.relic_coef * relics_held(o)
        elif t.mode == MODE_RUN:
            phi = hp_potential(cfg, o.hp, o.max_hp, t.max_hp)
            if phi != t.hp_last:                                          # 血量塑形：势能变化记在导致变化的那一步
                reward_event(t, "hp_potential", phi - t.hp_last)
                t.rewards[-1] += phi - t.hp_last; t.hp_shaped += phi - t.hp_last; t.hp_last = phi
            phi = cfg.relic_coef * relics_held(o)
            if phi != t.relic_phi:                                        # 遗物势能：记在拿到（失去）遗物的那一步
                reward_event(t, "relic_potential", phi - t.relic_phi)
                t.rewards[-1] += phi - t.relic_phi; t.relic_shaped += phi - t.relic_phi; t.relic_phi = phi
            if t.room is not None and (not o.in_combat or o.done):        # 战斗结束：非 Boss 战按净掉血扣分，记在最后一步
                if t.co_pred is not None:       # 跳过稳赢战斗：够条件但真打了的战斗，打完更新校准偏差、平均步数与药水头的校准统计
                    key, pm, mx0, vids, pps, step0 = t.co_pred
                    t.co_pred = None
                    rk = f"{key[0]}:{key[1]}"
                    if o.done and not o.victory:
                        t.choices[f"SKIP_ELIG_LOST:{rk}"] += 1
                    elif not o.stuck:
                        res = o.hp / max(1, mx0) - pm
                        skip_off[key] = (1 - cfg.skip_hp_ema) * skip_off.get(key, 0.0) + cfg.skip_hp_ema * res
                        t.choices[f"SKIP_RES_N:{rk}"] += 1; t.choices[f"SKIP_RES_SUM:{rk}"] += float(res)
                        skip_steps[key] = 0.98 * skip_steps.get(key, SKIP_STEPS0.get(key, 35)) + 0.02 * (len(t.obs) - step0)
                        if not o.done:
                            left_pots = Counter(int(x) for x in held_potion_ids(o) if x > 0)     # 不能叫 end：会遮住这里的 end() 函数
                            for v, q in zip(vids, pps):
                                if v <= 0:
                                    continue
                                kept = left_pots.get(int(v), 0) > 0
                                if kept:
                                    left_pots[int(v)] -= 1
                                t.choices["SKIP_POT_N"] += 1; t.choices["SKIP_POT_P"] += float(q); t.choices["SKIP_POT_Y"] += int(not kept)
                if t.room == "Boss" and not o.stuck:
                    act, _, enc = t.cb_key.split(":", 2)
                    t.bosses.append([f"{act}:{enc}", not o.done or o.victory])
                    # 第三幕（A10 连打两个 Boss）打赢第一个、局还没结束：里程碑奖励（2026-10-26 起，默认 0）
                    if act == "2" and not o.done and sum(1 for b in t.bosses if b[0].startswith("2:")) == 1:
                        ms = milestone_scale(cfg, t.version); t.ms_full += cfg.act3_boss_reward
                        t.rewards[-1] += ms * cfg.act3_boss_reward
                        reward_event(t, "boss3_first", ms * cfg.act3_boss_reward)
                        # 第三幕中途余血奖励（config.boss3_hp_coef，铁甲战士决策记录第 25 条）：按进第二个 Boss 时“不加血时还剩几成血”给
                        if cfg.boss3_hp_coef:
                            bh = int((getattr(t, "req", None) or {}).get("bonus_hp", 0))
                            frac = min(1.0, max(0.0, (int(o.hp) - bh) / max(1, int(o.max_hp) - bh)))
                            t.rewards[-1] += cfg.boss3_hp_coef * frac
                            reward_event(t, "boss3_hp", cfg.boss3_hp_coef * frac)
                            t.choices["BOSS3_HP:n"] += 1; t.choices["BOSS3_HP:hp"] += int(o.hp); t.choices["BOSS3_HP:frac"] += frac
                if t.room != "Boss":
                    r = cfg.combat_dmg_coef * max(0, t.cb_hp - o.hp) / max(1, t.cb_max)
                    t.rewards[-1] -= r; t.dmg_shaped -= r
                    reward_event(t, "combat_damage", -r)
                if t.cb_log is not None and not o.stuck:
                    rec, t.cb_log = t.cb_log, None
                    rec.update(hp1=int(o.hp), max1=int(o.max_hp), pots1=potions_held(o), win=bool(not o.done or o.victory), steps=len(t.obs) - rec.pop("step0"))
                    room_log.write(rec)
                t.room = None
            if o.floor > t.floor:
                ms = milestone_scale(cfg, t.version); t.ms_full += cfg.floor_reward * (o.floor - t.floor)
                t.rewards[-1] += ms * cfg.floor_reward * (o.floor - t.floor)   # 记在导致爬层的那一步
                reward_event(t, "floor", ms * cfg.floor_reward * (o.floor - t.floor))
                t.floor = o.floor
            for a in range(t.act + 1, o.act + 1):                         # 进入下一幕 = 打过了上一幕的 Boss
                r_ = {1: cfg.act1_boss_reward, 2: cfg.act2_boss_reward}.get(a, 0.0); ms = milestone_scale(cfg, t.version); t.ms_full += r_
                t.rewards[-1] += ms * r_
                reward_event(t, f"boss{a}", ms * r_)
        t.act = o.act
        if t.mode == MODE_RUN:
            settle_discards(cfg, t, o, o.done or o.stuck or o.num_actions == 0 or o.decision == DecisionKind.MAP)
            if not o.in_combat and not o.done and o.num_actions > 0:
                phi = deck_potential(cfg, o)
                if t.deck_phi is not None and t.rewards:
                    reward_event(t, "deck_potential", phi - t.deck_phi)
                    t.rewards[-1] += phi - t.deck_phi; t.deck_shaped += phi - t.deck_phi
                t.deck_phi = phi
        if o.done or o.stuck or o.num_actions == 0:
            if o.stuck:
                log_stuck(t, "引擎判卡住")
            end(t, o)
            start(i)
        elif len(t.obs) >= (cfg.max_steps_run if t.mode == MODE_RUN else cfg.max_steps_combat):
            # 来回打转：按卡住结束（记为失败），引擎直接开下一局
            log_stuck(t, "超过步数上限", f"{len(t.obs)} 步")
            end(t, SimpleNamespace(victory=False, stuck=True, hp=o.hp, floor=o.floor, act=o.act))
            start(i)
        elif t.mode == MODE_RUN and cfg.max_steps_fight and t.room is not None and len(t.obs) - t.fight0 >= cfg.max_steps_fight:
            # 整局里单场战斗打转（亡灵契约师决策记录第 4 条补记二：十足蜈蚣节段反复接回，无进展判输判不下来）：按卡住结束（记为失败）。
            # 整局只在选路前切段提交，不截的话这一场几千步、每帧越来越大的数据会一次交给学习端
            log_stuck(t, "单场战斗超过步数上限", f"{len(t.obs) - t.fight0} 步")
            end(t, SimpleNamespace(victory=False, stuck=True, hp=o.hp, floor=o.floor, act=o.act))
            start(i)
        else:
            ready_obs[i] = o

    def luck_req(st: dict, j: int) -> dict:
        """第 j 个推演：恢复到选“结束回合”之前的局面，按公开信息换随机数（保留已知的抽牌堆顶，不换战后奖励），走同一个动作。"""
        return {"op": "restore", "slot": cfg.luck_cv_slot, "reseed": st["seeds"][j], "first": st["a"], "keep_top": True, "rewards": False}

    def reply(i: int, o) -> None:
        """引擎回复的分流：运气校正的推演回复先收下，推演齐了再恢复局面走真实的那一步；真实那一步的回复照常交给 receive。"""
        st = cv.get(i)
        if st is None:
            receive(i, o)
            return
        if isinstance(o, EngineError) or st["sent_real"]:
            del cv[i]
            if not isinstance(o, EngineError):
                t_ = st["t"]
                ends = [luck_ended(b) for b in st["obs"]] + [luck_ended(o)]
                if any(ends):
                    t_.choices["LUCK_SKIP"] += 1
                    t_.choices["LUCK_MIXED"] += int(not all(ends))     # 推演与真实那一步有的打完了、有的没打完
                else:
                    cv_ready.append((t_, st["step"], st["obs"]))
            receive(i, o)
            return
        st["obs"].append(o)
        if len(st["obs"]) < cfg.luck_cv_k:
            env.send(i, luck_req(st, len(st["obs"])))
        else:
            st["sent_real"] = True
            env.send(i, {"op": "restore", "slot": cfg.luck_cv_slot, "first": st["a"]})

    for i in range(n):
        start(i)
    last_check = last_labels = time.time()
    try:
        while not stop_event.is_set():
            for i, o in env.recv_ready(timeout=0.5):
                reply(i, o)
            # 凑批：所有在途的引擎都回来，或等满 infer_wait_ms
            if ready_obs and env.pending:
                deadline = time.time() + cfg.infer_wait_ms / 1000
                while env.pending and time.time() < deadline:
                    for i, o in env.recv_ready(timeout=max(0.0, deadline - time.time())):
                        reply(i, o)
            if ready_obs:
                ids = list(ready_obs.keys())
                obs = [ready_obs[i] for i in ids]
                modes = np.array([trajs[i].mode for i in ids])
                # 整局里的非战斗决策：局面视图拼在同一批后面，一遍前向同时得到打分；
                # 同一幕内与上一个非战斗决策相比的变化记为上一个决策的奖励
                macro = [k for k, i in enumerate(ids) if trajs[i].mode == MODE_RUN and is_macro(obs[k])] if shaping > 0 else []
                views = [deck_view(obs[k]) for k in macro]
                # 运气校正的推演帧（config.luck_cv）拼在局面视图之后，同一遍前向得到它们的价值
                cvo = [b for e in cv_ready for b in e[2]]
                cvm = np.array([e[0].mode for e in cv_ready for _ in e[2]], dtype=modes.dtype)
                batch = collate_ragged(Ragged.from_obs(obs + views + cvo, np.concatenate([modes, np.full(len(views), MODE_RUN), cvm]))).to(device)
                with torch.no_grad():
                    logits, values, z = net.forward_phi(batch)
                    values_cv = values[len(ids) + len(views):].float().cpu().numpy() if cvo else None
                    logits, values = logits[:len(ids)], values[:len(ids)]
                    dist = torch.distributions.Categorical(logits=logits)
                    acts = dist.sample()
                    if cfg.combat_greedy:       # 只训局外（决策记录 267）：整局里的战斗取概率最大的动作，局外照常采样
                        gm = torch.tensor([trajs[i].mode == MODE_RUN and not is_macro(o) for i, o in zip(ids, obs)], device=logits.device)
                        acts = torch.where(gm, logits.argmax(-1), acts)
                    if cfg.macro_greedy:        # 战斗阶段（决策记录 270）：整局里的局外决策取概率最大的动作，战斗照常采样
                        gm = torch.tensor([trajs[i].mode == MODE_RUN and is_macro(o) for i, o in zip(ids, obs)], device=logits.device)
                        acts = torch.where(gm, logits.argmax(-1), acts)
                    logp = dist.log_prob(acts)
                acts, logp, values = acts.cpu().numpy(), logp.cpu().numpy(), values.cpu().numpy()
                if cvo:
                    j = 0
                    for t_, step, bobs in cv_ready:
                        vs = values_cv[j:j + len(bobs)]; j += len(bobs)
                        if t_.seg0 <= step < len(t_.nv):
                            t_.nv[step] = float(vs.mean()); t_.choices["LUCK_N"] += 1
                    cv_ready.clear()
                forced = np.zeros(len(ids), bool); logp_mu = logp.copy()
                anc_step = ancient_rows(obs, modes, vocab)  # 统计识别独立于强制探索开关
                lsm = None
                if explore and not cfg.macro_greedy and (cfg.explore_until is None or version < cfg.explore_until):
                    for k, i in enumerate(ids):
                        if trajs[i].mode != MODE_RUN or not is_macro(obs[k]) or (spec := explore_spec(obs[k], vocab, explore, temper)) is None:
                            continue
                        if lsm is None:
                            lsm = torch.log_softmax(logits.float(), -1).cpu().numpy()
                        pi_k = np.exp(lsm[k, :obs[k].num_actions].astype(np.float64))
                        if (ex := explore_choice(obs[k], int(acts[k]), vocab, explore, rng, spec, pi_k)) is not None:
                            acts[k], logp[k], forced[k] = ex[1], lsm[k, ex[1]], True
                            trajs[i].choices["EXPLORE:" + ex[0]] += 1
                        # 行为概率：会改选的界面上每一步（改没改都算）都按 μ 记，学习端用它算重要性比率（决策记录 197）
                        logp_mu[k] = explore_logmu(pi_k, int(acts[k]), spec)
                # 战斗定向探索（决策记录 238）：整局里的战斗与战斗练习都做，房间取整局当前战斗的 t.room 或练习题的 spec["room"]
                if cexp and (cfg.combat_explore_until is None or version < cfg.combat_explore_until):
                    for k, i in enumerate(ids):
                        tr = trajs[i]
                        room = tr.room if tr.mode == MODE_RUN else tr.spec.get("room")
                        if (spec := combat_explore_spec(obs[k], tr, vocab, cexp, room, cgate)) is None:
                            continue
                        if lsm is None:
                            lsm = torch.log_softmax(logits.float(), -1).cpu().numpy()
                        pi_k = np.exp(lsm[k, :obs[k].num_actions].astype(np.float64))
                        if (ex := combat_explore_choice(int(acts[k]), spec, rng)) is not None:
                            acts[k], logp[k], forced[k] = ex[1], lsm[k, ex[1]], True
                            tr.choices["EXPLORE:combat:" + ex[0]] += 1
                        logp_mu[k] = combat_explore_logmu(pi_k, int(acts[k]), spec)
                # 胜率势能（决策记录 189）：整局的每一步算特征，抽样留作胜率模型的训练样本；有模型、系数大于 0 时算势能
                det = np.zeros(len(ids), bool)
                if srch is not None or det_rooms:     # 照部署方式出手（覆盖采样与战斗定向探索），记为 det：search_collect_rooms 做搜索，det_greedy_rooms 只取最大
                    pr_all = None
                    for k, i in enumerate(ids):
                        tr = trajs[i]
                        if tr.mode != MODE_RUN or not obs[k].in_combat:
                            continue
                        use_s = srch is not None and srch.wants(tr, obs[k])
                        if not use_s and tr.room not in det_rooms:
                            continue
                        if pr_all is None:
                            pr_all = torch.softmax(logits.float(), -1).double().cpu().numpy()
                        nk = int(obs[k].num_actions)
                        a_k, did = srch.decide(tr, obs[k], pr_all[k, :nk], version) if use_s else (int(np.argmax(pr_all[k, :nk])), False)
                        acts[k] = a_k; logp[k] = float(np.log(max(pr_all[k, a_k], 1e-30))); logp_mu[k] = logp[k]; forced[k] = False; det[k] = True
                        tr.choices["SEARCH:det"] += 1
                        if did:
                            tr.choices["SEARCH:searched"] += 1
                            tr.choices["SEARCH:changed"] += int(a_k != int(np.argmax(pr_all[k, :nk])))
                wp_phi, pg_phi = {}, {}
                if cfg.wp_shaping > 0 or prog_on:
                    run_k = []
                    rows = []
                    for k, i in enumerate(ids):
                        t, o = trajs[i], obs[k]
                        if t.mode != MODE_RUN:
                            continue
                        key = winprob_online.checkpoint(o)      # 只在战斗外的决策、战斗中每回合第一步算
                        if key == t.wp_key and key[2] < 0:
                            continue
                        t.wp_key = key; run_k.append(k)
                        if prog_on:     # 进度势能：层数、打过的幕末 Boss（进到第二、三幕）、打过的第三幕第一个 Boss
                            b3 = any(b[0].startswith("2:") and b[1] for b in t.bosses)
                            pg_phi[k] = (cfg.prog_floor * float(o.floor) + cfg.prog_boss[0] * (o.act >= 1) + cfg.prog_boss[1] * (o.act >= 2)
                                         + cfg.prog_boss[2] * b3)
                        if cfg.wp_shaping <= 0:
                            continue
                        dk = winprob_online.deck_of(o)
                        if dk is not None:
                            t.wp_deck, t.wp_up = dk, winprob_online.upgraded(o)
                        row = winprob_online.features(o, t.wp_deck, t.wp_up)
                        rows.append(row)
                        fl = t.wp_rows.setdefault(int(o.floor), [0, []])
                        fl[0] += 1
                        if len(fl[1]) < cfg.wp_per_floor:
                            fl[1].append(row)
                        elif (j := int(rng.integers(fl[0]))) < cfg.wp_per_floor:
                            fl[1][j] = row
                    if wp.get("net") is not None and wp["coef"] > 0 and run_k:
                        p_win = winprob_online.predict(wp["net"], rows, wp["device"])
                        wp_phi = {k: float(wp["coef"] * p) for k, p in zip(run_k, p_win)}
                bv_phi = {}
                if bv_est is not None:      # 构筑价值势能：整局里每个战斗外的决策算一次
                    bv_c = cfg.bv_shaping if cfg.bv_start is None else cfg.bv_shaping * float(np.clip((version - cfg.bv_start) / max(1, cfg.bv_ramp), 0, 1))
                    bk, brows = [], []
                    for k, i in enumerate(ids):
                        if trajs[i].mode == MODE_RUN and bv_c > 0 and (bv_r := bv_row(obs[k])) is not None and bv_r[5] not in (0, 12):
                            bk.append(k); brows.append(bv_r)
                    if bk:
                        bv_phi = {k: float(bv_c * pb) for k, pb in zip(bk, bv_est(brows))}
                phi = np.full(len(ids), np.nan, np.float32)
                if macro:
                    phi[macro] = torch.sigmoid(z[len(ids):len(ids) + len(views)]).cpu().numpy()
                for k in macro:
                    t, o = trajs[ids[k]], obs[k]
                    if not np.isnan(phi[k]):
                        if t.phi_prev is not None and t.phi_prev[1] == o.act:
                            r = shaping * (float(phi[k]) - t.phi_prev[2])
                            t.rewards[t.phi_prev[0]] += r; t.shaped += r
                            reward_event(t, "learned_deck_potential", r, t.phi_prev[0])
                        t.phi_prev = (len(t.obs), o.act, float(phi[k]))
                # 跳过稳赢战斗（config.skip_easy）：开战第一帧够条件的，以 skip_prob 的概率不打，发 resolve 直接结束（这一帧不进轨迹）
                skip_k = {}
                if co is not None:
                    firsts = []
                    for k, i in enumerate(ids):
                        t = trajs[i]
                        if t.mode == MODE_RUN and t.cb_first:
                            # 第一帧要是普通的出牌决策：开战时就要选牌的（遗物、效果让开局先选牌）引擎的 resolve 不接，照常打
                            if (t.cb_first == "map" and obs[k].decision == DecisionKind.COMBAT and t.room in cfg.skip_rooms and co["ok"] and t.cb_key
                                    and (cfg.skip_from is None or version >= cfg.skip_from)
                                    and not t.cb_key.split(":", 2)[2].startswith(skip_black)):
                                firsts.append(k)
                            t.cb_first = False
                    if firsts:
                        sub = [obs[k] for k in firsts]
                        with torch.no_grad():
                            bt = collate_ragged(Ragged.from_obs(sub, np.full(len(sub), MODE_RUN))).to(device)
                            _, _, _, ex = net.forward_all(bt)
                            co_pw = torch.sigmoid(ex["aux"][:, 0].float()).cpu().numpy()
                            co_vid = np.stack([held_potion_ids(o_) for o_ in sub])
                            hl, pl = co["mod"](ex["g"], net.id_emb, torch.from_numpy(co_vid).to(device))
                            hp_pr = torch.softmax(hl.float(), -1).double().cpu().numpy(); pot_pr = torch.sigmoid(pl.float()).cpu().numpy()
                        cen = co["mod"].centers.double().cpu().numpy(); edg = co["mod"].edges.double().cpu().numpy()
                        for j, k in enumerate(firsts):
                            t, o = trajs[ids[k]], obs[k]
                            if co_pw[j] < cfg.skip_p:
                                continue
                            key = (int(o.act), t.room); rk = f"{key[0]}:{key[1]}"
                            t.choices[f"SKIP_ELIG:{rk}"] += 1
                            pm = float(hp_pr[j] @ cen)
                            has_pot = bool((co_vid[j] > 0).any())
                            go = (not t.co_real and skip_rng.random() < cfg.skip_prob
                                  and (not has_pot or cfg.skip_pot_from is None or version >= cfg.skip_pot_from))
                            if not go:      # 真打：打完拿来校准
                                t.co_pred = (key, pm, int(o.max_hp), co_vid[j].copy(), pot_pr[j].copy(), len(t.obs))
                                continue
                            pb = hp_pr[j] / hp_pr[j].sum()
                            b_ = int(skip_rng.choice(len(pb), p=pb))
                            frac = edg[b_] + skip_rng.random() * (edg[b_ + 1] - edg[b_]) + skip_off.get(key, 0.0)
                            mx = max(1, int(o.max_hp))
                            hp_new = int(min(max(1, round(frac * mx)), mx))
                            used = [vocab[int(v)] for v, q in zip(co_vid[j], pot_pr[j]) if v > 0 and skip_rng.random() < q]
                            skip_k[k] = {"op": "resolve", "hp": hp_new, "potions": used}
                            t.choices[f"SKIP_DONE:{rk}"] += 1
                            t.choices[f"SKIP_DHP_SUM:{rk}"] += (hp_new - int(o.hp)) / mx      # 抽到的血量变化（负为掉血），看板对照真打
                            t.choices["SKIP_POT_USED"] += len(used)
                            t.seg_fill += int(round(skip_steps.get(key, SKIP_STEPS0.get(key, 35))))
                pr = None       # 抽中另存回放的局要每步的出手概率，一批里有才算一次
                for k, i in enumerate(ids):
                    o, t, a = obs[k], trajs[i], int(acts[k])
                    if k in skip_k:
                        env.send(i, skip_k[k])
                        continue
                    if t.tape is not None:
                        if pr is None:
                            pr = torch.softmax(logits.float(), -1).cpu().numpy()
                        t.tape.append(json.dumps(describe(o, a, pr[k, :o.num_actions]), separators=(",", ":")))
                    # 整局分段提交：在选路线决策前切。白丢药水到这时已结算（settle_discards），牌组打分奖励记在上一个局外决策上、
                    # 它总在没送出的部分里，所以切开后不会再改已送出的步
                    if k in wp_phi:      # 胜率势能塑形：这一步的势能减上一步的，记在上一步（要在分段送出之前）
                        if t.wp_last is not None and t.rewards:
                            reward_event(t, "winprob_potential", wp_phi[k] - t.wp_last)
                            t.rewards[-1] += wp_phi[k] - t.wp_last; t.wp_shaped += wp_phi[k] - t.wp_last
                        t.wp_last = wp_phi[k]
                    if k in pg_phi:      # 进度势能塑形，同上
                        if t.pg_last is not None and t.rewards:
                            reward_event(t, "progress_potential", pg_phi[k] - t.pg_last)
                            t.rewards[-1] += pg_phi[k] - t.pg_last; t.pg_shaped += pg_phi[k] - t.pg_last
                        t.pg_last = pg_phi[k]
                    if k in bv_phi:      # 构筑价值势能塑形，同上
                        if t.bv_last is not None and t.rewards:
                            reward_event(t, "buildvalue_potential", bv_phi[k] - t.bv_last)
                            t.rewards[-1] += bv_phi[k] - t.bv_last; t.bv_shaped += bv_phi[k] - t.bv_last
                        t.bv_last = bv_phi[k]
                    if (cfg.run_segment_steps and t.mode == MODE_RUN and o.decision == DecisionKind.MAP
                            and len(t.obs) - t.seg0 + t.seg_fill >= cfg.run_segment_steps):
                        send_segment(t, float(values[k]), out_queue, rank, o)
                    t.phi.append(float(phi[k]))
                    # 连打练习：头部 23（引擎记的第几场）到 1，就是第二场开打了
                    if t.stage2 is None and t.spec.get("next_encounter") and int(o.header[23]) >= 1:
                        t.stage2 = len(t.obs)
                    t.obs.append(o); t.actions.append(a); t.logp.append(float(logp[k])); t.logp_mu.append(float(logp_mu[k])); t.values.append(float(values[k])); t.rewards.append(0.0); t.vers.append(version); t.nv.append(float("nan"))
                    if getattr(t, "pq_entry", None) is not None:
                        t.pq_entry["v"] = float(values[k]); t.pq_entry = None
                    t.forced.append(bool(forced[k])); t.anc.append(bool(anc_step[k])); t.det.append(bool(det[k]))
                    t.sig.append(frame_sig(o) if cfg.run_group > 1 and t.mode == MODE_RUN and is_macro(o) else 0)
                    if o.cand_ints[a, 0] == ActionKind.USE_POTION:
                        # 免扣的药水（config.potion_free，如药水形状的石头）：整局与战斗课程里用了都不扣分
                        pa = int(o.cand_ints[a, 1])
                        free = pa >= 0 and int(o.tok_ints[pa, 1]) in free_pots
                        t.potions += not free
                        t.elite_pots += t.mode == MODE_RUN and o.in_combat and t.room == "Elite"
                        # 非 Boss 战用药水扣分；elite_potion_free 时精英战也不扣（2026-10-27 起的对照实验）
                        if not free and t.mode == MODE_RUN and o.in_combat and t.room not in (None, "Boss") and not (cfg.elite_potion_free and t.room == "Elite"):
                            t.rewards[-1] -= cfg.run_potion_coef; t.pot_cost -= cfg.run_potion_coef
                            reward_event(t, "potion_use", -cfg.run_potion_coef)
                    if t.mode == MODE_RUN and is_macro(o):
                        t.choices[choice_key(o, a, vocab)] += 1
                        if (ro := relic_offer(o, a, vocab)) is not None:     # 跳过遗物的探针：训练里只计次数（采样的动作，详情看评测）
                            t.choices[f"RELIC_OFFER:{o.decision.name}"] += 1
                            if ro[3]:
                                t.choices[f"RELIC_SKIP:{o.decision.name}"] += 1
                        if o.decision == DecisionKind.REWARDS and any(cand_name(o, k, vocab) == "REWARD:GoldReward" for k in range(o.num_actions)):
                            t.choices["GOLD_OFFER"] += 1
                        nr, ng = skipped_free(o, a, vocab)
                        if nr or ng:        # 放弃无负面遗物、金币：衰减的惩罚（决策记录 160）
                            t.choices["FREE_SKIP:RELIC"] += nr; t.choices["FREE_SKIP:GOLD"] += ng
                            left = 1.0 - ramp(cfg.skip_penalty_start, cfg.skip_penalty_span, version) if cfg.skip_penalty_start is not None else 0.0
                            r = left * (cfg.relic_skip_penalty * nr + cfg.gold_skip_penalty * ng)
                            t.rewards[-1] -= r; t.skip_cost -= r
                            reward_event(t, "skip", -r)
                        if (anc := ancient_offer(o, a, vocab)) is not None:
                            t.ancient[(int(o.floor), anc[0])] = (int(o.act), anc[1])
                        if o.decision == DecisionKind.CARD_REWARD:
                            names, picked = card_offer(o, a, vocab)
                            t.cards[(int(o.floor), names)] = (int(o.act), picked)
                    if t.mode == MODE_RUN and o.cand_ints[a, 0] == ActionKind.DISCARD_POTION:
                        t.pot_pending.append(len(t.obs) - 1)
                    if (cfg.luck_cv and o.in_combat and not o.done and int(o.cand_ints[a, 0]) == ActionKind.END_TURN
                            and luck_room(cfg, t)):
                        # 运气校正：存下这一刻的局面，先走 luck_cv_k 个换了随机数的推演，齐了再恢复局面走真实的一步（reply 里接着发）
                        try:
                            env.envs[i].snapshot(cfg.luck_cv_slot)
                        except Exception as e:     # 存不下（引擎出错）：这一步不校正，照常走
                            print(f"采集进程 {rank} 引擎 {i} 运气校正存局面失败，这一步不校正：{str(e).splitlines()[0][:200]}", flush=True)
                            env.send_step(i, a)
                        else:
                            st = {"t": t, "step": len(t.obs) - 1, "a": a, "obs": [], "sent_real": False,
                                  "seeds": [int(x) for x in luck_rng.integers(1, 1 << 32, cfg.luck_cv_k)]}
                            cv[i] = st
                            env.send(i, luck_req(st, 0))
                    else:
                        env.send_step(i, a)
                ready_obs.clear()
            if time.time() - last_check > 1.0:
                last_check = time.time()
                if last_check - last_labels > 30:
                    last_labels = last_check
                    solver.refresh()
                if os.getppid() != parent:       # 学习进程已退出（含被强行终止），不再空转
                    break
                v, p, sh = load_weights(net, weights_path, version, wp, co=co)
                if v != version:
                    version = v; _cur_version[0] = v
                    net.to(device)
                    if co is not None:
                        co["mod"].to(device)
                if p >= 0:
                    progress, shaping = p, sh
    finally:
        env.close()
        if srch is not None:
            srch.close()
        if value_probe:
            value_probe.writer.close()


def save_run_replay(rec_dir: str, keep: int, t: Trajectory, win: bool, final, rng) -> None:
    """抽中的整局存成回放（格式同 evaluator.save_replay：第一行概要，之后每行一步），先写临时文件再改名；
    约每 200 局按修改时间删掉最旧的，目录里只留最近 keep 局（几个采集进程都会删，删重了就跳过）。存不下不影响采集。"""
    try:
        os.makedirs(rec_dir, exist_ok=True)
        seed = t.spec.get("seed")
        head = {"version": t.version, "seed": seed, "time": time.time(), "win": win, "floor": int(final.floor), "act": int(final.act),
                "combats": [], "steps": len(t.tape), "src": "train", "auto_gold": t.req.get("auto_gold", True) if t.req else True}
        name = f"v{t.version}_{seed}"
        tmp = os.path.join(rec_dir, f".{name}.tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as f:
            f.write(json.dumps(head, ensure_ascii=False) + "\n")
            for s in t.tape:
                f.write(s + "\n")
        os.replace(tmp, os.path.join(rec_dir, f"{name}.jsonl.gz"))
        if rng.random() < 1 / 200:
            fs = []
            for e in os.scandir(rec_dir):
                try:
                    if e.name.endswith(".jsonl.gz"):
                        fs.append((e.stat().st_mtime, e.path))
                except FileNotFoundError:
                    pass
            for _, path in sorted(fs)[:max(0, len(fs) - keep)]:
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass
    except Exception as e:
        print(f"整局回放保存失败：{e}", flush=True)


def _put(out_queue, item) -> None:
    """队列满时阻塞（反压），但每秒检查一次是否要停止。"""
    while True:
        try:
            out_queue.put(item, timeout=1.0)
            return
        except queue.Full:
            if (_STOP is not None and _STOP.is_set()) or os.getppid() == 1:
                return


def combat_aux_labels(obs: list, after) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """战斗短程辅助头的标签（决策记录 261）：逐步 [有没有标签, 本场战斗打赢没有, 从这一步到本场结束还要掉的血 / 这一步的最大生命]。
    一场战斗 = 连续的战斗中的步（头部 11），层数（7）或连打的第几场（23）变了另算一场；这场之后的下一步（没有就是 after：
    整局分段送出时为这一段之后的选路线帧，对局结束时为终局帧）已结束且没赢就是打输，结束血量记 0，否则取下一步的血量。
    整局每段都从选路线开始、在选路线之前送出，一场战斗不会跨段；下一步不明（卡住、缺血量）的那场不给标签。"""
    n = len(obs)
    m = np.zeros(n, bool); win = np.zeros(n, np.float32); hp = np.zeros(n, np.float32)

    def h(o, i: int) -> int:
        hd = getattr(o, "header", None)
        return int(hd[i]) if hd is not None and len(hd) > i else 0
    i = 0
    while i < n:
        if not h(obs[i], 11):
            i += 1; continue
        key = (h(obs[i], 7), h(obs[i], 23)); j = i
        while j + 1 < n and h(obs[j + 1], 11) and (h(obs[j + 1], 7), h(obs[j + 1], 23)) == key:
            j += 1
        nxt = obs[j + 1] if j + 1 < n else after
        if nxt is not None and not getattr(nxt, "stuck", False) and getattr(nxt, "hp", None) is not None:
            lost = bool(getattr(nxt, "done", False)) and not bool(getattr(nxt, "victory", False))
            end = 0 if lost else int(nxt.hp)
            for k in range(i, j + 1):
                m[k] = True; win[k] = 0.0 if lost else 1.0
                hp[k] = float(np.clip((int(obs[k].hp) - end) / max(int(obs[k].max_hp), 1), -1.0, 1.0))
        i = j + 1
    return m, win, hp


def _enemy_left(o) -> float:
    """活着的敌人剩余生命占最大生命的比例（同 tools/sim_pimc.enemy_left）；没有敌人记录时为 0。"""
    ti, tf = getattr(o, "tok_ints", None), getattr(o, "tok_feats", None)
    if ti is None or tf is None or not len(ti):
        return 0.0
    k = (ti[:, 0] == int(TokenKind.CREATURE)) & (tf[:, 5] == 0) & (tf[:, 3] > 0)
    hp, mx = float(tf[k, 0].sum()), float(tf[k, 1].sum())
    return hp / mx if mx > 0 else 0.0


_FH_LABELS = [0]        # 候选未来读数头（model.future_head）打开时为读数的步数 fh_h（采集进程开头设置），0 为不算标签


def future_labels(obs: list, values, rewards, boot: float, after, H: int) -> tuple[np.ndarray, np.ndarray]:
    """候选未来读数头的标签（愚者决策记录第 25 条）：逐步 [有没有标签]、[走这一步之后 H 步的读数]。只给战斗步（头部 11），
    一场战斗的划分同 combat_aux_labels。第 t 步的窗口为 t … e，e = min(t + H − 1, 这场最后一步)：
      0 窗口优势：rewards[t..e] 之和 + 末端价值 − values[t]（末端价值：窗口没到这场结束取 values[e+1]；这场在窗口内结束时取这场之后下一步的价值，
        对局已结束为 0、下一步不在这一段里取 boot——同战斗收尾用战后价值自举）；
      1 血量变化 / 这一步的最大生命；2 敌人剩余生命比例的变化（这场在窗口内打赢时末端为 0、输掉时取这场最后一步的）；3 其间打赢；4 其间输掉。
    这场之后的下一步不明（卡住、缺血量）时这场不给标签。"""
    n = len(obs)
    m = np.zeros(n, bool); y = np.zeros((n, 5), np.float32)
    if not n or H <= 0:
        return m, y

    def h(o, i: int) -> int:
        hd = getattr(o, "header", None)
        return int(hd[i]) if hd is not None and len(hd) > i else 0
    cum = np.concatenate([[0.0], np.cumsum(np.asarray(rewards, np.float64))])
    left = [None] * n
    i = 0
    while i < n:
        if not h(obs[i], 11):
            i += 1; continue
        key = (h(obs[i], 7), h(obs[i], 23)); j = i
        while j + 1 < n and h(obs[j + 1], 11) and (h(obs[j + 1], 7), h(obs[j + 1], 23)) == key:
            j += 1
        nxt = obs[j + 1] if j + 1 < n else after
        if nxt is not None and not getattr(nxt, "stuck", False) and getattr(nxt, "hp", None) is not None:
            lost = bool(getattr(nxt, "done", False)) and not bool(getattr(nxt, "victory", False))
            v_end = 0.0 if bool(getattr(nxt, "done", False)) else (float(values[j + 1]) if j + 1 < n else float(boot))
            for k in range(i, j + 1):
                if left[k] is None:
                    left[k] = _enemy_left(obs[k])
            for t in range(i, j + 1):
                e = min(t + H - 1, j)
                mx = max(int(obs[t].max_hp), 1)
                if e < j:
                    o2 = obs[e + 1]
                    y[t] = (cum[e + 1] - cum[t] + float(values[e + 1]) - float(values[t]), (int(o2.hp) - int(obs[t].hp)) / mx,
                            left[e + 1] - left[t], 0.0, 0.0)
                else:
                    hp_end = 0 if lost else int(nxt.hp)
                    l_end = left[j] if lost else 0.0
                    y[t] = (cum[j + 1] - cum[t] + v_end - float(values[t]), (hp_end - int(obs[t].hp)) / mx, l_end - left[t],
                            0.0 if lost else 1.0, 1.0 if lost else 0.0)
                m[t] = True
        i = j + 1
    return m, y


_CO_LABELS = [False]    # 学习端训练跳过稳赢战斗的结果头（config.co_heads）时，逐步数据另带结果头的标签（采集进程开头设置）


def co_labels(obs: list, after) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """跳过稳赢战斗的结果头（model.CombatOutcomeHeads）的标签：每场赢下的整局战斗，在开战第一帧上记
    [有没有标签, 战后血量 / 开战时最大生命（截到 0–1）, 开战时的药水词表 ID（按帧里的先后，补 0 到 CO_POT_SLOTS）, 每瓶打完后用掉没有（1 / 0，不明为 −1）]。
    一场战斗的划分同 combat_aux_labels；战后那一帧是这场之后的下一步（没有就是 after），输了、卡住、缺血量的不给标签，
    整局结束的终局帧不看药水（药水标签记为不明）。"""
    n = len(obs)
    m = np.zeros(n, bool); hp = np.zeros(n, np.float32)
    vid = np.zeros((n, CO_POT_SLOTS), np.int64); y = np.full((n, CO_POT_SLOTS), -1.0, np.float32)

    def h(o, i: int) -> int:
        hd = getattr(o, "header", None)
        return int(hd[i]) if hd is not None and len(hd) > i else 0
    i = 0
    while i < n:
        if not h(obs[i], 11):
            i += 1; continue
        key = (h(obs[i], 7), h(obs[i], 23)); j = i
        while j + 1 < n and h(obs[j + 1], 11) and (h(obs[j + 1], 7), h(obs[j + 1], 23)) == key:
            j += 1
        nxt = obs[j + 1] if j + 1 < n else after
        if (nxt is not None and not getattr(nxt, "stuck", False) and getattr(nxt, "hp", None) is not None
                and not (bool(getattr(nxt, "done", False)) and not bool(getattr(nxt, "victory", False)))):
            m[i] = True
            hp[i] = float(np.clip(int(nxt.hp) / max(int(obs[i].max_hp), 1), 0.0, 1.0))
            vid[i] = held_potion_ids(obs[i])
            if not bool(getattr(nxt, "done", False)) and getattr(nxt, "tok_ints", None) is not None:
                end = Counter(int(x) for x in held_potion_ids(nxt) if x > 0)
                for s_, v in enumerate(vid[i]):
                    if v <= 0:
                        continue
                    kept = end.get(int(v), 0) > 0
                    if kept:
                        end[int(v)] -= 1
                    y[i, s_] = 0.0 if kept else 1.0
        i = j + 1
    return m, hp, vid, y


def _steps(t: Trajectory, boot: float, after=None) -> dict:
    """还没送出的那一段（t.seg0 起）的逐步数据；boot 为这一段之后下一步的价值（局末为 0），学习端算优势时自举用。
    after 为这一段之后的那一帧（分段时的选路线帧、局末的终局帧），算战斗辅助头的标签用。"""
    s = t.seg0
    aux_m, aux_win, aux_hp = combat_aux_labels(t.obs[s:], after)
    co = {}
    if _FH_LABELS[0]:       # 候选未来读数头的标签（愚者决策记录第 25 条）
        fm, fy = future_labels(t.obs[s:], t.values[s:], t.rewards[s:], boot, after, _FH_LABELS[0])
        co = {"fh_m": fm, "fh_y": fy}
    if _CO_LABELS[0] and t.mode == MODE_RUN:     # 跳过稳赢战斗的结果头的标签（只用整局里的战斗）
        cm, ch, cv, cy = co_labels(t.obs[s:], after)
        co = {**co, "co_m": cm, "co_hp": ch, "co_vid": cv, "co_y": cy}
    return {**co, "aux_m": aux_m, "aux_win": aux_win, "aux_hp": aux_hp, "mode": t.mode, "macro": np.array([t.mode == MODE_RUN and is_macro(o) for o in t.obs[s:]], bool),
            "phi": np.array(t.phi[s:], np.float32), "ragged": Ragged.from_obs(t.obs[s:], np.full(len(t.obs) - s, t.mode)),
            "actions": np.array(t.actions[s:], np.int64), "logp": np.array(t.logp[s:], np.float32),
            "logp_mu": np.array(t.logp_mu[s:], np.float32), "forced": np.array(t.forced[s:], bool), "anc": np.array(t.anc[s:], bool),
            "det": np.array(t.det[s:], bool),
            "values": np.array(t.values[s:], np.float32), "rewards": np.array(t.rewards[s:], np.float32), "vers": np.array(t.vers[s:], np.int32),
            **({"nv": nv} if np.isfinite(nv := np.array(t.nv[s:], np.float32)).any() else {}),
            "len": len(t.obs) - s, "boot": float(boot), "v_send": _cur_version[0], "t_send": time.time(), "gid": t.gid,
            "sig": np.array(t.sig[s:], np.int64), "grp": [t.spec["seed"], int(t.spec["grp"])] if t.spec.get("grp") else None,
            **({"value_probe": {"episode_id": t.probe["episode_id"], "root_id": t.probe["root_id"],
                               "split": t.probe["split"], "start": s, "segment_id": len(t.probe["segments"])}} if t.probe else {})}


def send_segment(t: Trajectory, boot: float, out_queue, rank: int, after=None) -> None:
    """整局中途送出一段（run_segment_steps）：只有逐步数据，不计入对局统计；结局和整局统计随最后一段送。after 为这一段之后的选路线帧。"""
    _put(out_queue, ("traj", rank, {**_steps(t, boot, after), "segment": True, "version": t.version}))
    if t.probe:
        t.probe["segments"].append({"start": t.seg0, "end": len(t.obs), "boot": float(boot), "terminal": False})
    t.seg0 = len(t.obs)
    t.seg_fill = 0


def _finish(cfg, t: Trajectory, final, out_queue, rank: int) -> bool:
    """对局结束：补上终局奖励，把轨迹打包送出；返回是否打赢。"""
    win = bool(final.victory) and not final.stuck
    if not t.obs:
        return win
    if t.mode == MODE_RUN and cfg.skip_easy:     # 跳过稳赢战斗：按局的类别（完全真打 / 允许跳过）记局数与通关，看板对照
        kind = "real" if t.co_real else "skip"
        t.choices[f"SKIPRUN_N:{kind}"] += 1; t.choices[f"SKIPRUN_WIN:{kind}"] += int(win)
    if t.mode == MODE_RUN:
        if t.wp_last is not None:       # 胜率势能：对局结束时 Φ = 0
            reward_event(t, "winprob_potential_terminal", -t.wp_last)
            t.rewards[-1] -= t.wp_last; t.wp_shaped -= t.wp_last
        if t.pg_last is not None:       # 进度势能：对局结束时（输赢都算）Φ = 0
            reward_event(t, "progress_potential_terminal", -t.pg_last)
            t.rewards[-1] -= t.pg_last; t.pg_shaped -= t.pg_last
        if t.bv_last is not None:       # 构筑价值势能：对局结束时（输赢都算）Φ = 0
            reward_event(t, "buildvalue_potential_terminal", -t.bv_last)
            t.rewards[-1] -= t.bv_last; t.bv_shaped -= t.bv_last
        if win:
            # 打赢第三幕第二个 Boss 就是通关：通关奖励之外另加一笔里程碑（2026-10-26 起，用户决定单列；默认 0）
            t.rewards[-1] += cfg.victory_reward + cfg.act3_boss2_reward
            reward_event(t, "victory", cfg.victory_reward)
            reward_event(t, "boss3_second", cfg.act3_boss2_reward)
            # 通关余血奖励（config.victory_hp_bonus，铁甲战士决策记录第 24 条）：通关时余血大于门槛才给。
            # 终局帧的血量恒为 0（胜利后剧情“击杀”玩家，见引擎 GameShim 记胜负处的注释；评测逐局 hp_delta 全是 −0.8），
            # 所以取终局前最后一个血量大于 0 的决策帧（通关局是最后那场 Boss 战里打出致命一击之前的那一帧；之后的燃烧之血回血不算）
            if cfg.victory_hp_bonus:
                hp_end = next((int(o.hp) for o in reversed(t.obs) if int(o.hp) > 0), 0)
                ok = hp_end > cfg.victory_hp_margin
                t.choices["VICTORY_HP:ok" if ok else "VICTORY_HP:low"] += 1
                t.choices["VICTORY_HP:sum"] += hp_end
                if ok:
                    t.rewards[-1] += cfg.victory_hp_bonus
                    reward_event(t, "victory_hp", cfg.victory_hp_bonus)
            back = (1.0 - milestone_scale(cfg, t.version)) * t.ms_full     # 撤掉的里程碑在通关时补回（决策记录 271），通关局的回报不变
            if back:
                t.rewards[-1] += back
                reward_event(t, "milestone_back", back)
    elif t.spec.get("next_encounter"):
        # 连打练习（决策记录 187）：只按两场的总结果结算。两场都赢 = 这一幕打完（即通关），同 Boss 打赢；
        # 输了按进度给部分分：（打过的场数 + 输的那场打掉的血量比例）/ 2 × boss_dmg_coef，死在第二场总比死在第一场高。
        # 中途不扣血、不计药水：第一场省下的血和药水值多少，由第二场的胜负直接定价
        if t.stage2 is None and int(getattr(final, "header", [0] * 24)[23]) >= 1:
            t.stage2 = len(t.obs)       # 第二场一开打就死了（没有决策点）：算打到了第二场，部分分为 0
        if win:
            t.rewards[-1] += cfg.win_reward + cfg.boss_win_bonus
        else:
            done = 0.0 if t.stage2 is None else 1.0
            part = boss_score(t, False, 1.0, first=t.stage2) if t.stage2 is not None and t.stage2 < len(t.obs) else \
                (0.0 if t.stage2 is not None else boss_score(t, False, 1.0))
            t.rewards[-1] += cfg.lose_reward + cfg.boss_dmg_coef * (done + part) / 2
    else:
        if win and t.spec.get("slot") == "boss":
            t.rewards[-1] += cfg.boss_win_bonus
        # Boss 战用药水不扣分（2026-10-25 起，用户决定：药水一般留到 Boss 战用）；Boss 名额、真实快照与按进度抽到的 Boss 遭遇都算
        is_boss = t.spec.get("slot") == "boss" or t.spec.get("room") == "Boss" or str(t.spec.get("encounter", "")).lower() == "boss"
        is_elite = t.spec.get("room") == "Elite" or str(t.spec.get("encounter", "")).lower() == "elite"
        # 打输的 Boss 战按打掉的敌人血量比例给部分分（见 boss_score），让输局之间也分得出好坏
        if is_boss and not win and cfg.boss_dmg_coef > 0:
            t.rewards[-1] += cfg.boss_dmg_coef * boss_score(t, False, 1.0)
        t.rewards[-1] += (cfg.win_reward if win else cfg.lose_reward) + cfg.hp_coef * (final.hp - t.hp_start) / t.max_hp \
            - (0.0 if is_boss or (cfg.elite_potion_free and is_elite) else cfg.potion_coef * t.potions)
    phi_label = None
    deck_n = None       # 真实快照的 Boss 练习：快照那一刻的牌组张数（含诅咒）
    if t.label_view is not None:
        v = t.label_view
        deck_n = int(((v.tok_ints[:, 0] == TokenKind.CARD) & (v.tok_ints[:, 2] == Zone.DECK)).sum())
    if t.label_view is not None and not final.stuck:
        phi_label = {"view": Ragged.from_obs([t.label_view], np.full(1, MODE_RUN)), "act": int(t.spec["act"]),
                     "y": (boss_score(t, win, cfg.phi_dmg_weight) if not t.spec.get("next_encounter") or win       # 连打：按两场的进度
                           else cfg.phi_dmg_weight * ((t.stage2 is not None)
                                                      + (boss_score(t, False, 1.0, first=t.stage2 or 0) if (t.stage2 or 0) < len(t.obs) else 0.0)) / 2),
                     "probe": bool(t.spec.get("probe"))}
    _put(out_queue, ("traj", rank, {
        **_steps(t, 0.0, final),
        "shaped": t.shaped, "phi_label": phi_label, "hp_shaped": t.hp_shaped, "choices": dict(t.choices),
        "pot_wasted": t.pot_wasted, "deck_shaped": t.deck_shaped,
        "dmg_shaped": t.dmg_shaped, "pot_cost": t.pot_cost, "skip_cost": t.skip_cost, "relic_shaped": t.relic_shaped,
        "elite_fights": t.elite_fights, "elite_pots": t.elite_pots, "boss_pots": t.boss_pots, "ancient": offer_counts(t.ancient), "cards": offer_counts(t.cards),
        "reward": float(sum(t.rewards)),
        # len 为这最后一段的步数（学习用），run_len 为整局步数（统计用）
        "version": t.version, "win": win, "stuck": bool(final.stuck), "run_len": len(t.obs), "seconds": time.time() - t.t0,
        **({"probe_terminated": bool(getattr(final, "done", final.victory)) and not final.stuck} if t.probe else {}),
        "hp_delta": (final.hp - t.hp_start) / t.max_hp,
        # 战斗课程
        "act": t.spec.get("act", final.act), "encounter": t.spec.get("encounter", "run"), "slot": t.spec.get("slot"),
        "room": t.spec.get("room"), "real": "player_snapshot" in t.spec, "probe": bool(t.spec.get("probe")), "deck_n": deck_n,
        "cert": t.spec.get("cert"), "second": bool(t.spec.get("second")),
        "chain": bool(t.spec.get("next_encounter")), "chain_reached": t.stage2 is not None,
        # 胜率势能（决策记录 189）：这一局的塑形合计；整局（没卡住）的抽样步与标签（是否通关）、权重（1 / 这一层留下的步数）
        "wp_shaped": t.wp_shaped, "pg_shaped": t.pg_shaped, "bv_shaped": t.bv_shaped,
        "wp": [(r[0], r[1], float(win), 1.0 / len(v[1])) for v in t.wp_rows.values() for r in v[1]] if t.mode == MODE_RUN and not final.stuck else None,
        # 整局
        "floor": int(final.floor), "final_act": int(final.act), "start_act": int(t.spec.get("start_act", 0)),
        # 死因：输掉的局死在哪场战斗（键同评测的 deaths；死在战斗之外记“非战斗”，打赢或卡住为 None）；打过的 Boss 及胜负
        "death": None if win or final.stuck or t.mode != MODE_RUN else (t.cb_key if t.cb_key and t.cb_floor == final.floor else "非战斗"),
        "bosses": t.bosses,
    }))
    return win
