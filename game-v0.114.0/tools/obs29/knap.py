"""格式 29 包 2（背包，敌人记录 18“本回合最多还能打多少”、19“打不打得死”）的核对模块。

研究报告/2026-11-20_全角色观测缺口调研/R3-格式29规格.md 5.2。给 tools/knap_truth.py 与 tools/obs_v29_check.py（主会话的框架）共用：

  conditions(fr, foe)   这一帧、这个敌人涉及包 2 的哪些条件（规格 5.2 表里的每一行一类；都没有为空列表）
  FIXES                 obs_v29_check 的登记：每条 {"id", "where", "when"}，where 为允许不同的位置，when(fr) 为这一条可能起作用的谓词
  truth(...)            用局面复制（snapshot / restore）对小手牌暴力搜索，得出本回合实际能对每个敌人打到的伤害（真值）
  Frame                 一帧观测的便捷读法（手牌、敌人、玩家能力、遗物、能量、星）

真值的口径（规格 5.2 第一段）：
  - 只在战斗出牌决策、手牌 ≤ 7、能量 ≤ 5 时算；从这一帧出发深度优先枚举出牌序列（深度 ≤ 6），只打这一帧手里的牌
    （按（词表 ID，升级次数）的多重集合逐张扣，抽到的、生成的新牌不打），不用药水、不结束回合；中途的选牌决策每个选项都试。
  - 每个节点读敌人的生命 + 格挡（生物 0、2），比这一帧少了多少；敌人 f 的真值 = 各节点的最大值，截到这一帧的生命 + 格挡。
    敌人死了（记录消失）按全额；战斗结束（胜）全体按全额；玩家死了的节点不算。
  - 与规格的一处出入：多个敌人时，**有目标的牌只对 f 出**（每个敌人单独搜一遍）。打别的敌人不会增加对 f 的伤害
    （例外是靠先打死别的敌人让随机目标集中到 f，背包本来也不算），这样搜索量从（目标数）^张数 降到 张数的子集数。
  - 有随机目标（打随机敌人的牌、势不可当）且可攻击的敌人 ≥ 2 时，同一组牌用 6 个 reseed 各搜一遍，
    按“打出的牌的多重集合”取各 seed 的平均，再取最大（同一集合里不同顺序按各 seed 自己的最好顺序，略偏乐观）。
  - 同一局面（帧内容相同，不看帧头）只展开一次（换顺序打出同样几张牌、结果相同时不重复搜）。
  - “生命 + 格挡”的减少量是真值的口径：敌人在出牌途中加格挡（胆小）会抵掉一部分，背包按同一口径算。

比较（compare_row、summarize）：min(引擎 18, 17) 对 min(真值, 17)（18 只按硬化外壳封顶，打得死时常比 17 大），19 对“真值 ≥ 17”。
按条件分组报完全一致（差 ≤ 0.5）、差 ≤ 2、平均偏差（引擎 − 真值）、平均绝对误差、19 一致率，格式 28、29 各一份。
规格明确不做的“先打一张牌带来的增益”（上易伤、ONE_TWO_PUNCH 这张牌、降费、先打能力牌加大附带伤害）与 G48（打下箭再射）会让真值高于背包，
归包 4 的 PACTS_END（G13）会让背包高于真值，这样的帧标 oos，表里另给去掉它们的一份。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------- 卡牌表（与 Obs/Knap29.cs 的名单一致）

# 打随机敌人的牌（catalog 的 target 为 RandomEnemy）
RANDOM_TARGET = {"SWORD_BOOMERANG", "RIP_AND_TEAR", "RICOCHET", "FLAK_CANNON", "SWEEPING_GAZE", "MEMENTO_MORI", "STARDUST", "BEAT_DOWN",
                 "VOLLEY", "BOUNCING_FLASK"}
# X 费攻击（能量）与星 X 费（G18）
X_ATTACKS = {"WHIRLWIND", "VOLLEY", "SKEWER", "ERADICATE", "REPEAT_SHOT", "HEAVENLY_DRILL"}
STAR_X = {"STARDUST"}
# 无条件当场加能量的牌（Gains26.cs 的 EnergyCards）
ENERGY_CARDS = {"ADRENALINE", "ALIGNMENT", "BIG_BANG", "BLOODLETTING", "BORROWED_TIME", "BRIGHTEST_FLAME", "ENCORE", "FUEL", "JOY", "LUMINESCE",
                "NEUROSURGE", "OFFERING", "PRODUCTION", "SUPERCRITICAL", "TACTICIAN", "THE_EMPEROR", "THE_SUN", "TURBO", "WISP"}
# 储君当场得星的牌（OnPlay 顶层 PlayerCmd.GainStars，无条件）
STAR_CARDS = {"GATHER_LIGHT", "GLOW", "VENERATE", "SHINING_STRIKE", "SOLAR_STRIKE", "BIG_BANG", "ROYAL_GAMBLE", "HIDDEN_CACHE"}
# 本回合加力量的牌（G76、G69）
STR_CARDS = {"INFLAME", "FEEDING_FRENZY", "SETUP_STRIKE", "BRAND", "BULK_UP", "PROWESS", "STRENGTH", "FIGHT_ME"}
# 出牌自伤的牌（Characters/Ironclad.cs 的 SelfDamageCards；红骷髅、狱火用）
SELF_DAMAGE = {"BLOODLETTING", "OFFERING", "HEMOKINESIS", "BLOOD_WALL", "BRAND", "BREAKTHROUGH", "DEMONIC_SHIELD"}
SHIV = "SHIV"
# 带 Damage 数值的状态与诅咒牌（回合末打自己）：能打出时原来的背包把它算成对敌人的伤害
SELF_DMG_STATUS = {"BURN", "DECAY", "INFECTION", "TOXIC", "WITHER"}
# 规格 5.2 G76 明确不做的“先打一张牌带来的增益”：给敌人上易伤的牌（游戏源码里 Apply<VulnerablePower> 的牌）、ONE_TWO_PUNCH 这张牌本身（打出后下一张攻击打两次）、
# ENLIGHTENMENT（降费）。手里有这些牌（且敌人还没有易伤）时真值可能高于背包，这样的帧另外标出（oos），核对表里另给“去掉这些帧”的一份
VULN_CARDS = {"ASSASSINATE", "BASH", "BEAM_CELL", "BREAK", "COMET", "DOMINATE", "EXPOSE", "FALLING_STAR", "FEAR", "GAMMA_BLAST", "HIGH_FIVE", "HOBBLING_SHOT",
              "KNOW_THY_PLACE", "METEOR_SHOWER", "MOLTEN_FIST", "NAIL", "PUTREFY", "SHOCKWAVE", "SQUASH", "TAUNT", "THE_EMPRESS", "THUNDERCLAP", "TREMBLE",
              "UPPERCUT", "UPROOT", "WAH_TOUCH"}
# 先打一张能力牌让本回合的附带伤害或加成变大（再打一张幻影之刃、暴力和弦、爆炸箭矢，先打狱火、势不可当、致死性），同属“先打增益”，不做
OOS_PRE = {"ONE_TWO_PUNCH", "ENLIGHTENMENT", "PHANTOM_BLADES", "VIOLENT_CHORD", "EXPLOSIVE_ARROWS", "INFERNO", "JUGGERNAUT", "LETHALITY"}
# 归别的包的背包误差：PACTS_END 消耗堆不足时没有伤害（G13，包 4 改 PreviewCore，背包随之不计；本包合并前两边都按卡面算）
OTHER_PKG = {"PACTS_END"}


def out_of_scope(fr: Frame, foe: Foe) -> bool:
    """这一帧对这个敌人有没有规格不做的先打增益（上易伤、ONE_TWO_PUNCH 牌、降费），或归别的包的误差（PACTS_END，G13）。"""
    hn = set(fr.hand_names())
    # G48（不做）：用攻击把扎在敌人身上的箭打下来再射
    g48 = bool(hn & SHOOT_NAMES) and any("IMPALED_POWER" in x.powers for x in fr.foes)
    return g48 or bool(hn & (OOS_PRE | OTHER_PKG)) or (bool(hn & VULN_CARDS) and "VULNERABLE_POWER" not in foe.powers)

COND_ORDER = ["G16-凡庸", "G16-ENTHRALLED", "G16-鸣响", "G17-ONE_TWO_PUNCH", "G17-幻影之刃", "G17-致死性", "G17-虚空形态",
              "G18-X费", "G19-恶魔之焰", "G19-迟缓", "G19-软弱", "G19-胆小", "G20-随机目标", "G43-愚者", "G43-铁甲",
              "G64-撕咬", "G76-得能量得星", "G76-加力量", "附-状态牌自伤"]
# 有随机效果的条件（真值按 6 个 reseed 平均；规格的目标是平均绝对误差下降，不要求一致率）
RANDOM_CONDS = {"G20-随机目标"}


def _name(voc: list[str], vid: int) -> str:
    k = voc[vid] if 0 <= vid < len(voc) else "?"
    return k.split(".", 1)[1] if "." in k else k


@dataclass
class Foe:
    tok: int
    name: str
    hp: float
    max_hp: float
    block: float
    hittable: bool
    f16: float
    f17: float
    f18: float
    f19: float
    powers: dict = field(default_factory=dict)      # 名字 → (层数, 本回合已触发)


@dataclass
class Card:
    tok: int
    name: str
    up: int
    cost: float
    xcost: int          # 0 否，1 能量 X，2 星 X
    playable: bool
    affl: str | None = None


class Frame:
    """一帧观测（sts2env.protocol.Obs）的便捷读法。"""

    def __init__(self, o, voc: list[str]):
        self.o, self.voc = o, voc
        ti, tf = o.tok_ints, o.tok_feats
        self.player_tok = -1
        self.foes: list[Foe] = []
        self.hand: list[Card] = []
        self.ppowers: dict[str, tuple[float, float, list]] = {}
        self.relics: set[str] = set()
        self.energy = self.stars = 0.0
        for i in range(ti.shape[0]):
            kind, vid, zone, owner = int(ti[i, 0]), int(ti[i, 1]), int(ti[i, 2]), int(ti[i, 3])
            if kind == 1:
                self.energy, self.stars = float(tf[i, 6]), float(tf[i, 8])
            elif kind == 2:
                if tf[i, 5] == 1:
                    self.player_tok = i
                elif zone == 2:
                    self.foes.append(Foe(i, _name(voc, vid), float(tf[i, 0]), float(tf[i, 1]), float(tf[i, 2]), bool(tf[i, 4]),
                                         *map(float, tf[i, 16:20])))
            elif kind == 4 and zone == 3:
                self.hand.append(Card(i, _name(voc, vid), int(tf[i, 3]), float(tf[i, 0]), int(tf[i, 1]), bool(tf[i, 4])))
            elif kind == 5:
                self.relics.add(_name(voc, vid))
        byTok = {f.tok: f for f in self.foes}
        handTok = {c.tok: c for c in self.hand}
        for i in range(ti.shape[0]):
            if int(ti[i, 0]) != 3:
                continue
            owner, nm = int(ti[i, 3]), _name(voc, int(ti[i, 1]))
            if owner == self.player_tok:
                self.ppowers[nm] = (float(tf[i, 0]), float(tf[i, 6]), list(map(float, tf[i, 8:12])))
            elif owner in byTok:
                byTok[owner].powers[nm] = (float(tf[i, 0]), float(tf[i, 6]))
            elif owner in handTok and voc[int(ti[i, 1])].startswith("AFFLICTION."):
                handTok[owner].affl = nm

    def hand_names(self) -> list[str]:
        return [c.name for c in self.hand]

    def n_hittable(self) -> int:
        return sum(1 for f in self.foes if f.hittable)


def conditions(fr: Frame, foe: Foe) -> list[str]:
    """这一帧、这个敌人涉及包 2 的哪些条件（规格 5.2 的表；判断只看帧内容，与 Knap29.cs 的检测同口径但略宽）。"""
    out = []
    hn = fr.hand_names()
    pp = fr.ppowers
    attacks = [c for c in fr.hand if c.name in ATTACK_NAMES]
    if "NORMALITY" in hn:
        out.append("G16-凡庸")
    if "ENTHRALLED" in hn:
        out.append("G16-ENTHRALLED")
    if "RINGING_POWER" in pp:
        out.append("G16-鸣响")
    if "ONE_TWO_PUNCH_POWER" in pp and len(attacks) >= 2:
        out.append("G17-ONE_TWO_PUNCH")
    if "PHANTOM_BLADES_POWER" in pp and hn.count(SHIV) >= 2:
        out.append("G17-幻影之刃")
    if "LETHALITY_POWER" in pp and len(attacks) >= 2:
        out.append("G17-致死性")
    if "VOID_FORM_POWER" in pp and pp["VOID_FORM_POWER"][2][3] > 0:
        out.append("G17-虚空形态")
    if any(c.name in X_ATTACKS or c.name in STAR_X for c in fr.hand):
        out.append("G18-X费")
    if "FIEND_FIRE" in hn and len(hn) >= 2:
        out.append("G19-恶魔之焰")
    if "SLOW_POWER" in foe.powers:
        out.append("G19-迟缓")
    if "TENDER_POWER" in pp:
        out.append("G19-软弱")
    if "SKITTISH_POWER" in foe.powers and foe.powers["SKITTISH_POWER"][1] == 0:
        out.append("G19-胆小")
    if any(c.name in RANDOM_TARGET for c in fr.hand) and fr.n_hittable() >= 2:
        out.append("G20-随机目标")
    if "VIOLENT_CHORD_POWER" in pp or "EXPLOSIVE_ARROWS_POWER" in pp:
        out.append("G43-愚者")
    if "CHARONS_ASHES" in fr.relics or "INFERNO_POWER" in pp or "JUGGERNAUT_POWER" in pp:
        out.append("G43-铁甲")
    if hn.count("MAUL") >= 2:
        out.append("G64-撕咬")
    if any(c in ENERGY_CARDS or c in STAR_CARDS for c in hn):
        out.append("G76-得能量得星")
    if any(c in STR_CARDS for c in hn) or ("RED_SKULL" in fr.relics and any(c in SELF_DAMAGE for c in hn)) \
            or any(c.name == "MAD_SCIENCE" for c in fr.hand):
        out.append("G76-加力量")
    # 附带修正：能打出的状态牌（毒素）的 Damage 是打自己的，原来的背包算成对敌人的伤害
    if any(c.name in SELF_DMG_STATUS and c.playable for c in fr.hand):
        out.append("附-状态牌自伤")
    return out


def is_random(fr: Frame) -> bool:
    """这一帧的出牌伤害有没有随机目标（真值要按多个 reseed 平均）：可攻击的敌人 ≥ 2，且手里有打随机敌人的牌或玩家有势不可当。"""
    return fr.n_hittable() >= 2 and (any(c.name in RANDOM_TARGET for c in fr.hand) or "JUGGERNAUT_POWER" in fr.ppowers)


# 攻击牌的名字（catalog 的 type 为 Attack），conditions 判断“手里攻击 ≥ 2”用
def _attack_names() -> set[str]:
    try:
        cat = json.loads((REPO / "engine" / "catalog" / "v0.113.1" / "catalog.json").read_text(encoding="utf-8"))
        return {x["key"][5:] for x in cat["entities"] if x["category"] == "CARD" and x.get("type") == "Attack"}
    except Exception:  # noqa: BLE001
        return set()


ATTACK_NAMES = _attack_names()


def _shoot_names() -> set[str]:
    try:
        cat = json.loads((REPO / "engine" / "catalog" / "v0.113.1" / "catalog.json").read_text(encoding="utf-8"))
        return {x["key"][5:] for x in cat["entities"] if x["category"] == "CARD" and "Shoot" in (x.get("tags") or [])}
    except Exception:  # noqa: BLE001
        return set()


SHOOT_NAMES = _shoot_names()


# ---------------------------------------------------------------- obs_v29_check 的登记（规格 3.3：敌人 18–19 由 G13、G16–G20、G43、G64、G76 改）

def _any_cond(fr_obs, voc) -> bool:
    fr = Frame(fr_obs, voc)
    return any(conditions(fr, f) for f in fr.foes)


FIXES = [
    {"id": "包2-背包", "items": ["G16", "G17", "G18", "G19", "G20", "G43", "G64", "G66②", "G69", "G76"],
     "where": [("rec", 2, 18), ("rec", 2, 19)],
     # 只有出现 conditions 里的条件时敌人 18、19 才可能与格式 28 不同（Knap29 一个条件都没检测到时返回 −1，沿用原来的背包）。
     # when(obs, vocab)：obs 为 sts2env.protocol.Obs（格式 29 的帧），vocab 为词表
     "when": _any_cond},
]


# ---------------------------------------------------------------- 真值：局面复制上的暴力搜索

def _state_key(o) -> bytes:
    h = hashlib.blake2b(digest_size=16)
    for a in (o.tok_ints, o.tok_feats, o.cand_ints, o.cand_feats):
        h.update(np.ascontiguousarray(a).tobytes())
    return h.digest()


def _foe_hb(fr: Frame, ref: Foe, idx: int, refs: list[Foe]) -> float | None:
    """在节点帧里找到开局的第 idx 个敌人 ref，返回它现在的生命 + 格挡；死了（记录没了）为 0；对不上（同名同最大生命的有死有活、分不清）为 None。"""
    same0 = [k for k, r in enumerate(refs) if (r.name, r.max_hp) == (ref.name, ref.max_hp)]
    now = [f for f in fr.foes if (f.name, f.max_hp) == (ref.name, ref.max_hp)]
    if len(now) == len(same0):
        f = now[same0.index(idx)]
        return f.hp + f.block
    if not now:
        return 0.0
    return None


class Searcher:
    """一个引擎进程上的暴力搜索（槽位 0 存开局，1 起按搜索深度用）。"""

    def __init__(self, proc, voc: list[str], parse, max_depth: int = 6, budget: int = 400, seeds: int = 6):
        self.proc, self.voc, self.parse = proc, voc, parse
        self.max_depth, self.budget, self.seeds = max_depth, budget, seeds
        self.nodes = 0

    def _req(self, r: dict):
        self.proc.send(r)
        return self.parse(self.proc.recv())

    def truth(self, root_obs, rng) -> dict:
        """返回 {开局敌人下标: 真值 或 None（超出搜索预算 / 出错）}，以及是否按随机处理。调用前后引擎都停在开局（槽位 0）。"""
        fr0 = Frame(root_obs, self.voc)
        self.proc.send({"op": "snapshot", "slot": 0}); self.proc.recv()
        rnd = is_random(fr0)
        hand0: dict[tuple[str, int], int] = {}
        for c in fr0.hand:
            hand0[(c.name, c.up)] = hand0.get((c.name, c.up), 0) + 1
        res: dict[int, float | None] = {}
        seeds = [int(rng.integers(1, 1 << 31)) for _ in range(self.seeds)] if rnd else [None]
        for k, f in enumerate(fr0.foes):
            if not f.hittable:
                continue
            full = f.hp + f.block
            per_seed: list[dict] = []
            ok = True
            for s in seeds:
                vals: dict[tuple, float] = {}
                try:
                    o = self._req({"op": "restore", "slot": 0, "snapshot": 1, **({"reseed": s, "keep_top": True} if s is not None else {})})
                    self.nodes = 0
                    self._dfs(o, 1, 0, dict(hand0), (), fr0, k, f, full, vals, set())
                except _Budget:
                    ok = False
                except Exception:  # noqa: BLE001
                    ok = False
                if not ok:
                    break
                per_seed.append(vals)
            if not ok:
                res[k] = None
                continue
            if s is None or len(per_seed) == 1:
                res[k] = max(per_seed[0].values()) if per_seed[0] else 0.0
            else:
                keys = set().union(*per_seed)
                best = 0.0
                for key in keys:
                    xs = [v[key] for v in per_seed if key in v]
                    if len(xs) * 2 >= len(per_seed):
                        best = max(best, sum(xs) / len(xs))
                res[k] = best
        self._req({"op": "restore", "slot": 0})
        return {"truth": res, "random": rnd}

    def _dfs(self, o, slot: int, depth: int, left: dict, key: tuple, fr0: Frame, k: int, f: Foe, full: float, vals: dict, seen: set):
        self.nodes += 1
        if self.nodes > self.budget:
            raise _Budget()
        # 这个节点对 f 打了多少
        dmg = None
        fr = Frame(o, self.voc) if (o.in_combat and not o.done) else None
        if o.done:
            if o.victory:
                dmg = full
        elif fr is not None:
            hb = _foe_hb(fr, f, k, fr0.foes)
            if hb is not None:
                dmg = min(full, max(0.0, full - hb))
        if dmg is not None:
            sk = tuple(sorted(key))
            vals[sk] = max(vals.get(sk, 0.0), dmg)
        if o.done or not o.in_combat or depth >= self.max_depth:
            return
        dec = int(o.header[1])
        n = o.num_actions
        ci = o.cand_ints
        kids = []
        for a in range(n):
            kind = int(ci[a, 0])
            if dec == 1 and kind == 2:          # 出牌：只打开局手里的牌，有目标的只打 f（或自己、友方）
                ta, tb = int(ci[a, 1]), int(ci[a, 2])
                card = (_name(self.voc, int(o.tok_ints[ta, 1])), int(o.tok_feats[ta, 3]))
                if left.get(card, 0) <= 0:
                    continue
                if tb >= 0 and int(o.tok_ints[tb, 0]) == 2 and int(o.tok_ints[tb, 2]) == 2 and fr is not None:
                    tf = next((x for x in fr.foes if x.tok == tb), None)
                    same = [x for x in fr.foes if (x.name, x.max_hp) == (f.name, f.max_hp)]
                    same0 = [x for x in fr0.foes if (x.name, x.max_hp) == (f.name, f.max_hp)]
                    if tf is None or (tf.name, tf.max_hp) != (f.name, f.max_hp):
                        continue
                    if len(same) == len(same0) and same.index(tf) != same0.index(f):
                        continue
                kids.append((a, card, 1))
            elif dec == 2 and kind in (5, 6):   # 中途的选牌：每个选项都试（不算深度）
                kids.append((a, None, 0))
        for a, card, dd in kids:
            c = self._req({"op": "restore", "slot": slot, "first": a, "snapshot": slot + 1})
            sk = _state_key(c)
            if sk in seen:
                continue
            seen.add(sk)
            nl = left
            if card is not None:
                nl = dict(left); nl[card] -= 1
            self._dfs(c, slot + 1, depth + dd, nl, key + ((card,) if card else ()), fr0, k, f, full, vals, seen)


class _Budget(Exception):
    pass


# ---------------------------------------------------------------- 比较与统计

def compare_row(fr28: Frame, fr29: Frame, k: int, truth: float, conds: list[str], rnd: bool) -> dict:
    """一个（帧，敌人）的比较：min(引擎 18, 17) 对 min(真值, 17)，引擎 19 对“真值 ≥ 17”。
    敌人 18 只按硬化外壳封顶、不按生命封顶（打得死时常比 17 大），真值又截到了生命 + 格挡，所以两边都截到 17 再比（打死之后多出来的伤害不算）。
    17 为 0（回合末被动就打得死）时 18 不比，只比 19。"""
    f28, f29 = fr28.foes[k], fr29.foes[k]
    row = {"foe": f28.name, "hb": f28.hp + f28.block, "truth": truth, "conds": conds, "random": rnd, "oos": out_of_scope(fr29, f29),
           "pacts": "PACTS_END" in fr29.hand_names(),      # 合并后核对：G13（包 4）改了契约终结的预览之后，这些帧不该再算 oos
           "e17": [f28.f17, f29.f17], "e18": [f28.f18, f29.f18], "e19": [f28.f19, f29.f19]}
    for j, ff in enumerate((f28, f29)):
        tgt = min(truth, ff.f17) if ff.f17 > 0 else None
        row.setdefault("target", []).append(tgt)
        row.setdefault("e18c", []).append(min(ff.f18, ff.f17))
        row.setdefault("t19", []).append(1.0 if (ff.f17 <= 0 or truth >= ff.f17) else 0.0)
    return row


def summarize(rows: list[dict], key=None) -> dict:
    """按条件分组。非随机帧：18 完全一致（差 ≤ 0.5）、差 ≤ 2 的比例，平均偏差（引擎 − 真值），平均绝对误差；随机帧（真值为 6 个 reseed 的平均）：
    平均绝对误差；全部帧：19 一致率。格式 28、29 各一份。"""
    groups: dict[str, list[dict]] = {}
    for r in rows:
        cs = r["conds"] or ["无"]
        for c in cs:
            groups.setdefault(c, []).append(r)
        if len(r["conds"]) > 1:
            groups.setdefault("（多个条件）", []).append(r)
    out = {}
    for g in ["无"] + COND_ORDER + ["（多个条件）"]:
        rs = groups.get(g)
        if not rs:
            continue
        d = {"n": len(rs), "n_random": sum(1 for r in rs if r["random"])}
        for j, fmt in enumerate(("28", "29")):
            det = [r["e18c"][j] - r["target"][j] for r in rs if r["target"][j] is not None and not r["random"]]
            rnd = [r["e18c"][j] - r["target"][j] for r in rs if r["target"][j] is not None and r["random"]]
            n = len(det)
            d[fmt] = {
                "n18": n,
                "exact": round(sum(abs(x) <= 0.5 for x in det) / n, 4) if n else None,
                "within2": round(sum(abs(x) <= 2 for x in det) / n, 4) if n else None,
                "bias": round(float(np.mean(det)), 2) if n else None,
                "mae": round(float(np.mean(np.abs(det))), 2) if n else None,
                "n_rand": len(rnd),
                "mae_rand": round(float(np.mean(np.abs(rnd))), 2) if rnd else None,
                "agree19": round(sum(r["e19"][j] == r["t19"][j] for r in rs) / len(rs), 4),
            }
        out[g] = d
    return out
