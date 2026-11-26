"""格式 29 包 3（候选与增益）的核对（研究报告/2026-11-20_全角色观测缺口调研/R3-格式29规格.md 5.3）。

两个引擎进程同种子锁步：一个 obs_format 29，一个 28；动作取自格式 29 的帧（候选列表两边相同，不同即记下）。
每个战斗内决策帧（出牌、选牌）把两边都存进快照槽 0，对出牌、用药水、从手牌选牌的候选逐个“局面复制走一步”：恢复快照 → 两边同走这一步
（中途的选牌按固定随机数走完，多选时选完一张就确认）→ 读实际结果 → 再恢复。实际结果同时拿来比格式 28 与 29 的预测，所以两个格式的一致率可比。
  候选 12：实际能量变化 + 这张牌的费用；候选 13：手牌里新出现的牌数（整手离开的牌、药水按之后的手牌数，微光、光子切割按净变化）；
  候选 8：实际格挡变化；候选 10 自洽：预测 = 这一步实际掉的血 + 走完之后结束回合候选的 10（战斗当场结束时只算这一步掉的血）；
  候选 14：这一步之后战斗当场结束；候选 17：这一步之后的可用箭（战斗计数子类型 4 的 2）；
  候选 15：同一候选 reseed 走 6 遍，抽进手的牌在回合末对自己的伤害（目录里带 OnTurnEndInHand 的牌：HpLoss，否则 Damage，悔恨为手牌数）的平均；
  打随机敌人的候选 6：reseed 走 6 遍，敌人生命 + 格挡减少量的平均；伤害药水的 4–7：实际伤害与击杀；
  头部 32：结束回合走到下回合第一个决策帧，手牌 = 头部 32 + 33（牌堆够抽时才比，同 tools/obs113/hdraw_probe.py）；
  战斗外候选 14–17（G33）：整局模式随机走，按 op state 的牌组与目录复算。
只读；不改引擎。

  python -I tools/obs29/cand.py --dll engine/Sts2Headless/bin/Release/net9.0/Sts2Headless.dll --jobs 4 --out 草稿/cand29.json
  python -I tools/obs29/cand.py --dll ... --groups G02,G04 --combats 10      # 只跑几组
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import random
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "tools"))

import numpy as np  # noqa: E402
from sts2env.gamever import catalog_path  # noqa: E402
from sts2env.protocol import ActionKind, DecisionKind, TokenKind, Zone, parse  # noqa: E402
from obs_v21_check import Raw  # noqa: E402

CAT = {e["key"]: e for e in json.loads(Path(catalog_path()).read_text(encoding="utf-8"))["entities"]}
START = {
    "IRONCLAD": ["STRIKE_IRONCLAD"] * 4 + ["DEFEND_IRONCLAD"] * 4 + ["BASH"],
    "SILENT": ["STRIKE_SILENT"] * 4 + ["DEFEND_SILENT"] * 4 + ["NEUTRALIZE", "SURVIVOR"],
    "DEFECT": ["STRIKE_DEFECT"] * 4 + ["DEFEND_DEFECT"] * 4 + ["ZAP", "DUALCAST"],
    "REGENT": ["STRIKE_REGENT"] * 4 + ["DEFEND_REGENT"] * 4 + ["FALLING_STAR", "VENERATE"],
    "NECROBINDER": ["STRIKE_NECROBINDER"] * 4 + ["DEFEND_NECROBINDER"] * 4 + ["BODYGUARD", "UNLEASH"],
    "FOOL": ["STRIKE_FOOL"] * 4 + ["DEFEND_FOOL"] * 4 + ["FOOLISH_SHOT"],
}
START_RELIC = {"FOOL": ["SONG_BOW"], "SILENT": ["RING_OF_THE_SNAKE"]}
MIXED = [(0, "regular"), (1, "elite"), (2, "regular"), (0, "elite"), (1, "regular"), (2, "elite"), (0, "boss"), (1, "boss"), (2, "boss")]

# 每组：角色、牌组（初始牌组之外加的牌，各 2 张）、遗物、药水、遭遇轮换、只看的牌（None 为全部）
GROUPS: dict[str, dict] = {
    "G02": dict(char="IRONCLAD", add=["BURNING_PACT", "TRUE_GRIT", "FEEL_NO_PAIN", "DARK_EMBRACE", "DRUM_OF_BATTLE", "SECOND_WIND"], relics=["CHARONS_ASHES"],
                encs=[(1, "MECHA_KNIGHT_ELITE"), (0, "regular"), (1, "regular")], upg=0.5),
    "G45": dict(char="SILENT", add=["SURVIVOR", "ACROBATICS", "PREPARED", "DAGGER_THROW", "TACTICIAN", "REFLEX", "HIDDEN_DAGGERS"], relics=["TINGSHA", "TOUGH_BANDAGES"]),
    "G04": dict(char="IRONCLAD", add=["BLUDGEON", "HEMOKINESIS", "THUNDERCLAP", "CARNAGE", "WHIRLWIND"], relics=["CHARONS_ASHES"],
                encs=[(2, "THE_KIN_BOSS"), (2, "QUEEN_BOSS"), (2, "FABRICATOR_NORMAL"), (1, "OVICOPTER_NORMAL"), (1, "MECHA_KNIGHT_ELITE")], hp=300),
    "G06": dict(char="IRONCLAD", add=["POMMEL_STRIKE", "SHRUG_IT_OFF"],
                potions_cycle=[["ATTACK_POTION", "CURE_ALL", "CLARITY"], ["SKILL_POTION", "POWER_POTION", "RADIANT_TINCTURE"], ["COLORLESS_POTION", "SNECKO_OIL", "ENERGY_POTION"],
                               ["OROBIC_ACID", "LIQUID_MEMORIES", "DROPLET_OF_PRECOGNITION"], ["CUNNING_POTION", "BOTTLED_POTENTIAL", "SWIFT_POTION"],
                               ["CRYSTALLINE_EUPHORIA", "BOTTLED_SONG", "GAMBLERS_BREW"], ["GLOWWATER_POTION", "POT_OF_GHOULS", "COSMIC_CONCOCTION"]], only_potions=True),
    "G07a": dict(char="IRONCLAD", add=["POMMEL_STRIKE", "TWIN_STRIKE"], relics=["GREMLIN_HORN"], encs=[(1, "regular"), (2, "regular"), (1, "elite")]),
    "G07b": dict(char="IRONCLAD", add=["POMMEL_STRIKE", "INFLAME", "METALLICIZE", "TRUE_GRIT", "SHRUG_IT_OFF"], relics=["NUNCHAKU", "IRON_CLUB", "GAME_PIECE", "JOSS_PAPER", "MUSIC_BOX", "UNCEASING_TOP"]),
    "G07c": dict(char="NECROBINDER", add=["DEATHS_DOOR", "BLIGHT_STRIKE", "SCOURGE"], relics=["IVORY_TILE", "BURNING_STICKS"]),
    "G08": dict(char="IRONCLAD", add=["BLOODLETTING", "OFFERING", "HEMOKINESIS"], relics=["CENTENNIAL_PUZZLE"], header=True),
    "G09": dict(char="IRONCLAD", add=["BATTLE_TRANCE", "POMMEL_STRIKE", "INFERNAL_BLADE"], relics=["FIDDLE"]),
    "G10": dict(char="IRONCLAD", add=["SHRUG_IT_OFF"], potions_cycle=[["FIRE_POTION", "EXPLOSIVE_AMPOULE", "FOUL_POTION"], ["POTION_SHAPED_ROCK", "FOUL_POTION", "FIRE_POTION"]],
                encs=[(0, "regular"), (1, "regular"), (1, "elite"), (2, "regular")], only_potions=True),
    "G14i": dict(char="IRONCLAD", add=["DARK_EMBRACE", "DRUM_OF_BATTLE", "STOKE", "INFERNAL_BLADE", "PILLAGE", "JUGGLING", "VICIOUS", "CORRUPTION", "BASH"]),
    "G14c": dict(char="IRONCLAD", add=["JACKPOT", "JACK_OF_ALL_TRADES", "DISCOVERY", "SECRET_TECHNIQUE", "SECRET_WEAPON", "SEEKER_STRIKE", "ANOINTED", "SCRAWL", "IMPATIENCE", "CALAMITY"]),
    "G14s": dict(char="SILENT", add=["BLADE_DANCE", "CLOAK_AND_DAGGER", "LEADING_STRIKE", "UP_MY_SLEEVE", "BLADE_OF_INK", "HIDDEN_DAGGERS", "STORM_OF_STEEL", "FAN_OF_KNIVES", "CALCULATED_GAMBLE"]),
    "G14d": dict(char="DEFECT", add=["DUALCAST", "MULTI_CAST", "QUADCAST", "SHATTER", "SUBROUTINE", "FTL", "HOLOGRAM", "ALL_FOR_ONE", "WHITE_NOISE", "FERAL", "FUSION", "CLAW"]),
    "G14r": dict(char="REGENT", add=["COLLISION_COURSE", "CRASH_LANDING", "MANIFEST_AUTHORITY", "QUASAR", "BUNDLE_OF_JOY", "HEIRLOOM_HAMMER", "SUMMON_FORTH", "PARTICLE_WALL", "MAKE_IT_SO", "GLIMMER", "PHOTON_CUT"]),
    "G14n": dict(char="NECROBINDER", add=["FETCH", "SEVERANCE", "DREDGE", "GRAVEBLAST", "RIGHT_HAND_HAND"], relics=["IVORY_TILE"]),
    "G14f": dict(char="FOOL", add=["THE_WORLD", "THE_MOON", "THE_LOVERS", "THE_HIEROPHANT", "WHEEL_OF_FORTUNE"]),
    "G71": dict(char="IRONCLAD", add=["NEOWS_FURY", "WISH", "DUAL_WIELD", "DISTRACTION", "ABUNDANCE", "POMMEL_STRIKE"]),
    "G20": dict(char="IRONCLAD", add=["SWORD_BOOMERANG", "SWORD_BOOMERANG"], encs=[(2, "QUEEN_BOSS"), (1, "OVICOPTER_NORMAL"), (0, "regular"), (2, "THE_KIN_BOSS")], hp=300, reseed=True),
    # 合并后核对：打随机敌人的牌同时有附带伤害（狂怒的格挡触发势不可当）时，包 4 的 Riders29 会重算 4–7，看包 3 的 G20 期望伤害还在不在
    "G20R": dict(char="IRONCLAD", add=["SWORD_BOOMERANG", "SWORD_BOOMERANG", "RAGE", "JUGGERNAUT"], encs=[(2, "QUEEN_BOSS"), (1, "OVICOPTER_NORMAL"), (0, "regular"), (2, "THE_KIN_BOSS")], hp=300, reseed=True),
    "G22": dict(char="IRONCLAD", add=["DARK_EMBRACE", "AGGRESSION", "ASCENDERS_BANE", "ASCENDERS_BANE", "CORRUPTION"], relics=["JOSS_PAPER"], header=True),
    "G25a": dict(char="IRONCLAD", add=["POMMEL_STRIKE", "BATTLE_TRANCE", "SHRUG_IT_OFF", "BURNING_PACT", "SCRAWL", "JACKPOT"], encs=[(2, "AEONGLASS_BOSS")], hp=300, reseed=True),
    "G25b": dict(char="IRONCLAD", add=["POMMEL_STRIKE", "BATTLE_TRANCE", "BURN", "DECAY", "REGRET"], potions=["SWIFT_POTION"], reseed=True),
    "G32": dict(char="IRONCLAD", add=["NOT_YET", "FISTICUFFS", "THE_GAMBIT", "BLOODLETTING", "OFFERING", "HEMOKINESIS"], relics=["DEMON_TONGUE"], encs=[(1, "MECHA_KNIGHT_ELITE"), (0, "regular")]),
    "G37r": dict(char="REGENT", add=["PILLAR_OF_CREATION", "GENESIS"]),
    "G37n": dict(char="NECROBINDER", add=["SHROUD"]),
    "G46": dict(char="SILENT", add=["ESCAPE_PLAN", "ESCAPE_PLAN", "BACKFLIP", "DEFLECT"]),
    "G50": dict(char="FOOL", add=None),
    "G68": dict(char="IRONCLAD", add=["ENTRENCH", "SHRUG_IT_OFF", "IRON_WAVE"]),
    "G77": dict(char="IRONCLAD", add=["POMMEL_STRIKE", "BATTLE_TRANCE", "BURNING_PACT"], relics=["THE_ABACUS"]),
    "G78": dict(char="IRONCLAD", add=["FEEL_NO_PAIN", "DARK_EMBRACE", "DRUM_OF_BATTLE"], relics=["CHARONS_ASHES"], potions=["GLOWWATER_POTION", "GLOWWATER_POTION"], only_potions=True),
    "G65": dict(char="IRONCLAD", snapshot="mad"),
    "G81": dict(char="IRONCLAD", snapshot="adroit"),
    "G33": dict(char="ALL"),
}

# 条目 → (说明, 组, 指标, 只看的牌或药水（None 为组里全部候选）, 目标)
ITEMS = [
    ("G02", "消耗选牌候选 10 自洽", ["G02"], "c10", "select", 0.95),
    ("G03", "候选侧 fx.SelfLoss（随包 1 量）", [], None, None, None),
    ("G04", "候选 14 对当场获胜", ["G04"], "c14", None, 1.00),
    ("G04", "当场获胜候选的 10 自洽", ["G04"], "c10end", None, 1.00),
    ("G06", "药水 12", ["G06"], "c12", "potion", 0.98),
    ("G06", "药水 13", ["G06"], "c13", "potion", 0.98),
    ("G07", "遗物 12（地精之角、双截棍、象牙麻将牌）", ["G07a", "G07b", "G07c"], "c12", "play", 0.98),
    ("G07", "遗物 13（铁棒、棋子、金纸、音乐盒、陀螺、木棍）", ["G07a", "G07b", "G07c"], "c13", "play", 0.98),
    ("G08", "百年积木 13", ["G08"], "c13", "selfdmg", 0.98),
    ("G08", "头部 32（积木，下回合实际）", ["G08"], "h32", None, 0.95),
    ("G09", "小提琴 13", ["G09"], "c13", "play", 0.98),
    ("G10", "伤害药水 4–7", ["G10"], "c47pot", "dmgpot", 1.00),
    ("G10", "污浊药水 10 自洽", ["G10"], "c10", "FOUL_POTION", 0.98),
    ("G14", "铁甲 12", ["G14i"], "c12", "play", 0.98), ("G14", "铁甲 13", ["G14i"], "c13", "play", 0.98),
    ("G14", "无色 13", ["G14c"], "c13", "play", 0.98),
    ("G14", "猎人 13", ["G14s"], "c13", "play", 0.98),
    ("G14", "故障 12", ["G14d"], "c12", "play", 0.98), ("G14", "故障 13", ["G14d"], "c13", "play", 0.98),
    ("G14", "储君 13", ["G14r"], "c13", "play", 0.98),
    ("G14", "亡灵 12", ["G14n"], "c12", "play", 0.98), ("G14", "亡灵 13", ["G14n"], "c13", "play", 0.98),
    ("G14", "愚者 13", ["G14f"], "c13", "play", 0.98),
    ("G20", "打随机敌人的候选 6（reseed 平均，相对误差）", ["G20"], "g20", None, 0.10),
    ("G20", "打随机敌人 + 狂怒、势不可当的候选 6（合并后核对，相对误差）", ["G20R"], "g20", None, 0.10),
    ("G22", "头部 32（侵略、黑暗之拥、金纸）", ["G22"], "h32", None, 0.95),
    ("G25", "候选 15 的平均对实际（相对误差）", ["G25a", "G25b"], "c15", None, 0.10),
    ("G25", "手满时出牌候选 10 自洽", ["G25a"], "c10full", "play", 0.98),
    ("G32", "候选 10 自洽（恶魔之舌、自伤牌）", ["G32"], "c10", "play", 0.95),
    ("G32", "候选 8（拳拳到肉）", ["G32"], "c8", "FISTICUFFS", 1.00),
    ("G32", "候选 11（NOT_YET 回血）", ["G32"], "c11", "NOT_YET", 1.00),
    ("G33", "战斗外候选 14–17 复算", ["G33"], "g33", None, 1.00),
    ("G37", "候选 8（创世之柱、厄运之衣）", ["G37r", "G37n"], "c8", "PILLAR_OF_CREATION,GENESIS,SHROUD", 1.00),
    ("G45", "弃牌选牌候选 10 自洽", ["G45"], "c10", "select", 0.95),
    ("G45", "弃牌选牌候选 8", ["G45"], "c8", "select", 0.95),
    ("G46", "逃脱计划候选 8（预测 > 0 时与实际一致，不多算）", ["G46"], "c8sure", "ESCAPE_PLAN", 0.99),
    ("G46", "逃脱计划候选 8（全部，参考）", ["G46"], "c8", "ESCAPE_PLAN", None),
    ("G50", "候选 17 可用箭", ["G50"], "c17", "play", 0.98),
    ("G65", "疯狂科学 12", ["G65"], "c12", "MAD_SCIENCE", 1.00), ("G65", "疯狂科学 13", ["G65"], "c13", "MAD_SCIENCE", 1.00),
    ("G68", "壕沟候选 8", ["G68"], "c8", "ENTRENCH", 1.00),
    ("G71", "事件牌 13", ["G71"], "c13", "NEOWS_FURY,WISH,DUAL_WIELD,DISTRACTION,ABUNDANCE", 0.98),
    ("G77", "算盘候选 8", ["G77"], "c8", "shuffle", 1.00),
    ("G78", "发光水 8", ["G78"], "c8", "GLOWWATER_POTION", 1.00), ("G78", "发光水 12", ["G78"], "c12", "GLOWWATER_POTION", 1.00),
    ("G78", "发光水 13", ["G78"], "c13", "GLOWWATER_POTION", 0.98),
    ("G81", "伶俐附魔候选 8", ["G81"], "c8", "adroit", 1.00),
]

TURN_END = sorted(k.split(".", 1)[1] for k, v in CAT.items() if k.startswith("CARD.") and "hook:OnTurnEndInHand" in v.get("signature", []))
TURN_END_SET = set(TURN_END)
AUTOPLAY_HAND = {"CHO_CHORUS", "THE_FOOL", "THE_DEVIL"}      # 从手里自动打出的牌离手不记弃牌、消耗
HAND_CLEAR = {"STOKE", "STORM_OF_STEEL", "CALCULATED_GAMBLE", "WHEEL_OF_FORTUNE", "FIEND_FIRE", "GLOWWATER_POTION", "BOTTLED_POTENTIAL"}
NET13 = {"GLIMMER", "PHOTON_CUT", "THINKING_AHEAD"}
ZERO13 = {"GAMBLERS_BREW"}   # 要选牌，口径记 0（R3 规格 G06）
LOWER13 = {"PILLAGE"}      # 口径为下限（抽到非攻击为止，至少 1 张；R3 规格 G14），不比精确值


def end_self(cid: str, hand_n: int) -> float:
    e = CAT.get("CARD." + cid, {})
    if "hook:OnTurnEndInHand" not in e.get("signature", []): return 0.0
    if cid == "REGRET": return float(hand_n)
    v = e.get("vars") or {}
    return float(v.get("HpLoss", v.get("Damage", 0)))


def base_snapshot(E: Raw, char: str) -> dict:
    o = parse(E.req({"op": "reset", "mode": "run", "character": char, "ascension": 10, "seed": f"P3SNAP{char}", "snapshots": True}))
    rng = random.Random(5)
    for _ in range(300):
        if o.extra and "snapshot" in o.extra:
            return json.loads(o.extra["snapshot"])
        o = parse(E.req({"op": "step", "action": rng.randrange(o.num_actions)}))
    raise RuntimeError("拿不到玩家快照")


def make_spec(g: dict, name: str, i: int, rng: random.Random, snap: dict | None) -> dict:
    char = g["char"]
    act, enc = (g.get("encs") or MIXED)[i % len(g.get("encs") or MIXED)]
    add = g.get("add")
    if add is None:      # 愚者：初始牌组加随机的本角色牌
        ks = [k for k, v in CAT.items() if v.get("pool") == f"{char}_CARD_POOL" and v.get("rarity") in ("Common", "Uncommon", "Rare")]
        add = [k.split(".", 1)[1] for k in rng.sample(ks, 8)]
    deck = [{"id": c, "upgrades": 0} for c in START[char]]
    for c in add:
        if "CARD." + c not in CAT: continue
        for _ in range(2): deck.append({"id": c, "upgrades": int(rng.random() < g.get("upg", 0.3))})
    pots = g.get("potions") or []
    if g.get("potions_cycle"): pots = g["potions_cycle"][i % len(g["potions_cycle"])]
    sp = {"character": char, "ascension": 10, "seed": f"P3{name}{i}", "act": act, "encounter": enc, "deck": deck,
          "relics": START_RELIC.get(char, []) + (g.get("relics") or []), "potions": pots, "hp": g.get("hp", 200), "max_hp": g.get("hp", 200)}
    if snap is not None:
        s = json.loads(json.dumps(snap))
        cards = [{"id": "CARD." + c, "floor_added_to_deck": 1} for c in START[char]]
        if g["snapshot"] == "mad":
            # 疯狂科学技能型的三种附加效果（TinkerTimeType = 2 技能；Rider 4 充能、5 抽牌、6 混沌）
            for rider in (4, 5, 6):
                for _ in range(2):
                    cards.append({"id": "CARD.MAD_SCIENCE", "floor_added_to_deck": 1, "props": {"ints": [{"name": "TinkerTimeType", "value": 2}, {"name": "TinkerTimeRider", "value": rider}]}})
        else:
            for c in ["STRIKE_IRONCLAD", "BASH", "POMMEL_STRIKE", "TWIN_STRIKE", "DEFEND_IRONCLAD", "SHRUG_IT_OFF"]:
                cards.append({"id": "CARD." + c, "floor_added_to_deck": 1, "enchantment": {"id": "ENCHANTMENT.ADROIT", "amount": 3 + rng.randrange(3)}})
            s["relics"] = s.get("relics", [])
        s["deck"] = cards
        s["current_hp"] = s["max_hp"] = 200
        sp["player_snapshot"] = json.dumps(s)
    return sp


def stat4_arrows(o) -> float | None:
    ti, tf = o.tok_ints, o.tok_feats
    js = np.flatnonzero((ti[:, 0] == TokenKind.COMBAT_STAT) & (tf[:, 0] == 4))
    return float(tf[js[0], 2]) if len(js) else None


def et10(o) -> float | None:
    ks = [k for k in range(o.num_actions) if o.cand_ints[k, 0] == ActionKind.END_TURN]
    return float(o.cand_feats[ks[0], 10]) if ks else None


def hand_ids(st: dict) -> list[tuple[str, int]]:
    return [(c["id"], c.get("up", 0)) for c in st.get("hand", [])]


def stat_at(o, sub: int, col: int) -> float:
    ti, tf = o.tok_ints, o.tok_feats
    js = np.flatnonzero((ti[:, 0] == TokenKind.COMBAT_STAT) & (tf[:, 0] == sub))
    return float(tf[js[0], col]) if len(js) else 0.0


def counters(o, st: dict) -> dict:
    """本回合抽牌数（战斗计数 0 的 9）、消耗数（0 的 12）、弃牌数（子类型 2 的 1），消耗堆。"""
    return {"drawn": stat_at(o, 0, 9), "exh": stat_at(o, 0, 12), "disc": stat_at(o, 2, 1), "xpile": Counter(c["id"] for c in st.get("exhaust", []))}


def new_old(pre: list, post: list) -> tuple[int, int]:
    a, b = Counter(x[0] for x in pre), Counter(x[0] for x in post)
    return sum(max(0, b[k] - a[k]) for k in b), sum(max(0, a[k] - b[k]) for k in a)


class Pair:
    """格式 29、28 两个引擎锁步。"""

    def __init__(self, dll: Path, logdir: Path):
        self.a = Raw(dll, open(logdir / "e29.log", "w"))
        self.b = Raw(dll, open(logdir / "e28.log", "w"))

    def reset(self, req: dict):
        return parse(self.a.req({**req, "obs_format": 29})), parse(self.b.req({**req, "obs_format": 28}))

    def step(self, k: int):
        return parse(self.a.req({"op": "step", "action": int(k)})), parse(self.b.req({"op": "step", "action": int(k)}))

    def snap(self):
        self.a.req({"op": "snapshot", "slot": 0}); self.b.req({"op": "snapshot", "slot": 0})

    def restore(self):
        return parse(self.a.req({"op": "restore", "slot": 0})), parse(self.b.req({"op": "restore", "slot": 0}))

    def close(self):
        self.a.close(); self.b.close()


def hand_leave_prompt(o) -> bool:
    """这一帧的选牌是从手牌选牌消耗或弃掉（选牌说明记录 1 TO_EXHAUST、4 TO_DISCARD，12 来源是手牌）。"""
    ti, tf = o.tok_ints, o.tok_feats
    js = np.flatnonzero(ti[:, 0] == TokenKind.SELECT_PROMPT)
    return bool(len(js)) and tf[js[0], 12] == 1 and (tf[js[0], 1] == 1 or tf[js[0], 4] == 1)


def resolve(P: Pair, o29, o28, rng: random.Random, limit: int = 20, picked: int = 0, out: dict | None = None, first_only: bool = False):
    """走完中途的决策直到回到战斗决策或战斗结束。选牌：这一轮还没选过就随机选一张（不跳过），选过了有确认就确认；picked 为已经选了几张（选牌候选的探测为 1）。
    out["left"] 累加从手牌选去消耗、弃掉的张数（候选 13 的实际进手要加回）。"""
    g = 0
    while not o29.done and o29.decision != DecisionKind.COMBAT and g < limit:
        n = o29.num_actions
        if n == 0: break
        kinds = o29.cand_ints[:n, 0]
        conf = [k for k in range(n) if kinds[k] == ActionKind.CONFIRM_SELECTION]
        sel = [k for k in range(n) if kinds[k] in (ActionKind.SELECT_CARD, ActionKind.PICK_CARD_REWARD)]
        if conf and (not sel or (picked > 0 and first_only)): k = conf[0]; picked = 0
        elif sel:
            k = rng.choice(sel); picked += 1
            if out is not None and o29.decision == DecisionKind.CARD_SELECT and hand_leave_prompt(o29): out["left"] = out.get("left", 0) + 1
        else: k = rng.randrange(n)
        o29, o28 = P.step(k); g += 1
    return o29, o28


def run_group(dll: str, name: str, n: int, seedoff: int) -> dict:
    g = GROUPS[name]
    tmp = Path(tempfile.mkdtemp(prefix=f"p3-{name}-"))
    P = Pair(Path(dll), tmp)
    vocab = P.a.env.vocab()
    recs: list[dict] = []
    agg = Counter()
    ex: dict[str, list] = defaultdict(list)
    snap = base_snapshot(P.a, g["char"]) if g.get("snapshot") else None
    t0 = time.time()
    if name == "G33":
        return run_g33(P, vocab, n, seedoff, t0)
    for i in range(seedoff, seedoff + n):
        rng = random.Random(i * 31 + 7)
        sp = make_spec(g, name, i, rng, snap)
        o29, o28 = P.reset({"op": "reset", "mode": "combat", "spec": sp, "labels": True})
        step = 0
        while not (o29.done or o29.num_actions == 0 or step >= 400):
            n_act = o29.num_actions
            if not (np.array_equal(o29.cand_ints, o28.cand_ints)):
                agg["候选整数与格式 28 不同的帧"] += 1
            dk = o29.decision
            st = P.a.state()
            in_c = bool(st.get("in_combat")) and "player" in st
            if in_c and dk in (DecisionKind.COMBAT, DecisionKind.CARD_SELECT):
                agg["战斗帧"] += 1
                probe_frame(P, o29, o28, st, vocab, g, name, i, step, recs, agg, rng)
            # 主线：偏向出牌的随机策略
            kinds = o29.cand_ints[:n_act, 0]
            play = [k for k in range(n_act) if kinds[k] in (ActionKind.PLAY_CARD, ActionKind.USE_POTION)]
            if dk == DecisionKind.COMBAT and play and rng.random() < 0.85:
                if g.get("only_potions"):
                    pots = [k for k in play if kinds[k] == ActionKind.USE_POTION]
                    a = rng.choice(pots) if pots and rng.random() < 0.4 else rng.choice(play)
                else:
                    a = rng.choice(play)
            else:
                a = rng.randrange(n_act)
            o29, o28 = P.step(a); step += 1
    P.close()
    return {"group": name, "recs": recs, "agg": dict(agg), "seconds": round(time.time() - t0, 1)}


def card_of(o, k, vocab) -> str:
    a = int(o.cand_ints[k, 1])
    if a < 0: return ""
    key = vocab[int(o.tok_ints[a, 1])]
    return key.split(".", 1)[1] if "." in key else key


def probe_frame(P: Pair, o29, o28, st, vocab, g, name, i, step, recs, agg, rng):
    n = o29.num_actions
    kinds = o29.cand_ints[:n, 0]
    ti = o29.tok_ints
    hand_toks = [j for j in range(len(ti)) if ti[j, 0] == TokenKind.CARD and ti[j, 2] == Zone.HAND]
    seen = set(); todo = []
    for k in range(n):
        kd = int(kinds[k])
        if kd not in (ActionKind.PLAY_CARD, ActionKind.USE_POTION, ActionKind.SELECT_CARD): continue
        key = (o29.labels[k] if o29.labels else str(k), float(o29.tok_feats[o29.cand_ints[k, 1], 3]) if o29.cand_ints[k, 1] >= 0 else 0, tuple(np.round(o29.cand_feats[k], 2)))
        if key in seen: continue
        seen.add(key); todo.append(k)
    if g.get("header"):
        todo += [k for k in range(n) if kinds[k] == ActionKind.END_TURN]
    if not todo: return
    P.snap()
    pre_hand = hand_ids(st)
    cnt0 = counters(o29, st)
    pl = st["player"]
    D, R = len(st.get("draw", [])), len(st.get("discard", []))
    arrows0 = stat4_arrows(o29)
    for k in todo:
        kd = int(kinds[k])
        cname = card_of(o29, k, vocab)
        rec = {"g": name, "i": i, "step": step, "kind": kd, "card": cname, "label": o29.labels[k] if o29.labels else "",
               "f29": [float(x) for x in o29.cand_feats[k]], "f28": [float(x) for x in o28.cand_feats[k]],
               "pre": {"hp": pl["hp"], "block": pl["block"], "energy": pl["energy"], "hand": len(pre_hand), "D": D, "R": R, "arrows": arrows0,
                       "foes": [(e["hp"] + e["block"]) if e.get("alive") else -1 for e in st.get("enemies", [])]}}
        if kd == ActionKind.SELECT_CARD:
            rec["select"] = True
        if kd == ActionKind.END_TURN:
            rec["h32"] = (int(o29.header[32]), int(o29.header[33])); rec["h32_28"] = (int(o28.header[32]), int(o28.header[33]))
        # 费用
        cost = 0
        if kd == ActionKind.PLAY_CARD and int(o29.cand_ints[k, 1]) in hand_toks:
            hi = hand_toks.index(int(o29.cand_ints[k, 1]))
            if hi < len(st.get("hand", [])):
                hc = st["hand"][hi]
                cost = pl["energy"] if hc.get("cost", 0) == -1 else max(0, hc.get("cost", 0))
                played = (hc["id"], hc.get("up", 0))
            else:
                played = None
        else:
            played = None
        rec["cost"] = cost
        rec["endheal"] = sum(CAT.get("RELIC." + r, {}).get("vars", {}).get("Heal", 0) for r in pl.get("relics", []) if r in ("BURNING_BLOOD", "BLACK_BLOOD"))
        prng = random.Random(i * 1009 + step * 31 + k)
        a29, a28 = P.step(k)
        lv = {"left": 1 if kd == ActionKind.SELECT_CARD and hand_leave_prompt(o29) else 0}
        if kd == ActionKind.SELECT_CARD and not a29.done and a29.decision == DecisionKind.CARD_SELECT \
                and not any(int(a29.cand_ints[j, 0]) == ActionKind.CONFIRM_SELECTION for j in range(a29.num_actions)):
            rec["multi"] = True      # 还得再选（必须选满几张）：不是“只选这一张”的口径
        a29, a28 = resolve(P, a29, a28, prng, limit=40 if kd == ActionKind.END_TURN else 20, picked=1 if kd == ActionKind.SELECT_CARD else 0, out=lv, first_only=kd == ActionKind.SELECT_CARD)
        if not np.array_equal(a29.cand_ints, a28.cand_ints): agg["探测后两边候选整数不同"] += 1
        post_hp = int(a29.header[8])
        won = bool(a29.done) and int(a29.header[5]) == 1
        post = {"hp": post_hp, "won": won, "done": bool(a29.done)}
        s2 = P.a.state()
        post["foes"] = [(e["hp"] + e["block"]) if e.get("alive") else -1 for e in s2.get("enemies", [])] if "enemies" in s2 else None
        if not a29.done:
            if s2.get("in_combat") and "player" in s2:
                post.update(block=s2["player"]["block"], energy=s2["player"]["energy"], round=s2.get("round"))
                ph = hand_ids(s2)
                pre_minus = list(pre_hand)
                if played in pre_minus: pre_minus.remove(played)
                nw, lf = new_old(pre_minus, ph)
                cnt1 = counters(a29, s2)
                # 打出的这张自己被消耗：目录里带消耗，或腐化下的技能（同名牌被别的效果消耗时不算它）
                pe = CAT.get("CARD." + played[0], {}) if played else {}
                pexh = int(played is not None and cnt1["xpile"][played[0]] > cnt0["xpile"][played[0]]
                           and ("Exhaust" in (pe.get("keywords") or []) or (pe.get("type") == "Skill" and any(p[0] == "CORRUPTION_POWER" for p in pl.get("powers", [])))))
                # 离手张数：本回合弃牌数、消耗数的增量（打出的这张若被消耗要扣掉）
                lft = (cnt1["disc"] - cnt0["disc"]) + (cnt1["exh"] - cnt0["exh"]) - pexh
                post.update(hand=len(ph), new=nw, left=lf, leftsel=lv["left"], leftcnt=lft, drawn=cnt1["drawn"] - cnt0["drawn"],
                            same=(s2.get("round") == st.get("round")) and a29.decision == DecisionKind.COMBAT)
                post["newself"] = newself(a29, s2, pre_minus, wither=0)
                post["et10_29"], post["et10_28"] = et10(a29), et10(a28)
                post["arrows"] = stat4_arrows(a29)
                if kd == ActionKind.END_TURN:
                    post["next_hand"] = len(ph)
                    rec["pile"] = D + R + len(pre_hand)
        rec["post"] = post
        # reseed：打随机敌人的伤害（G20）、抽到回合末自伤牌（G25）
        if g.get("reseed") and kd in (ActionKind.PLAY_CARD, ActionKind.USE_POTION) and (rec["f29"][15] > 0 or rec["f29"][13] > 0 or "RandomEnemy" == CAT.get("CARD." + cname, {}).get("target")):
            dmg, selfs = [], []
            foes0 = {j: (e["hp"] + e["block"]) for j, e in enumerate(st.get("enemies", [])) if e.get("alive")}
            wither0 = stat0(o29, 19)
            for s in range(6):
                parse(P.a.req({"op": "restore", "slot": 0, "reseed": 100 + s}))
                r29 = parse(P.a.req({"op": "step", "action": int(k)}))
                g2 = 0
                while not r29.done and r29.decision != DecisionKind.COMBAT and g2 < 20:
                    # 中途选牌（燃烧契约等）挑不带回合末效果的牌，免得“消耗一张凋萎、又抽到一张凋萎”在张数比较里抵消
                    ok = [j for j in range(r29.num_actions) if not set((r29.labels[j] if r29.labels else "").split()) & TURN_END_SET]
                    r29 = parse(P.a.req({"op": "step", "action": prng.choice(ok) if ok else prng.randrange(r29.num_actions)})); g2 += 1
                s3 = P.a.state()
                if r29.done or "player" not in s3:
                    dmg.append(sum(foes0.values())); selfs.append(0.0); continue
                foes1 = {j: (e["hp"] + e["block"]) if e.get("alive") else 0 for j, e in enumerate(s3.get("enemies", []))}
                dmg.append(sum(v - foes1.get(j, 0) for j, v in foes0.items()))
                ph = hand_ids(s3)
                pre_minus = list(pre_hand)
                if played in pre_minus: pre_minus.remove(played)
                # 永世沙漏这一步塞进手里的凋萎不是抽到的（战斗计数主记录 19：再打一张牌塞进来的凋萎的伤害）
                # 沙漏的凋萎只在手牌没满时进手：新进手的张数多于实际抽的张数（战斗计数 9，本回合抽牌数）时才扣
                drawn = stat0(r29, 9) - stat0(o29, 9)
                nnew = len(new_cards(pre_minus, hand_ids(s3)))
                selfs.append(newself(r29, s3, pre_minus, wither=wither0 if kd == ActionKind.PLAY_CARD and nnew > drawn else 0))
            rec["reseed_dmg"] = float(np.mean(dmg)); rec["reseed_self"] = float(np.mean(selfs))
            rec["foes_hittable"] = len(foes0)
        recs.append(rec)
        P.restore()
    b29, b28 = P.restore()
    if not np.array_equal(b29.cand_ints, o29.cand_ints): agg["恢复后与原帧不同"] += 1


def stat0(o, col: int) -> float:
    ti, tf = o.tok_ints, o.tok_feats
    js = np.flatnonzero((ti[:, 0] == TokenKind.COMBAT_STAT) & (tf[:, 0] == 0))
    return float(tf[js[0], col]) if len(js) else 0.0


def newself(o, st: dict, pre_minus: list, wither: float = 0.0) -> float:
    """这一步新进手的牌在回合末对自己的伤害（同引擎 EndTurnSelfDmg：悔恨为手牌数，有 HpLoss 取目录值，否则取帧里这张牌的 Damage 现值，卡牌记录 5）。
    wither > 0 时扣掉一张永世沙漏塞进来的凋萎（它不是抽到的）。"""
    ti, tf = o.tok_ints, o.tok_feats
    hand = st.get("hand", [])
    toks = [j for j in range(len(ti)) if ti[j, 0] == TokenKind.CARD and ti[j, 2] == Zone.HAND]
    dmg = defaultdict(list)
    for (cid, up), j in zip(hand_ids(st), toks): dmg[(cid, up)].append(float(tf[j, 5]))
    tot = 0.0; wsub = wither
    for cid, up in new_cards(pre_minus, hand_ids(st)):
        e = CAT.get("CARD." + cid, {})
        if "hook:OnTurnEndInHand" not in e.get("signature", []): continue
        v = e.get("vars") or {}
        if cid == "REGRET": val = float(len(hand))
        elif "HpLoss" in v: val = float(v["HpLoss"])
        elif "Damage" in v: val = max(dmg.get((cid, up), [float(v["Damage"])]))
        else: val = 0.0
        if cid == "WITHER" and wsub > 0: wsub = 0; continue
        tot += val
    return tot


def new_cards(pre: list, post: list) -> list:
    a = Counter(pre); out = []
    for k, v in Counter(post).items():
        out += [k] * max(0, v - a[k])
    return out


def run_g33(P: Pair, vocab, n: int, seedoff: int, t0: float) -> dict:
    agg = Counter(); recs = []
    chars = list(START)
    for i in range(seedoff, seedoff + n):
        rng = random.Random(i)
        char = chars[i % len(chars)]
        o29, o28 = P.reset({"op": "reset", "mode": "run", "character": char, "ascension": 10, "seed": f"P3G33{i}"})
        for t in range(900):
            if o29.done or o29.num_actions == 0: break
            st = P.a.state()
            if not st.get("in_combat") and "run" in st and o29.decision not in (DecisionKind.COMBAT, DecisionKind.GAME_OVER, DecisionKind.STUCK):
                deck = st["run"]["deck"]
                for k in range(o29.num_actions):
                    a = int(o29.cand_ints[k, 1])
                    if a < 0 or int(o29.tok_ints[a, 0]) != TokenKind.CARD: continue
                    key = vocab[int(o29.tok_ints[a, 1])]
                    cid = key.split(".", 1)[1]
                    up = float(o29.tok_feats[a, 3]) > 0; ench = int(o29.tok_ints[a, 4]) != 0
                    in_deck = int(o29.tok_ints[a, 2]) == Zone.DECK
                    typ = CAT.get(key, {}).get("type")
                    same = [d for d in deck if d["id"] == cid]
                    exp = [len(same) - in_deck, sum(1 for d in same if d.get("up", 0) > 0) - (in_deck and up),
                           sum(1 for d in same if "ench" in d) - (in_deck and ench), sum(1 for d in deck if CAT.get("CARD." + d["id"], {}).get("type") == typ) - in_deck]
                    got = [float(x) for x in o29.cand_feats[k, 14:18]]
                    recs.append({"g": "G33", "exp": exp, "got": got, "card": cid, "zone": int(o29.tok_ints[a, 2])})
            n_act = o29.num_actions
            kinds = o29.cand_ints[:n_act, 0]
            play = [k for k in range(n_act) if kinds[k] == ActionKind.PLAY_CARD]
            a = rng.choice(play) if o29.decision == DecisionKind.COMBAT and play and rng.random() < 0.9 else rng.randrange(n_act)
            o29, o28 = P.step(a)
    P.close()
    return {"group": "G33", "recs": recs, "agg": dict(agg), "seconds": round(time.time() - t0, 1)}


# ---------------- 指标 ----------------

def want(rec: dict, flt: str | None) -> bool:
    if flt is None: return True
    if flt == "play": return rec["kind"] == ActionKind.PLAY_CARD
    if flt == "potion": return rec["kind"] == ActionKind.USE_POTION
    if flt == "select": return rec.get("select", False) and not rec.get("multi", False)
    if flt == "dmgpot": return rec["kind"] == ActionKind.USE_POTION and rec["card"] in ("FIRE_POTION", "EXPLOSIVE_AMPOULE", "FOUL_POTION", "POTION_SHAPED_ROCK")
    if flt == "selfdmg": return rec["kind"] == ActionKind.PLAY_CARD and rec["card"] in ("BLOODLETTING", "OFFERING", "HEMOKINESIS")
    if flt == "shuffle": return rec["kind"] in (ActionKind.PLAY_CARD, ActionKind.USE_POTION) and rec["f29"][13] > rec["pre"]["D"]
    if flt == "adroit": return rec["kind"] == ActionKind.PLAY_CARD
    return rec["kind"] in (ActionKind.PLAY_CARD, ActionKind.USE_POTION) and rec["card"] in flt.split(",")


def metric(m: str, rec: dict, fmt: int) -> bool | None:
    """一条探测记录在这个指标上：None 不参与，True / False 一致与否。"""
    f = rec["f29"] if fmt == 29 else rec["f28"] + [0.0] * 6
    post, pre = rec["post"], rec["pre"]
    same = post.get("same", False)
    if m == "c12":
        if not same or "energy" not in post: return None
        act = post["energy"] - pre["energy"] + rec["cost"]
        if act == 0 and rec["f29"][12] == 0 and rec["f28"][12] == 0: return None
        return abs(f[12] - act) < 0.5
    if m == "c13":
        if not same or "hand" not in post: return None
        if rec["card"] in LOWER13 or rec["card"] in ZERO13: return None
        net = post["hand"] - pre["hand"] + (1 if rec["kind"] == ActionKind.PLAY_CARD else 0)
        if rec["card"] in NET13: act = net
        elif rec["card"] in HAND_CLEAR and rec["kind"] != ActionKind.SELECT_CARD: act = post["hand"]
        elif "leftcnt" in post:
            act = net + post["leftcnt"]
            if rec["card"] in AUTOPLAY_HAND: act = max(act, post["new"])
        else: act = max(post["new"], net + post.get("leftsel", 0))
        if act == 0 and rec["f29"][13] == 0 and rec["f28"][13] == 0: return None
        return abs(f[13] - act) < 0.5
    if m == "c8sure":
        if not same or "block" not in post or f[8] <= 0: return None
        return abs(f[8] - (post["block"] - pre["block"])) < 0.5
    if m == "c8":
        if not same or "block" not in post: return None
        act = post["block"] - pre["block"]
        if act == 0 and rec["f29"][8] == 0 and rec["f28"][8] == 0: return None
        return abs(f[8] - act) < 0.5
    if m in ("c10", "c10full", "c10end"):
        if rec["kind"] == ActionKind.END_TURN: return None
        if m == "c10full" and not (pre["hand"] - 1 + rec["f29"][13] >= 10): return None
        if m == "c10end" and not (rec["f29"][14] == 1 or post.get("won")): return None
        if post.get("newself", 0) > 0: return None      # 这一步抽到了回合末自伤的牌（不在候选 10 的口径里，见候选 15）
        if post.get("won"): tgt = max(0, pre["hp"] - post["hp"] + rec.get("endheal", 0))
        elif same and post.get(f"et10_{fmt}") is not None: tgt = max(0, pre["hp"] - post["hp"]) + post[f"et10_{fmt}"]
        else: return None
        return abs(f[10] - tgt) < 0.5
        loss = pre["hp"] - post["hp"]
        if post.get("won"): tgt = loss
        elif same and post.get(f"et10_{fmt}") is not None: tgt = loss + post[f"et10_{fmt}"]
        else: return None
        return abs(f[10] - tgt) < 0.5
    if m == "c11":
        if not same: return None
        # NOT_YET：11 = 回血之后的生命 − 10，10 自洽时等价于 11 = 实际生命 − 之后结束回合的 10
        if post.get(f"et10_{fmt}") is None: return None
        return abs(f[11] - (post["hp"] - post[f"et10_{fmt}"])) < 0.5
    if m == "c14":
        if rec["kind"] not in (ActionKind.PLAY_CARD, ActionKind.USE_POTION, ActionKind.SELECT_CARD): return None
        p = f[14] if fmt == 29 else 0.0
        act = 1.0 if post.get("won") else 0.0
        if act == 0 and rec["f29"][14] == 0: return None
        return abs(p - act) < 0.5
    if m == "c17":
        if not same or post.get("arrows") is None: return None
        p = f[17] if fmt == 29 else -1.0
        return abs(p - post["arrows"]) < 0.5
    if m == "h32":
        if rec["kind"] != ActionKind.END_TURN or "next_hand" not in post: return None
        h32, h33 = rec["h32"] if fmt == 29 else rec["h32_28"]
        if rec.get("pile", 0) < h33 + h32: return None
        return post["next_hand"] == h32 + h33
    if m == "c47pot":
        # 伤害药水：7 = 实际打死的敌人数；没打死时 6 = 敌人生命 + 格挡的减少之和；4 > 0、5 = 1
        if rec["kind"] != ActionKind.USE_POTION or not (same or post.get("won")): return None
        f0, f1 = pre["foes"], post.get("foes")
        if f1 is None or len(f1) < len(f0):
            if not post.get("won"): return None
            f1 = [-1] * len(f0)
        kills = sum(1 for a, b in zip(f0, f1) if a >= 0 and b < 0)
        dec = sum(a - b for a, b in zip(f0, f1) if a >= 0 and b >= 0)
        if abs(f[7] - kills) >= 0.5 or f[4] <= 0 or abs(f[5] - 1) >= 0.5: return False
        return kills > 0 or abs(f[6] - dec) < 0.5
    return None


def table(results: list[dict]) -> list[dict]:
    by = defaultdict(list)
    for r in results:
        by[r["group"]].extend(r["recs"])
    rows = []
    for item, desc, groups, m, flt, target in ITEMS:
        recs = [x for gname in groups for x in by.get(gname, []) if want(x, flt)] if m else []
        row = {"条目": item, "说明": desc, "目标": target}
        if m is None:
            row.update(帧数=0, 备注="随包 1 的预演核对")
        elif m == "g20":
            xs = [x for x in recs if x["kind"] == ActionKind.PLAY_CARD and "reseed_dmg" in x and x.get("foes_hittable", 0) >= 2
                  and CAT.get("CARD." + x["card"], {}).get("target") == "RandomEnemy"]
            act = sum(x["reseed_dmg"] for x in xs)
            for fmt in (28, 29):
                pred = sum((x["f29"] if fmt == 29 else x["f28"])[6] for x in xs)
                row[f"{fmt}"] = round(abs(pred - act) / act, 3) if act else None
            row["帧数"] = len(xs); row["通过"] = row["29"] is not None and row["29"] <= target
        elif m == "c15":
            xs = [x for x in recs if "reseed_self" in x and (x["f29"][15] > 0 or x["reseed_self"] > 0)]
            act = sum(x["reseed_self"] for x in xs); pred = sum(x["f29"][15] for x in xs)
            row["28"] = 1.0 if act else None
            row["29"] = round(abs(pred - act) / act, 3) if act else None
            row["帧数"] = len(xs); row["通过"] = row["29"] is not None and row["29"] <= target
            row["平均"] = {"预测": round(pred / max(1, len(xs)), 3), "实际": round(act / max(1, len(xs)), 3)}
        elif m == "g33":
            xs = recs
            ok = sum(1 for x in xs if all(abs(a - b) < 0.5 for a, b in zip(x["exp"], x["got"])))
            row.update({"帧数": len(xs), "28": None, "29": round(ok / max(1, len(xs)), 4)}); row["通过"] = ok == len(xs) and len(xs) > 0
        else:
            for fmt in (28, 29):
                v = [metric(m, x, fmt) for x in recs]
                v = [b for b in v if b is not None]
                row[f"{fmt}"] = round(sum(v) / len(v), 4) if v else None
                row["帧数"] = len(v)
            row["通过"] = row["29"] is not None and (target is None or row["29"] >= target)
        rows.append(row)
    return rows


def examples(results: list[dict], per: int = 4) -> dict:
    """各指标里格式 29 不一致的例子（排查用）。"""
    by = defaultdict(list)
    for r in results: by[r["group"]].extend(r["recs"])
    out = {}
    for item, desc, groups, m, flt, target in ITEMS:
        if m in (None, "g20", "c15", "g33") or target is None: continue
        bad = []
        for gname in groups:
            for x in by.get(gname, []):
                if want(x, flt) and metric(m, x, 29) is False:
                    bad.append({k: x[k] for k in ("g", "i", "step", "card", "label", "cost") if k in x} |
                               {"f29": [round(v, 1) for v in x["f29"]], "f28": [round(v, 1) for v in x["f28"]], "pre": x["pre"], "post": x["post"]})
        if bad:
            cnt = Counter(b["card"] for b in bad)
            out[f"{item} {desc}"] = {"按牌": dict(cnt.most_common(12)), "例子": bad[:per]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dll", required=True)
    ap.add_argument("--groups", default=",".join(GROUPS))
    ap.add_argument("--combats", type=int, default=40)
    ap.add_argument("--g33-runs", type=int, default=12)
    ap.add_argument("--jobs", type=int, default=4, help="并行组数（每组两个引擎进程）")
    ap.add_argument("--seedoff", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--raw", default=None, help="存下全部探测记录（JSON）")
    a = ap.parse_args()
    names = [x for x in a.groups.split(",") if x]
    results = []
    with cf.ProcessPoolExecutor(max_workers=a.jobs) as pool:
        futs = {pool.submit(run_group, a.dll, nm, a.g33_runs if nm == "G33" else a.combats, a.seedoff): nm for nm in names}
        for fu in cf.as_completed(futs):
            r = fu.result()
            print(f"{r['group']}：{len(r['recs'])} 条探测，{r['seconds']} 秒，{r['agg']}", flush=True)
            results.append(r)
    rows = table(results)
    res = {"表": rows, "不一致的例子": examples(results), "各组": {r["group"]: r["agg"] for r in results}}
    txt = json.dumps(res, ensure_ascii=False, indent=1)
    if a.out: Path(a.out).write_text(txt, encoding="utf-8")
    if a.raw: Path(a.raw).write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")
    for r in rows:
        print(f"{r['条目']:4} {r['说明'][:28]:30} 帧数 {r.get('帧数')!s:>6}  28 {r.get('28')!s:>7}  29 {r.get('29')!s:>7}  目标 {r['目标']!s:>5}  {'通过' if r.get('通过') else '—'}")


if __name__ == "__main__":
    main()
