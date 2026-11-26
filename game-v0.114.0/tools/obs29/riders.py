"""格式 29 包 4（角色与附带效果，R3-格式29规格.md 5.4）的逐条核对：格式 28 与格式 29 各算一遍，对真实结算。

两个引擎进程同种子锁步（一个 reset 带 obs_format 29、一个带 28），动作取自格式 29 的帧（候选列表两边相同，不同即报错）。
观测格式不影响游戏，两边局面逐步相同，所以同一个局面上能比“格式 28 的预测”与“格式 29 的预测”哪个更接近真实结果（op state）。

指标（同 tools/obs_v22_check.py 的口径）：
  候选 8：打出后格挡的增加（打出途中要选牌的不比）；
  候选 6、7：有目标时能否打死目标、没打死时对目标的总伤害；无目标时打死几个、没打死时的总伤害；
  候选 10 自洽：打出前的候选 10 = 这一步实际掉的血 + 打出后结束回合候选的 10；
  候选 16（格式 29 新列，G40）：打出后立即结束回合，敌方回合里身上有灾厄、被处决的敌人数；
  卡牌 7（G55、G56）：手牌外扯碎的段数 = 1 + 战斗计数子类型 3 的 1；异蛇头骨下手牌致命毒药的层数 = 卡面 + 1。
打法：偏向出牌的随机策略，按组偏向打这一组关心的牌（focus），每组默认 40 场（--combats 改）。

  python -I tools/obs29/riders.py --groups all --combats 40 --workers 4 --dump .cache/obs29/riders
  python -I tools/obs29/riders.py --groups G01,G13 --combats 8      # 只跑几组
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "tools"))

import numpy as np  # noqa: E402

from sts2env.protocol import ActionKind, DecisionKind, TokenKind, Zone, parse  # noqa: E402
from obs_v21_check import Raw, match_enemy  # noqa: E402

DLL = REPO / "engine" / "Sts2Headless" / "bin" / "Release" / "net9.0" / "Sts2Headless.dll"
STARTER = {
    "IRONCLAD": ["STRIKE_IRONCLAD"] * 3 + ["DEFEND_IRONCLAD"] * 3,
    "SILENT": ["STRIKE_SILENT"] * 3 + ["DEFEND_SILENT"] * 3,
    "DEFECT": ["STRIKE_DEFECT"] * 3 + ["DEFEND_DEFECT"] * 3 + ["ZAP", "DUALCAST"],
    "REGENT": ["STRIKE_REGENT"] * 3 + ["DEFEND_REGENT"] * 3,
    "NECROBINDER": ["STRIKE_NECROBINDER"] * 3 + ["DEFEND_NECROBINDER"] * 3,
    "FOOL": [],
}
ENC = [(0, "regular"), (1, "elite"), (2, "boss"), (1, "regular"), (0, "elite"), (2, "regular"), (1, "boss"), (2, "elite")]
VUL_CARDS = {"BASH", "THUNDERCLAP", "TAUNT", "UPPERCUT", "SHOCKWAVE", "TREMBLE", "DOMINATE", "BREAK", "MOLTEN_FIST"}
DOOM_CARDS = {"SCOURGE", "NEGATIVE_PULSE", "DEATHBRINGER", "BLIGHT_STRIKE", "NO_ESCAPE", "END_OF_DAYS", "OBLIVION", "MISERY", "TIMES_UP"}
SELF_DMG = {"BLOODLETTING", "OFFERING", "HEMOKINESIS", "BLOOD_WALL", "BRAND", "BREAKTHROUGH", "DEMONIC_SHIELD"}

# 组：角色、牌池（每场抽 k 种各 2 张，加初始牌）、固定加的牌、遗物轮换、遭遇、药水、血量、偏向打的牌、附加检查
GROUPS: dict[str, dict] = {
    "G01": dict(char="IRONCLAD", fixed=["FEEL_NO_PAIN", "FEEL_NO_PAIN", "HAVOC", "HAVOC", "HAVOC", "THRASH", "THRASH", "STOKE", "STOKE"],
                pool=["SHRUG_IT_OFF", "TRUE_GRIT", "IRON_WAVE", "POMMEL_STRIKE", "JUGGERNAUT", "HEADBUTT", "BURNING_PACT"], k=3,
                relics=[[], ["CHARONS_ASHES"], ["FORGOTTEN_SOUL"]], focus=["FEEL_NO_PAIN", "JUGGERNAUT", "HAVOC", "THRASH", "STOKE"]),
    # 浩劫的顶牌已知：除浩劫、无惧疼痛外全是防御，抽牌堆常常全是同一张
    "G01T": dict(char="IRONCLAD", fixed=["FEEL_NO_PAIN", "HAVOC", "HAVOC", "HAVOC"] + ["DEFEND_IRONCLAD"] * 8, pool=[], k=0, starter=False, upg=0,
                 relics=[[], ["CHARONS_ASHES"]], focus=["FEEL_NO_PAIN", "HAVOC"]),
    "G02": dict(char="IRONCLAD", fixed=["SECOND_WIND", "SECOND_WIND", "FIEND_FIRE", "FIEND_FIRE", "STOKE", "STOKE"],
                pool=["FEEL_NO_PAIN", "SHRUG_IT_OFF", "IRON_WAVE", "TWIN_STRIKE"], k=2, relics=[[]], encounter=[(1, "ENCOUNTER.MECHA_KNIGHT_ELITE")],
                focus=["SECOND_WIND", "FIEND_FIRE", "STOKE"]),
    "G13": dict(char="IRONCLAD", fixed=["PACTS_END", "PACTS_END", "PACTS_END", "TRUE_GRIT", "BURNING_PACT", "SECOND_WIND"],
                pool=["SHRUG_IT_OFF", "IRON_WAVE"], k=1, relics=[[]], focus=["PACTS_END", "TRUE_GRIT"]),
    "G15": dict(char="IRONCLAD", fixed=["COLOSSUS", "COLOSSUS", "FIGHT_ME", "FIGHT_ME", "BASH", "THUNDERCLAP"],
                pool=["TAUNT", "UPPERCUT", "SHOCKWAVE", "TREMBLE", "DOMINATE", "BREAK"], k=3, relics=[[]],
                potions=["VULNERABLE_POTION", "VULNERABLE_POTION", "VULNERABLE_POTION"],
                focus=["COLOSSUS", "FIGHT_ME", "BASH", "THUNDERCLAP", "TAUNT", "UPPERCUT", "SHOCKWAVE", "TREMBLE", "DOMINATE", "BREAK", "VULNERABLE_POTION"]),
    "G15R": dict(char="REGENT", fixed=["MONARCHS_GAZE", "MONARCHS_GAZE", "SHINING_STRIKE", "SOLAR_STRIKE", "FALLING_STAR", "COLLISION_COURSE"], pool=["KNOCKOUT_BLOW", "GATHER_LIGHT"], k=1,
                 relics=[[]], focus=["MONARCHS_GAZE", "SHINING_STRIKE", "SOLAR_STRIKE", "FALLING_STAR", "COLLISION_COURSE", "STRIKE_REGENT", "KNOCKOUT_BLOW"]),
    "G15N": dict(char="NECROBINDER", fixed=["DEBILITATE", "DEBILITATE", "MISERY", "MISERY", "DEFY", "PUTREFY", "SCOURGE", "DEATHBRINGER"], pool=["NEGATIVE_PULSE", "FEAR"], k=1,
                 relics=[["BOUND_PHYLACTERY", "UNDYING_SIGIL"], ["BOUND_PHYLACTERY"]], focus=["DEBILITATE", "MISERY", "DEFY", "PUTREFY", "SCOURGE", "DEATHBRINGER", "NEGATIVE_PULSE"]),
    "G15S": dict(char="SILENT", fixed=["ENVENOM", "ENVENOM", "DAGGER_SPRAY", "DAGGER_SPRAY", "RICOCHET", "POISONED_STAB"], pool=["NEUTRALIZE", "SUCKER_PUNCH"], k=1,
                 relics=[[]], focus=["ENVENOM", "DAGGER_SPRAY", "RICOCHET", "POISONED_STAB", "STRIKE_SILENT", "NEUTRALIZE", "SUCKER_PUNCH"]),
    "G31": dict(char="IRONCLAD", fixed=["TWIN_STRIKE", "POMMEL_STRIKE", "INFLAME", "TRUE_GRIT"], pool=["IRON_WAVE", "SWORD_BOOMERANG", "DEMON_FORM", "BURNING_PACT"], k=2,
                relics=[["DAUGHTER_OF_THE_WIND"], ["DAUGHTER_OF_THE_WIND", "LOST_WISP"], ["FORGOTTEN_SOUL", "DAUGHTER_OF_THE_WIND"]],
                focus=["TWIN_STRIKE", "STRIKE_IRONCLAD", "INFLAME", "DEMON_FORM", "TRUE_GRIT", "BURNING_PACT"]),
    "G55": dict(char="IRONCLAD", fixed=["TEAR_ASUNDER", "TEAR_ASUNDER", "TEAR_ASUNDER"], pool=["HEMOKINESIS", "BLOODLETTING"], k=1, relics=[[]], focus=["HEMOKINESIS"]),
    "G56": dict(char="SILENT", fixed=["DEADLY_POISON", "DEADLY_POISON", "DEADLY_POISON"], pool=["NEUTRALIZE", "BACKFLIP"], k=1, relics=[["SNECKO_SKULL"]], focus=[]),
    "G66": dict(char="IRONCLAD", fixed=["HEMOKINESIS"] * 4 + ["BREAKTHROUGH"] * 2, pool=[], k=0, relics=[["RED_SKULL"], ["RED_SKULL", "RUINED_HELMET"]],
                hp=(41, 80), focus=["HEMOKINESIS", "BREAKTHROUGH"]),
    "G67": dict(char="IRONCLAD", fixed=["WHISTLE", "WHISTLE", "WHISTLE"], pool=["SHRUG_IT_OFF", "IRON_WAVE"], k=1, relics=[[]], focus=["WHISTLE"]),
    "G72": dict(char="IRONCLAD", fixed=["CALTROPS", "CALTROPS"], pool=["SHRUG_IT_OFF", "IRON_WAVE"], k=1, relics=[[]],
                encounter=[(0, "ENCOUNTER.TOADPOLES_WEAK"), (1, "regular"), (0, "regular")], focus=["CALTROPS"]),
    "G38": dict(char="REGENT", fixed=["PILLAR_OF_CREATION", "PILLAR_OF_CREATION", "COLLISION_COURSE", "COLLISION_COURSE", "MANIFEST_AUTHORITY", "CHILD_OF_THE_STARS"],
                pool=["GATHER_LIGHT", "GLOW", "VENERATE", "BIG_BANG", "QUASAR", "CRASH_LANDING", "BUNDLE_OF_JOY", "FALLING_STAR", "BLACK_HOLE", "SHINING_STRIKE"], k=4,
                relics=[[], ["GALACTIC_DUST"], ["REGALITE"], ["GALACTIC_DUST", "REGALITE"]],
                focus=["PILLAR_OF_CREATION", "CHILD_OF_THE_STARS", "BLACK_HOLE", "COLLISION_COURSE", "MANIFEST_AUTHORITY", "QUASAR", "CRASH_LANDING", "BUNDLE_OF_JOY",
                       "GATHER_LIGHT", "VENERATE", "BIG_BANG", "FALLING_STAR", "SHINING_STRIKE", "GLOW"]),
    "G39": dict(char="NECROBINDER", fixed=["SHROUD", "SHROUD", "SLEIGHT_OF_FLESH", "NEGATIVE_PULSE", "SCOURGE", "DEATHBRINGER"],
                pool=["BLIGHT_STRIKE", "NO_ESCAPE", "FEAR", "DEFY", "REAPER_FORM", "SPIRIT_OF_ASH", "DANSE_MACABRE", "HAUNT", "DEVOUR_LIFE", "DEBILITATE", "PUTREFY"], k=4,
                relics=[["BOUND_PHYLACTERY"]],
                focus=["SHROUD", "SLEIGHT_OF_FLESH", "REAPER_FORM", "SPIRIT_OF_ASH", "DANSE_MACABRE", "HAUNT", "DEVOUR_LIFE", "NEGATIVE_PULSE", "SCOURGE", "DEATHBRINGER",
                       "BLIGHT_STRIKE", "NO_ESCAPE", "FEAR", "DEFY", "DEBILITATE", "PUTREFY"]),
    "G40": dict(char="NECROBINDER", fixed=["SCOURGE", "SCOURGE", "SCOURGE", "DEATHBRINGER", "DEATHBRINGER", "NEGATIVE_PULSE", "NEGATIVE_PULSE", "NO_ESCAPE", "NO_ESCAPE"],
                pool=["BLIGHT_STRIKE", "REAPER_FORM", "END_OF_DAYS", "OBLIVION", "SHROUD", "SLEIGHT_OF_FLESH"], k=3, relics=[["BOUND_PHYLACTERY"]],
                encounter=[(0, "weak"), (0, "regular"), (1, "weak"), (1, "regular"), (0, "elite")],
                focus=["SCOURGE", "DEATHBRINGER", "NEGATIVE_PULSE", "NO_ESCAPE", "BLIGHT_STRIKE", "REAPER_FORM", "END_OF_DAYS", "OBLIVION", "SLEIGHT_OF_FLESH"]),
    "G41": dict(char="NECROBINDER", fixed=["DIRGE", "DIRGE", "SPUR", "SPUR", "BONE_SHARDS"], pool=["POKE", "FLATTEN", "BODYGUARD", "SIC_EM", "HIGH_FIVE"], k=3,
                relics=[["BOUND_PHYLACTERY"]], encounter=[(0, "ENCOUNTER.TOADPOLES_WEAK"), (0, "regular"), (1, "regular"), (1, "elite")],
                focus=["DIRGE", "SPUR", "BONE_SHARDS", "SIC_EM", "POKE", "HIGH_FIVE"]),
    "G42A": dict(char="DEFECT", fixed=["SHADOW_SHIELD", "SHADOW_SHIELD", "NULL", "NULL", "METEOR_STRIKE", "GLACIER", "CHILL", "CHILL", "COOLHEADED"], pool=[], k=0, relics=[[]],
                 focus=["SHADOW_SHIELD", "NULL", "METEOR_STRIKE", "GLACIER", "COOLHEADED"]),
    "G42B": dict(char="DEFECT", fixed=["STORM", "STORM", "DEFRAGMENT", "DEFRAGMENT", "CAPACITOR", "LOOP", "CHILL", "CHILL", "COOLHEADED"], pool=[], k=0, relics=[[]],
                 focus=["STORM", "DEFRAGMENT", "CAPACITOR", "LOOP"]),
    "G42E2": dict(char="DEFECT", fixed=["HAILSTORM", "THUNDER", "TEMPEST", "MULTI_CAST", "VOLTAIC", "SPINNER", "METEOR_STRIKE", "FUSION", "COOLHEADED", "CHILL"], pool=[], k=0,
                  relics=[[]], focus=["VOLTAIC", "SPINNER", "METEOR_STRIKE", "TEMPEST", "MULTI_CAST"]),
    "G42E3": dict(char="DEFECT", fixed=["COOLHEADED", "COOLHEADED", "CHILL", "ZAP", "BALL_LIGHTNING", "GLACIER"], pool=[], k=0, relics=[["METRONOME"]],
                  focus=["COOLHEADED", "CHILL", "ZAP", "BALL_LIGHTNING", "GLACIER"]),
    "G42E4": dict(char="DEFECT", fixed=["COOLHEADED", "COOLHEADED", "CHILL", "ZAP", "BALL_LIGHTNING", "GLACIER"], pool=[], k=0, relics=[["GOLD_PLATED_CABLES"]],
                  focus=["COOLHEADED", "CHILL", "ZAP", "BALL_LIGHTNING", "GLACIER", "DUALCAST"]),
    "G42H": dict(char="DEFECT", fixed=["BULK_UP", "BULK_UP", "SPINNER", "SPINNER", "MODDED", "CHAOS", "COOLHEADED", "GLACIER"], pool=[], k=0, relics=[[]], upg=1,
                 focus=["BULK_UP", "SPINNER", "MODDED", "CHAOS"]),
    "G45": dict(char="SILENT", fixed=["CALCULATED_GAMBLE", "CALCULATED_GAMBLE", "STORM_OF_STEEL", "SHADOW_STEP", "TACTICIAN", "REFLEX", "TACTICIAN", "REFLEX"],
                pool=["MEMENTO_MORI", "BACKFLIP", "SURVIVOR"], k=2, relics=[[], ["TINGSHA"], ["TOUGH_BANDAGES"], ["TINGSHA", "TOUGH_BANDAGES"]],
                focus=["CALCULATED_GAMBLE", "STORM_OF_STEEL", "SHADOW_STEP"]),
    "G46": dict(char="SILENT", fixed=["SPEEDSTER", "SPEEDSTER", "ECHOING_SLASH", "ECHOING_SLASH", "BACKFLIP", "BACKFLIP", "ACROBATICS", "PREPARED", "DAGGER_THROW"],
                pool=["ESCAPE_PLAN", "POMMEL_STRIKE", "ADRENALINE"], k=1, relics=[[]], encounter=[(0, "ENCOUNTER.TOADPOLES_WEAK"), (0, "weak"), (1, "regular"), (0, "regular")],
                hp=(400, 400), focus=["SPEEDSTER", "ECHOING_SLASH", "BACKFLIP", "ACROBATICS", "PREPARED", "DAGGER_THROW", "ESCAPE_PLAN", "POMMEL_STRIKE", "ADRENALINE"]),
}


def spec_for(name: str, g: dict, i: int, rng: random.Random, cards: set[str]) -> dict:
    ch = g["char"]
    pool = [c for c in g.get("pool", []) if c in cards]
    deck = [{"id": c, "upgrades": g.get("upg", 0) if g.get("upg") is not None else rng.randrange(2)} for c in g["fixed"] if c in cards]
    for c in rng.sample(pool, min(g.get("k", 0), len(pool))):
        deck += [{"id": c, "upgrades": rng.randrange(2)}] * 2
    if g.get("starter", True):
        deck += [{"id": c, "upgrades": 0} for c in STARTER[ch]]
    act, enc = (g.get("encounter") or ENC)[i % len(g.get("encounter") or ENC)]
    hp, mhp = g.get("hp", (250, 250))
    return {"character": ch, "ascension": 10, "seed": f"P4{name}{i}", "act": act, "encounter": enc, "deck": deck,
            "relics": g["relics"][(i // 3) % len(g["relics"])], "potions": g.get("potions", []), "hp": hp, "max_hp": mhp}


def end_turn_c10(o) -> float | None:
    n = o.num_actions
    ks = [k for k in range(n) if o.cand_ints[k, 0] == ActionKind.END_TURN]
    return float(o.cand_feats[ks[0], 10]) if ks else None


def stat3(o) -> np.ndarray | None:
    js = np.flatnonzero((o.tok_ints[:, 0] == TokenKind.COMBAT_STAT) & (o.tok_feats[:, 0] == 3))
    return o.tok_feats[js[0]] if len(js) else None


class Tally:
    def __init__(self):
        self.c = Counter(); self.ex: dict[str, list] = defaultdict(list)

    def note(self, item: str, label: str, ok29: bool, ok28: bool, msg: str = ""):
        key = f"{item}｜{label}"
        self.c[key + "｜n"] += 1; self.c[key + "｜29"] += int(ok29); self.c[key + "｜28"] += int(ok28)
        if not ok29 and msg and len(self.ex[key]) < 6: self.ex[key].append(msg)


def match_foe(pre: dict, post: dict, idx: int):
    """pre 里第 idx 个敌人在 post 里是哪一个：按（ID、最大生命）相同的第几个对应（前面同 ID 的敌人死了时按下标或只按 ID 对应会错人）；对不上时退回 match_enemy。"""
    e0 = pre["enemies"][idx]
    key = (e0["id"], e0["max_hp"])
    k = sum(1 for e in pre["enemies"][:idx] if (e["id"], e["max_hp"]) == key)
    same = [e for e in post.get("enemies", []) if (e["id"], e["max_hp"]) == key]
    if len([e for e in pre["enemies"] if (e["id"], e["max_hp"]) == key]) == 1:
        return same[0] if same else None
    return match_enemy(pre, post, idx)


def ppow(st: dict) -> dict:
    return {p[0]: p[1] for p in st["player"]["powers"]}


def epow(e: dict) -> dict:
    return {p[0]: p[1] for p in e.get("powers", [])}


TRACE: set[str] = set()


def run_group(args) -> dict:
    name, combats, seed0, trace = args
    TRACE.update(trace)
    g = GROUPS[name]
    log = open(REPO / ".cache" / "obs29" / f"riders_{name}.log", "w")
    a, b = Raw(DLL, log), Raw(DLL, log)
    vocab = a.env.vocab()
    cards = {v.split(".", 1)[1] for v in vocab if v.startswith("CARD.")}
    T = Tally(); frames = 0; desync = 0
    for i in range(combats):
        rng = random.Random(seed0 + i)
        spec = spec_for(name, g, i, rng, cards)
        req = {"op": "reset", "mode": "combat", "spec": spec, "labels": True, "loop_cut": True, "stall_turns": 8}
        o = parse(a.req({**req, "obs_format": 29})); o28 = parse(b.req({**req, "obs_format": 28}))
        step = 0; pend_exec = None
        while not (o.done or o.num_actions == 0 or step >= 600):
            n = o.num_actions
            if n != o28.num_actions or not np.array_equal(o.cand_ints[:n, 0], o28.cand_ints[:n, 0]):
                desync += 1; break
            ti, tf = o.tok_ints, o.tok_feats

            def act(k):
                nonlocal o, o28, step
                o = parse(a.req({"op": "step", "action": int(k)})); o28 = parse(b.req({"op": "step", "action": int(k)})); step += 1

            if o.decision != DecisionKind.COMBAT:
                act(rng.randrange(n)); continue
            frames += 1
            pre = a.state()
            # 帧级的检查：G55 手牌外的扯碎，G56 异蛇头骨下的致命毒药
            # 合并包 5 之后格式 29 可能多出记录（复活中的怪、招式计划），两边的记录下标不再对齐：格式 28 的记录按（类型、词表 ID、区域）在自己的帧里另找，按出现先后配对
            def cards_in(ob, key, zones):
                t = ob.tok_ints
                return [j for j in np.flatnonzero((t[:, 0] == TokenKind.CARD) & np.isin(t[:, 2], zones)) if vocab[t[j, 1]] == key]
            if name == "G55":
                s29, s28 = stat3(o), stat3(o28)
                zs = [Zone.DRAW, Zone.DISCARD, Zone.EXHAUST]
                for j, j28 in zip(cards_in(o, "CARD.TEAR_ASUNDER", zs), cards_in(o28, "CARD.TEAR_ASUNDER", zs)):
                    if s29 is not None:
                        want = 1 + s29[1]
                        T.note("G55", "手牌外扯碎 7 = 1 + 子类型 3 的 1", abs(tf[j, 7] - want) < 0.5, abs(o28.tok_feats[j28, 7] - want) < 0.5, f"{name}{i}:{step} 7={tf[j, 7]:g} 要 {want:g}")
            if name == "G56":
                for j, j28 in zip(cards_in(o, "CARD.DEADLY_POISON", [Zone.HAND]), cards_in(o28, "CARD.DEADLY_POISON", [Zone.HAND])):
                    want = (5 + 2 * tf[j, 3]) + 1      # 致命毒药 5（升级 7）层，异蛇头骨 +1
                    T.note("G56", "异蛇头骨下手牌致命毒药 7 = 卡面 + 1", abs(tf[j, 7] - want) < 0.5, abs(o28.tok_feats[j28, 7] - want) < 0.5, f"{name}{i}:{step} 7={tf[j, 7]:g} 要 {want:g}")
            if name == "G13":
                exh = len(pre["exhaust"])
                if exh < 3:
                    for j, j28 in zip(cards_in(o, "CARD.PACTS_END", [Zone.HAND]), cards_in(o28, "CARD.PACTS_END", [Zone.HAND])):
                        T.note("G13", "消耗堆不足 3 张时手牌 16 = 0", tf[j, 16] == 0, o28.tok_feats[j28, 16] == 0, f"{name}{i}:{step} 16={tf[j, 16]:g}")
            kinds = o.cand_ints[:n, 0]
            plays = [k for k in range(n) if kinds[k] in (ActionKind.PLAY_CARD, ActionKind.USE_POTION)]
            ends = [k for k in range(n) if kinds[k] == ActionKind.END_TURN]

            def cname(k):
                return vocab[ti[o.cand_ints[k, 1], 1]].split(".", 1)[1] if o.cand_ints[k, 1] >= 0 else ""

            focus = [k for k in plays if cname(k) in g["focus"]]
            # G40：上一步打了灾厄相关的牌，这一步结束回合，看敌方回合里被处决的敌人数
            if pend_exec is not None and ends:
                k = ends[0]
                doomed = [(idx, e) for idx, e in enumerate(pre["enemies"]) if e["alive"] and epow(e).get("DOOM_POWER", 0) > 0]
                act(k)
                post = a.state()
                if post.get("in_combat") and not post.get("over"):
                    died = sum(1 for idx, e in doomed if (lambda x: x is None or not x["alive"] or x["max_hp"] != e["max_hp"])(match_foe(pre, post, idx)))
                    p29, p28, card = pend_exec
                    T.note("G40", "灾厄处决数（打出后立即结束回合）", round(p29) == died, round(p28) == died, f"{name}{i}:{step} {card} 预测 {p29:g} 实际 {died}")
                    if p29 > 0 or died > 0:
                        T.note("G40", "其中预测或实际有处决的", round(p29) == died, round(p28) == died, f"{name}{i}:{step} {card} 预测 {p29:g} 实际 {died}")
                pend_exec = None
                continue
            pend_exec = None
            if focus and rng.random() < 0.75: k = rng.choice(focus)
            elif plays and rng.random() < 0.75: k = rng.choice(plays)
            elif ends and rng.random() < 0.85: k = ends[0]
            else: k = rng.randrange(n)
            if kinds[k] not in (ActionKind.PLAY_CARD, ActionKind.USE_POTION):
                act(k); continue
            cf, cf28 = o.cand_feats[k].copy(), o28.cand_feats[k].copy()
            card = cname(k) if kinds[k] == ActionKind.PLAY_CARD else vocab[ti[o.cand_ints[k, 1], 1]].split(".", 1)[1]
            is_card = kinds[k] == ActionKind.PLAY_CARD
            tgt_rec = int(o.cand_ints[k, 2])
            enemies = sorted([j for j in np.flatnonzero((ti[:, 0] == TokenKind.CREATURE) & (ti[:, 2] == Zone.ENEMY)) if ti[j, 4] >= 0], key=lambda j: ti[j, 4])
            tgt = enemies.index(tgt_rec) if tgt_rec in enemies else None
            pw = ppow(pre); relics = set(pre["player"]["relics"])
            hand_ids = [c["id"] for c in pre["hand"]]
            draw_same = len(pre["draw"]) > 0 and len({(c["id"], c["up"]) for c in pre["draw"]}) == 1
            top1 = [vocab[ti[j, 1]].split(".", 1)[1] for j in np.flatnonzero((ti[:, 0] == TokenKind.CARD) & (ti[:, 2] == Zone.DRAW_TOP1))]
            top_id = pre["draw"][0]["id"] if draw_same else top1[0] if top1 else None
            act(k)
            picked = 0
            while o.decision not in (DecisionKind.COMBAT, DecisionKind.GAME_OVER) and not o.done and o.num_actions > 0 and picked < 30:
                act(rng.randrange(o.num_actions)); picked += 1
            post = a.state()
            over = not post.get("in_combat") or post.get("over")
            if over and not post.get("won"): continue
            dmg, dead = [], []
            for idx, e0 in enumerate(pre["enemies"]):
                if not e0["alive"]: dmg.append(0); dead.append(False); continue
                e1 = None if over else match_foe(pre, post, idx)
                if over or e1 is None or not e1["alive"] or e1["max_hp"] != e0["max_hp"]: dmg.append(e0["hp"] + e0["block"]); dead.append(True)
                # 掉的血加打掉的格挡（挨打后加格挡的敌人——蜷缩、怯懦——新加的格挡不算成少挨了伤害）
                else: dmg.append(e0["hp"] - e1["hp"] + max(0, e0["block"] - e1["block"])); dead.append(False)
            gain = 0 if over else post["player"]["block"] - pre["player"]["block"]
            lost = 0 if over else pre["player"]["hp"] - post["player"]["hp"]
            e29 = end_turn_c10(o) if not over and o.decision == DecisionKind.COMBAT else None
            e28 = end_turn_c10(o28) if not over and o28.decision == DecisionKind.COMBAT else None
            tag = f"{name}{i}:{step} {card}"
            if f"{name}:{i}:{step}" in TRACE:
                print(f"=== 跟踪 {tag}：候选 29 {np.round(cf, 1).tolist()}\n    28 {np.round(cf28, 1).tolist()}", flush=True)
                for key in ("player", "pets", "enemies", "hand", "draw", "exhaust"):
                    print(f"  前 {key}：{json.dumps(pre.get(key), ensure_ascii=False)[:900]}")
                    print(f"  后 {key}：{json.dumps(post.get(key), ensure_ascii=False)[:900]}", flush=True)

            def c8(item, label):
                if not over and not picked:
                    T.note(item, label + "：候选 8 对格挡增加", abs(cf[8] - gain) < 0.5, abs(cf28[8] - gain) < 0.5, f"{tag} 预测 {cf[8]:g}（28：{cf28[8]:g}）实际 {gain}")

            def c10(item, label, over_only=False):
                if not over and not picked and e29 is not None and e28 is not None:
                    w29, w28 = lost + e29, lost + e28
                    if over_only:
                        T.note(item, label + "：候选 10 不多报", cf[10] <= w29 + 0.5, cf28[10] <= w28 + 0.5, f"{tag} 预测 {cf[10]:g}（28：{cf28[10]:g}），掉 {lost} + 打出后 {e29:g}")
                    else:
                        T.note(item, label + "：候选 10 自洽", abs(cf[10] - w29) < 0.5, abs(cf28[10] - w28) < 0.5, f"{tag} 预测 {cf[10]:g}（28：{cf28[10]:g}），掉 {lost} + 打出后 {e29:g}（28：{e28:g}）")

            def c67(item, label):
                if picked: return
                if tgt is not None:
                    T.note(item, label + "：能否打死目标", round(cf[7]) == int(dead[tgt]), round(cf28[7]) == int(dead[tgt]), f"{tag} 4–7 {np.round(cf[4:8], 1).tolist()}（28：{np.round(cf28[4:8], 1).tolist()}）目标掉 {dmg[tgt]}{' 死' if dead[tgt] else ''}")
                    if not dead[tgt]:
                        T.note(item, label + "：对目标的总伤害（没打死时）", abs(cf[6] - dmg[tgt]) < 0.5, abs(cf28[6] - dmg[tgt]) < 0.5, f"{tag} 6={cf[6]:g}（28：{cf28[6]:g}）实际 {dmg[tgt]}")
                else:
                    kills = sum(dead)
                    T.note(item, label + "：打死几个", round(cf[7]) == kills, round(cf28[7]) == kills, f"{tag} 4–7 {np.round(cf[4:8], 1).tolist()}（28：{np.round(cf28[4:8], 1).tolist()}）掉 {dmg}")
                    if kills == 0:
                        T.note(item, label + "：总伤害（没打死时）", abs(cf[6] - sum(dmg)) < 0.5, abs(cf28[6] - sum(dmg)) < 0.5, f"{tag} 6={cf[6]:g}（28：{cf28[6]:g}）实际 {sum(dmg)}")

            fnp = "FEEL_NO_PAIN_POWER" in pw
            nfoe = sum(1 for e in pre["enemies"] if e["alive"])
            nlab = "（单敌）" if nfoe == 1 else "（多敌）"
            # ---- 各组的指标
            if name in ("G01", "G01T") and card in ("HAVOC", "THRASH", "STOKE") and (fnp or "CHARONS_ASHES" in relics or "JUGGERNAUT_POWER" in pw):
                lab = card + (("（顶牌未知）" if top_id is None else "（顶牌是浩劫，连锁）" if top_id == "HAVOC" else "（顶牌已知）") if card == "HAVOC" else "")
                if fnp: c8("G01", lab + "，无惧疼痛")
                c10("G01", lab)
                if "CHARONS_ASHES" in relics: c67("G01", lab + "，卡戎之灰")
            if name == "G02" and card in ("SECOND_WIND", "FIEND_FIRE", "STOKE"):
                status = any(h in ("BURN", "WOUND", "DAZED", "SLIMED", "DECAY", "REGRET", "TOXIC", "INFECTION") for h in hand_ids if h != card)
                c10("G02", card + ("（手里有状态牌）" if status else "（没有状态牌）"))
            if name == "G13" and card == "PACTS_END":
                exh = len(pre["exhaust"])
                c67("G13", f"契约终结（消耗堆{'不足 3' if exh < 3 else '≥ 3'}）")
            if name == "G15":
                if card in ("COLOSSUS", "FIGHT_ME", "VULNERABLE_POTION"): c10("G15", card)
                elif "COLOSSUS_POWER" in pw and card in VUL_CARDS: c10("G15", "已有巨像时上易伤的牌")
            if name == "G15R" and is_card and "MONARCHS_GAZE_POWER" in pw and card != "MONARCHS_GAZE": c10("G15R", "王之凝视在场的攻击" + nlab)
            if name == "G15N" and is_card:
                if card in ("DEBILITATE", "MISERY"): c10("G15N", card)
                elif "UNDYING_SIGIL" in relics and card in DOOM_CARDS: c10("G15N", "不死符文：灾厄牌")
            if name == "G15S" and is_card and "ENVENOM_POWER" in pw and card != "ENVENOM": c10("G15S", "涂毒在场的攻击" + nlab)
            if name == "G31" and is_card:
                atk = card in ("TWIN_STRIKE", "POMMEL_STRIKE", "STRIKE_IRONCLAD", "IRON_WAVE", "SWORD_BOOMERANG")
                if "DAUGHTER_OF_THE_WIND" in relics and atk: c8("G31", "风的女儿：攻击牌")
                if "LOST_WISP" in relics and card in ("INFLAME", "DEMON_FORM"): c10("G31", "迷失鬼火：能力牌")
                if "FORGOTTEN_SOUL" in relics and card in ("TRUE_GRIT", "BURNING_PACT") and len([e for e in pre["enemies"] if e["alive"]]) == 1: c10("G31", "遗忘之魂：消耗牌（单敌）")
            if name == "G66" and card in ("HEMOKINESIS", "BREAKTHROUGH"):
                hp0, mhp = pre["player"]["hp"], pre["player"]["max_hp"]
                loss = 2 if card == "HEMOKINESIS" else 1
                if hp0 > mhp * 0.5 >= hp0 - loss:
                    c67("G66", card + "（自伤跨过一半）")
            if name == "G67" and card == "WHISTLE": c10("G67", "口哨")
            if name == "G72" and card == "CALTROPS": c10("G72", "铁蒺藜（荆棘结算待包 1、3 合并）")
            g37 = card in ("PILLAR_OF_CREATION", "SHROUD", "FISTICUFFS")      # 打出的牌本身的候选 8 归包 3（G37、G32），“在场”的几项不算它们
            if name == "G38" and is_card and not g37:
                if "PILLAR_OF_CREATION_POWER" in pw:
                    c8("G38", "创世之柱在场")
                    c10("G38", "创世之柱在场", over_only=True)
                if "CHILD_OF_THE_STARS_POWER" in pw: c8("G38", "群星之子在场")
                if "BLACK_HOLE_POWER" in pw: c67("G38", "黑洞在场")
            if name == "G38" and card == "PILLAR_OF_CREATION": c8("G37", "打出创世之柱（候选 8 的基数归包 3，合并后量）")
            if name == "G39" and card == "SHROUD": c8("G37", "打出厄运之衣（候选 8 的基数归包 3，合并后量）")
            if name == "G39" and is_card and not g37:
                if "SHROUD_POWER" in pw:
                    c8("G39", "厄运之衣在场"); c10("G39", "厄运之衣在场")
                if "SLEIGHT_OF_FLESH_POWER" in pw: c67("G39", "血肉戏法在场"); c10("G39", "血肉戏法在场")
                if "SPIRIT_OF_ASH_POWER" in pw or "DANSE_MACABRE_POWER" in pw: c8("G39", "灰烬之灵或死亡之舞在场")
            if name in ("G39", "G40", "G41") and is_card and (card in DOOM_CARDS or any(epow(e).get("DOOM_POWER", 0) > 0 for e in pre["enemies"] if e["alive"])):
                if not over and not picked and o.decision == DecisionKind.COMBAT:
                    pend_exec = (float(cf[16]), 0.0, card)
            if name == "G41" and is_card:
                if card in ("DIRGE", "SPUR"): c10("G41", card)
                if card == "BONE_SHARDS" and not (pre.get("pets") and pre["pets"][0]["alive"]): c8("G41", "碎骨（奥斯提不在）")
                if card in ("POKE", "HIGH_FIVE", "SIC_EM", "FLATTEN") and any(epow(e).get("THORNS_POWER", 0) > 0 for e in pre["enemies"] if e["alive"]): c10("G41", "奥斯提出手打荆棘敌人")
            if name.startswith("G42") and is_card and card in g["focus"]:
                c10("G42", f"{name[3:]} 组 {card}{nlab}")
            if name == "G45" and card in ("CALCULATED_GAMBLE", "STORM_OF_STEEL", "SHADOW_STEP"):
                sly = any(h in ("TACTICIAN", "REFLEX", "MEMENTO_MORI") for h in hand_ids)
                rel = "、".join(sorted(relics & {"TINGSHA", "TOUGH_BANDAGES"})) or "无遗物"
                lab = f"整手弃（{'有奇巧' if sly else '无奇巧'}，{rel}）{nlab if 'TINGSHA' in relics else ''}"
                c8("G45", lab)
                if sly or "TINGSHA" in relics: c67("G45", lab)
            if name == "G46" and is_card:
                if "SPEEDSTER_POWER" in pw and card in ("BACKFLIP", "ACROBATICS", "PREPARED", "DAGGER_THROW", "ESCAPE_PLAN", "POMMEL_STRIKE", "ADRENALINE"): c67("G46", "速行者：抽牌牌")
                if card == "ECHOING_SLASH": c67("G46", "回响斩击")
    a.close(); b.close()
    return {"c": dict(T.c), "ex": dict(T.ex), "frames": frames, "desync": desync, "group": name}


TARGET = {"G01": 0.95, "G02": 0.95, "G13": 1.0, "G15": 0.98, "G31": 0.98, "G55": 1.0, "G56": 1.0, "G66": 1.0, "G67": 1.0, "G72": 0.95,
          "G15R": 0.95, "G15N": 0.95, "G15S": 0.95, "G37": 0.95, "G38": 0.95, "G39": 0.95, "G40": 0.95, "G41": 0.95, "G42": 0.95, "G45": 0.95, "G46": 0.98}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--groups", default="all"); ap.add_argument("--combats", type=int, default=40)
    ap.add_argument("--workers", type=int, default=4, help="并行的组数（每组两个引擎进程）"); ap.add_argument("--seed", type=int, default=4000)
    ap.add_argument("--dump", default="")
    ap.add_argument("--trace", default="", help="组:场次:步数（例子里“组名场次:步数”的步数），逗号分隔，打印这一步出牌前后的完整状态")
    args = ap.parse_args()
    (REPO / ".cache" / "obs29").mkdir(parents=True, exist_ok=True)
    names = list(GROUPS) if args.groups == "all" else args.groups.split(",")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        res = list(ex.map(run_group, [(nm, args.combats, args.seed, set(filter(None, args.trace.split(",")))) for nm in names]))
    c = Counter(); exs: dict[str, list] = {}
    for r in res:
        c.update(r["c"]); exs.update({k: v for k, v in r["ex"].items() if v})
    rows = []
    for key in sorted({k.rsplit("｜", 1)[0] for k in c}):
        n = c[key + "｜n"]; item = key.split("｜")[0]
        r29, r28 = c[key + "｜29"] / n, c[key + "｜28"] / n
        tgt = 0.98 if "不多报" in key else TARGET.get(item, 0.95)
        rows.append({"条目": item, "指标": key.split("｜", 1)[1], "帧数": n, "28 一致率": round(r28, 4), "29 一致率": round(r29, 4), "目标": tgt, "通过": r29 >= tgt})
    for r in rows:
        print(f"{r['条目']:4s} {r['指标']:<44s} n={r['帧数']:5d}  28={r['28 一致率']:.3f}  29={r['29 一致率']:.3f}  目标 {r['目标']:.2f}  {'通过' if r['通过'] else '不通过'}")
    print(json.dumps({"frames": {r["group"]: r["frames"] for r in res}, "desync": {r["group"]: r["desync"] for r in res if r["desync"]}, "seconds": round(time.time() - t0, 1)}, ensure_ascii=False))
    if args.dump:
        d = Path(args.dump); d.mkdir(parents=True, exist_ok=True)
        (d / "riders.json").write_text(json.dumps({"rows": rows, "examples": exs, "frames": {r["group"]: r["frames"] for r in res}}, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
