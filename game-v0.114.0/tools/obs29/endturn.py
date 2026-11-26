"""格式 29 包 1（回合预演，R3-格式29规格.md 5.1）的核对：格式 28 与 29 对真实结果的一致率，按条目出表。

两个引擎进程同种子锁步，一个 reset 带 obs_format 28、一个带 29；动作取自格式 29 的帧（偏向出牌的随机策略，同 tools/clone/common.py 的 pick_bias），
观测格式不影响游戏，两边局面逐步相同。真实结果来自 op state。每个条目给一组单场战斗 spec 与一个指标：

  结束回合 c10   结束回合候选的 10（最坏口径的预计掉血）对敌方回合实际掉的血（到下一个决策帧为止的生命变化）
  玩家 17        玩家记录 17（期望口径的预计掉血）对同上
  敌人 16        单个敌人时，敌人记录 16、17 推出的预计掉血（生命 − 17 + max(0, 格挡 − 16)）对敌方回合之后敌人实际掉的血
  出牌自洽       出牌、用药水前预测的候选 10 = 这一步实际掉的血 + 打出后结束回合候选的 10（同 tools/obs_v23_check.py）
  致死           结束回合后玩家死了的帧：玩家 17（或结束回合 c10）≥ 当前生命

每条只看“这一条可能起作用”的帧（when），格式 28、29 各算一遍一致率（差 < 0.5 为一致）。依赖别包输出的条目（G02、G27、G28、G21、G50 等）
在别包合并之前，格式 29 的这几项输入为默认值，量出来与格式 28 相同属正常（报告里注明“合并后再量”）。

  python -I tools/obs29/endturn.py --new engine/Sts2Headless/bin/Release/net9.0/Sts2Headless.dll --combats 40 --jobs 4 --dump 输出.json
  python -I tools/obs29/endturn.py --new ... --items G03,G11 --combats 10
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "tools"))

import numpy as np  # noqa: E402

from sts2env.protocol import ActionKind, DecisionKind, TokenKind, parse  # noqa: E402
from obs_v21_check import Raw, enemy_records, match_enemy  # noqa: E402

IRON = [{"id": "STRIKE_IRONCLAD", "upgrades": 0}] * 5 + [{"id": "DEFEND_IRONCLAD", "upgrades": 0}] * 4 + [{"id": "BASH", "upgrades": 0}]
SILENT = [{"id": "STRIKE_SILENT", "upgrades": 0}] * 4 + [{"id": "DEFEND_SILENT", "upgrades": 0}] * 4 + [{"id": "NEUTRALIZE", "upgrades": 0}]
ATTACKS = {"STRIKE_IRONCLAD", "BASH", "HOWL_FROM_BEYOND", "TWIN_STRIKE", "POMMEL_STRIKE", "HEADBUTT", "IRON_WAVE", "THUNDERCLAP", "BREAKTHROUGH",
           "HEMOKINESIS", "STRIKE_SILENT", "NEUTRALIZE", "POISONED_STAB", "STRIKE_REGENT", "STRIKE_NECROBINDER"}
SINGLE = ["AXEBOTS_NORMAL", "CUBEX_CONSTRUCT_NORMAL", "FOGMOG_NORMAL", "FROG_KNIGHT_NORMAL", "MAWLER_NORMAL", "PUNCH_CONSTRUCT_NORMAL",
          "SEWER_CLAM_NORMAL", "VINE_SHAMBLER_NORMAL", "SLIMED_BERSERKER_NORMAL", "BYRDONIS_ELITE", "BYGONE_EFFIGY_ELITE", "CEREMONIAL_BEAST_BOSS",
          "MECHA_KNIGHT_ELITE", "HUNTER_KILLER_NORMAL", "GLOBE_HEAD_NORMAL"]


def cards(*xs) -> list[dict]:
    out = []
    for x in xs:
        name, _, n = x.partition("*")
        up = name.endswith("+")
        out += [{"id": name.rstrip("+"), "upgrades": int(up)}] * (int(n) if n else 1)
    return out


def spec(ch: str, seed: str, act: int, enc: str, deck: list[dict], relics: list[str] | None = None, hp: int = 200, potions: list[str] | None = None) -> dict:
    s = {"character": ch, "ascension": 10, "seed": seed, "act": act, "encounter": enc, "deck": deck, "relics": relics or [], "hp": hp, "max_hp": max(hp, 80)}
    if potions is not None: s["potions"] = potions
    return s


# ------------------------------------------------------------------ 各条目的局面与指标

def sp_g03(i):
    encs = [(0, "elite"), (1, "elite"), (2, "elite"), (0, "boss"), (1, "boss"), (2, "boss")]
    act, enc = encs[i % len(encs)]
    return spec("IRONCLAD", f"P1G03_{i}", act, enc, IRON + cards("ASCENDERS_BANE"), ["BEATING_REMNANT"], 200)


def sp_g04(i):
    enc = ["THE_KIN_BOSS", "QUEEN_BOSS", "FABRICATOR_NORMAL"][i % 3]
    act = {"THE_KIN_BOSS": 0, "QUEEN_BOSS": 2, "FABRICATOR_NORMAL": 2}[enc]
    if i % 3 == 0:
        deck = IRON + cards("ASCENDERS_BANE", "DAZED*2", "FEEL_NO_PAIN", "JUGGERNAUT*2", "STONE_ARMOR*2", "BLOODLETTING")
        return spec("IRONCLAD", f"P1G04_{i}", act, enc, deck, ["CHARONS_ASHES"], 250)
    deck = SILENT + cards("DEADLY_POISON*3", "NOXIOUS_FUMES*2", "BUBBLE_BUBBLE*2", "POISONED_STAB*2", "ACCELERANT", "FOOTWORK")
    return spec("SILENT", f"P1G04_{i}", act, enc, deck, ["SNECKO_SKULL"], 250)


def sp_g11(i):
    deck = IRON + cards("HOWL_FROM_BEYOND*2", "BURNING_PACT*2", "TRUE_GRIT+*2", "STAMPEDE*2", "RAGE")
    enc = SINGLE[i % len(SINGLE)]
    return spec("IRONCLAD", f"P1G11_{i}", i % 3, enc, deck, [], 250)


def sp_g12(i):
    deck = IRON + cards("ASCENDERS_BANE", "FEEL_NO_PAIN*2", "JUGGERNAUT*2", "STONE_ARMOR*2", "DAZED")
    enc = SINGLE[i % len(SINGLE)]
    return spec("IRONCLAD", f"P1G12_{i}", i % 3, enc, deck, ["ORICHALCUM"] if i % 2 else [], 250)


def sp_g23(i):
    enc = ["TEST_SUBJECT_BOSS", "SOUL_FYSH_BOSS"][i % 2]
    deck = IRON + cards("ASCENDERS_BANE", "DAZED*2", "FEEL_NO_PAIN", "JUGGERNAUT*2", "STONE_ARMOR")
    return spec("IRONCLAD", f"P1G23_{i}", 2 if enc == "TEST_SUBJECT_BOSS" else 1, enc, deck, ["CHARONS_ASHES"], 300)


def sp_g27(i):
    enc = ["SLITHERING_STRANGLER_NORMAL", "KNIGHTS_ELITE"][i % 2]
    deck = IRON + cards("TWIN_STRIKE*2", "POMMEL_STRIKE*2", "HEADBUTT", "IRON_WAVE")
    return spec("IRONCLAD", f"P1G27_{i}", 1 if enc == "SLITHERING_STRANGLER_NORMAL" else 2, enc, deck, [], 200)


def sp_g28(i):
    return spec("IRONCLAD", f"P1G28_{i}", 1, "SCROLLS_OF_BITING_NORMAL", IRON + cards("THUNDERCLAP*2", "BREAKTHROUGH"), [], 200)


def sp_g29(i):
    enc = ["THE_OBSCURA_NORMAL", "QUEEN_BOSS"][i % 2]
    return spec("IRONCLAD", f"P1G29_{i}", 2, enc, IRON + cards("SHRUG_IT_OFF*2", "TRUE_GRIT"), [], 250)


def sp_g32(i):
    k = i % 3
    if k == 0:      # THE_GAMBIT（无色），防御少
        deck = cards("STRIKE_IRONCLAD*6", "DEFEND_IRONCLAD", "BASH", "THE_GAMBIT*2", "POMMEL_STRIKE*2")
        return spec("IRONCLAD", f"P1G32_{i}", i % 3, ["elite", "boss", "regular"][i % 3], deck, [], 120)
    if k == 1:      # 恶魔之舌 + 自伤牌 + 灼伤
        deck = IRON + cards("BLOODLETTING*2", "OFFERING", "HEMOKINESIS*2")
        return spec("IRONCLAD", f"P1G32_{i}", 1, "MECHA_KNIGHT_ELITE", deck, ["DEMON_TONGUE"], 200)
    deck = IRON + cards("INFERNO*2", "BLOODLETTING", "HEMOKINESIS")       # 狱火 + 灼伤
    return spec("IRONCLAD", f"P1G32_{i}", 1, "MECHA_KNIGHT_ELITE", deck, [], 250)


def sp_g36(i):
    deck = cards("NEUROSURGE*2", "DEFEND_NECROBINDER+*9", "STRIKE_NECROBINDER*2")
    return spec("NECROBINDER", f"P1G36_{i}", i % 3, ["regular", "elite", "boss"][i % 3], deck, [], 14 + 6 * (i % 4))


def sp_g36h(i):
    """合并后核对（G36 的灾厄记录，包 4 的 DoomInfo 读包 1 的 HandLoss）：手里有悔恨（不可格挡的回合末自伤），灾厄结算前先掉血。"""
    deck = cards("NEUROSURGE*2", "DEFEND_NECROBINDER+*7", "REGRET*2", "STRIKE_NECROBINDER*2")
    return spec("NECROBINDER", f"P1G36H_{i}", 0, ["weak", "regular"][i % 2], deck, [], 14 + 6 * (i % 4))


def sp_g51(i):
    deck = [{"id": "STRIKE_REGENT", "upgrades": 0}] * 4 + [{"id": "DEFEND_REGENT", "upgrades": 0}] * 4 + cards("REFLECT*3", "REFLECT+*2")
    return spec("REGENT", f"P1G51_{i}", i % 3, ["regular", "elite", "boss"][i % 3], deck, ["BRONZE_SCALES"] if i % 2 else [], 200)


def sp_g21(i):
    k = i % 6
    if k == 4:      # 静默猎手：闪躲翻滚（下回合格挡）、残影，释
        deck = SILENT + cards("DODGE_AND_ROLL*3", "BLUR*2", "FOOTWORK")
        return spec("SILENT", f"P1G21_{i}", i % 3, ["regular", "elite", "boss"][i % 3], deck, ["SAI"], 200)
    if k == 5:      # 储君的托里克坚韧；铁甲的赤红斗篷混在铁甲牌组里
        deck = [{"id": "STRIKE_REGENT", "upgrades": 0}] * 4 + [{"id": "DEFEND_REGENT", "upgrades": 0}] * 4 + cards("TORIC_TOUGHNESS*3")
        return spec("REGENT", f"P1G21_{i}", i % 3, ["regular", "elite", "boss"][i % 3], deck, ["SELF_FORMING_CLAY"], 200)
    deck = IRON + cards("BARRICADE*2", "SHRUG_IT_OFF*2", "IMPERVIOUS", "CRIMSON_MANTLE")
    rel = [["SELF_FORMING_CLAY"], ["HORN_CLEAT"], ["STURDY_CLAMP"], ["SELF_FORMING_CLAY", "CAPTAINS_WHEEL"]][k]
    return spec("IRONCLAD", f"P1G21_{i}", i % 3, ["regular", "elite", "boss"][i % 3], deck, rel, 200)


ITEMS = {
    # 条目：(局面, [(指标, 帧条件, 目标, 说明)])
    "G03": (sp_g03, [("et_c10", "remnant", 0.97, "律动残余：结束回合 c10 对实际"), ("self", "remnant", 0.95, "律动残余：出牌自洽")]),
    "G04": (sp_g04, [("et_p17", "won_after", 1.0, "主要敌人被被动或毒打死、战斗当场结束的帧：玩家 17 对实际"),
                     ("et_c10", "won_after", 1.0, "同上：结束回合 c10 对实际")]),
    "G11": (sp_g11, [("e16", "howl_stampede", 0.95, "消耗堆有嚎叫或有践踏：敌人 16 对回合末实际")]),
    "G12": (sp_g12, [("e16", "juggernaut", 0.95, "有势不可当：敌人 16 对回合末实际")]),
    "G23": (sp_g23, [("e16", "foe_intangible", 1.0, "敌人有无实体、回合末有被动伤害：敌人 16 对实际")]),
    "G27": (sp_g27, [("self_kill_applier", "constrict", 0.98, "打死缠绕/魔法炸弹的施加者：出牌自洽（需包 3 的 CandFx.Dead）")]),
    "G28": (sp_g28, [("et_p17", "always", 0.95, "纸片割伤：玩家 17 对实际（最大生命损失见生物 28，包 5 写列）"), ("maxhp", "papercuts", 0.95, "生物 28 对最大生命实际减少（包 5 写列后）")]),
    "G29": (sp_g29, [("et_p17", "ally_buff", 0.95, "有队友加力量的招式：玩家 17 对实际"), ("et_c10", "ally_buff", 0.95, "同上：结束回合 c10")]),
    "G32": (sp_g32, [("dead17", "gambit", 1.0, "THE_GAMBIT：结束回合即死的帧，玩家 17 = 生命"),
                     ("et_c10", "tongue", 0.95, "恶魔之舌：结束回合 c10 对实际"), ("self", "tongue", 0.95, "恶魔之舌：出牌自洽（需包 3 的 CandFx.SelfLoss）"),
                     ("e16", "inferno", 0.90, "狱火：敌人 16 对实际")]),
    "G36": (sp_g36, [("dead17", "doom", 1.0, "玩家灾厄：结束回合即死的帧，玩家 17 = 生命"), ("dead_c10", "doom", 1.0, "同上：结束回合 c10 = 生命"),
                     ("doomrec", "doom", 1.0, "玩家灾厄能力记录 12（会被处决）对结束回合后死亡")]),
    "G36H": (sp_g36h, [("dead17", "doom", 1.0, "手里有悔恨：结束回合即死的帧，玩家 17 = 生命"),
                       ("doomrec", "doom", 1.0, "手里有悔恨：玩家灾厄能力记录 12（会被处决）对结束回合后死亡")]),
    "G51": (sp_g51, [("et_p17", "reflect", 0.97, "倒映：玩家 17 对实际"), ("et_c10", "reflect", 0.97, "倒映：结束回合 c10 对实际")]),
    "G21": (sp_g21, [("nextblock", "always", 0.95, "生物 29 对下一回合第一帧玩家的格挡（包 5 写列后）")]),
}


# ------------------------------------------------------------------ 帧与状态的读取

def player_rec(o) -> int | None:
    ti, tf = o.tok_ints, o.tok_feats
    js = np.flatnonzero((ti[:, 0] == TokenKind.CREATURE) & (tf[:, 5] > 0))
    return int(js[0]) if len(js) else None


def end_idx(o) -> int | None:
    ks = [k for k in range(o.num_actions) if o.cand_ints[k, 0] == ActionKind.END_TURN]
    return ks[0] if ks else None


def pw(e: dict) -> dict:
    return {p[0]: p[1] for p in e.get("powers", [])}


def hp_of(st: dict) -> int:
    if st.get("in_combat"): return int(st["player"]["hp"])
    return int(st.get("run", {}).get("hp", 0))


def maxhp_of(st: dict) -> int:
    if st.get("in_combat"): return int(st["player"]["max_hp"])
    return int(st.get("run", {}).get("max_hp", 0))


def cond(name: str, pre: dict, post: dict | None, o) -> bool:
    """帧条件（结束回合的帧；post 为结束回合之后的状态）。"""
    p = pre["player"]; ppw = pw(p)
    foes = [e for e in pre["enemies"] if e["alive"]]
    if name == "always": return True
    if name == "remnant": return "BEATING_REMNANT" in p["relics"]
    if name == "won_after": return post is not None and bool(post.get("won")) and hp_of(post) > 0
    if name == "howl_stampede": return any(c["id"] == "HOWL_FROM_BEYOND" for c in pre["exhaust"]) or "STAMPEDE_POWER" in ppw
    if name == "juggernaut": return "JUGGERNAUT_POWER" in ppw
    if name == "foe_intangible": return any("INTANGIBLE_POWER" in pw(e) for e in foes)
    if name == "constrict": return "CONSTRICT_POWER" in ppw or "MAGIC_BOMB_POWER" in ppw
    if name == "papercuts": return any("PAPER_CUTS_POWER" in pw(e) for e in foes)
    if name == "ally_buff":
        return any(e["id"] in ("QUEEN", "THE_OBSCURA") and any(m in str(e.get("move", "")) for m in ("BURN_BRIGHT", "SAIL")) for e in foes)
    if name == "gambit": return "THE_GAMBIT_POWER" in ppw
    if name == "tongue": return "DEMON_TONGUE" in p["relics"]
    if name == "inferno": return "INFERNO_POWER" in ppw
    if name == "doom": return "DOOM_POWER" in ppw
    if name == "reflect": return "REFLECT_POWER" in ppw
    raise KeyError(name)


def doom_rec12(o, doom_id: int) -> float | None:
    """玩家身上灾厄能力记录的 12（格式 22 起：会不会在玩家回合结束时被处决）；没有这条记录为 None。"""
    pj = player_rec(o)
    if pj is None: return None
    ti = o.tok_ints
    js = np.flatnonzero((ti[:, 0] == TokenKind.POWER) & (ti[:, 1] == doom_id) & (ti[:, 3] == pj))
    return float(o.tok_feats[js[0], 12]) if len(js) else None


def foe_pred(o, pre: dict) -> list[tuple[int, float]]:
    """每个活着的敌人：(在 pre['enemies'] 里的下标, 预计回合末掉的生命 = 生命 − 17 + max(0, 格挡 − 16))。"""
    out = []
    js = enemy_records(o)
    ens = pre["enemies"]
    if len(js) != len(ens): return out
    for idx, (j, e) in enumerate(zip(js, ens)):
        if not e["alive"]: continue
        f16, f17 = float(o.tok_feats[j, 16]), float(o.tok_feats[j, 17])
        out.append((idx, e["hp"] - f17 + max(0.0, e["block"] - f16)))
    return out


# ------------------------------------------------------------------ 主循环

def run_item(args_t) -> dict:
    item, n, dll, seed0, examples = args_t
    mk, metrics = ITEMS[item]
    logd = REPO / ".cache" / "obs29_endturn"; logd.mkdir(parents=True, exist_ok=True)
    a = Raw(Path(dll), open(logd / f"{item}_29.log", "w"))
    b = Raw(Path(dll), open(logd / f"{item}_28.log", "w"))
    c = Counter(); ex: dict[str, list] = {}
    vocab = a.req({"op": "vocab"}).decode("utf-8").split("\n")
    doom_id = vocab.index("POWER.DOOM_POWER") if "POWER.DOOM_POWER" in vocab else -1

    def note(key: str, ok29: bool, ok28: bool, msg: str, vals: tuple[float, float, float] | None = None):
        c[f"{key}|n"] += 1; c[f"{key}|29"] += int(ok29); c[f"{key}|28"] += int(ok28)
        if vals is not None:        # 随机的局面只能比期望：累计预测与实际，看平均偏差
            c[f"{key}|s29"] += vals[0]; c[f"{key}|s28"] += vals[1]; c[f"{key}|sact"] += vals[2]
        if not ok29 and len(ex.setdefault(key, [])) < examples: ex[key].append(msg)

    t0 = time.time()
    for i in range(n):
        sp = mk(i)
        rng = random.Random(seed0 * 7919 + i)
        req = {"op": "reset", "mode": "combat", "spec": sp, "loop_cut": True, "stall_turns": 12}
        try:
            o = parse(a.req({**req, "obs_format": 29})); o28 = parse(b.req({**req, "obs_format": 28}))
        except Exception as e:  # noqa: BLE001
            c["reset 失败"] += 1; ex.setdefault("reset 失败", []).append(f"{sp['encounter']}: {e}"[:200]); continue
        step = 0
        while not (o.done or o.num_actions == 0 or step >= 400):
            n_a = o.num_actions
            if o.decision != DecisionKind.COMBAT:
                k = rng.randrange(n_a)
                o = parse(a.req({"op": "step", "action": k})); o28 = parse(b.req({"op": "step", "action": k})); step += 1
                continue
            c["战斗帧"] += 1
            if o28.num_actions != n_a or not np.array_equal(o28.cand_ints[:n_a], o.cand_ints[:n_a]): c["候选与格式 28 不同的帧（应为 0）"] += 1
            kinds = o.cand_ints[:n_a, 0]
            plays = [k for k in range(n_a) if kinds[k] in (ActionKind.PLAY_CARD, ActionKind.USE_POTION)]
            ke = end_idx(o)
            k = rng.choice(plays) if plays and rng.random() < 0.75 else (ke if ke is not None and rng.random() < 0.85 else rng.randrange(n_a))
            pre = a.state()
            tag = f"{item}#{i} {sp['encounter']} 第 {step} 步"
            if kinds[k] == ActionKind.END_TURN:
                pj, pj28 = player_rec(o), player_rec(o28)
                p17, p17b = (float(o.tok_feats[pj, 17]) if pj is not None else 0.0), (float(o28.tok_feats[pj28, 17]) if pj28 is not None else 0.0)
                c10, c10b = float(o.cand_feats[k, 10]), float(o28.cand_feats[k, 10])
                fp, fp28 = foe_pred(o, pre), foe_pred(o28, pre)
                y28, y29 = (float(o.tok_feats[pj, 28]), float(o.tok_feats[pj, 29])) if pj is not None and o.tok_feats.shape[1] >= 30 else (0.0, 0.0)
                r12, r12b = doom_rec12(o, doom_id), doom_rec12(o28, doom_id)
                o = parse(a.req({"op": "step", "action": k})); o28 = parse(b.req({"op": "step", "action": k})); step += 1
                post = a.state()
                lost = pre["player"]["hp"] - hp_of(post)
                if post.get("won") and "BURNING_BLOOD" in pre["player"]["relics"]: lost += 6     # 燃烧之血在战斗胜利后回 6 血，不属于敌方回合
                dead = hp_of(post) <= 0 or (post.get("over") and not post.get("won"))
                hp0 = pre["player"]["hp"]
                ppw = pw(pre["player"])
                start_fx = {"REGEN_POWER", "DEMISE_POWER", "POISON_POWER"} & set(ppw)
                for met, cn, _, _ in metrics:
                    try:
                        if not cond(cn, pre, post, o): continue
                    except KeyError:
                        continue
                    key = f"{met}:{cn}"
                    if met in ("et_c10", "et_p17") and not start_fx:
                        x29, x28 = (c10, c10b) if met == "et_c10" else (p17, p17b)
                        if dead: ok29, ok28 = x29 >= hp0 - 0.5, x28 >= hp0 - 0.5
                        else: ok29, ok28 = abs(x29 - lost) < 0.5, abs(x28 - lost) < 0.5
                        note(key, ok29, ok28, f"{tag}：29 预测 {x29:g}，28 预测 {x28:g}，实际 {lost}{'（死）' if dead else ''}；生命 {hp0} 格挡 {pre['player']['block']}；"
                                              f"玩家能力 {sorted(ppw)}；敌人 {[(e['id'], e['hp'], e.get('move')) for e in pre['enemies'] if e['alive']]}")
                    elif met in ("dead17", "dead_c10") and dead:
                        x29, x28 = (p17, p17b) if met == "dead17" else (c10, c10b)
                        note(key, x29 >= hp0 - 0.5, x28 >= hp0 - 0.5, f"{tag}：死了；29 预测 {x29:g}，28 {x28:g}，生命 {hp0}；能力 {ppw}")
                    elif met == "e16" and not post.get("over"):
                        alive = [e for e in pre["enemies"] if e["alive"]]
                        if len(alive) != 1: continue
                        # 践踏从手里随机挑攻击牌：手里的攻击牌不全一样时只有期望对得上，单列（看平均偏差）
                        atk = {(x["id"], x["up"]) for x in pre["hand"] if x["id"] in ATTACKS}
                        rnd = "STAMPEDE_POWER" in ppw and len(atk) > 1
                        key2 = key + ("（践踏随机挑牌，只比期望）" if rnd else "（确定）")
                        d29, d28 = dict(fp), dict(fp28)
                        for idx, e0 in enumerate(pre["enemies"]):
                            if not e0["alive"] or idx not in d29 or idx not in d28: continue
                            e1 = match_enemy(pre, post, idx)
                            if e1 is None or e1["max_hp"] != e0["max_hp"]: continue
                            act = e0["hp"] - (e1["hp"] if e1["alive"] else 0)
                            # 狱火在下个玩家回合开始时先对自己造成伤害（AfterPlayerTurnStart），这一下又触发一次对全体的狱火，发生在下一个决策帧之前、
                            # 不属于回合末被动。它先打在敌人这回合新得的格挡上：下一帧敌人还有格挡就是全被挡住（不扣）；格挡为 0 时，
                            # 这一招不带防御意图（没新得格挡）就扣掉层数，带防御意图的分不清，跳过
                            if "INFERNO_POWER" in ppw and e1["alive"] and not dead and e1["block"] <= 0:
                                if any(str(x[0]).startswith("Defend") or "Block" in str(x[0]) for x in e0.get("intents", [])): continue
                                act -= ppw["INFERNO_POWER"]
                            note(key2, abs(d29[idx] - act) < 0.5, abs(d28[idx] - act) < 0.5,
                                 f"{tag} {e0['id']}：29 预测掉 {d29[idx]:g}，28 {d28[idx]:g}，实际 {act}；生命 {e0['hp']} 格挡 {e0['block']} 能力 {sorted(pw(e0))}；"
                                 f"玩家能力 {sorted(ppw)}；消耗堆 {[x['id'] for x in pre['exhaust']]}；手牌 {[x['id'] for x in pre['hand']]}",
                                 (d29[idx], d28[idx], act) if rnd else None)
                    elif met == "maxhp" and not dead:
                        act = pre["player"]["max_hp"] - maxhp_of(post)
                        note(key, abs(y28 - act) < 0.5, act == 0, f"{tag}：生物 28 = {y28:g}，实际最大生命少了 {act}")
                    elif met == "doomrec" and r12 is not None and r12b is not None:
                        # 玩家的灾厄在玩家回合结束时结算（手牌的回合末自伤之后、敌人出手之前）。只比死因分得清的帧：敌人这回合的攻击合计不超过现有格挡，
                        # 悔恨（不可格挡，每张按回合末手牌数）也不可能先把人打死（生命 > 悔恨张数 × 手牌数）。这些帧里死了就是被灾厄处决，12 应与之一致
                        atk = sum(int(x[1]) * int(x[2]) for e in pre["enemies"] if e["alive"] for x in e.get("intents", []))
                        nreg = sum(1 for x in pre["hand"] if x["id"] == "REGRET")
                        if atk > pre["player"]["block"] or hp0 <= nreg * len(pre["hand"]): continue
                        regret = nreg > 0
                        key2 = key + ("（手里有悔恨）" if regret else "（手里没有悔恨）")
                        note(key2, (r12 >= 0.5) == bool(dead), (r12b >= 0.5) == bool(dead),
                             f"{tag}：灾厄 {ppw.get('DOOM_POWER')}，生命 {hp0}，格挡 {pre['player']['block']}；29 记录 12 = {r12:g}，28 = {r12b:g}，"
                             f"{'死了' if dead else '没死'}；玩家 17 29 {p17:g}；手牌 {[x['id'] for x in pre['hand']]}")
                    elif met == "nextblock" and not dead and post.get("in_combat") and o.decision == DecisionKind.COMBAT:
                        act = post["player"]["block"]
                        note(key, abs(y29 - act) < 0.5, act == 0, f"{tag}：生物 29 = {y29:g}，下一帧格挡 {act}；能力 {sorted(ppw)} 遗物 {pre['player']['relics']}")
                continue
            if kinds[k] in (ActionKind.PLAY_CARD, ActionKind.USE_POTION):
                c10, c10b = float(o.cand_feats[k, 10]), float(o28.cand_feats[k, 10])
                ti = o.tok_ints
                name = ""
                try:
                    from sts2env.protocol import TokenKind as _T  # noqa: F401
                    name = str(int(ti[o.cand_ints[k, 1], 1]))
                except Exception:  # noqa: BLE001
                    pass
                o = parse(a.req({"op": "step", "action": k})); o28 = parse(b.req({"op": "step", "action": k})); step += 1
                picked = 0
                while o.decision not in (DecisionKind.COMBAT, DecisionKind.GAME_OVER) and not o.done and o.num_actions > 0 and picked < 30:
                    kk = rng.randrange(o.num_actions)
                    o = parse(a.req({"op": "step", "action": kk})); o28 = parse(b.req({"op": "step", "action": kk})); step += 1; picked += 1
                post = a.state()
                if picked or not post.get("in_combat") or post.get("over") or o.decision != DecisionKind.COMBAT: continue
                e29, e28 = end_idx(o), end_idx(o28)
                if e29 is None or e28 is None: continue
                lost = pre["player"]["hp"] - post["player"]["hp"]
                drew = len(post["hand"]) + 1 > len(pre["hand"])        # 抽了牌（抽到的回合末效果两个格式都不算），单列
                for met, cn, _, _ in metrics:
                    if not met.startswith("self"): continue
                    try:
                        if not cond(cn, pre, None, o): continue
                    except KeyError:
                        continue
                    if met == "self_kill_applier":
                        before = {x["id"] for x in pre["enemies"] if x["alive"]}
                        after = {x["id"] for x in post["enemies"] if x["alive"]}
                        if not (before - after): continue
                    key = f"{met}:{cn}" + ("（抽了牌）" if drew else "")
                    x29, x28 = float(o.cand_feats[e29, 10]), float(o28.cand_feats[e28, 10])
                    note(key, abs(c10 - (lost + x29)) < 0.5, abs(c10b - (lost + x28)) < 0.5,
                         f"{tag} 牌 {name}：29 预测 {c10:g}，实际掉 {lost} + 打出后 {x29:g}（28：{c10b:g} / {x28:g}）；玩家能力 {sorted(pw(pre['player']))}")
                continue
            o = parse(a.req({"op": "step", "action": k})); o28 = parse(b.req({"op": "step", "action": k})); step += 1
    a.close(); b.close()
    errs = 0
    for nm in (f"{item}_29.log", f"{item}_28.log"):
        log = (logd / nm).read_text(encoding="utf-8", errors="replace")
        errs += sum(1 for line in log.splitlines() if "Exception" in line or "[error]" in line.lower())
    c["引擎日志异常行"] = errs
    return {"item": item, "counts": dict(c), "examples": ex, "seconds": round(time.time() - t0, 1)}


def table(results: list[dict]) -> list[dict]:
    rows = []
    for r in results:
        item = r["item"]; cnt = r["counts"]
        for met, cn, target, desc in ITEMS[item][1]:
            for key in sorted({k.rsplit("|", 1)[0] for k in cnt if "|" in k and k.split("|")[0].startswith(f"{met}:{cn}")}):
                nn = cnt.get(f"{key}|n", 0)
                if not nn: continue
                r29, r28 = cnt[f"{key}|29"] / nn, cnt[f"{key}|28"] / nn
                row = {"条目": item, "指标": key, "说明": desc, "帧数": nn, "28": round(r28, 3), "29": round(r29, 3), "目标": target,
                       "通过": r29 >= target - 1e-9}
                if f"{key}|sact" in cnt:        # 只比期望的：平均预测与平均实际，不判通过
                    row.update({"平均预测 28": round(cnt[f"{key}|s28"] / nn, 2), "平均预测 29": round(cnt[f"{key}|s29"] / nn, 2),
                                "平均实际": round(cnt[f"{key}|sact"] / nn, 2), "目标": "期望无偏", "通过": None})
                rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--new", required=True)
    ap.add_argument("--items", default=",".join(ITEMS))
    ap.add_argument("--combats", type=int, default=40)
    ap.add_argument("--jobs", type=int, default=4, help="并行的条目数（每个条目两个引擎进程）")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--examples", type=int, default=6)
    ap.add_argument("--dump", default="")
    args = ap.parse_args()
    items = [x for x in args.items.split(",") if x]
    jobs = [(it, args.combats, args.new, args.seed, args.examples) for it in items]
    with Pool(min(args.jobs, len(jobs))) as p:
        res = p.map(run_item, jobs)
    rows = table(res)
    out = {"表": rows, "明细": res}
    for r in rows:
        bias = f"\t平均 28 {r['平均预测 28']} 29 {r['平均预测 29']} 实际 {r['平均实际']}" if "平均实际" in r else ""
        res_s = "—" if r["通过"] is None else "通过" if r["通过"] else "未通过"
        print(f"{r['条目']}\t{r['指标']}\t帧 {r['帧数']}\t28 {r['28']:.3f}\t29 {r['29']:.3f}\t目标 {r['目标']}\t{res_s}{bias}\t{r['说明']}")
    for r in res:
        print(r["item"], {k: v for k, v in r["counts"].items() if "|" not in k}, f"{r['seconds']}s")
    if args.dump:
        Path(args.dump).parent.mkdir(parents=True, exist_ok=True)
        Path(args.dump).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
