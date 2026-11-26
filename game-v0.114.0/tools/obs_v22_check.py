"""观测格式 22（铁甲战士、亡灵契约师）的核对：新字段对真实结果（铁甲与亡灵契约师观测调研.md 第三节）。

两个新引擎进程同步走同一局：一个按角色默认出格式 22，另一个 reset 带 obs_format 21（对照）。观测格式不影响游戏，两边的局面逐步相同
（engine_equal.py 的门控核对），所以同一个局面上能比“格式 22 的预测”与“格式 21 的预测”哪个更接近真实结果。真实结果来自 op state。

  出牌：候选 8（当场格挡）对打出后格挡的增加；有狂怒、无惧疼痛、骨笛、势不可当、卡戎之灰、狱火、末日降临，或奥斯提出手的牌时，
        候选 7（打死目标 / 打死几个）与候选 6（没打死时的总伤害）对真实结果；
        候选 10 的自洽：打出前预测的“做完这一步就结束回合掉多少血” = 这一步实际掉的血 + 打出后结束回合候选的 10（召唤、奥斯提被牺牲、自伤这几类单列）。
  结束回合：结束回合候选的 10（玩家预计掉的血）对敌方回合里实际掉的血，按“奥斯提活着 / 有荆棘或火焰屏障 / 两者都没有”分开统计，格式 22 与 21 各算一遍；
        奥斯提记录 16–17（预计掉的血、剩下的生命）对奥斯提实际的血量；灾厄 12 对敌方回合结束后是不是死了。
  战斗计数子类型 3 对手牌预览：扯碎、猛晃、亡魂牵引的段数，死亡行军的伤害。

  python -I tools/obs_v22_check.py effects --new .cache/eng-v22/Sts2Headless.dll --combats 160 --dump 报告目录
  python -I tools/obs_v22_check.py runs --new .cache/eng-v22/Sts2Headless.dll --runs 8     # 随机整局（帧的版本、卡住、异常、新字段出现次数）
  python -I tools/obs_v22_check.py effects --new ... --fmt 28    # 被核对的一方 reset 带 obs_format 28（格式 28 = 27 + 22，铁甲战士决策记录第 1 条）；同种子与 --fmt 22 逐项可比
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "tools"))

import numpy as np  # noqa: E402

from sts2env.protocol import ActionKind, DecisionKind, TokenKind, Zone, parse  # noqa: E402
from obs_v21_check import Raw, enemy_records, match_enemy, powers_of  # noqa: E402

IRON = {
    "self": ["BLOODLETTING", "OFFERING", "HEMOKINESIS", "BLOOD_WALL", "BRAND", "BREAKTHROUGH", "DEMONIC_SHIELD", "INFERNO", "RUPTURE"],
    "block": ["RAGE", "FEEL_NO_PAIN", "JUGGERNAUT", "TRUE_GRIT", "SECOND_WIND", "IMPERVIOUS", "EVIL_EYE", "SHRUG_IT_OFF", "CORRUPTION", "BURNING_PACT", "IRON_WAVE"],
    "thorns": ["FLAME_BARRIER", "JUGGERNAUT", "DEFEND_IRONCLAD", "TAUNT", "IMPERVIOUS", "STONE_ARMOR", "COLOSSUS", "EXPECT_A_FIGHT"],
    "exhaust": ["FEEL_NO_PAIN", "TRUE_GRIT", "SECOND_WIND", "FIEND_FIRE", "CINDER", "MOLTEN_FIST", "EVIL_EYE", "WHIRLWIND", "ASHEN_STRIKE", "TEAR_ASUNDER", "RAGE"],
}
NECRO = {
    "osty": ["POKE", "FLATTEN", "FETCH", "HIGH_FIVE", "BONE_SHARDS", "RATTLE", "SIC_EM", "SNAP", "RIGHT_HAND_HAND", "PROTECTOR", "UNLEASH", "SQUEEZE"],
    "summon": ["BODYGUARD", "AFTERLIFE", "PULL_AGGRO", "LEGION_OF_BONE", "REANIMATE", "SPUR", "SACRIFICE", "BONE_SHARDS", "NECRO_MASTERY", "INVOKE"],
    "doom": ["END_OF_DAYS", "NEGATIVE_PULSE", "DEATHBRINGER", "SCOURGE", "BLIGHT_STRIKE", "NO_ESCAPE", "TIMES_UP", "DEFY", "FEAR"],
    "stats": ["DEATH_MARCH", "PULL_FROM_BELOW", "RATTLE", "DEFILE", "FEAR", "PARSE", "FETCH", "SCOURGE", "VEILPIERCER", "SCULPTING_STRIKE"],
}
RELICS = {
    "IRONCLAD": [["CHARONS_ASHES"], ["BURNING_BLOOD"], ["CHARONS_ASHES", "BRONZE_SCALES"], []],
    "NECROBINDER": [["BOUND_PHYLACTERY", "BONE_FLUTE"], ["BOUND_PHYLACTERY"], ["BONE_FLUTE", "CHARONS_ASHES"], ["BOUND_PHYLACTERY", "BOOK_REPAIR_KNIFE"]],
}
STARTER = {"IRONCLAD": ["STRIKE_IRONCLAD", "DEFEND_IRONCLAD"], "NECROBINDER": ["STRIKE_NECROBINDER", "DEFEND_NECROBINDER"]}


def focus_spec(ch: str, group: str, pool: list[str], i: int, rng: random.Random) -> dict:
    k = min(8, len(pool))
    deck = [{"id": c, "upgrades": rng.randrange(2)} for c in rng.sample(pool, k) for _ in range(2)]
    deck += [{"id": STARTER[ch][0], "upgrades": 0}] * 2 + [{"id": STARTER[ch][1], "upgrades": 0}] * 2
    return {"character": ch, "ascension": 10, "seed": f"V22{ch[:2]}{group[:2]}{i}", "act": i % 3, "encounter": ["regular", "elite", "boss"][i % 3],
            "deck": deck, "relics": RELICS[ch][(i // 8) % 4], "hp": 250, "max_hp": 250}


def player_rec(o) -> int | None:
    ti, tf = o.tok_ints, o.tok_feats
    js = np.flatnonzero((ti[:, 0] == TokenKind.CREATURE) & (tf[:, 5] > 0))
    return int(js[0]) if len(js) else None


def osty_rec(o, vocab) -> int | None:
    ti = o.tok_ints
    js = [j for j in np.flatnonzero((ti[:, 0] == TokenKind.CREATURE) & (ti[:, 2] == Zone.ALLY)) if vocab[ti[j, 1]] == "MONSTER.OSTY"]
    return int(js[0]) if js else None


def end_turn_c10(o) -> float | None:
    n = o.num_actions
    ks = [k for k in range(n) if o.cand_ints[k, 0] == ActionKind.END_TURN]
    return float(o.cand_feats[ks[0], 10]) if ks else None


def cmd_effects(args) -> None:
    a = Raw(Path(args.new), open(args.log_dir / "v22.log", "w"))
    b = Raw(Path(args.new), open(args.log_dir / "v21.log", "w"))
    vocab = a.env.vocab()
    cards = {v.split(".", 1)[1] for v in vocab if v.startswith("CARD.")}
    c = Counter(); ex: dict[str, list] = {}

    def note(kind: str, ok: bool, msg: str = ""):
        c[f"{kind}：比对"] += 1
        c[f"{kind}：{'一致' if ok else '不一致'}"] += 1
        if not ok and msg and len(ex.setdefault(kind, [])) < 10:
            ex[kind].append(msg)

    groups = [("IRONCLAD", g, p) for g, p in IRON.items()] + [("NECROBINDER", g, p) for g, p in NECRO.items()]
    t0 = time.time()
    for i in range(args.combats):
        ch, group, pool = groups[i % len(groups)]
        pool = sorted({x for x in pool if x in cards})
        rng = random.Random(7000 + i)
        spec = focus_spec(ch, group, pool, i, rng)
        req = {"op": "reset", "mode": "combat", "spec": spec, "labels": True, "loop_cut": True, "stall_turns": 8}
        o = parse(a.req({**req, **({"obs_format": args.fmt} if args.fmt else {})})); o21 = parse(b.req({**req, "obs_format": 21}))
        tag = f"V22E{i}（{ch} {group}）"
        step = 0
        while not (o.done or o.num_actions == 0 or step >= 700):
            n = o.num_actions
            ti, tf = o.tok_ints, o.tok_feats
            if o.decision != DecisionKind.COMBAT:
                k = rng.randrange(n)
                o = parse(a.req({"op": "step", "action": k})); o21 = parse(b.req({"op": "step", "action": k})); step += 1
                continue
            c["战斗帧"] += 1
            if int(o.header[0]) != (args.fmt or 22): c[f"不是格式 {args.fmt or 22} 的帧"] += 1
            # ---- 战斗计数子类型 3 对手牌预览
            st = np.flatnonzero((ti[:, 0] == TokenKind.COMBAT_STAT) & (tf[:, 0] == 3))
            c["战斗帧有子类型 3"] += int(len(st) == 1)
            if len(st) == 1:
                s = tf[st[0]]
                pl = player_rec(o)
                pw = powers_of(o, pl, vocab) if pl is not None else {}
                strength = float(pw.get("POWER.STRENGTH_POWER", np.zeros(20))[0])
                plain = set(pw) <= {"POWER.STRENGTH_POWER", "POWER.DEXTERITY_POWER", "POWER.PLATING_POWER", "POWER.FEEL_NO_PAIN_POWER", "POWER.RAGE_POWER",
                                    "POWER.JUGGERNAUT_POWER", "POWER.FLAME_BARRIER_POWER", "POWER.NO_DRAW_POWER", "POWER.CORRUPTION_POWER", "POWER.RUPTURE_POWER"}
                for j in np.flatnonzero((ti[:, 0] == TokenKind.CARD) & (ti[:, 2] == Zone.HAND)):
                    name = vocab[ti[j, 1]].split(".", 1)[1]
                    if name == "TEAR_ASUNDER": note("扯碎：段数 = 1 + 子类型 3 的 1", abs(tf[j, 7] - (1 + s[1])) < 0.5, f"{tag} 第 {step} 步：卡面 {tf[j, 7]:g}，计数 {s[1]:g}")
                    elif name == "RATTLE": note("猛晃：段数 = 1 + 子类型 3 的 4", abs(tf[j, 7] - (1 + s[4])) < 0.5, f"{tag} 第 {step} 步：卡面 {tf[j, 7]:g}，计数 {s[4]:g}")
                    elif name == "PULL_FROM_BELOW": note("亡魂牵引：段数 = 子类型 3 的 3", abs(tf[j, 7] - s[3]) < 0.5, f"{tag} 第 {step} 步：卡面 {tf[j, 7]:g}，计数 {s[3]:g}")
                    elif name == "DEATH_MARCH" and plain and ti[j, 4] == 0:
                        want = (9 if tf[j, 3] else 8) + 4 * s[2] + strength
                        note("死亡行军：伤害预览 = 基础 + 4 × 子类型 3 的 2 + 力量", abs(tf[j, 16] - want) < 0.5, f"{tag} 第 {step} 步：预览 {tf[j, 16]:g}，按计数 {want:g}")
            kinds = o.cand_ints[:n, 0]
            plays = [k for k in range(n) if kinds[k] == ActionKind.PLAY_CARD]
            ends = [k for k in range(n) if kinds[k] == ActionKind.END_TURN]
            pre = a.state()
            enemies = enemy_records(o)
            trace = f"{i}:{step}" in args.trace
            k = rng.choice(plays) if plays and rng.random() < 0.78 else (ends[0] if ends and rng.random() < 0.8 else rng.randrange(n))
            cf, cf21 = o.cand_feats[k].copy(), o21.cand_feats[k].copy()
            ostyj = osty_rec(o, vocab)
            pets = pre.get("pets") or []
            osty_alive = bool(pets and pets[0]["alive"])
            ppw = {p[0]: p[1] for p in pre["player"]["powers"]}
            if kinds[k] == ActionKind.PLAY_CARD:
                card = vocab[ti[o.cand_ints[k, 1], 1]].split(".", 1)[1]
                card_rec = int(o.cand_ints[k, 1])
                tgt_rec = int(o.cand_ints[k, 2])
                tgt = enemies.index(tgt_rec) if tgt_rec in enemies else None
                post_o = parse(a.req({"op": "step", "action": k})); post21 = parse(b.req({"op": "step", "action": k})); step += 1
                picked = 0
                while post_o.decision not in (DecisionKind.COMBAT, DecisionKind.GAME_OVER) and not post_o.done and post_o.num_actions > 0 and picked < 30:
                    kk = rng.randrange(post_o.num_actions)
                    post_o = parse(a.req({"op": "step", "action": kk})); post21 = parse(b.req({"op": "step", "action": kk})); step += 1; picked += 1
                post = a.state()
                if trace:
                    print(f"=== 跟踪 {tag} 第 {step} 步 {card}：候选 {np.round(cf, 1).tolist()}（21：{np.round(cf21, 1).tolist()}）")
                    for key in ("player", "pets", "enemies", "hand"):
                        print(f"  前 {key}：{json.dumps(pre.get(key), ensure_ascii=False)[:700]}")
                        print(f"  后 {key}：{json.dumps(post.get(key), ensure_ascii=False)[:700]}")
                    print(f"  打出后结束回合候选 10：{end_turn_c10(post_o) if post_o.num_actions else None}")
                over = not post.get("in_combat") or post.get("over")
                if over and not post.get("won"):
                    o, o21 = post_o, post21
                    continue
                alive_pre = [(e["id"], e["hp"], e["block"]) if e["alive"] else None for e in pre["enemies"]]
                dmg, dead = [], []
                for idx, ap in enumerate(alive_pre):
                    if ap is None: dmg.append(0); dead.append(False); continue
                    e = None if over else match_enemy(pre, post, idx)
                    if over or e is None or not e["alive"] or e["max_hp"] != pre["enemies"][idx]["max_hp"]: dmg.append(ap[1] + ap[2]); dead.append(True)
                    else: dmg.append(ap[1] + ap[2] - e["hp"] - e["block"]); dead.append(False)
                hittable = [e for e in pre["enemies"] if e["alive"]]
                relics = set(pre["player"]["relics"])
                # 候选 8：打出后格挡的增加（打出途中选牌的、能抽牌的不比：抽到的牌可能带格挡效果）
                draws = card in ("BURNING_PACT", "SHRUG_IT_OFF", "POMMEL_STRIKE", "OFFERING", "FETCH", "PARSE", "SCOURGE", "BATTLE_TRANCE")
                gain = 0 if over else post["player"]["block"] - pre["player"]["block"]
                riders = [x for x, y in (("狂怒", "RAGE_POWER" in ppw and tf[card_rec, 5] > 0), ("无惧疼痛", "FEEL_NO_PAIN_POWER" in ppw),
                                         ("骨笛", "BONE_FLUTE" in relics and card in NECRO["osty"] and osty_alive), ("势不可当", "JUGGERNAUT_POWER" in ppw)) if y]
                if not over and not picked and not draws:
                    lab = "候选 8：打出后格挡的增加" + ("（" + "、".join(riders) + "）" if riders else "（无附带格挡）")
                    note(lab + " — 格式 22", abs(cf[8] - gain) < 0.5, f"{tag} 第 {step} 步 {card}：预测 {cf[8]:g}（21：{cf21[8]:g}），实际 {gain}，能力 {sorted(ppw)}")
                    if riders: note(lab + " — 格式 21", abs(cf21[8] - gain) < 0.5)
                # 候选 6–7：伤害类附带效果、奥斯提出手、末日降临
                kind = []
                if "JUGGERNAUT_POWER" in ppw and len(hittable) == 1 and (cf[8] > 0): kind.append("势不可当单敌")
                if "CHARONS_ASHES" in relics and (card in ("TRUE_GRIT", "SECOND_WIND", "FIEND_FIRE", "BURNING_PACT") or tf[card_rec, 11] > 0 or ("CORRUPTION_POWER" in ppw)): kind.append("卡戎之灰")
                if "INFERNO_POWER" in ppw and card in ("BLOODLETTING", "OFFERING", "HEMOKINESIS", "BLOOD_WALL", "BRAND", "BREAKTHROUGH", "DEMONIC_SHIELD"): kind.append("狱火")
                if card == "END_OF_DAYS": kind.append("末日降临")
                if card in NECRO["osty"]: kind.append("奥斯提出手" + ("" if osty_alive else "（奥斯提不在）"))
                if card in ("TRUE_GRIT", "SECOND_WIND", "FIEND_FIRE", "BURNING_PACT", "BRAND") and "卡戎之灰" in kind: kind = [x for x in kind if x != "卡戎之灰"] + ["卡戎之灰（打出时另消耗别的牌，不算）"]
                if kind and not picked:
                    lab = "、".join(kind)
                    kills = sum(dead)
                    if tgt is not None and tf[card_rec, 5] > 0:
                        note(f"{lab}：能否打死目标 — 格式 22", int(round(cf[7])) == int(dead[tgt]), f"{tag} 第 {step} 步 {card}：预测 {cf[7]:g}（21：{cf21[7]:g}），实际 {'死' if dead[tgt] else '活'}；4–7 {np.round(cf[4:8], 1).tolist()}；目标 {alive_pre[tgt]}")
                        note(f"{lab}：能否打死目标 — 格式 21", int(round(cf21[7])) == int(dead[tgt]))
                        if not dead[tgt] and "势不可当多敌" not in lab:
                            note(f"{lab}：目标总伤害（没打死时） — 格式 22", abs(cf[6] - dmg[tgt]) < 0.5, f"{tag} 第 {step} 步 {card}：预测 {cf[6]:g}（21：{cf21[6]:g}），实际 {dmg[tgt]}；目标 {alive_pre[tgt]}")
                            note(f"{lab}：目标总伤害（没打死时） — 格式 21", abs(cf21[6] - dmg[tgt]) < 0.5)
                    else:
                        note(f"{lab}：打死几个 — 格式 22", int(round(cf[7])) == kills, f"{tag} 第 {step} 步 {card}：预测 {cf[7]:g}（21：{cf21[7]:g}），实际 {kills}；4–7 {np.round(cf[4:8], 1).tolist()}；敌人 {alive_pre} 掉 {dmg}")
                        note(f"{lab}：打死几个 — 格式 21", int(round(cf21[7])) == kills)
                        if kills == 0:
                            note(f"{lab}：总伤害（没打死时） — 格式 22", abs(cf[6] - sum(dmg)) < 0.5, f"{tag} 第 {step} 步 {card}：预测 {cf[6]:g}（21：{cf21[6]:g}），实际 {sum(dmg)}；敌人 {alive_pre} 掉 {dmg}")
                            note(f"{lab}：总伤害（没打死时） — 格式 21", abs(cf21[6] - sum(dmg)) < 0.5)
                # 候选 10 的自洽：预测 = 实际掉的血 + 打出后结束回合候选的 10
                if not over and not picked and post_o.decision == DecisionKind.COMBAT:
                    e22, e21 = end_turn_c10(post_o), end_turn_c10(post21)
                    if e22 is not None:
                        lost = pre["player"]["hp"] - post["player"]["hp"]
                        cat = ("自伤牌" if card in IRON["self"][:7] else "增生（另治疗奥斯提，不算）" if card == "SPUR" else "召唤下回合（不算）" if card == "INVOKE"
                               else "召唤" if card in NECRO["summon"] and card not in ("SACRIFICE", "BONE_SHARDS")
                               else "奥斯提被牺牲" if card in ("SACRIFICE", "BONE_SHARDS") and osty_alive else "其他")
                        note(f"候选 10 自洽（{cat}） — 格式 22", abs(cf[10] - (lost + e22)) < 0.5, f"{tag} 第 {step} 步 {card}：预测 {cf[10]:g}，实际掉 {lost} + 打出后预计 {e22:g}（21：{cf21[10]:g} / {e21}）")
                        if e21 is not None: note(f"候选 10 自洽（{cat}） — 格式 21", abs(cf21[10] - (lost + e21)) < 0.5)
                o, o21 = post_o, post21
                continue
            if kinds[k] == ActionKind.END_TURN:
                pj = player_rec(o)
                pred22, pred21 = float(cf[10]), float(cf21[10])
                o_pred = (float(tf[ostyj, 16]), float(tf[ostyj, 17])) if ostyj is not None and osty_alive else None
                dooms = []
                for idx, j in enumerate(enemies):
                    p = powers_of(o, j, vocab).get("POWER.DOOM_POWER")
                    if p is not None and idx < len(pre["enemies"]) and pre["enemies"][idx]["alive"]: dooms.append((idx, int(p[12]), float(p[13]), pre["enemies"][idx]))
                thorns = "THORNS_POWER" in ppw or "FLAME_BARRIER_POWER" in ppw
                post_o = parse(a.req({"op": "step", "action": k})); post21 = parse(b.req({"op": "step", "action": k})); step += 1
                post = a.state()
                over = not post.get("in_combat") or post.get("over")
                if over:
                    c["结束回合后战斗结束，不比"] += 1
                    o, o21 = post_o, post21
                    continue
                # 回合开始时的掉血、回血（狱火、深红斗篷的自伤，燃烧之血不在战斗中）会混进来：有这些能力时不比
                start_fx = {"INFERNO_POWER", "CRIMSON_MANTLE_POWER", "REGEN_POWER", "DEMISE_POWER"} & set(ppw)
                lost = pre["player"]["hp"] - post["player"]["hp"]
                cat = "奥斯提活着" if osty_alive else "荆棘或火焰屏障" if thorns else "都没有"
                if not start_fx:
                    note(f"结束回合：玩家预计掉血（{cat}） — 格式 22", abs(pred22 - lost) < 0.5, f"{tag} 第 {step} 步：预测 {pred22:g}（21：{pred21:g}），实际 {lost}；奥斯提 {pets}；敌人 {[(e['id'], e['hp'], e['intent'] if 'intent' in e else '') for e in pre['enemies'] if e['alive']]}")
                    note(f"结束回合：玩家预计掉血（{cat}） — 格式 21", abs(pred21 - lost) < 0.5)
                if o_pred is not None:
                    # 下回合开始的召唤（缚魂命匣每回合 1 点、召唤下回合）在读到状态之前已经结算：奥斯提活着时加最大生命并回同样多的血，
                    # 死了则复活成召唤量的生命（OstyCmd.Summon）。召唤量从子类型 3 的“本场召唤总量”前后之差得到
                    def summoned(fr):
                        js = np.flatnonzero((fr.tok_ints[:, 0] == TokenKind.COMBAT_STAT) & (fr.tok_feats[:, 0] == 3))
                        return float(fr.tok_feats[js[0], 5]) if len(js) else 0.0
                    ds = summoned(post_o) - summoned(o)
                    ppets = post.get("pets") or []
                    if ppets:
                        q = ppets[0]
                        revived = ds > 0 and q["max_hp"] == ds and pets[0]["max_hp"] + ds != ds
                        died = (not q["alive"]) or revived
                        if o_pred[1] <= 0:
                            note("奥斯提 17：预测会被打死", died, f"{tag} 第 {step} 步：预测掉 {o_pred[0]:g} 剩 0，实际 {pets[0]} → {q}（下回合开始召唤 {ds:g}）")
                        else:
                            note("奥斯提 17：预测活着时剩下的生命", (not died) and abs(o_pred[1] + ds - q["hp"]) < 0.5,
                                 f"{tag} 第 {step} 步：预测掉 {o_pred[0]:g} 剩 {o_pred[1]:g}，实际 {pets[0]} → {q}（下回合开始召唤 {ds:g}）")
                for idx, doomed, margin, e0 in dooms:
                    e1 = match_enemy(pre, post, idx)
                    died = e1 is None or not e1["alive"] or e1["max_hp"] != e0["max_hp"]
                    note("灾厄 12：标了会被处决的都死了", (not doomed) or died, f"{tag} 第 {step} 步 {e0['id']}：预测 {doomed}（差 {margin:g}），生命 {e0['hp']} → {'死' if died else e1['hp']}")
                    if not doomed: c["灾厄 12 为 0：敌方回合后" + ("死了（别的伤害或预测漏了）" if died else "活着")] += 1
                o, o21 = post_o, post21
                continue
            o = parse(a.req({"op": "step", "action": k})); o21 = parse(b.req({"op": "step", "action": k})); step += 1
    a.close(); b.close()
    out = {"统计": dict(sorted(c.items())), "不一致的例子": ex, "场数": args.combats, "seconds": round(time.time() - t0, 1)}
    print(json.dumps(out, ensure_ascii=False, indent=1))
    if args.dump:
        d = Path(args.dump); d.mkdir(parents=True, exist_ok=True)
        (d / "格式22新字段对真实结果.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


def cmd_runs(args) -> None:
    """随机整局与课程战斗：帧的版本、卡住、引擎报错、新字段出现的次数。"""
    a = Raw(Path(args.new), open(args.log_dir / "runs.log", "w"))
    vocab = a.env.vocab()
    c = Counter()
    from engine_equal import play_bias
    for ch in ("IRONCLAD", "NECROBINDER"):
        for i in range(args.runs):
            o = parse(a.req({"op": "reset", "mode": "run", "character": ch, "ascension": 10, "seed": f"V22R{ch[:2]}{i}", "snapshots": True,
                             "route_summary": True, "privileged": True, "future_content": True, **({"obs_format": args.fmt} if args.fmt else {})}))
            pol = play_bias(random.Random(i), 0.85)
            step = 0
            while not (o.done or o.num_actions == 0 or step >= 4000):
                c[f"{ch} 帧（版本 {int(o.header[0])}）"] += 1
                c[f"{ch} 卡住"] += int(o.header[15])
                ti, tf = o.tok_ints, o.tok_feats
                c[f"{ch} 子类型 3 记录"] += int(((ti[:, 0] == TokenKind.COMBAT_STAT) & (tf[:, 0] == 3)).sum())
                c[f"{ch} 灾厄记录"] += int(sum(1 for j in np.flatnonzero(ti[:, 0] == TokenKind.POWER) if vocab[ti[j, 1]] == "POWER.DOOM_POWER"))
                oj = osty_rec(o, vocab)
                if oj is not None:
                    c[f"{ch} 奥斯提记录"] += 1
                    c[f"{ch} 奥斯提的能力在我方区域"] += int(((ti[:, 0] == TokenKind.POWER) & (ti[:, 3] == oj) & (ti[:, 2] == Zone.ALLY)).sum())
                    c[f"{ch} 奥斯提的能力在敌方区域（应为 0）"] += int(((ti[:, 0] == TokenKind.POWER) & (ti[:, 3] == oj) & (ti[:, 2] == Zone.ENEMY)).sum())
                    c[f"{ch} 奥斯提的招式记录（应为 0）"] += int(((ti[:, 0] == TokenKind.MOVE) & (ti[:, 3] == oj)).sum())
                o = parse(a.req({"op": "step", "action": pol(step, o)})); step += 1
            c[f"{ch} 局数"] += 1
            c[f"{ch} 最深层数"] = max(c[f"{ch} 最深层数"], int(o.header[7]))
    a.close()
    log = (args.log_dir / "runs.log").read_text(encoding="utf-8", errors="replace")
    c["引擎日志里的异常行"] = sum(1 for line in log.splitlines() if "Exception" in line or "[error]" in line.lower())
    print(json.dumps(dict(sorted(c.items())), ensure_ascii=False, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("effects")
    e.add_argument("--new", required=True); e.add_argument("--combats", type=int, default=160); e.add_argument("--dump", default="")
    e.add_argument("--trace", default="", help="场次:步数，逗号分隔（打出前的步数，同例子里的“第 N 步”减 1），打印这些出牌前后的完整状态")
    r = sub.add_parser("runs")
    r.add_argument("--new", required=True); r.add_argument("--runs", type=int, default=8)
    for p in (e, r):
        p.add_argument("--log-dir", type=Path, default=REPO / ".cache" / "obs_v22_check")
        p.add_argument("--fmt", type=int, default=0, help="被核对的一方 reset 带这个 obs_format（默认不带，按角色查表）")
    args = ap.parse_args()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    {"effects": cmd_effects, "runs": cmd_runs}[args.cmd](args)


if __name__ == "__main__":
    main()
