"""出牌候选的预测对真实结果（游戏更新v0.113.0/观测独立核查.md）：只读地用引擎打单场战斗，偏向多打牌的随机出牌，逐次比较
  候选 6（有目标的总伤害，敌人没死时对“生命 + 格挡”的减少）、候选 7（能否打死）、候选 8（这一步当场得到的格挡）、
  打全体的候选 6、7，候选 10 对“出牌后同回合下一帧的玩家 17”（自洽），以及结束回合时玩家 17 对真实掉血。
  牌组默认是初始牌组加 6–12 张本角色卡池的随机牌（普通 3、罕见 2、稀有 1 的权重）；环境变量 DECK、RELICS、POTIONS（逗号分隔的 ID）可另指定，
  SEEDOFF 改种子。敌人受击后加格挡、无实体、滑溜等会让“伤害”与“生命 + 格挡的减少”不同，属于正常偏差（猎人对照组约 1%）。
  python -I tools/obs113/pred_probe.py --dll engine/Sts2Headless/bin/Release/net9.0/Sts2Headless.dll --char FOOL --combats 600 --out /tmp/x.json
  SEEDOFF=1000 DECK=ICE_LANCE,REFRACT python -I tools/obs113/pred_probe.py --dll ... --char DEFECT --combats 150
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "tools"))

import numpy as np  # noqa: E402
from sts2env.gamever import catalog_path  # noqa: E402
from sts2env.protocol import ActionKind, DecisionKind, TokenKind, Zone, parse  # noqa: E402
from obs_v21_check import Raw, enemy_records, match_enemy, play_bias, powers_of  # noqa: E402

SEEDOFF = int(__import__('os').environ.get('SEEDOFF','0'))
CAT = {e["key"]: e for e in json.loads(Path(catalog_path()).read_text(encoding="utf-8"))["entities"]}
TAGS = {k: set(v.get("tags") or []) for k, v in CAT.items() if k.startswith("CARD.")}
W = {"Common": 3, "Uncommon": 2, "Rare": 1}
MULTI = {"CARD.COMPOST", "CARD.ENRAPTURE", "CARD.YOGA", "CARD.COVERING_FIRE", "CARD.ENSEMBLE", "CARD.SELF_DEPRECATE"}


def pool(char: str):
    p = f"{char}_CARD_POOL"
    ks = [k for k, v in CAT.items() if v.get("pool") == p and v.get("rarity") in W and k not in MULTI]
    return ks, [W[CAT[k]["rarity"]] for k in ks]


START = {
    "DEFECT": [("STRIKE_DEFECT", 4), ("DEFEND_DEFECT", 4), ("ZAP", 1), ("DUALCAST", 1)],
    "IRONCLAD": [("STRIKE_IRONCLAD", 4), ("DEFEND_IRONCLAD", 4), ("BASH", 1)],
    "NECROBINDER": [("STRIKE_NECROBINDER", 4), ("DEFEND_NECROBINDER", 4), ("BODYGUARD", 1), ("UNLEASH", 1)],
    "FOOL": [("STRIKE_FOOL", 5), ("DEFEND_FOOL", 4), ("FOOLISH_SHOT", 1)],
    "SILENT": [("STRIKE_SILENT", 5), ("DEFEND_SILENT", 5), ("NEUTRALIZE", 1), ("SURVIVOR", 1)],
}
RELIC = {"FOOL": ["SONG_BOW"], "SILENT": ["RING_OF_THE_SNAKE"]}


def spec(char, i, rng, ks, ws):
    import os
    deck = [{"id": k, "upgrades": 0} for k, n in START.get(char, []) for _ in range(n)]
    if os.environ.get("DECK"):
        deck += [{"id": k, "upgrades": int(rng.random() < 0.4)} for k in os.environ["DECK"].split(",")]
    else:
      for k in rng.choices(ks, ws, k=rng.randint(6, 12)):
        deck.append({"id": k.split(".", 1)[1], "upgrades": int(rng.random() < 0.4)})
    return {"character": char, "ascension": 10, "seed": f"PRB{char}{i + SEEDOFF}", "act": i % 3, "encounter": ["regular", "elite", "boss"][i % 3],
            "deck": deck, "relics": (__import__("os").environ["RELICS"].split(",") if __import__("os").environ.get("RELICS") else RELIC.get(char, [])), "potions": (__import__("os").environ["POTIONS"].split(",") if __import__("os").environ.get("POTIONS") else []), "hp": 200, "max_hp": 200}


def player_tok(o):
    ti, tf = o.tok_ints, o.tok_feats
    js = np.flatnonzero((ti[:, 0] == TokenKind.CREATURE) & (tf[:, 5] == 1))
    return int(js[0]) if len(js) else None


def stat4(o):
    ti, tf = o.tok_ints, o.tok_feats
    js = np.flatnonzero((ti[:, 0] == TokenKind.COMBAT_STAT) & (tf[:, 0] == 4))
    return tf[js[0]] if len(js) == 1 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dll", required=True)
    ap.add_argument("--char", default="FOOL")
    ap.add_argument("--combats", type=int, default=300)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ks, ws = pool(a.char)
    eng = Raw(Path(a.dll), open(Path(tempfile.mkdtemp(prefix="probe-")) / "eng.log", "w"))
    vocab = eng.env.vocab()
    c = Counter()
    per = defaultdict(Counter)       # 卡牌 → 计数
    ex = defaultdict(list)
    endturn = []                     # (预测掉血, 实际掉血, 有无随机目标标记)
    hand_frames = Counter()
    for i in range(a.combats):
        rng = random.Random(i * 7 + 1 + SEEDOFF)
        o = parse(eng.req({"op": "reset", "mode": "combat", "spec": spec(a.char, i, rng, ks, ws)}))
        pol = play_bias(rng, 0.85)
        step = 0
        while not (o.done or o.num_actions == 0 or step >= 500):
            if o.decision != DecisionKind.COMBAT:
                o = parse(eng.req({"op": "step", "action": pol(step, o)})); step += 1; continue
            ti, tf, ci, cf = o.tok_ints, o.tok_feats, o.cand_ints, o.cand_feats
            pj = player_tok(o)
            s4 = stat4(o)
            # 手牌层面：射击牌张数 vs 可用箭；瞄准 > 0 时手里射击牌数
            hand = [j for j in np.flatnonzero((ti[:, 0] == TokenKind.CARD) & (ti[:, 2] == Zone.HAND))]
            shoot_hand = [j for j in hand if "Shoot" in TAGS.get(vocab[ti[j, 1]], ()) and tf[j, 4] > 0]
            ppw = powers_of(o, pj, vocab) if pj is not None else {}
            aim = float(ppw["POWER.AIM_POWER"][0]) if "POWER.AIM_POWER" in ppw else 0.0
            hand_frames["战斗帧"] += 1
            if s4 is not None:
                if len(shoot_hand) >= 2: hand_frames["手里可打射击牌 ≥2"] += 1
                if len(shoot_hand) > s4[2]: hand_frames["手里可打射击牌 > 可用箭"] += 1
                if 0 < s4[2] < len(shoot_hand): hand_frames["0 < 可用箭 < 手里可打射击牌"] += 1
                if aim > 0 and len(shoot_hand) >= 2: hand_frames["瞄准>0 且手里可打射击牌 ≥2"] += 1
            act = pol(step, o)
            ck, cv = ci[act], cf[act]
            pre = eng.state()
            kind = int(ck[0])
            card = vocab[int(ti[int(ck[1]), 1])] if kind == ActionKind.PLAY_CARD and ck[1] >= 0 else None
            ens = enemy_records(o)
            tgt = ens.index(int(ck[2])) if card and int(ck[2]) in ens else None
            p_hp, p_blk = pre["player"]["hp"], pre["player"]["block"]
            o2 = parse(eng.req({"op": "step", "action": int(act)})); step += 1
            post = eng.state() if not o2.done else None
            if kind == ActionKind.END_TURN and pj is not None:
                pred = float(tf[pj, 17])
                # 下一个玩家决策时的血量（中途若有选牌等，继续用随机策略推进到下一战斗帧）
                o3, post3 = o2, post
                guard = 0
                while not o3.done and o3.decision != DecisionKind.COMBAT and guard < 20:
                    o3 = parse(eng.req({"op": "step", "action": pol(step, o3)})); step += 1; guard += 1
                    post3 = eng.state() if not o3.done else None
                if post3 is not None and post3.get("player"):
                    actual = p_hp - post3["player"]["hp"]
                    endturn.append((pred, actual, float(tf[pj, 18])))
                    if abs(actual - pred) >= 0.5 and len(ex["__END"]) < 40:
                        ex["__END"].append({"pred": pred, "actual": actual, "blk": p_blk, "hp": p_hp,
                            "ppow": [(p[0], p[1]) for p in pre["player"].get("powers", [])] if isinstance(pre["player"].get("powers"), list) else pre["player"].get("powers"),
                            "hand": [h.get("id") if isinstance(h, dict) else h for h in pre.get("hand", [])],
                            "enemies": [(e.get("id"), e.get("hp"), e.get("block"), e.get("intent") or e.get("move")) for e in pre.get("enemies", [])],
                            "relics": pre["player"].get("relics"), "pets": pre.get("pets"),
                            "post_enemies": [(e.get("id"), e.get("hp")) for e in post3.get("enemies", [])], "post_pow": post3["player"].get("powers")})
                o = o3; continue
            if kind in (ActionKind.PLAY_CARD, ActionKind.USE_POTION) and post is not None and post.get("in_combat") and o2.decision == DecisionKind.COMBAT and post.get("round") == pre.get("round"):
                pj2 = player_tok(o2)
                if pj2 is not None:
                    nm = card or (vocab[int(ti[int(ck[1]), 1])] if ck[1] >= 0 else "?")
                    okc = abs(float(o2.tok_feats[pj2, 17]) - float(cv[10])) < 0.5
                    c["候选10 vs 下一帧玩家17 比对"] += 1; c["候选10 vs 下一帧玩家17 一致"] += int(okc)
                    per[nm]["c10比对"] += 1; per[nm]["c10一致"] += int(okc)
                    if not okc and len(ex[nm]) < 6: ex[nm].append(f"c10 {cv[10]:.0f} 下一帧17 {float(o2.tok_feats[pj2, 17]):.0f}")
            if card and post is not None and post.get("in_combat"):
                tags = TAGS.get(card, set())
                cat = "射击" if "Shoot" in tags else ("旋律" if tags & {"Wah", "Cho", "Ki"} else "其他")
                if "Shoot" in tags:
                    cat += "·有箭" if (s4 is not None and s4[2] > 0) else "·没箭"
                    if aim > 0: cat += "·有瞄准"
                # 格挡：候选 8 vs 玩家格挡变化
                blk_gain = post["player"]["block"] - p_blk
                if cv[8] > 0 or blk_gain > 0:
                    ok = abs(cv[8] - blk_gain) < 0.5
                    per[card]["格挡比对"] += 1; per[card]["格挡一致"] += int(ok)
                    c[f"格挡（{cat}）比对"] += 1; c[f"格挡（{cat}）一致"] += int(ok)
                    if not ok and len(ex[card]) < 4: ex[card].append(f"格挡 预测{cv[8]:.0f} 实际{blk_gain}")
                # 伤害：有目标
                pe = pre.get("enemies", [])
                def loss(e0, e1):
                    if e1 is None: return None
                    return (e0["hp"] - e1["hp"]) + (e0["block"] - e1["block"])
                if tgt is not None and tgt < len(pe):
                    e0 = pe[tgt]; e1 = match_enemy(pre, post, tgt)
                    died = e1 is None or e1.get("hp", 1) <= 0
                    if cv[4] > 0 or cv[6] > 0 or (e1 is not None and loss(e0, e1) > 0):
                        per[card]["有目标"] += 1
                        if not died and e1 is not None:
                            L = loss(e0, e1); ok = abs(cv[6] - L) < 0.5
                            per[card]["伤害比对"] += 1; per[card]["伤害一致"] += int(ok)
                            c[f"有目标伤害（{cat}）比对"] += 1; c[f"有目标伤害（{cat}）一致"] += int(ok)
                            if not ok and len(ex[card]) < 6: ex[card].append(f"有目标 预测总伤{cv[6]:.0f}(单{cv[4]:.0f}×{cv[5]:.0f}) 实际{L}")
                        okk = (cv[7] > 0) == died
                        per[card]["击杀比对"] += 1; per[card]["击杀一致"] += int(okk)
                        c[f"击杀（{cat}）比对"] += 1; c[f"击杀（{cat}）一致"] += int(okk)
                        if not okk and len(ex[card]) < 6: ex[card].append(f"击杀 预测{cv[7]:.0f} 实际{'死' if died else '活'} 目标血{e0['hp']}+{e0['block']} 预测总伤{cv[6]:.0f}")
                    # 目标以外的敌人掉血（候选 6 不含）
                    side = 0
                    for k2, e in enumerate(pe):
                        if k2 == tgt: continue
                        e2 = match_enemy(pre, post, k2)
                        if e2 is not None: side += max(0, loss(e, e2))
                        else: side += e["hp"] + e["block"]
                    if side > 0:
                        per[card]["目标外掉血"] += 1
                        c[f"有目标出牌伤到其他敌人（{cat}）"] += 1
                        if len(ex[card]) < 6: ex[card].append(f"目标外敌人掉 {side}（候选只算目标）")
                elif ck[2] < 0:
                    tot = 0; dead = 0; alive_all = True
                    for k2, e in enumerate(pe):
                        e2 = match_enemy(pre, post, k2)
                        if e2 is None or e2.get("hp", 1) <= 0: dead += 1; alive_all = False; tot += e["hp"] + e["block"]
                        else: tot += max(0, loss(e, e2))
                    if cv[6] > 0 or tot > 0:
                        per[card]["无目标"] += 1
                        if alive_all:
                            ok = abs(cv[6] - tot) < 0.5
                            per[card]["无目标伤害比对"] += 1; per[card]["无目标伤害一致"] += int(ok)
                            c[f"无目标伤害（{cat}）比对"] += 1; c[f"无目标伤害（{cat}）一致"] += int(ok)
                            if not ok and len(ex[card]) < 6: ex[card].append(f"无目标 预测{cv[6]:.0f}(单{cv[4]:.0f}×{cv[5]:.0f}) 实际合计{tot}")
                        okk = int(cv[7]) == dead
                        c[f"无目标击杀数（{cat}）比对"] += 1; c[f"无目标击杀数（{cat}）一致"] += int(okk)
            o = o2
        c["场数"] += 1
    eng.close()
    res = {"总计": dict(sorted(c.items())), "手牌": dict(hand_frames)}
    et = np.array(endturn) if endturn else np.zeros((0, 3))
    if len(et):
        d = et[:, 1] - et[:, 0]
        res["回合结束预演"] = {"次数": int(len(et)), "精确一致": int((np.abs(d) < 0.5).sum()), "实际>预测": int((d > 0.5).sum()), "实际<预测": int((d < -0.5).sum()),
                         "平均绝对误差": float(np.abs(d).mean()), "实际>预测 平均差": float(d[d > 0.5].mean()) if (d > 0.5).any() else 0.0}
    bad = []
    for k, v in per.items():
        n = v["伤害比对"] + v["无目标伤害比对"] + v["格挡比对"] + v["击杀比对"]
        miss = (v["伤害比对"] - v["伤害一致"]) + (v["无目标伤害比对"] - v["无目标伤害一致"]) + (v["格挡比对"] - v["格挡一致"]) + (v["击杀比对"] - v["击杀一致"])
        miss += v["c10比对"] - v["c10一致"]
        if miss or v["目标外掉血"]:
            bad.append((k, dict(v), ex[k]))
    bad.sort(key=lambda t: -((t[1].get("伤害比对", 0) - t[1].get("伤害一致", 0)) + (t[1].get("格挡比对", 0) - t[1].get("格挡一致", 0)) + t[1].get("目标外掉血", 0) + (t[1].get("击杀比对", 0) - t[1].get("击杀一致", 0))))
    res["不一致的牌"] = [{"牌": k, "计数": v, "例子": e} for k, v, e in bad]
    res["回合结束不一致例子"] = ex.get("__END", [])
    txt = json.dumps(res, ensure_ascii=False, indent=1)
    if a.out: Path(a.out).write_text(txt, encoding="utf-8")
    print(txt[:6000])


if __name__ == "__main__":
    main()
