"""观测格式 24（Fool，游戏 v0.113.0）的核对：新字段的自洽与对真实结果（适配报告第五节）。
  combats：按 Fool 的射击、旋律、箭袋、钉住等牌组成牌组打单场战斗，偏向多打牌的随机出牌，逐帧检查：
    · Fool 的战斗帧都是格式 24，且恰有一条战斗计数子类型 4；
    · 子类型 4：箭位总数 = 可用 + 扎在敌人身上 + 已摧毁；扎在敌人身上的箭 = 各敌人扎穿记录 12 之和；扎穿记录 13 = 12 为 0 时 0，否则 ≥ 1；
    · 手里的射击牌：没有可用的箭时卡牌 16（伤害预览）为 0；
    · 打出射击牌：有可用的箭时下一帧“本场射出的箭”+1，没有时不变，且目标掉的血为 0（op state 对照）；
    · 打出带哇 / 啾 / 叽标签的牌后（同一回合）下一帧对应的旋律位为 1。
  runs：Fool 随机整局：帧的版本、卡住、引擎报错、新字段出现的次数。
  python -I tools/obs_v24_check.py combats --new .cache/eng-new/Sts2Headless.dll --combats 200
  python -I tools/obs_v24_check.py runs --new .cache/eng-new/Sts2Headless.dll --runs 20
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "tools"))

import numpy as np  # noqa: E402
from sts2env.gamever import catalog_path  # noqa: E402
from sts2env.protocol import ActionKind, DecisionKind, TokenKind, Zone, parse  # noqa: E402
from obs_v21_check import Raw, enemy_records, match_enemy, play_bias, powers_of  # noqa: E402

CAT = {e["key"]: e for e in json.loads(Path(catalog_path()).read_text(encoding="utf-8"))["entities"]}
FOOL = [k for k, v in CAT.items() if v.get("pool") == "FOOL_CARD_POOL" and v.get("rarity") not in ("Basic", "Ancient")]
TAGS = {k: set(v.get("tags") or []) for k, v in CAT.items() if k.startswith("CARD.")}
SHOOT = [k for k in FOOL if "Shoot" in TAGS[k]]
MELODY = [k for k in FOOL if TAGS[k] & {"Wah", "Cho", "Ki"}]
EXTRA = ["CARD.QUIVER", "CARD.NAIL", "CARD.BOUNCE", "CARD.RECALL", "CARD.MOURNING_BLOW", "CARD.ANGUISH", "CARD.MYTHIC_FORM", "CARD.EXPLOSIVE_ARROWS"]
RELICS = [["SONG_BOW"], ["SONG_BOW", "TWINE"], ["SONG_BOW", "CONDUCTORS_BATON"], []]


def spec(i: int, rng: random.Random) -> dict:
    pick = rng.sample(SHOOT, 5) + rng.sample(MELODY, 4) + rng.sample([k for k in EXTRA if k in CAT], 3)
    deck = [{"id": k.split(".", 1)[1], "upgrades": rng.randrange(2)} for k in pick]
    deck += [{"id": "STRIKE_FOOL", "upgrades": 0}] * 2 + [{"id": "DEFEND_FOOL", "upgrades": 0}] * 2
    return {"character": "FOOL", "ascension": 10, "seed": f"V24F{i}", "act": i % 3, "encounter": ["regular", "elite", "boss"][i % 3],
            "deck": deck, "relics": RELICS[(i // 4) % 4], "hp": 250, "max_hp": 250}


def stat4(o) -> np.ndarray | None:
    ti, tf = o.tok_ints, o.tok_feats
    js = np.flatnonzero((ti[:, 0] == TokenKind.COMBAT_STAT) & (tf[:, 0] == 4))
    return tf[js[0]] if len(js) == 1 else (np.full(20, -1.0) if len(js) > 1 else None)


def cmd_combats(args) -> None:
    eng = Raw(Path(args.new), open(Path(args.log_dir) / "combats.log", "w"))
    vocab = eng.env.vocab()
    c = Counter(); ex: dict[str, list] = {}

    def note(kind: str, ok: bool, msg: str = ""):
        c[f"{kind}：比对"] += 1
        c[f"{kind}：{'一致' if ok else '不一致'}"] += 1
        if not ok and len(ex.setdefault(kind, [])) < 8:
            ex[kind].append(msg)

    for i in range(args.combats):
        rng = random.Random(i)
        o = parse(eng.req({"op": "reset", "mode": "combat", "spec": spec(i, rng)}))
        pol = play_bias(rng, 0.9)
        step = 0
        while not (o.done or o.num_actions == 0 or step >= 600):
            if o.decision == DecisionKind.COMBAT:
                c[f"战斗帧（版本 {int(o.header[0])}）"] += 1
                s4 = stat4(o)
                note("恰有一条子类型 4", s4 is not None and s4[0] == 4, f"场 {i} 步 {step}")
                ti, tf = o.tok_ints, o.tok_feats
                if s4 is not None and s4[0] == 4:
                    note("箭位总数 = 可用 + 扎着 + 摧毁", s4[1] == s4[2] + s4[3] + s4[4], f"场 {i} 步 {step}：{s4[1:5]}")
                    imp = [powers_of(o, j, vocab).get("POWER.IMPALED_POWER") for j in enemy_records(o)]
                    away = sum(int(p[12]) for p in imp if p is not None)
                    note("扎着的箭 = 各敌人扎穿 12 之和", int(s4[3]) == away, f"场 {i} 步 {step}：{s4[3]} vs {away}")
                    for p in imp:
                        if p is not None:
                            note("扎穿 13 与 12 一致", (p[12] == 0 and p[13] == 0) or (p[12] >= 1 and p[13] >= 1), f"场 {i} 步 {step}：{p[12]},{p[13]}")
                            c["扎穿记录"] += 1
                    for j in np.flatnonzero((ti[:, 0] == TokenKind.CARD) & (ti[:, 2] == Zone.HAND)):
                        key = vocab[ti[j, 1]]
                        if "Shoot" in TAGS.get(key, ()):
                            if s4[2] == 0:
                                note("没箭时手里射击牌的伤害预览为 0", tf[j, 16] <= 0, f"场 {i} 步 {step} {key}：{tf[j, 16]}")
                            else:
                                c["有箭时手里射击牌的伤害预览 > 0"] += int(tf[j, 16] > 0)
            a = pol(step, o)
            pre_state = eng.state() if o.decision == DecisionKind.COMBAT else None
            ck = o.cand_ints[a] if o.decision == DecisionKind.COMBAT else None
            # 出牌候选的记录 A 是这张牌的卡牌记录（候选的词表 ID 一栏不是牌）
            card_key = vocab[int(o.tok_ints[int(ck[1]), 1])] if ck is not None and ck[0] == ActionKind.PLAY_CARD and ck[1] >= 0 else None
            s4pre = stat4(o) if o.decision == DecisionKind.COMBAT else None
            # 打的是哪个敌人（候选的记录 B），按 enemy_records 的顺序对应 op state 的 enemies
            ens = enemy_records(o) if card_key else []
            tidx = ens.index(int(ck[2])) if card_key and int(ck[2]) in ens else None
            o2 = parse(eng.req({"op": "step", "action": int(a)})); step += 1
            post_state = eng.state() if card_key else None
            if (card_key and "Shoot" in TAGS.get(card_key, ()) and s4pre is not None and s4pre[0] == 4 and s4pre[2] == 0 and tidx is not None
                    and pre_state is not None and tidx < len(pre_state.get("enemies", []))):
                # 没有可用的箭时打出射击牌：攻击不执行，目标的血（含格挡）不变
                e0, e1 = pre_state["enemies"][tidx], match_enemy(pre_state, post_state, tidx)
                if e1 is not None:
                    note("没箭时打出射击牌，目标血量与格挡不变", (e0.get("hp"), e0.get("block")) == (e1.get("hp"), e1.get("block")),
                         f"场 {i} 步 {step} {card_key}：{e0.get('hp')}/{e0.get('block')} → {e1.get('hp')}/{e1.get('block')}")
            if card_key and s4pre is not None and s4pre[0] == 4 and o2.decision == DecisionKind.COMBAT and not o2.done:
                s4post = stat4(o2)
                same_turn = pre_state is not None and post_state is not None and post_state.get("round") == pre_state.get("round")
                if s4post is not None and s4post[0] == 4:
                    if "Shoot" in TAGS.get(card_key, ()):
                        if card_key in ("CARD.RAPID_SHOT", "CARD.REPEAT_SHOT", "CARD.RAIN_SHOT", "CARD.SKY_SHOT"):
                            c[f"射击（多箭或不定）{card_key}"] += 1
                        else:
                            exp = s4pre[6] + (1 if s4pre[2] > 0 else 0)
                            note("打出射击牌后本场射出的箭", s4post[6] == exp, f"场 {i} 步 {step} {card_key}：可用 {s4pre[2]}，射出 {s4pre[6]} → {s4post[6]}")
                    for t, k in (("Wah", 7), ("Cho", 8), ("Ki", 9)):
                        if t in TAGS.get(card_key, ()) and same_turn:
                            note("打出旋律牌后对应旋律位为 1", s4post[k] == 1, f"场 {i} 步 {step} {card_key}")
            o = o2
        c["场数"] += 1
    eng.close()
    print(json.dumps(dict(sorted(c.items())), ensure_ascii=False, indent=1))
    for k, v in ex.items():
        print(f"\n不一致的例子（{k}）："); [print("  " + m) for m in v]


def cmd_runs(args) -> None:
    a = Raw(Path(args.new), open(Path(args.log_dir) / "runs.log", "w"))
    vocab = a.env.vocab()
    c = Counter()
    for i in range(args.runs):
        o = parse(a.req({"op": "reset", "mode": "run", "character": "FOOL", "ascension": 10, "seed": f"V24R{i}", "snapshots": True,
                         "route_summary": True, "privileged": True, "future_content": True}))
        pol = play_bias(random.Random(i), 0.85)
        step = 0
        while not (o.done or o.num_actions == 0 or step >= 4000):
            c[f"帧（版本 {int(o.header[0])}）"] += 1
            c["卡住"] += int(o.header[15])
            ti, tf = o.tok_ints, o.tok_feats
            c["子类型 4 记录"] += int(((ti[:, 0] == TokenKind.COMBAT_STAT) & (tf[:, 0] == 4)).sum())
            c["扎穿记录"] += int(sum(1 for j in np.flatnonzero(ti[:, 0] == TokenKind.POWER) if vocab[ti[j, 1]] == "POWER.IMPALED_POWER"))
            o = parse(a.req({"op": "step", "action": pol(step, o)})); step += 1
        c["局数"] += 1
        c["最深层数"] = max(c["最深层数"], int(o.header[7]))
    a.close()
    log = (Path(args.log_dir) / "runs.log").read_text(encoding="utf-8", errors="replace")
    c["引擎日志里的异常行"] = sum(1 for line in log.splitlines() if "Exception" in line or "[error]" in line.lower())
    print(json.dumps(dict(sorted(c.items())), ensure_ascii=False, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("combats"); e.add_argument("--new", required=True); e.add_argument("--combats", type=int, default=200)
    r = sub.add_parser("runs"); r.add_argument("--new", required=True); r.add_argument("--runs", type=int, default=20)
    for p in (e, r):
        p.add_argument("--log-dir", default=tempfile.mkdtemp(prefix="obs24-"))
    args = ap.parse_args()
    {"combats": cmd_combats, "runs": cmd_runs}[args.cmd](args)


if __name__ == "__main__":
    main()
