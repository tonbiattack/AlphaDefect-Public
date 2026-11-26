"""格式 27 的手牌记录 20 标黄、21 标红（CardModel.ShouldGlowGold / ShouldGlowRed）对另一条路径算出的条件（观测格式.md 四之十二）。

随机偏向出牌地打单场战斗，每个战斗帧逐张看手牌：
  - 射击牌（没箭时整张牌不生效）标红 ⇔ 箭袋里没有就绪的箭（战斗计数子类型 4 的 2：Retrieved 的箭数）；
  - 箭矢戳击、啾 坚定标黄 ⇔ 有就绪的箭；
  - 持弓猛击、叽 跳跃、钉住标黄 ⇔ 有敌人被扎穿；召回、拧下、连根拔起标黄 ⇔ 有敌人被玩家的箭扎穿（单人对局两者相同）；
  - 心神不宁标黄 ⇔ 手里只有这一张；
  - 手牌以外的牌、别的记录 20、21 一律为 0。
敌人能力、手牌、牌堆读 state，箭数读这一帧的战斗计数记录（AddFoolStats，与发光判断不是同一段代码）。只读；不改引擎。
  python -I tools/obs113/glow27_probe.py --dll .cache/eng-g27/Sts2Headless.dll --combats 60
  python -I tools/obs113/glow27_probe.py --dll ... --char DEFECT --obs-format 27 --deck FTL,...   # 别的角色只看“手牌以外为 0”与标黄标红出现次数
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
sys.path.insert(0, str(REPO / "tools" / "obs113"))

import numpy as np  # noqa: E402
from sts2env.protocol import DecisionKind, TokenKind, Zone, parse  # noqa: E402
from obs_v21_check import Raw, play_bias  # noqa: E402
from pred_probe import pool, spec  # noqa: E402

SHOTS = {"BLOSSOM_SHOT", "DIVINE_SHOT", "FOOLISH_SHOT", "HEAVY_SHOT", "HOBBLING_SHOT", "MELODIC_SHOT", "QUICK_SHOT", "RAIN_SHOT",
         "RAPID_SHOT", "TWIN_SHOT", "REPEAT_SHOT", "SKY_SHOT", "THE_SHOT", "TRICK_SHOT"}
IMPALE_GOLD = {"BOW_SMACK", "KI_HOP", "NAIL", "RECALL", "WREST", "UPROOT"}
from pred_probe import CAT  # noqa: E402
_TWIN = "TWIN_SHOT" if "CARD.TWIN_SHOT" in CAT else "RAPID_SHOT"      # v0.114.0 迅捷射击重做成双重射击（RapidShot 删掉）
# v0.114.0：箭矢戳击改成伤害 + 瞄准，不再按“有就绪的箭”标黄（ArrowStab 去掉了 ShouldGlowGoldInternal）
ARROW_GOLD = {"CHO_FIRM"} if _TWIN == "TWIN_SHOT" else {"ARROW_STAB", "CHO_FIRM"}
DEFAULT_DECK = f"TRICK_SHOT,QUICK_SHOT,HEAVY_SHOT,{_TWIN},SKY_SHOT,ARROW_STAB,CHO_FIRM,BOW_SMACK,KI_HOP,NAIL,RECALL,WREST,UPROOT,RESTLESSNESS"


def expect(cid: str, st: dict, ready: int) -> tuple[int | None, int | None]:
    """（应否标黄，应否标红）；None 为这张牌不核对这一项。"""
    impaled = any(p[0] == "IMPALED_POWER" for e in st["enemies"] if e.get("alive") for p in e.get("powers", []))
    gold = red = None
    if cid in SHOTS:
        red = int(ready == 0)
    if cid in ARROW_GOLD:
        gold = int(ready > 0)
    if cid in IMPALE_GOLD:
        gold = int(impaled)
    if cid == "RESTLESSNESS":
        gold = int(len(st["hand"]) == 1)
    return gold, red


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dll", required=True)
    ap.add_argument("--char", default="FOOL")
    ap.add_argument("--combats", type=int, default=60)
    ap.add_argument("--deck", default=None, help="逗号分隔的牌 ID，加在打击、防御之外（Fool 默认是射击牌与看箭、看扎穿的牌）")
    ap.add_argument("--obs-format", type=int, default=0, help="0 为按角色（Fool 27）")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    deck = a.deck or (DEFAULT_DECK if a.char == "FOOL" else "")
    ks, ws = pool(a.char)
    eng = Raw(Path(a.dll), open(Path(tempfile.mkdtemp(prefix="g27-")) / "eng.log", "w"))
    vocab = eng.env.vocab()
    c = Counter(); per = defaultdict(Counter); ex = defaultdict(list)
    for i in range(a.combats):
        rng = random.Random(i * 13 + 5)
        sp = spec(a.char, i, rng, ks, ws)
        sp["seed"] = f"GL27{a.char}{i}"
        if deck:
            sp["deck"] = [d for d in sp["deck"] if d["id"].startswith(("STRIKE", "DEFEND"))] + [{"id": k, "upgrades": int(rng.random() < 0.3)} for k in deck.split(",")]
        req = {"op": "reset", "mode": "combat", "spec": sp}
        if a.obs_format:
            req["obs_format"] = a.obs_format
        o = parse(eng.req(req))
        c["场数"] += 1
        pol = play_bias(rng, 0.85)
        step = 0
        while not (o.done or o.num_actions == 0 or step >= 400):
            ver = int(o.header[0])
            c[f"版本 {ver} 的帧"] += 1
            ti, tf = o.tok_ints, o.tok_feats
            card = ti[:, 0] == TokenKind.CARD
            hand = np.flatnonzero(card & (ti[:, 2] == Zone.HAND))
            other = np.flatnonzero(~(card & (ti[:, 2] == Zone.HAND)))
            c["手牌以外的记录"] += len(other)
            c["手牌以外 20/21 非 0"] += int((tf[other, 20:22] != 0).any(1).sum())
            if o.decision == DecisionKind.COMBAT and len(hand):
                st = eng.state()
                stat = np.flatnonzero((ti[:, 0] == TokenKind.COMBAT_STAT) & (tf[:, 0] == 4))
                ready = int(tf[stat[0], 2]) if len(stat) else 0
                for k, t in enumerate(hand):
                    g, r = int(tf[t, 20]), int(tf[t, 21])
                    c["手牌"] += 1; c["标黄"] += g; c["标红"] += r
                    cid = st["hand"][k]["id"] if k < len(st["hand"]) else vocab[int(ti[t, 1])]
                    if cid != vocab[int(ti[t, 1])].split(".")[-1] and not vocab[int(ti[t, 1])].endswith(cid):
                        c["手牌对不上 state"] += 1; continue
                    eg, er = expect(cid, st, ready)
                    for tag, got, want in (("黄", g, eg), ("红", r, er)):
                        if want is None:
                            continue
                        per[cid][f"{tag} 比对"] += 1; per[cid][f"{tag} 一致"] += int(got == want)
                        per[cid][f"{tag} 应为 1"] += want
                        c[f"{tag} 比对"] += 1; c[f"{tag} 一致"] += int(got == want)
                        if got != want and len(ex[cid]) < 4:
                            ex[cid].append(f"{tag} 观测{got} 应为{want} 就绪箭{ready} 手牌{[h['id'] for h in st['hand']]} 敌人能力{[e.get('powers') for e in st['enemies']]}")
                    if eg is None and er is None and (g or r):
                        per[cid]["别的牌 标黄"] += g; per[cid]["别的牌 标红"] += r
            o = parse(eng.req({"op": "step", "action": pol(step, o)})); step += 1
    eng.close()
    res = {"角色": a.char, "总计": dict(sorted(c.items())), "按牌": {k: dict(v) for k, v in sorted(per.items())}, "不一致的例子": dict(ex)}
    txt = json.dumps(res, ensure_ascii=False, indent=1)
    if a.out:
        Path(a.out).write_text(txt, encoding="utf-8")
    print(txt[:8000])


if __name__ == "__main__":
    main()
