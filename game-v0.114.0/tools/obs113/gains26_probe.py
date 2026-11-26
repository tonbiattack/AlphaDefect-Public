"""格式 26 的候选 12（这一步当场得到的能量）、13（这一步之后进手的牌数）对实际（游戏更新v0.113.0/观测独立核查.md 第十二节）。

随机偏向出牌地打单场战斗，逐步比较出牌、用药水候选：
  - 12 对“下一帧能量 − 这一帧能量 + 这张牌的费用”（X 费按全部能量）；
  - 13 对“下一帧手牌 − (这一帧手牌 − 1)”（出牌）或“下一帧手牌 − 这一帧手牌”（药水）。
这一步有选牌等中间决策时按随机策略走完再比。13 的实际值会被“这一步让别的牌离手”（弃牌、消耗、自动打出）拉低，按牌分开看。
只读；不改引擎。
  python -I tools/obs113/gains26_probe.py --dll .cache/eng-r5/Sts2Headless.dll --char FOOL --combats 200 [--deck CHO_CHORUS,...]
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
from sts2env.protocol import ActionKind, DecisionKind, TokenKind, Zone, parse  # noqa: E402
from obs_v21_check import Raw, play_bias  # noqa: E402
from pred_probe import pool, spec  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dll", required=True)
    ap.add_argument("--char", default="FOOL")
    ap.add_argument("--combats", type=int, default=200)
    ap.add_argument("--deck", default=None, help="逗号分隔的牌 ID，加在打击、防御之外（不给则随机加本角色卡池的牌）")
    ap.add_argument("--relics", default="")
    ap.add_argument("--potions", default="")
    ap.add_argument("--obs-format", type=int, default=26)
    ap.add_argument("--seedoff", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ks, ws = pool(a.char)
    eng = Raw(Path(a.dll), open(Path(tempfile.mkdtemp(prefix="g26-")) / "eng.log", "w"))
    vocab = eng.env.vocab()
    c = Counter(); per = defaultdict(Counter); ex = defaultdict(list)
    for i in range(a.seedoff, a.seedoff + a.combats):
        rng = random.Random(i * 11 + 3)
        sp = spec(a.char, i, rng, ks, ws)
        sp["seed"] = f"G26{a.char}{i}"
        sp["relics"] = [x for x in a.relics.split(",") if x]
        sp["potions"] = [x for x in a.potions.split(",") if x]
        if a.deck:
            sp["deck"] = [d for d in sp["deck"] if d["id"].startswith(("STRIKE", "DEFEND"))] + [{"id": k, "upgrades": int(rng.random() < 0.4)} for k in a.deck.split(",")]
        o = parse(eng.req({"op": "reset", "mode": "combat", "spec": sp, "obs_format": a.obs_format}))
        c["场数"] += 1
        pol = play_bias(rng, 0.85)
        step = 0
        while not (o.done or o.num_actions == 0 or step >= 500):
            if o.decision != DecisionKind.COMBAT:
                o = parse(eng.req({"op": "step", "action": pol(step, o)})); step += 1; continue
            if o.cand_feats.shape[1] < 14:
                c["候选不足 14 列"] += 1; break
            act = pol(step, o)
            ck, cv = o.cand_ints[act], o.cand_feats[act]
            kind = int(ck[0])
            if kind not in (ActionKind.PLAY_CARD, ActionKind.USE_POTION):
                o = parse(eng.req({"op": "step", "action": int(act)})); step += 1; continue
            pre = eng.state()
            ti = o.tok_ints
            hand_toks = [j for j in range(len(ti)) if ti[j, 0] == TokenKind.CARD and ti[j, 2] == Zone.HAND]
            cost = 0; name = "POTION"
            if kind == ActionKind.PLAY_CARD and ck[1] >= 0:
                name = vocab[int(ti[int(ck[1]), 1])]
                if int(ck[1]) in hand_toks and hand_toks.index(int(ck[1])) < len(pre["hand"]):
                    hc = pre["hand"][hand_toks.index(int(ck[1]))]
                    cost = pre["player"]["energy"] if hc.get("cost", 0) == -1 else max(0, hc.get("cost", 0))
            o2 = parse(eng.req({"op": "step", "action": int(act)})); step += 1
            g = 0
            while not o2.done and o2.decision != DecisionKind.COMBAT and g < 20:
                o2 = parse(eng.req({"op": "step", "action": pol(step, o2)})); step += 1; g += 1
            post = eng.state() if not o2.done else None
            if post is None or not post.get("player") or post.get("round") != pre.get("round"):
                o = o2; continue
            de = post["player"]["energy"] - pre["player"]["energy"] + cost
            dh = len(post["hand"]) - len(pre["hand"]) + (1 if kind == ActionKind.PLAY_CARD else 0)
            p12, p13 = float(cv[12]), float(cv[13])
            ok12, ok13 = abs(p12 - de) < 0.5, abs(p13 - dh) < 0.5
            c["比对"] += 1; c["12 一致"] += int(ok12); c["13 一致"] += int(ok13)
            c["12 > 0 的步"] += int(p12 > 0); c["13 > 0 的步"] += int(p13 > 0)
            c["实际得能量的步"] += int(de > 0); c["实际进手的步"] += int(dh > 0)
            if de > 0 or p12 > 0:
                c["12 相关步 比对"] += 1; c["12 相关步 一致"] += int(ok12)
            if dh > 0 or p13 > 0:
                c["13 相关步 比对"] += 1; c["13 相关步 一致"] += int(ok13)
            per[name]["比对"] += 1; per[name]["12一致"] += int(ok12); per[name]["13一致"] += int(ok13)
            if not ok12 and len(ex["12·" + name]) < 4:
                ex["12·" + name].append(f"预测{p12:.0f} 实际{de} 费{cost} 能量{pre['player']['energy']}→{post['player']['energy']} 能力{pre['player'].get('powers')} 目标{int(ck[2])}")
            if not ok13 and len(ex["13·" + name]) < 4:
                ex["13·" + name].append(f"预测{p13:.0f} 实际{dh} 手牌{[h['id'] for h in pre['hand']]} → {[h['id'] for h in post['hand']]} 能力{pre['player'].get('powers')}")
            o = o2
    eng.close()
    bad = {k: dict(v) for k, v in sorted(per.items(), key=lambda kv: -(2 * kv[1]["比对"] - kv[1]["12一致"] - kv[1]["13一致"])) if v["12一致"] < v["比对"] or v["13一致"] < v["比对"]}
    res = {"角色": a.char, "格式": a.obs_format, "总计": dict(sorted(c.items())), "不一致的牌": bad, "例子": dict(ex)}
    txt = json.dumps(res, ensure_ascii=False, indent=1)
    if a.out:
        Path(a.out).write_text(txt, encoding="utf-8")
    print(txt[:12000])


if __name__ == "__main__":
    main()
