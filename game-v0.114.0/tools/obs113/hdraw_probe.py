"""头部 32+33（留手 + 下回合实际进手）对下回合第一帧手牌张数。用法：hdraw_probe.py DLL CHAR N DECK"""
import json, random, sys, tempfile, os
from pathlib import Path
from collections import Counter
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python")); sys.path.insert(0, str(REPO / "tools"))
import numpy as np
from sts2env.protocol import TokenKind, Zone, ActionKind, DecisionKind, parse
from obs_v21_check import Raw, play_bias
dll, char, n, deck = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4].split(",")
E = Raw(Path(dll), open(Path(tempfile.mkdtemp()) / "e.log", "w"))
c = Counter(); ex = []
for g in range(n):
    rng = random.Random(g)
    spec = {"character": char, "ascension": 10, "seed": f"HD{char}{g}", "act": g % 3, "encounter": ["regular", "elite", "boss"][g % 3],
            "deck": [{"id": k, "upgrades": 0} for k in deck], "hp": 300, "max_hp": 300}
    o = parse(E.req({"op": "reset", "mode": "combat", "spec": spec})); pol = play_bias(rng, 0.9); step = 0
    while not (o.done or o.num_actions == 0 or step >= 400):
        a = pol(step, o)
        if o.decision == DecisionKind.COMBAT and o.cand_ints[a][0] == ActionKind.END_TURN:
            h32, h33 = int(o.header[32]), int(o.header[33])
            st = E.state(); pile = len(st.get("draw", [])) + len(st.get("discard", []))
            o = parse(E.req({"op": "step", "action": int(a)})); step += 1
            g2 = 0
            while not o.done and o.decision != DecisionKind.COMBAT and g2 < 10:
                o = parse(E.req({"op": "step", "action": pol(step, o)})); step += 1; g2 += 1
            if o.done: break
            hand = int(((o.tok_ints[:, 0] == TokenKind.CARD) & (o.tok_ints[:, 2] == Zone.HAND)).sum())
            if pile >= h33:      # 牌堆够抽时才比
                c["比对"] += 1; ok = hand == h32 + h33; c["一致"] += int(ok)
                if not ok and len(ex) < 6: ex.append(f"{g}:{step} 头部32={h32} 33={h33} 实际手牌={hand}")
            continue
        o = parse(E.req({"op": "step", "action": int(a)})); step += 1
E.close(); print(json.dumps(dict(c), ensure_ascii=False), ex)
