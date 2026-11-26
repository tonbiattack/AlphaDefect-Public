"""瓦皮（新宠物）的编码与回合结束预演对真实掉血（观测独立核查.md）：python -I tools/obs113/wuppy_probe.py <dll> <角色> <场数>"""
import json, random, sys, tempfile
from pathlib import Path
from collections import Counter
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python")); sys.path.insert(0, str(REPO / "tools"))
import numpy as np
from sts2env.protocol import ActionKind, DecisionKind, TokenKind, Zone, parse
from obs_v21_check import Raw, play_bias
dll, char, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
eng = Raw(Path(dll), open(Path(tempfile.mkdtemp()) / "e.log", "w")); vocab = eng.env.vocab()
c = Counter(); shown = 0; et = []
for i in range(n):
    rng = random.Random(i)
    spec = {"character": char, "ascension": 10, "seed": f"WP{i}", "act": i % 3, "encounter": ["regular", "elite", "boss"][i % 3], "relics": ["WUPPY"], "hp": 200, "max_hp": 200}
    o = parse(eng.req({"op": "reset", "mode": "combat", "spec": spec})); pol = play_bias(rng, 0.85); step = 0
    while not (o.done or o.num_actions == 0 or step >= 400):
        if o.decision == DecisionKind.COMBAT:
            ti, tf = o.tok_ints, o.tok_feats
            pets = [j for j in np.flatnonzero((ti[:, 0] == TokenKind.CREATURE) & (ti[:, 2] == Zone.ALLY) & (tf[:, 5] == 0))]
            c["战斗帧"] += 1; c["有我方宠物记录的帧"] += int(len(pets) > 0)
            for j in pets:
                pw = [(vocab[ti[k, 1]], int(ti[k, 2])) for k in np.flatnonzero((ti[:, 0] == TokenKind.POWER) & (ti[:, 3] == j))]
                mv = [vocab[ti[k, 1]] for k in np.flatnonzero((ti[:, 0] == TokenKind.MOVE) & (ti[:, 3] == j))]
                for name, z in pw: c[f"宠物能力 {name} 在区域 {z}"] += 1
                for m in mv: c[f"宠物招式记录 {m}"] += 1
                if shown < 2: print("宠物", vocab[ti[j, 1]], "特征", tf[j, :8].tolist(), "16-19", tf[j, 16:20].tolist(), "能力", pw, "招式", mv); shown += 1
            pj = [j for j in np.flatnonzero((ti[:, 0] == TokenKind.CREATURE) & (tf[:, 5] == 1))][0]
        a = pol(step, o)
        ck = o.cand_ints[a] if o.decision == DecisionKind.COMBAT else None
        if ck is not None and ck[0] == ActionKind.END_TURN:
            pred = float(o.tok_feats[pj, 17]); hp0 = eng.state()["player"]["hp"]
            o = parse(eng.req({"op": "step", "action": int(a)})); step += 1
            g = 0
            while not o.done and o.decision != DecisionKind.COMBAT and g < 20:
                o = parse(eng.req({"op": "step", "action": pol(step, o)})); step += 1; g += 1
            if not o.done:
                st = eng.state(); et.append((pred, hp0 - st["player"]["hp"]))
            continue
        o = parse(eng.req({"op": "step", "action": int(a)})); step += 1
eng.close()
e = np.array(et); d = e[:, 1] - e[:, 0]
c["回合结束比对"] = len(e); c["精确一致"] = int((abs(d) < .5).sum()); c["实际<预测"] = int((d < -.5).sum()); c["实际>预测"] = int((d > .5).sum())
print(json.dumps(dict(sorted(c.items())), ensure_ascii=False, indent=0))
