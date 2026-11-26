"""Fool 召回放到抽牌堆顶后，已知牌堆顶（区域 16–18）与真实顶部是否一致（观测独立核查.md）：python -I tools/obs113/drawtop_probe.py <dll> <场数>"""
import json, random, sys, tempfile
from pathlib import Path
from collections import Counter
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python")); sys.path.insert(0, str(REPO / "tools"))
import numpy as np
from sts2env.protocol import ActionKind, DecisionKind, TokenKind, Zone, parse
from obs_v21_check import Raw, play_bias
dll = sys.argv[1]; n = int(sys.argv[2])
# v0.114.0：召回不再把弃牌堆的牌放到抽牌堆顶（Recall 改成保留 + 拔箭 + 消耗），改用哇 叽（WahKi.OnPlay：从弃牌堆选一张放到抽牌堆顶）
from sts2env.gamever import catalog_path  # noqa: E402
import json as _json  # noqa: E402
TOPPER = "WAH_KI" if any(e["key"] == "CARD.TWIN_SHOT" for e in _json.loads(Path(catalog_path()).read_text(encoding="utf-8"))["entities"]) else "RECALL"
logd = Path(tempfile.mkdtemp(prefix="dtp-")); log = open(logd / "eng.log", "w")
eng = Raw(Path(dll), log); vocab = eng.env.vocab()
c = Counter()
for i in range(n):
    rng = random.Random(i)
    deck = [{"id": "STRIKE_FOOL", "upgrades": 0}] * 4 + [{"id": "DEFEND_FOOL", "upgrades": 0}] * 4 + [{"id": "FOOLISH_SHOT", "upgrades": 0}]
    deck += [{"id": k, "upgrades": rng.randrange(2)} for k in [TOPPER, TOPPER, TOPPER, "KI_CHANT", "QUICK_SHOT", "LOST_LUTE", "WAH_TOUCH", "CHO_READY", "KI_HOP"]]
    spec = {"character": "FOOL", "ascension": 10, "seed": f"DT{i}", "act": i % 3, "encounter": ["regular", "elite", "boss"][i % 3], "deck": deck, "relics": [], "hp": 200, "max_hp": 200}
    o = parse(eng.req({"op": "reset", "mode": "combat", "spec": spec})); pol = play_bias(rng, 0.9); step = 0
    while not (o.done or o.num_actions == 0 or step >= 400):
        if o.decision == DecisionKind.COMBAT:
            ti = o.tok_ints
            k = int(((ti[:, 0] == TokenKind.CARD) & (ti[:, 2] >= 16) & (ti[:, 2] <= 18)).sum())
            c["战斗帧"] += 1; c["有已知牌堆顶的帧"] += int(k > 0)
            if k > 0:
                st = eng.state(); top = st["draw"][:k]
                rec = [vocab[ti[j, 1]] for j in np.flatnonzero((ti[:, 0] == TokenKind.CARD) & (ti[:, 2] >= 16) & (ti[:, 2] <= 18))]
                ids = ["CARD." + (t["id"] if isinstance(t, dict) else t) for t in top]
                c["已知顶与真实顶一致"] += int(sorted(rec) == sorted(ids))
        a = pol(step, o)
        o = parse(eng.req({"op": "step", "action": a})); step += 1
    c["场数"] += 1
eng.close(); log.close()
txt = (logd / "eng.log").read_text(errors="replace")
c["引擎日志 [drawtop] 不一致行"] = txt.count("[drawtop]")
print(json.dumps(dict(c), ensure_ascii=False))
