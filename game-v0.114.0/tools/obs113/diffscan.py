"""两个引擎同步走同一局，逐帧解析后归类差异（编码只读，不影响局面）。用法：diffscan.py BASE NEW CHAR N [combat|run]"""
import json, random, sys, tempfile, os
from pathlib import Path
from collections import Counter, defaultdict
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python")); sys.path.insert(0, str(REPO / "tools"))
import numpy as np
from sts2env.protocol import TokenKind, Zone, ActionKind, DecisionKind, parse
from obs_v21_check import Raw, play_bias
base, new, char, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]); mode = sys.argv[5] if len(sys.argv) > 5 else "run"
d = Path(tempfile.mkdtemp())
A = Raw(Path(base), open(d / "a.log", "w")); B = Raw(Path(new), open(d / "b.log", "w")); voc = B.env.vocab()
K = {int(k): k.name for k in TokenKind}
cat = Counter(); ex = defaultdict(list); frames = 0; dframes = 0
def key(o, j): return voc[o.tok_ints[j, 1]] if 0 <= o.tok_ints[j, 1] < len(voc) else str(o.tok_ints[j, 1])
for g in range(n):
    rng = random.Random(g)
    req = {"op": "reset", "mode": "run", "character": char, "ascension": 10, "seed": f"DS{char}{g}", "route_summary": True, "privileged": True, "future_content": True} if mode == "run" else \
          {"op": "reset", "mode": "combat", "spec": {"character": char, "ascension": 10, "seed": f"DSC{char}{g}", "act": g % 3, "encounter": ["regular", "elite", "boss"][g % 3]}}
    a = parse(A.req(req)); b = parse(B.req(req)); pol = play_bias(rng, 0.85); step = 0
    while not (a.done or a.num_actions == 0 or step >= 3000):
        frames += 1; diffs = []
        for i in np.flatnonzero(a.header != b.header): diffs.append(f"头部{i}")
        if a.tok_ints.shape != b.tok_ints.shape:
            ka = Counter((K.get(int(r[0])), int(r[2])) for r in a.tok_ints); kb = Counter((K.get(int(r[0])), int(r[2])) for r in b.tok_ints)
            for k in set(ka) | set(kb):
                if ka[k] != kb[k]: diffs.append(f"记录数 {k[0]}@区域{k[1]} {ka[k]}→{kb[k]}")
        else:
            for j in np.flatnonzero((a.tok_ints != b.tok_ints).any(1)):
                for c in np.flatnonzero(a.tok_ints[j] != b.tok_ints[j]): diffs.append(f"记录整数 {K.get(int(a.tok_ints[j,0]))} {key(a,j)} 列{c}")
            m = ~np.isclose(a.tok_feats, b.tok_feats, equal_nan=True)
            for j, c in zip(*np.nonzero(m)): diffs.append(f"记录特征 {K.get(int(a.tok_ints[j,0]))}@区域{int(a.tok_ints[j,2])} {key(a,j)} 列{c}")
        if a.cand_feats.shape == b.cand_feats.shape:
            m = ~np.isclose(a.cand_feats, b.cand_feats, equal_nan=True)
            for i, c in zip(*np.nonzero(m)):
                ck = a.cand_ints[i]; obj = key(a, int(ck[1])) if ck[1] >= 0 and ck[1] < len(a.tok_ints) else "-"
                diffs.append(f"候选{c} {ActionKind(int(ck[0])).name} {obj}")
        else: diffs.append("候选数不同")
        if diffs:
            dframes += 1
            for x in set(diffs):
                cat[x] += 1
                if len(ex[x]) < 2: ex[x].append(f"{char}{g} 步{step}")
        act = pol(step, a)
        a = parse(A.req({"op": "step", "action": int(act)})); b = parse(B.req({"op": "step", "action": int(act)})); step += 1
A.close(); B.close()
print(json.dumps({"char": char, "frames": frames, "diff_frames": dframes}, ensure_ascii=False))
for k, v in cat.most_common(60): print(f"{v:6d}  {k}   {ex[k][:1]}")
