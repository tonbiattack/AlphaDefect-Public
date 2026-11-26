"""一次性诊断：知识恶魔的三次“知识诅咒”二选一，愚者与猎人各怎么选、选完胜率如何。在 B 上只读运行（nice 10）。
看这场战斗里玩家身上的瓦解（层数 = 每回合末掉的血）、心智腐蚀、懒惰、消瘦。"""
import glob
import gzip
import json
import os
import sys
from collections import Counter, defaultdict

DIS, MIND, SLOTH, WASTE = 1335, 1407, 1491, 1544
LINES = {"愚者": "/root/.local/share/sls2/runs/fool-eval", "猎人": "/root/.local/share/sls2/runs/silent-eval"}
RLO, RHI, MAXR = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])


def ver_of(p):
    return int(os.path.basename(p).split("_")[0][1:])


for name, d in LINES.items():
    files = sorted(f for f in glob.glob(f"{d}/eval_replays/v*_*.jsonl.gz") if RLO <= ver_of(f) <= RHI)[:MAXR]
    stats = defaultdict(list)       # 选法 → [是否赢, 回合数, 每回合掉血]
    order = Counter()
    nkd = 0
    for f in files:
        try:
            with gzip.open(f, "rt") as fh:
                head = json.loads(fh.readline())
                steps = [json.loads(l) for l in fh]
        except Exception:
            continue
        cs = head.get("combats") or []
        segs, cur = [], None
        for s in steps:
            g = s.get("g")
            if not g:
                continue
            if g[8]:
                cur = cur or []
                cur.append(s)
            elif cur is not None:
                segs.append(cur); cur = None
        if cur is not None:
            segs.append(cur)
        if len(segs) != len(cs):
            continue
        for k, (seg, c) in enumerate(zip(segs, cs)):
            if c["encounter"] != "KNOWLEDGE_DEMON_BOSS":
                continue
            nkd += 1
            won = (k < len(cs) - 1) or bool(head.get("win"))
            seq = []            # 每次诅咒后的选择，按出现顺序
            prev = {}
            hp_by_turn = {}
            for s in seg:
                if "pl" not in s:
                    continue
                pw = {p[0]: p[1] for p in s["pl"].get("p", [])}
                for pid, tag in ((DIS, "瓦解"), (MIND, "心智腐蚀"), (SLOTH, "懒惰"), (WASTE, "消瘦")):
                    if pid == DIS:
                        if pw.get(DIS, 0) > prev.get(DIS, 0):
                            seq.append("瓦解")
                    elif pid in pw and pid not in prev:
                        seq.append(tag)
                prev = pw
                t = s["g"][7]
                hp_by_turn.setdefault(t, s["pl"]["hp"])
            ts = sorted(hp_by_turn)
            taken = [max(0, hp_by_turn[ts[i]] - hp_by_turn[ts[i + 1]]) for i in range(len(ts) - 1)]
            key = "→".join(seq) if seq else "（还没到诅咒）"
            stats[key].append((won, len(ts), sum(taken) / len(taken) if taken else 0))
            for i, x in enumerate(seq):
                order[(i, x)] += 1
    print(f"== {name}：{len(files)} 局回放，知识恶魔 {nkd} 场")
    for i in range(3):
        a = order[(i, "瓦解")]; b = sum(v for (j, x), v in order.items() if j == i and x != "瓦解")
        if a + b:
            print(f"  第 {i + 1} 次诅咒：选瓦解 {a}（{a / (a + b):.0%}），选另一张 {b}")
    for key, v in sorted(stats.items(), key=lambda kv: -len(kv[1])):
        if len(v) < 8:
            continue
        print(f"  {key}: {len(v)} 场，胜率 {sum(w for w, _, _ in v) / len(v):.2f}，回合 {sum(t for _, t, _ in v) / len(v):.1f}，每回合掉血 {sum(x for _, _, x in v) / len(v):.1f}")
