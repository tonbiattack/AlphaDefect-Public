"""一次性诊断：愚者与猎人同版本段的 Boss 战。在 B 上只读运行（nice 19）。
第一部分（eval_runs 逐局记录）：进 Boss 战时的状态（生命比例、牌组张数、升级、遗物、药水）与胜负，按生命比例分组的胜率。
第二部分（eval_replays 回放）：每场 Boss 战的回合数、每回合打掉 Boss 多少、自己掉多少、输时 Boss 还剩多少。"""
import glob
import gzip
import json
import os
import statistics as st
import sys
from collections import defaultdict

LINES = {"愚者": "/root/.local/share/sls2/runs/fool-eval", "猎人": "/root/.local/share/sls2/runs/silent-eval"}
LO, HI = int(sys.argv[1]), int(sys.argv[2])            # 第一部分的版本段
RLO, RHI = int(sys.argv[3]), int(sys.argv[4])          # 第二部分（回放）的版本段
MAXR = int(sys.argv[5]) if len(sys.argv) > 5 else 2500


def boss_entries(r):
    """逐局记录里的 Boss 战：[(序号, 条目, 是否赢)]。最后一场且没通关的算输。"""
    cs = r.get("combats") or []
    out = []
    for i, c in enumerate(cs):
        if c["room"] != "Boss":
            continue
        won = (i < len(cs) - 1) or bool(r.get("win"))
        out.append((i, c, won))
    return out


def ver_of(path):
    b = os.path.basename(path)
    return int(b.split("_")[0][1:]) if "_" in b else int(b[1:].split(".")[0])


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


print(f"== 第一部分：第 {LO}–{HI} 版的逐局记录")
part1 = {}
for name, d in LINES.items():
    rows = []
    for f in glob.glob(f"{d}/eval_runs/v*.jsonl"):
        v = ver_of(f)
        if LO <= v <= HI:
            rows += [json.loads(l) for l in open(f) if l.strip()]
    if name == "猎人":
        pass
    part1[name] = rows
    print(f"{name}：{len(rows)} 局")
    for act in range(3):
        ent = [(c, w) for r in rows for (i, c, w) in boss_entries(r) if c["act"] == act]
        if not ent:
            continue
        def s(sel, k):
            return mean(c[k] for c, w in ent if sel(w))
        hpf = lambda c: c["hp"] / max(1, c["max_hp"])
        print(f"  第{act + 1}幕 Boss {len(ent)} 场，胜率 {mean(w for _, w in ent):.3f}；"
              f"开打 生命 {mean(c['hp'] for c, _ in ent):.1f}/{mean(c['max_hp'] for c, _ in ent):.1f}（{mean(hpf(c) for c, _ in ent):.2f}），"
              f"牌组 {mean(c['deck'] for c, _ in ent):.1f} 张、升级 {mean(c['upgrades'] for c, _ in ent):.1f}、遗物 {mean(c['relics'] for c, _ in ent):.1f}、药水 {mean(c['potions'] for c, _ in ent):.2f}")
        bins = [(0, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.01)]
        parts = []
        for a, b in bins:
            g = [w for c, w in ent if a <= hpf(c) < b]
            parts.append(f"{a:.2f}–{min(b, 1):.2f}: {mean(g):.2f}（{len(g)}）" if g else f"{a:.2f}–{min(b,1):.2f}: -")
        print("    按开打生命比例的胜率 " + "；".join(parts))
        dbins = [(0, 18), (18, 24), (24, 30), (30, 99)]
        parts = []
        for a, b in dbins:
            g = [w for c, w in ent if a <= c["deck"] < b]
            parts.append(f"{a}–{b - 1} 张: {mean(g):.2f}（{len(g)}）" if g else f"{a}–{b-1} 张: -")
        print("    按牌组张数的胜率 " + "；".join(parts))
        byb = defaultdict(list)
        for c, w in ent:
            byb[c["encounter"].replace("_BOSS", "")].append((c, w))
        print("    各 Boss：" + "；".join(f"{k} {mean(w for _, w in v):.2f}（{len(v)}，开打生命 {mean(hpf(c) for c, _ in v):.2f}）" for k, v in sorted(byb.items())))

print(f"\n== 第二部分：第 {RLO}–{RHI} 版的回放（每条线最多 {MAXR} 局）")
for name, d in LINES.items():
    files = sorted((f for f in glob.glob(f"{d}/eval_replays/v*_*.jsonl.gz") if RLO <= ver_of(f) <= RHI))[:MAXR]
    fights = []
    for f in files:
        try:
            with gzip.open(f, "rt") as fh:
                head = json.loads(fh.readline())
                steps = [json.loads(l) for l in fh]
        except Exception:
            continue
        cs = head.get("combats") or []
        # 战斗段：g[8] 为 1 的连续步；第 k 段对应 combats[k]
        segs, cur = [], None
        for s in steps:
            g = s.get("g")
            if not g:
                continue
            if g[8]:
                if cur is None:
                    cur = []
                cur.append(s)
            elif cur is not None:
                segs.append(cur); cur = None
        if cur is not None:
            segs.append(cur)
        if len(segs) != len(cs):
            continue
        for k, (seg, c) in enumerate(zip(segs, cs)):
            if c["room"] != "Boss":
                continue
            won = (k < len(cs) - 1) or bool(head.get("win"))
            turns = {}
            for s in seg:
                t = s["g"][7]
                if t not in turns and "pl" in s and "en" in s:
                    ehp = sum(e["hp"] + e["b"] for e in s["en"] if e.get("al", 1))
                    emax = sum(e["m"] for e in s["en"])
                    turns[t] = (s["pl"]["hp"], s["pl"]["m"], ehp, emax, len(s.get("pot") or []))
            if len(turns) < 1:
                continue
            ts = sorted(turns)
            p0, pm, e0, em0, pot0 = turns[ts[0]]
            p1, _, e1, em1, pot1 = turns[ts[-1]]
            n = len(ts)
            dealt = [max(0, turns[ts[i]][2] - turns[ts[i + 1]][2]) for i in range(n - 1)]
            taken = [max(0, turns[ts[i]][0] - turns[ts[i + 1]][0]) for i in range(n - 1)]
            fights.append({"act": c["act"], "boss": c["encounter"].replace("_BOSS", ""), "won": won, "turns": n,
                           "hp0": p0 / max(1, pm), "boss_left": e1 / max(1, em1) if not won else 0.0,
                           "dealt": mean(dealt) if dealt else float("nan"), "taken": mean(taken) if taken else float("nan"),
                           "emax": em0, "pot_used": max(0, pot0 - pot1), "pot0": pot0})
    print(f"{name}：{len(files)} 局回放，Boss 战 {len(fights)} 场")
    for act in range(3):
        fa = [x for x in fights if x["act"] == act]
        if not fa:
            continue
        lost = [x for x in fa if not x["won"]]; won = [x for x in fa if x["won"]]
        print(f"  第{act + 1}幕 {len(fa)} 场 胜 {len(won)}：回合 赢 {mean(x['turns'] for x in won):.1f} / 输 {mean(x['turns'] for x in lost):.1f}；"
              f"每回合打掉（含格挡）{mean(x['dealt'] for x in fa if x['dealt'] == x['dealt']):.1f}，每回合掉血 {mean(x['taken'] for x in fa if x['taken'] == x['taken']):.1f}；"
              f"输的时候 Boss 还剩 {mean(x['boss_left'] for x in lost):.2f}（中位 {st.median([x['boss_left'] for x in lost]) if lost else float('nan'):.2f}）；"
              f"开打药水 {mean(x['pot0'] for x in fa):.2f}、用掉 {mean(x['pot_used'] for x in fa):.2f}")
        if lost:
            near = mean(x["boss_left"] < 0.2 for x in lost); far = mean(x["boss_left"] > 0.5 for x in lost)
            print(f"    输的 {len(lost)} 场里：Boss 剩不到两成 {near:.2f}，剩一半以上 {far:.2f}")
        byb = defaultdict(list)
        for x in fa:
            byb[x["boss"]].append(x)
        for b, v in sorted(byb.items()):
            lv = [x for x in v if not x["won"]]
            print(f"    {b}: {len(v)} 场 胜率 {mean(x['won'] for x in v):.2f}，Boss 总血量 {mean(x['emax'] for x in v):.0f}，每回合打掉 {mean(x['dealt'] for x in v if x['dealt'] == x['dealt']):.1f}、"
                  f"掉血 {mean(x['taken'] for x in v if x['taken'] == x['taken']):.1f}，回合 {mean(x['turns'] for x in v):.1f}，输时 Boss 剩 {mean(x['boss_left'] for x in lv):.2f}")
