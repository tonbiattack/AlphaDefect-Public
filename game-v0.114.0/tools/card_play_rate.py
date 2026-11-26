"""各牌在评测回放里的出牌率（2026-11-01，决策记录 238）：按回合统计“这回合手里有这张牌、能量够、能打”时有没有打出，
并分房间（普通、精英、Boss），另记带着可打的这张牌结束回合（多余能量）的回合占比。只读回放，不碰训练。

    python3 tools/card_play_rate.py 回放目录 --min-version 136000 [--card CARD.DEFRAGMENT] [--min-turns 1500] [--out 结果.json]

回放格式见 python/sts2rl/replay.py：每步 hand 为 [词表 ID, 升级, 费用, 能否打出, 伤害, 格挡]，c 为候选，g[5] 能量、g[7] 回合、g[9] 房间类型（1 普通 2 精英 3 Boss）。
同一张牌的多个副本按一张算；按（卡、费用、房间）累计，费用取该回合第一次能打时的费用。
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
from multiprocessing import Pool
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOMS = {1: "普通", 2: "精英", 3: "Boss"}


def scan(f: Path) -> dict:
    out: dict = collections.defaultdict(lambda: [0, 0, 0])      # (卡, 费用, 房间) → [能打的回合数, 打出的回合数, 带着能打的它结束回合的回合数]
    try:
        lines = [json.loads(x) for x in gzip.open(f, "rt", encoding="utf-8")]
    except Exception:
        return {}
    turns: dict = collections.defaultdict(list)
    for s in lines[1:]:
        if s.get("hand") and s.get("c"):
            turns[(s["g"][1], s["g"][7])].append(s)
    for ss in turns.values():
        room = ss[0]["g"][9]
        first: dict = {}
        for s in ss:
            for h in s["hand"]:
                if h[3] and h[2] >= 0 and h[0] not in first and any(x[0] == 2 and x[1] and x[1][1] == h[0] for x in s["c"]):
                    first[h[0]] = (h[2], int(s["g"][5]))
        played = {h[0] for s in ss for x in [s["c"][s["a"]]] if x[0] == 2 and x[1] for h in s["hand"] if h[0] == x[1][1]}
        last = ss[-1]
        ended = last["c"][last["a"]][0] == 1
        for cid, (cost, en) in first.items():
            if cost > en:
                continue
            o = out[(cid, cost, room)]
            o[0] += 1
            o[1] += cid in played
            o[2] += cid not in played and ended and any(h[0] == cid and h[3] and h[2] <= int(last["g"][5]) for h in last["hand"])
    return dict(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--min-version", type=int, default=0)
    ap.add_argument("--kind", default="RT", help="文件名里版本号后的前缀（RT 复测、EVAL 评测）")
    ap.add_argument("--card", default=None)
    ap.add_argument("--min-turns", type=int, default=1500)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    names = [l.strip() for l in open(REPO / "engine/catalog/vocab.txt", encoding="utf-8")]
    fs = [f for f in sorted(Path(a.dir).glob(f"v*_{a.kind}*.jsonl.gz")) if int(f.name[1:].split("_")[0]) >= a.min_version]
    with Pool(a.procs) as p:
        res = p.map(scan, fs, chunksize=20)
    tot: dict = collections.defaultdict(lambda: [0, 0, 0])
    for r in res:
        for k, v in r.items():
            for i in range(3):
                tot[k][i] += v[i]
    by: dict = collections.defaultdict(lambda: {r: [0, 0, 0] for r in ROOMS})
    for (cid, cost, room), v in tot.items():
        if room in ROOMS:
            for i in range(3):
                by[cid][room][i] += v[i]
    rows = []
    for cid, d in by.items():
        n = sum(x[0] for x in d.values())
        if n < a.min_turns or (a.card and names[cid] != a.card):
            continue
        rows.append((names[cid], n, {ROOMS[r]: d[r] for r in ROOMS}))
    rows.sort(key=lambda x: x[2]["普通"][1] / max(1, x[2]["普通"][0]))
    print(f"{len(fs)} 局；每张牌：打出率 / 带着它结束回合占比（回合数）")
    for name, n, d in rows:
        cells = "  ".join(f"{k} {100 * v[1] / max(1, v[0]):.0f}%/{100 * v[2] / max(1, v[0]):.0f}%（{v[0]}）" for k, v in d.items())
        print(f"{name:32s}{cells}")
    if a.out:
        json.dump({n: d for n, _, d in rows}, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
