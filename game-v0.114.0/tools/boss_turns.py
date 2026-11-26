"""第三幕 Boss 战的逐回合效率（第十一份外部评审，决策记录 204）：从评测的整局回放（eval_replays，每局一个 .jsonl.gz）里，
把第三幕两个 Boss 的每一场按回合展开，分 Boss、分输赢统计“前面稳、后面撑不住”的过程：

  敌方剩余生命（占开场的比例，实验体各形态的生命合计）、我方生命比例、这一回合打掉的敌方生命、
  状态牌与诅咒在战斗牌堆（手牌 + 抽牌堆 + 弃牌堆）里的比例、手里能打的牌数、第一张能力牌落地的回合。

每回合取这一回合第一个出牌决策时的局面。只读回放，不碰训练。

    python3 tools/boss_turns.py 回放目录 --min-version 106370 --out boss_turns.json
"""
from __future__ import annotations

import argparse
import gzip
import json
import statistics
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MAX_T = 20


def catalog_types() -> dict[int, str]:
    cat = json.load(open(sorted((REPO / "engine/catalog").glob("v*/catalog.json"))[-1], encoding="utf-8"))
    return {e["id"]: e.get("type", "") for e in cat["entities"] if e.get("category") == "CARD"}


def fights(steps: list[dict], head: dict) -> list[tuple[str, int, list[dict], bool]]:
    """这一局里第三幕的 Boss 战：[(遭遇, 层, 这场的战斗步, 是否打赢)]。"""
    out = []
    for c in head.get("combats") or []:
        if c["room"] != "Boss" or c["act"] != 2:
            continue
        ss = [s for s in steps if s.get("en") and s["g"][1] == c["floor"] and s["g"][8]]
        if not ss:
            continue
        lost = not head["win"] and head["floor"] == c["floor"]
        out.append((c["encounter"], c["floor"], ss, not lost))
    return out


def turns(ss: list[dict], ctype: dict[int, str]) -> list[dict]:
    """一场战斗按回合取第一个决策时的局面。"""
    by_t: dict[int, dict] = {}
    first_power = None
    for s in ss:
        t = int(s["g"][7])
        c = s.get("c") or []
        a = c[s["a"]] if 0 <= s["a"] < len(c) else None
        if a and a[0] == 2 and a[1] and ctype.get(a[1][1]) == "Power" and first_power is None:
            first_power = t
        if t in by_t:
            continue
        en = sum(e["hp"] for e in s["en"] if e.get("al"))
        hand = s.get("hand") or []
        piles = [x[0] for x in hand] + [x[0] for x in s.get("drawl") or []] + [x[0] for x in s.get("disc") or []]
        bad = sum(1 for v in piles if ctype.get(v) in ("Status", "Curse"))
        pl = s.get("pl") or {}
        by_t[t] = {"t": t, "en": en, "hp": pl.get("hp", s["g"][2]), "mhp": max(1, pl.get("m", s["g"][3])),
                   "status": bad / max(1, len(piles)), "playable": sum(1 for x in hand if x[3]), "hand": len(hand)}
    rows = [by_t[t] for t in sorted(by_t)]
    for k, r in enumerate(rows):
        r["dmg"] = max(0, r["en"] - rows[k + 1]["en"]) if k + 1 < len(rows) else None
    if rows:
        rows[0]["first_power"] = first_power
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--min-version", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ctype = catalog_types()
    agg: dict[tuple, dict] = defaultdict(lambda: defaultdict(list))
    meta: dict[tuple, dict] = defaultdict(lambda: {"n": 0, "turns": [], "first_power": []})
    n_games = 0
    for f in sorted(Path(a.dir).glob("v*.jsonl.gz")):
        if int(f.name[1:].split("_")[0]) < a.min_version:
            continue
        try:
            lines = [json.loads(x) for x in gzip.open(f, "rt", encoding="utf-8")]
        except Exception:
            continue
        n_games += 1
        head, steps = lines[0], lines[1:]
        for enc, floor, ss, won in fights(steps, head):
            rows = turns(ss, ctype)
            if not rows:
                continue
            key = (enc, "赢" if won else "输")
            m = meta[key]; m["n"] += 1; m["turns"].append(len(rows))
            m["first_power"].append(rows[0].get("first_power"))
            en0 = max(1, rows[0]["en"])
            for r in rows[:MAX_T]:
                g = agg[key][r["t"]]
                g.append({"en": r["en"] / en0, "hp": r["hp"] / r["mhp"], "status": r["status"], "playable": r["playable"],
                          "dmg": None if r["dmg"] is None else r["dmg"] / en0})
    mean = lambda xs: round(statistics.mean(xs), 3) if xs else None
    out = {"games": n_games, "min_version": a.min_version, "fights": {}}
    for key in sorted(meta):
        m = meta[key]
        fp = [x for x in m["first_power"] if x is not None]
        curve = {t: {"n": len(v), **{k: mean([x[k] for x in v if x[k] is not None]) for k in ("en", "hp", "status", "playable", "dmg")}}
                 for t, v in sorted(agg[key].items())}
        out["fights"][f"{key[0]}·{key[1]}"] = {"n": m["n"], "turns_median": statistics.median(m["turns"]),
                                              "first_power_median": statistics.median(fp) if fp else None,
                                              "no_power": round(1 - len(fp) / max(1, m["n"]), 3), "curve": curve}
    if a.out:
        json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{n_games} 局（第 {a.min_version:,} 版起）")
    for k, v in out["fights"].items():
        c = v["curve"]
        pick = lambda t, f: c.get(t, {}).get(f)
        print(f"{k}：{v['n']} 场，回合数中位 {v['turns_median']}，第一张能力牌中位第 {v['first_power_median']} 回合（没打能力牌 {v['no_power']:.0%}）；"
              f"敌方剩余 第 1/3/5/8 回合 {[pick(t, 'en') for t in (1, 3, 5, 8)]}，我方血量 {[pick(t, 'hp') for t in (1, 3, 5, 8)]}，"
              f"状态牌比例 {[pick(t, 'status') for t in (1, 3, 5, 8)]}，能打的牌 {[pick(t, 'playable') for t in (1, 3, 5, 8)]}")


if __name__ == "__main__":
    main()
