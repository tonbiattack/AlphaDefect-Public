"""“有救命动作却结束回合”的统计（第十二份外部评审，决策记录 205）。从评测的整局回放（eval_replays）里找：

  模型选了结束回合，而观测的回合结束预演（玩家生物特征 17：现在结束回合预计掉多少血）不小于当前生命，
  也就是按确定性结算这回合会死；再看手里能打的格挡牌（按能量做 0–1 背包，格挡取手牌的界面预览）
  加上能用的格挡药水（目录里的 Block 数值），合计能不能把预计掉血压到生命以下。

只是近似：不算抽牌、能量药水、敌人的易伤与虚弱变化、需要目标的牌，也不算回血；固化药水这类按当前格挡翻倍的药水另算。
目的是把“带药死亡”里真有确定救命动作的局面挑出来，看模型给救命动作多少概率，不是判定每一例都是失误。

    python3 tools/lethal_check.py 回放目录 --min-version 107000 --out lethal.json
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOUBLE_BLOCK = {"POTION.FORTIFIER"}      # 固化药水：当前格挡翻 3 倍（游戏的 Fortifier，按当前格挡算）


def catalog() -> tuple[dict, dict, list[str]]:
    cat = json.load(open(sorted((REPO / "engine/catalog").glob("v*/catalog.json"))[-1], encoding="utf-8"))
    pot = {e["id"]: int((e.get("vars") or {}).get("Block", 0)) for e in cat["entities"] if e.get("category") == "POTION"}
    names = [l.strip() for l in open(REPO / "engine/catalog/vocab.txt", encoding="utf-8")]
    return pot, {}, names


def best_block(hand: list, energy: int) -> int:
    """能打的格挡牌在能量内最多凑多少格挡（0–1 背包）。hand 每项 [词表 ID, 升级, 费用（X 费为 −1）, 能否打出, 伤害, 格挡]。"""
    items = [(max(0, h[2]), h[5]) for h in hand if h[3] and h[5] > 0 and h[2] >= 0]
    dp = [0] * (max(0, energy) + 1)
    for c, b in items:
        for e in range(len(dp) - 1, c - 1, -1):
            dp[e] = max(dp[e], dp[e - c] + b)
    return dp[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--min-version", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    pot_block, _, names = catalog()
    n_games = lethal = rescuable = died = 0
    p_rescue, examples = [], []
    for f in sorted(Path(a.dir).glob("v*.jsonl.gz")):
        if int(f.name[1:].split("_")[0]) < a.min_version:
            continue
        try:
            lines = [json.loads(x) for x in gzip.open(f, "rt", encoding="utf-8")]
        except Exception:
            continue
        n_games += 1
        head, steps = lines[0], lines[1:]
        for i, s in enumerate(steps):
            c, pl = s.get("c") or [], s.get("pl")
            if not pl or not c or not (0 <= s["a"] < len(c)) or c[s["a"]][0] != 1:      # 只看选了结束回合的出牌决策
                continue
            hp, block, loss = pl["hp"], pl.get("b", 0), pl["et"][1]
            if loss < hp:
                continue
            lethal += 1
            need = loss - hp + 1
            card = best_block(s.get("hand") or [], int(s["g"][5]))
            pots = [(k, x) for k, x in enumerate(c) if x[0] == 3 and x[1]]      # 用药水的候选：[3, [类型, 词表 ID, ...], 目标, ...]
            pb = [pot_block.get(x[1][1], 0) for _, x in pots]
            dbl = any(names[x[1][1]] in DOUBLE_BLOCK for _, x in pots if 0 <= x[1][1] < len(names))
            gain = card + max(pb + [0]) + (2 * (block + card) if dbl else 0)
            rescue_idx = [k for k, x in enumerate(c) if (x[0] == 2 and x[1] and any(h[0] == x[1][1] and h[5] > 0 for h in s.get("hand") or []))]
            rescue_idx += [k for (k, x), b in zip(pots, pb) if b > 0 or names[x[1][1]] in DOUBLE_BLOCK]
            if gain >= need:
                rescuable += 1
                pr = dict((k, p) for k, p in s.get("pr", []))
                p_rescue.append(sum(pr.get(k, 0.0) for k in rescue_idx))
                lost_here = not head["win"] and not any(x.get("g", [0, 0])[1] != s["g"][1] for x in steps[i + 1:])
                died += int(lost_here)
                if len(examples) < 30:
                    examples.append({"file": f.name, "step": i, "hp": hp, "block": block, "loss": loss, "need": need, "card_block": card,
                                     "potions": [names[x[1][1]] for _, x in pots], "p_rescue_top5": round(p_rescue[-1], 3), "died_here": lost_here})
    out = {"games": n_games, "lethal_end_turn": lethal, "rescuable": rescuable, "rescuable_then_died": died,
           "p_rescue_median": sorted(p_rescue)[len(p_rescue) // 2] if p_rescue else None, "examples": examples}
    if a.out:
        json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{n_games} 局：选了结束回合而预演致死 {lethal} 次，其中按格挡牌与药水近似能救 {rescuable} 次（随后死在这一层 {died} 次），"
          f"救命动作在前五项里的概率合计中位 {out['p_rescue_median']}")


if __name__ == "__main__":
    main()
