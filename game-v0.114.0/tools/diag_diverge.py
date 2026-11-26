"""平台期诊断 D2（2026-11-03）：两个快照打同一个种子时，第一处出手不同的地方在哪。
引擎按种子确定、评测是贪心，所以两局在第一处不同的出手之前局面完全一样；那一步就是两局分开的起点。
读两次测试的回放目录（ability_test 的 A10/replays/，每局一个 .jsonl.gz：首行是整局摘要，其后每行一步，
d 决策类型、a 选的动作、g[0..1] 幕与层、g[8] 是否在战斗中、pr 前几名动作的概率），输出汇总 JSON。
    python -I tools/diag_diverge.py --a 目录1 --b 目录2 --out 结果.json
"""
import argparse
import gzip
import json
import statistics as S
from collections import Counter
from pathlib import Path

KIND = {0: "卡住", 1: "战斗", 2: "选牌（战斗内外的选卡）", 3: "卡牌奖励", 4: "卡包", 5: "奖励领取", 6: "事件", 7: "休息处",
        8: "商店", 9: "宝箱", 10: "小游戏", 11: "地图选路", 12: "结束"}


def load(p: Path):
    with gzip.open(p, "rt", encoding="utf-8") as f:
        head = json.loads(f.readline())
        return head, [json.loads(l) for l in f if l.strip()]


def gap(pr):
    return (pr[0][1] - (pr[1][1] if len(pr) > 1 else 0.0)) if pr else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ra = {p.name.split("_", 1)[1]: p for p in Path(a.a).glob("*/replays/*.jsonl.gz")} or \
         {p.name.split("_", 1)[1]: p for p in Path(a.a).glob("replays/*.jsonl.gz")}
    rb = {p.name.split("_", 1)[1]: p for p in Path(a.b).glob("*/replays/*.jsonl.gz")} or \
         {p.name.split("_", 1)[1]: p for p in Path(a.b).glob("replays/*.jsonl.gz")}
    common = sorted(set(ra) & set(rb))
    rows = []
    for s in common:
        ha, sa = load(ra[s]); hb, sb = load(rb[s])
        i = 0
        while i < min(len(sa), len(sb)) and sa[i]["a"] == sb[i]["a"] and sa[i]["d"] == sb[i]["d"]:
            i += 1
        r = {"seed": s[:-9], "win_a": bool(ha["win"]), "win_b": bool(hb["win"]), "steps_a": len(sa), "steps_b": len(sb), "div": None}
        if i < min(len(sa), len(sb)):
            x, y = sa[i], sb[i]
            r["div"] = {"step": i, "kind": KIND.get(x["d"], x["d"]), "act": x["g"][0], "floor": x["g"][1], "combat": bool(x["g"][8]),
                        "gap_a": gap(x.get("pr")), "gap_b": gap(y.get("pr")), "p_a": x.get("pr"), "p_b": y.get("pr"), "a_a": x["a"], "a_b": y["a"],
                        "hp": x["g"][2], "maxhp": x["g"][3]}
        rows.append(r)
    div = [r for r in rows if r["div"]]
    flip = [r for r in div if r["win_a"] != r["win_b"]]
    same = [r for r in div if r["win_a"] == r["win_b"]]

    def summ(rs):
        if not rs:
            return {}
        d = [r["div"] for r in rs]
        g = [x["gap_a"] for x in d if x["gap_a"] is not None]
        return {"n": len(rs), "kind": Counter(x["kind"] for x in d).most_common(), "act": Counter(x["act"] for x in d).most_common(),
                "floor_median": S.median(x["floor"] for x in d), "floor_hist": sorted(Counter((x["floor"] // 5) * 5 for x in d).items()),
                "step_median": S.median(x["step"] for x in d), "combat_share": sum(x["combat"] for x in d) / len(d),
                "gap_a_median": S.median(g) if g else None, "gap_a_lt_0.1": sum(v < 0.1 for v in g) / len(g) if g else None,
                "gap_a_lt_0.3": sum(v < 0.3 for v in g) / len(g) if g else None}
    res = {"a": a.a, "b": a.b, "common": len(common), "identical": len(rows) - len(div),
           "outcome_flip": len(flip), "flip_a_wins": sum(r["win_a"] for r in flip),
           "all_div": summ(div), "flip": summ(flip), "same_outcome": summ(same), "rows": rows}
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
