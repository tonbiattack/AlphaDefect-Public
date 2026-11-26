"""汇总 tools/pimc_rust_eval.py 的逐题结果：各臂连打两个 Boss 的胜率、与贪心的配对差（95% 区间、精确符号检验 p 值）、
按第一个 Boss 分组、根局面核对（Rust 重放到同一局面与 C# 帧逐项比较）、搜索次数与耗时。

  python3 tools/pimc_rust_report.py 结果目录 [--json 汇总.json]
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path


def sign_p(x: int, y: int) -> float:
    """双侧精确符号检验（McNemar 精确版）：只有一边赢的题数 x、y。"""
    n = x + y
    if n == 0:
        return 1.0
    k = min(x, y)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def paired(rows: list[dict], arm: str, base: str = "greedy") -> dict:
    n = len(rows)
    a = [int(r[arm]["win"]) for r in rows]
    b = [int(r[base]["win"]) for r in rows]
    only_a = sum(x and not y for x, y in zip(a, b))
    only_b = sum(y and not x for x, y in zip(a, b))
    d = [x - y for x, y in zip(a, b)]
    m = sum(d) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1)) if n > 1 else 0.0
    se = sd / math.sqrt(n)
    return {"n": n, arm: sum(a), base: sum(b), "rate_" + arm: round(sum(a) / n, 4), "rate_" + base: round(sum(b) / n, 4),
            "diff_pp": round(100 * m, 2), "se_pp": round(100 * se, 2), "ci95_pp": [round(100 * (m - 1.96 * se), 2), round(100 * (m + 1.96 * se), 2)],
            "only_" + arm: only_a, "only_" + base: only_b, "p_sign": round(sign_p(only_a, only_b), 5)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    rows = []
    for f in sorted(Path(a.dir).glob("r_s*.jsonl")):
        rows += [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
    rows.sort(key=lambda r: r["i"])
    arms = [k for k in ("greedy", "rs", "imp", "cs") if rows and k in rows[0]]
    out: dict = {"problems": len(rows), "arms": arms}
    print(f"{len(rows)} 道题，臂 {arms}")
    for arm in arms:
        if arm == "greedy":
            continue
        pr = paired(rows, arm)
        out[f"{arm}_vs_greedy"] = pr
        print(f"{arm} 对贪心：{pr[arm]}/{pr['n']}（{100 * pr['rate_' + arm]:.1f}%）对 {pr['greedy']}/{pr['n']}（{100 * pr['rate_greedy']:.1f}%），"
              f"差 {pr['diff_pp']:+.2f} ± {pr['se_pp']:.2f} 个百分点，95% 区间 {pr['ci95_pp']}，只有搜索赢 {pr['only_' + arm]}、只有贪心赢 {pr['only_greedy']}，p = {pr['p_sign']}")
        by = {}
        for enc in sorted({r["encounter"] for r in rows}):
            sub = [r for r in rows if r["encounter"] == enc]
            by[enc] = {"n": len(sub), arm: sum(r[arm]["win"] for r in sub), "greedy": sum(r["greedy"]["win"] for r in sub)}
        out[f"{arm}_by_first_boss"] = by
        print("  按第一个 Boss：" + "；".join(f"{k} {v[arm]}/{v['n']} 对 {v['greedy']}/{v['n']}" for k, v in by.items()))
        both2 = [r for r in rows if r[arm].get("stage", 0) >= 1 or r[arm]["win"]]
        reach = (sum(1 for r in rows if r[arm]["win"] or r[arm].get("stage", 0) >= 1), sum(1 for r in rows if r["greedy"]["win"] or r["greedy"].get("stage", 0) >= 1))
        out[f"{arm}_reach_second"] = reach
        print(f"  打到第二场：{reach[0]} 对 {reach[1]}")
        S = [r[arm] for r in rows]
        dec = sum(s["decisions"] for s in S); se = sum(s["searched"] for s in S); ch = sum(s["changed"] for s in S); sims = sum(s["sims"] for s in S)
        times = sorted(t for s in S for t in s.get("times", []))
        q = lambda p: times[min(len(times) - 1, int(p * len(times)))] if times else None
        out[f"{arm}_search"] = {"decisions": dec, "searched": se, "changed": ch, "sims": sims, "rollout_errors": sum(s.get("rollout_errors", 0) for s in S), "fallback": sum(s.get("fallback", 0) for s in S),
                                "truth_errors": sum(bool(s.get("error")) for s in S),
                                "time_p50_s": q(0.5), "time_p95_s": q(0.95), "time_mean_s": round(sum(times) / len(times), 4) if times else None}
        print(f"  每题平均 {dec / len(rows):.1f} 个决策、搜索 {se / len(rows):.1f} 次、改选 {ch / len(rows):.1f} 次（{100 * ch / max(1, se):.1f}%）、推演 {sims / len(rows):.0f} 次；"
              f"推演报错 {out[f'{arm}_search']['rollout_errors']}，根局面不可用退回贪心 {out[f'{arm}_search']['fallback']} 次，真实对局报错 {out[f'{arm}_search']['truth_errors']}")
        if times:
            print(f"  每次搜索耗时（多题并行时的墙钟）：中位 {q(0.5):.3f} 秒，p95 {q(0.95):.3f} 秒")
        if "root_checked" in S[0]:
            chk = sum(s["root_checked"] for s in S); ok = sum(s["root_matched"] for s in S)
            mism = Counter()
            for s in S:
                mism.update(s["root_mismatch"])
            clean = sum(1 for s in S if s["first_bad"] is None)
            out[f"{arm}_root"] = {"checked": chk, "matched": ok, "rate": round(ok / max(1, chk), 4), "problems_all_match": clean,
                                  "mismatch_by_segment": dict(mism.most_common())}
            print(f"  根局面核对：{ok}/{chk} 个决策一致（{100 * ok / max(1, chk):.2f}%），全程一致的题 {clean}/{len(rows)}")
            print("  不一致的第一处：" + "；".join(f"{k} {v}" for k, v in mism.most_common(12)))
            okl = sum(s["root_matched_loose"] for s in S)
            ml = Counter()
            for s in S:
                ml.update(s["root_mismatch_loose"])
            cleanl = sum(1 for s in S if s["first_bad_loose"] is None)
            fb = sorted(s["first_bad_loose"] for s in S if s["first_bad_loose"] is not None)
            out[f"{arm}_root"].update(matched_loose=okl, rate_loose=round(okl / max(1, chk), 4), problems_all_match_loose=cleanl,
                                      first_bad_loose_at_start=sum(1 for x in fb if x == 0),
                                      first_bad_loose_median=fb[len(fb) // 2] if fb else None,
                                      mismatch_by_segment_loose=dict(ml.most_common()))
            print(f"  宽松口径（不比金币、第二个 Boss、Boss 数、删牌次数与价格）：{okl}/{chk}（{100 * okl / max(1, chk):.2f}%），全程一致的题 {cleanl}/{len(rows)}，"
                  f"开局就不一致 {out[f'{arm}_root']['first_bad_loose_at_start']} 题，其余第一次不一致的决策序号中位 {out[f'{arm}_root']['first_bad_loose_median']}")
            print("  宽松口径不一致的第一处：" + "；".join(f"{k} {v}" for k, v in ml.most_common(12)))
            errs = Counter((s.get("rollout_error_text") or "")[:80] for s in S if s.get("rollout_error_text"))
            out[f"{arm}_rollout_error_texts"] = dict(errs.most_common(10))
            for k, v in errs.most_common(6):
                print(f"  推演报错 {v} 题：{k}")
    if a.json:
        Path(a.json).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
