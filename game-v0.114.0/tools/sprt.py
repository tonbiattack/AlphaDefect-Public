"""配对序贯检验（SPRT，2026-10-31 起，后续计划第九节、决策记录 219）：两个检查点在同一批种子上的整局输赢，逐种子配对。

只看只有一方赢的种子（共 x + y 个，x 为新版赢、y 为旧版赢）：零假设 H0 新版不比旧版强（差 0），备择 H1 新版强 δ 个百分点。
给定分歧率 d（只有一方赢的种子占比，用已打的数据估，不低于 0.1），只有一方赢时新版赢的概率在 H0 下为 0.5，在 H1 下为
q1 = 0.5 + δ / (2d)。对数似然比 LLR = x·ln(q1/0.5) + y·ln((1 − q1)/0.5)；越过 ln((1 − β)/α) 接受 H1，低于 ln(β/(1 − α)) 接受 H0。
d 用已打的数据估是近似（Fishtest 一类做法里的“插值”），每批（默认 400 个种子）更新一次；到上限仍未决就按未决报告，附上差与 95% 区间。

  python3 tools/sprt.py 新版结果目录 旧版结果目录 [--delta 0.02] [--alpha 0.05] [--beta 0.05] [--json 输出.json]
结果目录里是 tools/ability_test.py 的 runs_s*.jsonl。
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load(d: str) -> dict[str, int]:
    out = {}
    for f in sorted(Path(d).glob("runs_s*.jsonl")):
        for line in open(f, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                out[r["seed"]] = int(bool(r["win"]) and not r.get("stuck"))
    return out


def sprt(new: dict, old: dict, delta: float, alpha: float, beta: float) -> dict:
    seeds = sorted(s for s in new if s in old)
    n = len(seeds)
    x = sum(new[s] > old[s] for s in seeds); y = sum(old[s] > new[s] for s in seeds)
    d = max(0.1, (x + y) / max(1, n))
    q1 = min(0.95, 0.5 + delta / (2 * d))
    llr = x * math.log(q1 / 0.5) + y * math.log((1 - q1) / 0.5)
    hi, lo = math.log((1 - beta) / alpha), math.log(beta / (1 - alpha))
    diff = (x - y) / max(1, n)
    se = math.sqrt(max(0.0, (x + y) / max(1, n) - diff ** 2) / max(1, n - 1))
    verdict = "新版更强（接受 H1）" if llr >= hi else ("没有强出 δ（接受 H0）" if llr <= lo else "未决")
    return {"n": n, "win_new": sum(new[s] for s in seeds), "win_old": sum(old[s] for s in seeds), "only_new": x, "only_old": y,
            "d": round((x + y) / max(1, n), 4), "diff": round(diff, 4), "ci95": [round(diff - 1.96 * se, 4), round(diff + 1.96 * se, 4)],
            "delta": delta, "q1": round(q1, 4), "llr": round(llr, 3), "bounds": [round(lo, 3), round(hi, 3)], "verdict": verdict}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("new"); ap.add_argument("old")
    ap.add_argument("--delta", type=float, default=0.02); ap.add_argument("--alpha", type=float, default=0.05); ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    r = sprt(load(a.new), load(a.old), a.delta, a.alpha, a.beta)
    print(f"{r['n']} 个种子：新版 {r['win_new']}，旧版 {r['win_old']}；只有新版赢 {r['only_new']}、只有旧版赢 {r['only_old']}（d = {r['d']}）；"
          f"差 {100 * r['diff']:+.1f} 个百分点（95% 区间 {100 * r['ci95'][0]:+.1f} 至 {100 * r['ci95'][1]:+.1f}）；LLR {r['llr']}，界 {r['bounds']}：{r['verdict']}")
    if a.json:
        Path(a.json).write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
