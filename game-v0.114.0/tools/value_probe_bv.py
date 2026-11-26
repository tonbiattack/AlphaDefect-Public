"""局外价值探针在构筑价值势能（决策记录 253）前后的可比统计（2026-11-03）。

势能改变了价值头要学的目标：有势能时价值预测的是“含势能的剩余回报”G′ = G − Φ(当前)。势能本身是可预测的那部分，
直接拿 V′ 对 G′ 算 EV 会机械地变低，不能和势能前的基线比。这里把剩余的势能塑形从两边扣掉：
不含势能的剩余回报 G = G′ − S（S 为这一步起往后的势能塑形合计，含局末清零），价值头对 G 的预测为 V′ − S（S 在采集时可知）。
只用整局档案（episodes/*.npz，tools/value_probe 的格式；有 reward_component_buildvalue_potential* 才算势能，没有就当 0）。

按时间段分组：势能前（版本 < bv_start）、爬坡期（bv_start ≤ 版本 < bv_start + bv_ramp）、满额后；每组按幕列 EV、MSE、平均偏差。
    python3 tools/value_probe_bv.py --dir 档案目录 [--dir ...] --bv-start 157500 --bv-ramp 300 [--last 2000] [--out 结果.json]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def episode(path: Path):
    try:
        z = np.load(path)
        meta = json.loads(bytes(z["_meta"]).decode())
    except Exception:
        return None
    if not meta.get("mc_valid"):
        return None
    mi = z["macro_index"]
    bv = np.zeros(len(z["rewards"]))
    for k in ("reward_component_buildvalue_potential", "reward_component_buildvalue_potential_terminal"):
        if k in z.files:
            bv = bv + z[k]
    rest = np.cumsum(bv[::-1])[::-1]            # 这一步起往后的势能塑形合计
    g_shaped = z["mc_behavior_target"].astype(np.float64)
    v_shaped = z["values"][mi].astype(np.float64)
    s = rest[mi]
    return {"version": int(meta["version_start"]), "root": meta["root_id"], "act": z["act"][mi].astype(int),
            "v": v_shaped - s, "g": g_shaped - s, "v_raw": v_shaped, "g_raw": g_shaped, "bv_total": float(bv.sum())}


def stats(v, g):
    if len(g) < 2:
        return None
    err = v - g
    var = float(np.var(g))
    return {"n": int(len(g)), "ev": round(1 - float(np.var(err)) / var, 4) if var > 0 else None,
            "mse": round(float(np.mean(err ** 2)), 4), "bias": round(float(np.mean(err)), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", nargs="+", required=True)
    ap.add_argument("--bv-start", type=int, required=True)
    ap.add_argument("--bv-ramp", type=int, default=300)
    ap.add_argument("--last", type=int, default=2000, help="每组最多取最新的多少局")
    ap.add_argument("--before", type=int, default=2000, help="势能前取最新的多少局作基线")
    ap.add_argument("--out", default=None)
    ap.add_argument("--names", default="势能前,爬坡期,满额后", help="三组的名字（逗号分隔）；按别的改动点分组时用（决策记录 261：上线前、上线后 300 版内、之后）")
    a = ap.parse_args()
    n0, n1, n2 = a.names.split(",")
    files = [p for d in a.dir for p in Path(d).glob("*.npz")]
    eps = [e for e in map(episode, files) if e]
    end = a.bv_start + a.bv_ramp
    groups = {n0: [e for e in eps if e["version"] < a.bv_start],
              n1: [e for e in eps if a.bv_start <= e["version"] < end],
              n2: [e for e in eps if e["version"] >= end]}
    out = {"bv_start": a.bv_start, "bv_ramp_end": end, "groups": {}}
    for name, es in groups.items():
        es = sorted(es, key=lambda e: e["version"])[-(a.before if name == n0 else a.last):]
        if not es:
            out["groups"][name] = {"局数": 0}
            continue
        rec = {"局数": len(es), "版本": [es[0]["version"], es[-1]["version"]],
               "每局势能合计": round(float(np.mean([e["bv_total"] for e in es])), 4)}
        cat = lambda k, m=None: np.concatenate([e[k] if m is None else e[k][e["act"] == m] for e in es])
        rec["扣掉势能（可与势能前比）"] = {"全部": stats(cat("v"), cat("g")), **{f"第{m}幕": stats(cat("v", m), cat("g", m)) for m in (1, 2, 3)}}      # 档案里的幕从 1 起
        rec["含势能（原始）"] = {"全部": stats(cat("v_raw"), cat("g_raw"))}
        out["groups"][name] = rec
        print(name, json.dumps(rec, ensure_ascii=False), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
