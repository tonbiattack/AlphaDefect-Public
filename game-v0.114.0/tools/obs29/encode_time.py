"""格式 28 与格式 29 在铁甲战士整局帧上的编码耗时（R3-格式29规格.md 1.1 第 7 条：全部合并后增加不超过 20%）。

tools/engine_bench.py 只按角色的默认格式开局（请求里不能带 obs_format），题目是第三幕 Boss 战，量不了格式 29 的整局帧。这里改用引擎自带的
STS2_TIMING 计时：Server.cs 对每个请求里 Observation.Build + Serialize 的墙钟按“encode”（局外地图另记“encode_map”）累计，进程关闭时打到标准错误。

  1. 录：检查点贪心（按格式 28 的帧）打 N 局铁甲战士整局（进阶 10，route_summary、loop_cut 同训练配置），录下动作序列；
  2. 量：开两个引擎（格式 28、29，都带 STS2_TIMING），同一动作序列逐步交替重放（每一步先发给一个、读完再发给另一个，两边承受同样的机器负载），
     关闭后读各自的平均编码耗时。重复 P 遍，单数遍先 28、双数遍先 29。两边候选数逐步核对，不同即报错。

  python -I tools/obs29/encode_time.py --dll engine/Sts2Headless/bin/Release/net9.0/Sts2Headless.dll \
      --ckpt ~/.local/share/sls2/runs/remote/board-silent-dread/ironclad-eval/high/v24505.pt --runs 8 --passes 4 --out 草稿/enc.json
同时运行的引擎进程：录的阶段 --jobs 个，量的阶段 2 × --jobs 个。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
from multiprocessing import Pool
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "tools"))

from sts2env.env import Sts2Env  # noqa: E402
from sts2env.protocol import parse  # noqa: E402

TIMING = re.compile(r"\[timing\]\s+(\S+)\s+(\d+)\s+次\s+平均\s+([\d.]+)\s+ms")


def reset_req(seed: str, fmt: int | None) -> dict:
    r = {"op": "reset", "mode": "run", "character": "IRONCLAD", "ascension": 10, "seed": seed, "route_summary": True, "loop_cut": True, "stall_turns": 8}
    if fmt is not None: r["obs_format"] = fmt
    return r


def req(e: Sts2Env, r: dict) -> bytes:
    e._proc.send(r)
    return e._proc.recv()


def record(args_t) -> dict:
    """检查点贪心打一局（格式 28），返回动作序列。"""
    dll, ckpt, seed, max_steps = args_t
    from obs29.creatures import ckpt_policy
    act = ckpt_policy(Path(ckpt))
    e = Sts2Env(dll=Path(dll))
    o = parse(req(e, reset_req(seed, 28)))
    acts: list[int] = []
    while not (o.done or o.num_actions == 0) and len(acts) < max_steps:
        k = act([o])[0]
        acts.append(k)
        o = parse(req(e, {"op": "step", "action": k}))
    e.close()
    return {"seed": seed, "actions": acts}


def measure(args_t) -> dict:
    """同一组动作序列在格式 28、29 两个引擎上逐步交替重放，返回两边的 [timing] 统计。"""
    dll, runs, first = args_t
    os.environ["STS2_TIMING"] = "1"
    tmpd = tempfile.TemporaryDirectory(prefix="enc29_")
    tmp = Path(tmpd.name)
    logs = {f: open(tmp / f"{f}.log", "w") for f in (28, 29)}
    envs = {f: Sts2Env(dll=Path(dll), log=logs[f]) for f in (28, 29)}
    order = (first, 57 - first)
    frames, bad = 0, 0
    for run in runs:
        obs = {}
        for f in order: obs[f] = parse(req(envs[f], reset_req(run["seed"], f)))
        for k in run["actions"]:
            if obs[28].num_actions != obs[29].num_actions: bad += 1; break
            for f in order: obs[f] = parse(req(envs[f], {"op": "step", "action": k}))
            frames += 1
    out = {"frames": frames, "desync": bad}
    for f in (28, 29):
        envs[f].close(); logs[f].close()
        st = {}
        for line in (tmp / f"{f}.log").read_text(encoding="utf-8", errors="replace").splitlines():
            m = TIMING.search(line)
            if m: st[m.group(1)] = {"n": int(m.group(2)), "ms": float(m.group(3))}
        out[str(f)] = st
    tmpd.cleanup()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dll", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--runs", type=int, default=8)
    ap.add_argument("--passes", type=int, default=4)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=4000)
    ap.add_argument("--trajs", default="", help="已录好的动作序列（JSON）；没有就先录，录完存到 --out 旁边")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    if a.trajs and Path(a.trajs).exists():
        runs = json.loads(Path(a.trajs).read_text())["runs"]
    else:
        t0 = time.time()
        with Pool(a.jobs) as p:
            runs = p.map(record, [(a.dll, a.ckpt, f"ENC29_{k}", a.max_steps) for k in range(a.runs)])
        Path(str(out) + ".trajs.json").write_text(json.dumps({"runs": runs}))
        print(f"录了 {len(runs)} 局，共 {sum(len(r['actions']) for r in runs)} 步，{time.time() - t0:.0f}s", flush=True)
    shards = [runs[i::a.jobs] for i in range(a.jobs)]
    passes = []
    for p in range(a.passes):
        first = 28 if p % 2 == 0 else 29
        t0 = time.time()
        with Pool(a.jobs) as pool:
            res = pool.map(measure, [(a.dll, s, first) for s in shards if s])
        agg: dict = {"first": first, "frames": sum(r["frames"] for r in res), "desync": sum(r["desync"] for r in res)}
        for f in ("28", "29"):
            for key in ("encode", "encode_map", "step", "decision"):
                n = sum(r[f].get(key, {}).get("n", 0) for r in res)
                tot = sum(r[f].get(key, {}).get("n", 0) * r[f].get(key, {}).get("ms", 0.0) for r in res)
                agg[f"{f}_{key}_n"] = n; agg[f"{f}_{key}_ms"] = tot / max(1, n)
            n = agg[f"{f}_encode_n"] + agg[f"{f}_encode_map_n"]
            agg[f"{f}_all_ms"] = (agg[f"{f}_encode_n"] * agg[f"{f}_encode_ms"] + agg[f"{f}_encode_map_n"] * agg[f"{f}_encode_map_ms"]) / max(1, n)
        for f in ("28", "29"):     # 引擎这一侧每步的总耗时：走一步 + 取决策 + 编码（含地图），按步平均
            agg[f"{f}_engine_ms"] = (sum(agg[f"{f}_{k}_n"] * agg[f"{f}_{k}_ms"] for k in ("encode", "encode_map", "step", "decision"))) / max(1, agg["frames"])
        agg["ratio_engine"] = agg["29_engine_ms"] / agg["28_engine_ms"]
        agg["ratio_encode"] = agg["29_encode_ms"] / agg["28_encode_ms"]
        agg["ratio_all"] = agg["29_all_ms"] / agg["28_all_ms"]
        agg["seconds"] = round(time.time() - t0, 1)
        passes.append(agg)
        print(f"第 {p + 1} 遍（先 {first}）：帧 {agg['frames']}，失步 {agg['desync']}；局内编码 28 {agg['28_encode_ms']:.3f} ms、29 {agg['29_encode_ms']:.3f} ms"
              f"（{100 * (agg['ratio_encode'] - 1):+.1f}%）；含地图 28 {agg['28_all_ms']:.3f}、29 {agg['29_all_ms']:.3f}（{100 * (agg['ratio_all'] - 1):+.1f}%）；引擎每步合计 28 {agg['28_engine_ms']:.3f}、29 {agg['29_engine_ms']:.3f}（{100 * (agg['ratio_engine'] - 1):+.1f}%）；{agg['seconds']}s",
              flush=True)
    s28 = sum(p["28_encode_ms"] for p in passes); s29 = sum(p["29_encode_ms"] for p in passes)
    a28 = sum(p["28_all_ms"] for p in passes); a29 = sum(p["29_all_ms"] for p in passes)
    print(f"各遍合计：局内编码 {100 * (s29 / s28 - 1):+.1f}%，含地图 {100 * (a29 / a28 - 1):+.1f}%")
    out.write_text(json.dumps({"passes": passes, "runs": len(runs), "steps": sum(len(r["actions"]) for r in runs)}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
