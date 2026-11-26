"""带搜索评测的对局资料 → 胜率小模型用的整局回放（储君版胜率小模型，2026-11-09，储君决策记录第 30 条）。

tools/sim_pimc_eval.py --records 每个种子记一行：主干整局贪心的全部动作（tape）与每道题的分叉位置、两臂打完这一幕的动作。
“学生 + 第三幕搜索”实际打出的一局 = 主干分叉前的动作 + 搜索臂的动作（没到第三幕 Boss 的局两臂相同，就是主干）。
这里在真实引擎里按这串动作逐步重放（引擎是确定的），每步照 python/sts2rl/replay.describe 记下，存成 evaluator.save_replay 的格式，
概要的输赢取搜索臂的结局。策略概率（pr）不重要（胜率小模型不用），记为空。

    .venv/bin/python -I tools/winprob/pimc_replays.py --dll Sts2Headless.dll --records 目录 [目录 ...] --out 回放目录 [--shard 0/8]
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

from sts2env.env import Sts2Env  # noqa: E402
from sts2env.protocol import DecisionKind, parse  # noqa: E402
from sts2rl.replay import describe  # noqa: E402


def load_records(dirs: list[str]) -> dict[str, dict]:
    """各目录的 records_s*.jsonl 按种子去重（同一种子各处结果相同，sim_pimc_eval 的分片号与种子表一致时搜索随机数也相同）。"""
    out: dict[str, dict] = {}
    for d in dirs:
        for f in sorted(glob.glob(f"{d}/records_s*.jsonl")):
            for l in open(f, encoding="utf-8"):
                if l.strip():
                    r = json.loads(l)
                    out.setdefault(r["seed"], r)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dll", required=True)
    ap.add_argument("--records", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--character", default="REGENT")
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    k, n = map(int, a.shard.split("/"))
    recs = load_records(a.records)
    seeds = sorted(recs)[k::n]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    env = Sts2Env(dll=Path(a.dll), log=open(out / f"engine_s{k}.log", "w"))
    call = lambda req: (env._proc.send(req), parse(env._proc.recv()))[1]
    nz = np.zeros(64)
    done = bad = 0; t0 = time.time()
    for seed in seeds:
        name = out / f"pimc_{seed}.jsonl.gz"
        if name.exists():
            continue
        r = recs[seed]
        probs = r.get("problems") or []
        if probs:
            p = probs[0]
            acts, win, arm = r["tape"][:int(p["fork_at"])] + list(p["search_actions"]), bool(p["search_win"]), "search"
        else:
            acts, win, arm = list(r["tape"]), bool(r["win"]), "trunk"
        o = call({"op": "reset", "mode": "run", "character": a.character, "ascension": 10, "seed": seed, "route_summary": bool(r.get("route_summary", True))})
        steps, ok = [], True
        for i, act in enumerate(acts):
            if int(o.header[1]) == int(DecisionKind.GAME_OVER):
                ok = False; break                 # 动作还没用完游戏就结束：与资料对不上
            steps.append(json.dumps(describe(o, int(act), nz), separators=(",", ":")))
            o = call({"op": "step", "action": int(act)})
        if not ok:
            bad += 1
            print(f"{seed}：第 {len(steps)} 个动作时游戏已结束，资料对不上，跳过", flush=True)
            continue
        g = json.loads(steps[-1])["g"] if steps else [0, 0]
        head = {"version": r.get("version"), "seed": seed, "time": time.time(), "win": win, "floor": g[1], "act": g[0],
                "combats": [], "steps": len(steps), "arm": arm, "source": "sim_pimc_eval --records"}
        tmp = out / f".pimc_{seed}.tmp"
        with gzip.open(tmp, "wt", encoding="utf-8") as f:
            f.write(json.dumps(head, ensure_ascii=False) + "\n")
            for s in steps:
                f.write(s + "\n")
        os.replace(tmp, name)
        done += 1
    print(f"分片 {k}/{n}：写出 {done} 局，对不上 {bad} 局，用时 {time.time() - t0:.0f} 秒", flush=True)


if __name__ == "__main__":
    main()
