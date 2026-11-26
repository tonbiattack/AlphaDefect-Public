"""逐种子级联的工作进程（2026-11-02，决策记录 251 补记二）：一个进程装着全部快照，带一组引擎；每个种子从给定的快照往下打，
输了就立刻换下一个快照接着打这个种子，直到通关或快照用完。种子之间互不依赖，所以把种子分给各个进程，各自走完，不用跨机器协调。
用户：“一个种子，如果一个模型输掉了，就可以让下一个补位了。如果这样子可以充分调度性能的话，你考虑一下。”

出手同复测（tools/ability_test.py → evaluator.play）：纯贪心逐候选取最大、带路线摘要（route_summary），整局步数上限 6000（超过按卡住、算输）。
每一步把等决策的局面按快照分组，各自批量推理。

    python -I tools/cascade_worker.py --models 快照表.txt --tasks 任务.json --out 结果.jsonl --envs 30 --device cuda
      快照表：每行 “名字 检查点路径”，按级联顺序；任务：{"tasks": [[种子, 起始快照下标], ...]}
      结果：每局一行 {"seed", "level", "model", "win", "floor", "stuck"}；已有结果的（种子, 下标）跳过，可续跑
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

from sts2env.env import EngineError, VecEnv  # noqa: E402
from sts2rl.actor import MODE_RUN  # noqa: E402
from sts2rl.features import Ragged, collate_ragged  # noqa: E402
from sts2rl.model import ModelConfig, PolicyValueNet, load_state  # noqa: E402

MAX_STEPS = 6000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--envs", type=int, default=30)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--character", default="DEFECT")
    ap.add_argument("--ascension", type=int, default=10)
    a = ap.parse_args()
    torch.set_num_threads(2)
    tasks = json.loads(Path(a.tasks).read_text(encoding="utf-8"))["tasks"]
    lo = min((lv for _, lv in tasks), default=0)     # 排在最小起点之前的快照用不到，不载入（省显存）
    names, nets = [], []
    for line in Path(a.models).read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        name, path = line.split()[:2]
        if len(nets) < lo:
            names.append(name); nets.append(None); continue
        blob = torch.load(path, map_location="cpu", weights_only=False)
        net = PolicyValueNet(ModelConfig(**blob["config"]["model"])).eval().to(a.device)
        load_state(net, blob["state"])
        names.append(name); nets.append(net)
    m = len(nets)
    done = set()
    outp = Path(a.out)
    if outp.exists():
        for line in open(outp, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                done.add((r["seed"], r["level"]))
                if r["win"]:
                    done.add((r["seed"], "won"))
    q: deque = deque()
    for seed, lv in tasks:
        if (seed, "won") in done:
            continue
        while lv < m and (seed, lv) in done:      # 续跑：已打过且输掉的快照跳过
            lv += 1
        if lv < m:
            q.append((seed, lv))
    print(f"{len(nets)} 个快照，{len(q)} 个种子待打，{a.envs} 个引擎", flush=True)
    fh = open(outp, "a", encoding="utf-8")
    env = VecEnv(min(a.envs, max(1, len(q))))
    slot: dict[int, list] = {}      # 引擎下标 → [种子, 快照下标, 步数]
    obs: dict[int, object] = {}
    t0, n_games = time.time(), 0

    def start(i: int) -> None:
        if not q:
            slot.pop(i, None); return
        seed, lv = q.popleft()
        slot[i] = [seed, lv, 0]
        env.send(i, {"op": "reset", "mode": "run", "character": a.character, "ascension": a.ascension, "seed": seed,
                     "snapshots": False, "route_summary": True})

    def finish(i: int, win: bool, floor: int, stuck: bool) -> None:
        nonlocal n_games
        seed, lv, _ = slot[i]
        fh.write(json.dumps({"seed": seed, "level": lv, "model": names[lv], "win": bool(win), "floor": int(floor), "stuck": bool(stuck)},
                            ensure_ascii=False) + "\n"); fh.flush()
        n_games += 1
        if not win and lv + 1 < m:
            q.appendleft((seed, lv + 1))      # 输了：下一个快照立刻补位
        start(i)

    for i in range(len(env)):
        start(i)
    while slot:
        for i, o in env.recv_ready(timeout=1.0):
            if i not in slot:
                continue
            if isinstance(o, EngineError):
                finish(i, False, 0, True); continue
            slot[i][2] += 1
            if o.done or o.stuck or slot[i][2] > MAX_STEPS:
                finish(i, bool(o.done and o.victory), int(o.floor), bool(o.stuck or slot[i][2] > MAX_STEPS))
            else:
                obs[i] = o
        if not obs:
            continue
        by: dict[int, list[int]] = {}
        for i in obs:
            by.setdefault(slot[i][1], []).append(i)
        with torch.no_grad():
            for lv, ids in by.items():
                rg = Ragged.from_obs([obs[i] for i in ids], np.full(len(ids), MODE_RUN))
                logits = nets[lv](collate_ragged(rg).to(a.device))[0]
                for k, i in enumerate(ids):
                    env.send_step(i, int(logits[k, :obs[i].num_actions].argmax()))
        obs.clear()
    print(f"完成：{n_games} 局，用时 {time.time() - t0:.0f} 秒", flush=True)


if __name__ == "__main__":
    main()
