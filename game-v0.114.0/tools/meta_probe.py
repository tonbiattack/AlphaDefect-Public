"""局外选牌的搜索信号探针（2026-10-31，决策记录 217 延长的窗口）：只测“搜索在选牌上找不找得到显著更好的选项”，不改出手、不测胜率。

每个种子用整局模式贪心打完。在第一、二幕的卡牌奖励决策上（网络最大概率 < tau），取概率前 k 个候选（含跳过一类的候选），
每个候选在同一批 W 个重洗世界里推演：从整局开头按已走的动作重放、reseed（只换以后的随机数，玩家看得到的局面不变）、走这个候选，
之后贪心打到进入下一幕或死亡。得分：死亡时 0.02 ×（死亡层 − 当前层）；进入下一幕时 0.02 × 层差 + 0.5（本幕 Boss 奖励）
+ 局外价值头在下一幕第一个决策上的估值（与训练回报同一尺度）。逐世界配对，记下每个候选与贪心项的差、标准误。
零假设下（候选与贪心一样好），单个候选“差 > 2 个标准误”的概率按 t 分布（自由度 W − 1）约 4%。

  .venv/bin/python -I tools/meta_probe.py --ckpt 检查点.pt --out 结果.jsonl --seed-prefix PX --runs 40 --shard 0/4 --envs 12 --device cuda
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--seed-prefix", default="PX"); ap.add_argument("--runs", type=int, default=40); ap.add_argument("--shard", default="0/1")
    ap.add_argument("--envs", type=int, default=12); ap.add_argument("--device", default="cuda")
    ap.add_argument("--worlds", type=int, default=8); ap.add_argument("--k", type=int, default=3); ap.add_argument("--tau", type=float, default=0.9)
    ap.add_argument("--pmin", type=float, default=0.02); ap.add_argument("--acts", default="0,1"); ap.add_argument("--deadline", type=float, default=0)
    a = ap.parse_args()
    import numpy as np
    import torch
    from sts2env import DecisionKind, EngineError, VecEnv
    from sts2rl.actor import MODE_RUN
    from sts2rl.features import Ragged, collate_ragged
    from sts2rl.model import ModelConfig, PolicyValueNet, load_state
    torch.set_num_threads(1)
    blob = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    net = PolicyValueNet(ModelConfig(**blob["config"]["model"])).eval().to(a.device)
    load_state(net, blob["state"])
    acts = {int(x) for x in a.acts.split(",")}
    k0, n0 = map(int, a.shard.split("/"))
    seeds = [f"{a.seed_prefix}{j:04d}" for j in range(a.runs) if j % n0 == k0]
    base = {"mode": "run", "character": "DEFECT", "ascension": 10, "route_summary": True}
    out = open(a.out, "a", encoding="utf-8")
    t0 = time.time()

    # 任务：("main", 种子, 动作前缀) 贪心整局；("roll", 决策编号, 候选, 世界) 推演
    queue: list = [("main", s) for s in seeds]
    task_of: dict[int, dict] = {}       # 引擎 → 任务状态
    obs_of: dict[int, object] = {}
    probes: dict[int, dict] = {}        # 决策编号 → 现场与得分表
    pid_seq = [0]
    n_main_left = len(seeds)

    def score_leaf(st, o, value=None):
        if o is None or o.done or o.stuck or o.num_actions == 0:
            fl = o.floor if o is not None else st["floor0"]
            win = bool(o is not None and o.victory)
            return 0.02 * (fl - st["floor0"]) + (3.0 if win else 0.0)
        return 0.02 * (o.floor - st["floor0"]) + 0.5 + float(value)

    def finish_roll(st, sc):
        pr = probes[st["pid"]]
        pr["sc"][st["cand"]][st["world"]] = sc
        pr["left"] -= 1
        if pr["left"] == 0:
            g = pr["greedy"]
            rows = {}
            for c in pr["cands"]:
                if c == g:
                    continue
                d = [pr["sc"][c][w] - pr["sc"][g][w] for w in range(a.worlds)]
                m = sum(d) / len(d); sd = math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1)) if len(d) > 1 else 0.0
                rows[str(c)] = {"mean": round(m, 4), "se": round(sd / math.sqrt(len(d)), 4), "p": round(float(pr["p"][c]), 4)}
            out.write(json.dumps({"seed": pr["seed"], "act": pr["act"], "floor": pr["floor"], "step": pr["step"], "n": pr["n"], "greedy": g,
                                  "p_greedy": round(float(pr["p"][g]), 4), "labels": pr["labels"], "greedy_mean": round(sum(pr["sc"][g]) / a.worlds, 4),
                                  "alts": rows, "t": round(time.time() - t0)}, ensure_ascii=False) + "\n")
            out.flush()
            del probes[st["pid"]]

    with VecEnv(a.envs) as env:
        free = list(range(len(env)))

        def launch():
            while free and queue:
                i = free.pop()
                t = queue.pop()      # 后进先出：推演优先，主局慢慢推进
                if t[0] == "main":
                    task_of[i] = {"kind": "main", "seed": t[1], "prefix": []}
                    env.send(i, dict(base, op="reset", seed=t[1]))
                else:
                    _, pid, c, w = t
                    pr = probes[pid]
                    task_of[i] = {"kind": "roll", "pid": pid, "cand": c, "world": w, "floor0": pr["floor"], "act0": pr["act"], "steps": 0}
                    env.send(i, dict(base, op="reset", seed=pr["seed"], replay=pr["prefix"], reseed=(1000003 * (w + 1) + 7919 * pr["step"]) & 0xFFFFFFFF, first=c))

        def release(i):
            del task_of[i]
            free.append(i)

        launch()
        while task_of or queue:
            if a.deadline and time.time() > a.deadline:
                queue[:] = [t for t in queue if t[0] == "roll"]       # 截止后不再开新局，已开始的探针做完
            for i, o in env.recv_ready(timeout=0.5):
                st = task_of[i]
                if isinstance(o, EngineError):
                    if st["kind"] == "roll":
                        finish_roll(st, score_leaf(st, None))
                    else:
                        n_main_left -= 1
                    release(i)
                    continue
                if st["kind"] == "roll":
                    if o.done or o.stuck or o.num_actions == 0 or st["steps"] > 3000:
                        finish_roll(st, score_leaf(st, o)); release(i); continue
                    if o.act > st["act0"] and not o.in_combat:
                        st["leaf"] = True
                else:
                    if o.done or o.stuck or o.num_actions == 0 or len(st["prefix"]) > 6000:
                        n_main_left -= 1
                        print(f"  {st['seed']}：{'通关' if o.victory else '止于'} 第 {o.floor} 层（{time.time() - t0:.0f} 秒）", flush=True)
                        release(i); continue
                    if a.deadline and time.time() > a.deadline:
                        n_main_left -= 1
                        release(i); continue
                obs_of[i] = o
            launch()
            if not obs_of:
                continue
            ids = list(obs_of)
            batch = [obs_of[i] for i in ids]
            with torch.no_grad():
                logits, values = net(collate_ragged(Ragged.from_obs(batch, np.full(len(batch), MODE_RUN))).to(a.device))
            logits, values = logits.float().cpu(), values.float().cpu()
            for j, i in enumerate(ids):
                o = obs_of.pop(i)
                st = task_of[i]
                n = o.num_actions
                if st["kind"] == "roll":
                    if st.get("leaf"):
                        finish_roll(st, score_leaf(st, o, values[j])); release(i); continue
                    st["steps"] += 1
                    env.send(i, {"op": "step", "action": int(logits[j, :n].argmax())})
                    continue
                p = torch.softmax(logits[j, :n], -1).numpy()
                g = int(p.argmax())
                if o.decision == DecisionKind.CARD_REWARD and o.act in acts and n > 1 and p[g] < a.tau and not (a.deadline and time.time() > a.deadline):
                    order = np.argsort(-p)
                    cands = [int(c) for c in order[: a.k] if p[c] >= a.pmin]
                    if len(cands) > 1:
                        pid = pid_seq[0]; pid_seq[0] += 1
                        probes[pid] = {"seed": st["seed"], "prefix": list(st["prefix"]), "act": int(o.act), "floor": int(o.floor), "step": len(st["prefix"]),
                                       "n": n, "greedy": g, "p": p, "cands": cands, "labels": list(o.labels or [])[:n],
                                       "sc": {c: [0.0] * a.worlds for c in cands}, "left": len(cands) * a.worlds}
                        queue.extend(("roll", pid, c, w) for c in cands for w in range(a.worlds))
                st["prefix"].append(g)
                env.send(i, {"op": "step", "action": g})
            launch()
    print(f"完成，用时 {time.time() - t0:.0f} 秒", flush=True)


if __name__ == "__main__":
    main()
