"""选路风险修正的进阶 10 配对评测（决策记录 152）：同一检查点、同一组种子，打法只差在地图决策上。

- β = 0：纯贪心（每步取策略概率最高的候选），与评测、能力测试相同；
- β > 0：地图决策（多于一个候选时）改为按 策略对数几率 + β · log(V + 0.001) 取最高，V 是规划器（planner.py）估的
  “从这个节点走下去、活着打过本幕 Boss 的概率”。其余决策都是贪心。
每局记：输赢、层数、到了哪一幕、每场 Boss 战开打时的血量、精英战场数、死在哪（幕:房间类型），以及地图决策数、被修正改掉的次数。
结果每局一行写到 --out；最后按种子配对汇总。

    python route_eval.py --ckpt 主线检查点.pt --rm /root/roommodel/model.pt --vocab vocab.txt --log ~/roomlog --out route.jsonl --betas 0,1,3 --runs 1000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="仓库目录（python/ 与引擎所在）")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--rm", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--log", required=True, help="房间记录目录（统计遭遇分布）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--betas", default="0,1,3")
    ap.add_argument("--runs", type=int, default=1000)
    ap.add_argument("--seed-prefix", default="ROUTE")
    ap.add_argument("--envs", type=int, default=48)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--debug", type=int, default=0, help="打印前几个地图决策的策略概率与规划器估值")
    a = ap.parse_args()
    sys.path.insert(0, str(Path(a.repo) / "python"))
    from sts2env import VecEnv
    from sts2env.env import EngineError
    from sts2env.protocol import ActionKind, DecisionKind
    from sts2rl.actor import MODE_RUN
    from sts2rl.features import Ragged, collate_ragged
    from sts2rl.model import ModelConfig, PolicyValueNet, load_state
    from planner import Planner, build_prior
    from roommodel import RoomModel, load_vocab

    torch.set_num_threads(4)
    blob = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    net = PolicyValueNet(ModelConfig(**blob["config"]["model"])).eval().to(a.device)
    load_state(net, blob["state"])
    rm = RoomModel().to(a.device)
    rm.load_state_dict(torch.load(a.rm, map_location=a.device, weights_only=False)["model"])
    vocab = load_vocab(Path(a.vocab))
    t0 = time.time()
    prior = build_prior(Path(a.log))
    print(f"主线第 {blob.get('version')} 版；遭遇分布 {len(prior)} 格（{time.time() - t0:.0f} 秒）", flush=True)
    planner = Planner(rm, vocab, prior, a.device)
    betas = [float(x) for x in a.betas.split(",")]
    todo = deque((f"{a.seed_prefix}{j:04d}", b) for j in range(a.runs) for b in betas)
    out = open(a.out, "a", encoding="utf-8")
    done_n = 0
    stats = {}

    with VecEnv(a.envs) as env:
        cur: dict[int, dict] = {}

        def start(i: int) -> None:
            if not todo:
                return
            seed, b = todo.popleft()
            cur[i] = {"seed": seed, "beta": b, "bosses": [], "elites": 0, "maps": 0, "changed": 0, "agree": 0, "vgap": 0.0, "last": None, "steps": 0, "plan_s": 0.0}
            env.send(i, {"op": "reset", "mode": "run", "character": "DEFECT", "ascension": 10, "seed": seed, "snapshots": True, "route_summary": True})

        for i in range(len(env)):
            start(i)
        ready: dict[int, object] = {}
        while cur:
            for i, o in env.recv_ready(timeout=0.05):
                r = cur[i]
                if isinstance(o, EngineError) or o.done or o.stuck or o.num_actions == 0 or r["steps"] > 6000:
                    ok = not isinstance(o, EngineError)
                    rec = {"seed": r["seed"], "beta": r["beta"], "win": bool(ok and o.victory and not o.stuck), "floor": int(o.floor) if ok else 0,
                           "act": int(o.act) if ok else 0, "bosses": r["bosses"], "elites": r["elites"], "maps": r["maps"], "changed": r["changed"], "agree": r["agree"], "vgap": round(r["vgap"], 3),
                           "death": None if (ok and o.victory) else r["last"], "err": not ok or bool(o.stuck), "plan_s": round(r["plan_s"], 2)}
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n"); out.flush()
                    done_n += 1
                    s = stats.setdefault(r["beta"], [0, 0]); s[0] += rec["win"]; s[1] += 1
                    if done_n % 100 == 0:
                        print(f"{time.time() - t0:6.0f} 秒 完成 {done_n} 局；" + "，".join(f"β={b:g} {w}/{n}" for b, (w, n) in sorted(stats.items())), flush=True)
                    del cur[i]
                    start(i)
                    continue
                if o.extra and "combat" in o.extra:
                    c = o.extra["combat"]
                    r["last"] = f"{c['act']}:{c['room']}"
                    if c["room"] == "Boss":
                        r["bosses"].append([int(c["act"]), int(o.hp), int(o.max_hp)])
                    elif c["room"] == "Elite":
                        r["elites"] += 1
                if o.extra and "snapshot" in o.extra:
                    r["snap"] = o.extra["snapshot"]
                ready[i] = o
            if not ready:
                continue
            ids = list(ready)
            obs = [ready[i] for i in ids]
            ready.clear()
            with torch.no_grad():
                logits, _ = net(collate_ragged(Ragged.from_obs(obs, np.full(len(obs), MODE_RUN))).to(a.device))
            logits = logits.float().cpu().numpy()
            for k, (i, o) in enumerate(zip(ids, obs)):
                r = cur[i]
                lg = logits[k, : o.num_actions]
                act = int(np.argmax(lg))
                if o.decision == DecisionKind.MAP and o.num_actions > 1:
                    r["maps"] += 1
                    if r.get("snap"):          # β = 0 也算，只记一致程度不改选
                        ts = time.time()
                        V = planner.node_values(o, r["snap"], int(o.act))
                        r["plan_s"] += time.time() - ts
                        adj = lg.copy()
                        for j in range(o.num_actions):
                            if int(o.cand_ints[j, 0]) == ActionKind.MAP_MOVE:
                                v = V.get(int(o.cand_ints[j, 1]))
                                if v is not None:
                                    adj[j] += r["beta"] * np.log(v + 1e-3)
                        a2 = int(np.argmax(adj)) if r["beta"] > 0 else act
                        vs = [V.get(int(o.cand_ints[j, 1])) for j in range(o.num_actions) if int(o.cand_ints[j, 0]) == ActionKind.MAP_MOVE]
                        vs = [v for v in vs if v is not None]
                        vp = V.get(int(o.cand_ints[act, 1])) if int(o.cand_ints[act, 0]) == ActionKind.MAP_MOVE else None
                        if vs and vp is not None:
                            r["agree"] += vp >= max(vs) - 1e-9
                            r["vgap"] += max(vs) - vp
                        if a.debug > 0:
                            a.debug -= 1
                            pp = np.exp(lg - lg.max()); pp /= pp.sum()
                            print(f"幕{int(o.act)} 层{int(o.floor)} 血{int(o.hp)}/{int(o.max_hp)} " + "  ".join(
                                f"[{j}] p={pp[j]:.3f} V={V.get(int(o.cand_ints[j, 1]), float('nan')):.3f}" for j in range(o.num_actions)) + f" → {act}/{a2}", flush=True)
                        r["changed"] += a2 != act
                        act = a2
                r["steps"] += 1
                env.send(i, {"op": "step", "action": act})
    print(f"完成，用时 {time.time() - t0:.0f} 秒", flush=True)


if __name__ == "__main__":
    main()
