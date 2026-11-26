"""按 p(1−p) 选题的离线核对（决策记录 225）：第三幕第一个 Boss 的快照开局后，价值头对第一个观测的估值能不能估出这个快照的胜率 p。

  .venv/bin/python -I tools/health/p_calib.py values --ckpt 检查点.pt --exam 题.json [题2.json ...] --out 估值.jsonl --envs 24
  python3 tools/health/p_calib.py fit --values 估值.jsonl --p learn_p.json --out 结果.json

values：每道题（exam_learn 的第一个世界的种子）开局，取第一个观测，分别按战斗练习与整局两种模式标记让价值头估值。
fit：估值与实测 p（tools/search_test 可学性测量，每个快照 8 个世界）的排序相关；用逻辑函数拟合 p̂ = σ(a·v + b)，
     再算按 max(0.02, p̂(1−p̂)) 加权（混 20% 随机）时，抽到的题平均 p(1−p) 与必输局面占比。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))


def cmd_values(a) -> None:
    import numpy as np
    import torch
    from sts2env import EngineError, VecEnv
    from sts2rl.actor import MODE_COMBAT, MODE_RUN
    from sts2rl.features import Ragged, collate_ragged
    from sts2rl.model import ModelConfig, PolicyValueNet, load_state
    torch.set_num_threads(2)
    blob = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    net = PolicyValueNet(ModelConfig(**blob["config"]["model"])).eval().to(a.device)
    load_state(net, blob["state"])
    items = []
    for f, off in zip(a.exam, (0, 100000, 200000)):
        for it in json.load(open(f, encoding="utf-8"))["items"]:
            spec = dict(it["spec"], seed=f"LX{off + it['id']:06d}0")
            items.append((off + it["id"], spec))
    out = open(a.out, "w", encoding="utf-8")
    queue, task, done = list(reversed(items)), {}, 0
    with VecEnv(a.envs) as env:
        def launch(i):
            if queue:
                sid, spec = queue.pop()
                task[i] = sid
                env.send(i, {"op": "reset", "mode": "combat", "spec": spec})
        for i in range(len(env)):
            launch(i)
        while env.pending:
            got = [(i, o) for i, o in env.recv_ready(timeout=1.0)]
            ok = [(i, o) for i, o in got if not isinstance(o, EngineError) and o.num_actions > 0]
            if ok:
                obs = [o for _, o in ok]
                with torch.no_grad():
                    _, vc = net(collate_ragged(Ragged.from_obs(obs, np.full(len(obs), MODE_COMBAT))).to(a.device))
                    _, vr = net(collate_ragged(Ragged.from_obs(obs, np.full(len(obs), MODE_RUN))).to(a.device))
                for (i, o), x, y in zip(ok, vc.float().cpu().tolist(), vr.float().cpu().tolist()):
                    out.write(json.dumps({"id": task[i], "v_combat": round(x, 5), "v_run": round(y, 5), "hp": int(o.hp), "max_hp": int(o.max_hp)}) + "\n")
                    done += 1
            for i, _ in got:
                launch(i)
            if done and done % 200 == 0:
                out.flush()
    out.close()
    print(f"{done} 个局面 → {a.out}")


def spearman(x, y):
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v)
        i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
                j += 1
            for k in range(i, j + 1):
                r[o[k]] = (i + j) / 2
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    return num / math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))


def cmd_fit(a) -> None:
    P = {int(k): v for k, v in json.load(open(a.p)).items()}
    V = [json.loads(l) for l in open(a.values, encoding="utf-8") if l.strip()]
    V = [r for r in V if r["id"] in P]
    out = {"n": len(V)}
    for key in ("v_combat", "v_run"):
        x = [r[key] for r in V]; y = [P[r["id"]] for r in V]
        # 逻辑函数拟合 p̂ = σ(a·x + b)：对 8 次里的胜负做最大似然（梯度下降，x 先标准化）
        mu = sum(x) / len(x); sd = math.sqrt(sum((t - mu) ** 2 for t in x) / len(x)) or 1.0
        z = [(t - mu) / sd for t in x]; A, B = 1.0, 0.0
        for _ in range(3000):
            ga = gb = 0.0
            for zi, yi in zip(z, y):
                q = 1 / (1 + math.exp(-(A * zi + B))); ga += (yi - q) * zi; gb += (yi - q)
            A += 0.5 * ga / len(z); B += 0.5 * gb / len(z)
        ph = [1 / (1 + math.exp(-(A * zi + B))) for zi in z]
        w = [0.8 * max(0.02, q * (1 - q)) / sum(max(0.02, t * (1 - t)) for t in ph) + 0.2 / len(ph) for q in ph]
        s = sum(w)
        mean_pq = sum(wi * yi * (1 - yi) for wi, yi in zip(w, y)) / s
        dead = sum(wi for wi, yi in zip(w, y) if yi == 0) / s
        out[key] = {"spearman": round(spearman(x, y), 4), "a": round(A / sd, 5), "b": round(B - A * mu / sd, 5),
                    "weighted_mean_pq": round(mean_pq, 4), "weighted_dead": round(dead, 4),
                    "uniform_mean_pq": round(sum(t * (1 - t) for t in y) / len(y), 4), "uniform_dead": round(sum(t == 0 for t in y) / len(y), 4)}
        print(key, out[key])
    if a.out:
        Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("values"); v.add_argument("--ckpt", required=True); v.add_argument("--exam", nargs="+", required=True); v.add_argument("--out", required=True)
    v.add_argument("--envs", type=int, default=24); v.add_argument("--device", default="cuda")
    f = sub.add_parser("fit"); f.add_argument("--values", required=True); f.add_argument("--p", required=True); f.add_argument("--out", default=None)
    a = ap.parse_args()
    {"values": cmd_values, "fit": cmd_fit}[a.cmd](a)


if __name__ == "__main__":
    main()
