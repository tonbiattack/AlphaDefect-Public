"""测上限：不读档条件下“大量搜索”能把学生 e 推到多高（决策记录 284）。

整局里局外有多个选项、且策略不太确定（首选概率 < --thr）的决策，对首选在内的前 --k 个（不同名的）选项，各用同一组 --salts 个“可能的未来”推演：
每个未来是引擎的公平重设（Branch.Apply：玩家看得到的不变，看不到的——后面的遭遇、奖励、事件、各幕、抽牌顺序——换掉），
重放到这一步、走这个选项、用同一个网络贪心打完，记输赢。第一段挑出配对胜负差最大的备选；它若领先，第二段只比首选与它，
另取 --salts2 个新的未来，配对差超过 --z 倍标准误才换（试跑时一段 16 个未来、1 倍标准误就换，噪声改选把一局本来通关的打输了）。
战斗决策与确定的局外决策照常贪心。只用决策时能知道的东西（真实的未来从不看），是不读档条件下能做的“对所有可能的未来求平均”。

    python -I tools/search_ceiling.py --ckpt 学生e.pt --seeds-file 种子.json --from 0 --games 200 --shard 0/4 --out 目录 [--envs 48] [--k 3] [--salts 32] [--salts2 128] [--z 2] [--thr 0.9]
结果：<out>/runs_s{k}.jsonl（每局：种子、输赢、层数、搜了几次、改了几次）与 <out>/search_s{k}.jsonl（每次搜索的明细）。已打过的种子跳过。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "tools"))

from branch_label import RESET, run_batch, _name  # noqa: E402

KINDS = {2, 3, 4, 6, 7, 8, 9, 11}       # 选牌、卡牌奖励、卡包、事件、休息处、商店、宝箱、选路线（protocol.DecisionKind）


def main() -> None:
    import torch
    from sts2env.env import VecEnv
    from sts2rl.actor import MODE_RUN
    from sts2rl.features import Ragged, collate_ragged
    from sts2rl.model import ModelConfig, PolicyValueNet, load_state
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seeds-file", required=True)
    ap.add_argument("--from", dest="start", type=int, default=0)
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--envs", type=int, default=48)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--salts", type=int, default=32)
    ap.add_argument("--salts2", type=int, default=128)
    ap.add_argument("--thr", type=float, default=0.9)
    ap.add_argument("--z", type=float, default=2.0)
    ap.add_argument("--dll", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-steps", type=int, default=6000)
    a = ap.parse_args()
    dev = torch.device(a.device)
    blob = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    mc = dict(blob["config"]["model"]); mc.pop("oracle", None)
    net = PolicyValueNet(ModelConfig(**mc)).to(dev).eval(); load_state(net, blob["state"])
    k, n = map(int, a.shard.split("/"))
    seeds = json.loads(Path(a.seeds_file).read_text(encoding="utf-8"))["seeds"][a.start:a.start + a.games]
    seeds = [s for i, s in enumerate(seeds) if i % n == k]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    runs_f, srch_f = out / f"runs_s{k}.jsonl", out / f"search_s{k}.jsonl"
    done = {json.loads(l)["seed"] for f in out.glob("runs_s*.jsonl") for l in open(f, encoding="utf-8") if l.strip()}
    kw = {"dll": Path(a.dll)} if a.dll else {}
    pool = VecEnv(a.envs, **kw)
    main_env = VecEnv(1, **kw)
    rng = np.random.default_rng(1000 + k)

    def rollouts(seed, tape, cands, m) -> np.ndarray:
        """每个选项、同一组 m 个公平重设的未来，贪心打完；返回 [选项, 未来] 的输赢。"""
        salts = [int(x) for x in rng.integers(1, 2 ** 31, size=m)]
        res = run_batch(net, dev, pool, [{**RESET, "seed": seed, "replay": list(tape), "branch": s, "first": c} for c in cands for s in salts], record=False)
        return np.array([r["win"] for r in res], float).reshape(len(cands), m)

    def probs(o) -> np.ndarray:
        with torch.no_grad():
            lg = net(collate_ragged(Ragged.from_obs([o], np.full(1, MODE_RUN))).to(dev))[0][0, :o.num_actions].float()
        return torch.softmax(lg, -1).cpu().numpy()

    for seed in seeds:
        if seed in done:
            continue
        t0 = time.time()
        main_env.send(0, {**RESET, "seed": seed})
        o = _recv(main_env)
        tape: list[int] = []
        n_search = n_change = 0
        while not (isinstance(o, Exception) or o.done or o.stuck or len(tape) > a.max_steps):
            p = probs(o)
            act = int(p.argmax())
            if not o.in_combat and int(o.decision) in KINDS and o.num_actions >= 2 and p.max() < a.thr:
                cands, seen = [], set()
                for x in np.argsort(-p):
                    nm = _name(o, int(x)) if int(o.decision) == 2 else int(x)
                    if nm not in seen:            # 选牌界面同名的牌（如三张一样的打击）只留概率最高的一个；事件选项名都是 <unk>、同类地图节点同名，不能按名字去重
                        seen.add(nm); cands.append(int(x))
                    if len(cands) == a.k:
                        break
                rec = {"seed": seed, "t": len(tape), "d": int(o.decision), "act": int(o.act), "floor": int(o.floor),
                       "cands": [_name(o, c) for c in cands], "p": [round(float(p[c]), 4) for c in cands]}
                if len(cands) >= 2:
                    # 第一段：各选项同一组 --salts 个未来，挑出配对胜负差最大的备选
                    w = rollouts(seed, tape, cands, a.salts)
                    d = w[1:] - w[0]
                    best = int(np.argmax(d.mean(1)))
                    rec["rate"] = [round(float(x), 4) for x in w.mean(1)]
                    if d[best].mean() > 0:
                        # 第二段：只比首选与这个备选，另取 --salts2 个新的未来（不复用第一段，免得挑选偏差），配对差超过 --z 倍标准误才换
                        w2 = rollouts(seed, tape, [cands[0], cands[best + 1]], a.salts2)
                        d2 = w2[1] - w2[0]
                        se = d2.std(ddof=1) / np.sqrt(len(d2))
                        rec["stage2"] = {"alt": rec["cands"][best + 1], "rate": [round(float(x), 4) for x in w2.mean(1)], "se": round(float(se), 4)}
                        if d2.mean() > a.z * max(se, 1e-9):
                            act = cands[best + 1]; n_change += 1
                    n_search += 1
                rec["chosen"] = _name(o, act)
                with open(srch_f, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            tape.append(act)
            main_env.send_step(0, act)
            o = _recv(main_env)
        win = bool(not isinstance(o, Exception) and o.done and o.victory)
        with open(runs_f, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"seed": seed, "win": win, "floor": int(getattr(o, "floor", 0)), "steps": len(tape), "searched": n_search,
                                 "changed": n_change, "seconds": round(time.time() - t0, 1)}, ensure_ascii=False) + "\n")
        print(f"{time.strftime('%H:%M:%S')} {seed} {'通关' if win else '第 ' + str(getattr(o, 'floor', '?')) + ' 层'}：搜 {n_search} 次、改 {n_change} 次，{time.time() - t0:.0f} 秒", flush=True)


def _recv(env):
    while True:
        for i, o in env.recv_ready(timeout=5.0):
            if i == 0:
                return o


if __name__ == "__main__":
    main()
