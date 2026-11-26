"""AlphaZero 式搜索（python/sts2rl/azmcts.py）在固定考卷上的核对与配对比较，后端是 C# 引擎 + 重放（决策记录 145）。

C# 引擎不能复制状态，这里的“复制”就是记下开局请求和动作序列，用的时候让引擎重放（reset 带 replay，再按需 reseed）。
很慢，只用来在几十道题上核对算法的正确性；Rust 引擎能复制状态以后换后端。

每道题（exam.json 的一个 Boss 局面）按 --modes 打几遍，真实对局都用题目自带的种子：
- greedy：每步取网络概率最高的候选（同评测与考卷）；
- greedy_key：先把动作键相同的候选（一模一样的两张牌）概率相加，再取最高——模拟 1 次的搜索应与它逐步完全一致；
- az：每个有多个候选的决策做 --sims 次模拟（每次一个新的确定化世界），选根节点访问最多的动作。
每道题一行写到 --out：各打法的输赢、剩余血量、动作序列；az 另记搜索过的决策数、改选数（与 greedy_key 的选择不同）、
确定化后根局面对不上的次数（应为 0）、引擎出错数。

    .venv/bin/python -I tools/az_eval.py --ckpt 检查点.pt --exam exam.json --out az.jsonl --per-act 20 --sims 1 --modes greedy,greedy_key,az
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from sts2env import VecEnv  # noqa: E402
from sts2env.env import EngineError  # noqa: E402
from sts2rl.actor import MODE_COMBAT  # noqa: E402
from sts2rl.azmcts import Tree, action_keys, is_terminal, simulate  # noqa: E402
from sts2rl.features import Ragged, collate_ragged  # noqa: E402
from sts2rl.model import ModelConfig, PolicyValueNet, load_state  # noqa: E402

MAX_DECISIONS = 600


def outcome(o, seq: list) -> dict:
    if o is None:
        return {"win": False, "hp": 0, "max_hp": 1, "err": True, "seq": seq}
    return {"win": bool(o.victory) and not o.stuck, "hp": int(o.hp), "max_hp": max(1, int(o.max_hp)), "seq": seq}


class Prog:
    """调度器上跑的一个小程序：生成器 + 开局请求（spec、已走的动作）+ 结束回调。持有引擎从第一次 reset 到结束。"""
    __slots__ = ("gen", "spec", "prefix", "on_done", "engine", "pending")

    def __init__(self, gen, spec: dict, prefix: list, on_done):
        self.gen, self.spec, self.prefix, self.on_done = gen, spec, list(prefix), on_done
        self.engine, self.pending = None, None


def greedy_prog(by_key: bool):
    o = yield ("reset", None)
    seq = []
    while o is not None and not is_terminal(o) and len(seq) < MAX_DECISIONS:
        p, _ = yield ("eval", o)
        if by_key:
            keys = action_keys(o)
            acc: dict = {}
            for k, x in zip(keys, p):
                acc[k] = acc.get(k, 0.0) + float(x)
            a = keys.index(max(acc, key=acc.get))
        else:
            a = int(np.argmax(p))
        seq.append(a)
        o = yield ("step", a)
    return outcome(o if o is not None and len(seq) < MAX_DECISIONS else None, seq)


def obs_prog():
    o = yield ("reset", None)
    return o


class Scheduler:
    def __init__(self, env: VecEnv, net, device: str, max_batch: int = 2048):
        self.env, self.net, self.device, self.max_batch = env, net, device, max_batch
        self.free = list(range(len(env)))
        self.wait: deque = deque()          # 等引擎的程序
        self.busy: dict[int, Prog] = {}
        self.nn: list = []                  # (程序, 观测)
        self.live = 0
        self.n_eval = 0

    def spawn(self, prog: Prog) -> None:
        self.live += 1
        self._drive(prog, None)

    def _drive(self, prog: Prog, value) -> None:
        try:
            req = prog.gen.send(value)
        except StopIteration as e:
            if prog.engine is not None:
                self.busy.pop(prog.engine, None); self.free.append(prog.engine); prog.engine = None
            self.live -= 1
            prog.on_done(e.value)
            return
        if req[0] == "reset":
            prog.pending = req
            if prog.engine is None:
                if self.free:
                    prog.engine = self.free.pop(); self.busy[prog.engine] = prog
                else:
                    self.wait.append(prog); return
            self._send_reset(prog)
        elif req[0] == "step":
            self.env.send(prog.engine, {"op": "step", "action": int(req[1])})
        elif req[0] == "eval":
            self.nn.append((prog, req[1]))
        else:
            raise ValueError(req)

    def _send_reset(self, prog: Prog) -> None:
        r = {"op": "reset", "mode": "combat", "spec": prog.spec, "replay": prog.prefix}
        if prog.pending[1] is not None:
            r["reseed"] = int(prog.pending[1]) & 0xFFFFFFFF
        self.env.send(prog.engine, r)

    def _eval(self) -> None:
        batch, self.nn = self.nn[: self.max_batch], self.nn[self.max_batch:]
        obs = [o for _, o in batch]
        with torch.no_grad():
            logits, values = self.net(collate_ragged(Ragged.from_obs(obs, np.full(len(obs), MODE_COMBAT))).to(self.device))
        logits, values = logits.float().cpu(), values.float().cpu()
        self.n_eval += len(obs)
        for k, (prog, o) in enumerate(batch):
            p = torch.softmax(logits[k, : o.num_actions], -1).numpy()
            self._drive(prog, (p, float(values[k])))

    def run(self, done) -> None:
        while not done():
            while self.free and self.wait:
                prog = self.wait.popleft()
                prog.engine = self.free.pop(); self.busy[prog.engine] = prog
                self._send_reset(prog)
            if self.busy:
                for i, o in self.env.recv_ready(timeout=0.002 if self.nn else 0.5):
                    prog = self.busy.get(i)
                    if prog is None:
                        continue
                    self._drive(prog, None if isinstance(o, EngineError) else o)
            if self.nn:
                self._eval()


class AzGame:
    """一道题的搜索对局：每个真实决策重放到当前局面，建一棵树做 sims 次模拟（同时最多 par 次），走访问最多的动作。"""

    def __init__(self, sch: Scheduler, item: dict, a, seed_base: int, finish):
        self.sch, self.item, self.a, self.finish = sch, item, a, finish
        self.spec = item["spec"]
        self.hp0 = float(json.loads(self.spec["player_snapshot"]).get("current_hp", 0)) if self.spec.get("player_snapshot") else 0.0
        self.prefix: list = []
        self.seed_base = seed_base
        self.stats = {"searched": 0, "changed": 0, "sims": 0}
        self.t0 = time.time()

    def start(self) -> None:
        self.sch.spawn(Prog(obs_prog(), self.spec, self.prefix, self._real))

    def _real(self, o) -> None:
        if o is None or is_terminal(o) or len(self.prefix) >= MAX_DECISIONS:
            res = outcome(o if len(self.prefix) < MAX_DECISIONS else None, self.prefix)
            res.update(self.stats, secs=round(time.time() - self.t0, 1))
            self.finish(res)
            return
        self.keys = action_keys(o)
        if len(set(self.keys)) == 1:
            self._act(0)
            return
        self.stats["searched"] += 1
        self.tree = Tree(self.a.c_puct)
        self.launched = self.completed = 0
        self.decision_seed = self.seed_base + 7919 * (len(self.prefix) + 1)
        for _ in range(min(self.a.par, self.a.sims)):
            self._launch()

    def _launch(self) -> None:
        s = self.decision_seed + self.launched
        self.launched += 1
        self.sch.spawn(Prog(simulate(self.tree, s, self.hp0, self.keys, self.stats), self.spec, self.prefix, self._sim_done))

    def _sim_done(self, _) -> None:
        self.completed += 1
        if self.launched < self.a.sims:
            self._launch()
        elif self.completed == self.launched:
            self.stats["sims"] += self.tree.sims
            best = self.tree.best(self.keys)
            # 与 greedy_key 的选择比较（根节点先验最大的动作）
            gk = max(self.tree.root.edges, key=lambda k: self.tree.root.edges[k].P)
            self.stats["changed"] += best != gk
            self._act(self.keys.index(best))

    def _act(self, a: int) -> None:
        self.prefix.append(a)
        self.start()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--exam", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-act", type=int, default=20, help="第二、三幕各取前几道题")
    ap.add_argument("--modes", default="greedy,greedy_key,az")
    ap.add_argument("--sims", type=int, default=32)
    ap.add_argument("--par", type=int, default=8, help="一棵树同时挂几次模拟（虚拟损失分散路径）")
    ap.add_argument("--c-puct", type=float, default=1.25)
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--live", type=int, default=8, help="同时在做的题数")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False      # 推理用 fp32，贪心与搜索的先验逐位一致
    blob = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    net = PolicyValueNet(ModelConfig(**blob["config"]["model"])).eval().to(a.device)
    load_state(net, blob["state"])
    exam = json.load(open(a.exam, encoding="utf-8"))["items"]
    items, per = [], {}
    for i, it in enumerate(exam):
        if per.get(it["act"], 0) < a.per_act:
            per[it["act"]] = per.get(it["act"], 0) + 1
            items.append((i, it))
    modes = a.modes.split(",")
    print(f"版本 {blob.get('version')}，{len(items)} 道题（各幕 {per}），打法 {modes}，sims={a.sims} par={a.par} c={a.c_puct}", flush=True)
    out = open(a.out, "w", encoding="utf-8")
    results: dict[int, dict] = {}
    todo = deque((i, it, m) for i, it in items for m in modes)
    t0 = time.time()
    running = [0]

    with VecEnv(a.envs) as env:
        sch = Scheduler(env, net, a.device)

        def finish(i, m, res):
            running[0] -= 1
            rec = results.setdefault(i, {"i": i, "act": exam[i]["act"], "encounter": exam[i]["encounter"]})
            rec[m] = res
            if all(x in rec for x in modes):
                out.write(json.dumps(rec, ensure_ascii=False) + "\n"); out.flush()
                print(f"{time.time() - t0:7.0f} 秒 题 {i:3d} " + " ".join(f"{x}={'赢' if rec[x]['win'] else '输'}" for x in modes)
                      + (f" 搜索 {rec['az']['searched']} 改选 {rec['az']['changed']} 根对不上 {rec['az'].get('root_mismatch', 0)} 出错 {rec['az'].get('error', 0)} 用时 {rec['az']['secs']}s" if "az" in rec else ""), flush=True)
            feed()

        def feed():
            while todo and running[0] < a.live:
                i, it, m = todo.popleft()
                running[0] += 1
                cb = (lambda res, i=i, m=m: finish(i, m, res))
                if m == "az":
                    AzGame(sch, it, a, 100003 * (i + 1), cb).start()
                else:
                    sch.spawn(Prog(greedy_prog(m == "greedy_key"), it["spec"], [], cb))

        feed()
        sch.run(lambda: running[0] == 0 and not todo)
    # 汇总
    recs = list(results.values())
    for m in modes:
        print(f"{m:10s} 赢 {sum(r[m]['win'] for r in recs)}/{len(recs)}", flush=True)
    if "az" in modes and "greedy_key" in modes:
        same = sum(r["az"]["seq"] == r["greedy_key"]["seq"] for r in recs)
        print(f"az 与 greedy_key 动作序列完全相同 {same}/{len(recs)}", flush=True)
        for base in ("greedy", "greedy_key"):
            if base in modes:
                d = [int(r["az"]["win"]) - int(r[base]["win"]) for r in recs]
                print(f"az − {base}：多赢 {sum(x > 0 for x in d)}，少赢 {sum(x < 0 for x in d)}", flush=True)
    print(f"网络前向 {sch.n_eval} 次，用时 {time.time() - t0:.0f} 秒", flush=True)


if __name__ == "__main__":
    main()
