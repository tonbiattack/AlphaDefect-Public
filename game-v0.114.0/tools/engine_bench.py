"""C# 无头引擎与 Rust 引擎的同类操作测速（2026-11-01，交付版 Rust 引擎测试）。

两边都走训练端的帧协议（sts2env 的 reset / step，返回格式 19 的观测帧），计时只算请求发出到帧字节读完（不含 Python 解析，解析另计）。
局面用真实的第三幕 Boss 快照题（tools/search_test.py exam 建的题，连打两个 Boss），随机打法。

  single  单进程：开局、单步、整场随机推演、按动作重放到第 L 步（C# 的“还原”）的耗时
  clone   Rust 搜索协议（sts2-serve search）：复制、reseed、走一步、编码一帧的耗时（C# 没有复制，对应的是开局 + 重放）
  inproc  Rust 进程内（JSON 协议的 bench 操作）：开局与随机走步的纯引擎耗时，不编码观测、没有进程间通信
  thru    多进程吞吐：P 个进程各自不停地随机打整场，统计每秒步数与场数

  .venv/bin/python -I tools/engine_bench.py single --exam 题.json --engine cs --n 100 --out cs_single.json
  .venv/bin/python -I tools/engine_bench.py clone --exam 题.json --n 50 --out rs_clone.json
  .venv/bin/python -I tools/engine_bench.py thru --exam 题.json --engine rs --procs 8 --seconds 30 --out rs_thru.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import struct
import subprocess
import sys
import time
from multiprocessing import Process, Queue
from pathlib import Path

REPO = Path(os.environ.get("SLS2_REPO") or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(REPO / "python"))

from sts2env.env import RUST_BIN, Sts2Env  # noqa: E402
from sts2env.protocol import parse  # noqa: E402


def specs(path: str) -> list[dict]:
    return [it["spec"] for it in json.loads(Path(path).read_text(encoding="utf-8"))["items"]]


def call(e: Sts2Env, req: dict) -> tuple[bytes, float]:
    t = time.perf_counter()
    e._proc.send(req)
    b = e._proc.recv()
    return b, time.perf_counter() - t


def nact(b: bytes) -> tuple[int, bool]:
    """帧头：候选数（第 3 项）、是否结束（第 4 项）。"""
    _, _, _, m, done = struct.unpack_from("<5i", b)
    return m, bool(done)


def pct(xs: list[float]) -> dict:
    xs = sorted(xs)
    if not xs:
        return {}
    q = lambda p: xs[min(len(xs) - 1, int(p * len(xs)))]
    return {"n": len(xs), "mean_us": round(1e6 * sum(xs) / len(xs), 1), "p50_us": round(1e6 * q(0.5), 1),
            "p95_us": round(1e6 * q(0.95), 1), "max_us": round(1e6 * xs[-1], 1)}


def play_random(e: Sts2Env, spec: dict, rng: random.Random, step_t: list | None = None) -> tuple[list[int], float, bool]:
    """随机打完一整场（连打两个 Boss），返回动作序列、总耗时、是否赢。"""
    t0 = time.perf_counter()
    b, _ = call(e, {"op": "reset", "mode": "combat", "spec": spec, "labels": False})
    acts: list[int] = []
    m, done = nact(b)
    while not done and m > 0 and len(acts) < 600:
        a = rng.randrange(m)
        b, dt = call(e, {"op": "step", "action": a})
        if step_t is not None:
            step_t.append(dt)
        acts.append(a)
        m, done = nact(b)
    return acts, time.perf_counter() - t0, bool(struct.unpack_from("<6i", b)[5])


def greedy_trajs(e: Sts2Env, sp: list[dict], ckpt: str) -> list[list[int]]:
    """用网络贪心在这个引擎上打完每一场，记下动作序列（之后按固定动作重放计时，计时里没有网络）。"""
    import numpy as np
    import torch
    from sts2rl.actor import MODE_COMBAT
    from sts2rl.features import Ragged, collate_ragged
    from sts2rl.model import ModelConfig, PolicyValueNet, load_state
    torch.set_num_threads(2)
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    net = PolicyValueNet(ModelConfig(**blob["config"]["model"])).eval()
    load_state(net, blob["state"])
    out = []
    for s in sp:
        o = parse(call(e, {"op": "reset", "mode": "combat", "spec": s, "labels": False})[0])
        acts: list[int] = []
        while not o.done and o.num_actions > 0 and len(acts) < 600:
            with torch.no_grad():
                lg, _ = net(collate_ragged(Ragged.from_obs([o], np.array([MODE_COMBAT]))).to("cpu"))
            x = int(lg[0, : o.num_actions].argmax())
            acts.append(x)
            o = parse(call(e, {"op": "step", "action": x})[0])
        out.append(acts)
    return out


def play_fixed(e: Sts2Env, spec: dict, acts: list[int], step_t: list) -> float:
    """按固定动作序列一步一步打（每步返回一帧），返回总耗时。"""
    t0 = time.perf_counter()
    call(e, {"op": "reset", "mode": "combat", "spec": spec, "labels": False})
    for x in acts:
        step_t.append(call(e, {"op": "step", "action": x})[1])
    return time.perf_counter() - t0


def cmd_single(a) -> None:
    sp = specs(a.exam)[: a.n]
    e = Sts2Env(engine="rust" if a.engine == "rs" else "csharp")
    rng = random.Random(1)
    for s in sp[: a.warmup]:        # 预热（C# 的 JIT）
        play_random(e, s, rng)
    greedy = {}
    if a.ckpt:
        t = time.perf_counter()
        gt = greedy_trajs(e, sp, a.ckpt)
        gstep, gcombat, greplay = [], [], []
        for s, acts in zip(sp, gt):
            gcombat.append(play_fixed(e, s, acts, gstep))
            greplay.append(call(e, {"op": "reset", "mode": "combat", "spec": s, "replay": acts, "labels": False})[1])
        greedy = {"trajectory_steps_mean": round(sum(map(len, gt)) / len(gt), 1), "step": pct(gstep),
                  "combat_stepwise": pct(gcombat), "combat_one_request_replay": pct(greplay)}
        replay_g = {}
        for L in (25, 50, 100, 200):
            ts = [call(e, {"op": "reset", "mode": "combat", "spec": s, "replay": acts[:L], "labels": False})[1]
                  for s, acts in zip(sp, gt) if len(acts) >= L]
            replay_g[str(L)] = pct(ts)
        greedy["replay_to_L"] = replay_g
    reset_t, step_t, combat_t, combat_steps, parse_t, trajs = [], [], [], [], [], []
    for s in sp:
        _, dt = call(e, {"op": "reset", "mode": "combat", "spec": s, "labels": False})
        reset_t.append(dt)
    for s in sp:
        acts, dt, _ = play_random(e, s, rng, step_t)
        combat_t.append(dt); combat_steps.append(len(acts)); trajs.append(acts)
    b, _ = call(e, {"op": "reset", "mode": "combat", "spec": sp[0], "labels": False})
    for _ in range(200):
        t = time.perf_counter(); parse(b); parse_t.append(time.perf_counter() - t)
    replay = {}
    for L in (10, 25, 50, 100):
        ts = []
        for s, acts in zip(sp, trajs):
            if len(acts) >= L:
                _, dt = call(e, {"op": "reset", "mode": "combat", "spec": s, "replay": acts[:L], "labels": False})
                ts.append(dt)
        replay[str(L)] = pct(ts)
    reseed_t = []
    for s, acts in zip(sp, trajs):
        L = len(acts) // 2
        _, dt = call(e, {"op": "reset", "mode": "combat", "spec": s, "replay": acts[:L], "reseed": 12345, "first": 0, "labels": False})
        reseed_t.append(dt)
    e.close()
    res = {"engine": a.engine, "specs": len(sp), "reset": pct(reset_t), "step": pct(step_t), "parse_python": pct(parse_t),
           "combat_random": {**pct(combat_t), "steps_mean": round(sum(combat_steps) / len(combat_steps), 1)},
           "replay_to_L": replay, "replay_half_reseed_first": pct(reseed_t), "greedy": greedy}
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


class Search:
    """sts2-serve search 的二进制协议（u32 长度 + u8 op + 负载 → u32 长度 + u8 状态 + 负载）。"""

    def __init__(self):
        self.p = subprocess.Popen([str(RUST_BIN), "search"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0)

    def call(self, op: int, payload: bytes = b"") -> bytes:
        self.p.stdin.write(struct.pack("<IB", len(payload) + 1, op) + payload)
        self.p.stdin.flush()
        (n,) = struct.unpack("<I", self._read(4))
        body = self._read(n)
        if body[0] != 0:
            raise RuntimeError(body[1:].decode("utf-8", "replace"))
        return body[1:]

    def _read(self, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            c = self.p.stdout.read(n - len(buf))
            if not c:
                raise RuntimeError("search 进程已退出")
            buf += c
        return buf

    def import_spec(self, spec: dict) -> int:
        return struct.unpack_from("<I", self.call(1, b"\x00" + json.dumps(spec).encode()))[0]

    def steps(self, pairs: list[tuple[int, int]]) -> bytes:
        return self.call(5, struct.pack("<I", len(pairs)) + b"".join(struct.pack("<II", i, a) for i, a in pairs))

    def clone(self, i: int) -> int:
        return struct.unpack("<I", self.call(2, struct.pack("<I", i)))[0]


def cmd_clone(a) -> None:
    sp = specs(a.exam)[: a.n]
    e = Sts2Env(engine="rust")
    rng = random.Random(2)
    trajs = [play_random(e, s, rng)[0] for s in sp]
    e.close()
    S = Search()
    imp_t, clone_t, drop_t, reseed_t, step_t, frame_t, branch_t, bstep_t, bframe_t = [], [], [], [], [], [], [], [], []
    for s, acts in zip(sp, trajs):
        t = time.perf_counter(); root = S.import_spec(s); imp_t.append(time.perf_counter() - t)
        L = len(acts) // 2
        if L:
            S.steps([(root, x) for x in acts[:L]])
        for k in range(a.clones):
            t = time.perf_counter(); c = S.clone(root); clone_t.append(time.perf_counter() - t)
            t = time.perf_counter(); S.call(4, struct.pack("<IQ", c, 1000 + k)); reseed_t.append(time.perf_counter() - t)
            t = time.perf_counter(); S.steps([(c, 0)]); step_t.append(time.perf_counter() - t)
            t = time.perf_counter(); S.call(6, struct.pack("<II", 1, c)); frame_t.append(time.perf_counter() - t)
            t = time.perf_counter(); S.call(3, struct.pack("<I", c)); drop_t.append(time.perf_counter() - t)
        # 一次往返做完“复制 + reseed + 走一步”的批量版：32 个世界
        t = time.perf_counter()
        ids = [S.clone(root) for _ in range(32)]
        for k, c in enumerate(ids):
            S.call(4, struct.pack("<IQ", c, 5000 + k))
        S.steps([(c, 0) for c in ids])
        branch_t.append((time.perf_counter() - t) / 32)
        t = time.perf_counter(); S.steps([(c, 0) for c in ids]); bstep_t.append((time.perf_counter() - t) / 32)
        t = time.perf_counter(); S.call(6, struct.pack("<I", 32) + b"".join(struct.pack("<I", c) for c in ids)); bframe_t.append((time.perf_counter() - t) / 32)
        for c in ids:
            S.call(3, struct.pack("<I", c))
        S.call(3, struct.pack("<I", root))
    S.call(8)
    res = {"engine": "rs-search", "specs": len(sp), "import_spec": pct(imp_t), "clone": pct(clone_t), "reseed": pct(reseed_t),
           "batch_step_1": pct(step_t), "batch_frame_1": pct(frame_t), "drop": pct(drop_t), "clone_reseed_step_per_world_32": pct(branch_t),
           "batch_step_32_per_item": pct(bstep_t), "batch_frame_32_per_item": pct(bframe_t)}
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


def _thru_worker(engine: str, sp: list[dict], seconds: float, wid: int, q: Queue) -> None:
    e = Sts2Env(engine="rust" if engine == "rs" else "csharp")
    rng = random.Random(100 + wid)
    for s in sp[:3]:
        play_random(e, s, rng)
    steps = combats = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        acts, _, _ = play_random(e, sp[rng.randrange(len(sp))], rng)
        steps += len(acts); combats += 1
    q.put((steps, combats, time.perf_counter() - t0))
    e.close()


def cmd_thru(a) -> None:
    sp = specs(a.exam)
    q: Queue = Queue()
    ps = [Process(target=_thru_worker, args=(a.engine, sp, a.seconds, w, q)) for w in range(a.procs)]
    for p in ps:
        p.start()
    rs = [q.get() for _ in ps]
    for p in ps:
        p.join()
    steps = sum(r[0] for r in rs); combats = sum(r[1] for r in rs); secs = max(r[2] for r in rs)
    res = {"engine": a.engine, "procs": a.procs, "seconds": round(secs, 1), "steps_per_s": round(steps / secs, 1),
           "combats_per_s": round(combats / secs, 2), "steps_per_combat": round(steps / max(1, combats), 1)}
    print(json.dumps(res, ensure_ascii=False))
    if a.out:
        Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


def cmd_inproc(a) -> None:
    """Rust 进程内测速（JSON 协议的 bench：同一个开局换种子随机打，不编码观测、没有进程间通信）。"""
    sp = specs(a.exam)[: a.n]
    p = subprocess.Popen([str(RUST_BIN)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    steps = combats = 0
    secs = setup = 0.0
    for s in sp:
        p.stdin.write(json.dumps({"op": "bench", "spec": s, "combats": a.combats}) + "\n"); p.stdin.flush()
        r = json.loads(p.stdout.readline())
        steps += r["steps"]; combats += r["combats"]; secs += r["seconds"]; setup += r["setup_seconds"]
    p.stdin.write('{"op":"close"}\n'); p.stdin.flush(); p.wait()
    res = {"engine": "rs-inproc", "specs": len(sp), "combats": combats, "steps": steps,
           "setup_us_per_combat": round(1e6 * setup / combats, 1), "step_us": round(1e6 * (secs - setup) / steps, 2),
           "combat_us": round(1e6 * secs / combats, 1), "steps_per_combat": round(steps / combats, 1)}
    print(json.dumps(res, ensure_ascii=False))
    if a.out:
        Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["single", "clone", "thru", "inproc"])
    ap.add_argument("--exam", required=True)
    ap.add_argument("--engine", choices=["cs", "rs"], default="rs")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--clones", type=int, default=20)
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--combats", type=int, default=20)
    ap.add_argument("--ckpt", default=None, help="single：另用网络贪心打出动作序列，测真实长度的逐步打与重放")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    {"single": cmd_single, "clone": cmd_clone, "thru": cmd_thru, "inproc": cmd_inproc}[a.cmd](a)


if __name__ == "__main__":
    main()
