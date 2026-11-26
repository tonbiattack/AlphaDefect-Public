"""交付版 Rust 引擎做 PIMC 推演的配对测试（2026-11-01）：真实对局在 C# 引擎上，推演交给 Rust 引擎，与贪心逐题配对。

题来自千种子复测里打到第三幕第一个 Boss 的快照（tools/search_test.py exam 建题），从快照连打两个 Boss。每题按“臂”各打一遍，
同一题的各臂用同一个种子（共同随机数）：
- greedy：每步取网络概率最高的候选；
- rs：搜索，推演在 Rust 引擎上；
- cs：同样的搜索，推演在 C# 引擎上（参照，慢）。
搜索沿用决策记录 217 的设置（mcts_eval / search_test 的同一算法）：网络最想选的项概率不到 --tau 时取前 --k 个候选（各至少 --pmin），
每个可能世界都是“从战斗开头按已走的动作重放 → reseed（换随机数、重洗抽牌堆）→ 走候选 → 贪心推演 --leaf 步后用价值头估值”，
各候选用同一组世界，逐轮减半 --rounds 轮、首轮每候选 --sims0 次，与贪心项逐世界配对、平均差超过 --z 个标准误才改选。
推演前一律 reseed，价值头不带特权输入：搜索不读真实种子（公开信息口径）。

真实对局的那一局每臂占一个专用的 C# 进程，只往前走、不重放。rs 臂每个决策还让 Rust 重放到同一局面、与 C# 的帧逐项比较
（根局面核对：头部除进程级的第 12、14 项外、记录整数与特征、候选整数与特征；不比特权特征），记一致与否和第一处不同的段。

结果每题一行写到 --out：各臂的输赢、血量、第几场、决策数、搜索数、改选数、推演数，rs 臂另有根局面核对与每次搜索的耗时。

  .venv/bin/python -I tools/pimc_rust_eval.py --ckpt 模型.pt --exam 题.json --out 结果_s0.jsonl --arms greedy,rs --shard 0/8 --device cuda
环境变量：STS2_RUST_BIN（Rust 引擎）、STS2_ENCODE_VERSION=19（Rust 的观测格式，须与 C# 相同）。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import selectors
import struct
import sys
import time
from pathlib import Path

REPO = Path(os.environ.get("SLS2_REPO") or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(REPO / "python"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from sts2env.env import EngineError, Sts2Env  # noqa: E402
from sts2env.protocol import TokenKind, parse  # noqa: E402
from sts2rl.actor import MODE_COMBAT  # noqa: E402
from sts2rl.features import Ragged, collate_ragged  # noqa: E402
from sts2rl.model import ModelConfig, PolicyValueNet, load_state  # noqa: E402

MAX_STEPS = 600
LOSE = {"win": False, "hp": 0, "max_hp": 1, "stage": 0}
HEADER_SKIP = (12, 14)      # 本进程累计赢下的战斗数、本进程累计遇到的词表外键数：与进程有关，不比
# 宽松口径另外不比的头部：金币（10）、第二个 Boss（17）、还要打的 Boss 数（25）、商店删牌次数与价格（28、29）。
# 交付版 Rust 在战斗模式下不填这几项（只影响网络看到的输入，不影响规则）
HEADER_SKIP_LOOSE = HEADER_SKIP + (10, 17, 25, 28, 29)


def terminal(r: dict) -> float:
    """推演结果的得分（同 search_diag.terminal，连打口径）：赢 1.5；输 −1 + 0.25 ×（已打过的场数 + 当前这场打掉的敌人生命比例）。"""
    if r["win"]:
        return 1.5
    return -1.0 + 0.25 * (r.get("stage", 0) + 1.0 - r.get("enemy_left", 1.0))


def enemy_left(o) -> float:
    """当前这场敌人剩余生命的比例（战斗结束那一帧没有敌人记录，所以推演时每步记下）。"""
    k = np.flatnonzero((o.tok_ints[:, 0] == TokenKind.CREATURE) & (o.tok_feats[:, 5] == 0) & (o.tok_feats[:, 3] > 0))
    hp, mx = float(o.tok_feats[k, 0].sum()), float(o.tok_feats[k, 1].sum())
    return hp / mx if mx > 0 else 0.0


def outcome(o, left: float | None = None) -> dict:
    r = {"win": bool(o.victory) and not o.stuck, "hp": int(o.hp), "max_hp": max(1, int(o.max_hp)), "stage": int(o.header[23])}
    if left is not None:
        r["enemy_left"] = round(min(1.0, left), 4)
    return r


def strip_pool(o):
    """去掉出怪池记录（类型 16，在帧末尾）并相应改头部第 2 项：导入接法按整局模式编码会带上它，战斗模式的 C# 帧没有。"""
    import copy
    keep = o.tok_ints[:, 0] != TokenKind.ENCOUNTER_POOL if hasattr(TokenKind, "ENCOUNTER_POOL") else o.tok_ints[:, 0] != 16
    if keep.all():
        return o
    o = copy.copy(o)
    o.header = o.header.copy(); o.header[2] = int(keep.sum())
    o.tok_ints, o.tok_feats, o.tok_priv = o.tok_ints[keep], o.tok_feats[keep], o.tok_priv[keep]
    return o


def frame_diff(a, b, skip=HEADER_SKIP) -> str | None:
    """两帧不同时返回第一处不同的段名，相同返回 None。"""
    ha, hb = a.header.copy(), b.header.copy()
    for k in skip:
        ha[k] = hb[k] = 0
    if not np.array_equal(ha, hb):
        k = int(np.flatnonzero(ha != hb)[0])
        return f"头部[{k}]"
    if a.tok_ints.shape != b.tok_ints.shape:
        return "记录数"
    if not np.array_equal(a.tok_ints, b.tok_ints):
        r, c = np.argwhere(a.tok_ints != b.tok_ints)[0]
        return f"记录整数/{int(a.tok_ints[r, 0])}/{int(c)}"
    if not np.array_equal(a.tok_feats, b.tok_feats):
        r, c = np.argwhere(a.tok_feats != b.tok_feats)[0]
        return f"记录特征/{int(a.tok_ints[r, 0])}/{int(c)}"
    if a.cand_ints.shape != b.cand_ints.shape or not np.array_equal(a.cand_ints, b.cand_ints):
        return "候选整数"
    if not np.array_equal(a.cand_feats, b.cand_feats):
        r, c = np.argwhere(a.cand_feats != b.cand_feats)[0]
        return f"候选特征/{int(c)}"
    return None


class Task:
    """推演池里的一件事：开局并按 prefix 重放、（可选）reseed 后走 first → 返回观测（mode="obs"）或贪心推演（mode="rollout"）。"""
    __slots__ = ("spec", "prefix", "mode", "seed", "first", "engine", "owner", "steps", "left", "result", "root", "stage0", "key")

    def __init__(self, spec, prefix, mode, engine, seed=None, first=None, owner=None):
        self.spec, self.prefix, self.mode, self.engine = spec, list(prefix), mode, engine
        self.seed, self.first, self.owner = seed, first, owner
        self.steps, self.left, self.result = 0, None, None
        self.root, self.stage0, self.key = None, 0, None


class Procs:
    """一组引擎进程（可以混合 C# 与 Rust），一个选择器等结果。"""

    def __init__(self):
        self.envs: list[Sts2Env] = []
        self.kind: list[str] = []
        self.sel = selectors.DefaultSelector()
        self.pending: set[int] = set()

    def add(self, engine: str, n: int) -> list[int]:
        ids = []
        for _ in range(n):
            e = Sts2Env(engine="rust" if engine == "rs" else "csharp")
            i = len(self.envs)
            self.envs.append(e); self.kind.append(engine)
            self.sel.register(e._proc.p.stdout, selectors.EVENT_READ, i)
            ids.append(i)
        return ids

    def send(self, i: int, req: dict) -> None:
        self.envs[i]._proc.send(req)
        self.pending.add(i)

    def recv_ready(self, timeout: float) -> list:
        if not self.pending:
            return []
        out = []
        for key, _ in self.sel.select(timeout):
            i = key.data
            if i not in self.pending:
                continue
            self.pending.discard(i)
            try:
                out.append((i, parse(self.envs[i]._proc.recv())))
            except EngineError as e:
                if self.envs[i]._proc.p.poll() is not None:
                    self.sel.unregister(self.envs[i]._proc.p.stdout)
                    self.envs[i] = Sts2Env(engine="rust" if self.kind[i] == "rs" else "csharp")
                    self.sel.register(self.envs[i]._proc.p.stdout, selectors.EVENT_READ, i)
                out.append((i, e))
        return out

    def close(self) -> None:
        self.sel.close()
        for e in self.envs:
            e.close()


class Imp:
    """交付版的搜索协议（sts2-serve search）：导入一帧、复制、reseed、批量走步。同步调用，Rust 很快，不另开选择器。"""

    def __init__(self):
        import subprocess
        self.p = subprocess.Popen([os.environ["STS2_RUST_BIN"], "search"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, bufsize=0)

    def call(self, op: int, payload: bytes = b"") -> bytes:
        self.p.stdin.write(struct.pack("<IB", len(payload) + 1, op) + payload)
        self.p.stdin.flush()
        (n,) = struct.unpack("<I", self._read(4))
        body = self._read(n)
        if body[0] != 0:
            raise EngineError(body[1:].decode("utf-8", "replace"))
        return body[1:]

    def _read(self, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            c = self.p.stdout.read(n - len(buf))
            if not c:
                raise EngineError("search 进程已退出")
            buf += c
        return buf

    def import_wire(self, wire: dict) -> tuple[int, bytes]:
        b = self.call(1, bytes([1]) + json.dumps(wire).encode())     # 1 = wire state dict 第 1 版
        (sid, n) = struct.unpack_from("<II", b)
        return sid, b[8:8 + n]

    def clone(self, sid: int) -> int:
        return struct.unpack("<I", self.call(2, struct.pack("<I", sid)))[0]

    def reseed(self, sid: int, seed: int) -> None:
        self.call(4, struct.pack("<IQ", sid, seed & 0xFFFFFFFF))

    def drop(self, sid: int) -> None:
        self.call(3, struct.pack("<I", sid))

    def steps(self, pairs: list[tuple[int, int]]) -> list[tuple[int, bytes]]:
        b = self.call(5, struct.pack("<I", len(pairs)) + b"".join(struct.pack("<II", i, x) for i, x in pairs))
        (n,), off, out = struct.unpack_from("<I", b), 4, []
        for _ in range(n):
            kind, ln = struct.unpack_from("<BI", b, off)
            out.append((kind, b[off + 5:off + 5 + ln]))
            off += 5 + ln
        return out

    def close(self) -> None:
        try:
            self.call(8)
        except Exception:
            self.p.kill()


def greedy_ctrl(spec: dict):
    o = yield ("reset", spec)
    n = 0
    while not (isinstance(o, EngineError) or o.done or o.stuck or o.num_actions == 0 or n > MAX_STEPS):
        p = yield ("policy", o)
        o = yield ("step", int(np.argmax(p)))
        n += 1
    res = dict(LOSE) if isinstance(o, EngineError) or n > MAX_STEPS else outcome(o)
    res.update(decisions=n, error=isinstance(o, EngineError))
    return res


def search_ctrl(spec: dict, engine: str, a, seed_base: int):
    """搜索臂：真实对局在专用 C# 进程上往前走；搜索的推演交给 engine：
    "rs" 从战斗开头按已走的动作在 Rust 上重放；"cs" 同样重放，在 C# 上；"imp" 把 C# 当前的帧经翻译层导入 Rust（直播的接法），复制后推演。
    rs、imp 每个决策先核对 Rust 的根局面与 C# 的帧；Rust 给不出根局面（报错）时这一步不搜索，按贪心走（记一次退回）。
    推演报错的世界不参与比较（只用两边都有效的世界配对）；贪心项有效世界不到 2 个时不改选。"""
    prefix: list[int] = []
    n_search = n_sims = n_changed = n_err = n_fallback = 0
    checked = matched = matched_loose = 0
    mism: dict[str, int] = {}
    mism_loose: dict[str, int] = {}
    first_bad = first_bad_loose = None      # 第一次根局面不一致的决策序号（严格、宽松口径）
    first_detail = None                     # 那一次两边的头部（C#、Rust）或 Rust 的报错文本
    err_text = None                         # 推演里第一次报错的文本
    times: list[float] = []
    check = engine in ("rs", "imp") and a.root_check
    o = yield ("reset", spec)
    while not (isinstance(o, EngineError) or o.done or o.stuck or o.num_actions == 0 or len(prefix) > MAX_STEPS):
        root = None
        if check:
            if engine == "rs":
                (ro,) = yield ("tasks", [Task(spec, prefix, "obs", "rs")])
            else:
                root = yield ("import", o)
                ro = root if isinstance(root, EngineError) else strip_pool(root[2])
            checked += 1
            d = "引擎报错" if isinstance(ro, EngineError) else frame_diff(o, ro)
            dl = "引擎报错" if isinstance(ro, EngineError) else frame_diff(o, ro, HEADER_SKIP_LOOSE)
            if dl is None:
                matched_loose += 1
            else:
                mism_loose[dl] = mism_loose.get(dl, 0) + 1
                if first_bad_loose is None:
                    first_bad_loose = len(prefix)
            if d is None:
                matched += 1
            else:
                mism[d] = mism.get(d, 0) + 1
                if first_bad is None:
                    first_bad = len(prefix)
                    first_detail = str(ro)[:300] if isinstance(ro, EngineError) else {"seg": d, "cs": o.header.tolist(), "rs": ro.header.tolist(),
                                                                                      "n_tok": [len(o.tok_ints), len(ro.tok_ints)]}
            # 候选对不上（数量或内容不同）时，C# 的候选下标在 Rust 里指的不是同一个动作：这一步不搜索
            root_ok = not isinstance(ro, EngineError) and (not a.cand_guard or (ro.cand_ints.shape == o.cand_ints.shape
                                                                                and np.array_equal(ro.cand_ints, o.cand_ints)))
        else:
            root_ok = True
        p = yield ("policy", o)
        order = np.argsort(-p)
        act = int(order[0])
        if o.num_actions > 1 and p[act] < a.tau:
            cands = [int(c) for c in order[: a.k] if p[c] >= a.pmin] or [act]
            if len(cands) > 1 and not root_ok:
                n_fallback += 1
            elif len(cands) > 1:
                t0 = time.perf_counter()
                n_search += 1
                sc: dict = {c: {} for c in cands}       # 候选 → {世界种子: 得分}，报错的世界不记
                alive, n_new, used = list(cands), a.sims0, 0
                for _ in range(a.rounds):
                    seeds = [seed_base + 7919 * (len(prefix) + 1) + used + j for j in range(n_new)]
                    used += n_new
                    tasks = [Task(spec, prefix, "rollout", engine, seed=s, first=c, owner=(c, s)) for c in alive for s in seeds]
                    if engine == "imp":
                        for t in tasks:
                            t.root, t.stage0 = root, int(o.header[23])
                    results = yield ("tasks", tasks)
                    n_sims += len(tasks)
                    for t, r in zip(tasks, results):
                        c, s_ = t.owner
                        if not isinstance(r, dict):
                            n_err += 1
                            if err_text is None:
                                err_text = str(r)[:300]
                            continue
                        sc[c][s_] = r["value"] if "value" in r else terminal(r)
                    if len(alive) <= 2:
                        continue
                    mean = lambda c: sum(sc[c].values()) / len(sc[c]) if sc[c] else -9.0
                    rest = sorted((c for c in alive if c != act), key=lambda c: (-mean(c), -p[c]))
                    alive = [act] + rest[: max(1, math.ceil(len(alive) / 2) - 1)]
                    n_new *= 2
                best, best_m = act, 0.0
                for c in sc:
                    if c == act:
                        continue
                    common = [s_ for s_ in sc[c] if s_ in sc[act]]
                    if len(common) < 2:
                        continue
                    d = [sc[c][s_] - sc[act][s_] for s_ in common]
                    m = sum(d) / len(d)
                    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1))
                    if m > best_m and m > a.z * sd / math.sqrt(len(d)):
                        best, best_m = c, m
                n_changed += best != act
                act = best
                times.append(round(time.perf_counter() - t0, 4))
        if root is not None and not isinstance(root, EngineError):
            yield ("drop", root)
        prefix.append(act)
        o = yield ("step", act)
    res = dict(LOSE) if isinstance(o, EngineError) or len(prefix) > MAX_STEPS else outcome(o)
    res.update(decisions=len(prefix), searched=n_search, sims=n_sims, changed=n_changed, rollout_errors=n_err, fallback=n_fallback,
               error=isinstance(o, EngineError), times=times)
    if check:
        res.update(root_checked=checked, root_matched=matched, root_mismatch=mism, first_bad=first_bad, first_detail=first_detail,
                   root_matched_loose=matched_loose, root_mismatch_loose=mism_loose, first_bad_loose=first_bad_loose)
    res.update(rollout_error_text=err_text, actions=prefix)
    return res


def run(a, items, net, device) -> None:
    arms = a.arms.split(",")
    done_ids = set()
    tag = Path(a.out).stem.rsplit("_s", 1)[0]
    for f in Path(a.out).parent.glob(f"{tag}_s*.jsonl"):
        for l in open(f, encoding="utf-8"):
            if l.strip():
                done_ids.add(json.loads(l)["i"])
    items = [(i, it) for i, it in items if i not in done_ids]
    print(f"{len(items)} 道题待做（已做 {len(done_ids)}），臂 {arms}", flush=True)
    out = open(a.out, "a", encoding="utf-8")
    P = Procs()
    truth_free = P.add("cs", a.live * len(arms))
    pool_free = {"rs": P.add("rs", a.rs_envs) if "rs" in arms else [], "cs": P.add("cs", a.cs_envs) if "cs" in arms else []}
    queue = {"rs": [], "cs": [], "imp": []}
    imps = [Imp() for _ in range(a.imp_procs)] if "imp" in arms else []
    imp_rr = [0]
    imp_task: dict = {}         # (导入进程, 状态 id) → Task
    imp_obs: dict = {}          # (导入进程, 状态 id) → 等网络选动作的观测
    if imps:
        sys.path.insert(0, os.environ["PIMC_IMPORT_DIR"])
        from live_import import frame_to_wire     # 交付版配套的翻译层（帧 → 内核 wire state）
    ctrls: dict = {}            # 控制器 id → [生成器, 题号, 臂, 真实对局的进程]
    partial: dict = {}
    waiting: dict = {}          # 控制器 id → [任务列表, 已完成数]
    pol_req: list = []
    env_task: dict = {}         # 推演池进程 → Task
    env_ctrl: dict = {}         # 真实对局进程 → 控制器 id
    obs_of: dict = {}           # 推演池进程 → 等网络选动作的观测
    it_iter = iter(items)
    n_live = 0
    cid_seq = [0]
    t_start = time.time()
    n_done = 0
    stat = {"fwd": 0, "obs": 0, "t": time.time()}

    def advance(cid, value):
        nonlocal n_live, n_done
        gen, i, arm, te = ctrls[cid]
        try:
            kind, payload = gen.send(value)
        except StopIteration as e:
            del ctrls[cid]
            truth_free.append(te)
            rec = partial.setdefault(i, {})
            rec[arm] = e.value
            if len(rec) == len(arms):
                it = items_by_id[i]
                line = {"i": i, "encounter": it["encounter"], "next": it["next_encounter"], "hp_frac": round(it["hp_frac"], 4),
                        **{k: rec[k] for k in arms}, "t": round(time.time())}
                out.write(json.dumps(line, ensure_ascii=False) + "\n"); out.flush()
                del partial[i]
                n_live -= 1
                n_done += 1
                if n_done % 10 == 0:
                    print(f"{time.strftime('%H:%M:%S')} 做完 {n_done} 题，用时 {time.time() - t_start:.0f} 秒", flush=True)
                start_more()
            return
        if kind == "policy":
            pol_req.append((cid, payload))
        elif kind == "reset":
            env_ctrl[te] = cid
            P.send(te, {"op": "reset", "mode": "combat", "spec": payload, "labels": False})
        elif kind == "step":
            env_ctrl[te] = cid
            P.send(te, {"op": "step", "action": int(payload)})
        elif kind == "import":
            k = imp_rr[0] % len(imps); imp_rr[0] += 1
            try:
                sid, fb = imps[k].import_wire(frame_to_wire(payload, seed=0))
                r = (k, sid, parse(fb))
            except (EngineError, ValueError) as e:
                r = e if isinstance(e, EngineError) else EngineError(str(e))
            advance(cid, r)
        elif kind == "drop":
            try:
                imps[payload[0]].drop(payload[1])
            except EngineError:
                pass
            advance(cid, None)
        else:
            for t in payload:
                t.owner = (cid, t.owner)
                queue[t.engine].append(t)
            waiting[cid] = [payload, 0]

    items_by_id = dict(items)

    def start_more():
        nonlocal n_live
        while n_live < a.live:
            if a.deadline and time.time() > a.deadline:
                return
            try:
                i, it = next(it_iter)
            except StopIteration:
                return
            n_live += 1
            for arm in arms:
                gen = greedy_ctrl(it["spec"]) if arm == "greedy" else search_ctrl(it["spec"], arm, a, 100003 * (i + 1))
                cid = cid_seq[0]; cid_seq[0] += 1
                ctrls[cid] = [gen, i, arm, truth_free.pop()]
                advance(cid, None)

    def task_done(t: Task, value):
        cid, owner = t.owner
        t.owner, t.result = owner, value
        w = waiting[cid]
        w[1] += 1
        if w[1] == len(w[0]):
            del waiting[cid]
            advance(cid, [x.result for x in w[0]])

    def imp_feed(key, t: Task, kind: int, body: bytes):
        """导入接法的一步走完：kind 0 为帧，1 为终局摘要（JSON）。"""
        if kind == 1:
            j = json.loads(body)
            del imp_task[key]; imps[key[0]].drop(key[1])
            r = {"win": bool(j.get("won")), "hp": int(j.get("hp", 0)), "max_hp": max(1, int(j.get("max_hp", 1))), "stage": t.stage0}
            if not r["win"]:
                r["enemy_left"] = round(min(1.0, t.left if t.left is not None else 1.0), 4)
            task_done(t, r)
            return
        o = parse(body)
        if o.done or o.stuck or o.num_actions == 0 or t.steps > MAX_STEPS:
            del imp_task[key]; imps[key[0]].drop(key[1])
            task_done(t, outcome(o, t.left) if t.steps <= MAX_STEPS else dict(LOSE))
            return
        t.left = enemy_left(o)
        imp_obs[key] = o

    def imp_start(batch: list[Task]):
        """导入接法：每个世界从根状态复制、reseed，再走候选；同一个进程的第一步合成一批。"""
        by: dict = {}
        for t in batch:
            k, root = t.root[0], t.root[1]
            if t.first >= t.root[2].num_actions:
                task_done(t, EngineError(f"候选越界：Rust 根局面只有 {t.root[2].num_actions} 个候选"))
                continue
            try:
                c = imps[k].clone(root)
                imps[k].reseed(c, t.seed)
            except EngineError as e:
                task_done(t, e)
                continue
            t.key = (k, c); imp_task[t.key] = t
            by.setdefault(k, []).append((c, t.first))
        imp_steps(by)

    def imp_steps(by: dict):
        for k, pairs in by.items():
            try:
                res = imps[k].steps(pairs)
            except EngineError as e:
                # 整批里有一项出错时协议只回错误、之前的步已经走了：这一批的推演全部作废（不重试，免得重复走步）
                res = [(2, str(e).encode())] * len(pairs)
            for (sid, _), (kind, body) in zip(pairs, res):
                key = (k, sid); t = imp_task.pop(key)
                if kind == 2:
                    try:
                        imps[k].drop(sid)
                    except EngineError:
                        pass
                    task_done(t, EngineError(body.decode("utf-8", "replace")))
                else:
                    imp_task[key] = t
                    imp_feed(key, t, kind, body)

    def feed(i, o):
        if i in env_ctrl:
            advance(env_ctrl.pop(i), o)
            return
        t = env_task[i]
        if isinstance(o, EngineError) or t.mode == "obs":
            del env_task[i]; pool_free[P.kind[i]].append(i)
            task_done(t, o)
            return
        if o.done or o.stuck or o.num_actions == 0 or t.steps > MAX_STEPS:
            del env_task[i]; pool_free[P.kind[i]].append(i)
            task_done(t, outcome(o, t.left) if t.steps <= MAX_STEPS else dict(LOSE))
            return
        t.left = enemy_left(o)
        obs_of[i] = o

    start_more()
    while ctrls:
        for eng in ("rs", "cs"):
            while pool_free[eng] and queue[eng]:
                i = pool_free[eng].pop(); t = queue[eng].pop()
                env_task[i] = t
                req = {"op": "reset", "mode": "combat", "spec": t.spec, "replay": t.prefix, "labels": False}
                if t.seed is not None:
                    req["reseed"] = int(t.seed) & 0xFFFFFFFF
                    if t.first is not None:
                        req["first"] = int(t.first)
                P.send(i, req)
        if queue["imp"]:
            b_ = queue["imp"]; queue["imp"] = []
            imp_start(b_)
        for i, o in P.recv_ready(0.5 if not (obs_of or pol_req or imp_obs) else 0.0):
            feed(i, o)
        deadline = time.time() + 0.002
        while P.pending and time.time() < deadline:
            for i, o in P.recv_ready(max(0.0, deadline - time.time())):
                feed(i, o)
        if not obs_of and not pol_req and not imp_obs:
            continue
        ids = list(obs_of)
        keys = list(imp_obs)
        reqs = list(pol_req); pol_req.clear()
        batch = [obs_of[i] for i in ids] + [imp_obs[k] for k in keys] + [o for _, o in reqs]
        stat["fwd"] += 1; stat["obs"] += len(batch)
        if time.time() - stat["t"] > 60:
            print(f"{time.strftime('%H:%M:%S')} 网络前向 {stat['fwd']} 次、{stat['obs']} 个观测（平均每批 {stat['obs'] / stat['fwd']:.1f}），"
                  f"在做 {n_live} 题、排队推演 {len(queue['rs']) + len(queue['cs'])}", flush=True)
            stat["t"] = time.time()
        with torch.no_grad():
            logits, values = net(collate_ragged(Ragged.from_obs(batch, np.array([MODE_COMBAT] * len(batch)))).to(device))
        logits, values = logits.float().cpu(), values.float().cpu()
        for k, i in enumerate(ids):
            o = obs_of.pop(i)
            t = env_task[i]
            if a.leaf and t.mode == "rollout" and t.seed is not None and t.steps >= a.leaf:
                del env_task[i]; pool_free[P.kind[i]].append(i)
                task_done(t, {"win": False, "hp": int(o.hp), "max_hp": max(1, int(o.max_hp)), "value": float(values[k])})
                continue
            t.steps += 1
            P.send(i, {"op": "step", "action": int(logits[k, : o.num_actions].argmax())})
        by: dict = {}
        for k, key in enumerate(keys):
            o = imp_obs.pop(key)
            t = imp_task[key]
            j = len(ids) + k
            if a.leaf and t.steps >= a.leaf:
                del imp_task[key]; imps[key[0]].drop(key[1])
                task_done(t, {"win": False, "hp": int(o.hp), "max_hp": max(1, int(o.max_hp)), "value": float(values[j])})
                continue
            t.steps += 1
            by.setdefault(key[0], []).append((key[1], int(logits[j, : o.num_actions].argmax())))
        imp_steps(by)
        off = len(ids) + len(keys)
        for k, (cid, o) in enumerate(reqs):
            advance(cid, torch.softmax(logits[off + k, : o.num_actions], -1).numpy())
    P.close()
    for x in imps:
        x.close()
    print(f"完成 {n_done} 题，用时 {time.time() - t_start:.0f} 秒", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--exam", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", default="greedy,rs", help="greedy / rs / cs / imp，逗号分隔")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--live", type=int, default=4, help="同时在做的题数")
    ap.add_argument("--rs-envs", type=int, default=8)
    ap.add_argument("--cs-envs", type=int, default=8)
    ap.add_argument("--imp-procs", type=int, default=2, help="导入接法用几个 sts2-serve search 进程（同步调用）")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--tau", type=float, default=0.9)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--pmin", type=float, default=0.02)
    ap.add_argument("--sims0", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--z", type=float, default=2.0)
    ap.add_argument("--leaf", type=int, default=12)
    ap.add_argument("--no-root-check", dest="root_check", action="store_false")
    ap.add_argument("--no-cand-guard", dest="cand_guard", action="store_false", help="候选对不上时照样搜索（默认按贪心走）")
    ap.add_argument("--deadline", type=float, default=0, help="Unix 时间：过了就不再开新题")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    blob = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    net = PolicyValueNet(ModelConfig(**blob["config"]["model"])).eval().to(a.device)
    load_state(net, blob["state"])
    exam = json.loads(Path(a.exam).read_text(encoding="utf-8"))["items"]
    items = [(it["id"], it) for it in exam]
    if a.limit:
        items = items[: a.limit]
    k, n = map(int, a.shard.split("/"))
    items = [x for j, x in enumerate(items) if j % n == k]
    print(f"版本 {blob.get('version')}，{len(items)} 道题；tau={a.tau} k={a.k} pmin={a.pmin} sims0={a.sims0} rounds={a.rounds} "
          f"z={a.z} leaf={a.leaf}；Rust {os.environ.get('STS2_RUST_BIN')}", flush=True)
    run(a, items, net, a.device)


if __name__ == "__main__":
    main()
