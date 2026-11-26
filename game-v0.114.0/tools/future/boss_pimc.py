"""未来推演方案初测之一：第三幕双 Boss 题上的公开信息 PIMC 老师（2026-11-16）。

依据 reports/2026-11-15_未来推演上下文与低算力规划（下称“报告”）第 8 节阶段 1：先证明老师强于当前强基线（愚者 L1000），
再谈蒸馏或候选未来上下文。用户：“现在B机器空着，请你充分利用B机器对这些方案做初步测试，然后选一个你觉得最靠谱的方案实践于愚者训练”。

题目：L1000 千种池（W1 前 1,000 个种子）整局评测里打到第三幕第一个 Boss 时记下的开打前快照（ability_test 结果的 boss3 项），
连打两个 Boss（同 tools/boss3_exam.py 与训练的连打练习）。“实机”是战斗模式的一局，用题目自己的种子。

快速“复制”：战斗模式开局只需要这场战斗的快照，所以推演的每个世界 = reset（同一题）+ replay（这场战斗里已走的动作）+ reseed（换掉以后所有随机数、
重洗抽牌堆）+ first（要比较的候选），一次往返，不用像直播那样从整局开头重放（engine/Sts2SimServe/直播接入.md）。
PIMC 照直播的无头搜索（python/sts2rl/headless_search.py 的 _search，sls2-live113）：最想选的项概率 < tau 才搜，前 k 个候选（各 ≥ pmin），
共同随机数逐轮减半，走候选后贪心推演 leaf 步用价值头估值，与贪心逐世界配对、平均差超过 z 个标准误才改选。

每个搜过的根记一行标签（根帧原始字节、候选、各世界得分与叶子读数），给后面的离线学习对照用（tools/future/heads.py）。
--temp > 0 且 --no-search 时按温度采样出手、逐步存帧（同策略采样数据，给“只用真实所选动作的未来标签”那一组）。

推演的几个可选开关（2026-11-16，引擎的局面复制，engine/Sts2Headless/Engine/Clone/；默认全关，默认行为与以前完全相同）：
  --worlds clone：世界不再 reset + replay。每个推演进程先跟到根局面——第一次 reset + replay 到根并顺手存进槽位 0（"snapshot":0），
      之后 restore 上一个根 + replay 实机新走的几步 + "snapshot":0；然后每个世界 {"op":"restore","slot":0,"reseed":s,"first":c}，接着照常推演。
      跟根的请求与这个进程这一轮第一个世界的请求一起发出（不多等一次往返）。规则上与 replay 完全等价（tools/clone/clone_fidelity.py 验收：
      恢复出的每一帧与 reset + replay 逐字节相同），帧相同则网络的出手、估值相同，搜索结果（各世界得分、改选、胜负）应当一模一样。
  --async：一轮里不再等最慢的进程锁步。哪个世界的帧到了，就和同一时刻已经到了的帧一起合成一批过网络，算完立刻发下一步。
      同一组世界（候选 × 种子）、同样的种子和步数规则、轮与轮之间照旧等齐（逐轮减半要用上一轮的结果），所以与锁步版统计等价；
      唯一的差别是每批合成的帧不同，网络前向的浮点结果可能在最后几位不同（批里补齐的长度不同），出手的最大项碰上几乎打平时可能翻转，
      所以单个世界的得分不保证逐位相同（锁步版换个推演进程数也一样会变：批的组成跟着变）。
      --async-wait 毫秒：第一个帧到了之后最多再等这么久凑批（默认 0 为纯异步）。CPU 推理时网络前向是瓶颈，纯异步每批的行数变少、前向次数变多，
      不一定更快；工作进程多、引擎耗时参差（如带重放）或用 GPU 时才明显得益。
  --auto-single：推演请求带 "auto_single":N（N 为离叶子估值还剩几步，不越过叶子那一帧），只有一个候选的决策由引擎自己走过（帧的附加信息 auto_steps 为走了几步），
      省掉这些帧的编码与网络前向。计步为 1 + auto_steps，且自动走不越过第 leaf 帧（引擎 auto_single 带上限），估值的那一帧与逐步走相同。
      （剩下的差别只来自网络前向的批组成不同带来的浮点误差，见 pimc_compare 的说明。）
  --keep-top：世界的 reseed 带 "keep_top":true，已知的抽牌堆顶（被效果放到顶部、玩家知道位置的牌）不洗。改变世界的分布（更接近玩家知道的信息）。
  --infer：推理路径（默认空，原路径 net(batch) + no_grad）。"fp32" 为推理专用前向 InferEngine；"unix:/tmp/pimc.sock" 为共用推理进程
      （先 tools/pimc_infer.py serve 起，分片带 --device cpu，不载网络、不建显卡上下文；显存 16 份 → 1 份）。精度保持 fp32（fp16/bf16 会改决策），
      结论与测量见 reports/PIMC推理侧提速.md。
  环境变量 STS2_TIMING=1 时引擎在退出时把各项耗时写进 engine_s{分片}.log（[timing] 行）。

  build：python3 tools/future/boss_pimc.py build --runs '/root/ens_fool/L1/runs_s*.jsonl' --out items.json
  run：  .venv/bin/python -I tools/future/boss_pimc.py run --items items.json --ckpt L1000.pt --out 目录 --shard 0/16 [--no-search] [--temp 1 --save-frames]
         [--worlds clone] [--async] [--auto-single] [--keep-top]
"""
from __future__ import annotations

import argparse
import base64
import glob
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

EXTRA = {"loop_cut": True, "stall_turns": 20}       # 同评测（eval_stall_turns 20）
MODE_COMBAT = 0


def build(a) -> None:
    """--key boss3（默认）：第三幕第一个 Boss 战前的快照 + 两个 Boss，连打；题号按读到的先后。
    --key boss1 / boss2（2026-11-17，ability_test --boss-snaps 的结果）：第一 / 二幕 Boss 战前的快照，单打这一场；给了 --seeds 时按种子池里的顺序编号。"""
    act = {"boss1": 0, "boss2": 1, "boss3": 2}[a.key]
    rows = []
    for f in sorted(glob.glob(a.runs)):
        for l in open(f, encoding="utf-8"):
            if not l.strip():
                continue
            r = json.loads(l)
            b = r.get(a.key)
            if not (b and b.get("snap") and (b.get("next_encounter") or a.key != "boss3")):
                continue
            rows.append((r, b))
    if a.seeds:
        order = {s: i for i, s in enumerate(json.loads(Path(a.seeds).read_text(encoding="utf-8"))["seeds"])}
        rows.sort(key=lambda x: order.get(x[0]["seed"], 1 << 30))
    items = []
    for r, b in rows:
        spec = {"character": r.get("character", "FOOL"), "ascension": 10, "seed": r["seed"], "act": act, "encounter": b["encounter"],
                "room": "Boss", "player_snapshot": b["snap"]}
        if a.key == "boss3":
            spec["next_encounter"] = b["next_encounter"]
        items.append({"id": len(items), "seed": r["seed"], "run_win": int(r["win"]), "hp": b.get("hp"), "max_hp": b.get("max_hp"), "spec": spec})
    Path(a.out).write_text(json.dumps({"from": a.runs, "key": a.key, "items": items}, ensure_ascii=False), encoding="utf-8")
    print(f"{len(items)} 题（整局里通关 {sum(i['run_win'] for i in items)}）→ {a.out}")


class Runner:
    def __init__(self, a):
        import torch
        from sts2env.env import _Process
        import sim_pimc as X
        from sts2rl.model import ModelConfig, PolicyValueNet, load_state
        self.a, self.torch, self.X = a, torch, X
        blob = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        self.version = int(blob.get("version", 0))
        remote = (a.infer or "").startswith("unix:")
        if remote and a.residual:
            raise SystemExit("--residual 挂在本地网络上，共用推理进程（--infer unix:…）里没有；要用就给 pimc_infer.py serve 加同样的挂法")
        net = None
        if not remote:      # 共用推理进程时本分片不载网络、不建显卡上下文（reports/PIMC推理侧提速.md 第九节）
            net = PolicyValueNet(ModelConfig(**blob["config"]["model"])).eval().to(a.device)
            state = blob.get("ema_state") if a.ema and blob.get("ema_state") else blob["state"]
            # 换游戏版本的旧检查点（2026-11-21，论文补充实验：故障机器人 v0.111.0 的网络在 v0.113.1 上搜索）：同 tools/ability_test.load_net，
            # 内容先验按特征名对齐、检查点之后追加的词表行清零。在当前版本练的检查点（content_names 相同、vocab_n 等于词表大小）上两步都不起作用
            load_state(net, state, blob.get("content_names"))
            from sts2rl.catalog import Catalog
            from sts2rl.model import zero_new_vocab
            old_n, vocab_n = blob.get("vocab_n") or 1873, Catalog().vocab_size
            if old_n < vocab_n:
                with torch.no_grad():
                    n_unk = zero_new_vocab(net, old_n, vocab_n)
                print(f"词表 {old_n} → {vocab_n} 项：新增 ID 的嵌入行已清零（其中 {n_unk} 个旧版记 <unk> 的取 <unk> 那一行）", flush=True)
        self.net = net
        self.residual = None
        if a.residual:
            from future_heads import attach_residual
            self.residual = attach_residual(net, a.residual, a.device, scale=a.residual_scale)
        # 推理路径（--infer，sim_pimc.make_evaluator）：空为原路径 net(batch)；"fp32" 等为 InferEngine；"unix:路径" 为共用推理进程
        self.ev = X.make_evaluator(a.infer or None, net, a.device, policy_mode=MODE_COMBAT, value_mode=MODE_COMBAT, strip_priv=True)
        self.P = X.PimcParams(tau=a.tau, k=a.k, sims0=a.sims0, rounds=a.rounds, leaf=a.leaf, z=a.z)
        log = open(os.path.join(a.out, f"engine_s{a.k_shard}.log"), "ab")
        dll = Path(a.dll)
        self.real = _Process(dll, log)
        self.workers = [] if a.no_search else [_Process(dll, log) for _ in range(a.procs)]
        self.rng = np.random.default_rng(1000003 * (a.policy_seed or 0) + a.k_shard)
        # --worlds clone：每个推演进程槽位 0 里是哪个根（(题目键, 根之前走过的动作数)）；None 为还没有或已作废
        self.wroot: list = [None] * len(self.workers)
        self.cur_key = None
        self.sel = None
        self.stat = {"auto_steps": 0, "replies": 0, "batches": 0, "rows": 0, "sync": 0, "overshoot": 0}
        if a.warm and self.workers:
            self.warm()

    def warm(self) -> None:
        """--warm：推演进程先各自开一局、走一次 replay + reseed + first（.NET 第一次跑要 JIT），--worlds clone 时再拍一次快照、恢复一次
        （第一次拍快照要找一遍静态根，约 0.4 秒）。不影响之后的结果（clone 模式下槽位 0 作废，第一个根照样从 reset 跟）；只让计时不含这些一次性开销。"""
        # 只走一步（开局第一个候选），不会走到战斗结束；clone 时就在这一步顺手拍快照
        warm = {"op": "reset", "mode": "combat", "spec": {"character": "FOOL", "ascension": 10, "seed": "WARM", "act": 2, "encounter": "QUEEN_BOSS"},
                **EXTRA, "labels": False, "replay": [], "reseed": 1, "first": 0}
        if self.a.worlds == "clone":
            warm["snapshot"] = 0
        for wk in self.workers:
            wk.send(warm)
            if self.a.worlds == "clone":
                wk.send({"op": "restore", "slot": 0, "reseed": 2, "first": 0})
                wk.send({"op": "restore", "slot": 0})
                wk.send({"op": "drop"})
        from sts2env.env import EngineError
        for wk in self.workers:
            for _ in range(4 if self.a.worlds == "clone" else 1):
                try:
                    wk.recv()       # 原始字节，不解析（drop 的回复是 JSON）
                except EngineError:
                    pass
        self.wroot = [None] * len(self.workers)

    def close(self):
        for w in [self.real, *self.workers]:
            w.close()

    def recv(self, w):
        from sts2env.env import EngineError
        from sts2env.protocol import parse
        try:
            raw = w.recv()
        except EngineError as e:
            return None, None, str(e)[:200]
        return parse(raw), raw, None

    def gated(self, p: np.ndarray) -> bool:
        n = len(p)
        if n <= 1:
            return False
        order = np.argsort(-p)
        return bool(p[order[0]] < self.P.tau and sum(1 for k in order[: self.P.k] if p[k] >= self.P.pmin) > 1)

    def search(self, base: dict, hist: list[int], root, p: np.ndarray, dec_seed: int) -> dict:
        """PIMC（同 headless_search._search）；另记每个世界的叶子读数。"""
        P, X = self.P, self.X
        t0 = time.perf_counter()
        order = np.argsort(-p)
        act = int(order[0])
        cands = [int(c) for c in order[: P.k] if p[c] >= P.pmin] or [act]
        sc: dict[int, dict[int, float]] = {c: {} for c in cands}
        meas: list[list] = []
        alive, n_new, used, errors = list(cands), P.sims0, 0, 0
        for _ in range(P.rounds):
            seeds = [dec_seed + used + j for j in range(n_new)]
            used += n_new
            worlds = [{"cand": c, "seed": s} for c in alive for s in seeds]
            self._run_worlds(worlds, base, hist, root)
            for w in worlds:
                if w.get("error"):
                    errors += 1
                elif w.get("score") is not None:
                    sc[w["cand"]][w["seed"]] = w["score"]
                    meas.append([w["cand"], w["seed"], round(w["score"], 4), w["hp"], round(w["left"], 4), w["stage"], w["term"], w["steps"]])
            if len(alive) <= 2:
                continue
            mean = lambda c: sum(sc[c].values()) / len(sc[c]) if sc[c] else -9.0
            rest = sorted((c for c in alive if c != act), key=lambda c: (-mean(c), -p[c]))
            alive = [act] + rest[: max(1, math.ceil(len(alive) / 2) - 1)]
            n_new *= 2
        best, best_m, diff = act, 0.0, {}
        for c in sc:
            if c == act:
                continue
            common = [s for s in sc[c] if s in sc[act]]
            if len(common) < 2:
                continue
            d = [sc[c][s] - sc[act][s] for s in common]
            m = sum(d) / len(d)
            sd = math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1))
            se = sd / math.sqrt(len(d))
            diff[c] = (round(m, 4), round(se, 4), len(d))
            if m > best_m and m > P.z * se:
                best, best_m = c, m
        return {"best": int(best), "greedy": act, "cands": cands, "diff": {str(k): v for k, v in diff.items()}, "meas": meas,
                "errors": errors, "ms": round((time.perf_counter() - t0) * 1000.0, 1)}

    def _world_reqs(self, j: int, w: dict, base: dict, hist: list[int]) -> list[dict]:
        """把世界 w 交给第 j 个推演进程要发的请求：最后一个的回复是这个世界的第一帧，之前的（跟根）回复丢掉。"""
        a = self.a
        rs = {"reseed": int(w["seed"]) & 0xFFFFFFFF, "first": int(w["cand"])}
        if a.keep_top:
            rs["keep_top"] = True
        # 战后奖励也确定化（2026-11-17，直播会话找到的漏口：Reseed 只换整局随机数流，推演走到战后奖励帧时看到的是真实的金币、药水、遗物；
        # 引擎 RunSession.ReseedRewards）。默认开；--no-reseed-rewards 复现旧结果。旧引擎不认这个键，照旧（漏口还在）
        if getattr(a, "reseed_rewards", True):
            rs["rewards"] = True
        if a.auto_single:
            rs["auto_single"] = self._auto_cap(0)
        if a.worlds != "clone":
            return [{**base, "labels": False, "replay": list(hist), **rs}]
        pre = []
        st = self.wroot[j]
        if st is None or st[0] != self.cur_key or st[1] > len(hist):
            pre.append({**base, "labels": False, "replay": list(hist), "snapshot": 0})
        elif st[1] < len(hist):
            pre.append({"op": "restore", "slot": 0, "replay": list(hist[st[1]:]), "snapshot": 0})
        self.stat["sync"] += len(pre)
        self.wroot[j] = (self.cur_key, len(hist))
        return pre + [{"op": "restore", "slot": 0, **rs}]

    def _auto_cap(self, steps: int):
        """auto_single 最多自动走几步：已收到 steps 帧，这次请求回来的是第 steps + 1 帧，自动走 k 步后是第 steps + 1 + k 帧；
        不越过第 leaf 帧（叶子估值的那一帧，原来逐步走时在这一帧估值），计步与逐步走完全相同。没有叶子估值时不限。"""
        return max(0, self.P.leaf - steps - 1) if self.P.leaf else True

    def _dispatch(self, j: int, w: dict, base: dict, hist: list[int], left0: float) -> None:
        w.update(left=left0, steps=0, frame=None)
        reqs = self._world_reqs(j, w, base, hist)
        w["skip"] = len(reqs) - 1
        for r in reqs:
            self.workers[j].send(r)

    def _take(self, j: int, w: dict) -> str:
        """读第 j 个进程的一个回复。返回 "skip"（跟根的回复，已丢掉）、"dead"（这个世界作废，回复已读完）、"frame"（w["frame"] 已更新）。"""
        o, _, err = self.recv(self.workers[j])
        self.stat["replies"] += 1
        if w["skip"] > 0:
            w["skip"] -= 1
            if err:
                # 跟根失败：槽位 0 不可信，这个世界作废（之后那个 restore 的回复照样要读掉）；下次这个进程从 reset 重新跟
                w["error"] = err
                self.wroot[j] = None
            return "skip"
        if w.get("error"):
            return "dead"
        if err:
            w["error"] = err
            if self.a.worlds == "clone":
                self.wroot[j] = None
            return "dead"
        w["frame"] = o
        n = 1
        if self.a.auto_single and isinstance(o.extra, dict):
            k = int(o.extra.get("auto_steps", 0))
            self.stat["auto_steps"] += k
            if self.P.leaf and w["steps"] < self.P.leaf < w["steps"] + 1 + k:
                self.stat["overshoot"] += 1      # 第 leaf 步落在被自动走过的单候选帧上（估值改在之后第一个多候选的帧）
            n += k
        w["steps"] += n
        return "frame"

    def _judge(self, w: dict, stage: int) -> bool:
        """世界的新帧：打完了或步数超限就记分（返回 False），否则记下敌人剩余生命、要网络出手（返回 True）。"""
        P, X = self.P, self.X
        o = w["frame"]
        if X.combat_over(o):
            r = X.outcome(o, w["left"])
            w.update(score=X.terminal(r), hp=int(o.hp), stage=int(r["stage"]) + int(r["win"]), term=1 if r["win"] else 2)
            return False
        if w["steps"] > P.max_steps:
            w.update(score=X.terminal({"win": False, "stage": stage}), hp=int(o.hp), stage=stage, term=3)
            return False
        w["left"] = X.enemy_left(o)
        return True

    def _act(self, ask: list[int], busy: dict) -> None:
        """一批要出手的世界合成一次前向；到 leaf 步的记叶子估值，其余发下一步。"""
        P = self.P
        leaf = [bool(P.leaf) and busy[j]["steps"] >= P.leaf for j in ask]
        acts, vals = self.ev.rollout([busy[j]["frame"] for j in ask], leaf, [(busy[j]["seed"], busy[j]["steps"]) for j in ask])
        self.stat["batches"] += 1
        self.stat["rows"] += len(ask)
        for j, l, a_, v in zip(ask, leaf, acts, vals):
            w = busy[j]
            if l:
                o = w["frame"]
                w.update(score=float(v), hp=int(o.hp), stage=int(o.header[23]), term=0)
                del busy[j]
            else:
                req = {"op": "step", "action": int(a_)}
                if self.a.auto_single:
                    req["auto_single"] = self._auto_cap(w["steps"])
                self.workers[j].send(req)

    def _run_worlds(self, worlds: list[dict], base: dict, hist: list[int], root) -> None:
        if self.a.worlds != "replay" or self.a.async_ or self.a.auto_single or self.a.keep_top:
            return self._run_worlds_new(worlds, base, hist, root)
        P, X = self.P, self.X
        stage = int(root.header[23])
        left0 = X.enemy_left(root)
        queue = list(worlds)
        busy: dict[int, dict] = {}
        while queue or busy:
            for j, wk in enumerate(self.workers):
                if j in busy or not queue:
                    continue
                w = queue.pop(0)
                w.update(left=left0, steps=0, frame=None)
                wk.send({**base, "labels": False, "replay": list(hist), "reseed": int(w["seed"]) & 0xFFFFFFFF, "first": int(w["cand"])})
                busy[j] = w
            if not busy:
                break
            for j in list(busy):
                w = busy[j]
                o, _, err = self.recv(self.workers[j])
                if err:
                    w["error"] = err
                    del busy[j]
                    continue
                w["frame"] = o
                w["steps"] += 1
            ask = []
            for j in list(busy):
                w = busy[j]
                o = w["frame"]
                if X.combat_over(o):
                    r = X.outcome(o, w["left"])
                    w.update(score=X.terminal(r), hp=int(o.hp), stage=int(r["stage"]) + int(r["win"]), term=1 if r["win"] else 2)
                elif w["steps"] > P.max_steps:
                    w.update(score=X.terminal({"win": False, "stage": stage}), hp=int(o.hp), stage=stage, term=3)
                else:
                    w["left"] = X.enemy_left(o)
                    ask.append(j)
                    continue
                del busy[j]
            if ask:
                leaf = [bool(P.leaf) and busy[j]["steps"] >= P.leaf for j in ask]
                acts, vals = self.ev.rollout([busy[j]["frame"] for j in ask], leaf, [(busy[j]["seed"], busy[j]["steps"]) for j in ask])
                for j, l, a_, v in zip(ask, leaf, acts, vals):
                    w = busy[j]
                    if l:
                        o = w["frame"]
                        w.update(score=float(v), hp=int(o.hp), stage=int(o.header[23]), term=0)
                        del busy[j]
                    else:
                        self.workers[j].send({"op": "step", "action": int(a_)})

    def _run_worlds_new(self, worlds: list[dict], base: dict, hist: list[int], root) -> None:
        """带可选开关（--worlds clone、--async、--auto-single、--keep-top）的推演。不带 --async 时与上面的锁步版一步一步对应：
        每一轮先把空闲进程派满，再按进程顺序把忙着的都读一遍、合成一批过网络、发下一步。"""
        stage = int(root.header[23])
        left0 = self.X.enemy_left(root)
        queue = list(worlds)
        busy: dict[int, dict] = {}
        if self.a.async_ and self.sel is None:
            import selectors
            self.sel = selectors.DefaultSelector()
            for j, wk in enumerate(self.workers):
                self.sel.register(wk.p.stdout, selectors.EVENT_READ, j)
        while queue or busy:
            for j in range(len(self.workers)):
                if j in busy or not queue:
                    continue
                w = queue.pop(0)
                self._dispatch(j, w, base, hist, left0)
                busy[j] = w
            if not busy:
                break
            ask = []
            if self.a.async_:
                # 等到至少一个进程有回复；同一时刻已经到了的都收下，合成一批。--async-wait > 0 时第一个到了之后再等最多这么多毫秒，
                # 让其余忙着的进程也赶上这一批（网络前向是瓶颈、每批行数太少时用；0 为纯异步）
                ready = {key.data for key, _ in self.sel.select()}
                if self.a.async_wait > 0:
                    deadline = time.perf_counter() + self.a.async_wait / 1000.0
                    while len(ready & busy.keys()) < len(busy):
                        left = deadline - time.perf_counter()
                        if left <= 0:
                            break
                        more = {key.data for key, _ in self.sel.select(left)}
                        if not more:
                            break
                        ready |= more
                for j in sorted(ready):
                    if j not in busy:
                        continue
                    w = busy[j]
                    got = self._take(j, w)
                    if got == "skip":
                        continue
                    if got == "dead":
                        del busy[j]
                        continue
                    if self._judge(w, stage):
                        ask.append(j)
                    else:
                        del busy[j]
            else:
                for j in list(busy):
                    w = busy[j]
                    got = self._take(j, w)
                    while got == "skip":
                        got = self._take(j, w)
                    if got == "dead":
                        del busy[j]
                        continue
                    if self._judge(w, stage):
                        ask.append(j)
                    else:
                        del busy[j]
            if ask:
                self._act(ask, busy)

    def play(self, item: dict, lab_fh, frame_fh) -> dict:
        a, X, torch = self.a, self.X, self.torch
        base = {"op": "reset", "mode": "combat", "spec": item["spec"], **EXTRA}
        self.cur_key = (item["id"], item["spec"].get("seed"))       # --worlds clone：根换了题就从 reset 重新跟
        # --privileged（2026-11-17，公开信息价值头的训练数据）：只给实机开局要特权特征（抽牌顺序等，同训练采集），推演世界照旧不要
        self.real.send({**base, "labels": False, **({"privileged": True} if getattr(a, "privileged", False) else {})})
        o, raw, err = self.recv(self.real)
        hist: list[int] = []
        t0 = time.perf_counter()
        nsearch = nchange = nsteps = 0
        stage_max = 0
        traj = []
        left = X.enemy_left(o) if o is not None else 1.0
        while err is None and not X.combat_over(o):
            stage_max = max(stage_max, int(o.header[23]))
            n = int(o.num_actions)
            logits, values = self.ev.forward([o], [MODE_COMBAT])
            lg = logits[0, :n].double()
            p = torch.softmax(lg, -1).numpy()
            greedy = int(np.argmax(p))
            act = greedy
            if not a.no_search and n > 1 and self.gated(p):
                dec_seed = (hash((item["id"], nsteps, a.policy_seed or 0)) & 0x3FFFFFFF)
                res = self.search(base, hist, o, p, dec_seed)
                act = res["best"]
                nsearch += 1
                nchange += int(act != greedy)
                if lab_fh is not None:
                    lab_fh.write(json.dumps({"id": item["id"], "t": nsteps, "stage": int(o.header[23]), "n": n, "v0": round(float(values[0]), 4),
                                             "p": [round(float(x), 5) for x in p], **res, "frame": base64.b64encode(raw).decode("ascii")}) + "\n")
            elif a.temp > 0 and n > 1:
                q = torch.softmax(lg / a.temp, -1).numpy()
                act = int(self.rng.choice(n, p=q / q.sum()))
            if frame_fh is not None:
                off = frame_fh.tell()
                frame_fh.write(raw)
                traj.append([off, len(raw), act, n, int(o.hp), round(left, 4), int(o.header[23]), round(float(values[0]), 4), round(float(p[act]), 5)])
            hist.append(act)
            self.real.send({"op": "step", "action": act})
            o, raw, err = self.recv(self.real)
            nsteps += 1
            if o is not None and not X.combat_over(o):
                left = X.enemy_left(o)
            if nsteps > a.cap:
                err = "步数超上限"
        if err is not None:
            out = {"id": item["id"], "win": 0, "stuck": 1, "err": err, "stage2": int(stage_max >= 1), "steps": nsteps}
        else:
            r = X.outcome(o, left)
            out = {"id": item["id"], "win": int(r["win"]), "stuck": int(bool(o.stuck)), "stage2": int(stage_max >= 1 or (r["win"] and r["stage"] >= 1)),
                   "hp": int(o.hp), "steps": nsteps}
        out.update(searches=nsearch, changed=nchange, s=round(time.perf_counter() - t0, 1))
        if a.worlds != "replay" or a.async_ or a.auto_single or a.keep_top:
            out["stat"] = dict(self.stat)       # 推演的计数（只在带可选开关时记，默认输出不变）
            self.stat = {k: 0 for k in self.stat}
        if frame_fh is not None:
            out["traj"] = traj
        return out


def run(a) -> None:
    k, n = map(int, a.shard.split("/"))
    a.k_shard = k
    os.makedirs(a.out, exist_ok=True)
    items = json.loads(Path(a.items).read_text(encoding="utf-8"))["items"]
    if a.item_mod:
        m, r_ = a.item_mod.split(":")
        items = [it for it in items if (it["id"] % int(m) != int(r_[1:]) if r_.startswith("!") else it["id"] % int(m) == int(r_))]
    if a.limit:
        items = items[: a.limit]
    reps = max(1, a.reps)
    todo = [(it, r) for r in range(reps) for i, it in enumerate(items) if (i + r * len(items)) % n == k]
    res_path = Path(a.out) / f"res_s{k}.jsonl"
    done = set()
    if res_path.exists():
        for l in open(res_path, encoding="utf-8"):
            if l.strip():
                d = json.loads(l)
                done.add((d["id"], d.get("rep", 0)))
    todo = [(it, r) for it, r in todo if (it["id"], r) not in done]
    print(f"分片 {k}/{n}：{len(todo)} 题待打", flush=True)
    if not todo:
        return
    R = Runner(a)
    fh = open(res_path, "a", encoding="utf-8")
    lab = open(Path(a.out) / f"labels_s{k}.jsonl", "a", encoding="utf-8") if not a.no_search else None
    frames = open(Path(a.out) / f"frames_s{k}.bin", "ab") if a.save_frames else None
    t0 = time.time()
    try:
        for i, (it, r) in enumerate(todo):
            if a.reps > 1:
                it = {**it, "spec": {**it["spec"], "seed": f"{it['spec']['seed']}R{r}"}}
            res = R.play(it, lab, frames)
            res["rep"] = r
            fh.write(json.dumps(res) + "\n"); fh.flush()
            if lab:
                lab.flush()
            if frames:
                frames.flush()
            if i % 10 == 0:
                print(f"{time.strftime('%H:%M:%S')} 分片 {k}：{i + 1}/{len(todo)}，{time.time() - t0:.0f} 秒", flush=True)
    finally:
        R.close()
    print("完", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--runs", required=True); b.add_argument("--out", required=True)
    b.add_argument("--key", default="boss3", choices=["boss1", "boss2", "boss3"]); b.add_argument("--seeds", default="", help="种子池文件：题号按池里的顺序")
    r = sub.add_parser("run")
    r.add_argument("--items", required=True); r.add_argument("--ckpt", required=True); r.add_argument("--out", required=True)
    r.add_argument("--dll", default=str(REPO / "engine/Sts2Headless/bin/Release/net9.0/Sts2Headless.dll"))
    r.add_argument("--device", default="cuda"); r.add_argument("--ema", action="store_true")
    r.add_argument("--shard", default="0/1"); r.add_argument("--procs", type=int, default=8)
    r.add_argument("--no-search", action="store_true"); r.add_argument("--temp", type=float, default=0.0); r.add_argument("--policy-seed", type=int, default=0)
    r.add_argument("--save-frames", action="store_true"); r.add_argument("--reps", type=int, default=1); r.add_argument("--limit", type=int, default=0)
    r.add_argument("--privileged", action="store_true", help="实机开局带特权特征（存帧给公开信息价值头当老师读数用；出手照旧去掉特权）")
    r.add_argument("--item-mod", default="", help="只打 id %% m == r 的题（m:r；r 前加 ! 为不等于）")
    r.add_argument("--tau", type=float, default=0.9); r.add_argument("--k", type=int, default=4); r.add_argument("--sims0", type=int, default=4)
    r.add_argument("--rounds", type=int, default=2); r.add_argument("--leaf", type=int, default=12); r.add_argument("--z", type=float, default=2.0)
    r.add_argument("--cap", type=int, default=3000)
    r.add_argument("--residual", default=""); r.add_argument("--residual-scale", type=float, default=1.0)
    r.add_argument("--infer", default="", help="推理路径（sim_pimc.make_evaluator）：空为原路径；fp32 等为 InferEngine；unix:套接字 为共用推理进程"
                   "（tools/pimc_infer.py serve 起，分片用 --device cpu）。搜索上用 fp32，见 reports/PIMC推理侧提速.md")
    r.add_argument("--worlds", choices=["replay", "clone"], default="replay", help="世界怎么到根：reset + replay（默认）或局面复制（restore 槽位 0）")
    r.add_argument("--async", dest="async_", action="store_true", help="一轮里不锁步：哪个世界的帧到了就和已到的一起过网络")
    r.add_argument("--async-wait", type=float, default=0.0, help="--async 时第一个帧到了之后最多再等几毫秒凑批（默认 0：纯异步）")
    r.add_argument("--auto-single", action="store_true", help="推演请求带 auto_single：单候选的决策由引擎走过，计步 1 + auto_steps")
    r.add_argument("--keep-top", action="store_true", help="世界的 reseed 带 keep_top：已知的抽牌堆顶不洗")
    r.add_argument("--warm", action="store_true", help="开始前让推演进程各跑一次（JIT 与第一次拍快照的一次性开销不算进搜索耗时）")
    a = ap.parse_args()
    {"build": build, "run": run}[a.cmd](a)


if __name__ == "__main__":
    main()
