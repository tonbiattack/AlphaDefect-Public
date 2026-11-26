"""第三幕双 Boss 题上的信息集树搜索（Gumbel 式，2026-11-17）。

用户：“停下吧，我们不考虑继续训练，而是考虑一下更优秀的搜索算法”；看过候选方案后：“先做树搜索”。

为什么：现在的公开信息 PIMC（boss_pimc.py，愚者决策记录第 25 条测试一的老师）每个犹豫的决策只比较前几个候选，走完候选以后一律贪心推演 12 步再估值，
连着两三步都不走贪心的打法它找不到；每个世界推演约 11 步、每步过一次网络，推理成了瓶颈（reports/局面复制无头引擎报告.md 第六节）。
引擎有了局面快照（restore 约 1 ms）以后，可以改成树：每次模拟只在新节点上过一次网络（价值头估值 + 策略头给先验），不再贪心推演。

一、树（单观察者、随机处分叉的闭环树）
  - 每次模拟用自己的世界：restore 根快照 + reseed（换掉以后所有随机数、重洗抽牌堆，玩家看得到的局面不变）+ first（根动作），之后按树往下走。
    搜索只用公开信息（价值头 strip_priv、世界 reseed），同 boss_pimc，属于决策记录 214 允许的公开信息搜索。
  - 动作按语义键认（sim_pimc.cand_keys 去掉同名序号：牌的 ID、升级、附魔、费用、区域，目标是第几个敌人等），同一个动作在不同世界里下标不同也对得上；
    同一帧里键相同的候选（两张一样的牌）合成一个动作，先验相加，走下标最小的那个。
  - 节点 = 父节点 + 动作 + 走完以后的公开局面签名（血量、格挡、能量、手牌、敌人的生命与格挡、决策类型、全部候选的键）。
    确定的出牌在各个世界里得到同一个签名，共用节点；结束回合、抽牌这类随机结果不同时签名不同，分成不同的子节点（机会节点）。
    冒烟时先做的是开环树（只按动作序列分节点）：结束回合以后各个世界的手牌不同，并进同一个节点，几乎每次模拟都碰到先验里没有的牌，所以改成闭环。
  - 第一次到达的节点就是叶子：这一帧过一次网络，价值头的估值回传，策略头的概率存成这个节点的先验。战斗（两个 Boss）打完按 sim_pimc.terminal 记分，
    与价值头同一尺度。只有一个候选的决策由引擎自己走过（auto_single），不进树。

二、非根节点的选择（Gumbel MuZero，Danihelka et al. 2022，mctx 的 gumbel_muzero_policy）
  π' = softmax(log 先验 + σ(补全 Q))，选 argmax_a [π'(a) − N(a) / (1 + Σ_b N(b))]；
  补全 Q：访问过的用 Q(a)，没访问的用混合值 v_mix = (v_节点 + ΣN · Σ_访问过 π(b)Q(b) / Σ_访问过 π(b)) / (1 + ΣN)；
  σ(q) = (c_visit + max_b N(b)) · c_scale · q̃，q̃ 是补全 Q 按本节点的最小、最大值归一到 [0, 1]（--qnorm minmax，mctx 默认）
  或除以固定的价值幅度（--qnorm fixed，--value-range）。动作的访问次数 N 在下行时就加（并发模拟的虚拟访问），Q 只算已回传的。

三、根（逐轮减半，与老师同形）
  候选 = 前 k 个（各 ≥ pmin，按键合并后），贪心项一定在内。预算 --sims 次模拟，ceil(log2 m) 轮；每轮把这一轮的预算平分给还在的候选，
  同一轮里各候选用同一组世界种子（共同随机数）。每轮后留贪心项与其余里得分最高的一半（--root-keep：mean 按平均回报，gumbel 按 log 先验 + σ(补全 Q)）。
  最后的选择（--decide）：
    kl（默认）：argmax log 先验 + Q / λ（KL 正则化的策略改进，Grill et al. 2020；只比模拟次数 ≥ --min-visits 的候选）。λ 越小越敢改：
      离开先验 p_g 的首选、改走 p_a 的候选，要 Q_a − Q_g > λ·log(p_g / p_a)。
    gumbel：argmax log 先验 + σ(补全 Q)（不加 Gumbel 噪声，评测用确定性版本）；
    z：与贪心项逐世界配对、平均差超过 z 个标准误才改选（老师的判据）。树里不适用：同一回合里出牌后的局面各世界相同、叶子估值相同，
      配对差的标准误为 0，退化成“高一点就换”（冒烟 3 道题改选 22%–30%，老师 12.5%）；同一根动作的各次模拟深度不同，回报也不是独立样本。

四、叶子短推演（--roll L，2026-11-17；默认 0 = 叶子直接用价值头，与以前逐位相同）
  留出题 618 道上 32 次模拟（λ 0.03）比贪心 +5.50，但 128 次、256 次没有更好（逐题配对 −2.76、−1.92，各约 300 题），
  怀疑叶子只靠价值头时模拟越多越会挑中价值头高估的走法。用户：“接着试”（试叶子加短推演）。
  --roll L：到叶子时这一帧照常过网络（先验、价值头），然后按策略头最大项贪心走 L 步（单候选由引擎走过，计步同 boss_pimc 的 leaf），
  第 L 步那一帧用价值头估值；推演中战斗打完按 terminal 记分。回传的值 = (1 − m)·叶子那一帧的价值头 + m·推演得分（--roll-mix m，默认 1，
  即同推演搜索 W8 L12 只用推演得分；AlphaGo 用 0.5）。这次模拟展开的节点，v0（补全 Q 里没访问的动作用的混合值）改成回传的值。

  --value-ckpts（2026-11-17）：叶子与推演末端的估值取主网络与另几份网络价值头的平均（先验、推演出手、实机贪心仍只用主网络）。
  用来看“更准的战斗价值”能不能让搜索更强（单独练战斗网络之前的便宜检验）。
  --oracle（上限诊断）：世界不 reseed，恢复出的局面带着实机真实的未来随机数，搜索等于看得见以后的抽牌与敌人行动。
  违反公开信息，不能当搜索用；只用来量这批题里还有多少局是“看得见未来也赢不了”的，即搜索还剩多少空间。

五、结果与老师、贪心逐题配对：res_s*.jsonl 与 boss_pimc 同格式（id、win、searches、changed、s），i2a.py status 直接能读；另记 tree（模拟、估值、深度等）。

  run：.venv/bin/python -I tools/future/tree_search.py run --items 题.json --item-mod 4:0 --ckpt L1000.pt --out 目录 --shard 0/8 --sims 128
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import boss_pimc  # noqa: E402

MODE_COMBAT = 0


class Node:
    """一个公开局面。按动作记边上的统计；动作走完以后按签名分到子节点。"""
    __slots__ = ("N", "Nq", "W", "out", "v0", "prior", "expanded")

    def __init__(self):
        self.N: dict = {}         # 动作键 → 访问次数（下行时就加，含还没回传的）
        self.Nq: dict = {}        # 动作键 → 已回传的次数
        self.W: dict = {}         # 动作键 → 回报之和
        self.out: dict = {}       # 动作键 → {签名: Node}
        self.v0 = 0.0             # 第一次到达时价值头的估值
        self.prior: dict = {}     # 动作键 → 先验概率（第一次到达那一帧的策略头，按键合并）
        self.expanded = False

    def q(self, k) -> float:
        n = self.Nq.get(k, 0)
        return self.W[k] / n if n else float("nan")


def frame_keys(o, X) -> tuple[list, dict]:
    """这一帧每个候选的键（去掉同名序号），以及键 → 下标最小的候选。"""
    keys = [k[:-1] for k in X.cand_keys(o)]
    first: dict = {}
    for i, k in enumerate(keys):
        first.setdefault(k, i)
    return keys, first


def signature(o, X, keys: list) -> tuple:
    """公开局面的签名：sim_pimc.check_items（血量、格挡、能量、手牌、敌人）+ 决策类型 + 全部候选的键。"""
    c = X.check_items(o)
    return (int(o.decision), c["hp"], c["block"], c["energy"], tuple(c["hand"]), tuple(c["enemies"]), tuple(sorted(map(repr, keys))))


def merged_prior(p: np.ndarray, keys: list) -> dict:
    out: dict = {}
    for i, k in enumerate(keys):
        out[k] = out.get(k, 0.0) + float(p[i])
    return out


class TreeRunner(boss_pimc.Runner):
    """boss_pimc.Runner 的引擎、推理与跟根（--worlds clone）照用，只换 search()。"""

    def __init__(self, a):
        super().__init__(a)
        # --value-ckpts：另载几份网络，叶子（及推演末端）的估值取主网络与它们价值头的平均；先验、推演出手、实机贪心仍只用主网络
        self.vev = []
        if a.value_ckpts:
            from sts2rl.model import ModelConfig, PolicyValueNet, load_state
            for path in a.value_ckpts.split(","):
                b = self.torch.load(path, map_location="cpu", weights_only=False)
                net = PolicyValueNet(ModelConfig(**b["config"]["model"])).eval().to(a.device)
                load_state(net, b["state"])
                self.vev.append(self.X.make_evaluator(None, net, a.device, policy_mode=MODE_COMBAT, value_mode=MODE_COMBAT, strip_priv=True))

    def _values(self, frames: list, values, need: list[int]):
        """need 里那几行的估值换成主网络与 --value-ckpts 各网络价值头的平均（其余行不用估值，不另算）。"""
        if not self.vev or not need:
            return values
        sub = [frames[i] for i in need]
        # --value-replace：只用另几份网络的价值头（主网络只给先验与出手），即单独的战斗价值网络
        acc = np.zeros(len(need)) if self.a.value_replace else np.array([float(values[i]) for i in need])
        for e in self.vev:
            _, v = e.forward(sub, [MODE_COMBAT] * len(sub))
            acc += v.numpy().astype(np.float64).reshape(-1)
        acc /= len(self.vev) + (0 if self.a.value_replace else 1)
        out = [float(x) for x in values]
        for i, x in zip(need, acc):
            out[i] = float(x)
        return out

    # ---------------------------------------------------------------- Gumbel 的打分

    def _completed(self, node: Node, legal: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """legal 里各动作的 log 先验、补全 Q、访问次数。"""
        lp = np.array([math.log(max(node.prior.get(k, 1e-6), 1e-6)) for k in legal])
        Nk = np.array([node.N.get(k, 0) for k in legal], np.float64)
        Qk = np.array([node.q(k) for k in legal])
        vis = ~np.isnan(Qk)
        sum_n = float(sum(node.Nq.get(k, 0) for k in legal))
        if vis.any():
            pr = np.exp(lp[vis]); pr = pr / max(pr.sum(), 1e-12)
            vmix = (node.v0 + sum_n * float((pr * Qk[vis]).sum())) / (sum_n + 1.0)
        else:
            vmix = node.v0
        return lp, np.where(vis, Qk, vmix), Nk

    def _sigma(self, cq: np.ndarray, Nk: np.ndarray) -> np.ndarray:
        a = self.a
        if a.qnorm == "minmax":
            lo, hi = float(cq.min()), float(cq.max())
            qn = (cq - lo) / max(hi - lo, 1e-8)
        else:
            qn = cq / a.value_range
        return (a.c_visit + float(Nk.max() if len(Nk) else 0.0)) * a.c_scale * qn

    def _select(self, node: Node, legal: list):
        lp, cq, Nk = self._completed(node, legal)
        z = lp + self._sigma(cq, Nk)
        pi = np.exp(z - z.max()); pi /= pi.sum()
        score = pi - Nk / (1.0 + Nk.sum())
        return legal[int(np.argmax(score))]

    # ---------------------------------------------------------------- 搜索

    def search(self, base: dict, hist: list[int], root, p: np.ndarray, dec_seed: int) -> dict:
        a, P, X = self.a, self.P, self.X
        t0 = time.perf_counter()
        keys, first = frame_keys(root, X)
        prior = merged_prior(p, keys)
        greedy_idx = int(np.argmax(p))
        gk = keys[greedy_idx]
        order = sorted(prior, key=lambda k: -prior[k])
        cands = [k for k in order[: P.k] if prior[k] >= P.pmin]
        if gk not in cands:
            cands = [gk] + cands[: P.k - 1]
        if len(cands) <= 1 or a.sims <= 0:
            return {"best": greedy_idx, "greedy": greedy_idx, "cands": [first[k] for k in cands], "sims": 0, "ms": 0.0}
        rootn = Node()
        rootn.prior = prior
        rootn.expanded = True
        self.tstat = {"sims": 0, "evals": 0, "term": 0, "depth": 0, "maxd": 0, "err": 0, "nodes": 0, "chance": 0, "roll": 0, "rterm": 0}
        ret: dict = {k: {} for k in cands}      # 根动作键 → {世界种子: 回报}
        alive = list(cands)
        phases = max(1, math.ceil(math.log2(len(cands))))
        used = 0
        for ph in range(phases):
            per = max(1, (a.sims // phases) // len(alive))
            seeds = [dec_seed + used + j for j in range(per)]
            used += per
            self._simulate([{"seed": s, "key0": k} for s in seeds for k in alive], rootn, first, base, hist, root, ret)
            if len(alive) <= 2 or ph == phases - 1:
                continue
            rest = [k for k in alive if k != gk]
            if a.root_keep == "gumbel":
                lp, cq, Nk = self._completed(rootn, rest)
                sc = lp + self._sigma(cq, Nk)
            else:
                sc = np.array([np.mean(list(ret[k].values())) if ret[k] else -9.0 for k in rest])
            keep = [rest[i] for i in np.argsort(-sc, kind="stable")[: max(1, math.ceil(len(alive) / 2) - 1)]]
            alive = [gk] + keep
        # 逐世界配对差（只作记录；z 判据用）
        diff = {}
        for k in cands:
            if k == gk:
                continue
            common = [s for s in ret[k] if s in ret[gk]]
            if len(common) >= 2:
                d = [ret[k][s] - ret[gk][s] for s in common]
                m = sum(d) / len(d)
                sd = math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1))
                diff[k] = (round(m, 4), round(sd / math.sqrt(len(d)), 4), len(d))
        best = gk
        if a.decide == "kl":
            sc = {k: math.log(max(prior[k], 1e-6)) + rootn.q(k) / a.lam for k in cands if rootn.Nq.get(k, 0) >= a.min_visits}
            if sc:
                best = max(sc, key=lambda k: (sc[k], k == gk))
        elif a.decide == "z":
            best_m = 0.0
            for k, (m, se, _) in diff.items():
                if m > best_m and m > P.z * se:
                    best, best_m = k, m
        else:
            lp, cq, Nk = self._completed(rootn, alive)
            best = alive[int(np.argmax(lp + self._sigma(cq, Nk)))]
        st = self.tstat
        for kk in ("sims", "evals", "term", "err", "depth", "nodes", "chance", "roll", "rterm"):
            self.acc[kk] = self.acc.get(kk, 0) + st[kk]
        self.acc["maxd"] = max(self.acc.get("maxd", 0), st["maxd"])
        self.acc["ms"] = self.acc.get("ms", 0.0) + (time.perf_counter() - t0) * 1000.0
        q = {first[k]: (round(rootn.q(k), 4), rootn.Nq.get(k, 0)) for k in cands if rootn.Nq.get(k, 0)}
        return {"best": int(first[best]), "greedy": greedy_idx, "cands": [first[k] for k in cands],
                "diff": {str(first[k]): v for k, v in diff.items()}, "q": {str(k): v for k, v in q.items()},
                "sims": st["sims"], "evals": st["evals"], "term": st["term"], "err": st["err"], "nodes": st["nodes"],
                "roll": st["roll"], "rterm": st["rterm"],
                "depth": round(st["depth"] / max(1, st["sims"]), 2), "maxd": st["maxd"], "ms": round((time.perf_counter() - t0) * 1000.0, 1)}

    def _simulate(self, sims: list[dict], rootn: Node, first: dict, base: dict, hist: list[int], root, ret: dict) -> None:
        """一批模拟（每个推演进程同时跑一个）：restore 根 + reseed + 走根动作，之后每一帧按签名找到节点，在节点上按树选动作，
        直到第一次到达的节点（叶子）或战斗结束；叶子合成一批过网络，回传。锁步：每一轮把空闲进程派满，按进程顺序把忙着的都读一遍。"""
        a, P, X = self.a, self.P, self.X
        stage = int(root.header[23])
        left0 = X.enemy_left(root)
        queue = list(sims)
        busy: dict[int, dict] = {}
        while queue or busy:
            for j in range(len(self.workers)):
                if j in busy or not queue:
                    continue
                w = queue.pop(0)
                k0 = w["key0"]
                rootn.N[k0] = rootn.N.get(k0, 0) + 1
                w.update(cand=first[k0], edge=(rootn, k0), path=[(rootn, k0)])
                self._dispatch(j, w, base, hist, left0)
                busy[j] = w
            if not busy:
                break
            leaves, rolls = [], []
            for j in list(busy):
                w = busy[j]
                got = self._take(j, w)
                while got == "skip":
                    got = self._take(j, w)
                if got == "dead":
                    self._abort(w)
                    del busy[j]
                    continue
                o = w["frame"]
                if X.combat_over(o):
                    self._backup(w, self._leaf_value(w, X.terminal(X.outcome(o, w["left"]))), ret)
                    self.tstat["rterm" if "roll" in w else "term"] += 1
                    del busy[j]
                    continue
                if w["steps"] > P.max_steps:
                    self._backup(w, self._leaf_value(w, X.terminal({"win": False, "stage": stage})), ret)
                    del busy[j]
                    continue
                w["left"] = X.enemy_left(o)
                if "roll" in w:          # 叶子之后的贪心推演
                    rolls.append(j)
                    continue
                keys, fk = frame_keys(o, X)
                parent, k = w["edge"]
                kids = parent.out.setdefault(k, {})
                sig = signature(o, X, keys)
                node = kids.get(sig)
                if node is None:
                    node = kids[sig] = Node()
                    self.tstat["nodes"] += 1
                    if len(kids) > 1:
                        self.tstat["chance"] += 1
                w["node"], w["keys"] = node, keys
                if not node.expanded or len(w["path"]) >= a.max_depth:
                    leaves.append(j)
                    continue
                k = self._select(node, list(fk))
                node.N[k] = node.N.get(k, 0) + 1
                w["edge"] = (node, k)
                w["path"].append((node, k))
                req = {"op": "step", "action": int(fk[k])}
                if a.auto_single:
                    req["auto_single"] = True
                self.workers[j].send(req)
            if leaves or rolls:
                js = leaves + rolls
                frames = [busy[j]["frame"] for j in js]
                logits, values = self.ev.forward(frames, [MODE_COMBAT] * len(frames))
                self.tstat["evals"] += len(frames)
                if self.vev:
                    need = [i for i, j in enumerate(js) if "roll" not in busy[j] or busy[j]["steps"] - busy[j]["roll"]["start"] >= a.roll]
                    values = self._values(frames, values, need)
                for i, j in enumerate(js):
                    w = busy[j]
                    o = w["frame"]
                    v = float(values[i])
                    if "roll" not in w:      # 树的叶子
                        node = w["node"]
                        fresh = not node.expanded
                        if fresh:
                            n = int(o.num_actions)
                            pr = self.torch.softmax(logits[i, :n].double(), -1).numpy()
                            node.prior = merged_prior(pr, w["keys"])
                            node.v0 = v
                            node.expanded = True
                        if a.roll <= 0:
                            del busy[j]
                            self._backup(w, v, ret)
                            continue
                        w["roll"] = {"start": w["steps"], "v0": v, "node": node if fresh else None}
                    elif w["steps"] - w["roll"]["start"] >= a.roll:
                        del busy[j]
                        self._backup(w, self._leaf_value(w, v), ret)
                        continue
                    # 推演：策略头最大项（同 sim_pimc.ModelEvaluator.rollout）；自动走单候选不越过第 roll 步那一帧（同 boss_pimc._auto_cap）
                    act = int(logits[i, : int(o.num_actions)].argmax())
                    req = {"op": "step", "action": act}
                    if a.auto_single:
                        req["auto_single"] = max(0, a.roll - (w["steps"] - w["roll"]["start"]) - 1)
                    self.workers[j].send(req)
                    self.tstat["roll"] += 1

    def _world_reqs(self, j: int, w: dict, base: dict, hist: list[int]) -> list[dict]:
        """--oracle：世界不 reseed，恢复出的就是实机真实的未来（抽牌、敌人意图的随机数都与实机相同），只作上限诊断，不是可用的搜索。"""
        reqs = super()._world_reqs(j, w, base, hist)
        if self.a.oracle:
            reqs[-1].pop("reseed", None)
        return reqs

    def _leaf_value(self, w: dict, v: float) -> float:
        """没有推演时就是 v；推演的模拟回传 (1 − m)·叶子价值头 + m·推演得分，并把这次展开的节点的 v0 改成它。"""
        r = w.get("roll")
        if r is None:
            return v
        m = self.a.roll_mix
        out = (1.0 - m) * r["v0"] + m * v
        if r["node"] is not None:
            r["node"].v0 = out
        return out

    def _backup(self, w: dict, v: float, ret: dict) -> None:
        for node, k in w["path"]:
            node.Nq[k] = node.Nq.get(k, 0) + 1
            node.W[k] = node.W.get(k, 0.0) + v
        ret[w["key0"]][w["seed"]] = v
        d = len(w["path"])
        self.tstat["sims"] += 1
        self.tstat["depth"] += d
        self.tstat["maxd"] = max(self.tstat["maxd"], d)

    def _abort(self, w: dict) -> None:
        """推演进程出错：这次模拟作废，下行时加的访问次数撤回。"""
        for node, k in w.get("path", []):
            node.N[k] -= 1
        self.tstat["err"] += 1


def run(a) -> None:
    k, n = map(int, a.shard.split("/"))
    a.k_shard = k
    os.makedirs(a.out, exist_ok=True)
    items = json.loads(Path(a.items).read_text(encoding="utf-8"))["items"]
    if a.item_mod:
        m, r = a.item_mod.split(":")
        m = int(m)
        items = [it for it in items if (it["id"] % m != int(r[1:]) if r.startswith("!") else it["id"] % m == int(r))]
    if a.limit:
        items = items[: a.limit]
    todo = [it for i, it in enumerate(items) if i % n == k]
    done = set()
    for f in Path(a.out).glob("res_s*.jsonl"):
        for l in open(f, encoding="utf-8"):
            if l.strip():
                try:
                    done.add(json.loads(l)["id"])
                except ValueError:
                    pass
    todo = [it for it in todo if it["id"] not in done]
    print(f"分片 {k}/{n}：{len(todo)} 题待打（树搜索，{a.sims} 次模拟，decide {a.decide}，λ {a.lam}，叶子推演 {a.roll} 步 m {a.roll_mix}）", flush=True)
    if not todo:
        return
    R = TreeRunner(a)
    fh = open(Path(a.out) / f"res_s{k}.jsonl", "a", encoding="utf-8")
    lab = open(Path(a.out) / f"labels_s{k}.jsonl", "a", encoding="utf-8") if a.labels else None
    t0 = time.time()
    try:
        for i, it in enumerate(todo):
            R.acc = {}
            res = R.play(it, lab, None)
            res["rep"] = 0
            acc = R.acc
            res["tree"] = {kk: acc.get(kk, 0) for kk in ("sims", "evals", "term", "err", "nodes", "chance", "maxd", "roll", "rterm")}
            res["tree"]["depth"] = round(acc.get("depth", 0) / max(1, acc.get("sims", 0)), 2)
            res["tree"]["ms"] = round(acc.get("ms", 0.0))
            fh.write(json.dumps(res) + "\n"); fh.flush()
            if lab:
                lab.flush()
            if i % 10 == 0:
                print(f"{time.strftime('%H:%M:%S')} 分片 {k}：{i + 1}/{len(todo)}，{time.time() - t0:.0f} 秒", flush=True)
    finally:
        R.close()
    print("完", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--items", required=True); r.add_argument("--ckpt", required=True); r.add_argument("--out", required=True)
    r.add_argument("--dll", default=str(boss_pimc.REPO / "engine/Sts2Headless/bin/Release/net9.0/Sts2Headless.dll"))
    r.add_argument("--device", default="cuda"); r.add_argument("--ema", action="store_true"); r.add_argument("--infer", default="")
    r.add_argument("--shard", default="0/1"); r.add_argument("--procs", type=int, default=8)
    r.add_argument("--item-mod", default=""); r.add_argument("--limit", type=int, default=0); r.add_argument("--policy-seed", type=int, default=0)
    r.add_argument("--tau", type=float, default=0.9); r.add_argument("--k", type=int, default=4); r.add_argument("--z", type=float, default=2.0)
    r.add_argument("--cap", type=int, default=3000)
    r.add_argument("--sims", type=int, default=128, help="每个犹豫决策的模拟次数（0 = 不搜，等于贪心）")
    r.add_argument("--decide", choices=["kl", "z", "gumbel"], default="kl",
                   help="根上最后怎么选：kl = log 先验 + Q/λ；z = 与贪心逐世界配对过 z 个标准误（老师的判据，树里不适用，见文件头）；gumbel = log 先验 + σ(补全 Q)")
    r.add_argument("--lam", type=float, default=0.03, help="kl 的 λ（价值尺度同 sim_pimc.terminal）"); r.add_argument("--min-visits", type=int, default=4)
    r.add_argument("--root-keep", choices=["mean", "gumbel"], default="mean", help="逐轮减半时留哪些候选")
    r.add_argument("--qnorm", choices=["minmax", "fixed"], default="minmax"); r.add_argument("--value-range", type=float, default=2.5)
    r.add_argument("--c-visit", type=float, default=50.0); r.add_argument("--c-scale", type=float, default=0.1)
    r.add_argument("--max-depth", type=int, default=200, help="一次模拟最多走几条树边（超过就在那一帧估值，不再展开）")
    r.add_argument("--roll", type=int, default=0, help="叶子之后贪心推演几步再用价值头估值（0 = 叶子直接估值，见文件头第四节）")
    r.add_argument("--roll-mix", type=float, default=1.0, help="回传值里推演得分的权重 m（其余 1 − m 给叶子那一帧的价值头）")
    r.add_argument("--value-replace", action="store_true", help="叶子估值只用 --value-ckpts 的网络（不与主网络平均）：单独的战斗价值网络")
    r.add_argument("--keep-top", action="store_true", help="世界 reseed 时不洗已知的抽牌堆顶（被效果放到顶部、玩家知道位置的牌；同 boss_pimc --keep-top）")
    r.add_argument("--value-ckpts", default="", help="另几份检查点（逗号分隔）：叶子估值取主网络与它们价值头的平均（先验、出手仍用主网络）")
    r.add_argument("--oracle", action="store_true", help="上限诊断：世界不 reseed，搜索看得到实机真实的未来随机数（违反公开信息，不能当搜索用）")
    r.add_argument("--labels", action="store_true", help="每个搜过的根记一行（含根帧原始字节，很大）")
    a = ap.parse_args()
    # boss_pimc.Runner 要的其余参数：局面复制跟根、单候选由引擎走过；不贪心推演（leaf 0 只影响 _auto_cap：不设上限）
    a.no_search = False; a.warm = True; a.worlds = "clone"; a.async_ = False; a.async_wait = 0.0; a.auto_single = True
    a.residual = ""; a.residual_scale = 1.0; a.temp = 0.0; a.save_frames = False; a.sims0 = 1; a.rounds = 1; a.leaf = 0
    {"run": run}[a.cmd](a)


if __name__ == "__main__":
    main()
