"""整局评测：局外照常贪心，指定房间（默认 Boss）的战斗里用树搜索（2026-11-17）。

用户：“虽然第三幕BOSS是大头，但是我目前仍在考虑给前两幕BOSS甚至精英加搜索……可以考虑增加一个就是在三幕都采用搜索的一个愚者胜率测试，做千种测试。”

和第三幕双 Boss 题（tree_search.py，战斗模式开局、战斗价值头、按 sim_pimc.terminal 记分）的区别：这里是真实的整局，网络全程用整局模式
（与 ability_test 的常规评测同一个出手），树里的先验与估值也用整局模式的策略头与价值头。理由：
  - 整局模式的价值头估的是这一局以后的总奖励，Boss、精英打完时的血量对后面的影响自动算进去；战斗模式的终局分只看这一场的输赢。
  - 第三幕两个 Boss 在整局里的帧（头部 23 恒为 0、头部 17 两场都是第二个 Boss）与战斗模式连打题的帧不同，战斗价值头分不出是第几场。
  - 模拟次数为 0 时与整局贪心逐步相同（搜索返回贪心项时取贪心的下标），配对干净。

回传值 = 模拟路上拿到的奖励 + 末帧的整局价值（同训练的奖励定义，gamma = 1，读检查点里的配置）：
  - 爬层 floor_reward × 层数；第三幕第一个 Boss 打完 act3_boss_reward（打完那一步发，模拟穿过奖励、地图两个单候选帧直接进第二个 Boss）；
    通关 victory_reward + act3_boss2_reward；进入下一幕 act1/act2_boss_reward（一般停在奖励帧、还没进下一幕，价值头里已含）。
  - 死亡记 0（整局模式没有输的惩罚，以后也没有奖励）；卡住记 0。
  - 战斗打完、停在有多个候选的局外帧（奖励等）：这一帧的整局价值，不展开节点。战斗中的帧照 tree_search 的树。
  - L1000 的配置里血量塑形、药水扣分、战斗掉血扣分都是 0，所以战斗里没有别的奖励项。换检查点时核对（启动时打印）。
λ：整局价值的幅度（第三幕 Boss 开打约 2–5.5，死亡 0）约为战斗终局分（1.5 对约 −0.75）的两倍，默认 --lam 0.06（题目上是 0.03）。

世界跟根：boss_pimc 的 --worlds clone（reset 整局 + 重放这一局已走的全部动作 + 快照，之后 restore + 增量重放）；每场要搜的战斗第一次搜索时
推演进程要从头重放到这里（几百步），之后同一场里只重放新走的几步。

  .venv/bin/python -I tools/future/run_search.py --seeds 种子池.json --runs 1000 --ckpt L1000.pt --out 目录 --shard 0/16 --sims 32 [--rooms Boss,Elite]
  --sims 0 为整局贪心（应与 ability_test 同种子逐局相同）
  --replays 目录：每局另存一份整局回放（evaluator.save_replay 的格式，每步 replay.describe，概率为根上的策略概率），给胜率小模型做带搜索的数据（2026-11-21）
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import boss_pimc  # noqa: E402
from sts2rl.actor import cand_name  # noqa: E402
from sts2rl.replay import describe  # noqa: E402
import tree_search as T  # noqa: E402

MODE_RUN = 1
MODE_COMBAT = 0
EXTRA = {"loop_cut": True, "stall_turns": 20}       # 同常规评测与 /root/future-c/c_games.sh


def save_replay(rec_dir: str, version, seed: str, result: dict, steps: list[str], arm: str) -> None:
    """同 evaluator.save_replay（第一行概要、之后每行一步，先写临时文件再改名），概要多记 arm 与 source。"""
    import gzip
    os.makedirs(rec_dir, exist_ok=True)
    head = {"version": version, "seed": seed, "time": time.time(), "win": result["win"], "floor": result["floor"], "act": result["act"],
            "combats": result["combats"], "steps": len(steps), "arm": arm, "source": "run_search --replays"}
    name = f"v{version}_{seed}"
    tmp = os.path.join(rec_dir, f".{name}.tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        f.write(json.dumps(head, ensure_ascii=False) + "\n")
        for x in steps:
            f.write(x + "\n")
    os.replace(tmp, os.path.join(rec_dir, f"{name}.jsonl.gz"))


def kl_target(keys: list, p: np.ndarray, res: dict, lam: float, min_visits: int):
    """搜索改进后的分布（与最终选法同一个算子，同 tree_distill.load_roots）：访问数 ≥ min_visits 的候选按语义键分组，组 k 的质量 ∝ P_k · exp(Q_k / λ)，
    这些组的总质量保持不变、组内按原概率分，其余候选保持原概率。少于两个合格候选、或贪心项不在其中时返回 None（这样的根不当标签）。"""
    q = {int(c): (float(v[0]), int(v[1])) for c, v in (res.get("q") or {}).items()}
    C = [c for c in q if q[c][1] >= min_visits]
    g = int(res["greedy"])
    if len(C) < 2 or keys[g] not in {keys[c] for c in C}:
        return None
    groups: dict = {}
    for i, k in enumerate(keys):
        groups.setdefault(k, []).append(i)
    gk = [keys[c] for c in C]
    Pk = np.array([p[groups[k]].sum() for k in gk])
    s = np.log(np.maximum(Pk, 1e-12)) + np.array([q[c][0] for c in C]) / lam
    w = np.exp(s - s.max()); w /= w.sum()
    tgt = np.array(p, np.float64).copy()
    M = Pk.sum()
    for k, pk, wk in zip(gk, Pk, w):
        idx = groups[k]
        tgt[idx] = M * wk * (p[idx] / max(pk, 1e-12))
    return tgt / tgt.sum()


class RunTreeRunner(T.TreeRunner):
    def __init__(self, a):
        super().__init__(a)
        c = self.torch.load(a.ckpt, map_location="cpu", weights_only=False)["config"]
        self.rw = {k: float(c.get(k) or 0.0) for k in ("floor_reward", "act1_boss_reward", "act2_boss_reward", "act3_boss_reward",
                                                      "victory_reward", "act3_boss2_reward")}
        zero = {k: c.get(k) for k in ("run_hp_shaping", "run_potion_coef", "combat_dmg_coef", "wp_shaping", "bv_shaping", "relic_coef")}
        # 里程碑撤掉（决策记录 271，故障机器人第 164,606 版起）：训练时层数、各幕 Boss、第三幕第一个 Boss 的奖励按 ms_scale 发，撤掉的部分在通关时一次补回
        # （按这一局原本能拿到的里程碑合计），价值头学的就是这个口径。模拟里照同样记（2026-11-22，论文补充实验：故障机器人在 v0.113.1 上树搜索）：
        # 路上的里程碑 × ms_scale，通关时补 (1 − ms_scale) × (根之前这一局已得的里程碑 + 模拟里得的)。没有 milestone_from 的检查点 ms_scale = 1，与原来逐位相同
        ver = int(self.version)
        mf, mr = c.get("milestone_from"), int(c.get("milestone_ramp") or 500)
        self.ms_scale = 1.0 if mf is None or ver < mf else (0.0 if mr <= 0 else 1.0 - min(1.0, (ver - mf) / mr))
        self.ms_root = 0.0
        print(f"整局奖励（检查点配置）：{self.rw}；应为 0 的项：{zero}；milestone_from {mf}（ramp {mr}，第 {ver} 版的里程碑系数 {self.ms_scale}）", flush=True)
        self.ctx = {}
        self.lab = None     # --labels：run() 打开的标签文件
        self.map_bias = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in (a.map_bias or "").split(",") if "=" in kv}
        from sts2rl.catalog import Catalog
        self.vocab = Catalog().vocab

    # ---------------------------------------------------------------- 模拟（tree_search._simulate 的整局版）

    def _path_reward(self, w: dict, o) -> None:
        """这一帧相对上一帧拿到的奖励，累加进 w["acc"]（同 actor 的记法）。"""
        rw, ctx, ms = self.rw, self.ctx, self.ms_scale
        if o.floor > w["floor"]:
            m = rw["floor_reward"] * (o.floor - w["floor"]); w["acc"] += ms * m; w["ms"] = w.get("ms", 0.0) + m
        for act in range(w["act"] + 1, o.act + 1):
            m = {1: rw["act1_boss_reward"], 2: rw["act2_boss_reward"]}.get(act, 0.0); w["acc"] += ms * m; w["ms"] = w.get("ms", 0.0) + m
        # 第三幕第一个 Boss：这场打完（离开战斗或已进到下一层）且没死，发 act3_boss_reward（只发一次）
        if ctx.get("boss3_first") and not w.get("b3") and not o.done and (not o.in_combat or o.floor > ctx["floor0"]):
            m = rw["act3_boss_reward"]; w["acc"] += ms * m; w["ms"] = w.get("ms", 0.0) + m; w["b3"] = True
        if o.done and o.victory and not o.stuck:
            w["acc"] += rw["victory_reward"] + rw["act3_boss2_reward"]
            if ms < 1.0:        # 撤掉的里程碑在通关时补回（同 actor._finish 的 milestone_back）
                w["acc"] += (1.0 - ms) * (self.ms_root + w.get("ms", 0.0))
        w["floor"], w["act"] = max(w["floor"], int(o.floor)), max(w["act"], int(o.act))

    @staticmethod
    def cframe(o, act3: bool):
        """--net-mode combat：整局的帧给战斗网络看之前，第三幕 Boss 的头部 23（这一幕第几场，战斗模式连打题第二场为 1）按头部 24 补上
        （整局模式下头部 23 恒为 0，头部 24、25 已标出第二场）。其余字段不动。"""
        if not act3 or int(o.header[23]) == int(o.header[24]):
            return o
        import copy
        o2 = copy.copy(o)
        o2.header = o.header.copy()
        o2.header[23] = o.header[24]
        return o2

    def _cm_terminal(self, w, o) -> float:
        """战斗模式的终局分（同 sim_pimc.terminal 与连打练习）：打赢 1.5；输了 −1 + 0.25 ×（这一串已打赢的场数 + 这场打掉的敌人生命比例）。"""
        if o.victory and not o.stuck:
            return 1.5
        return -1.0 + 0.25 * (w.get("stage", 0) + 1.0 - w.get("left", 1.0))

    def _simulate(self, sims, rootn, first, base, hist, root, ret) -> None:
        if self.ctx.get("cm"):
            return self._simulate_cm(sims, rootn, first, base, hist, root, ret)
        a, P, X = self.a, self.P, self.X
        queue = list(sims)
        busy: dict[int, dict] = {}
        while queue or busy:
            for j in range(len(self.workers)):
                if j in busy or not queue:
                    continue
                w = queue.pop(0)
                k0 = w["key0"]
                rootn.N[k0] = rootn.N.get(k0, 0) + 1
                w.update(cand=first[k0], edge=(rootn, k0), path=[(rootn, k0)], acc=0.0, floor=int(root.floor), act=int(root.act), ms=0.0)
                self._dispatch(j, w, base, hist, 1.0)
                busy[j] = w
            if not busy:
                break
            leaves, posts = [], []
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
                self._path_reward(w, o)
                if o.done or o.stuck or o.num_actions == 0:
                    self._backup(w, w["acc"], ret)          # 通关的奖励已在 acc 里；死亡、卡住以后没有奖励
                    self.tstat["term"] += 1
                    del busy[j]
                    continue
                if not o.in_combat or w["steps"] > P.max_steps or self.ctx.get("macro"):
                    posts.append(j)                         # 战斗打完停在局外帧（或步数超限），或局外决策的一步前瞻：这一帧的整局价值，不展开
                    continue
                keys, fk = T.frame_keys(o, X)
                parent, k = w["edge"]
                kids = parent.out.setdefault(k, {})
                sig = T.signature(o, X, keys)
                node = kids.get(sig)
                if node is None:
                    node = kids[sig] = T.Node()
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
                self.workers[j].send({"op": "step", "action": int(fk[k]), "auto_single": True})
            js = leaves + posts
            if js:
                frames = [busy[j]["frame"] for j in js]
                logits, values = self.ev.forward(frames, [MODE_RUN] * len(frames))
                self.tstat["evals"] += len(frames)
                for i, j in enumerate(js):
                    w = busy.pop(j)
                    v = float(values[i])
                    if i < len(leaves):
                        node, o = w["node"], w["frame"]
                        if not node.expanded:
                            n = int(o.num_actions)
                            pr = self.torch.softmax(logits[i, :n].double(), -1).numpy()
                            node.prior = T.merged_prior(pr, w["keys"])
                            node.v0 = w["acc"] + v
                            node.expanded = True
                    else:
                        self.tstat["roll"] += 1     # 借 roll 计数：停在局外帧估值的模拟数
                    self._backup(w, w["acc"] + v, ret)

    def _simulate_cm(self, sims, rootn, first, base, hist, root, ret) -> None:
        """--net-mode combat 的模拟：先验与叶子估值用战斗模式（帧经 cframe 补头部 23），记分同题目（打赢 1.5、输了按进度）。
        第三幕第一个 Boss 打完，引擎的 auto_single 会走过奖励、地图两个单候选帧直接进第二个 Boss，模拟接着在第二个 Boss 里走（同连打题）。"""
        a, P, X = self.a, self.P, self.X
        act3 = bool(self.ctx.get("act3"))
        stage0 = int(root.header[24]) if act3 else 0
        queue = list(sims)
        busy: dict[int, dict] = {}
        while queue or busy:
            for j in range(len(self.workers)):
                if j in busy or not queue:
                    continue
                w = queue.pop(0)
                k0 = w["key0"]
                rootn.N[k0] = rootn.N.get(k0, 0) + 1
                w.update(cand=first[k0], edge=(rootn, k0), path=[(rootn, k0)], stage=stage0)
                self._dispatch(j, w, base, hist, X.enemy_left(root))
                busy[j] = w
            if not busy:
                break
            leaves, posts = [], []
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
                if o.done or o.stuck or o.num_actions == 0:
                    self._backup(w, self._cm_terminal(w, o), ret)
                    self.tstat["term"] += 1
                    del busy[j]
                    continue
                if not o.in_combat:
                    if act3 and w["stage"] == 0:
                        posts.append(j)                     # 第三幕两个 Boss 之间停在多候选的局外帧（少见）：照样给战斗网络估值，计数
                    else:
                        self._backup(w, 1.5, ret)           # 第一、二幕 Boss 或精英打赢（第三幕第二个打赢会直接 done）
                        self.tstat["term"] += 1
                        del busy[j]
                    continue
                if act3:
                    w["stage"] = max(w["stage"], int(o.header[24]))
                w["left"] = X.enemy_left(o)
                if w["steps"] > P.max_steps:
                    self._backup(w, -1.0 + 0.25 * (w["stage"] + 1.0 - w["left"]), ret)
                    del busy[j]
                    continue
                keys, fk = T.frame_keys(o, X)
                parent, k = w["edge"]
                kids = parent.out.setdefault(k, {})
                sig = T.signature(o, X, keys)
                node = kids.get(sig)
                if node is None:
                    node = kids[sig] = T.Node()
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
                self.workers[j].send({"op": "step", "action": int(fk[k]), "auto_single": True})
            js = leaves + posts
            if js:
                frames = [self.cframe(busy[j]["frame"], act3) for j in js]
                logits, values = self.ev.forward(frames, [MODE_COMBAT] * len(frames))
                if self.ctx.get("hybrid"):          # 混合：先验取整局模式的策略头（原帧），估值仍是战斗模式
                    logits, _ = self.ev.forward([busy[j]["frame"] for j in js], [MODE_RUN] * len(js))
                    self.tstat["evals"] += len(frames)
                self.tstat["evals"] += len(frames)
                for i, j in enumerate(js):
                    w = busy.pop(j)
                    v = float(values[i])
                    if i < len(leaves):
                        node, o = w["node"], w["frame"]
                        if not node.expanded:
                            n = int(o.num_actions)
                            pr = self.torch.softmax(logits[i, :n].double(), -1).numpy()
                            node.prior = T.merged_prior(pr, w["keys"])
                            node.v0 = v
                            node.expanded = True
                    else:
                        self.tstat["roll"] += 1     # 借 roll 计数：第三幕两个 Boss 之间停在局外帧的模拟数
                    self._backup(w, v, ret)

    # ---------------------------------------------------------------- 实机整局

    def play_run(self, seed: str) -> dict:
        a, X, torch = self.a, self.X, self.torch
        base = {"op": "reset", "mode": "run", "character": a.character, "ascension": 10, "seed": seed, "snapshots": True,
                "route_summary": True, **EXTRA, **a.req_extra_d}
        self.cur_key = ("run", seed)
        self.real.send({**base, "labels": False})
        o, raw, err = self.recv(self.real)
        hist: list[int] = []
        t0 = time.perf_counter()
        room, fight = None, None
        fight_ctx = {}
        macro = {"searches": 0, "changed": 0}
        mbias = {"n": 0, "changed": 0}
        nsteps = 0
        fights = []
        act3_bosses = 0
        tsearch = 0.0
        prev_ic = False
        rsteps: list[str] | None = [] if a.replays else None
        while err is None and not (o.done or o.stuck or o.num_actions == 0):
            if a.all_rooms and o.in_combat and not prev_ic and not (o.extra and "combat" in o.extra):
                # 没带战斗快照就开打的战斗（不是 CombatRoom）：只在 --rooms all 时另记一场，免得统计算到上一场头上（其余设置的行为不变）
                room = "Other"
                fight = {"room": room, "act": int(o.act), "floor": int(o.floor), "enc": None, "hp0": int(o.hp), "searches": 0, "changed": 0}
                fights.append(fight)
                fight_ctx = {"boss3_first": False, "floor0": int(o.floor)}
            prev_ic = bool(o.in_combat)
            if o.extra and "combat" in o.extra:
                c = o.extra["combat"]
                room = c.get("room")
                if room == "Boss" and int(c.get("act", -1)) == 2:
                    act3_bosses += 1
                fight = {"room": room, "act": int(c.get("act", -1)), "floor": int(c.get("floor", o.floor)), "enc": c.get("encounter"),
                         "hp0": int(o.hp), "searches": 0, "changed": 0}
                fights.append(fight)
                fight_ctx = {"boss3_first": room == "Boss" and int(c.get("act", -1)) == 2 and act3_bosses == 1, "floor0": int(o.floor)}
            n = int(o.num_actions)
            sel = a.all_rooms or room in a.room_set
            if a.act_set is not None:       # --acts（2026-11-21）：只搜这几幕的这些房间；不给时与原来完全相同
                sel = sel and fight is not None and int(fight["act"]) in a.act_set
            cm = a.net_mode == "combat" and o.in_combat and sel     # 这场战斗整场交给战斗模式（搜或不搜）
            hy = a.net_mode == "hybrid" and o.in_combat and sel     # 混合：出手与先验用整局模式，搜索的估值用战斗模式
            act3 = room == "Boss" and fight is not None and fight["act"] == 2
            root = self.cframe(o, act3) if cm else o
            logits, _ = self.ev.forward([root], [MODE_COMBAT if cm else MODE_RUN])
            p = torch.softmax(logits[0, :n].double(), -1).numpy()
            greedy = int(np.argmax(p))
            act = greedy
            if self.map_bias and n > 1 and o.decision.name == "MAP":
                # 选路偏置（局外适应搜索的探针）：给指定类型的地图节点在对数概率上加分后取最大；不给 --map-bias 时不走这里
                lg = np.log(np.maximum(p, 1e-12))
                for c in range(n):
                    nm = cand_name(o, c, self.vocab)
                    t = nm.split(":", 1)[1] if nm.startswith("MAPNODE:") else None
                    if t in self.map_bias:
                        lg[c] += self.map_bias[t]
                act = int(np.argmax(lg))
                mbias["n"] += 1; mbias["changed"] += int(act != greedy)
            if cm and fight is not None:
                fight["cm"] = 1
            in_fight = o.in_combat and sel and a.sims > 0
            is_macro = a.macro_sims > 0 and not o.in_combat
            if (in_fight or is_macro) and n > 1 and self.gated(p):
                ts = time.perf_counter()
                dec_seed = zlib.crc32(f"{seed}:{nsteps}:{a.policy_seed}".encode()) & 0x3FFFFFFF
                # 根之前这一局按原系数已得的里程碑（同 actor 的 ms_full：层数从 0 起算、进第二三幕各补上一幕 Boss、第三幕第一个 Boss 打赢后补 act3_boss_reward），
                # 只在 ms_scale < 1 时用到
                rw_ = self.rw
                self.ms_root = (rw_["floor_reward"] * int(o.floor) + (rw_["act1_boss_reward"] if int(o.act) >= 1 else 0.0)
                                + (rw_["act2_boss_reward"] if int(o.act) >= 2 else 0.0)
                                + (rw_["act3_boss_reward"] if act3_bosses >= 2 or (act3_bosses == 1 and room != "Boss") else 0.0))
                self.ctx = ({"cm": True, "act3": act3, "hybrid": hy} if (cm or hy) else fight_ctx) if in_fight else {"macro": True, "floor0": int(o.floor)}
                sims0, self.a.sims = self.a.sims, (self.a.sims if in_fight else a.macro_sims)
                res = self.search(base, hist, root, p, dec_seed)
                self.a.sims = sims0
                tsearch += time.perf_counter() - ts
                keys = [k[:-1] for k in X.cand_keys(o)]
                if self.lab is not None and in_fight and not (cm or hy):
                    tgt = kl_target(keys, p, res, a.lam, a.min_visits)
                    if tgt is not None:
                        self.lab.write(json.dumps({"v": self.version, "seed": seed, "step": nsteps, "room": room, "act": fight["act"] if fight else None,
                                                   "frame": base64.b64encode(raw).decode(), "tgt": [round(float(x), 6) for x in tgt],
                                                   "greedy": greedy, "best": int(res["best"])}) + "\n")
                st = fight if in_fight else macro
                if keys[int(res["best"])] != keys[greedy]:
                    act = int(res["best"])
                    st["changed"] += 1
                st["searches"] += 1
            if rsteps is not None:
                rsteps.append(json.dumps(describe(o, act, p), separators=(",", ":")))
            hist.append(act)
            self.real.send({"op": "step", "action": act})
            o, raw, err = self.recv(self.real)
            nsteps += 1
            if o is not None and fight is not None and not o.in_combat and "hp1" not in fight:
                fight["hp1"] = int(o.hp)
            if nsteps > a.cap:
                err = "步数超上限"
        win = err is None and bool(o.victory) and not o.stuck
        out = {"seed": seed, "win": int(win), "stuck": int(err is not None or bool(o is not None and o.stuck)), "err": err,
               "floor": int(o.floor) if o is not None else None, "act": int(o.act) if o is not None else None, "steps": nsteps,
               "searches": sum(f["searches"] for f in fights), "changed": sum(f["changed"] for f in fights), "macro": macro,
               "s": round(time.perf_counter() - t0, 1), "s_search": round(tsearch, 1),
               "death": None if win or err is not None else (f"{fights[-1]['room']}:{fights[-1]['act']}" if fights and fights[-1]["floor"] == int(o.floor) else "非战斗"),
               "fights": [f for f in fights if f["room"] in ("Boss", "Elite")], **({"map_bias": mbias} if self.map_bias else {})}
        by_room = {}
        for f in fights:          # 每种房间的战斗：场数、搜索与改选次数、掉血合计（打完的场）
            b = by_room.setdefault(f["room"] or "None", {"n": 0, "searches": 0, "changed": 0, "hp_lost": 0, "n_done": 0})
            b["n"] += 1; b["searches"] += f["searches"]; b["changed"] += f["changed"]
            if "hp1" in f:
                b["hp_lost"] += f["hp0"] - f["hp1"]; b["n_done"] += 1
        out["by_room"] = by_room
        if rsteps is not None and err is None:
            save_replay(a.replays, self.version, seed, {"win": win, "floor": out["floor"], "act": out["act"], "combats": []}, rsteps,
                        arm=f"search:{a.rooms}:{a.sims}")
        return out


def run(a) -> None:
    k, n = map(int, a.shard.split("/"))
    a.k_shard = k
    a.room_set = set(a.rooms.split(","))
    a.act_set = {int(x) for x in a.acts.split(",")} if a.acts else None
    a.req_extra_d = json.loads(a.req_extra) if a.req_extra else {}     # 实机与推演的 reset 都带（base 传给 _world_reqs）
    a.all_rooms = "all" in a.room_set           # 所有战斗都搜（普通怪、精英、Boss，以及没有战斗快照的战斗）
    os.makedirs(a.out, exist_ok=True)
    seeds = json.loads(Path(a.seeds).read_text(encoding="utf-8"))["seeds"][: a.runs]
    todo = [s for i, s in enumerate(seeds) if i % n == k]
    res_path = Path(a.out) / f"runs_s{k}.jsonl"
    done = set()
    if res_path.exists():
        for l in open(res_path, encoding="utf-8"):
            if l.strip():
                done.add(json.loads(l)["seed"])
    todo = [s for s in todo if s not in done]
    print(f"分片 {k}/{n}：{len(todo)} 局待打（搜索房间 {a.rooms}，{a.sims} 次模拟，λ {a.lam}{'，reset 另加 ' + json.dumps(a.req_extra_d) if a.req_extra_d else ''}）", flush=True)
    if not todo:
        return
    R = RunTreeRunner(a)
    R.lab = open(Path(a.out) / f"labels_s{k}.jsonl", "a", encoding="utf-8") if a.run_labels else None
    fh = open(res_path, "a", encoding="utf-8")
    t0 = time.time()
    try:
        for i, s in enumerate(todo):
            R.acc = {}
            res = R.play_run(s)
            res["tree"] = {kk: R.acc.get(kk, 0) for kk in ("sims", "evals", "term", "err", "nodes", "chance", "roll")}
            fh.write(json.dumps(res, ensure_ascii=False) + "\n"); fh.flush()
            if R.lab is not None:
                R.lab.flush()
            if i % 5 == 0:
                print(f"{time.strftime('%H:%M:%S')} 分片 {k}：{i + 1}/{len(todo)}，{time.time() - t0:.0f} 秒", flush=True)
    finally:
        R.close()
        if R.lab is not None:
            R.lab.close()
    print("完", flush=True)


def parse(argv: list[str] | None = None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", required=True); ap.add_argument("--runs", type=int, default=1000); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--shard", default="0/1"); ap.add_argument("--character", default="FOOL")
    ap.add_argument("--rooms", default="Boss", help="哪些房间的战斗用搜索（逗号分隔：Boss、Elite、Monster）；all 为所有战斗")
    ap.add_argument("--acts", default="", help="只在这几幕搜（逗号分隔，0 起：2 为第三幕），与 --rooms 同时成立；如 --rooms Boss --acts 2 为只搜第三幕两个 Boss，"
                    "同 v0.111.0 三个角色 PIMC 的范围（2026-11-21，论文补充实验 4）。默认空：各幕都搜，与原来完全相同")
    ap.add_argument("--net-mode", choices=["run", "combat", "hybrid"], default="run",
                    help="--rooms 里的战斗用哪个模式的网络：run 为整局模式（默认，回传整局价值）；combat 为战斗模式（整场战斗贪心与搜索都用它，记分同题目，λ 宜用 0.03）；"
                         "hybrid 为出手与搜索先验用整局模式、叶子估值与记分用战斗模式")
    ap.add_argument("--macro-sims", type=int, default=0, help="局外决策（选牌、选路、商店、事件等）的一步前瞻：每个候选走一步，用整局价值估走完的局面；0 为不搜")
    ap.add_argument("--dll", default=str(boss_pimc.REPO / "engine/Sts2Headless/bin/Release/net9.0/Sts2Headless.dll"))
    ap.add_argument("--device", default="cuda"); ap.add_argument("--procs", type=int, default=8); ap.add_argument("--policy-seed", type=int, default=0)
    ap.add_argument("--tau", type=float, default=0.9); ap.add_argument("--k", type=int, default=4); ap.add_argument("--z", type=float, default=2.0)
    ap.add_argument("--cap", type=int, default=6000)
    ap.add_argument("--sims", type=int, default=32); ap.add_argument("--decide", choices=["kl", "gumbel"], default="kl")
    ap.add_argument("--lam", type=float, default=0.06, help="kl 的 λ（整局价值的尺度约为战斗终局分的两倍，题目上的 0.03 → 0.06）")
    ap.add_argument("--min-visits", type=int, default=4); ap.add_argument("--root-keep", choices=["mean", "gumbel"], default="mean")
    ap.add_argument("--qnorm", choices=["minmax", "fixed"], default="minmax"); ap.add_argument("--value-range", type=float, default=5.0)
    ap.add_argument("--c-visit", type=float, default=50.0); ap.add_argument("--c-scale", type=float, default=0.1)
    ap.add_argument("--max-depth", type=int, default=200)
    ap.add_argument("--no-reseed-rewards", action="store_true", help="推演的 reseed 不换战后奖励（复现 2026-11-17 以前带漏口的结果）")
    ap.add_argument("--map-bias", default="", help="选路偏置：节点类型=加分（对数概率上），逗号分隔，如 Elite=1.0；局外适应搜索的探针（2026-11-17）")
    ap.add_argument("--req-extra", default="", help='reset 另加的键（JSON），如 \'{"obs_format": 28}\' 给格式 28 时练的检查点（同 ability_test --req-extra，2026-11-21）')
    ap.add_argument("--replays", default="", help="每局另存整局回放到这个目录（胜率小模型的带搜索数据）；默认不存")
    ap.add_argument("--labels", action="store_true",
                    help="每个搜过的根写一行到 labels_s<分片>.jsonl：根帧原始字节与改进后的分布（训练中用搜索的试点，sts2rl/search_kl.py 读）")
    a = ap.parse_args(argv)
    a.run_labels = a.labels
    a.reseed_rewards = not a.no_reseed_rewards
    # tree_search / boss_pimc.Runner 要的其余参数（同 tree_search.main）
    a.no_search = False; a.warm = True; a.worlds = "clone"; a.async_ = False; a.async_wait = 0.0; a.auto_single = True; a.keep_top = False
    a.residual = ""; a.residual_scale = 1.0; a.temp = 0.0; a.save_frames = False; a.sims0 = 1; a.rounds = 1; a.leaf = 0
    a.ema = False; a.infer = ""; a.roll = 0; a.roll_mix = 1.0; a.oracle = False; a.value_ckpts = ""; a.labels = False; a.value_replace = False
    a.items = ""
    return a


def main() -> None:
    run(parse())


if __name__ == "__main__":
    main()
