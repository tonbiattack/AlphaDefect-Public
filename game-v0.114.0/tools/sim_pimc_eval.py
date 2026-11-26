"""PIMC 配对评测：真实引擎执行、sts2-sim 规划（engine/Sts2SimServe/方案.md 第 6 步的验收）。

每个种子一条主干：真实引擎（Sts2Headless）整局贪心打（整局模式，同评测：route_summary、自动领金币），
到 --acts 里那几幕的选路决策、选中的格子是 --rooms 里的房间（默认只有 Boss）时，进这一格之前取游戏自己的整局存档（save 操作），
这场战斗就是一道题：
- 贪心臂：主干自己接着贪心打完这场，之后主干照常往下打（第三幕两个 Boss 各是一道题；第一个输了就没有第二道）；
- 搜索臂：另一个真实引擎进程按主干走过的全部动作重放到同一局面（reset 带 replay，一次往返；第一帧与主干逐项核对），
  sts2-sim 用同一份存档开同一场战斗做镜像；之后真实引擎每走一步，镜像也走同样的动作（按候选语义对应下标）；
  到要搜索的决策（tools/sim_pimc.py 的门限）时在镜像上 save 根局面、按 PIMC 推演，选出的动作交给真实引擎执行。
  镜像与真实局面不一致（Check.cs 的那几项）或候选对不上时，这一步退回贪心并记原因；动作没法对应到镜像时镜像作废，这场余下全按贪心。
两臂从同一个局面开打、真实对局都用游戏原来的随机数，差别只在搜索改了哪几步。

每题一行写到 <out>/items_s<分片>.jsonl；每个种子打完写一行到 <out>/seeds_s<分片>.jsonl（续跑时跳过已打完的种子）。
可用内存低于 --mem-floor-gb 时退出；--pause-file 存在时（复测中）每 30 秒看一次，等它消失再继续。

    .venv/bin/python -I tools/sim_pimc_eval.py run --ckpt model.pt --real-dll Sts2Headless.dll --sim-dll Sts2SimServe.dll \
        --out 目录 --seeds 30 --shard 0/4 [--evaluator heuristic|model] [--sim-procs 2]
    python3 tools/sim_pimc_eval.py report 目录
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "tools"))

MAX_COMBAT_STEPS = 600
HEADER_SKIP = (12, 13, 14)      # 本进程累计赢下的战斗数、步数、词表外键数：与进程有关，核对分叉时不比


def mem_avail_gb() -> float:
    try:
        for l in open("/proc/meminfo"):
            if l.startswith("MemAvailable:"):
                return int(l.split()[1]) / 1048576
    except OSError:
        pass
    return 99.0


def guard(a) -> bool:
    """每题、每个种子开打前：内存不够就退出（返回假），复测中就等。"""
    if mem_avail_gb() < a.mem_floor_gb:
        print(f"可用内存 {mem_avail_gb():.1f} GB < {a.mem_floor_gb} GB，退出", flush=True)
        return False
    waited = False
    while a.pause_file and os.path.exists(a.pause_file):
        if not waited:
            print(f"{time.strftime('%H:%M:%S')} {a.pause_file} 存在（复测中），暂停", flush=True)
            waited = True
        time.sleep(30)
    if waited:
        print(f"{time.strftime('%H:%M:%S')} 继续", flush=True)
    if a.deadline and time.time() > a.deadline:
        print("到截止时刻，不再开新题", flush=True)
        return False
    return True


def frames_equal(x, y) -> str | None:
    import numpy as np
    hx, hy = x.header.copy(), y.header.copy()
    hx[list(HEADER_SKIP)] = 0; hy[list(HEADER_SKIP)] = 0
    if not np.array_equal(hx, hy):
        return f"头部[{int(np.flatnonzero(hx != hy)[0])}]"
    for n in ("tok_ints", "tok_feats", "cand_ints", "cand_feats"):
        u, v = getattr(x, n), getattr(y, n)
        if u.shape != v.shape or not np.array_equal(u, v):
            return n
    return None


# ---------- 决策器：搜索臂每个决策调它；镜像的导入、同步、动作对应与退回贪心由框架管

class Decider:
    """可替换的搜索算法。框架在“镜像与真实局面一致、动作对应得上”时才调 decide；
    wants(o, p) 说这个决策要不要搜（门限，用来统计“该搜却退回贪心”）；decide 返回统一的字典：
    searched、best（真实候选下标）、changed、total_s、sim_s、net_s、work（推演世界数或展开节点数）、errors、err_text、restore_bad、detail。"""
    name = "?"

    def wants(self, o, p) -> bool:
        raise NotImplementedError

    def decide(self, mirror, o, p, cmap, seed_base: int, n: int) -> dict:
        raise NotImplementedError


class PimcDecider(Decider):
    """决策记录 217 的 PIMC 改选（sim_pimc.pimc_decide）。"""
    name = "pimc"

    def __init__(self, S, ev, P):
        self.S, self.ev, self.P = S, ev, P

    def wants(self, o, p) -> bool:
        import numpy as np
        order = np.argsort(-p); act = int(order[0])
        return o.num_actions > 1 and p[act] < self.P.tau and sum(1 for c in order[: self.P.k] if p[c] >= self.P.pmin) > 1

    def decide(self, mirror, o, p, cmap, seed_base, n) -> dict:
        info = self.S.pimc_decide(mirror, self.ev, o, p, cmap, self.P, seed_base, n)
        return {"searched": info.searched, "best": int(info.best), "changed": bool(info.changed), "total_s": info.total_s, "sim_s": info.sim_s,
                "net_s": info.net_s, "work": info.rollouts, "errors": info.errors, "err_text": info.err_text, "restore_bad": info.restore_bad,
                "detail": info.diff.get(info.best), "greedy": int(info.greedy),
                "diff": {int(c): [float(v[0]), float(v[1]), int(v[2])] for c, v in info.diff.items()}}


class TurnDecider(Decider):
    """回合级束搜索（tools/turn_search.py 的 turn_decide）。"""
    name = "turn"

    def __init__(self, T, ev, TP):
        self.T, self.ev, self.TP = T, ev, TP

    def wants(self, o, p) -> bool:
        import numpy as np
        return o.num_actions > 1 and float(np.max(p)) < self.TP.tau

    def decide(self, mirror, o, p, cmap, seed_base, n) -> dict:
        info = self.T.turn_decide(mirror, self.ev, o, p, cmap, self.TP, seed_base + 7919 * (n + 1))     # 种子按题、场、决策号派生，同 PIMC
        return {"searched": info.searched, "best": int(info.best), "changed": bool(info.changed), "total_s": info.total_s, "sim_s": info.sim_s,
                "net_s": info.net_s, "work": info.nodes, "errors": info.errors, "err_text": info.err_text, "restore_bad": 0,
                "detail": {"greedy_score": info.greedy_score, "best_score": info.best_score, "depth": info.depth, "leaves": info.leaves}}


def cmd_run(a) -> None:
    import numpy as np
    import torch
    from sts2env.env import EngineError, Sts2Env
    from sts2env.protocol import DecisionKind, parse
    from sts2rl.model import ModelConfig, PolicyValueNet, load_state
    import sim_pimc as S

    torch.set_num_threads(a.threads)
    blob = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    net = PolicyValueNet(ModelConfig(**blob["config"]["model"])).eval().to(a.device)
    load_state(net, blob["state"])
    route_summary = bool(blob["config"].get("route_summary", True))
    P = S.PimcParams(tau=a.tau, k=a.k, pmin=a.pmin, sims0=a.sims0, rounds=a.rounds, z=a.z, leaf=a.leaf)
    modes = {"run": S.MODE_RUN, "combat": S.MODE_COMBAT}
    real_pol = S.ModelEvaluator(net, a.device, policy_mode=S.MODE_RUN)      # 真实对局的出手（同评测：整局模式）
    ev = (S.ModelEvaluator(net, a.device, policy_mode=modes[a.policy_mode], value_mode=modes[a.value_mode])
          if a.evaluator == "model" else S.HeuristicEvaluator(salt=a.salt))
    if a.algo == "turn":
        if a.evaluator != "model":
            raise SystemExit("回合级束搜索要用策略价值网络（--evaluator model）")
        import turn_search as T
        TP = T.TurnParams(tau=a.tau, beam=a.beam, expand=a.expand, max_depth=a.max_depth, max_nodes=a.max_nodes, margin=a.margin,
                          win_hp=a.win_hp, worlds=a.worlds)
        decider = TurnDecider(T, ev, TP)
    else:
        decider = PimcDecider(S, ev, P)
    acts = {int(x) for x in a.acts.split(",")}
    rooms = set(a.rooms.split(","))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    k_sh, n_sh = map(int, a.shard.split("/"))
    starts = {}
    if a.starts:
        # 从分叉点开打（猎人决策记录第 19 条第 2c 步）：每行 {"key", "seed", "replay", "branch", "first", "bias_names", "bias"}；
        # key 代替种子作为这一局的标识（输出、续跑都按 key），推演世界的种子按行号派生
        rows = [json.loads(l) for l in open(a.starts, encoding="utf-8") if l.strip()]
        starts = {r["key"]: r for r in rows}
        idx_of = {r["key"]: j for j, r in enumerate(rows)}
        seeds = [r["key"] for j, r in enumerate(rows) if j % n_sh == k_sh]
    elif a.seeds_file:
        # 现成的一批种子（例如学生 e 那 1,500 局的 seeds_cyc_ens.json）：按列表下标分片，推演世界的种子也按下标派生
        txt = Path(a.seeds_file).read_text(encoding="utf-8")
        lst = json.loads(txt) if txt.lstrip()[:1] in "[{" else txt.split()
        lst = lst["seeds"] if isinstance(lst, dict) else lst
        idx_of = {x: j for j, x in enumerate(lst)}
        seeds = [x for j, x in enumerate(lst) if j % n_sh == k_sh]
    else:
        idx_of = {f"{a.prefix}{j:04d}": j for j in range(a.start, a.start + a.seeds)}
        seeds = [x for x, j in idx_of.items() if j % n_sh == k_sh]
    if a.first_seeds:
        # 先打这些种子（例如别的算法已经出过题的那批，便于逐题对比），再打其余的；分片仍按种子编号分
        first = {x.strip() for x in Path(a.first_seeds).read_text(encoding="utf-8").split() if x.strip()}
        seeds = [x for x in seeds if x in first] + [x for x in seeds if x not in first]
    if a.only_seeds:
        # 只重打这些种子（例如上一轮里镜像退回过贪心的题，换个设置单独复跑；其余题的结果不变，直接沿用）
        only = {x.strip() for x in Path(a.only_seeds).read_text(encoding="utf-8").split() if x.strip()}
        seeds = [x for x in seeds if x in only]
    done = set()
    for f in out.glob("seeds_s*.jsonl"):
        for l in open(f, encoding="utf-8"):
            if l.strip():
                done.add(json.loads(l)["seed"])
    seeds = [s for s in seeds if s not in done]
    meta = out / "meta.json"
    if not meta.exists():
        meta.write_text(json.dumps({"version": blob.get("version"), "ckpt": a.ckpt, "evaluator": a.evaluator, "policy_mode": a.policy_mode,
                                    "value_mode": a.value_mode, "sim_procs": a.sim_procs, "sim_tables": a.sim_tables, "sim_dll": a.sim_dll, "sync": a.sync, "acts": sorted(acts), "rooms": sorted(rooms),
                                    "params": P.__dict__, "algo": a.algo,
                                    "turn_params": TP.__dict__ if a.algo == "turn" else None, "start": round(time.time())}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"版本 {blob.get('version')}，{len(seeds)} 个种子待打（已打完 {len(done)}）；算法 {a.algo}，推演 {a.evaluator}，{a.sim_procs} 个 sts2-sim 进程；"
          f"tau={P.tau} k={P.k} sims0={P.sims0} rounds={P.rounds} z={P.z} leaf={P.leaf}", flush=True)

    elog = open(out / f"engine_s{k_sh}.log", "ab")
    trunk = Sts2Env(dll=Path(a.real_dll), log=elog)
    branch = Sts2Env(dll=Path(a.real_dll), log=elog)
    vocab = trunk.vocab()
    vocab_path = Path(a.sim_vocab) if a.sim_vocab else out / "vocab.txt"     # sts2-sim 的编码器靠它填记录的词表 ID（须与真实引擎同一份）
    if not a.sim_vocab:
        vocab_path.write_text("\n".join(vocab), encoding="utf-8")
    elif vocab_path.read_text(encoding="utf-8").split("\n")[: len(vocab)] != vocab:
        raise SystemExit(f"{vocab_path} 与真实引擎的词表不同")
    mirror = S.SimMirror(a.sim_procs, a.sim_dll, log=elog, vocab=vocab_path, tables=a.sim_tables)
    f_items = open(out / f"items_s{k_sh}.jsonl", "a", encoding="utf-8")
    # 搜索蒸馏的标签（决策记录 274，--labels）：每场搜索过的战斗存一个文件，里面是这场每个搜索过的决策的真实帧（Ragged）与
    # 搜索结果（贪心项、改选项、各候选相对贪心项的 [平均差, 标准误, 配对数]、网络当时的出手概率）
    lab_dir = out / "labels"
    if a.labels:
        lab_dir.mkdir(exist_ok=True)
    f_seeds = open(out / f"seeds_s{k_sh}.jsonl", "a", encoding="utf-8")
    # 对局资料（--records）：每个种子一行——主干整局贪心的全部动作（reset 带 replay 可重放到任意决策）、结局，
    # 每道题的分叉位置（tape 下标）、分叉前的游戏存档、两臂打完这一幕的全部动作
    f_rec = open(out / f"records_s{k_sh}.jsonl", "a", encoding="utf-8") if a.records else None

    cur_bias = [None]      # (牌名集合, β)：--starts 里带 bias 的局，在选卡奖励界面给这些牌的对数几率加 β（第 19 条的 C 臂）

    def greedy(o) -> int:
        p = real_pol.policy(o)
        if cur_bias[0] is not None and not o.in_combat and int(o.decision) == 3 and o.num_actions > 1:
            from sts2rl.actor import cand_name
            lp = np.log(np.maximum(np.asarray(p, dtype=np.float64), 1e-30))
            for j in range(o.num_actions):
                if cand_name(o, j, vocab) in cur_bias[0][0]:
                    lp[j] += cur_bias[0][1]
            return int(np.argmax(lp))
        return int(np.argmax(p))

    def call(env, req):
        env._proc.send(req)
        return parse(env._proc.recv())

    def names(o) -> list[str]:
        return [vocab[int(o.tok_ints[t, 1])] if 0 <= int(o.tok_ints[t, 1]) < len(vocab) else "?" for t in S._enemies(o)]

    def target_of(o, act: int, sv: dict) -> tuple[str, list[int]]:
        """选路候选指向的格子：节点记录的特征 0、1 是行、列（同 sim_corpus）；房间类型从存档的地图里查（Boss 节点单独存）。"""
        node = int(o.cand_ints[act][1]); f = o.tok_feats[node]
        row, col = int(f[0]), int(f[1])
        m = sv["save"]["acts"][sv["save"]["current_act_index"]]["saved_map"]
        pts = list(m["points"]) + [x for x in (m.get("boss"), m.get("second_boss")) if isinstance(x, dict)]
        pt = next((p for p in pts if p["coord"]["col"] == col and p["coord"]["row"] == row), None)
        return (pt or {}).get("type", "?"), [col, row]

    def play_greedy(env, o) -> tuple[object, dict, list[int]]:
        n, acts_ = 0, []
        while not S.combat_over(o) and n < MAX_COMBAT_STEPS:
            x = greedy(o); acts_.append(x)
            o = call(env, {"op": "step", "action": x}); n += 1
        r = S.outcome(o) if n < MAX_COMBAT_STEPS else {"win": False, "hp": 0, "max_hp": 1}
        r["decisions"] = n
        return o, r, acts_

    def call_raw(env, req):
        env._proc.send(req)
        raw = env._proc.recv()
        return raw, parse(raw)

    def play_search(env, o, raw0, sv, target, room, seed_base) -> tuple[object, dict, list[int]]:
        """搜索臂的一场战斗：env 上的真实对局 + 镜像上的推演。"""
        res = {"decisions": 0, "searched": 0, "changed": 0, "fallback": 0, "fallback_why": {}, "rollouts": 0, "rollout_errors": 0,
               "restore_bad": 0, "times": [], "sim_s": 0.0, "net_s": 0.0, "sim_ops": 0, "mirror_ok": 0, "mirror_bad": 0,
               "first_bad": None, "first_bad_why": None, "lost_at": None, "lost_why": None, "map_by_index": 0, "changes": []}
        t0 = time.perf_counter()
        div0 = mirror.diverged
        try:
            m = mirror.reset(sv["save"], sv["next_room_id"], target, room, raw0 if a.sim_tables else None)
            res["transplant_s"] = round(time.perf_counter() - t0, 3)
        except EngineError as e:
            m = None
            res["lost_at"], res["lost_why"] = 0, f"导入失败：{str(e)[:300]}"
        n = 0
        lab_obs, lab_meta = [], []
        acts_: list[int] = []
        while not S.combat_over(o) and n < MAX_COMBAT_STEPS:
            p = real_pol.policy(o)
            act = int(np.argmax(p))
            cmap, why = None, res["lost_why"] or "镜像作废"
            if m is not None:
                if S.combat_over(m):
                    why = "镜像已结束"
                    cmap = None
                else:
                    # 动作对应先按语义做（局面差一点不影响对应，镜像照样能跟着走）；局面也一致才允许搜索
                    cmap, how = S.cand_map(o, m)
                    if how != "语义" and "map_note" not in res:     # 第一次对应不上语义时记下两边的差，给 sts2-sim 那边查
                        res["map_note"] = {"n": n, "how": how, **S.cand_diff(o, m, vocab)}
                    why = S.compare(o, m) or (how if cmap is None else None)
                    if why is not None and a.sync and o.decision == DecisionKind.COMBAT and not S._is_select(m):
                        # 规则差异让镜像走岔了：按真实帧把镜像看得到的部分校正过来再对一次（只让推演有偏差，不让搜索停掉）
                        try:
                            m = mirror._all(S.sync_req(o, vocab))
                            res["sync"] = res.get("sync", 0) + 1
                            cmap, how = S.cand_map(o, m)
                            why = S.compare(o, m) or (how if cmap is None else None)
                            res["sync_ok"] = res.get("sync_ok", 0) + (why is None)
                            if why is not None and "sync_note" not in res:
                                res["sync_note"] = {"n": n, "why": why}
                        except EngineError as e:
                            res["sync_err"] = res.get("sync_err", 0) + 1
                            why = f"同步失败：{str(e)[:120]}"
                    res["map_by_index"] += how == "按下标"
                    res["map_loose"] = res.get("map_loose", 0) + (how == "宽松语义")
                    res["sim_extra_cands"] = res.get("sim_extra_cands", 0) + (how == "镜像多出候选")
                if a.sim_tables and not S.combat_over(m):
                    # 移植版编码器：整帧逐项比（不比进程相关的头部 12–14），记一致的个数与第一处不同
                    fd = frames_equal(o, m)
                    res["frame_eq"] = res.get("frame_eq", 0) + (fd is None)
                    if fd is not None and "frame_first_diff" not in res:
                        res["frame_first_diff"] = {"n": n, "seg": fd}
                if why is None:
                    res["mirror_ok"] += 1
                else:
                    res["mirror_bad"] += 1
                    if res["first_bad"] is None:
                        res["first_bad"], res["first_bad_why"] = n, why
            gate = decider.wants(o, p)
            if gate and (why is not None or cmap is None):
                res["fallback"] += 1
                key = why.split(":")[0].split("：")[0]
                res["fallback_why"][key] = res["fallback_why"].get(key, 0) + 1
            elif gate:
                info = decider.decide(mirror, o, p, cmap, seed_base, n)
                if not info["searched"] and info["err_text"]:
                    res["fallback"] += 1
                    k_ = "无锚点" if "锚点" in info["err_text"] else info["err_text"].split("：")[0][:20]
                    res["fallback_why"][k_] = res["fallback_why"].get(k_, 0) + 1
                if info["searched"] and a.labels:
                    lab_obs.append(o)
                    lab_meta.append({"n": n, "greedy": info["greedy"], "best": info["best"], "changed": bool(info["changed"]), "diff": info["diff"],
                                     "p": [round(float(x), 5) for x in p[:o.num_actions]], "floor": int(o.floor), "seed_base": int(seed_base)})
                if info["searched"]:
                    res["searched"] += 1
                    res["rollouts"] += info["work"]; res["rollout_errors"] += info["errors"]; res["restore_bad"] += info["restore_bad"]
                    res["times"].append(round(info["total_s"], 4))
                    res["sim_s"] += info["sim_s"]; res["net_s"] += info["net_s"]
                    if info["err_text"] and "err_text" not in res:
                        res["err_text"] = info["err_text"]
                    if info["changed"]:
                        res["changed"] += 1
                        res["changes"].append({"n": n, "from": act, "to": info["best"], "p": [round(float(p[act]), 3), round(float(p[info["best"]]), 3)],
                                               "detail": info["detail"]})
                    act = info["best"]
            o = call(env, {"op": "step", "action": act}); n += 1; acts_.append(act)
            if m is not None and not S.combat_over(m):
                if cmap is None:
                    res["lost_at"], res["lost_why"] = n, "动作对应不到镜像"
                    m = None
                else:
                    try:
                        m = mirror.step(cmap[act])
                    except EngineError as e:
                        res["lost_at"], res["lost_why"] = n, f"镜像走步报错：{str(e)[:200]}"
                        m = None
        r = S.outcome(o) if n < MAX_COMBAT_STEPS else {"win": False, "hp": 0, "max_hp": 1}
        res.update(r)
        if lab_obs:
            import pickle
            from sts2rl.actor import MODE_RUN
            from sts2rl.features import Ragged
            fn = lab_dir / f"s{k_sh}_{seed_base}_{os.getpid()}.pkl"
            with open(fn.with_suffix(".tmp"), "wb") as fh:
                pickle.dump({"ragged": Ragged.from_obs(lab_obs, np.full(len(lab_obs), MODE_RUN)), "meta": lab_meta, "win": bool(r.get("win")),
                             "version": a.ckpt}, fh)
            fn.with_suffix(".tmp").replace(fn)
        res["decisions"] = n
        res["sim_s"], res["net_s"] = round(res["sim_s"], 3), round(res["net_s"], 3)
        res["wall_s"] = round(time.perf_counter() - t0, 2)
        res["procs_diverged"] = mirror.diverged - div0      # 搜索后各镜像进程回到的局面逐字节不同的次数（应为 0）
        if m is not None:
            # 真实对局结束时镜像也该结束、结果相同（输赢不同说明镜像在后段走偏了）
            res["mirror_end"] = {"over": S.combat_over(m), **({"win": S.outcome(m)["win"], "hp": int(m.hp)} if S.combat_over(m) else {})}
        return o, res, acts_

    SUM_KEYS = ("decisions", "searched", "changed", "fallback", "rollouts", "rollout_errors", "restore_bad", "sim_s", "net_s", "sim_ops",
                "mirror_ok", "mirror_bad", "map_by_index", "sim_extra_cands", "frame_eq", "procs_diverged", "wall_s", "sync", "sync_ok", "sync_err", "map_loose")

    def play_chain(env, o, raw, sv, target, room, search: bool, seed_idx: int) -> tuple[object, dict, list[int]]:
        """一条臂从分叉点打到这一幕结束（第三幕即整局结束）：符合 --rooms 的战斗逐场打（搜索臂搜索、贪心臂贪心），
        其余决策都贪心；到下一场符合的战斗之前在选路决策上取存档，搜索臂用自己的存档给镜像导入。
        结果：win = 这一幕的 Boss 都打赢（通关或进了下一幕），fights 为逐场结果。"""
        act0, fights, acts_ = int(o.act), [], []
        while True:
            if search:
                o, r, xs = play_search(env, o, raw, sv, target, room, 100003 * (seed_idx * 8 + len(fights) + 1)); acts_ += xs
            else:
                o, r, xs = play_greedy(env, o); acts_ += xs
            r.update(room=room, enemies=names_last[0])
            fights.append(r)
            nxt = False
            steps = 0
            while not o.done and int(o.act) == act0 and steps < 3000:
                x = greedy(o); acts_.append(x); steps += 1
                if o.decision == DecisionKind.MAP:
                    env._proc.send({"op": "save"}); sv = json.loads(env._proc.recv())
                    room, target = target_of(o, x, sv)
                    raw, o = call_raw(env, {"op": "step", "action": x})
                    if room in rooms and o.in_combat and not o.done:
                        names_last[0] = names(o)
                        nxt = True
                        break
                    continue
                raw, o = call_raw(env, {"op": "step", "action": x})
            if not nxt:
                break
        win = bool(o.victory) or (not o.done and int(o.act) > act0)
        res = {"win": win, "hp": int(fights[-1]["hp"]), "max_hp": int(fights[-1].get("max_hp", 1)), "n_fights": len(fights),
               "fights_won": sum(bool(f["win"]) for f in fights)}
        if search:
            for k in SUM_KEYS:
                res[k] = round(sum(f.get(k, 0) for f in fights), 3)
            res["times"] = [t for f in fights for t in f["times"]]
            res["changes"] = [dict(c, fight=j) for j, f in enumerate(fights) for c in f["changes"]]
            res["fallback_why"] = {}
            for f in fights:
                for k, v in f["fallback_why"].items():
                    res["fallback_why"][k] = res["fallback_why"].get(k, 0) + v
            for f in fights:        # 逐场只留镜像诊断，去掉已汇总的大字段
                for k in ("times", "changes"):
                    f.pop(k, None)
        else:
            res["decisions"] = sum(f["decisions"] for f in fights)
        res["fights"] = fights
        return o, res, acts_

    names_last = [[]]       # 当前这场战斗的敌人名字（play_chain 记进逐场结果）

    t_start = time.time()
    n_items = 0
    extra = json.loads(a.req_extra) if a.req_extra else {}

    def start_req(key):
        if key in starts:
            st = starts[key]
            return {"op": "reset", "mode": "run", "character": a.character, "ascension": 10, "seed": st["seed"], "route_summary": route_summary,
                    "replay": st["replay"], "branch": st["branch"], "first": st["first"], **extra}
        return {"op": "reset", "mode": "run", "character": a.character, "ascension": 10, "seed": key, "route_summary": route_summary, **extra}

    for seed in seeds:
        if not guard(a):
            break
        st_ = starts.get(seed)
        cur_bias[0] = (set(st_["bias_names"]), float(st_["bias"])) if st_ and st_.get("bias") else None
        o = call(trunk, start_req(seed))
        tape: list[int] = []
        probs: list[dict] = []
        k_boss = steps = 0
        forked: set[int] = set()
        stopped = False
        while not o.done and steps < 6000:
            act = greedy(o)
            if o.decision == DecisionKind.MAP and o.act in acts:
                trunk._proc.send({"op": "save"}); sv = json.loads(trunk._proc.recv())
                room, target = target_of(o, act, sv)
                o = call(trunk, {"op": "step", "action": act}); tape.append(act); steps += 1
                if room in rooms and o.in_combat and not o.done and int(o.act) not in forked:
                    forked.add(int(o.act))
                    # 分叉只在这一幕第一场符合的战斗开打前做一次；两臂各自打完这一幕（第三幕即两个 Boss），按“这一幕都赢”配对
                    if not guard(a):
                        stopped = True
                        break
                    names_last[0] = names(o)
                    item = {"seed": seed, "k": 0, "act": int(o.act), "floor": int(o.floor), "room": room, "target": target,
                            "enemies": names(o), "hp0": int(o.hp), "max_hp": int(o.max_hp), "version": blob.get("version")}
                    if st_ is None:
                        branch._proc.send({"op": "reset", "mode": "run", "character": a.character, "ascension": 10, "seed": seed,
                                           "route_summary": route_summary, "replay": tape, **extra})
                        raw0 = branch._proc.recv(); ob = parse(raw0)
                    else:
                        # 引擎的 reset 先重放再分枝，分枝之后的动作只能逐步走
                        branch._proc.send(start_req(seed)); raw0 = branch._proc.recv()
                        for x in tape:
                            branch._proc.send({"op": "step", "action": x}); raw0 = branch._proc.recv()
                        ob = parse(raw0)
                    item["branch_diff"] = frames_equal(o, ob)
                    seed_idx = idx_of[seed]         # 推演世界的种子由种子编号（列表下标）与第几场决定，与真实随机数无关
                    fork_at = len(tape)
                    _, item["search"], s_acts = play_chain(branch, ob, raw0, sv, target, room, True, seed_idx)
                    names_last[0] = item["enemies"]
                    o, item["greedy"], acts_ = play_chain(trunk, o, None, sv, target, room, False, seed_idx)
                    if f_rec:
                        probs.append({"k": k_boss, "act": item["act"], "floor": item["floor"], "room": room, "target": target, "fork_at": fork_at,
                                      "save": sv, "greedy_actions": acts_, "search_actions": s_acts,
                                      "greedy_win": bool(item["greedy"]["win"]), "search_win": bool(item["search"]["win"])})
                    tape += acts_; steps += len(acts_)
                    f_items.write(json.dumps(item, ensure_ascii=False) + "\n"); f_items.flush()
                    n_items += 1
                    s_, g_ = item["search"], item["greedy"]
                    ts = s_["times"]
                    fs = lambda r: "/".join(("赢" if f["win"] else "输") + str(f["hp"]) for f in r["fights"])
                    lost = [f"第{j + 1}场第{f['lost_at']}步（{f.get('lost_why') or ''}）" for j, f in enumerate(s_["fights"]) if f.get("lost_at") is not None]
                    print(f"{time.strftime('%H:%M:%S')} {seed} 第{item['act'] + 1}幕 {'/'.join(item['enemies'])} 起：贪心 {'赢' if g_['win'] else '输'}（{fs(g_)}）"
                          f"，搜索 {'赢' if s_['win'] else '输'}（{fs(s_)}）；搜索 {s_['searched']} 次、改选 {s_['changed']}、退回 {s_['fallback']}"
                          f"、镜像一致 {s_['mirror_ok']}/{s_['mirror_ok'] + s_['mirror_bad']}{'、镜像作废 ' + '；'.join(lost)[:200] if lost else ''}"
                          f"；每次搜索 {np.mean(ts) if ts else 0:.2f} 秒（搜索臂共 {s_['wall_s']:.0f} 秒）", flush=True)
                    k_boss += 1
                continue
            o = call(trunk, {"op": "step", "action": act}); tape.append(act); steps += 1
        if stopped:
            break
        f_seeds.write(json.dumps({"seed": seed, "win": bool(o.victory), "floor": int(o.floor), "act": int(o.act), "items": k_boss,
                                  "t": round(time.time())}, ensure_ascii=False) + "\n"); f_seeds.flush()
        if f_rec:
            f_rec.write(json.dumps({"seed": seed, "seed_idx": idx_of[seed], "win": bool(o.victory), "floor": int(o.floor), "act": int(o.act),
                                    "version": blob.get("version"), "route_summary": route_summary, "tape": tape, "problems": probs},
                                   ensure_ascii=False) + "\n"); f_rec.flush()
    print(f"完成 {n_items} 题，用时 {time.time() - t_start:.0f} 秒", flush=True)
    for x in (trunk, branch, mirror):
        x.close()


# ---------- report：配对统计

def _binom_two_sided(k: int, n: int) -> float:
    """McNemar 精确检验：不一致对里“只有搜索赢”的个数 k 服从 Bin(n, 1/2)，双侧 p。"""
    if n == 0:
        return 1.0
    pk = [math.comb(n, i) / 2 ** n for i in range(n + 1)]
    return min(1.0, sum(x for x in pk if x <= pk[k] * (1 + 1e-9)))


def _pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else 0.0


def cmd_report(a) -> None:
    items, seen = [], set()
    for d in a.dirs:
        for f in sorted(Path(d).glob("items_s*.jsonl")):
            for l in open(f, encoding="utf-8"):
                if l.strip():
                    r = json.loads(l)
                    key = (d, r["seed"], r["k"])
                    if key not in seen:         # 中断续跑时同一种子的题可能重做，留第一次
                        seen.add(key); items.append(r)
    if not items:
        print("没有结果"); return
    n = len(items)
    gw = sum(r["greedy"]["win"] for r in items); sw = sum(r["search"]["win"] for r in items)
    only_s = sum(r["search"]["win"] and not r["greedy"]["win"] for r in items)
    only_g = sum(r["greedy"]["win"] and not r["search"]["win"] for r in items)
    d = [int(r["search"]["win"]) - int(r["greedy"]["win"]) for r in items]
    md = sum(d) / n
    se = math.sqrt(sum((x - md) ** 2 for x in d) / (n - 1) / n) if n > 1 else 0.0
    print(f"{n} 题：贪心 {gw}（{100 * gw / n:.1f}%），搜索 {sw}（{100 * sw / n:.1f}%），差 {100 * md:+.1f} ± {100 * se:.1f} 个百分点；"
          f"只有搜索赢 {only_s}、只有贪心赢 {only_g}，McNemar 精确 p = {_binom_two_sided(only_s, only_s + only_g):.3g}")
    both = [r for r in items if r["search"]["win"] and r["greedy"]["win"]]
    if both:
        dh = [(r["search"]["hp"] - r["greedy"]["hp"]) / max(1, r["max_hp"]) for r in both]
        print(f"两边都赢的 {len(both)} 题：剩余生命（占最大生命）搜索减贪心平均 {100 * sum(dh) / len(dh):+.1f} 个百分点"
              f"（搜索多 {sum(x > 0 for x in dh)}、少 {sum(x < 0 for x in dh)}、相同 {sum(x == 0 for x in dh)}）")
    S_ = [r["search"] for r in items]
    times = [t for s in S_ for t in s["times"]]
    dec = sum(s["decisions"] for s in S_)
    print(f"搜索臂共 {dec} 个决策：搜索 {sum(s['searched'] for s in S_)} 次（每题 {sum(s['searched'] for s in S_) / n:.1f}）、改选 {sum(s['changed'] for s in S_)}、"
          f"退回贪心 {sum(s['fallback'] for s in S_)}；推演 {sum(s['rollouts'] for s in S_)} 个世界（回合搜索为展开的节点数）、报错 {sum(s['rollout_errors'] for s in S_)}、"
          f"还原根局面不一致 {sum(s['restore_bad'] for s in S_)}、镜像进程之间不一致 {sum(s.get('procs_diverged', 0) for s in S_)}")
    if times:
        sim_s = sum(s["sim_s"] for s in S_); net_s = sum(s["net_s"] for s in S_)
        print(f"每次搜索耗时：平均 {sum(times) / len(times):.3f} 秒，中位 {_pct(times, 0.5):.3f}，p90 {_pct(times, 0.9):.3f}，最长 {max(times):.3f}；"
              f"其中 sts2-sim 往返 {sim_s:.0f} 秒、推演出手与估值 {net_s:.0f} 秒（合计 {sum(times):.0f} 秒），"
              f"每个世界平均 {1000 * sum(times) / max(1, sum(s['rollouts'] for s in S_)):.1f} 毫秒")
    F_ = [f for s in S_ for f in (s.get("fights") or [s])]        # 逐场（连打口径的结果里一题有几场）
    if any("fights" in r["greedy"] for r in items):
        for j in range(max(len(r["greedy"]["fights"]) for r in items)):
            gj = [r["greedy"]["fights"][j] for r in items if len(r["greedy"]["fights"]) > j]
            sj = [r["search"]["fights"][j] for r in items if len(r["search"]["fights"]) > j]
            print(f"第 {j + 1} 场：贪心打到 {len(gj)}、赢 {sum(f['win'] for f in gj)}；搜索打到 {len(sj)}、赢 {sum(f['win'] for f in sj)}")
    ok = sum(f["mirror_ok"] for f in F_); bad = sum(f["mirror_bad"] for f in F_)
    n_s = sum(s["searched"] for s in S_); n_f = sum(s["fallback"] for s in S_)
    print(f"该搜索的决策 {n_s + n_f} 个：搜索 {n_s}、退回贪心 {n_f}（{100 * n_f / max(1, n_s + n_f):.1f}%）；改选率 {100 * sum(s['changed'] for s in S_) / max(1, n_s):.1f}%")
    if any(f.get("sync") for f in F_):
        print(f"同步 {sum(f.get('sync', 0) for f in F_)} 次，同步后一致 {sum(f.get('sync_ok', 0) for f in F_)} 次，同步报错 {sum(f.get('sync_err', 0) for f in F_)} 次")
    why: dict = {}
    for s in S_:
        for k, v in s["fallback_why"].items():
            why[k] = why.get(k, 0) + v
    lost = [f for f in F_ if f["lost_at"] is not None]
    lw: dict = {}
    for f in lost:
        k = (f.get("lost_why") or "?").split("：")[0]
        lw[k] = lw.get(k, 0) + 1
    print(f"镜像（逐场 {len(F_)} 场）：决策一致 {ok}/{ok + bad}（{100 * ok / max(1, ok + bad):.1f}%），整场一致 {sum(f['mirror_bad'] == 0 and f['lost_at'] is None for f in F_)}/{len(F_)}；"
          f"宽松对应 {sum(f.get('map_loose', 0) for f in F_)} 次、按下标对应 {sum(f['map_by_index'] for f in F_)} 次、镜像多出候选 {sum(f.get('sim_extra_cands', 0) for f in F_)} 次；退回原因 {why}；镜像作废 {len(lost)} 场 {lw}")
    if any("frame_eq" in f for f in F_):
        segs: dict = {}
        for f in F_:
            if f.get("frame_first_diff"):
                segs[f["frame_first_diff"]["seg"]] = segs.get(f["frame_first_diff"]["seg"], 0) + 1
        print(f"整帧一致（移植版编码器）：{sum(f.get('frame_eq', 0) for f in F_)}/{ok + bad} 个决策；每场第一处不同的段 {segs}")
    fb: dict = {}
    for f in F_:
        if f["first_bad_why"]:
            k = f["first_bad_why"].split(":")[0]
            fb[k] = fb.get(k, 0) + 1
    if fb:
        print(f"第一处不一致的项：{fb}")
    br = [r for r in items if r.get("branch_diff")]
    if br:
        print(f"分叉核对不一致 {len(br)} 题：{[r['branch_diff'] for r in br][:10]}")
    mend = [f for f in F_ if f.get("mirror_end", {}).get("over") and f["mirror_end"].get("win") != f["win"]]
    if mend:
        print(f"镜像跟到最后、但结果与真实不同 {len(mend)} 场")
    by: dict = {}
    for r in items:
        e = "/".join(r["enemies"]) or "?"
        b = by.setdefault(e, [0, 0, 0])
        b[0] += 1; b[1] += r["greedy"]["win"]; b[2] += r["search"]["win"]
    print("分 Boss（题数 贪心赢 搜索赢）：" + "，".join(f"{k} {v[0]} {v[1]} {v[2]}" for k, v in sorted(by.items(), key=lambda kv: -kv[1][0])))
    if a.items:
        for r in items:
            g, s = r["greedy"], r["search"]
            print(f"  {r['seed']}#{r['k']} 第{r['act'] + 1}幕 {'/'.join(r['enemies'])} 开战 {r['hp0']}/{r['max_hp']}：贪心 {'赢' if g['win'] else '输'} {g['hp']}，"
                  f"搜索 {'赢' if s['win'] else '输'} {s['hp']}（搜索 {s['searched']}、改选 {s['changed']}、退回 {s['fallback']}）")


def _load_items(dirs) -> dict:
    out = {}
    for d in dirs:
        for f in sorted(Path(d).glob("items_s*.jsonl")):
            for l in open(f, encoding="utf-8"):
                if l.strip():
                    r = json.loads(l)
                    out.setdefault(r["seed"], r)
    return out


def cmd_compare(a) -> None:
    """几组评测逐题对比（同一种子）：各组的贪心臂应完全一致（核对），搜索臂各列一栏。"""
    groups = [(g.split("=")[0], _load_items(g.split("=")[1].split(","))) for g in a.groups]
    common = sorted(set.intersection(*(set(d) for _, d in groups)))
    if not common:
        print("没有共同的题"); return
    names = [n for n, _ in groups]
    fs = lambda r: "/".join(("赢" if f["win"] else "输") + str(f["hp"]) for f in r["fights"])
    bad_g = 0
    print(f"{len(common)} 道共同的题：" + "，".join(names))
    tally = {n: 0 for n in names}; tally_g = 0
    for sd in common:
        g0 = groups[0][1][sd]["greedy"]
        same = all(json.dumps([(f["win"], f["hp"], f["decisions"]) for f in d[sd]["greedy"]["fights"]]) ==
                   json.dumps([(f["win"], f["hp"], f["decisions"]) for f in g0["fights"]]) for _, d in groups)
        bad_g += not same
        tally_g += g0["win"]
        for n, d in groups:
            tally[n] += d[sd]["search"]["win"]
        if a.items:
            print(f"  {sd} 贪心 {fs(g0)}{'' if same else '（各组贪心臂不一致！）'}｜" +
                  "｜".join(f"{n} {fs(d[sd]['search'])}（搜 {d[sd]['search']['searched']} 改 {d[sd]['search']['changed']}）" for n, d in groups))
    print(f"两个 Boss 都赢：贪心 {tally_g}，" + "，".join(f"{n} {tally[n]}" for n in names) + f"；贪心臂不一致的题 {bad_g}")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            A, B = groups[i][1], groups[j][1]
            oa = sum(A[s_]["search"]["win"] and not B[s_]["search"]["win"] for s_ in common)
            ob = sum(B[s_]["search"]["win"] and not A[s_]["search"]["win"] for s_ in common)
            print(f"  {names[i]} 对 {names[j]}：只有 {names[i]} 赢 {oa}、只有 {names[j]} 赢 {ob}，p = {_binom_two_sided(oa, oa + ob):.3g}")
    for n, d in groups:
        oa = sum(d[s_]["search"]["win"] and not d[s_]["greedy"]["win"] for s_ in common)
        ob = sum(d[s_]["greedy"]["win"] and not d[s_]["search"]["win"] for s_ in common)
        print(f"  {n} 对贪心：只有搜索赢 {oa}、只有贪心赢 {ob}，p = {_binom_two_sided(oa, oa + ob):.3g}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--ckpt", required=True)
    r.add_argument("--real-dll", required=True, help="真实引擎 Sts2Headless.dll（要带只读的 save 操作）")
    r.add_argument("--sim-dll", default=None, help="Sts2SimServe.dll（默认环境变量 STS2SIM_DLL 或仓库里的编译输出）")
    r.add_argument("--sync", action="store_true", help="镜像与真实局面不一致时先用服务的 sync 操作校正再比（要带 sync 的服务）")
    r.add_argument("--sim-vocab", default=None, help="给 sts2-sim 的词表文件（默认把真实引擎 vocab 操作的输出写到 <out>/vocab.txt）")
    r.add_argument("--sim-tables", default=None, help="sim_tables.json：给了就让 sts2-sim 用移植版观测编码器（服务要按 -p:WithObs=true 编译），"
                                                      "开战时把真实引擎的第一帧作为根帧传过去")
    r.add_argument("--out", required=True)
    r.add_argument("--seeds", type=int, default=20)
    r.add_argument("--start", type=int, default=0)
    r.add_argument("--prefix", default="SIMS", help="种子前缀（SIMS0000…，不与评测、复测、语料的种子重叠）")
    r.add_argument("--character", default="DEFECT", help="打哪个角色（检查点要是替这个角色训练的；储君要用移植了观测格式 20 的 sts2-sim）")
    r.add_argument("--shard", default="0/1")
    r.add_argument("--first-seeds", default=None, help="文件，每行一个种子：范围内的这些种子先打")
    r.add_argument("--only-seeds", default=None, help="文件，每行一个种子：范围内只打这些种子")
    r.add_argument("--seeds-file", default=None, help="现成的种子列表（JSON 列表、带 seeds 的 JSON 或每行一个）：代替 --prefix/--start/--seeds，按下标分片")
    r.add_argument("--records", action="store_true", help="对局资料：每个种子把主干全部动作、结局、每道题的分叉位置、存档与两臂动作写进 <out>/records_s<分片>.jsonl")
    r.add_argument("--acts", default="2", help="哪几幕的战斗出题（0 起算，默认第三幕）")
    r.add_argument("--rooms", default="boss", help="哪类房间出题：boss、elite、monster，逗号分隔")
    r.add_argument("--algo", default="pimc", choices=["pimc", "turn"], help="搜索臂的决策器：PIMC 改选（决策记录 217）或回合级束搜索（tools/turn_search.py）")
    r.add_argument("--beam", type=int, default=6, help="回合级束搜索：每层留几个节点")
    r.add_argument("--expand", type=int, default=4, help="回合级束搜索：每个节点展开策略最高的几个候选（另加结束回合）")
    r.add_argument("--max-depth", type=int, default=10)
    r.add_argument("--max-nodes", type=int, default=400)
    r.add_argument("--margin", type=float, default=0.03, help="回合级束搜索：最好的方案比贪心路线高出多少才改")
    r.add_argument("--win-hp", type=float, default=0.5, help="回合级束搜索：打赢的得分 = 1.5 + 它 × 剩余生命比例")
    r.add_argument("--worlds", type=int, default=1, help="回合级束搜索：几个重抽的世界")
    r.add_argument("--evaluator", default="heuristic", choices=["heuristic", "model"], help="推演的出手与估值：替身或策略价值网络")
    r.add_argument("--policy-mode", default="run", choices=["run", "combat"], help="网络推演出手用的对局模式")
    r.add_argument("--value-mode", default="combat", choices=["run", "combat"], help="叶子估值用的对局模式（战斗课程的价值与 terminal() 同一尺度）")
    r.add_argument("--salt", type=int, default=0, help="替身随机出牌的盐")
    r.add_argument("--sim-procs", type=int, default=1, help="镜像用几个 sts2-sim 进程（各自推演一部分世界）")
    r.add_argument("--device", default="cpu")
    r.add_argument("--threads", type=int, default=2)
    r.add_argument("--starts", default=None, help="分叉点开局文件（jsonl，tools/branch_force.py 的节点转成；猎人决策记录第 19 条第 2c 步）：代替种子列表")
    r.add_argument("--req-extra", default=None, help="并进每局 reset 请求的额外字段（JSON），如常规评测的截断开关 {\"loop_cut\": true, \"stall_turns\": 20}")
    r.add_argument("--tau", type=float, default=0.9)
    r.add_argument("--k", type=int, default=4)
    r.add_argument("--pmin", type=float, default=0.02)
    r.add_argument("--sims0", type=int, default=4)
    r.add_argument("--rounds", type=int, default=2)
    r.add_argument("--z", type=float, default=2.0)
    r.add_argument("--leaf", type=int, default=12)
    r.add_argument("--deadline", type=float, default=0, help="Unix 时间：过了就不再开新题")
    r.add_argument("--labels", action="store_true", help="搜索蒸馏的标签：每场搜索过的战斗把搜索过的决策（真实帧与各候选的搜索结果）存进 <out>/labels/（决策记录 274）")
    r.add_argument("--pause-file", default="/root/.local/share/sls2/runs/defect-eval/PAUSE", help="这个文件存在时（复测中）暂停")
    r.add_argument("--mem-floor-gb", type=float, default=3.0)
    q = sub.add_parser("report")
    q.add_argument("dirs", nargs="+")
    q.add_argument("--items", action="store_true", help="逐题列出")
    c = sub.add_parser("compare", help="几组评测逐题对比：组名=目录[,目录…]，第一组的贪心臂作基准")
    c.add_argument("groups", nargs="+")
    c.add_argument("--items", action="store_true")
    a = ap.parse_args()
    {"run": cmd_run, "report": cmd_report, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    main()
