"""复制保真验收：在大样本的随机分叉点上，比较“快照 → 恢复”与“从头重放”两条路得到的每一帧（主会话加局面复制的核心验收）。

两条路（同一个 dll，两个进程）：
  - 复制路径 C：主线走到分叉点 → 拍快照（槽位 k）→ 对每个世界 restore（+ reseed + first）→ 再走 K 步；
  - 重放路径 R：每个世界 reset + replay（主线走过的动作）+ reseed + first → 再走同样的 K 步（现在推演的做法，tools/future/boss_pimc.py）。
  两边同样的请求、同样的随机动作（由帧内容与世界自己的随机数决定，帧相同则动作相同），逐帧比原始字节，跳过头部 12、14（随进程累计）。
  一个世界里第一次出现不同就记下第一处差异并结束这个世界；两边对同一请求报错相同算一致。

每个分叉点 4 个世界：世界 0 不带 reseed（restore 后 step 首步：沿用原来的随机数，最能查出随机数状态没恢复），
世界 1–3 各带不同的 reseed 与首步（尽量取不同的候选）。另外覆盖：
  - 同一快照恢复多次：4 个世界 + 主线回到分叉点再走，都从同一槽位恢复；世界 0 与主线恢复的帧还要与拍快照时的帧逐字节全同（含头部 12、14）。
  - 恢复前进程先走过别的世界：世界 1–3 都是在走完上一个世界之后恢复的。
  - 快照之后 reset 开了别的局，再恢复旧槽位（“他局”）：一部分分叉点在跑世界之前，C 先开另一局（换模式、换角色、换开关、换 labels），
    走几步、拍一份别的槽位再丢掉，然后恢复旧槽位。
  - 带 "snapshot":k 的请求：一部分分叉点的快照由主线的 step（第 0 步由 reset）顺手存；一部分世界 3 的 restore 带 "snapshot"
    （嵌套：从“reseed + first 之后”的局面再分一个子世界，R 这边用 reset + replay + reseed + first、再单独 reseed、再 step 对照）。
  - 旧槽位：一局结束时，从这一局较早的分叉点（槽位还在的）再分一个世界；主线终局帧与 R 的 reset + replay 全部动作比一次。
  - 槽位号循环使用（k = 分叉序号 mod 4，覆盖旧快照），一局结束 drop 全部并核对回复。

两种模式：
  - combat（战斗练习）：reset 带 "loop_cut":true,"stall_turns":20。语料（--sources，名字:场数）：
      starter:/random:角色  同 pair_diff 的单场战斗（种子 RD{i}、遭遇轮转、random 由 random.Random(f"deck{i}") 抽牌组）；
      drawtop:角色          针对性牌组（主会话指定）：会往抽牌堆顶放牌的牌，覆盖“拍快照时已知牌堆顶非空”的局面（储君格式 20、愚者 24 起读它）；
      boss                  items_t1.json 的第三幕 Boss 题（愚者，连打第二个 Boss）。
    主线：偏向多打牌的随机出牌（同 engine_equal.play_bias），最多 300 步。
  - run（整局）：随机出牌打整局，reset 带训练采集的开关 privileged、route_summary、future_content、snapshots（engine_equal.RUN_FLAGS），
    偶数局再带 labels。分叉概率战斗外高、战斗里低，覆盖战斗与非战斗决策。最多 1,200 步。
分叉：主线每一帧以一定概率分叉（已知牌堆顶非空的帧概率更高），每局最多 --max-forks 个。

口径与目标：战斗练习至少 1,000 个分叉点，整局至少 300 个。输出分叉点数、世界数、帧数、不同数、第一处差异，
以及按决策类型、已知牌堆顶、他局、嵌套、旧槽位分的计数。每个不同都写一份复现文件（repro/…json：C、R 两边各自的请求序列），
`repro` 子命令照着重放并打印差异。

反向核对：--env-c STS2_CLONE_WEAK_LEAF=1 让复制进程退回旧的漏记写法（弱引用表当叶子），同样的语料应当报出不同。

  python -I tools/clone/clone_fidelity.py run --mode combat --out 目录 [--sources drawtop:REGENT=60,boss=40] [--workers 2]
  python -I tools/clone/clone_fidelity.py run --mode run --out 目录
  python -I tools/clone/clone_fidelity.py summary --out 目录
  python -I tools/clone/clone_fidelity.py repro 目录/repro/xxx.json
分片续跑：每个分片把做完的局记进 episodes_s{k}.jsonl，重跑同一命令跳过已做的局。默认 2 个分片（每片 C、R 两个引擎进程，合计 4 个），nice 10。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from common import EngineError  # noqa: E402

ITEMS = Path.home() / ".local" / "share" / "sls2" / "clone-bench" / "items_t1.json"
CHARS = ["DEFECT", "FOOL", "REGENT", "SILENT", "IRONCLAD", "NECROBINDER"]
# 针对性牌组：往抽牌堆顶放牌（主会话 2026-11-16 指定）。储君：光子切割、漠然、闪耀打击、微光、我无敌；愚者：气功吟唱、回想、未雨绸缪；
# 铁甲：头槌、破坏、倾泻（HEADBUTT、HAVOC、CASCADE）
# v0.114.0：召回不再把弃牌堆的牌放到抽牌堆顶，改用哇 叽（WahKi.OnPlay：从弃牌堆选一张放到抽牌堆顶）
FOOL_TOPPER = "WAH_KI" if tuple(int(x) for x in C.CATALOG_VERSION.lstrip("v").split(".")) >= (0, 114, 0) else "RECALL"
DRAWTOP = {
    "REGENT": ["STRIKE_REGENT"] * 3 + ["DEFEND_REGENT"] * 3 + ["FALLING_STAR", "VENERATE"] + ["PHOTON_CUT"] * 3 + ["COSMIC_INDIFFERENCE"] * 3
              + ["SHINING_STRIKE"] * 2 + ["GLIMMER"] * 2 + ["I_AM_INVINCIBLE"],
    "FOOL": ["STRIKE_FOOL"] * 3 + ["DEFEND_FOOL"] * 3 + ["FOOLISH_SHOT"] + ["KI_CHANT"] * 3 + [FOOL_TOPPER] * 3 + ["THINKING_AHEAD"] * 2,
    "IRONCLAD": ["STRIKE_IRONCLAD"] * 3 + ["DEFEND_IRONCLAD"] * 3 + ["BASH"] + ["HEADBUTT"] * 3 + ["HAVOC"] * 2 + ["CASCADE"] * 2,
}
DEFAULT_SOURCES = {
    "combat": "starter:DEFECT=80,random:DEFECT=80,random:FOOL=50,random:REGENT=50,random:SILENT=30,random:IRONCLAD=30,random:NECROBINDER=30,"
              "drawtop:REGENT=60,drawtop:FOOL=60,drawtop:IRONCLAD=40,boss=60",
    "run": "run:FOOL=16,run:DEFECT=8,run:REGENT=8,run:SILENT=6,run:IRONCLAD=6,run:NECROBINDER=6",
}
DEC_NAMES = {0: "卡住", 1: "战斗", 2: "选牌", 3: "卡牌奖励", 4: "卡包", 5: "奖励", 6: "事件", 7: "休息", 8: "商店", 9: "宝箱", 10: "小游戏", 11: "地图", 12: "结局"}


def parse_sources(s: str) -> list[tuple[str, int]]:
    out = []
    for part in s.split(","):
        part = part.strip()
        if part:
            name, n = part.rsplit("=", 1)
            out.append((name, int(n)))
    return out


def episodes(a) -> list[dict]:
    """全部局：[{id, src, i, reset}]，按来源轮流排开（各分片都分到各种来源）。"""
    lists = []
    items = None
    encs = C.encounters()
    pools = {}
    for src, n in parse_sources(a.sources):
        kind, _, char = src.partition(":")
        lst = []
        for i in range(a.start, a.start + n):
            if kind in ("starter", "random"):
                if kind == "random" and char not in pools:
                    pools[char] = C.load_pools(char)
                spec = C.combat_spec(i, kind, char, encs, pools.get(char))
                reset = {"op": "reset", "mode": "combat", "spec": spec, **C.COMBAT_FLAGS}
            elif kind == "drawtop":
                spec = C.combat_spec(i, "starter", char, encs)
                spec["deck"] = [{"id": c, "upgrades": 1 if random.Random(f"dt{i}{j}").random() < 0.3 else 0} for j, c in enumerate(DRAWTOP[char])]
                reset = {"op": "reset", "mode": "combat", "spec": spec, **C.COMBAT_FLAGS}
            elif kind == "boss":
                items = items or C.boss_items(a.items)
                it = items[i % len(items)]
                reset = {"op": "reset", "mode": "combat", "spec": it["spec"], **C.COMBAT_FLAGS}
            elif kind == "run":
                reset = {"op": "reset", "mode": "run", "character": char, "ascension": 10, "seed": f"CF{i}", **C.RUN_FLAGS}
            else:
                raise SystemExit(f"不认识的来源 {src}")
            if a.mode == "run" or kind == "run" or i % 2 == 0:
                reset["labels"] = (i % 2 == 0)
            lst.append({"id": f"{src}#{i}", "src": src, "i": i, "reset": reset})
        lists.append(lst)
    out = []
    for k in range(max((len(x) for x in lists), default=0)):
        for lst in lists:
            if k < len(lst):
                out.append(lst[k])
    return out


class Pair:
    """复制进程 C 与重放进程 R。各自记下本局发过的请求（复现用）。"""

    def __init__(self, a, tag: str):
        logdir = Path(a.out) / "logs"
        logdir.mkdir(parents=True, exist_ok=True)
        env_c = dict(x.split("=", 1) for x in a.env_c)
        self.c = C.Eng(a.dll_c, open(logdir / f"c_{tag}.log", "ab"), "C", env=env_c)
        self.r = C.Eng(a.dll_r, open(logdir / f"r_{tag}.log", "ab"), "R")
        self.c_log: list[dict] = []
        self.t_restore = []      # C 的 restore 往返（毫秒）
        self.t_replay = []       # R 的 reset + replay 往返（毫秒）

    def creq(self, r: dict) -> bytes | EngineError:
        self.c_log.append(r)
        t = time.perf_counter()
        out = self.c.req(r)
        if r.get("op") == "restore":
            self.t_restore.append((time.perf_counter() - t) * 1000)
        return out

    def both(self, rc: dict, rr: dict) -> tuple:
        """C、R 同时发、再各自收（两个进程并行算）。"""
        self.c_log.append(rc)
        t = time.perf_counter()
        self.c.send(rc)
        self.r.send(rr)
        fc = self.c.recv()
        tc = time.perf_counter()
        fr = self.r.recv()
        tr = time.perf_counter()
        if rc.get("op") == "restore":
            self.t_restore.append((tc - t) * 1000)
        if rr.get("op") == "reset" and rr.get("replay"):
            self.t_replay.append((tr - t) * 1000)
        return fc, fr

    def close(self):
        self.c.close()
        self.r.close()


def cmp(fc, fr, after_replay: bool = False) -> dict | None:
    """C 与 R 的一帧：None 为一致；否则第一处差异。两边同样报错算一致（返回 None），只一边报错或报错不同为差异。
    after_replay：R 的这一帧是 reset + replay 之后的第一帧，容许它只多出重放特有的开战信息、开幕快照（common.replay_extra_only），
    这种帧返回 {"replay_extra": True}（调用方单独计数，不算不同）。"""
    ec, er = isinstance(fc, EngineError), isinstance(fr, EngineError)
    if ec or er:
        if ec and er and str(fc) == str(fr):
            return None
        return {"seg": "报错", "desc": f"C：{str(fc)[:200] if ec else '无'}；R：{str(fr)[:200] if er else '无'}"}
    d = C.first_diff(fc, fr)
    if d is not None and after_replay and C.replay_extra_only(fc, fr):
        return {"replay_extra": True}
    return d


class Episode:
    def __init__(self, P: Pair, ep: dict, a, writer):
        self.P, self.ep, self.a, self.w = P, ep, a, writer
        self.reset = ep["reset"]
        self.rng = random.Random(f"main/{ep['id']}")
        self.forks: list[dict] = []
        self.slots: dict[int, dict] = {}     # 槽位 → {hist, frame, fork}
        self.n_frames = 0

    # -- 一个世界：C、R 各自的起始请求，之后走 K 步
    def world(self, fork: dict, w: int, rc: dict, rr_list: list[dict], first_step: int | None, k: int, tag: str,
              expect: bytes | None = None) -> dict:
        """expect：C 恢复出的第一帧应与它逐字节全同（含头部 12、14），记在 restore_exact。"""
        P = self.P
        r_log = list(rr_list)
        fc, fr = None, None
        # R 可能要多个请求才到起点（嵌套世界：reset … 再单独 reseed、step）；只比最后一个
        for j, rr in enumerate(rr_list):
            if j == len(rr_list) - 1:
                fc, fr = P.both(rc, rr)
            else:
                x = P.r.req(rr)
                if isinstance(x, EngineError):
                    fr = x
        res = {"w": w, "tag": tag, "frames": 0, "diff": None}
        if expect is not None:
            res["restore_exact"] = (not isinstance(fc, EngineError)) and fc == expect
            if not res["restore_exact"] and not isinstance(fc, EngineError) and C.extra_only(fc, expect, ()):
                # 拍快照的那一帧正好是开战（或换幕）的第一帧：服务端在回复它时已把“在战斗中”记下、开幕快照已发出，
                # 快照存的是回复之后的样子，所以恢复后重新编码的帧不再带开战信息、开幕快照；其余逐字节全同。单独计数，不算不同
                res["restore_exact"] = True
                res["restore_extra"] = True
        rng = random.Random(f"w/{self.ep['id']}/{fork['fork']}/{tag}")
        steps = 0
        pending_first = first_step
        while True:
            d = cmp(fc, fr, after_replay=(steps == 0 and len(rr_list) == 1))
            res["frames"] += 1
            if d is not None and d.get("replay_extra"):
                res["replay_extra"] = res.get("replay_extra", 0) + 1
                d = None
            if d is not None:
                res["diff"] = d
                res["diff_step"] = steps
                if not isinstance(fc, EngineError):
                    res["at"] = C.frame_brief(fc)
                self.repro(fork, tag, r_log, d)
                break
            if isinstance(fc, EngineError) or C.is_done(fc) or steps >= k:
                break
            act = pending_first if pending_first is not None else C.pick_bias(rng, fc)
            pending_first = None
            req = self.wf({"op": "step", "action": act})
            r_log.append(req)
            fc, fr = P.both(req, req)
            steps += 1
        res["steps"] = steps
        return res

    def wf(self, req: dict) -> dict:
        """世界里的请求加上 --world-flags：keep_top 加在带 reseed 的请求上，auto_single 加在走一步的请求上
        （世界 0 的 restore 与 R 的 reset + replay 不加：它们要回到分叉点本身那一帧）。C、R 两边加的一样。"""
        fl = self.a.world_flags
        if not fl:
            return req
        req = dict(req)
        if "keep_top" in fl and (req.get("op") == "reseed" or "reseed" in req):
            req["keep_top"] = True
        if "auto_single" in fl and (req.get("op") == "step" or "first" in req):
            req["auto_single"] = True
        return req

    def repro(self, fork: dict, tag: str, r_log: list[dict], d: dict) -> None:
        rd = Path(self.a.out) / "repro"
        rd.mkdir(parents=True, exist_ok=True)
        name = f"{self.ep['id'].replace(':', '_').replace('#', '_')}_f{fork['fork']}_{tag}.json"
        (rd / name).write_text(json.dumps({"episode": self.ep["id"], "fork": fork["fork"], "world": tag, "diff": d,
                                           "dll_c": str(self.a.dll_c), "dll_r": str(self.a.dll_r), "env_c": self.a.env_c,
                                           "c_requests": self.P.c_log, "r_requests": r_log}, ensure_ascii=False), encoding="utf-8")

    def other_game(self) -> dict:
        """他局：C 开另一局（换模式、换角色、换开关），走几步，拍一份别的槽位再丢掉。"""
        rng = self.rng
        if self.reset["mode"] == "combat" or rng.random() < 0.3:
            req = {"op": "reset", "mode": "run", "character": rng.choice(CHARS), "ascension": 10, "seed": f"OTHER{rng.randrange(10**6)}",
                   **({} if rng.random() < 0.5 else C.RUN_FLAGS), "labels": rng.random() < 0.5}
        else:
            spec = C.combat_spec(rng.randrange(10**4), "starter", rng.choice(CHARS))
            req = {"op": "reset", "mode": "combat", "spec": spec, "privileged": rng.random() < 0.5, "labels": rng.random() < 0.5,
                   **({} if rng.random() < 0.5 else {"obs_format": 27})}
        f = self.P.creq(req)
        n = rng.randrange(0, 9)
        for _ in range(n):
            if isinstance(f, EngineError) or C.is_done(f):
                break
            f = self.P.creq({"op": "step", "action": C.pick_bias(rng, f)})
        if not isinstance(f, EngineError) and not C.is_done(f):
            self.P.creq({"op": "snapshot", "slot": 77})
            self.P.creq({"op": "drop", "slot": 77})
        return {"mode": req["mode"], "steps": n}

    def fork(self, slot: int, hist: list[int], frame: bytes, method: str) -> None:
        P, a, rng = self.P, self.a, self.rng
        fork = {"ep": self.ep["id"], "src": self.ep["src"], "fork": len(self.forks), "slot": slot, "step": len(hist), "method": method,
                "dec": C.decision(frame), "combat": C.in_combat(frame), "drawtop": C.draw_top_cards(frame), "n": C.num_actions(frame),
                "worlds": [], "frames": 0, "diffs": 0}
        self.forks.append(fork)
        self.slots[slot] = {"hist": list(hist), "frame": frame, "fork": fork["fork"]}
        if rng.random() < a.p_other:
            fork["other"] = self.other_game()
        n = C.num_actions(frame)
        order = list(range(n))
        rng.shuffle(order)
        firsts = [order[j % n] for j in range(4)]
        base = {**self.reset, "replay": list(hist)}
        nested = rng.random() < a.p_nested
        for w in range(4):
            if w == 0:
                # 不带 reseed：restore 后首步单独 step；恢复的帧要与拍快照时的帧逐字节全同（含头部 12、14）
                rc = {"op": "restore", "slot": slot}
                wr = self.world(fork, 0, rc, [base], firsts[0], a.k, "w0", expect=frame)
                fork["worlds"].append(wr)
                continue
            s = rng.getrandbits(32)
            rc = self.wf({"op": "restore", "slot": slot, "reseed": s, "first": firsts[w]})
            if w == 3 and nested:
                rc["snapshot"] = 1000 + slot
            rr = self.wf({**base, "reseed": s, "first": firsts[w]})
            wr = self.world(fork, w, rc, [rr], None, a.k, f"w{w}")
            fork["worlds"].append(wr)
            if w == 3 and nested and wr["diff"] is None:
                # 嵌套：从“reseed + first 之后”的槽位再分一个子世界。R：reset + replay + reseed + first，再单独 reseed、再 step
                s2 = rng.getrandbits(32)
                st = self.c_restore_probe(1000 + slot)
                if st is not None and not C.is_done(st):
                    b = rng.randrange(C.num_actions(st))
                    rc2 = self.wf({"op": "restore", "slot": 1000 + slot, "reseed": s2, "first": b})
                    rr2 = [rr, self.wf({"op": "reseed", "seed": s2}), self.wf({"op": "step", "action": b})]
                    wn = self.world(fork, 4, rc2, rr2, None, max(4, a.k // 2), "nested")
                    fork["worlds"].append(wn)
                    fork["nested"] = True
                P.creq({"op": "drop", "slot": 1000 + slot})
        # 主线回到分叉点：恢复的帧也要与拍快照时全同
        g = P.creq({"op": "restore", "slot": slot})
        same_g = (not isinstance(g, EngineError)) and (g == frame or C.extra_only(g, frame, ()))
        if same_g and g != frame:
            fork["restore_extra"] = True        # 同世界 0 的 restore_extra：恢复后不重发开战信息、开幕快照
        fork["restore_exact"] = same_g and bool(fork["worlds"][0].get("restore_exact"))
        for wr in fork["worlds"]:
            fork["frames"] += wr["frames"]
            fork["diffs"] += int(wr["diff"] is not None)
        if not fork["restore_exact"]:
            fork["diffs"] += 1
            fork["restore_diff"] = C.first_diff(g, frame, ()) if not isinstance(g, EngineError) else {"desc": str(g)[:200]}
        self.n_frames += fork["frames"]
        return g

    def c_restore_probe(self, slot: int):
        """恢复嵌套槽位（不带 reseed）取它的帧，只用来决定子世界的首步。"""
        f = self.P.creq({"op": "restore", "slot": slot})
        return None if isinstance(f, EngineError) else f

    def run(self) -> dict:
        P, a, rng = self.P, self.a, self.rng
        P.c_log = []
        is_run = self.reset["mode"] == "run"
        max_steps = a.max_steps or (1200 if is_run else 300)
        hist: list[int] = []
        inline = rng.random() < a.p_inline
        slot_next = 0
        req = dict(self.reset)
        if inline:
            req["snapshot"] = slot_next
        f = P.creq(req)
        res = {"id": self.ep["id"], "src": self.ep["src"], "steps": 0, "error": None}
        while True:
            if isinstance(f, EngineError):
                res["error"] = str(f)[:200]
                break
            if C.is_done(f):
                break
            want = False
            method = ""
            if len(self.forks) < a.max_forks:
                if inline:
                    want, method = True, ("reset" if not hist else "step")
                else:
                    p = a.p_fork
                    if C.draw_top_cards(f) > 0:
                        p = a.p_drawtop
                    elif is_run and not C.in_combat(f):
                        p = a.p_noncombat
                    if rng.random() < p:
                        want, method = True, "op"
                        rep = P.c.req({"op": "snapshot", "slot": slot_next})
                        P.c_log.append({"op": "snapshot", "slot": slot_next})
                        if isinstance(rep, EngineError):
                            res["error"] = f"拍快照报错：{str(rep)[:160]}"
                            break
                        info = json.loads(rep.decode("utf-8"))
                        res.setdefault("objects", []).append(info.get("objects"))
            elif inline:
                # 超过分叉上限、却已顺手存了快照：照样不分叉
                pass
            if want:
                g = self.fork(slot_next, hist, f, method)
                slot_next = (slot_next + 1) % 4
                if isinstance(g, EngineError):
                    res["error"] = f"主线恢复报错：{str(g)[:160]}"
                    break
                f = g
                self.w(self.forks[-1])
            if len(hist) >= max_steps:
                break
            act = C.pick_bias(rng, f)
            inline = len(self.forks) < a.max_forks and rng.random() < a.p_inline
            req = {"op": "step", "action": act}
            if inline:
                req["snapshot"] = slot_next
                # 这一步顺手存的快照会覆盖槽位 slot_next；原来记着的世界不能再按它恢复（v0.114.0 验收时 boss#10 主线最后一步覆盖了槽位 0，
                # “旧槽位”世界按第 0 个分叉点去恢复、拿到终局帧，报了 1 个假的不同）
                self.slots.pop(slot_next, None)
            hist.append(act)
            f = P.creq(req)
        res["steps"] = len(hist)
        # 终局核对：C 的主线最后一帧与 R 的 reset + replay 全部动作
        if not isinstance(f, EngineError):
            fr = P.r.req({**self.reset, "replay": list(hist)})
            d = cmp(f, fr, after_replay=True)
            if d is not None and d.get("replay_extra"):
                res["final_replay_extra"] = True
                d = None
            res["final_same"] = d is None
            if d is not None:
                res["final_diff"] = d
                self.repro({"fork": "final"}, "final", [{**self.reset, "replay": list(hist)}], d)
        # 旧槽位：从这一局较早的分叉点再分一个世界
        old = [s for s, v in self.slots.items() if v["fork"] < len(self.forks) - 1]
        if old and rng.random() < a.p_old:
            s_ = rng.choice(old)
            v = self.slots[s_]
            fork = {"ep": self.ep["id"], "src": self.ep["src"], "fork": f"old{v['fork']}", "slot": s_, "step": len(v["hist"]), "method": "old",
                    "dec": C.decision(v["frame"]), "combat": C.in_combat(v["frame"]), "drawtop": C.draw_top_cards(v["frame"]),
                    "n": C.num_actions(v["frame"]), "worlds": [], "frames": 0, "diffs": 0, "old": True}
            sd = rng.getrandbits(32)
            b = rng.randrange(C.num_actions(v["frame"]))
            wr = self.world(fork, 0, self.wf({"op": "restore", "slot": s_, "reseed": sd, "first": b}),
                            [self.wf({**self.reset, "replay": v["hist"], "reseed": sd, "first": b})], None, a.k, "old")
            fork["worlds"].append(wr)
            fork["frames"], fork["diffs"] = wr["frames"], int(wr["diff"] is not None)
            fork["restore_exact"] = True
            self.n_frames += wr["frames"]
            self.w(fork)
            res["old_slot"] = True
        rep = P.c.req({"op": "drop"})
        P.c_log.append({"op": "drop"})
        if not isinstance(rep, EngineError):
            dj = json.loads(rep.decode("utf-8"))
            res["drop_ok"] = dj.get("slots") == 0
        res["forks"] = len(self.forks)
        res["frames"] = self.n_frames
        return res


def run_shard(job) -> None:
    k, eps, a = job
    C.be_nice(a.nice)
    out = Path(a.out)
    done = set()
    ep_path = out / f"episodes_s{k}.jsonl"
    if ep_path.exists():
        for l in open(ep_path, encoding="utf-8"):
            if l.strip():
                done.add(json.loads(l)["id"])
    todo = [e for e in eps if e["id"] not in done]
    if not todo:
        return
    P = Pair(a, f"s{k}")
    fh = open(out / f"forks_s{k}.jsonl", "a", encoding="utf-8")
    eh = open(ep_path, "a", encoding="utf-8")

    def writer(fork):
        fh.write(json.dumps(fork, ensure_ascii=False) + "\n")
        fh.flush()

    t0 = time.time()
    try:
        for j, ep in enumerate(todo):
            r = Episode(P, ep, a, writer).run()
            r["restore_ms"] = round(sum(P.t_restore) / max(1, len(P.t_restore)), 2)
            r["replay_ms"] = round(sum(P.t_replay) / max(1, len(P.t_replay)), 2)
            r["n_restore"], r["n_replay"] = len(P.t_restore), len(P.t_replay)
            P.t_restore, P.t_replay = [], []
            eh.write(json.dumps(r, ensure_ascii=False) + "\n")
            eh.flush()
            if j % 20 == 0:
                print(f"{time.strftime('%H:%M:%S')} 分片 {k}：{j + 1}/{len(todo)} 局，{time.time() - t0:.0f} 秒", flush=True)
    finally:
        P.close()


def run(a) -> None:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if not a.sources:
        a.sources = DEFAULT_SOURCES[a.mode]
    a.world_flags = [x for x in a.world_flags.split(",") if x]
    if a.max_forks <= 0:
        a.max_forks = 12 if a.mode == "run" else 6
    if a.p_fork < 0:
        a.p_fork = 0.03 if a.mode == "run" else 0.12
    eps = episodes(a)
    (out / "config.json").write_text(json.dumps({"mode": a.mode, "sources": a.sources, "k": a.k, "world_flags": a.world_flags, "dll_c": C.dll_info(a.dll_c),
                                                  "dll_r": C.dll_info(a.dll_r), "env_c": a.env_c, "episodes": len(eps),
                                                  "time": time.strftime("%F %T")}, ensure_ascii=False, indent=1), encoding="utf-8")
    w = max(1, min(a.workers, 2))
    jobs = [(k, eps[k::w], a) for k in range(w)]
    print(f"{len(eps)} 局，{w} 个分片；C = {C.dll_info(a.dll_c)} {a.env_c or ''}；R = {C.dll_info(a.dll_r)}", flush=True)
    t0 = time.time()
    if w == 1:
        run_shard(jobs[0])
    else:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(w) as ex:
            list(ex.map(run_shard, jobs))
    print(f"用时 {time.time() - t0:.0f} 秒", flush=True)
    summary(a)


def summary(a) -> None:
    out = Path(a.out)
    forks = [json.loads(l) for f in sorted(out.glob("forks_s*.jsonl")) for l in open(f, encoding="utf-8") if l.strip()]
    eps = [json.loads(l) for f in sorted(out.glob("episodes_s*.jsonl")) for l in open(f, encoding="utf-8") if l.strip()]
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8")) if (out / "config.json").exists() else {}
    main_forks = [f for f in forks if not f.get("old")]
    worlds = [w for f in forks for w in f["worlds"]]
    frames = sum(w["frames"] for w in worlds)
    wdiff = [w for w in worlds if w["diff"] is not None]
    rx = sum(1 for f in main_forks if not f.get("restore_exact"))
    fin = [e for e in eps if "final_same" in e]
    print(f"目录 {out}：模式 {cfg.get('mode')}，C = {cfg.get('dll_c')} {cfg.get('env_c') or ''}")
    print(f"局 {len(eps)}（报错停下 {sum(1 for e in eps if e.get('error'))}），分叉点 {len(main_forks)}（另旧槽位 {len(forks) - len(main_forks)}），"
          f"世界 {len(worlds)}，帧 {frames}；不同的世界 {len(wdiff)}，主线恢复帧不全同 {rx}，终局核对不同 {sum(1 for e in fin if not e['final_same'])}/{len(fin)}")
    by = Counter()
    byd = Counter()
    for f in main_forks:
        by[DEC_NAMES.get(f["dec"], str(f["dec"])) + ("（战斗中）" if f["combat"] and f["dec"] != 1 else "")] += 1
        if f["diffs"]:
            byd[DEC_NAMES.get(f["dec"], str(f["dec"]))] += 1
    print("分叉点按决策类型：", dict(by.most_common()))
    rx_extra = sum(w.get("replay_extra", 0) for w in worlds) + sum(1 for e in eps if e.get("final_replay_extra"))
    if rx_extra:
        print(f"重放特有的附加信息（R 的第一帧多出开战信息或开幕快照，其余全同；不算不同，见 common.replay_extra_only）：{rx_extra} 帧")
    rs_extra = sum(1 for f in main_forks if f.get("restore_extra"))
    if rs_extra:
        print(f"在开战（换幕）的第一帧拍的快照，恢复后不重发开战信息、开幕快照（其余全同，含头部 12、14；不算不同）：{rs_extra} 个分叉点")
    print(f"已知牌堆顶非空的分叉点 {sum(1 for f in main_forks if f['drawtop'] > 0)}（其中有不同 {sum(1 for f in main_forks if f['drawtop'] > 0 and f['diffs'])}）；"
          f"他局 {sum(1 for f in main_forks if f.get('other'))}，嵌套 {sum(1 for f in main_forks if f.get('nested'))}，"
          f"快照方式 {dict(Counter(f['method'] for f in main_forks))}")
    srcs = Counter(f["src"] for f in main_forks)
    sd = Counter(f["src"] for f in main_forks if f["diffs"])
    print("按来源（分叉点/有不同）：", {s: f"{n}/{sd.get(s, 0)}" for s, n in srcs.items()})
    if byd:
        print("有不同的分叉点按决策类型：", dict(byd))
    cats = Counter(C.diff_key(w["diff"]) if w["diff"].get("seg") != "报错" else "报错" for w in wdiff)
    if cats:
        print("世界第一处差异的分类：", dict(cats.most_common(15)))
    shown = 0
    for f in forks:
        for w in f["worlds"]:
            if w["diff"] is not None and shown < a.show:
                shown += 1
                print(f"  {f['ep']} 分叉 {f['fork']}（第 {f['step']} 步，{DEC_NAMES.get(f['dec'])}，已知顶 {f['drawtop']}）世界 {w['tag']} 第 {w.get('diff_step')} 步："
                      f"{w['diff']['desc']}")
    for f in [f for f in main_forks if not f.get("restore_exact")][:5]:
        print(f"  主线恢复帧不全同：{f['ep']} 分叉 {f['fork']}：{(f.get('restore_diff') or {}).get('desc')}")
    rms = [e["restore_ms"] for e in eps if e.get("n_restore")]
    pms = [e["replay_ms"] for e in eps if e.get("n_replay")]
    if rms:
        print(f"往返耗时（局平均的平均）：C restore {sum(rms) / len(rms):.1f} ms，R reset + replay {sum(pms) / max(1, len(pms)):.1f} ms（含编码，两进程同时算）")
    obj = [x for e in eps for x in e.get("objects", []) if x]
    if obj:
        print(f"快照对象数：中位 {sorted(obj)[len(obj) // 2]}，最大 {max(obj)}")
    late, check = set(), set()
    for lf in sorted((out / "logs").glob("c_*.log")) if (out / "logs").exists() else []:
        for l in open(lf, encoding="utf-8", errors="replace"):
            if l.startswith("[clone] 类型 "):
                t = l[len("[clone] 类型 "):].split(" ", 1)[0]
                (check if "要核查" in l else late).add(t)
    if late or check:
        print(f"C 的日志：拍快照之后才初始化的类型 {len(late) + len(check)} 个，其中有可写静态字段、引擎标“要核查”的 {len(check)} 个：")
        for t in sorted(check):
            print("    " + t)
    s = {"episodes": len(eps), "forks": len(main_forks), "old_forks": len(forks) - len(main_forks), "worlds": len(worlds), "frames": frames,
         "diff_worlds": len(wdiff), "restore_not_exact": rx, "final_diff": sum(1 for e in fin if not e["final_same"]),
         "drawtop_forks": sum(1 for f in main_forks if f["drawtop"] > 0), "by_decision": dict(by), "categories": dict(cats),
         "replay_extra_frames": rx_extra, "restore_extra_forks": rs_extra}
    (out / "summary.json").write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding="utf-8")


def repro(a) -> None:
    """照复现文件重放：新开 C、R 两个进程，各自发请求序列，比最后一帧并打印第一处差异。"""
    d = json.loads(Path(a.file).read_text(encoding="utf-8"))
    env_c = dict(x.split("=", 1) for x in d.get("env_c", []))
    c = C.Eng(a.dll_c or d["dll_c"], None, "C", env=env_c)
    r = C.Eng(a.dll_r or d["dll_r"], None, "R")
    fc = fr = None
    for q in d["c_requests"]:
        fc = c.req(q)
    for q in d["r_requests"]:
        fr = r.req(q)
    print(f"C {len(d['c_requests'])} 个请求、R {len(d['r_requests'])} 个请求；记录的差异：{d['diff'].get('desc')}")
    dd = cmp(fc, fr)
    print("重放后：", "一致" if dd is None else dd["desc"])
    if dd is not None and not isinstance(fc, EngineError):
        print("C 帧概况", C.frame_brief(fc), "R 帧概况", C.frame_brief(fr) if not isinstance(fr, EngineError) else fr)
    c.close(); r.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--mode", choices=["combat", "run"], default="combat")
    r.add_argument("--sources", default="", help="名字:角色=局数,…（默认见 DEFAULT_SOURCES）")
    r.add_argument("--start", type=int, default=0, help="每个来源的局序号起点")
    r.add_argument("--items", default=str(ITEMS))
    r.add_argument("--dll-c", default=str(C.DLL_CLONE))
    r.add_argument("--dll-r", default=str(C.DLL_CLONE))
    r.add_argument("--env-c", action="append", default=[], help="复制进程额外的环境变量（KEY=VALUE，可多次），反向核对用")
    r.add_argument("--k", type=int, default=12, help="每个世界首步之后再走几步")
    r.add_argument("--world-flags", default="", help="世界里的请求带的可选开关（逗号分隔：keep_top、auto_single），C、R 两边一样")
    r.add_argument("--max-steps", type=int, default=0)
    r.add_argument("--max-forks", type=int, default=0, help="每局最多几个分叉点（默认战斗 6、整局 12）")
    r.add_argument("--p-fork", type=float, default=-1, help="主线每一帧分叉的概率（整局模式为战斗里的帧；默认战斗 0.12、整局 0.03）")
    r.add_argument("--p-noncombat", type=float, default=0.35, help="整局模式战斗外的帧分叉的概率")
    r.add_argument("--p-drawtop", type=float, default=0.5, help="已知牌堆顶非空的帧分叉的概率")
    r.add_argument("--p-inline", type=float, default=0.03, help="主线 step（第 0 步为 reset）顺手带 snapshot 的概率（带了就在那一帧分叉）")
    r.add_argument("--p-other", type=float, default=0.15, help="分叉点先开别的局再恢复旧槽位的概率")
    r.add_argument("--p-nested", type=float, default=0.15, help="世界 3 的 restore 带 snapshot、再分嵌套子世界的概率")
    r.add_argument("--p-old", type=float, default=0.5, help="局末从较早的槽位再分一个世界的概率")
    r.add_argument("--workers", type=int, default=2)
    r.add_argument("--nice", type=int, default=10)
    r.add_argument("--out", required=True)
    r.add_argument("--show", type=int, default=12)
    s = sub.add_parser("summary")
    s.add_argument("--out", required=True)
    s.add_argument("--show", type=int, default=12)
    p = sub.add_parser("repro")
    p.add_argument("file")
    p.add_argument("--dll-c", default="")
    p.add_argument("--dll-r", default="")
    a = ap.parse_args()
    {"run": run, "summary": summary, "repro": repro}[a.cmd](a)


if __name__ == "__main__":
    main()
