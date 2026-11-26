"""局外分枝公平重设（engine Branch.Apply，决策记录 258）的检验。在装了新引擎的目录里跑：
    .venv/bin/python -I tools/branch_check.py --ckpt 检查点.pt --seeds 40 --out 结果.json

1. 看得到的不变：贪心打到第一幕的一个局外决策（选卡或选路线），重放到那里，分枝前后的观测逐字节相同；当前这一幕的 Boss、
   已显示的地图在存档里不变。
2. 确定：同一个 salt 做两次，整局存档逐字相同；接着贪心打完，每一步动作、结局都相同。
3. 换掉了未来：不同 salt、以及不分枝（真实的未来），后几幕的 Boss 与遭遇、当前幕没走到的遭遇与事件、玩家随机数计数互不相同。
4. 分枝出去的局正常：同一批种子在第一个决策（涅奥）处分枝与不分枝各贪心打完，比较卡住数、通关率、平均层数。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

from sts2env.env import Sts2Env as Env, VecEnv  # noqa: E402
from sts2env.protocol import parse  # noqa: E402
from sts2env.protocol import DecisionKind  # noqa: E402
from sts2rl.actor import MODE_RUN  # noqa: E402
from sts2rl.features import Ragged, collate_ragged  # noqa: E402
from sts2rl.model import ModelConfig, PolicyValueNet, load_state  # noqa: E402

RESET = {"op": "reset", "mode": "run", "character": "DEFECT", "ascension": 10, "route_summary": True}


def raw(env: Env, req: dict) -> bytes:
    env._proc.send(req)
    return env._proc.recv()


TIME_KEYS = ("run_time", "save_time", "start_time", "win_time")


def save(env: Env) -> dict:
    """整局存档，去掉时间戳（每次都不同，不是局面的一部分）。"""
    sv = json.loads(raw(env, {"op": "save"}))["save"]
    for k in TIME_KEYS:
        sv.pop(k, None)
    return sv


def same_obs(*bs: bytes) -> bool:
    """观测相同：头部第 12 位是引擎进程累计赢的战斗场数（UiFlows.CombatsWon，开新局不清零、模型不用），比较时去掉。"""
    def norm(b):
        o = parse(b); h = o.header.copy(); h[12] = 0
        return [h.tobytes()] + [getattr(o, k).tobytes() for k in ("tok_ints", "tok_feats", "tok_priv", "cand_ints", "cand_feats") if getattr(o, k, None) is not None]
    first = norm(bs[0])
    return all(norm(b) == first for b in bs[1:])


def greedy(net, device, o) -> int:
    with torch.no_grad():
        lg = net(collate_ragged(Ragged.from_obs([o], np.full(1, MODE_RUN))).to(device))[0][0, :o.num_actions]
    return int(lg.argmax())


def play_out(env: Env, net, device, o, cap=6000) -> tuple[list[int], bool, int, bool]:
    acts = []
    while not (o.done or o.stuck) and len(acts) < cap:
        a = greedy(net, device, o); acts.append(a); o = parse(raw(env, {"op": "step", "action": a}))
    return acts, bool(o.done and o.victory), int(o.floor), bool(o.stuck)


def diff_keys(a, b, path="", out=None, depth=0):
    out = [] if out is None else out
    if type(a) is not type(b):
        out.append(path); return out
    if isinstance(a, dict) and depth < 4:
        for k in sorted(set(a) | set(b)):
            if a.get(k) != b.get(k):
                diff_keys(a.get(k), b.get(k), f"{path}.{k}", out, depth + 1)
    elif isinstance(a, list) and depth < 4 and len(a) == len(b) and all(isinstance(x, dict) for x in a):
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                diff_keys(x, y, f"{path}[{i}]", out, depth + 1)
    elif a != b:
        out.append(path)
    return out


def rooms(sv: dict) -> list:
    return [a.get("rooms") for a in sv.get("acts", [])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--seeds", type=int, default=40); ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--envs", type=int, default=24)
    ap.add_argument("--dll", default=None, help="引擎程序集路径（默认仓库里的 Release 输出）")
    ap.add_argument("--character", default="DEFECT", help="角色（猎人决策记录第 19 条起可换，默认故障机器人）")
    ap.add_argument("--seed-list", default="BRCHK01,BRCHK02,BRCHK03,BRCHK04,BRCHK05,BRCHK06", help="逐项检验用的种子")
    a = ap.parse_args()
    RESET["character"] = a.character
    blob = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    net = PolicyValueNet(ModelConfig(**blob["config"]["model"])).eval().to(a.device)
    load_state(net, blob["state"])
    rep = {"checks": [], "fails": 0}

    def check(name, ok, detail=""):
        rep["checks"].append({"name": name, "ok": bool(ok), "detail": detail}); rep["fails"] += 0 if ok else 1
        print(("通过 " if ok else "失败 ") + name + (f"：{detail}" if detail else ""), flush=True)

    dkw = {"dll": Path(a.dll)} if a.dll else {}
    env = Env(**dkw)
    for seed in a.seed_list.split(","):
        # 贪心打到第一幕第 6 层以后的第一个选卡或选路线决策，记下动作带
        o = parse(raw(env, {**RESET, "seed": seed})); tape = []
        while not (o.done or o.stuck):
            if o.floor >= 6 and o.decision in (DecisionKind.CARD_REWARD, DecisionKind.MAP):
                break
            act = greedy(net, a.device, o); tape.append(act); o = parse(raw(env, {"op": "step", "action": act}))
        if o.done or o.stuck:
            check(f"{seed} 打到第一幕局外决策", False, "没到就结束了"); continue
        kind = DecisionKind(o.decision).name
        b0 = raw(env, {**RESET, "seed": seed, "replay": tape}); s0 = save(env)
        b1 = raw(env, {**RESET, "seed": seed, "replay": tape, "branch": 7}); s1 = save(env)
        b1b = raw(env, {**RESET, "seed": seed, "replay": tape, "branch": 7}); s1b = save(env)
        b2 = raw(env, {**RESET, "seed": seed, "replay": tape, "branch": 8}); s2 = save(env)
        check(f"{seed} 第 {o.floor} 层 {kind}：分枝前后观测相同（不计进程累计胜场）", same_obs(b0, b1, b2), f"{len(b0)} 字节")
        check(f"{seed} 同一 salt 两次存档相同", s1 == s1b)
        ca = s0.get("current_act_index", 0)
        r0, r1, r2 = rooms(s0), rooms(s1), rooms(s2)
        same_boss = all(r0[ca].get(k) == r1[ca].get(k) for k in ("boss_id", "ancient_id", "second_boss_id") if k in r0[ca])
        check(f"{seed} 当前幕的 Boss、先古之民不变", same_boss, json.dumps({k: r0[ca].get(k) for k in r0[ca] if "boss" in k or "ancient" in k}, ensure_ascii=False)[:200])
        check(f"{seed} 当前幕的地图与走过的路线不变", s0["acts"][ca].get("saved_map") == s1["acts"][ca].get("saved_map")
              and s0.get("map_point_history") == s1.get("map_point_history") and s0.get("visited_map_coords") == s1.get("visited_map_coords"))
        fut0 = [r0[i] for i in range(ca + 1, len(r0))]; fut1 = [r1[i] for i in range(ca + 1, len(r1))]; fut2 = [r2[i] for i in range(ca + 1, len(r2))]
        check(f"{seed} 后几幕房间：分枝与真实不同、两个 salt 互不相同", fut0 != fut1 and fut1 != fut2 and fut0 != fut2)
        d01 = diff_keys(s0, s1)
        rep.setdefault("save_diff", {})[seed] = d01
        print(f"  {seed} 分枝后存档里变了的项（前 30）：{d01[:30]}", flush=True)
        # 确定性：同一 salt 分枝后贪心打完，动作与结局相同
        parse(raw(env, {**RESET, "seed": seed, "replay": tape, "branch": 7}))
        ob = parse(b1); pa = play_out(env, net, a.device, ob)
        parse(raw(env, {**RESET, "seed": seed, "replay": tape, "branch": 7}))
        pb = play_out(env, net, a.device, parse(b1b))
        check(f"{seed} 同一 salt 打完：动作序列与结局相同", pa == pb, f"{len(pa[0])} 步，{'通关' if pa[1] else '第 ' + str(pa[2]) + ' 层输'}")
    # 幕初重抽遭遇守规矩（2026-11-03 补，另一会话的意见）：第一幕开头的选路线点上各打 20 个 salt，重抽后
    #   弱怪位（原列表里带 _WEAK 的那些位置）仍是弱怪、其余仍是常规战，与原列表的弱怪场数一致；没走到的部分没有相邻两场是同一场；
    #   本轮已抽的不再出现（本幕弱怪位一共不重复）
    bad = []
    for seed in a.seed_list.split(","):
        o = parse(raw(env, {**RESET, "seed": seed})); tape = []
        while not (o.done or o.stuck) and not (o.decision == DecisionKind.MAP and o.floor >= 1):
            act = greedy(net, a.device, o); tape.append(act); o = parse(raw(env, {"op": "step", "action": act}))
        raw(env, {**RESET, "seed": seed, "replay": tape}); s0 = save(env); ca = s0["current_act_index"]
        n0 = s0["acts"][ca]["rooms"]["normal_encounter_ids"]; v0 = s0["acts"][ca]["rooms"]["normal_encounters_visited"]
        weak = [x.endswith("_WEAK") for x in n0]
        for salt in range(1, 21):
            raw(env, {**RESET, "seed": seed, "replay": tape, "branch": salt}); n1 = save(env)["acts"][ca]["rooms"]["normal_encounter_ids"]
            ok = len(n1) == len(n0) and n1[:v0] == n0[:v0] and [x.endswith("_WEAK") for x in n1] == weak \
                and all(n1[i] != n1[i - 1] for i in range(max(1, v0), len(n1))) \
                and len(set(x for x, w in zip(n1, weak) if w)) == sum(weak)
            if not ok:
                bad.append((seed, salt, v0, n0[:5], n1[:5]))
    check("幕初重抽：弱怪位仍是弱怪、场数一致、没有相邻重复、本幕弱怪不重复（6 个种子 × 20 个 salt）", not bad, str(bad[:2]))
    env.close()

    # 分枝出去的局正常：同一批种子，开局第一个决策（涅奥）处分枝 vs 不分枝，各贪心打完（--seeds 0 时跳过）
    seeds = [f"BRV{j:04d}" for j in range(a.seeds)]
    if not seeds:
        Path(a.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"共 {len(rep['checks'])} 项，失败 {rep['fails']} 项", flush=True)
        return
    res = {}
    for arm in ("plain", "branch"):
        venv = VecEnv(min(a.envs, len(seeds)), **dkw); it = iter(seeds); slot = {}; obs = {}; out = []
        def start(i):
            s = next(it, None)
            if s is None:
                slot.pop(i, None); return
            slot[i] = [s, 0]
            venv.send(i, {**RESET, "seed": s, **({"branch": 1000 + int(s[3:])} if arm == "branch" else {})})
        for i in range(len(venv)):
            start(i)
        while slot:
            for i, o in venv.recv_ready(timeout=1.0):
                if i not in slot:
                    continue
                if isinstance(o, Exception) or o.done or o.stuck or slot[i][1] > 6000:
                    out.append({"seed": slot[i][0], "win": (not isinstance(o, Exception)) and bool(o.done and o.victory),
                                "floor": int(getattr(o, "floor", 0)), "stuck": isinstance(o, Exception) or bool(getattr(o, "stuck", False)) or slot[i][1] > 6000})
                    start(i); continue
                slot[i][1] += 1; obs[i] = o
            if obs:
                ids = sorted(obs)
                with torch.no_grad():
                    lg = net(collate_ragged(Ragged.from_obs([obs[i] for i in ids], np.full(len(ids), MODE_RUN))).to(a.device))[0]
                for k, i in enumerate(ids):
                    venv.send_step(i, int(lg[k, :obs[i].num_actions].argmax()))
                obs.clear()
        res[arm] = out
        print(f"{arm}：{len(out)} 局，通关 {np.mean([r['win'] for r in out]):.1%}，平均层数 {np.mean([r['floor'] for r in out]):.2f}，卡住 {sum(r['stuck'] for r in out)}", flush=True)
    pw, bw = np.mean([r["win"] for r in res["plain"]]), np.mean([r["win"] for r in res["branch"]])
    stuck = sum(r["stuck"] for r in res["branch"])
    check(f"分枝出去的局正常：卡住 {stuck} 局，通关 {bw:.1%}（不分枝 {pw:.1%}）", stuck == 0 and abs(bw - pw) < 0.15)
    rep["rollouts"] = {k: {"n": len(v), "win": float(np.mean([r['win'] for r in v])), "floor": float(np.mean([r['floor'] for r in v])),
                           "stuck": int(sum(r['stuck'] for r in v))} for k, v in res.items()}
    Path(a.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"共 {len(rep['checks'])} 项，失败 {rep['fails']} 项", flush=True)


if __name__ == "__main__":
    main()
