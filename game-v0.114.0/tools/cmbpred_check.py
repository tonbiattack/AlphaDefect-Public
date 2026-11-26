"""战斗结果预测检验（2026-11-11，用户：“能否把战斗从实际打一遍变成某种预测，整局训练里直接跳过战斗”）。

要回答的是：现有的战斗短程辅助头（决策记录 261，aux：本场输赢、到本场结束还要掉的血 / 最大生命）在开战第一帧的预测，
够不够准、能不能代替真打。只读训练（固定一份权重拷贝、在单独目录里跑真实引擎），不改训练、配置、引擎。

子命令：
  runs  真打整局，出手与训练采集相同（按策略分布采样，带特权特征，同样的截断开关）。每场战斗记开战第一帧的辅助头预测、
        战斗价值、实际输赢与掉血，以及开战前的玩家快照（只收选路线后紧接着开打的那种，与练习池同一口径）。
  cf    从 runs 记下的快照里挑局面，造变体（改牌组、改血量），每个变体用同一组种子（共同随机数）按快照各打 K 场单场战斗，
        记每场开战第一帧的预测与实际结果。比较“变体减原样”的真实变化与预测变化。
  report 汇总两者。

在 B 上跑（借用部署目录的代码与引擎，不改它）：
  cd /root/cmbpred && nice -n 15 /root/silent-deploy/sls2/.venv/bin/python -I cmbpred_check.py runs --runs 300 --envs 16 --out runs.jsonl
"""
from __future__ import annotations

import argparse
import copy
import gzip
import json
import os
import random
import sys
import time
from pathlib import Path

DEPLOY = os.environ.get("CMB_DEPLOY", "/root/silent-deploy/sls2")
sys.path.insert(0, os.path.join(DEPLOY, "python"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import base64  # noqa: E402

import sts2env.env as _envmod  # noqa: E402
from sts2env.env import EngineError, VecEnv  # noqa: E402
from sts2env.protocol import parse as _parse0  # noqa: E402
from sts2rl.actor import MODE_COMBAT, MODE_RUN, load_weights, potions_held  # noqa: E402
from sts2rl.config import TrainConfig  # noqa: E402
from sts2rl.features import Ragged, collate_ragged  # noqa: E402
from sts2rl.model import PolicyValueNet  # noqa: E402


def _parse_keep(buf: bytes):
    """解析观测帧时留下原始字节（存开战帧给战斗结果模型训练用）。VecEnv 在模块里按名字调 parse，替换后各处都生效。"""
    o = _parse0(buf)
    o.raw = buf
    return o


_envmod.parse = _parse_keep


def b64(o) -> str:
    return base64.b64encode(o.raw).decode("ascii")


def player_snap(env, i: int) -> str | None:
    """同步向第 i 个引擎要当前玩家快照（引擎 player 操作，不走一步、不编码观测）；只在这个引擎没有在途请求时调用。"""
    p = env.envs[i]._proc
    try:
        p.send({"op": "player"})
        r = json.loads(p.recv().decode("utf-8"))
        s = r.get("snapshot")
        return s if isinstance(s, str) else json.dumps(s)
    except Exception:
        return None


def open_out(path: str):
    return gzip.open(path, "at", encoding="utf-8") if path.endswith(".gz") else open(path, "a", encoding="utf-8")


def read_lines(path: str):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue        # 进程被停时写了半行


DEV = "cpu"


def load(cfg_path: str, weights: str, threads: int, device: str = "cpu"):
    global DEV
    torch.set_num_threads(threads)
    DEV = device
    rd = Path("/root/cmbpred/rd"); rd.mkdir(parents=True, exist_ok=True)
    cfg = TrainConfig.from_args(["--config", cfg_path, "--run-dir", str(rd)])
    net = PolicyValueNet(cfg.model).eval()
    ver, _, _ = load_weights(net, weights, -1)
    if ver < 0:
        raise SystemExit(f"权重读不出来：{weights}")
    return cfg, net.to(device), ver


def infer(net, obs: list, modes: list):
    b = collate_ragged(Ragged.from_obs(obs, np.array(modes))).to(DEV)
    with torch.no_grad():
        logits, value, _, extra = net.forward_all(b)
        acts = torch.distributions.Categorical(logits=logits.float()).sample()
    aux = extra.get("aux")
    pw = torch.sigmoid(aux[:, 0].float()).cpu().numpy() if aux is not None else np.full(len(obs), np.nan)
    ph = aux[:, 1].float().cpu().numpy() if aux is not None else np.full(len(obs), np.nan)
    return acts.cpu().numpy(), value.float().cpu().numpy(), pw, ph


def hdr(o, i: int) -> int:
    h = getattr(o, "header", None)
    return int(h[i]) if h is not None and len(h) > i else 0


# ---------------------------------------------------------------- runs

class RunState:
    __slots__ = ("seed", "n", "snap", "snap_n", "cur", "combats", "bosses", "t0")

    def __init__(self, seed: str):
        self.seed, self.n, self.snap, self.snap_n, self.cur, self.combats, self.bosses, self.t0 = seed, 0, None, -9, None, [], [], time.time()


def close_combat(st: RunState, o, lost: bool) -> None:
    c = st.cur
    if c is None:
        return
    hp_end = 0 if lost else int(o.hp)
    c["win"] = 0 if lost else 1
    c["hp_end"] = hp_end
    c["loss"] = float(np.clip((c["hp0"] - hp_end) / max(c["max0"], 1), -1, 1))
    c["pots_end"] = int(potions_held(o)) if not lost else None
    c["steps"] = st.n - c["n0"]
    st.combats.append(c)
    if c["room"] == "Boss":
        st.bosses.append(f"{c['act']}:{c['encounter']}")
    st.cur = None


def cmd_runs(a) -> None:
    cfg, net, ver = load(a.config, a.weights, a.threads, a.device)
    print(f"权重第 {ver} 版，{a.envs} 个引擎，打 {a.runs} 局", flush=True)
    base_req = {"op": "reset", "mode": "run", "character": cfg.character, "ascension": 10, "snapshots": True,
                "route_summary": cfg.route_summary, "privileged": True, **cfg.cut_flags()}
    if cfg.model.future_content:
        base_req["future_content"] = True
    if cfg.model.oracle:
        base_req["oracle"] = True
    fh = open_out(a.out)
    rng = np.random.default_rng(a.seed)
    started = done = 0
    states: dict[int, RunState] = {}
    t_start, steps = time.time(), 0
    with VecEnv(a.envs, log=open("/root/cmbpred/engine_runs.log", "ab")) as env:
        def start(i: int):
            nonlocal started
            seed = f"CMB{a.seed}_{started:05d}"
            started += 1
            states[i] = RunState(seed)
            env.send(i, dict(base_req, seed=seed))
        for i in range(min(a.envs, a.runs)):
            start(i)
        ready: dict[int, object] = {}
        while done < a.runs:
            for i, o in env.recv_ready(timeout=0.5):
                st = states[i]
                if isinstance(o, EngineError):
                    print(f"{st.seed} 引擎报错，丢掉：{str(o)[:200]}", flush=True)
                    done += 1
                    if started < a.runs:
                        start(i)
                    continue
                st.n += 1
                if o.extra and "snapshot" in o.extra:
                    st.snap, st.snap_n = o.extra["snapshot"], st.n
                # 战斗边界（同 actor.combat_aux_labels：战斗中的步，层数或连打序号变了另算一场）
                if st.cur is not None and (not o.in_combat or o.done or (hdr(o, 7), hdr(o, 23)) != st.cur["key"]):
                    lost = bool(o.done) and not bool(o.victory)
                    if a.frames and not lost:
                        st.cur["snap1"] = player_snap(env, i)     # 战后玩家快照：比对战前得出血量、药水、牌组等的变化
                    close_combat(st, o, lost)
                if o.in_combat and not o.done and st.cur is None:
                    ex = (o.extra or {}).get("combat") or {}
                    room, act, enc = ex.get("room"), ex.get("act", int(o.act)), ex.get("encounter")
                    second = room == "Boss" and any(b.startswith(f"{act}:") for b in st.bosses)
                    prev = next((b.split(":", 1)[1] for b in st.bosses if b.startswith(f"{act}:")), None) if second else None
                    st.cur = {"act": int(act), "floor": int(o.floor), "room": room, "encounter": enc, "second": second, "prev": prev,
                              "key": (hdr(o, 7), hdr(o, 23)), "hp0": int(o.hp), "max0": int(o.max_hp), "pots0": int(potions_held(o)),
                              "snap": st.snap if (ex and st.snap and st.n == st.snap_n + 1) else None, "n0": st.n, "first": True}
                    if a.frames:
                        st.cur["frame"] = b64(o)                   # 开战第一帧原始字节（base64）
                        st.cur["snap0"] = player_snap(env, i)      # 开战时的玩家快照（含开战效果之后的血量）
                if o.done:
                    win = bool(o.victory)
                    rec = {"seed": st.seed, "version": ver, "win": win, "floor": int(o.floor), "steps": st.n, "sec": round(time.time() - st.t0, 1),
                           "combats": [{k: v for k, v in c.items() if k not in ("key", "first", "n0")} for c in st.combats]}
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); fh.flush()
                    done += 1
                    if done % 10 == 0:
                        el = time.time() - t_start
                        print(f"{done}/{a.runs} 局 {el:.0f}s 每秒 {steps / el:.0f} 步", flush=True)
                    if started < a.runs:
                        start(i)
                    continue
                ready[i] = o
            if not ready:
                continue
            ids = list(ready)
            obs = [ready[i] for i in ids]
            acts, val, pw, ph = infer(net, obs, [MODE_RUN] * len(ids))
            for k, i in enumerate(ids):
                st = states[i]
                if st.cur is not None and st.cur.get("first"):
                    st.cur.update(first=False, p=float(pw[k]), h=float(ph[k]), v=float(val[k]))
                env.send_step(i, int(acts[k]))
                steps += 1
            ready.clear()
    print(f"完成 {done} 局，{time.time() - t_start:.0f}s", flush=True)


# ---------------------------------------------------------------- cf

def card_pool(kind: str) -> list[str]:
    import glob
    p = sorted(glob.glob(os.path.join(DEPLOY, "engine", "catalog", "v*", "catalog.json")))[-1]
    cat = json.load(open(p, encoding="utf-8"))
    cards = [e for e in cat["entities"] if e.get("category") == "CARD"]
    if kind == "silent":
        return [e["key"] for e in cards if e.get("pool") == "SILENT_CARD_POOL" and e.get("rarity") in ("Common", "Uncommon", "Rare")]
    if kind == "silent_all":
        return [e["key"] for e in cards if e.get("pool") in ("SILENT_CARD_POOL", "COLORLESS_CARD_POOL")]
    if kind == "curse":
        return [e["key"] for e in cards if e.get("pool") == "CURSE_CARD_POOL"]
    if kind == "upgradable":
        return [e["key"] for e in cards if int(e.get("max_upgrade") or 0) > 0]
    raise ValueError(kind)


UPG = "current_upgrade_level"


BASIC = ("CARD.STRIKE_SILENT", "CARD.DEFEND_SILENT")


def variants(snap: str, rng: random.Random, pools: dict) -> list[tuple[str, str]]:
    """原样 + 七种改动：升级一张没升级的牌、去掉一张打击或防御、去掉一张非基础牌、加一张随机猎人牌、加一张诅咒、
    血量少 15% 最大生命、血量多 15%（不超过上限）。牌组改动对应选卡、删牌、升级这些局外决策，血量对应休息处与选路。"""
    d = json.loads(snap)
    out = [("base", snap)]
    deck = d.get("deck") or []
    up = [i for i, c in enumerate(deck) if not c.get(UPG) and c["id"] in pools["upgradable"]]
    if up:
        e = copy.deepcopy(d); i = rng.choice(up); e["deck"][i][UPG] = 1
        out.append((f"upgrade:{deck[i]['id']}", json.dumps(e)))
    basic = [i for i, c in enumerate(deck) if c["id"] in BASIC]
    if basic:
        e = copy.deepcopy(d); i = rng.choice(basic); del e["deck"][i]
        out.append((f"remove:{deck[i]['id']}", json.dumps(e)))
    nonbasic = [i for i, c in enumerate(deck) if c["id"] not in BASIC and c["id"] in pools["silent_all"]]
    if nonbasic:
        e = copy.deepcopy(d); i = rng.choice(nonbasic); del e["deck"][i]
        out.append((f"remove_nb:{deck[i]['id']}", json.dumps(e)))
    fl = max((c.get("floor_added_to_deck", 1) for c in deck), default=1)
    for kind in ("silent", "curse"):
        e = copy.deepcopy(d); cid = rng.choice(pools[kind])
        e["deck"].append({"floor_added_to_deck": fl, "id": cid})
        out.append((f"add_{kind}:{cid}", json.dumps(e)))
    hp, mhp = int(d.get("current_hp", 0)), int(d.get("max_hp", 1))
    dh = round(0.15 * mhp)
    if hp - dh >= 1:
        e = copy.deepcopy(d); e["current_hp"] = hp - dh
        out.append(("hp-15%", json.dumps(e)))
    if hp < mhp:
        e = copy.deepcopy(d); e["current_hp"] = min(mhp, hp + dh)
        out.append(("hp+15%", json.dumps(e)))
    return out


def cmd_cf(a) -> None:
    cfg, net, ver = load(a.config, a.weights, a.threads, a.device)
    rng = random.Random(a.seed)
    pools = {k: card_pool(k) for k in ("silent", "silent_all", "curse", "upgradable")}
    pools["upgradable"] = set(pools["upgradable"]); pools["silent_all"] = set(pools["silent_all"])
    # 挑局面：runs 里有快照的战斗，按（幕 × 房间）分层，每层优先取开战预测胜率在 [lo, hi] 的
    cands = []
    for fn in a.runs_file.split(","):
        for line in open(fn, encoding="utf-8"):
            r = json.loads(line)
            for c in r["combats"]:
                if c.get("snap") and c.get("encounter") and c.get("p") is not None:
                    cands.append(c)
    cands.sort(key=lambda c: (c["act"], c["floor"], c["encounter"], c["hp0"], c["snap"]))     # 与读文件的先后无关，各分片挑出同一批
    rng.shuffle(cands)
    strata: dict[tuple, list] = {}
    for c in cands:
        strata.setdefault((c["act"], c["room"], bool(c.get("second"))), []).append(c)
    pick = []
    quota = json.loads(a.quota)       # {"幕:房间[:second]": 个数}
    for key, n in quota.items():
        parts = key.split(":"); k = (int(parts[0]), parts[1], len(parts) > 2)
        pool = strata.get(k, [])
        mid = [c for c in pool if a.lo <= c["p"] <= a.hi]
        rest = [c for c in pool if not (a.lo <= c["p"] <= a.hi)]
        pick += (mid + rest)[:n]
    jobs = []
    for sid, c in enumerate(pick):
        for vname, snap in variants(c["snap"], rng, pools):
            for k in range(a.k):
                spec = {"character": cfg.character, "ascension": 10, "seed": f"CF{a.seed}_{sid}_{k}", "act": c["act"],
                        "encounter": c["encounter"], "room": c["room"], "player_snapshot": snap}
                if c.get("second"):
                    spec["second"] = True
                    if c.get("prev"):
                        spec["prev_encounter"] = c["prev"]
                jobs.append((sid, vname, k, spec))
    si, sn = (int(x) for x in a.shard.split("/"))
    jobs = [j for n, j in enumerate(jobs) if n % sn == si]
    if a.done_files:        # 已经打完的（同局面、同变体、同种子）不再打：换小 K 重启时沿用先前的结果
        done_set = set()
        for fn in a.done_files.split(","):
            if os.path.exists(fn):
                for line in open(fn, encoding="utf-8"):
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if "error" not in r:
                        done_set.add((r["sid"], r["var"], r["k"]))
        n0 = len(jobs)
        jobs = [j for j in jobs if (j[0], j[1], j[2]) not in done_set]
        print(f"沿用已打完的 {n0 - len(jobs)} 场", flush=True)
    meta = [{"sid": sid, "act": c["act"], "room": c["room"], "encounter": c["encounter"], "second": bool(c.get("second")),
             "run_p": c["p"], "run_h": c["h"], "run_win": c["win"], "run_loss": c["loss"], "hp0": c["hp0"], "max0": c["max0"]}
            for sid, c in enumerate(pick)]
    Path(a.out + ".meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    if a.frames_only:       # 只开战取开战帧（不打）：每个变体取前几个种子，给战斗结果模型出预测用
        jobs = [j for j in jobs if j[2] < a.frames_only]
    if a.max_jobs:
        jobs = jobs[:a.max_jobs]
    print(f"权重第 {ver} 版：{len(pick)} 个局面，{len(jobs)} 场（每个变体 {a.k} 场{f'，只取前 {a.frames_only} 个种子的开战帧' if a.frames_only else ''}），"
          f"{a.envs} 个引擎", flush=True)
    play_specs(a, cfg, net, [({"sid": sid, "var": vname, "k": k}, spec) for sid, vname, k, spec in jobs],
               frames_only=bool(a.frames_only), save_frames=bool(a.frames_only or a.save_frames), post_snap=False)


def play_specs(a, cfg, net, jobs: list, frames_only: bool, save_frames: bool, post_snap: bool) -> None:
    """按战斗规格（玩家快照 + 遭遇）开单场战斗并打完，出手同训练采集（按策略分布采样、带特权特征）。
    jobs 为 [(记录键, 规格)]；frames_only 时只取开战帧不打；save_frames 时记开战帧原始字节；post_snap 时打完要战后玩家快照。"""
    fh = open_out(a.out)
    req0 = {"op": "reset", "mode": "combat", "privileged": True, **cfg.cut_flags()}
    cur: dict[int, dict] = {}
    nxt = 0
    done = 0
    t0, steps = time.time(), 0
    with VecEnv(a.envs, log=open("/root/cmbpred/engine_cf.log", "ab")) as env:
        def start(i: int):
            nonlocal nxt
            key, spec = jobs[nxt]; nxt += 1
            cur[i] = {"key": key, "n": 0, "first": None}
            env.send(i, dict(req0, spec=spec))
        for i in range(min(a.envs, len(jobs))):
            start(i)
        ready: dict[int, object] = {}

        def finish(i: int, rec: dict) -> None:
            nonlocal done
            fh.write(json.dumps({**cur[i]["key"], **rec}, ensure_ascii=False) + "\n")
            done += 1
            if done % 200 == 0:
                fh.flush()
                el = time.time() - t0
                print(f"{done}/{len(jobs)} 场 {el:.0f}s 每秒 {steps / el:.0f} 步", flush=True)
            if nxt < len(jobs):
                start(i)

        def handle(i: int, o) -> None:
            """收到一帧：打完的记结果、换下一场；没打完的放进待推理。"""
            j = cur[i]
            if isinstance(o, EngineError):
                finish(i, {"error": str(o)[:300]})
                return
            j["n"] += 1
            if j["first"] is None:
                j["first"] = {"hp0": int(o.hp), "max0": int(o.max_hp), "pots0": int(potions_held(o))}
                if save_frames:
                    j["frame"] = b64(o)
            if o.done or not o.in_combat:
                lost = not bool(o.victory)
                f = j["first"]
                hp_end = 0 if lost else int(o.hp)
                rec = {"p": j.get("p"), "h": j.get("h"), "v": j.get("v"), "win": 0 if lost else 1,
                       "loss": float(np.clip((f["hp0"] - hp_end) / max(f["max0"], 1), -1, 1)), "hp0": f["hp0"], "max0": f["max0"],
                       "hp_end": hp_end, "max_end": int(o.max_hp), "steps": j["n"], "stuck": bool(getattr(o, "stuck", False))}
                if save_frames:
                    rec["frame"] = j.get("frame")
                if post_snap and not lost:
                    rec["snap1"] = player_snap(env, i)
                finish(i, rec)
                return
            ready[i] = o

        while done < len(jobs):
            for i, o in env.recv_ready(timeout=0.5):
                handle(i, o)
            if not ready:
                continue
            # 凑批（同采集进程的 infer_wait_ms）：小批前向的固定开销大，多等几毫秒让更多引擎回来
            deadline = time.time() + a.wait_ms / 1000
            while len(ready) < a.envs // 2 and time.time() < deadline:
                for i, o in env.recv_ready(timeout=max(0.0, deadline - time.time())):
                    handle(i, o)
            ids = list(ready)
            obs = [ready[i] for i in ids]
            ready.clear()
            acts, val, pw, ph = infer(net, obs, [MODE_COMBAT] * len(ids))
            for k, i in enumerate(ids):
                j = cur[i]
                if "p" not in j:
                    j.update(p=float(pw[k]), h=float(ph[k]), v=float(val[k]))
                    if frames_only:
                        f = j["first"]
                        finish(i, {"p": j["p"], "h": j["h"], "v": j["v"], "hp0": f["hp0"], "max0": f["max0"], "frame": j.get("frame")})
                        continue
                env.send_step(i, int(acts[k]))
                steps += 1
    fh.flush()
    print(f"完成 {done} 场，{time.time() - t0:.0f}s", flush=True)


def potion_pool() -> list[str]:
    import glob
    p = sorted(glob.glob(os.path.join(DEPLOY, "engine", "catalog", "v*", "catalog.json")))[-1]
    cat = json.load(open(p, encoding="utf-8"))
    return [e["key"] for e in cat["entities"] if e.get("category") == "POTION" and e.get("rarity") in ("Common", "Uncommon", "Rare")]


def perturb_snap(snap: str, rng: random.Random, pools: dict, n_edits: int) -> tuple[str, list[str]]:
    """随机改 n_edits 处：加牌（猎人牌或无色牌）、去一张牌、升级一张、变换一张（换成随机猎人牌）、血量 ±5–30% 最大生命、
    加一瓶药水（有空位时）、去一瓶药水。返回改后的快照与改动说明。"""
    d = json.loads(snap)
    deck = d.setdefault("deck", [])
    pots = d.get("potions") or []
    edits = []
    fl = max((c.get("floor_added_to_deck", 1) for c in deck), default=1)
    for _ in range(n_edits):
        kinds = ["add", "remove", "upgrade", "transform", "hp", "pot_add", "pot_remove"]
        kind = rng.choice(kinds)
        if kind == "add":
            cid = rng.choice(pools["silent"] if rng.random() < 0.85 else pools["colorless"])
            deck.append({"floor_added_to_deck": fl, "id": cid}); edits.append(f"add:{cid}")
        elif kind == "remove" and len(deck) > 5:
            i = rng.randrange(len(deck)); edits.append(f"remove:{deck[i]['id']}"); del deck[i]
        elif kind == "upgrade":
            up = [i for i, c in enumerate(deck) if not c.get(UPG) and c["id"] in pools["upgradable"]]
            if up:
                i = rng.choice(up); deck[i][UPG] = 1; edits.append(f"upgrade:{deck[i]['id']}")
        elif kind == "transform" and deck:
            i = rng.randrange(len(deck)); cid = rng.choice(pools["silent"])
            edits.append(f"transform:{deck[i]['id']}>{cid}"); deck[i] = {"floor_added_to_deck": deck[i].get("floor_added_to_deck", fl), "id": cid}
        elif kind == "hp":
            mhp = int(d.get("max_hp", 1)); hp = int(d.get("current_hp", 1))
            dh = round(rng.uniform(0.05, 0.30) * mhp) * (1 if rng.random() < 0.5 else -1)
            d["current_hp"] = int(min(mhp, max(1, hp + dh))); edits.append(f"hp:{d['current_hp'] - hp:+d}")
        elif kind == "pot_add":
            used = {int(p.get("slot_index", -1)) for p in pots}
            free = [s for s in range(int(d.get("max_potion_slot_count", 0) or 0)) if s not in used]
            if free:
                pid = rng.choice(pools["potion"]); pots.append({"id": pid, "slot_index": rng.choice(free)}); edits.append(f"pot_add:{pid}")
        elif kind == "pot_remove" and pots:
            i = rng.randrange(len(pots)); edits.append(f"pot_remove:{pots[i]['id']}"); del pots[i]
    if pots:
        d["potions"] = pots
    else:
        d.pop("potions", None)      # 没有药水时原版快照不写这个字段；写成 null 游戏读档会崩（Player.LoadPotions 空引用）
    return json.dumps(d), edits


def cmd_perturb(a) -> None:
    """D2：从 collect 记下的开战快照出发，随机改 0–3 处，每个改后的局面按快照单打一场（存开战帧与战后快照）。"""
    cfg, net, ver = load(a.config, a.weights, a.threads, a.device)
    rng = random.Random(a.seed)
    pools = {k: card_pool(k) for k in ("silent", "silent_all", "curse", "upgradable")}
    pools["upgradable"] = set(pools["upgradable"])
    pools["colorless"] = [c for c in card_pool("silent_all") if c not in set(pools["silent"])] or pools["silent"]
    pools["potion"] = potion_pool()
    hard, easy = [], []
    for fn in a.src.split(","):
        for r in read_lines(fn):
            for ci, c in enumerate(r["combats"]):
                if c.get("snap") and c.get("encounter"):
                    (hard if (c.get("p") is not None and c["p"] < 0.995) else easy).append((r["seed"], ci, c))
    # 有放回地抽：难打的战斗（开战预测胜率 < 0.995）只占约 17%，同一局面可以换不同的改动多打几次
    n_hard = int(a.n * a.hard_share) if hard else 0
    picks = (rng.choices(hard, k=n_hard) if hard else []) + (rng.choices(easy, k=a.n - n_hard) if easy else [])
    rng.shuffle(picks)
    jobs = []
    for n, (seed, ci, c) in enumerate(picks):
        ne = rng.choices([0, 1, 2, 3], weights=[0.15, 0.45, 0.25, 0.15])[0]
        snap, edits = perturb_snap(c["snap"], rng, pools, ne)
        spec = {"character": cfg.character, "ascension": 10, "seed": f"PT{a.seed}_{n}", "act": c["act"],
                "encounter": c["encounter"], "room": c["room"], "player_snapshot": snap}
        if c.get("second"):
            spec["second"] = True
            if c.get("prev"):
                spec["prev_encounter"] = c["prev"]
        jobs.append(({"src": seed, "ci": ci, "act": c["act"], "room": c["room"], "encounter": c["encounter"], "second": bool(c.get("second")),
                       "edits": edits, "snap0": snap, "run_p": c.get("p")}, spec))
    si, sn = (int(x) for x in a.shard.split("/"))
    jobs = [j for n, j in enumerate(jobs) if n % sn == si]
    if a.max_jobs:
        jobs = jobs[:a.max_jobs]
    print(f"权重第 {ver} 版：来源战斗 难 {len(hard)} / 易 {len(easy)}，本分片 {len(jobs)} 场，{a.envs} 个引擎", flush=True)
    play_specs(a, cfg, net, jobs, frames_only=False, save_frames=True, post_snap=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["runs", "cf", "perturb"])
    ap.add_argument("--config", default="/root/silent_train_config.json")
    ap.add_argument("--weights", default="/root/cmbpred/w.pt")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--runs-file", default="runs.jsonl")
    ap.add_argument("--quota", default='{"0:Monster": 20, "0:Elite": 20, "0:Boss": 20, "1:Monster": 20, "1:Elite": 20, "1:Boss": 20, '
                                       '"2:Monster": 20, "2:Elite": 20, "2:Boss": 20, "2:Boss:second": 20}')
    ap.add_argument("--lo", type=float, default=0.3)
    ap.add_argument("--hi", type=float, default=0.97)
    ap.add_argument("--k", type=int, default=24)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--shard", default="0/1", help="cf 的任务按下标取模分片：i/n")
    ap.add_argument("--done-files", default="", help="cf：这些文件里已经打完的（局面、变体、种子）跳过")
    ap.add_argument("--wait-ms", type=float, default=8.0, help="凑批最多等几毫秒")
    ap.add_argument("--frames", action="store_true", help="runs：每场战斗另存开战帧（base64）与开战、战后的玩家快照")
    ap.add_argument("--frames-only", type=int, default=0, help="cf：只开战取开战帧、不打，每个变体取前几个种子")
    ap.add_argument("--save-frames", action="store_true", help="cf：打的同时存开战帧")
    ap.add_argument("--src", default="", help="perturb：collect 的输出文件（逗号分隔）")
    ap.add_argument("--n", type=int, default=1000, help="perturb：一共打几场（各分片合计）")
    ap.add_argument("--hard-share", type=float, default=0.7, help="perturb：从开战预测胜率 < 0.995 的战斗里取的比例")
    ap.add_argument("--max-jobs", type=int, default=0, help="冒烟用：cf、perturb 最多打几场")
    a = ap.parse_args()
    {"runs": cmd_runs, "cf": cmd_cf, "perturb": cmd_perturb}[a.cmd](a)


if __name__ == "__main__":
    main()
