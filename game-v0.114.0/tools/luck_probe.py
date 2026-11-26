"""运气校正（config.luck_cv，铁甲战士决策记录第 20 条）的离线量尺：用训练过的权重打整局（按策略采样出手，同训练），
在 Boss 战里选“结束回合”时做与采集端相同的推演（存局面 → 按公开信息换 k 次随机数各走一遍 → 恢复局面走真实的一步），
量出运气有多大、能从优势里去掉多少方差。不训练、不改任何东西。

两种读数（默认只看 Boss 战，--rooms all 为全部战斗；战斗目标取这场打完后第一帧的价值，死了为 0，同 combat_fight_boot 的口径；γ = 1）：
  1. 每步 δ：校正的那些步上 δ = V(s') − V(s) 与 δ' = V̄ − V(s) 的均方根（与学习端日志的 luck 同口径）；
  2. 整场的优势：这场里每个决策 A = 目标 − V(s)，A' = A − 它之后各次运气之和；比较 A 与 A' 的方差（λ = 1 时策略梯度实际用的量）。

用法（CPU 即可，C 上用部署目录的虚拟环境）：
  python -I tools/luck_probe.py --ckpt /root/ens_ironclad/ck/v22968.pt --runs 40 --envs 8 --k 4 --out luck_probe.json
"""
from __future__ import annotations

import argparse
import json
import secrets
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))
import torch  # noqa: E402

from sts2env import VecEnv  # noqa: E402
from sts2env.protocol import ActionKind  # noqa: E402
from sts2rl.actor import MODE_RUN  # noqa: E402
from sts2rl.features import Ragged, collate_ragged  # noqa: E402
from sts2rl.infer import load_net  # noqa: E402


def ended(o) -> bool:
    return bool(o.done or o.stuck or o.num_actions == 0 or not o.in_combat)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--runs", type=int, default=40)
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--slot", type=int, default=7)
    ap.add_argument("--rooms", choices=["boss", "all"], default="boss", help="boss 只量 Boss 战（同 luck_cv_rooms 默认）；all 为全部战斗")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=20000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    net = load_net(a.ckpt, a.device)
    rng = np.random.default_rng(secrets.randbits(32))
    env = VecEnv(a.envs)

    def values_of(obs: list) -> np.ndarray:
        if not obs:
            return np.zeros(0, np.float32)
        b = collate_ragged(Ragged.from_obs(obs, np.full(len(obs), MODE_RUN))).to(a.device)
        with torch.no_grad():
            _, v, _ = net.forward_phi(b)
        return v.float().cpu().numpy()

    def new_req() -> dict:
        seed = "".join(secrets.choice("0123456789ABCDEFGHJKLMNPQRSTUVWXYZ") for _ in range(10))
        return {"mode": "run", "character": "IRONCLAD", "ascension": 10, "seed": seed, "snapshots": True,
                "loop_cut": True, "stall_turns": 20}

    started = 0
    obs = env.reset([new_req() for _ in range(a.envs)]); started += a.envs
    room = [None] * a.envs          # 当前战斗的房间类型
    fight: list[list | None] = [None] * a.envs     # 这场 Boss 战里各决策：[V(s), 之后这一步的运气（没校正为 None）]
    steps = [0] * a.envs
    td = []                          # (δ, δ')：校正的步
    adv = []                         # (A, A')：Boss 战里的每个决策
    n_skip = n_mixed = n_cv = runs_done = 0
    t0 = time.time()
    live = list(range(a.envs))
    while live:
        cur = [obs[i] for i in live]
        b = collate_ragged(Ragged.from_obs(cur, np.full(len(cur), MODE_RUN))).to(a.device)
        with torch.no_grad():
            logits, vals, _ = net.forward_phi(b)
            acts = torch.distributions.Categorical(logits=logits.float()).sample().cpu().numpy()
        vals = vals.float().cpu().numpy()
        nxt: dict[int, object] = {}
        plain: list[int | None] = [None] * a.envs
        branch_obs, branch_of = [], []      # 推演帧与它属于哪个引擎
        for k, i in enumerate(live):
            o, act = obs[i], int(acts[k])
            if fight[i] is not None:
                fight[i].append([float(vals[k]), None])
            if fight[i] is not None and int(o.cand_ints[act, 0]) == ActionKind.END_TURN:
                e = env.envs[i]
                e.snapshot(a.slot)
                bo = [e.restore(a.slot, reseed=int(rng.integers(1, 1 << 32)), first=act, keep_top=True, rewards=False) for _ in range(a.k)]
                real = e.restore(a.slot, first=act)
                ends = [ended(x) for x in bo] + [ended(real)]
                if any(ends):
                    n_skip += 1; n_mixed += int(not all(ends))
                else:
                    branch_obs += bo; branch_of += [(i, len(bo))]
                nxt[i] = real
            else:
                plain[i] = act
        for i, o in enumerate(env.step(plain)):
            if o is not None:
                nxt[i] = o
        # 推演帧的价值与真实下一帧的价值（同一遍前向），记下运气
        reals = [i for i, _ in branch_of]
        vv = values_of(branch_obs + [nxt[i] for i in reals])
        j = 0
        for (i, m), vr in zip(branch_of, vv[len(branch_obs):]):
            vbar = float(vv[j:j + m].mean()); j += m
            vs = fight[i][-1][0]
            td.append((float(vr) - vs, vbar - vs)); n_cv += 1
            fight[i][-1][1] = float(vr) - vbar
        # 推进各引擎：战斗开始、结束、局结束
        for i in list(live):
            o = nxt[i]; steps[i] += 1
            if o.extra and "combat" in o.extra:
                room[i] = o.extra["combat"]["room"]
                fight[i] = [] if (room[i] == "Boss" or a.rooms == "all") else None
            if fight[i] is not None and ended(o):
                target = 0.0 if (o.done or o.stuck) else float(values_of([o])[0])
                luck_after = 0.0
                for v_s, lk in reversed(fight[i]):
                    if lk is not None:
                        luck_after += lk
                    adv.append((target - v_s, target - v_s - luck_after))
                fight[i] = None
            if o.done or o.stuck or o.num_actions == 0 or steps[i] >= a.max_steps:
                runs_done += 1
                if started < a.runs:
                    obs[i] = env.reset_some({i: new_req()})[i]; started += 1; steps[i] = 0; room[i] = None; fight[i] = None
                else:
                    live.remove(i)
            else:
                obs[i] = o
        if runs_done and runs_done % 10 == 0:
            print(f"{runs_done} 局，校正 {n_cv} 步，用时 {time.time() - t0:.0f} 秒", flush=True)
    td_a, ad_a = np.array(td), np.array(adv)
    res = {"ckpt": a.ckpt, "runs": runs_done, "k": a.k, "seconds": round(time.time() - t0, 1),
           "cv_steps": n_cv, "skipped": n_skip, "mixed": n_mixed,
           "td_rms": float(np.sqrt((td_a[:, 0] ** 2).mean())) if n_cv else None,
           "td_rms_cv": float(np.sqrt((td_a[:, 1] ** 2).mean())) if n_cv else None,
           "luck_mean": float((td_a[:, 0] - td_a[:, 1]).mean()) if n_cv else None,
           "luck_rms": float(np.sqrt(((td_a[:, 0] - td_a[:, 1]) ** 2).mean())) if n_cv else None,
           "boss_decisions": len(adv),
           "adv_var": float(ad_a[:, 0].var()) if len(adv) else None,
           "adv_var_cv": float(ad_a[:, 1].var()) if len(adv) else None}
    if res["adv_var"]:
        res["adv_var_ratio"] = round(res["adv_var_cv"] / res["adv_var"], 4)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False), flush=True)
    env.close() if hasattr(env, "close") else None


if __name__ == "__main__":
    main()
