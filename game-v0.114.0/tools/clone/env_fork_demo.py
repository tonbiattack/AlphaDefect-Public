"""Sts2Env 局面复制接口（snapshot / restore / drop）的示例与自检：在决策点分出 K 个推演、推完恢复、接着走（训练时在线推演用）。

示例的写法（main 环境）：
    o = env.reset_combat(spec)
    while not o.done:
        if 要在这一步推演:
            env.snapshot(0)                                   # 存下决策点
            for s, a in 推演列表:
                r = env.restore(0, reseed=s, first=a)         # 回到决策点、换掉看不到的未来、走候选 a
                ... 用 r 接着 env.step(...) 推几步，记下结果 ...
            o = env.restore(0)                                # 推完回到决策点（不 reseed：真实的未来不变）
        o = env.step(按策略选的动作)
    env.drop()

自检：另开一个 plain 环境不分叉，走同样的动作，每一步的观测都要与 main 相同（头部 12、14 随进程累计，不比）——
推演不能扰动真实的那条线；restore(0) 回来的观测也要与分叉前的完全相同。同一个 (s, a) 推两遍结果相同（快照只读、可反复恢复）。

  python -I tools/clone/env_fork_demo.py [--episodes 6] [--k 4] [--m 8] [--every 3] [--dll 复制版引擎]
同时开 2 个引擎进程（nice 10）。退出码：有不一致为 1。
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from sts2env.env import Sts2Env  # noqa: E402


def same_obs(a, b) -> str | None:
    """两个观测是否相同（头部 12、14 不比；浮点按 NaN 相等）；不同时返回第一处不同的部分名。"""
    ha, hb = a.header.copy(), b.header.copy()
    ha[[12, 14]] = 0
    hb[[12, 14]] = 0
    if not np.array_equal(ha, hb):
        return f"头部第 {int(np.flatnonzero(ha != hb)[0])} 项"
    for name in ("tok_ints", "tok_feats", "tok_priv", "cand_ints", "cand_feats"):
        x, y = getattr(a, name), getattr(b, name)
        if x.shape != y.shape or not np.array_equal(x, y, equal_nan=x.dtype.kind == "f"):
            return name
    if a.labels != b.labels:
        return "labels"
    return None


def rollout(env: Sts2Env, slot: int, seed: int, first: int, m: int, rng: random.Random) -> tuple:
    """一次推演：恢复、换未来、走候选，再随机走 m 步；返回 (走到的步数, 最后的生命, 是否打完)。"""
    o = env.restore(slot, reseed=seed, first=first)
    steps = 1
    while not o.done and o.num_actions > 0 and steps <= m:
        o = env.step(rng.randrange(o.num_actions))
        steps += 1
    return steps, int(o.hp), bool(o.done)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dll", default=str(C.DLL_CLONE))
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--k", type=int, default=4, help="每个决策点分几个推演")
    ap.add_argument("--m", type=int, default=8, help="每个推演走候选之后再走几步")
    ap.add_argument("--every", type=int, default=3, help="主线每隔几步推演一次")
    ap.add_argument("--nice", type=int, default=10)
    a = ap.parse_args()
    C.be_nice(a.nice)
    main_env, plain = Sts2Env(Path(a.dll)), Sts2Env(Path(a.dll))
    chars = ["DEFECT", "FOOL", "REGENT"]
    bad, n_steps, n_roll, t_roll = 0, 0, 0, 0.0
    t0 = time.time()
    for ep in range(a.episodes):
        spec = C.combat_spec(ep, "random", chars[ep % len(chars)])
        o, q = main_env.reset_combat(spec), plain.reset_combat(spec)
        rng, rr = random.Random(f"main{ep}"), random.Random(f"roll{ep}")
        t = 0
        while not o.done and o.num_actions > 0 and t < 200:
            if (d := same_obs(o, q)) is not None:
                print(f"第 {ep} 局第 {t} 步：main 与 plain 不同（{d}）")
                bad += 1
                break
            if t % a.every == 0:
                main_env.snapshot(0)
                plan = [(rr.getrandbits(32), rr.randrange(o.num_actions)) for _ in range(a.k)]
                t1 = time.perf_counter()
                outs = [rollout(main_env, 0, s, c, a.m, random.Random(f"{ep}/{t}/{j}")) for j, (s, c) in enumerate(plan)]
                again = rollout(main_env, 0, plan[0][0], plan[0][1], a.m, random.Random(f"{ep}/{t}/0"))
                t_roll += time.perf_counter() - t1
                n_roll += a.k + 1
                if again != outs[0]:
                    print(f"第 {ep} 局第 {t} 步：同一个推演两遍结果不同 {outs[0]} / {again}")
                    bad += 1
                back = main_env.restore(0)
                if (d := same_obs(back, o)) is not None:
                    print(f"第 {ep} 局第 {t} 步：restore(0) 回来的观测与分叉前不同（{d}）")
                    bad += 1
                o = back
            act = rng.randrange(o.num_actions)
            o, q = main_env.step(act), plain.step(act)
            t += 1
            n_steps += 1
        main_env.drop()
        print(f"第 {ep} 局（{spec['character']} {spec['encounter']}）：{t} 步，{'打完' if o.done else '截断'}，生命 {int(o.hp)}", flush=True)
    main_env.close(); plain.close()
    print(f"共 {n_steps} 步、{n_roll} 次推演（平均每次 {1000 * t_roll / max(1, n_roll):.1f} ms），用时 {time.time() - t0:.0f} 秒；不一致 {bad} 处")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
