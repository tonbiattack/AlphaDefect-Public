"""随机策略测吞吐：python -I bench_random.py [进程数] [秒数] [run|combat]"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

from sts2env import VecEnv, ActionKind  # noqa: E402

CHARS = os.environ.get("STS2_CHARS", "DEFECT").split(",")


def make_req(i: int, k: int, mode: str) -> dict:
    c = CHARS[(i + k) % len(CHARS)]
    if mode == "combat":
        enc = ["weak", "regular", "elite", "boss"][k % 4]
        return {"mode": "combat", "spec": {"character": c, "ascension": 10, "seed": f"B{i}_{k}", "act": k % 3, "encounter": enc}}
    return {"mode": "run", "character": c, "ascension": 10, "seed": f"B{i}_{k}"}


def pick(rng, obs) -> int:
    n = obs.num_actions
    if n > 1 and obs.cand_ints[-1, 0] == ActionKind.END_TURN and rng.random() > 0.15:
        return int(rng.integers(n - 1))
    return int(rng.integers(n))


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 20
    mode = sys.argv[3] if len(sys.argv) > 3 else "combat"
    rng = np.random.default_rng(0)
    t0 = time.time()
    with VecEnv(n) as env:
        episodes = [0] * n
        obs = env.reset([make_req(i, 0, mode) for i in range(n)])
        t_ready = time.time()
        steps = wins = done_eps = stuck = 0
        max_tok = max_cand = 0
        while time.time() - t_ready < seconds:
            ended = {i: o for i, o in enumerate(obs) if o.done or o.stuck}
            if ended:
                for i, o in ended.items():
                    done_eps += 1; wins += o.victory; stuck += o.stuck; episodes[i] += 1
                for i, o in env.reset_some({i: make_req(i, episodes[i], mode) for i in ended}).items():
                    obs[i] = o
            acts = []
            for o in obs:
                max_tok = max(max_tok, o.tok_ints.shape[0]); max_cand = max(max_cand, o.num_actions)
                acts.append(pick(rng, o))
            obs = env.step(acts)
            steps += n
        dt = time.time() - t_ready
    print(f"{mode}：{n} 进程，启动 {t_ready - t0:.1f}s，{steps} 步 / {dt:.1f}s = {steps / dt:.0f} 决策/秒；"
          f"结束 {done_eps} 局（胜 {wins}，卡住 {stuck}），最多 {max_tok} 条记录、{max_cand} 个候选")


if __name__ == "__main__":
    main()
