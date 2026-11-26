"""异步收发的随机策略测吞吐：python -I bench_async.py [进程数] [秒数] [run|combat]"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

from sts2env import VecEnv  # noqa: E402
from bench_random import make_req, pick  # noqa: E402


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 20
    mode = sys.argv[3] if len(sys.argv) > 3 else "combat"
    rng = np.random.default_rng(0)
    t0 = time.time()
    with VecEnv(n) as env:
        episodes = [0] * n
        for i in range(n):
            env.send(i, dict(make_req(i, 0, mode), op="reset"))
        steps = done_eps = wins = stuck = 0
        t_start = None
        while True:
            for i, o in env.recv_ready():
                if t_start is None:
                    t_start = time.time()
                if o.done or o.stuck:
                    done_eps += 1; wins += o.victory; stuck += o.stuck; episodes[i] += 1
                    env.send(i, dict(make_req(i, episodes[i], mode), op="reset"))
                else:
                    env.send_step(i, pick(rng, o)); steps += 1
            if t_start and time.time() - t_start > seconds:
                break
        dt = time.time() - t_start
        while env.pending:
            env.recv_ready()
    print(f"{mode}（异步）：{n} 进程，{steps} 步 / {dt:.1f}s = {steps / dt:.0f} 决策/秒；结束 {done_eps} 局（胜 {wins}，卡住 {stuck}）")


if __name__ == "__main__":
    main()
