"""远程采集机（A）：起若干采集进程，从主线学习机（B）取权重、把轨迹送回 B（双机协同，决策记录 155；协议见 python/sts2rl/bridge.py）。

采集进程就是训练里的 sts2rl.actor.run_actor，代码不改：它照常每秒查一次本机 /dev/shm 里的权重文件，打出的轨迹放进本机的队列；
这里一个线程从 B 取最新权重写进那个文件，一个线程把队列里的轨迹编码后送给 B。
- 配置：B 的 train_config.json（与主线完全相同，保证打法、课程、奖励一致），再按命令行改本机的采集进程数与每进程引擎数；
  run_dir 换成本机目录（卡住记录等写在这里）；房间结果记录关掉。
- 采集进程编号从 --rank0 起（与 B 本机的 0…actors−1 不重叠，随机数种子也就不同）；进程退出就重启。
- 采集进程出错（("error", …) 项）只记在本机，不送给 B，免得 B 的学习进程因此退出。
- 每 30 秒写一次 <run_dir>/collector.json（连接状态、权重版本、已送轨迹数与流量、采集进程存活数），并照训练那样记 resource.jsonl。

    .venv/bin/python -I tools/remote/remote_actors.py --config b_train_config.json --run-dir ~/.local/share/sls2/runs/collector \\
        --host B_LAN_IP --port 47321 --token-file ~/.sls2_bridge_token --actors 10 --envs 20
停止：kill 本进程（会通知采集进程退出）。
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import signal
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

from sts2rl.actor import run_actor  # noqa: E402
from sts2rl.bridge import load_token, traj_loop, weights_loop  # noqa: E402
from sts2rl.config import TrainConfig  # noqa: E402
from sts2rl.resources import start_sampler  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--token-file", required=True)
    ap.add_argument("--actors", type=int, default=10)
    ap.add_argument("--envs", type=int, default=20)
    ap.add_argument("--rank0", type=int, default=100)
    # 本机权重文件（采集进程与评测读它）：同一台机器上同时有两条训练线的采集时各用各的（储君为 /dev/shm/sts2rl_regent.pt）；默认同原来
    ap.add_argument("--weights", default="")
    a = ap.parse_args()
    run_dir = Path(a.run_dir).expanduser(); run_dir.mkdir(parents=True, exist_ok=True)
    cfg = TrainConfig.from_args(["--config", a.config, "--run-dir", str(run_dir)])
    cfg.actors, cfg.envs_per_actor, cfg.room_log = a.actors, a.envs, 0.0
    cfg.actor_device = "cuda"
    token = load_token(a.token_file)
    shm = Path("/dev/shm") if Path("/dev/shm").exists() else run_dir
    weights = Path(a.weights) if a.weights else shm / "sts2rl_remote.pt"
    stop_t = threading.Event()
    status: dict = {"started": time.time(), "host": a.host, "actors": a.actors, "envs": a.envs}
    threading.Thread(target=weights_loop, args=(a.host, a.port, token, weights, stop_t, status), daemon=True).start()
    print("等第一份权重…", flush=True)
    while "version" not in status:
        time.sleep(1)
    print(f"拿到权重第 {status['version']} 版，起 {a.actors} 个采集进程 × {a.envs} 个引擎", flush=True)

    ctx = mp.get_context("spawn")
    # 本机队列只当缓冲（约 0.25 批）：B 那边的队列满了会反压到这里，囤多了只会让轨迹变旧、在 B 上被当作过旧丢掉
    q = ctx.Queue(maxsize=max(64, int(0.25 * cfg.batch_steps / cfg.queue_item_steps)))
    stop = ctx.Event()
    threading.Thread(target=traj_loop, args=(a.host, a.port, token, q, stop_t, status), daemon=True).start()
    procs: dict[int, mp.Process] = {}

    def spawn(r: int) -> None:
        p = ctx.Process(target=run_actor, args=(r, cfg, str(weights), q, stop), daemon=True)
        p.start(); procs[r] = p

    for r in range(a.rank0, a.rank0 + a.actors):
        spawn(r)
    sampler_stop = threading.Event(); start_sampler(run_dir, sampler_stop)
    quit_ = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: quit_.set())
    signal.signal(signal.SIGINT, lambda *_: quit_.set())
    restarts = 0
    while not quit_.wait(30):
        for r, p in list(procs.items()):
            if not p.is_alive():
                print(f"{time.strftime('%H:%M:%S')} 采集进程 {r} 退出（{p.exitcode}），重启", flush=True)
                restarts += 1; spawn(r)
        st = dict(status, time=time.time(), alive=sum(p.is_alive() for p in procs.values()), restarts=restarts)
        tmp = run_dir / "collector.json.tmp"; tmp.write_text(json.dumps(st, ensure_ascii=False)); tmp.replace(run_dir / "collector.json")
    stop.set(); stop_t.set(); sampler_stop.set()
    for p in procs.values():
        p.join(timeout=20)
        if p.is_alive():
            p.terminate()


if __name__ == "__main__":
    main()
