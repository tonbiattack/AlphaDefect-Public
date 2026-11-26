"""租用机标定：测引擎、推理、学习、采集进程四项吞吐，推荐训练配置。

  python -I calibrate.py [--seconds 12] [--out calib.json] [--quick]

输出：屏幕报告 + JSON（含推荐配置，可直接 `python -I train_defect.py --config 推荐配置.json`）。
"""
import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from sts2env import VecEnv  # noqa: E402
from sts2rl.catalog import Catalog  # noqa: E402
from sts2rl.config import TrainConfig  # noqa: E402
from sts2rl.curriculum import CombatCurriculum  # noqa: E402
from sts2rl.features import Ragged, collate_ragged  # noqa: E402
from sts2rl.model import ModelConfig, PolicyValueNet  # noqa: E402


def machine_info() -> dict:
    info = {"cpus": os.cpu_count(), "python": platform.python_version(), "torch": torch.__version__,
            "cuda": torch.cuda.is_available()}
    try:
        info["cpu_model"] = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
        info["mem_gb"] = round(int(next(l.split()[1] for l in open("/proc/meminfo") if l.startswith("MemTotal"))) / 1e6, 1)
    except Exception:
        pass
    if info["cuda"]:
        p = torch.cuda.get_device_properties(0)
        info["gpu"] = p.name; info["gpu_mem_gb"] = round(p.total_memory / 1e9, 1); info["gpu_count"] = torch.cuda.device_count()
    return info


def engine_throughput(n_proc: int, seconds: float, cur: CombatCurriculum) -> float:
    """随机策略、异步收发、战斗模式，测 n_proc 个引擎的总决策/秒。"""
    rng = np.random.default_rng(0)
    with VecEnv(n_proc) as env:
        for i in range(n_proc):
            env.send(i, {"op": "reset", "mode": "combat", "spec": cur.sample(0.6)})
        steps, t0 = 0, None
        while True:
            for i, o in env.recv_ready():
                if t0 is None:
                    t0 = time.time()
                if o.done or o.stuck or o.num_actions == 0:
                    env.send(i, {"op": "reset", "mode": "combat", "spec": cur.sample(0.6)})
                else:
                    env.send_step(i, int(rng.integers(o.num_actions))); steps += 1
            if t0 and time.time() - t0 > seconds:
                break
        dt = time.time() - t0
        while env.pending:
            env.recv_ready()
    return steps / dt


def sample_obs(n: int, cur: CombatCurriculum) -> list:
    rng = np.random.default_rng(1); out = []
    with VecEnv(4) as env:
        obs = env.reset([{"mode": "combat", "spec": cur.sample(0.8)} for _ in range(4)])
        while len(out) < n:
            out += [o for o in obs if not o.done]
            nxt = env.step([None if o.done else int(rng.integers(o.num_actions)) for o in obs])
            obs = [x if x is not None else o for x, o in zip(nxt, obs)]
            for i, o in enumerate(obs):
                if o.done:
                    obs[i] = env.envs[i].reset_combat(cur.sample(0.8))
    return out[:n]


def inference_throughput(net, ragged: Ragged, device, batch: int, seconds: float) -> float:
    idx = np.arange(batch) % len(ragged)
    b = collate_ragged(ragged.select(idx))
    t0, k = time.time(), 0
    with torch.no_grad():
        while time.time() - t0 < seconds:
            bb = b.to(device)                      # 含组批后的拷贝，贴近采集进程的真实开销
            logits, v = net(bb)
            if device.type == "cuda": torch.cuda.synchronize()
            k += 1
    return k * batch / (time.time() - t0)


def learn_throughput(net, ragged: Ragged, device, minibatch: int, seconds: float) -> float:
    opt = torch.optim.Adam(net.parameters(), 1e-4)
    t0, k = time.time(), 0
    rng = np.random.default_rng(0)
    while time.time() - t0 < seconds:
        idx = rng.integers(0, len(ragged), minibatch)
        b = collate_ragged(ragged.select(idx)).to(device)
        logits, v = net(b)
        loss = torch.logsumexp(logits, -1).mean() + v.pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if device.type == "cuda": torch.cuda.synchronize()
        k += 1
    return k * minibatch / (time.time() - t0)


def actor_overhead(ragged: Ragged, batch: int, seconds: float) -> float:
    """采集进程 Python 侧（拼接、组批，不含网络和引擎）每秒能处理的决策数。"""
    obs_like = [ragged.select(np.array([i])) for i in range(min(len(ragged), 256))]
    t0, k = time.time(), 0
    while time.time() - t0 < seconds:
        parts = [obs_like[(k * batch + j) % len(obs_like)] for j in range(batch)]
        collate_ragged(Ragged.concat(parts)); k += 1
    return k * batch / (time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--out", default="calibration.json")
    ap.add_argument("--quick", action="store_true", help="少测几个点")
    ap.add_argument("--model-d", type=int, default=128); ap.add_argument("--model-layers", type=int, default=2)
    a = ap.parse_args()
    info = machine_info(); print("机器：", json.dumps(info, ensure_ascii=False), flush=True)
    cur = CombatCurriculum(Catalog(), seed=7)
    cpus = info["cpus"] or 4
    report = {"machine": info, "engine": {}, "inference": {}, "learn": {}, "actor_python": {}}

    # 1. 引擎吞吐随进程数
    points = sorted({1, max(1, cpus // 4), max(1, cpus // 2), max(1, (3 * cpus) // 4), max(1, cpus - 2)})
    if a.quick: points = sorted({1, max(1, cpus // 2), max(1, cpus - 2)})
    for n in points:
        report["engine"][n] = round(engine_throughput(n, a.seconds, cur))
        print(f"引擎 {n:3d} 进程：{report['engine'][n]:7,} 决策/秒", flush=True)

    # 2–4. 网络
    obs = sample_obs(4096, cur); rag = Ragged.from_obs(obs)
    mc = ModelConfig(d=a.model_d, layers=a.model_layers)
    for dev_name in (["cuda"] if info["cuda"] else []) + ["cpu"]:
        dev = torch.device(dev_name)
        torch.set_num_threads(max(1, cpus // 2) if dev_name == "cpu" else 4)
        net = PolicyValueNet(mc).to(dev).eval()
        for bs in ([16, 64, 256, 1024] if dev_name == "cuda" else [16, 64, 256]):
            r = inference_throughput(net, rag, dev, bs, a.seconds / 3)
            report["inference"][f"{dev_name}@{bs}"] = round(r)
            print(f"推理 {dev_name} 批 {bs:5d}：{r:9,.0f} 决策/秒", flush=True)
        net.train()
        for mb in ([1024, 2048, 4096] if dev_name == "cuda" else [1024]):
            r = learn_throughput(net, rag, dev, mb, a.seconds / 2)
            report["learn"][f"{dev_name}@{mb}"] = round(r)
            print(f"训练 {dev_name} 小批 {mb:5d}：{r:9,.0f} 样本/秒（一遍）", flush=True)
    for bs in (8, 16, 32):
        r = actor_overhead(rag, bs, a.seconds / 3)
        report["actor_python"][bs] = round(r)
        print(f"采集进程 Python 侧（批 {bs}）：{r:9,.0f} 决策/秒（不含网络与引擎）", flush=True)

    # 推荐配置
    best_n = max(report["engine"], key=lambda n: report["engine"][n])
    engine_rate = report["engine"][best_n]
    cuda = info["cuda"]
    learn_key = max((k for k in report["learn"] if k.startswith("cuda" if cuda else "cpu")), key=lambda k: report["learn"][k])
    learn_rate = report["learn"][learn_key] / TrainConfig().epochs
    envs_per_actor = 16 if cuda else 8
    actors = max(1, math.ceil(best_n / envs_per_actor))
    rec = TrainConfig()
    rec.actors, rec.envs_per_actor = actors, math.ceil(best_n / actors)
    rec.actor_device = "cuda" if cuda else "cpu"; rec.learner_device = "cuda" if cuda else "cpu"
    rec.minibatch = int(learn_key.split("@")[1]); rec.batch_steps = max(16384, rec.minibatch * 8)
    rec.learner_threads = 4 if cuda else max(2, cpus // 4)
    rec.model.d, rec.model.layers = a.model_d, a.model_layers
    report["recommended"] = rec.to_dict()
    report["projected_steps_per_s"] = round(min(engine_rate, learn_rate))
    report["bottleneck"] = "引擎（CPU 核数）" if engine_rate < learn_rate else "学习（显卡/CPU 训练）"
    print(f"\n推荐：{actors} 个采集进程 × {rec.envs_per_actor} 个引擎，推理 {rec.actor_device}，学习 {rec.learner_device}，"
          f"小批 {rec.minibatch}；预计 {report['projected_steps_per_s']:,} 决策/秒，瓶颈在{report['bottleneck']}", flush=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    cfg_path = os.path.splitext(a.out)[0] + "_train_config.json"
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(report["recommended"], f, ensure_ascii=False, indent=1)
    print(f"报告：{a.out}；训练配置：{cfg_path}")


if __name__ == "__main__":
    main()
