"""租用机基准测试：引擎单进程吞吐、不同“采集进程 × 每进程引擎”组合的采集吞吐（用检查点推理、不训练）、
学习进程一批（batch_steps 步、epochs 遍）的耗时，给出推荐的硬件配置。一般由 bench.sh 调用，也可直接运行：

  .venv/bin/python -I tools/remote/bench.py --config 训练配置.json [--ckpt 检查点.pt] [--grid 12x16,16x16,12x20,16x20]
                                            [--quick] [--no-eval] [--no-pin] [--out 输出目录]

测什么：
  1. 引擎：1 个引擎随机策略打战斗、整局各 15 秒；再开与逻辑核数相同的引擎测随机策略的总吞吐（只有引擎、没有网络时的上限）。
  2. 采集：按 --grid 逐个组合启动真正的采集进程（sts2rl.actor.run_actor，推理设备、凑批等待、奖励、课程都取训练配置），
     同时按配置启动评测进程（与训练时一样 SCHED_IDLE + cgroup 限核）。决策数在每个采集进程里直接计数（每发出一步加一），
     不等对局结束，所以整局很长也不影响读数。每个组合测两段：
       采集单独跑：只有采集与评测；
       采集 + 学习：主进程同时在显卡上反复做 PPO 更新（与 train.py 相同的前向、损失、bf16、融合 Adam、后台组批），
       每次更新后照常发布一次权重（内容仍是检查点，策略不漂移），采集进程照常重新加载。这一段最接近真实训练。
     同一 envs_per_actor 的组合按采集进程数从少到多依次追加进程，不重复启动已有的引擎。
  3. 亲和性（只在大小核 CPU 上）：在推荐组合上比较三种放法：不绑定；学习与采集进程绑性能核、评测绑能效核；
     再把训练引擎也绑到能效核。
  4. 学习单独跑：停掉全部采集与评测，显卡独占时一次更新的耗时。

预计训练吞吐 ≈ min(采集 + 学习时的采集速度, batch_steps / (学习一次的耗时 + 取数与准备约 0.45 秒))；取最高者，
在相差 3% 以内时取引擎较少的组合（内存与过旧数据都更少）。学习用的数据是测试期间收到的真实轨迹（最新的 batch_steps 步）；
测试时间短，长的整局收到得少，局外决策（含地图节点、记录最多的那类）占比会低于真实训练，学习耗时可能略偏低，结果里给出占比供对照。
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import platform
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from sts2env import VecEnv  # noqa: E402
from sts2rl.config import TrainConfig  # noqa: E402
from sts2rl.features import Ragged  # noqa: E402
from sts2rl.model import ModelConfig, PolicyValueNet, load_state, use_tf32  # noqa: E402

# 迁移时作为“与机器相关”的键（与 tools/remote/hosts.sh 的 HW_KEYS 一致）
# 学习进程每次更新除 PPO 本身之外的开销：从队列取一批轨迹（反序列化）约 0.3 秒、算优势与拼接约 0.15 秒（A 上 t_prep 约 0.15 秒）
OVERHEAD_S = 0.45
HW_KEYS = ("actors", "envs_per_actor", "actor_device", "learner_device", "actor_threads", "learner_threads", "infer_wait_ms",
           "eval_workers", "eval_cpu_cores", "eval_envs", "eval_device")
def say(*a) -> None:
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)


# ---------------------------------------------------------------- 机器信息与采样
def _parse_cpus(s: str) -> list[int]:
    out = []
    for part in s.strip().split(","):
        if not part:
            continue
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    return out


def cpu_groups() -> tuple[list[int], list[int]]:
    """大小核：性能核与能效核的逻辑 CPU 编号（Intel 混合架构内核导出 cpu_core / cpu_atom）；不是混合架构时返回两个空表。"""
    try:
        p = _parse_cpus(Path("/sys/devices/cpu_core/cpus").read_text())
        e = _parse_cpus(Path("/sys/devices/cpu_atom/cpus").read_text())
        return p, e
    except OSError:
        return [], []


def machine_info() -> dict:
    info = {"host": platform.node(), "cpus": os.cpu_count(), "python": platform.python_version(), "torch": torch.__version__,
            "cuda": torch.cuda.is_available()}
    try:
        info["cpu_model"] = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
        info["mem_gb"] = round(mem()[1], 1)
    except Exception:
        pass
    p, e = cpu_groups()
    if p:
        info["p_cpus"], info["e_cpus"] = p, e
    if info["cuda"]:
        prop = torch.cuda.get_device_properties(0)
        info["gpu"] = prop.name
        info["gpu_mem_gb"] = round(prop.total_memory / 1e9, 1)
        try:
            info["driver"] = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                            capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:
            pass
        info["mps"] = subprocess.run(["pgrep", "-x", "nvidia-cuda-mps"], capture_output=True).returncode == 0 or \
            subprocess.run(["pgrep", "-f", "nvidia-cuda-mps-control -d"], capture_output=True).returncode == 0
    return info


def cpu_times() -> tuple[int, int]:
    with open("/proc/stat") as f:
        v = [int(x) for x in f.readline().split()[1:]]
    return sum(v), v[3] + v[4]


def mem() -> tuple[float, float]:
    """(已用 GB, 总量 GB)，已用 = 总量 − 可用。"""
    m = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":", 1)
            m[k] = int(v.split()[0])
    return (m["MemTotal"] - m["MemAvailable"]) / 1e6, m["MemTotal"] / 1e6


class GpuSampler:
    """后台每 2 秒记一次显卡占用；没有显卡时什么也不做。"""
    def __init__(self):
        self.samples: list[tuple[float, float, float]] = []
        self.stop = threading.Event()
        if torch.cuda.is_available():
            threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while not self.stop.wait(2.0):
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=5).stdout.strip().splitlines()[0]
                u, m = (float(x) for x in out.split(","))
                self.samples.append((time.time(), u, m / 1024))
            except Exception:
                pass

    def mean(self, t0: float, t1: float) -> tuple[float | None, float | None]:
        s = [x for x in self.samples if t0 <= x[0] <= t1]
        if not s:
            return None, None
        return round(float(np.mean([x[1] for x in s])), 1), round(max(x[2] for x in s), 1)


# ---------------------------------------------------------------- 进程树与亲和性
def _ppid_map() -> dict[int, int]:
    out = {}
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                with open(f"/proc/{d}/stat") as f:
                    s = f.read()
                out[int(d)] = int(s[s.rindex(")") + 2:].split()[1])
            except (OSError, ValueError):
                pass
    return out


def descendants(roots: list[int]) -> list[int]:
    pp = _ppid_map()
    kids: dict[int, list[int]] = {}
    for c, p in pp.items():
        kids.setdefault(p, []).append(c)
    out, stack = [], list(roots)
    while stack:
        for c in kids.get(stack.pop(), []):
            out.append(c); stack.append(c)
    return out


def set_affinity(pids: list[int], cpus: list[int]) -> int:
    """把这些进程的全部线程绑到 cpus（sched_setaffinity 只作用于单个线程，所以逐个线程设）。返回设置成功的线程数。"""
    n = 0
    for pid in pids:
        try:
            tids = os.listdir(f"/proc/{pid}/task")
        except OSError:
            continue
        for t in tids:
            try:
                os.sched_setaffinity(int(t), cpus); n += 1
            except OSError:
                pass
    return n


# ---------------------------------------------------------------- 1. 引擎
def engine_rate(n: int, seconds: float, mode: str) -> tuple[float, float]:
    """n 个引擎、随机策略、异步收发，返回 (决策/秒, 启动秒数)。"""
    from bench_random import make_req, pick
    rng = np.random.default_rng(0)
    t0 = time.time()
    with VecEnv(n) as env:
        episodes = [0] * n
        for i in range(n):
            env.send(i, dict(make_req(i, 0, mode), op="reset"))
        steps, t_start = 0, None
        while True:
            for i, o in env.recv_ready(timeout=1.0):
                if t_start is None:
                    t_start = time.time()
                if o.done or o.stuck or o.num_actions == 0:
                    episodes[i] += 1
                    env.send(i, dict(make_req(i, episodes[i], mode), op="reset"))
                else:
                    env.send_step(i, pick(rng, o)); steps += 1
            if t_start and time.time() - t_start > seconds:
                break
        dt = time.time() - t_start
        while env.pending:
            env.recv_ready(timeout=1.0)
    return steps / dt, t_start - t0


# ---------------------------------------------------------------- 2. 采集
def _actor_entry(rank: int, cfg, weights_path: str, out_q, stop, counts) -> None:
    """采集进程入口：照常运行 run_actor，只是每发出一步就在共享计数里加一。"""
    import sts2env
    orig = sts2env.VecEnv.send_step

    def send_step(self, i, a):
        counts[rank] += 1
        return orig(self, i, a)
    sts2env.VecEnv.send_step = send_step
    from sts2rl.actor import run_actor
    run_actor(rank, cfg, weights_path, out_q, stop)


class TrajBuffer:
    """收到的轨迹（只留学习要用的几项），保留最新的约 3 批。"""
    def __init__(self, keep_steps: int):
        self.items: deque = deque()
        self.steps = 0
        self.keep = keep_steps
        self.lock = threading.Lock()
        self.n = {"traj": 0, "run": 0, "steps": 0}

    def add(self, item: dict) -> None:
        with self.lock:
            self.items.append({"ragged": item["ragged"], "macro": item["macro"]})
            self.steps += item["len"]
            self.n["traj"] += 1; self.n["run"] += int(item["mode"] == 1); self.n["steps"] += item["len"]
            while self.steps - len(self.items[0]["macro"]) >= self.keep:
                self.steps -= len(self.items.popleft()["macro"])

    def newest(self, steps: int) -> list[dict]:
        with self.lock:
            out, n = [], 0
            for it in reversed(self.items):
                out.append(it); n += len(it["macro"])
                if n >= steps:
                    break
            return out


class Fleet:
    """一组采集进程（同一个 envs_per_actor），可逐步追加。"""
    def __init__(self, cfg, envs: int, weights_path: str, buf: TrajBuffer, max_actors: int):
        self.ctx = mp.get_context("spawn")
        self.cfg = cfg
        self.cfg.envs_per_actor = envs
        self.envs = envs
        self.wpath = weights_path
        self.q = self.ctx.Queue(maxsize=4096)
        self.stop = self.ctx.Event()
        self.counts = self.ctx.RawArray("q", max_actors)
        self.procs: list = []
        self.buf = buf
        self.errors: list[str] = []
        self._drain_stop = threading.Event()
        self._drain = threading.Thread(target=self._drain_loop, daemon=True)
        self._drain.start()

    def _drain_loop(self):
        import queue as _q
        while not self._drain_stop.is_set():
            try:
                kind, rank, item = self.q.get(timeout=0.5)
            except _q.Empty:
                continue
            except (EOFError, OSError):
                break
            if kind == "error":
                self.errors.append(f"采集进程 {rank}：{item}")
            else:
                self.buf.add(item)

    def add(self, n: int) -> None:
        for _ in range(n):
            r = len(self.procs)
            p = self.ctx.Process(target=_actor_entry, args=(r, self.cfg, self.wpath, self.q, self.stop, self.counts), daemon=True)
            p.start(); self.procs.append(p)

    @property
    def actors(self) -> int:
        return len(self.procs)

    def total(self) -> int:
        return int(sum(self.counts[:len(self.procs)]))

    def wait_stepping(self, timeout: float) -> float:
        """等到每个采集进程都已发出过决策（引擎都启动好了）；返回等了多少秒。"""
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.errors:
                raise RuntimeError(self.errors[0])
            if all(self.counts[r] > 0 for r in range(len(self.procs))):
                return time.time() - t0
            dead = [r for r, p in enumerate(self.procs) if not p.is_alive()]
            if dead:
                raise RuntimeError(f"采集进程 {dead} 已退出（返回码 {self.procs[dead[0]].exitcode}）")
            time.sleep(1.0)
        raise RuntimeError(f"{timeout:.0f} 秒内仍有采集进程没开始出决策")

    def pids(self) -> list[int]:
        return [p.pid for p in self.procs if p.pid]

    def close(self) -> None:
        engines = descendants(self.pids())
        self.stop.set()
        for p in self.procs:
            p.join(timeout=20)
        for p in self.procs:
            if p.is_alive():
                p.kill()
        time.sleep(1.0)
        for pid in engines:       # 兜底：被强制结束的采集进程留下的引擎
            try:
                os.kill(pid, 9)
            except OSError:
                pass
        self._drain_stop.set()
        self._drain.join(timeout=5)
        self.q.cancel_join_thread()


# ---------------------------------------------------------------- 3. 学习
class Learner:
    """与 train.py 同样的 PPO 更新（只为计时：动作从当前网络采样，优势与回报取随机数）。"""
    def __init__(self, cfg, model_cfg: ModelConfig, state: dict | None, device: torch.device):
        self.cfg, self.device = cfg, device
        self.net = PolicyValueNet(model_cfg).to(device)
        if state is not None:
            load_state(self.net, state)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr, eps=1e-5, fused=device.type == "cuda")
        self.data = None

    def prepare(self, buf: TrajBuffer) -> dict:
        n_want = self.cfg.batch_steps
        items = buf.newest(n_want)
        t0 = time.time()
        data = Ragged.concat([x["ragged"] for x in items])
        macro = np.concatenate([x["macro"] for x in items])
        t_concat = time.time() - t0
        if len(data) == 0:
            raise RuntimeError("还没有收到任何轨迹，无法测学习")
        idx = np.arange(n_want) % len(data)        # 不足一批时循环补足
        self.data = data.select(idx)
        self.macro = macro[idx]
        N = len(self.data)
        from sts2rl.features import collate_ragged
        acts, logps = [], []
        self.net.eval()
        with torch.no_grad():
            for s in range(0, N, 4096):
                b = collate_ragged(self.data.select(np.arange(s, min(N, s + 4096)))).to(self.device)
                logits, _ = self.net(b)
                d = torch.distributions.Categorical(logits=logits.float())
                a = d.sample(); acts.append(a.cpu()); logps.append(d.log_prob(a).cpu())
        rng = np.random.default_rng(0)
        self.actions = torch.cat(acts).numpy().astype(np.int64)
        self.old_logp = torch.cat(logps).numpy().astype(np.float32)
        self.adv = rng.standard_normal(N).astype(np.float32)
        self.ret = (0.5 * rng.standard_normal(N)).astype(np.float32)
        return {"steps": N, "unique_steps": int(min(len(data), n_want)), "macro_frac": round(float(self.macro.mean()), 4),
                "t_concat": round(t_concat, 3)}

    def update(self) -> float:
        from sts2rl.train import minibatches
        cfg, dev, net = self.cfg, self.device, self.net
        t0 = time.time()
        net.train()
        N = len(self.data)
        for _ in range(cfg.epochs):
            for idx, b in minibatches(self.data, np.random.permutation(N), cfg.minibatch, dev):
                a = torch.from_numpy(self.actions[idx]).to(dev); olp = torch.from_numpy(self.old_logp[idx]).to(dev)
                A = torch.from_numpy(self.adv[idx]).to(dev); R = torch.from_numpy(self.ret[idx]).to(dev)
                with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=cfg.learner_bf16 and dev.type == "cuda"):
                    logits, v = net(b)
                logits, v = logits.float(), v.float()
                dist = torch.distributions.Categorical(logits=logits)
                lp = dist.log_prob(a); ratio = torch.exp(lp - olp)
                l_pi = -torch.min(ratio * A, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * A).mean()
                l_v = F.mse_loss(v, R)
                ent_all = dist.entropy()
                ent_w = torch.where(b["macro"], cfg.ent_coef_macro, cfg.ent_coef)
                loss = l_pi + cfg.vf_coef * l_v - (ent_w * ent_all).mean()
                self.opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm); self.opt.step()
                with torch.no_grad():        # train.py 每个小批都把这几项取回 CPU，这里同样同步
                    (ratio - 1).abs().mean().item(); ent_all.mean().item()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        net.eval()
        return time.time() - t0


# ---------------------------------------------------------------- 主流程
def load_ckpt(path: str | None):
    """返回 (state, progress, 检查点里的模型配置或 None, 版本)。支持训练检查点（含 state）与只有权重的留档文件。"""
    if not path:
        return None, 0.0, None, None
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(blob, dict) and "state" in blob:
        model = (blob.get("config") or {}).get("model")
        return blob["state"], float(blob.get("progress", 0.0)), model, blob.get("version")
    return blob, 0.0, None, None


def window(fleet: Fleet, seconds: float, gpu: GpuSampler, learner: Learner | None = None, publish_fn=None) -> dict:
    """测一段：采集速度（每秒决策数）、CPU、显卡占用；有 learner 时同时反复更新，记每次更新的耗时。"""
    c0, s0, t0 = fleet.total(), cpu_times(), time.time()
    learn = []
    while time.time() - t0 < seconds:
        if learner is not None:
            learn.append(learner.update())
            if publish_fn:
                publish_fn()
        else:
            time.sleep(0.5)
        if fleet.errors:
            raise RuntimeError(fleet.errors[0])
    t1, c1, s1 = time.time(), fleet.total(), cpu_times()
    dt = t1 - t0
    g, gm = gpu.mean(t0, t1)
    out = {"sps": round((c1 - c0) / dt), "cpu": round(100 * (1 - (s1[1] - s0[1]) / max(1, s1[0] - s0[0])), 1), "gpu": g, "gpu_mem_gb": gm,
           "mem_gb": round(mem()[0], 1)}
    if learn:
        out["learn_s"] = round(float(np.median(learn)), 3); out["learn_n"] = len(learn)
    return out


def parse_grid(s: str) -> list[tuple[int, int]]:
    out = []
    for x in s.split(","):
        a, e = x.lower().split("x")
        out.append((int(a), int(e)))
    return out


def main():
    ap = argparse.ArgumentParser(description="租用机基准测试")
    ap.add_argument("--config", required=True, help="训练配置 JSON（奖励、课程、评测、模型等都取它；硬件项由本测试决定）")
    ap.add_argument("--ckpt", default="", help="检查点（latest.pt、备份或 best/ 留档）；不给则随机初始化（读数偏差较大）")
    ap.add_argument("--grid", default="", help="要测的组合（采集进程 x 每进程引擎），默认 12x16,16x16,12x20,16x20；--quick 为 12x16,16x16。"
                    "每个采集进程在显卡上约占 0.4 GB（A 上学习进程约占 15 GB），24 GB 显卡上采集进程不宜超过 18 个")
    ap.add_argument("--quick", action="store_true", help="少测几个组合、不测亲和性（约 5 分钟）")
    ap.add_argument("--warm", type=float, default=15.0, help="引擎都开始出决策之后再等几秒才计数")
    ap.add_argument("--collect-s", type=float, default=20.0, help="采集单独跑的计时秒数")
    ap.add_argument("--learn-s", type=float, default=25.0, help="采集 + 学习的计时秒数")
    ap.add_argument("--engine-s", type=float, default=15.0, help="引擎测试每项秒数；0 跳过引擎测试")
    ap.add_argument("--no-eval", action="store_true", help="不启动评测进程")
    ap.add_argument("--no-pin", action="store_true", help="不测亲和性")
    ap.add_argument("--device", default="", help="覆盖推理与学习设备（cpu 只用于本机试跑）")
    ap.add_argument("--batch-steps", type=int, default=0, help="覆盖每批决策数（只用于本机试跑）")
    ap.add_argument("--minibatch", type=int, default=0, help="覆盖小批大小（只用于本机试跑）")
    ap.add_argument("--pe", default="", help="手动指定性能核:能效核的逻辑 CPU，如 0-15:16-31（内核没有导出 cpu_core/cpu_atom 时用）")
    ap.add_argument("--mem-reserve-gb", type=float, default=12.0, help="推荐组合时给评测与系统预留的内存")
    ap.add_argument("--out", default="bench-out")
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    run_dir = out / "run"; run_dir.mkdir(exist_ok=True)
    T0 = time.time()
    info = machine_info()
    say("机器：", json.dumps(info, ensure_ascii=False))
    report: dict = {"machine": info, "args": vars(a), "engine": {}, "points": [], "pin": {}, "learn_alone": {}}

    def save():
        report["elapsed_s"] = round(time.time() - T0)
        (out / "bench.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    cfg = TrainConfig.from_args(["--config", a.config])
    cfg.run_dir = str(run_dir)
    state, progress, ck_model, ck_ver = load_ckpt(a.ckpt)
    if ck_model:
        for k, v in ck_model.items():
            setattr(cfg.model, k, v)
    if a.device:
        cfg.actor_device = cfg.learner_device = a.device
    elif not torch.cuda.is_available():
        say("警告：没有可用的显卡，推理与学习都在 CPU 上（只适合试跑）")
        cfg.actor_device = cfg.learner_device = "cpu"
    if a.batch_steps:
        cfg.batch_steps = a.batch_steps
    if a.minibatch:
        cfg.minibatch = a.minibatch
    report["ckpt"] = {"path": a.ckpt, "version": ck_ver, "progress": progress, "model": cfg.to_dict()["model"]}
    say(f"检查点：{a.ckpt or '（无，随机初始化）'}{f'，版本 {ck_ver}' if ck_ver is not None else ''}；模型 d={cfg.model.d} 层数={cfg.model.layers}；"
        f"每批 {cfg.batch_steps} / 小批 {cfg.minibatch} / {cfg.epochs} 遍；推理 {cfg.actor_device}，学习 {cfg.learner_device}")

    cpus = os.cpu_count() or 8
    if a.grid:
        grid = parse_grid(a.grid)
    elif a.quick:
        grid = [(12, 16), (16, 16)]
    else:
        grid = [(12, 16), (16, 16), (12, 20), (16, 20)]
    series: dict[int, list[int]] = {}
    for n_act, envs in grid:
        series.setdefault(envs, []).append(n_act)

    # ---- 1. 引擎
    if a.engine_s > 0:
        for name, n, mode in (("单引擎·战斗", 1, "combat"), ("单引擎·整局", 1, "run"), (f"{cpus} 引擎·战斗（随机策略上限）", cpus, "combat")):
            r, st = engine_rate(n, a.engine_s, mode)
            report["engine"][name] = {"engines": n, "mode": mode, "sps": round(r), "startup_s": round(st, 1)}
            say(f"引擎 {name}：{r:,.0f} 决策/秒（启动 {st:.1f} 秒）")
        save()

    # ---- 权重文件与评测
    use_tf32()
    model_cpu = PolicyValueNet(cfg.model).eval()
    if state is not None:
        load_state(model_cpu, state)
    from sts2rl.train import publish
    shm = Path("/dev/shm") if Path("/dev/shm").exists() else run_dir
    wpath = shm / f"sts2rl_bench_{os.getpid()}.pt"
    ver = [1]

    def publish_fn():
        ver[0] += 1
        publish(model_cpu, wpath, ver[0], progress, 0.0)
    publish_fn()

    gpu = GpuSampler()
    torch.set_num_threads(cfg.learner_threads)
    device = torch.device(cfg.learner_device)
    learner = Learner(cfg, cfg.model, state, device)
    buf = TrajBuffer(keep_steps=3 * cfg.batch_steps)
    ctx = mp.get_context("spawn")
    eval_proc = None
    eval_stop = ctx.Event()
    if not a.no_eval and cfg.eval_runs > 0 and cfg.eval_envs > 0:
        from sts2rl.evaluator import run_evaluator
        eval_proc = ctx.Process(target=run_evaluator, args=(cfg, str(wpath), str(run_dir), eval_stop), daemon=False)
        eval_proc.start()
        say(f"评测进程已启动：{cfg.eval_workers} 个工作进程 × {cfg.eval_envs} 个引擎，限 {cfg.eval_cpu_cores} 核（SCHED_IDLE）")
    report["eval"] = None if eval_proc is None else {"workers": cfg.eval_workers, "envs": cfg.eval_envs, "cpu_cores": cfg.eval_cpu_cores}

    fleet: Fleet | None = None
    max_actors = max(n for n, _ in grid) + 4
    p_cpus, e_cpus = cpu_groups()
    if a.pe:
        p_cpus, e_cpus = (_parse_cpus(x) for x in a.pe.split(":"))
    all_cpus = sorted(os.sched_getaffinity(0))

    def measure_point(fl: Fleet, tag: str) -> dict:
        r1 = window(fl, a.collect_s, gpu)
        if not learner_ready[0]:
            report["learn_batch"] = learner.prepare(buf); learner_ready[0] = True
            say(f"  学习用的数据：{report['learn_batch']}")
            learner.update()       # 预热（cuDNN、分配器）
        r2 = window(fl, a.learn_s, gpu, learner, publish_fn)
        cap = cfg.batch_steps / (r2["learn_s"] + OVERHEAD_S)
        pt = {"actors": fl.actors, "envs_per_actor": fl.envs, "engines": fl.actors * fl.envs, "collect": r1, "collect_learn": r2,
              "learn_cap_sps": round(cap), "predicted_sps": round(min(r2["sps"], cap)), "tag": tag}
        say(f"  {fl.actors:2d} × {fl.envs:2d} = {fl.actors * fl.envs:3d} 引擎：采集单独 {r1['sps']:6,} 决策/秒（CPU {r1['cpu']}%，显卡 {r1['gpu']}%）；"
            f"采集+学习 {r2['sps']:6,} 决策/秒、学习一次 {r2['learn_s']:.2f} 秒（上限 {cap:,.0f}）（CPU {r2['cpu']}%，显卡 {r2['gpu']}%，"
            f"显存峰值 {r2['gpu_mem_gb']} GB）；内存 {r2['mem_gb']} GB → 预计 {pt['predicted_sps']:,} 决策/秒")
        return pt

    learner_ready = [False]
    try:
        # ---- 2. 采集网格
        for envs, counts in series.items():
            counts = sorted(counts)
            if fleet is not None:
                fleet.close(); fleet = None
            fleet = Fleet(cfg, envs, str(wpath), buf, max_actors)
            for n_act in counts:
                t_add = time.time()
                try:
                    fleet.add(n_act - fleet.actors)
                    fleet.wait_stepping(300)
                    say(f"[{time.time() - T0:5.0f}s] {fleet.actors} 个采集进程 × {envs} 个引擎已全部出决策（追加用时 {time.time() - t_add:.0f} 秒），预热 {a.warm:.0f} 秒")
                    time.sleep(a.warm)
                    report["points"].append(measure_point(fleet, "grid")); save()
                except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                    # 显存或内存不够、采集进程出错：记下来，这一组更大的组合不再测，换下一组
                    msg = str(e).strip().splitlines()[-1][:300] if str(e).strip() else type(e).__name__
                    say(f"  {n_act} × {envs} 失败：{msg}")
                    report.setdefault("failed", []).append({"actors": n_act, "envs_per_actor": envs, "error": msg}); save()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    fleet.close(); fleet = None
                    break

        # ---- 推荐
        mem_total = mem()[1]
        if not report["points"]:
            raise RuntimeError("没有一个组合测成功，见上面的错误")
        ok = [p for p in report["points"] if p["collect_learn"]["mem_gb"] + a.mem_reserve_gb <= 0.9 * mem_total] or report["points"]
        best = max(p["predicted_sps"] for p in ok)
        rec = min((p for p in ok if p["predicted_sps"] >= 0.97 * best), key=lambda p: (p["engines"], -p["predicted_sps"]))
        report["recommended_point"] = rec

        # ---- 3. 亲和性
        if p_cpus and e_cpus and not a.no_pin and not a.quick:
            if fleet is None or (fleet.actors, fleet.envs) != (rec["actors"], rec["envs_per_actor"]):
                if fleet is not None:
                    fleet.close()
                fleet = Fleet(cfg, rec["envs_per_actor"], str(wpath), buf, max_actors)
                fleet.add(rec["actors"]); fleet.wait_stepping(300); time.sleep(a.warm)
            say(f"亲和性（{rec['actors']} × {rec['envs_per_actor']}）：性能核 {p_cpus[0]}–{p_cpus[-1]}，能效核 {e_cpus[0]}–{e_cpus[-1]}")
            me = [os.getpid()]
            ev = ([eval_proc.pid] + descendants([eval_proc.pid])) if eval_proc else []
            for mode in ("none", "p", "pe"):
                engines = descendants(fleet.pids())
                set_affinity(me + fleet.pids(), all_cpus); set_affinity(engines + ev, all_cpus)
                if mode in ("p", "pe"):
                    set_affinity(me + fleet.pids(), p_cpus); set_affinity(ev, e_cpus)
                if mode == "pe":
                    set_affinity(engines, e_cpus)
                time.sleep(3)
                r = window(fleet, a.learn_s, gpu, learner, publish_fn)
                cap = cfg.batch_steps / (r["learn_s"] + OVERHEAD_S)
                r["learn_cap_sps"] = round(cap); r["predicted_sps"] = round(min(r["sps"], cap))
                report["pin"][mode] = r
                desc = {"none": "不绑定", "p": "学习与采集→性能核，评测→能效核，引擎不限", "pe": "学习与采集→性能核，评测与引擎→能效核"}[mode]
                say(f"  {desc}：采集+学习 {r['sps']:,} 决策/秒、学习一次 {r['learn_s']:.2f} 秒 → 预计 {r['predicted_sps']:,}（CPU {r['cpu']}%）")
                save()
            set_affinity(me + fleet.pids() + descendants(fleet.pids()) + ev, all_cpus)
    finally:
        if fleet is not None:
            fleet.close()
        if eval_proc is not None:
            eval_stop.set(); eval_proc.join(timeout=20)
            if eval_proc.is_alive():
                eval_proc.kill()
            for pid in descendants([eval_proc.pid]) if eval_proc.pid else []:
                try:
                    os.kill(pid, 9)
                except OSError:
                    pass
        gpu.stop.set()

    # ---- 4. 学习单独跑
    if learner_ready[0]:
        time.sleep(3)
        ts = [learner.update() for _ in range(3)]
        report["learn_alone"] = {"learn_s": round(float(np.median(ts)), 3), "sps_cap": round(cfg.batch_steps / (float(np.median(ts)) + OVERHEAD_S))}
        say(f"学习单独跑（显卡独占）：一次 {np.median(ts):.2f} 秒（{cfg.batch_steps} 步 × {cfg.epochs} 遍，小批 {cfg.minibatch}）")
    try:
        wpath.unlink()
    except OSError:
        pass

    # ---- 汇总
    rec = report.get("recommended_point")
    if rec:
        hw = {k: getattr(cfg, k) for k in HW_KEYS}
        hw.update(actors=rec["actors"], envs_per_actor=rec["envs_per_actor"])
        report["recommended_hw"] = hw
        (out / "recommend_hw.json").write_text(json.dumps(hw, ensure_ascii=False, indent=1), encoding="utf-8")
        bottleneck = "采集（CPU）" if rec["collect_learn"]["sps"] <= rec["learn_cap_sps"] else f"学习（{cfg.learner_device}）"
        say(f"\n推荐：{rec['actors']} 个采集进程 × {rec['envs_per_actor']} 个引擎，预计约 {rec['predicted_sps']:,} 决策/秒，瓶颈在{bottleneck}；"
            f"内存约 {rec['collect_learn']['mem_gb']} / {mem()[1]:.0f} GB")
        pin = report.get("pin") or {}
        if pin.get("none"):
            base = pin["none"]["predicted_sps"]
            for m in ("p", "pe"):
                if m in pin:
                    say(f"  亲和性 {m}：相对不绑定 {100 * (pin[m]['predicted_sps'] / max(1, base) - 1):+.1f}%")
        say(f"硬件配置：{out / 'recommend_hw.json'}（迁移时可用 --hw @该文件）")
    save()
    say(f"报告：{out / 'bench.json'}；总用时 {time.time() - T0:.0f} 秒")


if __name__ == "__main__":
    main()
