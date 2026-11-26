"""机器资源采样（学习进程里的后台线程）：每 30 秒记一次 CPU、内存、显卡，写 run_dir/resource.jsonl。"""
from __future__ import annotations

import json
import subprocess
import threading
import time
from pathlib import Path


def _cpu_times():
    with open("/proc/stat") as f:
        v = [int(x) for x in f.readline().split()[1:]]
    idle = v[3] + v[4]
    return sum(v), idle


def _mem():
    m = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":", 1)
            m[k] = int(v.split()[0])
    return (m["MemTotal"] - m["MemAvailable"]) / 1e6, m["MemTotal"] / 1e6


def _gpu():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
        u, mu, mt, t = [float(x) for x in out[0].split(",")]
        return {"gpu_util": u, "gpu_mem_gb": mu / 1024, "gpu_mem_total_gb": mt / 1024, "gpu_temp": t}
    except Exception:
        return {}


def start_sampler(run_dir: Path, stop: threading.Event, every: float = 30.0) -> threading.Thread:
    def loop():
        prev = _cpu_times()
        while not stop.wait(every):
            cur = _cpu_times()
            dt, di = cur[0] - prev[0], cur[1] - prev[1]
            prev = cur
            used, total = _mem()
            rec = {"time": time.time(), "cpu_util": round(100 * (1 - di / max(dt, 1)), 1), "mem_gb": round(used, 1), "mem_total_gb": round(total, 1), **_gpu()}
            with open(run_dir / "resource.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return t
