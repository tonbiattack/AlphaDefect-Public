"""采集端用搜索出手（2026-11-17，局外适应搜索战斗）。

用户：“如果测下来没收益，考虑让模型在训练过程中也能用搜索，也就是说培养局外决策适应搜索战斗能力的水平”；“现在开始准备”。

做法：整局里 cfg.search_collect_rooms（默认 Boss、精英）的战斗照部署的方式出手——首选概率 ≥ tau（不犹豫）取概率最大的动作，犹豫时做整局模式的树搜索
（tools/future/run_search.py 的 RunTreeRunner，32 次模拟、λ 0.06，同整局千种测试里 68.4% 那组），搜索改选就用搜索的。这些步记为 det（确定性的部署出手）：
学习端不让它们进策略梯度（动作不是策略采样的），回报照常算，所以局外决策的优势与价值头学到的是“战斗有搜索时”的局面价值。局外决策与其余战斗照常采样。

世界跟根：每局的 reset 请求（t.req）+ 这一局已走的全部动作（t.actions），推演进程 clone 模式快照、恢复（同 run_search）。
一个采集进程同时打几局时，换局要从头重放，所以用它的采集进程宜每进程 1 个引擎（envs_per_actor 1）。
网络：建搜索器时把采集进程当前的权重写成临时检查点给它载入；之后采集进程换权重（版本变了）就把权重拷进搜索器的网络。
"""
from __future__ import annotations

import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]


class ActorSearch:
    def __init__(self, cfg, net, device, rank: int, version: int):
        if cfg.skip_easy:
            raise SystemExit("search_collect_rooms 与 skip_easy 不能同时开（跳过战斗的局没法照动作重放）")
        if str(REPO / "tools" / "future") not in sys.path:
            sys.path.insert(0, str(REPO / "tools" / "future"))
        import run_search as RS
        self.rooms = {r for r in cfg.search_collect_rooms.split(",") if r}
        self.net = net
        out = os.path.join(cfg.run_dir, "search_collect")
        os.makedirs(out, exist_ok=True)
        tmp = f"/dev/shm/sts2rl_srch_{os.getpid()}.pt"
        torch.save({"version": int(version), "config": cfg.to_dict(),
                    "state": {k: v.detach().cpu() for k, v in net.state_dict().items()}}, tmp)
        try:
            a = RS.parse(["--seeds", "-", "--ckpt", tmp, "--out", out, "--sims", str(cfg.search_collect_sims), "--lam", str(cfg.search_collect_lam),
                          "--procs", str(cfg.search_collect_procs), "--device", str(device), "--rooms", cfg.search_collect_rooms,
                          "--character", cfg.character, "--tau", str(cfg.search_collect_tau), "--min-visits", str(cfg.search_collect_min_visits)]
                         + (["--dll", cfg.search_collect_dll] if cfg.search_collect_dll else []))
            a.k_shard = rank
            self.R = RS.RunTreeRunner(a)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        self.R.acc = {}
        self.version = int(version)
        self.stat = {"searches": 0, "changed": 0, "det": 0, "ms": 0.0}

    def sync(self, version: int) -> None:
        if int(version) != self.version:
            self.R.net.load_state_dict(self.net.state_dict())
            self.version = int(version)

    def wants(self, t, o) -> bool:
        return bool(o.in_combat) and t.room in self.rooms

    def decide(self, t, o, p: np.ndarray, version: int) -> tuple[int, bool]:
        """部署方式的出手：返回 (动作, 是否做了搜索)。p 为策略在全部合法候选上的概率（float64）。"""
        g = int(np.argmax(p))
        self.stat["det"] += 1
        if len(p) <= 1 or not self.R.gated(p):
            return g, False
        self.sync(version)
        R = self.R
        R.cur_key = ("run", t.spec.get("seed"), id(t))
        R.ctx = t.srch_ctx or {"boss3_first": False, "floor0": int(o.floor)}
        if len(R.acc) > 64:
            R.acc = {}
        hist = list(t.actions)
        dec = zlib.crc32(f"{t.spec.get('seed')}:{len(hist)}".encode()) & 0x3FFFFFFF
        t0 = time.perf_counter()
        res = R.search(t.req, hist, o, p, dec)
        self.stat["ms"] += (time.perf_counter() - t0) * 1000.0
        self.stat["searches"] += 1
        b = int(res["best"])
        keys = [k[:-1] for k in R.X.cand_keys(o)]
        if keys[b] != keys[g]:
            self.stat["changed"] += 1
            return b, True
        return g, True

    def close(self) -> None:
        try:
            self.R.close()
        except Exception:
            pass
