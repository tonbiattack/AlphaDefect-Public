"""搜索标签的 KL 辅助损失（2026-11-17，训练中用搜索的试点）。

用户：“开始试点，两台机器还是给你，充分利用”（试点 = 搜索结果当辅助训练目标：采集照常由当前策略出手、PPO 不变，
另对搜索过的局面加一项 KL(搜索改进后的分布 ‖ 当前策略)，reports/整夜搜索探究总结.md 第六节、对话里的方案 A）。

标签由 tools/future/search_stream.py（内部调 run_search.py --labels）产出：整局模式、局外贪心，Boss 与精英战里首选概率 < 0.9 的决策做 32 次模拟的树搜索，
每个搜过的根写一行 {"v": 产标签的网络版本, "frame": 根帧原始字节（base64）, "tgt": 改进后的分布（候选数长）, ...}。
改进后的分布与搜索的最终选法同一个算子（tree_distill.py）：访问数 ≥ 4 的候选按语义键分组，组 k 的质量 ∝ P_k · exp(Q_k / λ)（λ 0.06），总质量不变，组内按原概率分。

本模块：一个线程每隔几秒扫 label_dir 下所有 labels_*.jsonl 的新行放进缓冲（最多 buffer 个，旧的挤掉）；另一个线程随机取 batch 个（max_age > 0 时只取
最近 max_age 版网络产的），解析、组批（与学习端的小批同一套 collate_ragged / packed_meta），放进队列。学习端每个小批非阻塞地取一批，取不到就这个小批不加。
"""
from __future__ import annotations

import base64
import glob
import json
import os
import queue
import random
import threading
import time
from collections import deque

import numpy as np
import torch

from sts2env.protocol import parse

from .actor import MODE_RUN
from .features import Ragged, collate_ragged
from .model import packed_meta


class SearchKL:
    def __init__(self, label_dir: str, batch: int, buffer: int, max_age: int, min_n: int, packed: bool, device, seed: int = 0):
        self.dir, self.batch, self.max_age, self.min_n, self.packed, self.device = label_dir, batch, max_age, min_n, packed, device
        self.buf: deque = deque(maxlen=buffer)
        self.lock = threading.Lock()
        self.offsets: dict[str, int] = {}
        self.version = 0            # 学习端当前版本（max_age 过滤用），学习端每版更新
        self.rng = random.Random(seed)
        self.q: queue.Queue = queue.Queue(maxsize=4)
        self.stat = {"read": 0, "bad": 0, "batches": 0, "empty": 0, "err": ""}
        threading.Thread(target=self._scan_loop, daemon=True, name="search_kl_scan").start()
        threading.Thread(target=self._produce_loop, daemon=True, name="search_kl_batch").start()

    # ------------------------------------------------------------------ 读标签
    def _scan_once(self) -> None:
        files = sorted(glob.glob(os.path.join(self.dir, "**", "labels_*.jsonl"), recursive=True), key=lambda f: os.path.getmtime(f))
        for f in files:
            off = self.offsets.get(f, 0)
            try:
                size = os.path.getsize(f)
            except OSError:
                continue
            if size <= off:
                continue
            with open(f, "rb") as fh:
                fh.seek(off)
                data = fh.read(size - off)
            end = data.rfind(b"\n")
            if end < 0:
                continue
            self.offsets[f] = off + end + 1
            new = []
            for line in data[:end].split(b"\n"):
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                    new.append((base64.b64decode(r["frame"]), np.asarray(r["tgt"], np.float32), int(r.get("v", 0))))
                except Exception:
                    self.stat["bad"] += 1
            with self.lock:
                self.buf.extend(new)
            self.stat["read"] += len(new)

    def _scan_loop(self) -> None:
        while True:
            try:
                if self.dir and os.path.isdir(self.dir):
                    self._scan_once()
            except Exception as e:      # 不让读标签的异常打断训练
                self.stat["err"] = f"读标签：{e}"[:200]
            time.sleep(10.0)

    # ------------------------------------------------------------------ 组批
    def _sample(self):
        with self.lock:
            n = len(self.buf)
            if n < self.min_n:
                return None
            lo = self.version - self.max_age if self.max_age > 0 else None
            out, tries = [], 0
            while len(out) < self.batch and tries < self.batch * 20:
                it = self.buf[self.rng.randrange(n)]
                tries += 1
                if lo is None or it[2] >= lo:
                    out.append(it)
        return out if len(out) >= max(8, self.batch // 4) else None

    def _produce_loop(self) -> None:
        while True:
            try:
                items = self._sample()
                if items is None:
                    time.sleep(5.0)
                    continue
                obs = [parse(raw) for raw, _, _ in items]
                bt = collate_ragged(Ragged.from_obs(obs, np.full(len(obs), MODE_RUN)))
                d = dict(bt.__dict__)
                if self.packed:
                    d.update(packed_meta(bt.core_idx, bt.core_mask))
                width = max(len(t) for _, t, _ in items)
                tgt = np.zeros((len(items), width), np.float32)
                for j, (_, t, _) in enumerate(items):
                    tgt[j, :len(t)] = t
                ages = np.array([self.version - v for _, _, v in items], np.float32)

                def cpu(v):
                    return torch.from_numpy(np.ascontiguousarray(v)).pin_memory() if torch.cuda.is_available() else torch.from_numpy(np.ascontiguousarray(v))
                dc = {k: (cpu(v) if isinstance(v, np.ndarray) else [(cpu(i), kg) for i, kg in v] if k == "groups" else v) for k, v in d.items()}
                self.q.put((dc, cpu(tgt), float(ages.mean())))
            except Exception as e:
                self.stat["err"] = f"组批：{e}"[:200]
                time.sleep(5.0)

    def next_batch(self):
        """非阻塞：有现成的一批就拷到显卡返回 (批, 目标分布, 平均标签年龄)，没有返回 None。"""
        try:
            dc, tgt, age = self.q.get_nowait()
        except queue.Empty:
            self.stat["empty"] += 1
            return None
        dev = self.device
        d = {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else [(i.to(dev, non_blocking=True), kg) for i, kg in v] if k == "groups" else v)
             for k, v in dc.items()}
        self.stat["batches"] += 1
        return d, tgt.to(dev, non_blocking=True), age

    def size(self) -> int:
        with self.lock:
            return len(self.buf)


def kl_loss(logits: torch.Tensor, tgt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """KL(tgt ‖ softmax(logits))，按行平均；另返回首选与目标首选一致的比例。logits 里非法候选为 -inf，tgt 在那里为 0。"""
    a = logits.shape[1]
    t = tgt[:, :a] if tgt.shape[1] >= a else torch.nn.functional.pad(tgt, (0, a - tgt.shape[1]))
    lp = torch.log_softmax(logits.float(), -1)
    pos = t > 0
    ce = -(t * torch.where(pos, lp, torch.zeros_like(lp))).sum(-1)
    ent = -(t * torch.log(t.clamp(min=1e-12))).sum(-1)
    agree = (logits.argmax(-1) == t.argmax(-1)).float().mean()
    return (ce - ent).mean(), agree
