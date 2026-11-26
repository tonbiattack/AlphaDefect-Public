"""搜索蒸馏（2026-10-27 起）：把 Boss 战确定化搜索选出的动作当作策略的辅助目标。

标签由 A 上的 tools/distill/label_worker.py 产出（每个搜索过的决策一条：观测、网络原本的选择、搜索的选择），
经本机中转 tools/distill/relay.py 放进 run_dir/distill/inbox/*.pkl.gz。学习进程每 distill_poll_s 秒收一次，
收进来的文件随即删掉；标签按产出时用的检查点版本计龄，超过 distill_max_age 版或总数超过 distill_cap 时丢最旧的。

每次 PPO 更新之后（与牌组打分头同样的做法）抽 distill_batch 条训练一步：
损失 = distill_coef × 加权交叉熵（目标为搜索的选择；改选了的权重 1，没改选的 distill_same_w）。
没改选的标签只说明“没有哪个候选显著更好”，权重给小，免得只是把策略压得更尖。

诊断（每批新到的标签在训练之前先看一眼）：当前网络的首选与搜索的选择一致的比例，改选与没改选的分开记。
产出这些标签的网络在改选处的首选必定是原来那个（一致率为 0），所以改选处的一致率上升，说明网络在没见过的局面上学到了搜索的选择。
"""
from __future__ import annotations

import gzip
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sts2env.protocol import FORMAT_VERSION

from .features import Ragged, collate_ragged


class DistillBuffer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.inbox = Path(cfg.run_dir) / "distill" / "inbox"
        self.inbox.mkdir(parents=True, exist_ok=True)
        self.chunks: list[dict] = []      # 每个标签文件一块：view、label、weight、version、changed、act
        self.last_poll = 0.0
        self.fresh = [0, 0, 0, 0]         # 本次记录以来新到的：[改选条数, 其中网络首选一致, 没改选条数, 其中一致]
        self.new_files = self.bad_files = 0
        self.steps = 0

    def __len__(self) -> int:
        return sum(len(c["label"]) for c in self.chunks)

    @torch.no_grad()
    def _agree(self, net, view: Ragged, label: np.ndarray, device, bf16: bool) -> np.ndarray:
        out = []
        for s in range(0, len(label), 512):
            idx = np.arange(s, min(len(label), s + 512))
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16 and device.type == "cuda"):
                logits, _ = net(collate_ragged(view.select(idx)).to(device))
            out.append(logits.float().argmax(-1).cpu().numpy())
        return np.concatenate(out) == label

    def poll(self, net, version: int, device, bf16: bool) -> None:
        """收新到的标签文件（并先记一致率），丢掉过旧的标签。"""
        if time.time() - self.last_poll < self.cfg.distill_poll_s:
            return
        self.last_poll = time.time()
        for f in sorted(self.inbox.glob("*.pkl.gz")):
            try:
                with gzip.open(f, "rb") as fh:
                    rec = pickle.load(fh)
                ok = rec.get("obs_format") == FORMAT_VERSION and len(rec["label"]) > 0
            except Exception:
                ok, rec = False, None
            f.unlink(missing_ok=True)
            if not ok:
                self.bad_files += 1      # 观测格式不对或文件坏了：丢掉，记数
                continue
            label, changed = rec["label"], rec["changed"]
            was = net.training
            net.eval()
            agree = self._agree(net, rec["view"], label, device, bf16)
            net.train(was)
            self.fresh[0] += int(changed.sum()); self.fresh[1] += int((agree & changed).sum())
            self.fresh[2] += int((~changed).sum()); self.fresh[3] += int((agree & ~changed).sum())
            self.chunks.append({"view": rec["view"], "label": label, "changed": changed, "act": rec["act"], "version": int(rec["version"]),
                                "weight": np.where(changed, 1.0, self.cfg.distill_same_w).astype(np.float32)})
            self.new_files += 1
        self.chunks = [c for c in self.chunks if version - c["version"] <= self.cfg.distill_max_age]
        while len(self.chunks) > 1 and len(self) > self.cfg.distill_cap:
            self.chunks.pop(0)

    def step(self, net, opt, device, bf16: bool) -> dict:
        """抽一批训练一步，返回这一步的损失与命中率（训练之前的）。标签不够一批时不训练。"""
        n = len(self)
        if not self.chunks or n < self.cfg.distill_batch or self.cfg.distill_coef <= 0:
            return {}
        sizes = np.array([len(c["label"]) for c in self.chunks])
        pick = np.sort(np.random.choice(n, self.cfg.distill_batch, replace=False))
        which = np.searchsorted(np.cumsum(sizes), pick, side="right")
        base = np.concatenate([[0], np.cumsum(sizes)])[which]
        parts, labels, weights, changed = [], [], [], []
        for ci in np.unique(which):
            idx = pick[which == ci] - base[which == ci]
            c = self.chunks[ci]
            parts.append(c["view"].select(idx)); labels.append(c["label"][idx]); weights.append(c["weight"][idx]); changed.append(c["changed"][idx])
        b = collate_ragged(Ragged.concat(parts)).to(device)
        y = torch.from_numpy(np.concatenate(labels)).to(device)
        w = torch.from_numpy(np.concatenate(weights)).to(device)
        ch = np.concatenate(changed)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16 and device.type == "cuda"):
            logits, _ = net(b)
        logits = logits.float()
        ce = F.cross_entropy(logits, y, reduction="none")
        loss = (w * ce).sum() / w.sum()
        opt.zero_grad(set_to_none=True); (self.cfg.distill_coef * loss).backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), self.cfg.max_grad_norm); opt.step()
        self.steps += 1
        with torch.no_grad():
            hit = (logits.argmax(-1) == y).cpu().numpy()
        return {"distill_loss": round(loss.item(), 4), "distill_acc": [round(float(hit[ch].mean()), 4) if ch.any() else None,
                                                                     round(float(hit[~ch].mean()), 4) if (~ch).any() else None]}

    def stats(self) -> dict:
        """记到 log.jsonl：缓冲里的标签数（其中改选的）、本次记录以来新到的一致率计数、收到的文件数、坏文件数、已训练步数。"""
        out = {"distill_n": [len(self), int(sum(c["changed"].sum() for c in self.chunks))], "distill_fresh": self.fresh,
               "distill_files": self.new_files, "distill_bad": self.bad_files, "distill_steps": self.steps}
        self.fresh = [0, 0, 0, 0]; self.new_files = 0
        return out
