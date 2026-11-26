"""求解器筛选的 Boss 练习（2026-10-25 起）。

采集进程在整局里打到各幕 Boss 时（第三幕起于 2026-10-26；A10 最后一幕连打两个 Boss，第二个也导出），按时间间隔把开打前的玩家快照导出到 run_dir/solver/pending/（一份一个文件），
本机用 Combat Solver 模组求解（tools/solver/label_loop.py），把结果追加到 run_dir/solver/labeled.jsonl。
求解器知道本场全部随机数，“能赢”只说明这副牌、这个血量在完全信息下有赢法，是上界；它只用来挑练习题，不给模型模仿。

战斗课程的 Boss 名额里，用真实快照的那部分按 solver_share 改从“求解器判定能赢”的局面里抽：
模型还没赢过的局面权重最高，赢得越多权重越低；一半用求解器解过的那个种子（确定有赢法），一半换新种子（防止背下这一场）。
每个采集进程各自读标签、各自统计胜负。
"""
from __future__ import annotations

import json
import os
import time
from collections import deque
from pathlib import Path

import numpy as np


class SolverPool:
    def __init__(self, cfg, rank: int):
        self.cfg, self.rank = cfg, rank
        self.dir = Path(cfg.run_dir) / "solver"
        self.pending = self.dir / "pending"
        self.pending.mkdir(parents=True, exist_ok=True)
        self.labeled = self.dir / "labeled.jsonl"
        self.offset = 0
        self.items = [deque(maxlen=cfg.solver_pool) for _ in range(3)]   # 各幕 Boss，能赢的局面
        self.by_id: dict[str, dict] = {}
        self.last_export = [0.0, 0.0, 0.0]
        self.exported = 0
        # 固定考卷里的局面不拿来练（见 tools/exam_build.py）
        try:
            self.exam_ids = {it["id"] for it in json.loads((Path(cfg.run_dir) / "exam.json").read_text(encoding="utf-8"))["items"]}
        except Exception:
            self.exam_ids = set()

    def maybe_export(self, combat: dict, snap: str) -> None:
        """整局里一场 Boss 战开打：按幕的时间间隔导出开打前的快照（积压太多时不导出）。"""
        act = combat["act"]
        if combat["room"] != "Boss" or act >= len(self.cfg.solver_export_every_s):
            return
        every = self.cfg.solver_export_every_s[act]
        now = time.time()
        if every <= 0 or now - self.last_export[act] < every:
            return
        try:
            if len(os.listdir(self.pending)) >= self.cfg.solver_pending_max:
                return
        except OSError:
            return
        self.last_export[act] = now
        self.exported += 1
        sid = f"r{self.rank}_{int(now)}_{self.exported}"
        rec = {"id": sid, "label": {"act": act, "encounter": combat["encounter"], "deck": combat.get("deck"), "hp": combat.get("hp"),
                                    "max_hp": combat.get("max_hp")},
               "spec": {"character": self.cfg.character, "ascension": 10, "seed": f"V{sid}", "act": act, "encounter": combat["encounter"],
                        "player_snapshot": snap}}
        tmp = self.pending / f".{sid}.tmp"
        tmp.write_text(json.dumps(rec, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(tmp, self.pending / f"{sid}.json")

    def refresh(self) -> None:
        """读入新追加的标签（只读到最后一个完整行），能赢的放进练习池。"""
        try:
            size = self.labeled.stat().st_size
        except OSError:
            return
        if size <= self.offset:
            return
        with open(self.labeled, "rb") as f:
            f.seek(self.offset)
            data = f.read(size - self.offset)
        end = data.rfind(b"\n")
        if end < 0:
            return
        self.offset += end + 1
        for line in data[:end].split(b"\n"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("win") and r.get("act") in (0, 1, 2) and r["id"] not in self.by_id and r["id"] not in self.exam_ids:
                item = {"id": r["id"], "spec": r["spec"], "end_hp": r.get("end_hp"), "attempts": 0, "wins": 0}
                pool = self.items[r["act"]]
                if len(pool) == pool.maxlen:
                    self.by_id.pop(pool[0]["id"], None)
                pool.append(item)
                self.by_id[r["id"]] = item

    def sample(self, act: int, rng: np.random.Generator, counter: int) -> dict | None:
        """按权重抽一个能赢的局面：模型还没赢过的为 1，赢过 k 次的为 1/(1+2k)；没有则返回 None。"""
        if act > 2 or not self.items[act]:
            return None
        pool = list(self.items[act])
        w = np.array([1.0 if x["wins"] == 0 else 1.0 / (1 + 2 * x["wins"]) for x in pool])
        x = pool[int(rng.choice(len(pool), p=w / w.sum()))]
        spec = dict(x["spec"])
        if rng.random() >= self.cfg.solver_same_seed:
            spec["seed"] = f"W{counter}_{int(rng.integers(1 << 30))}"
        spec.update(room="Boss", slot="boss", cert=x["id"])
        return spec

    def record(self, cert: str, win: bool) -> None:
        x = self.by_id.get(cert)
        if x is not None:
            x["attempts"] += 1
            x["wins"] += int(win)


class LabelStats:
    """学习进程用：统计标签文件里已标注的局面数和其中能赢的数（各幕），增量读取。"""

    def __init__(self, run_dir: str):
        self.path = Path(run_dir) / "solver" / "labeled.jsonl"
        self.pending = Path(run_dir) / "solver" / "pending"
        self.offset = 0
        self.counts = [[0, 0], [0, 0], [0, 0]]       # [幕][能赢, 已标注]

    def update(self) -> dict:
        try:
            size = self.path.stat().st_size
            if size > self.offset:
                with open(self.path, "rb") as f:
                    f.seek(self.offset)
                    data = f.read(size - self.offset)
                end = data.rfind(b"\n")
                if end >= 0:
                    self.offset += end + 1
                    for line in data[:end].split(b"\n"):
                        try:
                            r = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if r.get("act") in (0, 1, 2):
                            self.counts[r["act"]][0] += int(bool(r.get("win")))
                            self.counts[r["act"]][1] += 1
        except OSError:
            pass
        try:
            pending = sum(1 for p in self.pending.iterdir() if p.suffix == ".json")
        except OSError:
            pending = None
        return {"solver_labels": [list(c) for c in self.counts], "solver_pending": pending}
