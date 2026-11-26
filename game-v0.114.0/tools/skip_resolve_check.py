"""引擎 resolve 请求的检验（reports/跳过稳赢战斗方案.md 阶段 0）：整局里遇到普通战、精英战就用 resolve 直接结束，
血量设为开战血量减 --dhp、身上有药水就去掉第一瓶，核对下一帧（奖励界面）的血量与药水，再照常往下打（按策略贪心）。

  CMB_DEPLOY=/root/skipdev/sls2 /root/skipdev/sls2/.venv/bin/python -I skip_resolve_check.py --seeds 4 --ckpt /root/cmbpred/w.pt
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
from pathlib import Path

DEPLOY = os.environ.get("CMB_DEPLOY", "/root/skipdev/sls2")
sys.path.insert(0, os.path.join(DEPLOY, "python"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from sts2env.env import Sts2Env  # noqa: E402
from sts2env.protocol import TokenKind, Zone, parse  # noqa: E402
from sts2rl.actor import MODE_RUN, load_weights  # noqa: E402
from sts2rl.config import TrainConfig  # noqa: E402
from sts2rl.features import Ragged, collate_ragged  # noqa: E402
from sts2rl.model import PolicyValueNet  # noqa: E402


def pots(o, vocab) -> list[str]:
    ti = o.tok_ints
    m = (ti[:, 0] == TokenKind.POTION) & (ti[:, 2] == Zone.ALLY)
    return [vocab[int(x)] for x in ti[m, 1]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/root/cmbpred/w.pt")
    ap.add_argument("--config", default="/root/silent_train_config.json")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--prefix", default="RSV")
    ap.add_argument("--dhp", type=int, default=3)
    a = ap.parse_args()
    torch.set_num_threads(4)
    cfg = TrainConfig.from_args(["--config", a.config, "--run-dir", "/root/skipdev/rd"])
    net = PolicyValueNet(cfg.model).eval()
    load_weights(net, a.ckpt, -1)
    dll = Path(DEPLOY) / "engine" / "Sts2Headless" / "bin" / "Release" / "net9.0" / "Sts2Headless.dll"
    env = Sts2Env(dll=dll)
    proc = env._proc
    vocab = [l.rstrip("\n") for l in open(os.path.join(DEPLOY, "engine", "catalog", "vocab.txt"), encoding="utf-8")]

    def req(r):
        proc.send(r); raw = proc.recv(); return raw, parse(raw)

    def greedy(o):
        with torch.no_grad():
            lg, _ = net(collate_ragged(Ragged.from_obs([o], np.array([MODE_RUN]))).to("cpu"))
        return int(lg[0, :o.num_actions].argmax())

    stat = collections.Counter(); bad = []; tres = []
    base = {"op": "reset", "mode": "run", "character": cfg.character, "ascension": 10, "snapshots": True,
            "route_summary": cfg.route_summary, "privileged": True, **cfg.cut_flags()}
    for s in range(a.seeds):
        seed = f"{a.prefix}{s:04d}"
        raw, o = req(dict(base, seed=seed))
        steps = 0
        while not o.done and steps < 4000:
            ex = (o.extra or {}).get("combat") if o.extra else None
            if o.in_combat and ex and ex.get("room") in ("Monster", "Elite"):
                hp0, mx0, p0 = int(o.hp), int(o.max_hp), pots(o, vocab)
                want = max(1, hp0 - a.dhp)
                drop = p0[:1]
                t0 = time.time()
                raw, o2 = req({"op": "resolve", "hp": want, "potions": drop})
                tres.append(time.time() - t0)
                stat["resolve"] += 1; stat[f"room:{ex['room']}"] += 1
                p1 = pots(o2, vocab)
                ok_hp = int(o2.hp) == min(want, int(o2.max_hp))
                left = collections.Counter(p0); left.subtract(drop)
                ok_pot = collections.Counter(p1) == +left
                if o2.in_combat or o2.done:
                    stat["after_in_combat_or_done"] += 1; bad.append((seed, int(o.floor), ex, "战斗没结束或整局结束", int(o2.hp), o2.decision.name))
                if not ok_hp:
                    stat["hp_mismatch"] += 1; bad.append((seed, int(o.floor), ex["encounter"], f"血量 想要 {want} 实际 {int(o2.hp)}/{int(o2.max_hp)}（开战 {hp0}/{mx0}）"))
                if not ok_pot:
                    stat["pot_mismatch"] += 1; bad.append((seed, int(o.floor), ex["encounter"], f"药水 开战 {p0} 去掉 {drop} 之后 {p1}"))
                if int(o2.max_hp) != mx0:
                    stat["max_hp_changed"] += 1
                stat[f"next:{o2.decision.name}"] += 1
                o = o2; steps += 1
                continue
            raw, o = req({"op": "step", "action": greedy(o)}); steps += 1
        stat["runs"] += 1; stat["win"] += int(o.victory); stat["stuck"] += int(o.stuck)
        print(f"{seed}：{'通关' if o.victory else '失败'}，第 {int(o.floor)} 层，{steps} 步", flush=True)
    print(json.dumps(stat, ensure_ascii=False))
    if tres:
        print(f"resolve 每次 {np.mean(tres) * 1000:.1f} ms（中位 {np.median(tres) * 1000:.1f}，最大 {np.max(tres) * 1000:.1f}）")
    for b in bad[:20]:
        print("  问题：", b)
    proc.send({"op": "close"})


if __name__ == "__main__":
    main()
