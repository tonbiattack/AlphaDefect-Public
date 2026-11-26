"""跳过遗物的探针（2026-10-28，决策记录 159）：用某个检查点贪心打若干局整局（同评测：A10 故障机器人、种子 EVAL000 起），
统计奖励界面与宝箱里每件遗物出现、被主动跳过（奖励界面点离开、宝箱选跳过）的次数，并列出每次跳过的现场
（第几幕第几层、血量、手上遗物与药水数、牌组张数、模型给“拿”和“跳过”的概率、奖励界面上同时还剩什么）。
评测进程每次评测也记同样的东西（eval.jsonl 的 relic、run_dir/relic_skip.jsonl），这个工具用于单独抽查、核对。

    .venv/bin/python -I tools/relic_probe.py --ckpt latest.pt [--config train_config.json] --runs 96 --envs 16 --out relic_probe.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

import torch  # noqa: E402

from sts2rl import evaluator  # noqa: E402
from sts2rl.actor import MODE_RUN  # noqa: E402
from sts2rl.model import ModelConfig, PolicyValueNet, load_state  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="检查点或采集用的权重文件")
    ap.add_argument("--config", default="", help="训练配置 train_config.json（权重文件里不带模型结构时用它）")
    ap.add_argument("--runs", type=int, default=96)
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--seed-prefix", default="EVAL")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--manual-gold", action="store_true", help="金币不自动领，由策略自己领（决策记录 160），并统计领了几份、放弃几份")
    ap.add_argument("--out", default="", help="每次跳过的现场写到这里（JSON Lines），不给就只打印")
    a = ap.parse_args()
    blob = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = blob.get("config") or json.load(open(a.config, encoding="utf-8"))
    net = PolicyValueNet(ModelConfig(**cfg["model"])).eval().to(a.device)
    load_state(net, blob["state"])
    version = blob.get("version")
    route = bool(cfg.get("route_summary", True))
    tasks = iter([({"mode": "run", "character": "DEFECT", "ascension": 10, "seed": f"{a.seed_prefix}{k:03d}", "snapshots": True,
                    "route_summary": route, **({"auto_gold": False} if a.manual_gold else {})}, MODE_RUN, "run") for k in range(a.runs)])
    t0 = time.time()
    res = evaluator.play(net, torch.device(a.device), lambda: next(tasks, None), a.envs, threading.Event(), version=version)
    runs = [r for r in res if r["kind"] == "run"]
    rel = evaluator.merge_cards(r.get("relic") for r in runs)
    skips = [dict(d, v=version) for r in runs for d in r.get("relic_skips") or []]
    print(f"第 {version} 版，{len(runs)} 局，用时 {time.time() - t0:.0f} 秒，通关 {sum(r['win'] for r in runs)} 局")
    for scr in ("_REWARDS", "_TREASURE"):
        v = rel.get(scr, [0] * 6)
        print(f"{scr[1:]:9s} 出现 {sum(v[:3]):4d} 次（各幕 {v[:3]}），主动跳过 {sum(v[3:]):3d} 次（各幕 {v[3:]}）")
    ch = sum((collections.Counter(r["choices"]) for r in runs), collections.Counter())
    print(f"领金币 {ch['REWARDS:TAKE_REWARD:REWARD:GoldReward']} 份，放弃金币 {ch['FREE_SKIP:GOLD']} 份，放弃无负面遗物 {ch['FREE_SKIP:RELIC']} 件")
    print("被跳过过的遗物（出现 / 跳过）：")
    for n, v in sorted(((n, v) for n, v in rel.items() if not n.startswith("_") and sum(v[3:])), key=lambda x: -sum(x[1][3:])):
        print(f"  {n:28s} {sum(v[:3]):3d} / {sum(v[3:]):3d}")
    for d in skips:
        print("  " + json.dumps(d, ensure_ascii=False))
    if a.out:
        with open(a.out, "a", encoding="utf-8") as f:
            for d in skips:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
