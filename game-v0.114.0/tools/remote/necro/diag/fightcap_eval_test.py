"""评测的单场战斗步数上限（caps["fight"]，亡灵契约师决策记录第 10 条）实测：用当前检查点贪心打几局整局，
一次把上限设得很低（默认 25 步，正常战斗也会碰到），看是否按卡住结束、原因与 stuck.jsonl 记录对；一次不设上限，看与原来一样正常结束。

    REPO=部署目录 .venv/bin/python -I fightcap_eval_test.py --ckpt latest.pt --config train_config.json --out /tmp/fc --runs 6
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from pathlib import Path

REPO = Path(os.environ.get("REPO") or Path(__file__).resolve().parents[4])
sys.path.insert(0, str(REPO / "python"))

import torch  # noqa: E402

from sts2rl.actor import MODE_RUN  # noqa: E402
from sts2rl.config import TrainConfig  # noqa: E402
from sts2rl.evaluator import play  # noqa: E402
from sts2rl.model import PolicyValueNet, load_state  # noqa: E402


def run(net, cfg, seeds, caps, out_dir: Path) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    lc = cfg.cut_flags(evaluating=True)
    tasks = [({"mode": "run", "character": cfg.character, "ascension": 10, "seed": s, "snapshots": True, "route_summary": cfg.route_summary, **lc},
              MODE_RUN, "run") for s in seeds]
    it = iter(tasks)
    return play(net, torch.device("cpu"), lambda: next(it, None), min(4, len(seeds)), threading.Event(), caps=caps,
                loop_log=str(out_dir / "eval_loops.txt"), version=0, time_cap=600.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", type=int, default=6)
    ap.add_argument("--low", type=int, default=25)
    a = ap.parse_args()
    torch.set_num_threads(4)
    cfg = TrainConfig.from_args(["--config", a.config, "--run-dir", a.out])
    blob = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    net = PolicyValueNet(cfg.model).eval()
    load_state(net, blob["state"], blob.get("content_names"))
    seeds = [f"FIGHTCAP{i:03d}" for i in range(a.runs)]
    for name, fight in (("低上限", a.low), ("不设上限", None)):
        caps = {"run": cfg.max_steps_run, "combat": cfg.max_steps_combat, "exam": cfg.max_steps_combat, "fight": fight}
        d = Path(a.out) / ("low" if fight else "none")
        res = run(net, cfg, seeds, caps, d)
        st = [json.loads(l) for l in open(d / "stuck.jsonl")] if (d / "stuck.jsonl").exists() else []
        print(json.dumps({"组": name, "上限": fight, "局数": len(res), "卡住": sum(r["stuck"] for r in res), "通关": sum(r["win"] for r in res),
                          "层数": [r["floor"] for r in res], "stuck.jsonl 原因": sorted({s["reason"] for s in st})}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
