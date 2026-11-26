"""核对评测在 CPU 与显卡上推理的结果是否一致：用同一个检查点把固定考卷各打一遍（贪心、同种子），按题比对胜负，并记两边的耗时。
在租用机上以空闲优先级运行，只用空出来的 CPU：

    chrt -i 0 nice -n 19 .venv/bin/python -I tools/eval_device_check.py --run-dir ~/.local/share/sls2/runs/defect [--envs 12] [--limit 0]
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

import torch  # noqa: E402

from sts2rl import evaluator  # noqa: E402
from sts2rl.actor import MODE_COMBAT  # noqa: E402
from sts2rl.config import TrainConfig  # noqa: E402
from sts2rl.model import PolicyValueNet, load_state  # noqa: E402


def run(net, device, items, envs: int) -> tuple[str, float]:
    tasks = [({"mode": "combat", "spec": it["spec"], "tag": k}, MODE_COMBAT, "exam") for k, it in enumerate(items)]
    it_ = iter(tasks)
    t0 = time.time()
    res = evaluator.play(net, device, lambda: next(it_, None), envs, threading.Event())
    out = ["-"] * len(items)
    for r in res:
        out[r["tag"]] = "1" if r["win"] else ("x" if r["stuck"] else "0")
    return "".join(out), time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--envs", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    rd = Path(a.run_dir).expanduser()
    items = json.loads((rd / "exam.json").read_text(encoding="utf-8"))["items"]
    if a.limit:
        items = items[:a.limit]
    blob = torch.load(rd / "latest.pt", map_location="cpu", weights_only=False)
    cfg = TrainConfig()
    for k, v in (blob.get("config") or {}).get("model", {}).items():
        setattr(cfg.model, k, v)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = {}
    for dev in ("cuda", "cpu"):
        net = PolicyValueNet(cfg.model).eval()
        load_state(net, blob["state"])
        net.to(dev)
        out[dev] = run(net, torch.device(dev), items, a.envs)
        print(f"{dev}: 胜 {out[dev][0].count('1')} / {len(items)}，卡住 {out[dev][0].count('x')}，用时 {out[dev][1]:.0f} 秒", flush=True)
    g, c = out["cuda"][0], out["cpu"][0]
    diff = [i for i in range(len(items)) if g[i] != c[i]]
    print(f"第 {blob.get('version')} 版，{len(items)} 题：两边胜负不同 {len(diff)} 题（显卡赢 CPU 输 {sum(g[i] == '1' for i in diff)}，"
          f"反之 {sum(c[i] == '1' for i in diff)}）", flush=True)


if __name__ == "__main__":
    main()
