"""服务端口径的编码 / step 计时（引擎提速核对用，2026-11-16）：经管道协议照录下的动作序列走 boss_pimc 的题，读引擎退出时打印的 [timing]。

和 bench-encode 不同，这里走的是 Server 的真实路径（reset、step 与 decision、Observation.Build + Serialize 一起计在 encode 里），
可选 --snapshots N：每题开打时往 N 个槽位各存一份快照（推演进程常驻快照、堆更大，GC 更贵），看编码、step 在大堆下的开销。

  STS2_TIMING=1 python -I tools/perf/serve_timing.py --dll 新.dll --items items_t1.json --traj traj.json --n 16 [--snapshots 8]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from boss_equal import EXTRA, Raw  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dll", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--traj", required=True)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--snapshots", type=int, default=0)
    ap.add_argument("--log", default=str(REPO / ".cache" / "serve_timing.log"))
    a = ap.parse_args()
    os.environ["STS2_TIMING"] = "1"
    items = {it["id"]: it for it in json.loads(Path(a.items).read_text(encoding="utf-8"))["items"]}
    trajs = json.loads(Path(a.traj).read_text(encoding="utf-8"))["trajs"][:a.n]
    log = open(a.log, "w")
    e = Raw(Path(a.dll), log)
    for _ in range(a.passes):
        for t in trajs:
            e.req({"op": "reset", "mode": "combat", "spec": items[t["id"]]["spec"], **EXTRA})
            for s in range(a.snapshots):
                e.req({"op": "snapshot", "slot": s})
            for k in t["actions"]:
                e.req({"op": "step", "action": k})
    e.env._proc.send({"op": "close"})
    e.env._proc.p.wait(timeout=60)
    log.close()
    for line in open(a.log, encoding="utf-8"):
        if line.startswith("[timing]"):
            print(line.rstrip())


if __name__ == "__main__":
    main()
