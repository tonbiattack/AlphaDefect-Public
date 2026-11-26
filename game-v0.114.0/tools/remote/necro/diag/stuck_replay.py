"""重放 stuck.jsonl 里的卡住局（开局请求 + 出手序列），逐帧记 token 数与构成（2026-11-17，决策记录第 4 条补记：显存不足的超大帧）。
    REPO=/root/necro-deploy/sls2 $REPO/.venv/bin/python -I stuck_replay.py --dll 引擎.dll --stuck runs/necro/stuck.jsonl [--line -1]
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path

REPO = Path(os.environ.get("REPO") or Path(__file__).resolve().parents[4])
sys.path.insert(0, str(REPO / "python"))

from sts2env.env import Sts2Env  # noqa: E402
from sts2env.protocol import parse  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dll", required=True)
    ap.add_argument("--stuck", required=True)
    ap.add_argument("--line", type=int, default=-1)
    a = ap.parse_args()
    recs = [json.loads(l) for l in open(a.stuck) if l.strip()]
    r = recs[a.line]
    acts = r["actions"]
    print(f"第 {r['version']} 版 {r['req']['spec'].get('encounter')} 出手 {len(acts)} 个；牌组 {len(json.loads(r['req']['spec']['player_snapshot'])['deck']) if isinstance(r['req']['spec']['player_snapshot'], str) else len(r['req']['spec']['player_snapshot']['deck'])} 张", flush=True)
    sizes = []
    with Sts2Env(Path(a.dll)) as e:
        e._proc.send(r["req"]); o = parse(e._proc.recv())
        for i, act in enumerate(acts):
            sizes.append(int(o.tok_ints.shape[0]))
            if i in (0, 100, 500, 1000, 2000, 3000, 4000, len(acts) - 1) or (i % 500 == 0):
                c = collections.Counter(zip(o.tok_ints[:, 0].tolist(), o.tok_ints[:, 2].tolist()))
                print(f"第 {i} 步 token {sizes[-1]} 回合 {int(o.header[12]) if len(o.header) > 12 else '?'} 动作数 {o.num_actions} 构成（类型,区域:数）{[(k, z, v) for (k, z), v in c.most_common(5)]}", flush=True)
            if o.done:
                print("提前结束", i); break
            if act >= o.num_actions:
                print(f"第 {i} 步动作 {act} 越界（{o.num_actions}），停", flush=True); break
            o = e.step(int(act))
    import numpy as np
    s = np.array(sizes)
    print(json.dumps({"帧数": int(s.size), "总 token": int(s.sum()), "平均": round(float(s.mean()), 1), "最大": int(s.max()),
                      "分位50/90/99": [float(np.percentile(s, p)) for p in (50, 90, 99)]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
