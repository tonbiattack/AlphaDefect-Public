"""内容先验追加卡牌升级变化（决策记录 264，ModelConfig.content_upgrade）的检验，本机 CPU。

  1. 内容表：打开后原有各列（名字与数值）逐位不变，新列只出现在能升级的卡牌行上；抽查几张牌的升级变化。
  2. 模型：同一检查点载入打开与不打开两种配置，在真实帧上（检查点贪心打几局整局）策略与价值输出逐位相同（新列的映射清零）；
     把新列的映射换成随机权重后输出才变，且只在帧里有能升级的卡牌时变。

    .venv/bin/python -I tools/content_upgrade_check.py --dll 引擎/Sts2Headless.dll --ckpt 检查点.pt [--games 2]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

from sts2env.env import _Process  # noqa: E402
from sts2env.protocol import parse  # noqa: E402

fails: list[str] = []


def check(ok: bool, msg: str) -> bool:
    if not ok:
        fails.append(msg)
        print("  不通过：", msg, flush=True)
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dll", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--games", type=int, default=2)
    a = ap.parse_args()
    import torch
    from sts2rl.actor import MODE_RUN
    from sts2rl.catalog import Catalog
    from sts2rl.content import content_table
    from sts2rl.features import Ragged, collate_ragged
    from sts2rl.model import ModelConfig, PolicyValueNet, load_state
    torch.set_num_threads(4)
    blob = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    mc = dict(blob["config"]["model"]); mc.pop("content_upgrade", None)

    # 一、内容表
    n0, t0 = content_table(mc.get("vocab_size", 4096), enchant=bool(mc.get("content_enchant")))
    n1, t1 = content_table(mc.get("vocab_size", 4096), enchant=bool(mc.get("content_enchant")), upgrade=True)
    check(n1[:len(n0)] == n0 and np.array_equal(t1[:, :len(n0)], t0), "原有各列变了")
    new = t1[:, len(n0):]
    cat = Catalog()
    rows = np.flatnonzero(np.abs(new).sum(1) > 0)
    keys = [cat.key(int(i)) if hasattr(cat, "key") else str(i) for i in rows[:3]]
    print(f"内容表：原有 {len(n0)} 列不变，新增 {len(n1) - len(n0)} 列，{len(rows)} 行（能升级的卡牌）有值，例如 {keys}", flush=True)

    # 二、模型
    base = PolicyValueNet(ModelConfig(**mc)).eval(); load_state(base, blob["state"])
    new_net = PolicyValueNet(ModelConfig(**mc, content_upgrade=True)).eval(); load_state(new_net, blob["state"])

    def pol(o) -> int:
        with torch.no_grad():
            lg, _ = base(collate_ragged(Ragged.from_obs([o], np.full(1, MODE_RUN))).to("cpu"))
        return int(lg[0, :o.num_actions].argmax())
    proc = _Process(Path(a.dll), open("/dev/null", "wb"))
    obs = []
    for gi in range(a.games):
        proc.send({"op": "reset", "mode": "run", "character": "DEFECT", "ascension": 10, "seed": f"UPCHK{gi:04d}", "privileged": True, "route_summary": True})
        o = parse(proc.recv())
        for _ in range(3000):
            if o.done or o.stuck or o.num_actions == 0:
                break
            obs.append(o); proc.send({"op": "step", "action": pol(o)}); o = parse(proc.recv())
        print(f"  第 {gi} 局：第 {o.floor} 层，累计 {len(obs)} 帧", flush=True)
    b = collate_ragged(Ragged.from_obs(obs, np.full(len(obs), MODE_RUN))).to("cpu")
    with torch.no_grad():
        l0, v0 = base(b); l1, v1 = new_net(b)
        same = lambda x, y: bool(torch.equal(torch.nan_to_num(x, neginf=-1e30), torch.nan_to_num(y, neginf=-1e30)))
        check(same(l0, l1) and torch.equal(v0, v1), f"打开后输出变了：策略最大差 {float((torch.nan_to_num(l0 - l1)).abs().max()):.2e}，价值 {float((v0 - v1).abs().max()):.2e}")
        w = new_net.content_proj.weight
        w[:, len(n0):] = torch.randn_like(w[:, len(n0):]) * 0.05
        new_net._content_key = None
        l2, v2 = new_net(b)
        changed = (torch.nan_to_num(l2 - l0).abs().amax(-1) > 1e-6) | ((v2 - v0).abs() > 1e-6)
        print(f"新列换随机权重后：{int(changed.sum())}/{len(obs)} 帧的输出变了", flush=True)
        check(bool(changed.any()), "新列换随机权重后输出没变（新列没接上）")
    print("全部通过" if not fails else f"共 {len(fails)} 项不通过", flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
