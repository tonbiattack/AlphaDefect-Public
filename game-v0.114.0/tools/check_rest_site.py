"""休息处选项数值与满血屏蔽（观测格式版本 8）的核对：本机引擎用模型贪心打若干整局，每次休息处决策核对
休息选项的 2 实际能回的血 / 最大生命 = min(名义回血, 缺的血) / 最大生命、3 名义回血 / 最大生命 ≥ 0.3（取整前）、
4 附带效果条数 = 身上捕梦网、小邮箱、石头加湿器的件数；升级选项的 2 = 可升级的牌数 / 牌组张数（只核对范围）；
以及“满血、没有附带效果、能升级”时不出现休息候选，其余时候休息候选都在。

    .venv/bin/python -I tools/check_rest_site.py --ckpt 检查点.pt --runs 32
"""
from __future__ import annotations

import argparse
import collections
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from sts2env import VecEnv  # noqa: E402
from sts2env.protocol import ActionKind, DecisionKind, TokenKind  # noqa: E402
from sts2rl.actor import MODE_RUN  # noqa: E402
from sts2rl.catalog import Catalog  # noqa: E402
from sts2rl.features import Ragged, collate_ragged  # noqa: E402
from sts2rl.model import ModelConfig, PolicyValueNet, load_state  # noqa: E402

EXTRA = {"RELIC.DREAM_CATCHER", "RELIC.TINY_MAILBOX", "RELIC.STONE_HUMIDIFIER"}


def check(o, vocab, stats, bad):
    ti, tf = o.tok_ints, o.tok_feats
    g = tf[np.flatnonzero(ti[:, 0] == TokenKind.GLOBAL)[0]]
    hp, mx = int(g[4]), int(g[5])
    relics = [vocab[int(ti[k, 1])] for k in np.flatnonzero(ti[:, 0] == TokenKind.RELIC)]
    n_extra = sum(r in EXTRA for r in relics)
    pillow = "RELIC.REGAL_PILLOW" in relics
    opts = {}
    for c in range(o.num_actions):
        ci = o.cand_ints[c]
        if ci[0] == ActionKind.REST_OPTION:
            opts[vocab[int(ti[ci[1], 1])].split(":")[1]] = tf[ci[1]]
    stats["rest"] += 1
    full = hp >= mx
    stats["full"] += full
    if "HEAL" in opts:
        f = opts["HEAL"]
        nominal = f[3] * mx
        exp_actual = min(round(nominal), mx - hp) / mx
        ok = abs(f[2] - exp_actual) < 1e-4 and nominal >= math.floor(0.3 * mx) - 1e-3 and int(f[4]) == n_extra
        stats["heal_feat_ok" if ok else "heal_feat_bad"] += 1
        stats["pillow_nominal"] += pillow and nominal > 0.3 * mx + 1
        stats["extra>0"] += f[4] > 0
        if not ok and len(bad) < 8:
            bad.append(f"血 {hp}/{mx} 特征 {f[2:5].round(3).tolist()} 期望实际 {exp_actual:.3f} 附带 {n_extra} 遗物 {[r for r in relics if r in EXTRA or 'PILLOW' in r]}")
    if "SMITH" in opts:
        stats["smith_ok" if 0 < opts["SMITH"][2] <= 1 else "smith_bad"] += 1
    should_mask = full and n_extra == 0 and "SMITH" in opts
    if should_mask:
        stats["masked_ok" if "HEAL" not in opts else "masked_missing"] += 1
    elif "HEAL" not in opts:
        stats["heal_absent_unexpected"] += 1
        if len(bad) < 8:
            bad.append(f"不该屏蔽却没有休息：血 {hp}/{mx} 选项 {list(opts)} 附带 {n_extra}")
    if full and n_extra > 0:
        stats["full_with_extra_kept"] += "HEAL" in opts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--runs", type=int, default=32)
    ap.add_argument("--envs", type=int, default=8)
    a = ap.parse_args()
    blob = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    net = PolicyValueNet(ModelConfig(**blob["config"]["model"])).eval()
    load_state(net, blob["state"])
    vocab = Catalog().vocab
    stats, bad = collections.Counter(), []
    with VecEnv(a.envs, log=open("/dev/null", "ab")) as env:
        todo = list(range(a.runs))
        def launch(i):
            if todo:
                k = todo.pop()
                env.send(i, {"op": "reset", "mode": "run", "character": "DEFECT", "ascension": 10, "seed": f"REST{k:03d}", "route_summary": True})
                return True
            return False
        live = {i for i in range(a.envs) if launch(i)}
        while live:
            batch = []
            for i, o in env.recv_ready(timeout=60):
                if isinstance(o, Exception) or o.done or o.stuck or o.num_actions == 0:
                    if not launch(i):
                        live.discard(i)
                    continue
                batch.append((i, o))
            if not batch:
                continue
            with torch.no_grad():
                logits, _ = net(collate_ragged(Ragged.from_obs([o for _, o in batch], np.array([MODE_RUN] * len(batch)))).to("cpu"))
            for (i, o), lg in zip(batch, logits):
                if o.decision == DecisionKind.REST_SITE:
                    check(o, vocab, stats, bad)
                env.send(i, {"op": "step", "action": int(lg[: o.num_actions].argmax())})
    print(dict(stats))
    for b in bad:
        print("  ", b)


if __name__ == "__main__":
    main()
