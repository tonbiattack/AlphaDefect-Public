"""换观测格式 22 → 28 对现有策略的扰动（2026-11-17，亡灵契约师决策记录第 2 条）：同一份检查点，两个引擎同种子、同一串动作同步走，
一个强制格式 22（reset 带 "obs_format": 22），一个用默认格式（28）；网络按新配置建（obs_v26、obs_v27 开，新层清零），
在每一对帧上比首选是否相同、KL(28 ‖ 22)，按战斗 / 局外分开。出手按格式 28 那一边的贪心。

    .venv/bin/python -I tools/remote/necro/diag/f28_pair.py --dll 引擎.dll --ckpt latest.pt --config tools/remote/necro/train_config.json --runs 24
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "python"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from sts2env.env import Sts2Env  # noqa: E402
from sts2env.protocol import parse  # noqa: E402
from sts2rl.actor import MODE_RUN  # noqa: E402
from sts2rl.config import TrainConfig  # noqa: E402
from sts2rl.features import Ragged, collate_ragged  # noqa: E402
from sts2rl.model import PolicyValueNet, load_state  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dll", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--runs", type=int, default=24)
    ap.add_argument("--old-format", type=int, default=22)
    a = ap.parse_args()
    torch.set_num_threads(4)
    cfg = TrainConfig.from_args(["--config", a.config, "--run-dir", "/tmp/f28_pair"])
    blob = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    net = PolicyValueNet(cfg.model).eval()
    load_state(net, blob["state"], blob.get("content_names"))
    print(f"检查点第 {blob['version']} 版（obs_format {blob.get('obs_format')}），网络 obs_v26={cfg.model.obs_v26} obs_v27={cfg.model.obs_v27}", flush=True)
    base = {"op": "reset", "mode": "run", "character": cfg.character, "ascension": 10, "route_summary": cfg.route_summary, **cfg.cut_flags(evaluating=True)}
    stats = {k: [0, 0, 0.0] for k in ("战斗", "局外")}     # 帧数、首选相同、KL 之和
    mism = vers = 0
    floors = []
    with Sts2Env(Path(a.dll)) as e_old, Sts2Env(Path(a.dll)) as e_new:
        for i in range(a.runs):
            seed = f"F28PAIR{i:03d}"
            e_old._proc.send({**base, "seed": seed, "obs_format": a.old_format}); o1 = parse(e_old._proc.recv())
            e_new._proc.send({**base, "seed": seed}); o2 = parse(e_new._proc.recv())
            while True:
                vers += int(int(o2.header[0]) != 28) + int(int(o1.header[0]) != a.old_format)
                if (o1.done, o1.num_actions, int(o1.decision)) != (o2.done, o2.num_actions, int(o2.decision)):
                    mism += 1
                    break
                if o2.done:
                    floors.append(int(o2.floor))
                    break
                with torch.no_grad():
                    lg, _ = net(collate_ragged(Ragged.from_obs([o1, o2], np.full(2, MODE_RUN))).to("cpu"))
                lp = torch.log_softmax(lg.float(), -1)
                n = o2.num_actions
                if n >= 2:
                    p1, p2 = lp[0, :n], lp[1, :n]
                    kl = float((p2.exp() * (p2 - p1)).sum())
                    s = stats["战斗" if o2.in_combat else "局外"]
                    s[0] += 1; s[1] += int(int(p1.argmax()) == int(p2.argmax())); s[2] += kl
                act = int(lp[1, :n].argmax())
                o1, o2 = e_old.step(act), e_new.step(act)
    out = {k: {"帧": v[0], "首选相同": v[1], "首选相同比例": round(v[1] / max(1, v[0]), 4), "平均KL": round(v[2] / max(1, v[0]), 5)} for k, v in stats.items()}
    print(json.dumps({"局数": a.runs, "两边走岔的局": mism, "格式号不符的帧": vers, "格式28贪心平均层数": round(float(np.mean(floors)), 2) if floors else None,
                      **out}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
