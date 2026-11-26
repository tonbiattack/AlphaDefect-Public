"""第三幕双 Boss 题（boss_pimc 的题目）上的逐字节对拍，顺带录下动作序列给 bench-encode 用（引擎提速的验证，2026-11-16）。

两个引擎用同一题的 CombatSpec 开战（战斗模式，带 boss_pimc 同样的 loop_cut、stall_turns），同样的动作逐帧比较返回的字节；
一处不同就记下并结束这一题。动作由帧决定（检查点贪心或按种子的偏向出牌随机），两边帧相同就一定发同样的动作。

  python -I tools/perf/boss_equal.py --base 参照.dll --new 新.dll --items items_t1.json --n 24 --ckpt L1000.pt
  python -I tools/perf/boss_equal.py --base 参照.dll --new 新.dll --items items_t1.json --n 24 --policy random --obs-format 19,26,27
  python -I tools/perf/boss_equal.py --base 参照.dll --items items_t1.json --n 16 --ckpt L1000.pt --save-traj traj.json   # 只录动作

退出码：有不同为 1。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "tools"))

from sts2env.env import Sts2Env  # noqa: E402
from sts2env.protocol import ActionKind, DecisionKind, parse  # noqa: E402

EXTRA = {"loop_cut": True, "stall_turns": 20}       # 同 tools/future/boss_pimc.py
MODE_COMBAT = 0


class Raw:
    def __init__(self, dll: Path, log):
        self.env = Sts2Env(dll=dll, log=log)

    def req(self, r: dict) -> bytes:
        self.env._proc.send(r)
        return self.env._proc.recv()

    def close(self):
        self.env.close()


def greedy(ckpt: Path):
    import torch
    from sts2rl.features import Ragged, collate_ragged
    from sts2rl.model import ModelConfig, PolicyValueNet, load_state
    torch.set_num_threads(2)
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    net = PolicyValueNet(ModelConfig(**blob["config"]["model"])).eval()
    load_state(net, blob["state"])

    def act(step: int, o) -> int:
        with torch.no_grad():
            logits, _ = net(collate_ragged(Ragged.from_obs([o], np.array([MODE_COMBAT]))).to("cpu"))
        return int(logits[0, :o.num_actions].argmax())
    return act


def play_bias(rng: random.Random, p_play: float = 0.85):
    def act(step: int, o) -> int:
        n = o.num_actions
        if o.decision == DecisionKind.COMBAT:
            kinds = o.cand_ints[:n, 0]
            play = [k for k in range(n) if kinds[k] in (ActionKind.PLAY_CARD, ActionKind.USE_POTION)]
            if play and rng.random() < p_play:
                return rng.choice(play)
        return rng.randrange(n)
    return act


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--new", default="")
    ap.add_argument("--items", required=True)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--policy", choices=["greedy", "random"], default="greedy")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--privileged", action="store_true", help="reset 带 privileged（搜索不带；对拍时可多覆盖特权特征）")
    ap.add_argument("--obs-format", default="")
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--save-traj", default="", help="把每题的动作序列写成 JSON（bench-encode 用）")
    ap.add_argument("--log-dir", type=Path, default=REPO / ".cache" / "boss_equal")
    a = ap.parse_args()
    a.log_dir.mkdir(parents=True, exist_ok=True)
    items = json.loads(Path(a.items).read_text(encoding="utf-8"))["items"][a.start:a.start + a.n]
    base = Raw(Path(a.base), open(a.log_dir / "base.log", "w"))
    new = Raw(Path(a.new), open(a.log_dir / "new.log", "w")) if a.new else None
    pol = greedy(Path(a.ckpt).expanduser()) if a.policy == "greedy" else None
    fmts = [int(x) for x in a.obs_format.split(",") if x] or [None]
    t0 = time.time()
    total = {"frames": 0, "diff": 0}
    trajs = []
    for fm in fmts:
        st = {"frames": 0, "diff": 0, "items": 0, "examples": [], "ver": {}}
        for it in items:
            extra = dict(EXTRA)
            if fm is not None:
                extra["obs_format"] = fm
            if a.privileged:
                extra["privileged"] = True
            reset = {"op": "reset", "mode": "combat", "spec": it["spec"], **extra}
            act = pol if pol is not None else play_bias(random.Random(f"boss-{it['id']}"))
            x = base.req(reset)
            y = new.req(reset) if new else x
            acts, step = [], 0
            while True:
                st["frames"] += 1
                if x != y:
                    st["diff"] += 1
                    if len(st["examples"]) < 8:
                        st["examples"].append({"id": it["id"], "step": step, "len": [len(x), len(y)]})
                    break
                o = parse(x)
                st["ver"][int(o.header[0])] = st["ver"].get(int(o.header[0]), 0) + 1
                if o.done or o.num_actions == 0 or step >= a.max_steps:
                    break
                k = act(step, o)
                acts.append(int(k))
                x = base.req({"op": "step", "action": k})
                y = new.req({"op": "step", "action": k}) if new else x
                step += 1
            st["items"] += 1
            if fm is None:
                trajs.append({"id": it["id"], "actions": acts})
        tag = "按角色" if fm is None else f"格式 {fm}"
        print(f"{tag}：{st['items']} 题 {st['frames']} 帧，不同 {st['diff']}，版本 {st['ver']}，例 {st['examples']}（累计 {time.time() - t0:.0f} 秒）", flush=True)
        total["frames"] += st["frames"]; total["diff"] += st["diff"]
    base.close()
    if new:
        new.close()
    if a.save_traj:
        Path(a.save_traj).write_text(json.dumps({"items": a.items, "policy": a.policy, "ckpt": a.ckpt, "trajs": trajs}), encoding="utf-8")
        print(f"动作序列 {len(trajs)} 题、{sum(len(t['actions']) for t in trajs)} 步 → {a.save_traj}")
    print(json.dumps({"total": total, "seconds": round(time.time() - t0, 1)}, ensure_ascii=False))
    sys.exit(1 if total["diff"] else 0)


if __name__ == "__main__":
    main()
