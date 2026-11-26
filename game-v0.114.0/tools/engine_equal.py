"""引擎改动前后的逐字节对拍（五个角色通用；引擎重构、拆文件、加新格式时用）。

同一组请求、同样的动作同时发给两个引擎（改动前的基线、改动后的新引擎），逐帧比较返回的字节；一处不同就记下并结束这一局
（分歧之后两边的局面不同，不再往下比）。动作只由帧内容决定（偏向多打牌的随机出牌、按课程抽题的单场战斗、检查点贪心、
故障机器人的评测回放），两边帧相同就一定发同样的动作。reset 带训练采集的全部开关（特权、路线摘要、整局预生成内容、快照）。

  python -I tools/engine_equal.py --base .cache/eng-base/Sts2Headless.dll --new .cache/eng-new/Sts2Headless.dll
  python -I tools/engine_equal.py --base ... --new ... --chars DEFECT,REGENT --runs 6 --combats 40 --defect-replays 20 \\
      --greedy REGENT=~/.local/share/sls2/live/ckpt-regent/v45900-20261108-储君.pt
  python -I tools/engine_equal.py --base ... --new ... --chars SILENT --obs-format 19,20   # 强制格式（门控核对）

退出码：有不同为 1。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "tools"))

from sts2env.env import Sts2Env  # noqa: E402
from sts2env.protocol import ActionKind, DecisionKind, parse  # noqa: E402

RUN_FLAGS = {"snapshots": True, "route_summary": True, "privileged": True, "future_content": True}
CHARS = ["DEFECT", "REGENT", "SILENT", "IRONCLAD", "NECROBINDER"]
# 课程的初始牌组只登记了在训练的角色；这里补上其余角色（游戏 CharacterModel.StartingDeck，v0.111.0），只在本工具里用
EXTRA_DECKS = {
    "SILENT": ["STRIKE_SILENT"] * 5 + ["DEFEND_SILENT"] * 5 + ["NEUTRALIZE", "SURVIVOR"],
    "IRONCLAD": ["STRIKE_IRONCLAD"] * 5 + ["DEFEND_IRONCLAD"] * 4 + ["BASH"],
    "NECROBINDER": ["STRIKE_NECROBINDER"] * 4 + ["DEFEND_NECROBINDER"] * 4 + ["BODYGUARD", "UNLEASH"],
}


class Raw:
    """引擎进程，按原始字节收发（不解析），便于逐字节比较。"""

    def __init__(self, dll: Path, log):
        self.env = Sts2Env(dll=dll, log=log)

    def req(self, r: dict) -> bytes:
        self.env._proc.send(r)
        return self.env._proc.recv()

    def close(self):
        self.env.close()


def play_bias(rng: random.Random, p_play: float = 0.9):
    """随机出牌，偏向多打牌、少提前结束回合（多碰到出牌候选的预览与各种附带效果）。"""
    def act(step: int, o) -> int:
        n = o.num_actions
        if o.decision == DecisionKind.COMBAT:
            kinds = o.cand_ints[:n, 0]
            play = [k for k in range(n) if kinds[k] in (ActionKind.PLAY_CARD, ActionKind.USE_POTION)]
            if play and rng.random() < p_play:
                return rng.choice(play)
        return rng.randrange(n)
    return act


def compare(base: Raw, new: Raw, reset: dict, actions, max_steps: int, st: dict, tag: str) -> None:
    a, b = base.req(reset), new.req(reset)
    step = 0
    while True:
        st["frames"] += 1
        if a != b:
            st["diff"] += 1
            if len(st["examples"]) < 12:
                try:
                    oa, ob = parse(a), parse(b)
                    st["examples"].append({"tag": tag, "step": step, "len": [len(a), len(b)], "ver": [int(oa.header[0]), int(ob.header[0])],
                                           "tok": [int(oa.tok_ints.shape[0]), int(ob.tok_ints.shape[0])], "decision": int(oa.decision)})
                except Exception as e:  # noqa: BLE001
                    st["examples"].append({"tag": tag, "step": step, "len": [len(a), len(b)], "parse": str(e)[:120]})
            return
        o = parse(a)
        st["ver"][int(o.header[0])] = st["ver"].get(int(o.header[0]), 0) + 1
        if o.done or o.num_actions == 0 or step >= max_steps:
            return
        k = actions(step, o)
        if k >= o.num_actions:
            st["desync"] += 1
            k %= o.num_actions
        a, b = base.req({"op": "step", "action": k}), new.req({"op": "step", "action": k})
        step += 1


def greedy_policy(ckpt: Path):
    from obs_v21_check import greedy_policy as g  # 检查点贪心（同 sim_corpus.py）
    return g(ckpt)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--chars", default=",".join(CHARS))
    ap.add_argument("--runs", type=int, default=4, help="每个角色随机出牌的整局数")
    ap.add_argument("--max-steps", type=int, default=2500)
    ap.add_argument("--combats", type=int, default=30, help="每个角色按课程抽题的单场战斗数")
    ap.add_argument("--defect-replays", type=int, default=0, help="故障机器人评测回放的整局数（obs_v20_check.py 的回放）")
    ap.add_argument("--greedy", action="append", default=[], help="角色=检查点，检查点贪心打整局（可多次）")
    ap.add_argument("--greedy-runs", type=int, default=6)
    ap.add_argument("--obs-format", default="", help="逗号分隔的格式版本：reset 带 obs_format 强制成这些格式各跑一遍")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-dir", type=Path, default=REPO / ".cache" / "engine_equal")
    args = ap.parse_args()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    base = Raw(Path(args.base), open(args.log_dir / "base.log", "w"))
    new = Raw(Path(args.new), open(args.log_dir / "new.log", "w"))
    from sts2rl import curriculum as C
    from sts2rl.catalog import Catalog
    for k, v in EXTRA_DECKS.items():
        C.STARTING_DECK.setdefault(k, v)
    cat = Catalog()
    fmts = [int(x) for x in args.obs_format.split(",") if x] or [None]
    out = {}
    t0 = time.time()
    for ch in [c for c in args.chars.split(",") if c]:
        for fm in fmts:
            key = ch if fm is None else f"{ch}@{fm}"
            st = {"frames": 0, "diff": 0, "desync": 0, "games": 0, "ver": {}, "examples": []}
            extra = {} if fm is None else {"obs_format": fm}
            for i in range(args.runs):
                seed = f"EQ{args.seed}{ch[:2]}{i:03d}"
                compare(base, new, {"op": "reset", "mode": "run", "character": ch, "ascension": 10, "seed": seed, **RUN_FLAGS, **extra},
                        play_bias(random.Random(f"{args.seed}-{ch}-{i}")), args.max_steps, st, f"{key} 整局 {seed}")
                st["games"] += 1
            cur = C.CombatCurriculum(cat, C.CurriculumConfig.for_character(ch), seed=args.seed)
            rng = random.Random(args.seed)
            for i in range(args.combats):
                spec = cur.sample(rng.random(), boss_fraction=0.3, mix_fraction=0.3)
                compare(base, new, {"op": "reset", "mode": "combat", "spec": spec, "privileged": True, **extra},
                        play_bias(random.Random(5000 + i)), 600, st, f"{key} 战斗 {i}")
                st["games"] += 1
            for g in args.greedy:
                gc, path = g.split("=", 1)
                if gc != ch:
                    continue
                pol = greedy_policy(Path(path).expanduser())
                for i in range(args.greedy_runs):
                    seed = f"EQG{args.seed}{ch[:2]}{i:03d}"
                    compare(base, new, {"op": "reset", "mode": "run", "character": ch, "ascension": 10, "seed": seed, **RUN_FLAGS, **extra},
                            pol, 6000, st, f"{key} 贪心 {seed}")
                    st["games"] += 1
            if ch == "DEFECT" and args.defect_replays and fm is None:
                from obs_v20_check import RUN_FLAGS as RF, pick_replays, replay_actions
                for p in pick_replays(args.defect_replays, 0, args.seed):
                    seed, acts = replay_actions(p)
                    compare(base, new, {"op": "reset", "mode": "run", "character": "DEFECT", "ascension": 10, "seed": seed, **RF},
                            lambda i, o, acts=acts: acts[i] if i < len(acts) else 0, 6000, st, f"DEFECT 回放 {p.name}")
                    st["games"] += 1
            out[key] = st
            print(f"{key}：{st['games']} 局 {st['frames']} 帧，不同 {st['diff']}，动作越界 {st['desync']}，版本 {st['ver']}（累计 {time.time() - t0:.0f} 秒）", flush=True)
    base.close(); new.close()
    total = {"frames": sum(v["frames"] for v in out.values()), "diff": sum(v["diff"] for v in out.values())}
    print(json.dumps({"total": total, **out, "seconds": round(time.time() - t0, 1)}, ensure_ascii=False, indent=1))
    sys.exit(1 if total["diff"] else 0)


if __name__ == "__main__":
    main()
