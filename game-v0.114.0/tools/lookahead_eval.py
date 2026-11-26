"""牌组前瞻的评测（技术路线第七节“可选牌组前瞻”，2026-10-27 起在 A 上做，只评测、不训练）。

整局里遇到卡牌奖励时，对每个选项（拿其中一张牌，或跳过），用“当前牌组 + 这张牌”去打本幕 Boss 若干场（满血、各场换种子、贪心），
比较结果；与网络原本的选择在同一批种子上逐个配对取差，平均差超过 z 个标准误才改选。得分 = 赢 1 + 0.1 × 剩余血量比例，
输了按打掉的敌人生命比例给 0.5 × 比例。牌组快照由引擎的 player 操作给出（与战斗课程、求解器用的同一种快照）。

每个种子打两遍整局：纯贪心、贪心 + 选牌前瞻，按种子配对比较平均层数、进第二幕、进第三幕、通关。两遍从同一个种子开局，
第一次选牌不同之后就走上不同的路，所以要很多个种子才看得出差别。结果每个种子一行写到 --out。

    .venv/bin/python -I tools/lookahead_eval.py --ckpt 检查点.pt --out 结果.jsonl --runs 256 --shard 0/16 --envs 8 --device cuda
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from sts2env import Sts2Env, VecEnv  # noqa: E402
from sts2env.env import EngineError  # noqa: E402
from sts2env.protocol import ActionKind, DecisionKind, TokenKind, parse  # noqa: E402
from sts2rl.actor import MODE_COMBAT, MODE_RUN  # noqa: E402
from sts2rl.catalog import Catalog  # noqa: E402
from sts2rl.features import Ragged, collate_ragged  # noqa: E402
from sts2rl.model import ModelConfig, PolicyValueNet, load_state  # noqa: E402

MAX_RUN_STEPS, MAX_FIGHT_STEPS = 6000, 600


def logits_of(net, device, obs: list, mode: int) -> list[np.ndarray]:
    with torch.no_grad():
        lg, _ = net(collate_ragged(Ragged.from_obs(obs, np.array([mode] * len(obs)))).to(device))
    lg = lg.float().cpu().numpy()
    return [lg[k, : o.num_actions] for k, o in enumerate(obs)]


def enemy_hp(o) -> tuple[float, float]:
    k = np.flatnonzero((o.tok_ints[:, 0] == TokenKind.CREATURE) & (o.tok_feats[:, 5] == 0) & (o.tok_feats[:, 3] > 0))
    return float(o.tok_feats[k, 0].sum()), float(o.tok_feats[k, 1].sum())


def fights(env: VecEnv, net, device, specs: list[dict]) -> list[float]:
    """把每个战斗规格贪心打完，返回得分（见文件头）。引擎数少于场数时排队。"""
    out: list[float | None] = [None] * len(specs)
    queue = list(range(len(specs)))
    task: dict[int, list] = {}         # 引擎 → [场次下标, 步数, 开打时敌人最大生命合计, 最近一次敌人剩余生命合计]
    free = list(range(len(env)))
    pend: dict[int, object] = {}
    while queue or task:
        while free and queue:
            i, k = free.pop(), queue.pop()
            task[i] = [k, 0, None, None]
            env.send(i, {"op": "reset", "mode": "combat", "spec": specs[k]})
        for i, o in env.recv_ready(timeout=5.0):
            t = task[i]
            if isinstance(o, EngineError) or o.done or o.stuck or o.num_actions == 0 or t[1] > MAX_FIGHT_STEPS:
                if isinstance(o, EngineError) or t[1] > MAX_FIGHT_STEPS:
                    sc = 0.0
                elif o.victory:
                    sc = 1.0 + 0.1 * o.hp / max(1, o.max_hp)
                else:
                    sc = 0.5 * (1 - t[3] / t[2]) if t[2] else 0.0
                out[t[0]] = sc
                del task[i]; free.append(i)
                continue
            hp_, mx_ = enemy_hp(o)
            if t[2] is None:
                t[2] = mx_
            t[3] = hp_
            pend[i] = o
        if pend:
            ids = list(pend)
            lgs = logits_of(net, device, [pend[i] for i in ids], MODE_COMBAT)
            for i, lg in zip(ids, lgs):
                task[i][1] += 1
                env.send(i, {"op": "step", "action": int(lg.argmax())})
            pend.clear()
    return [x if x is not None else 0.0 for x in out]


def card_options(o, vocab: list[str]) -> list[tuple[int, str | None, int]]:
    """卡牌奖励界面的选项：(候选下标, 牌的词表键或 None 表示跳过, 升级次数)；重抽等其他替代项不考虑。"""
    opts = []
    for c in range(o.num_actions):
        k = int(o.cand_ints[c, 0])
        if k == ActionKind.PICK_CARD_REWARD:
            t = int(o.cand_ints[c, 1])
            opts.append((c, vocab[int(o.tok_ints[t, 1])], int(o.tok_feats[t, 3])))
        elif k == ActionKind.CARD_REWARD_ALT and vocab[int(o.cand_ints[c, 3])] == "ALT:SKIP":
            opts.append((c, None, 0))
    return opts


def lookahead(main: Sts2Env, o, greedy: int, env: VecEnv, net, device, vocab, a, seed_base: str) -> int:
    opts = card_options(o, vocab)
    if len(opts) < 2 or greedy not in {c for c, _, _ in opts}:
        return greedy
    info = main.player()
    snap = json.loads(info["snapshot"])
    snap["current_hp"] = snap.get("max_hp", snap.get("current_hp"))          # 满血打：比的是牌组本身
    boss = vocab[o.bosses[0]].split(".", 1)[-1] if o.bosses[0] else None
    if not boss:
        return greedy
    specs, owner = [], []
    for c, card, up in opts:
        sp = dict(snap); sp["deck"] = list(snap["deck"])
        if card:
            sp["deck"].append({"id": card, "current_upgrade_level": up, "floor_added_to_deck": int(info["floor"])} if up else
                              {"id": card, "floor_added_to_deck": int(info["floor"])})
        js = json.dumps(sp, ensure_ascii=False)
        for j in range(a.n):
            specs.append({"character": "DEFECT", "ascension": 10, "seed": f"{seed_base}_{j}", "act": int(info["act"]), "encounter": boss,
                          "player_snapshot": js, "room": "Boss"})
            owner.append(c)
    sc = fights(env, net, device, specs)
    by = {c: sc[k * a.n:(k + 1) * a.n] for k, (c, _, _) in enumerate(opts)}
    best, best_m = greedy, 0.0
    for c, xs in by.items():
        if c == greedy:
            continue
        d = [x - y for x, y in zip(xs, by[greedy])]
        m = sum(d) / len(d)
        sd = math.sqrt(sum((v - m) ** 2 for v in d) / max(1, len(d) - 1))
        if m > best_m and m > a.z * sd / math.sqrt(len(d)):
            best, best_m = c, m
    return best


def play(main: Sts2Env, env: VecEnv, net, device, vocab, a, seed: str, use_look: bool) -> dict:
    main._proc.send({"op": "reset", "mode": "run", "character": "DEFECT", "ascension": 10, "seed": seed, "route_summary": True})
    o = parse(main._proc.recv())
    n_dec = n_changed = steps = 0
    while not (o.done or o.stuck or o.num_actions == 0) and steps < MAX_RUN_STEPS:
        (lg,) = logits_of(net, device, [o], MODE_RUN)
        act = int(lg.argmax())
        if use_look and o.decision == DecisionKind.CARD_REWARD:
            n_dec += 1
            new = lookahead(main, o, act, env, net, device, vocab, a, f"{seed}_{o.floor}")
            n_changed += new != act
            act = new
        o = main.step(act)
        steps += 1
    return {"floor": int(o.floor), "act": int(o.act), "win": bool(o.victory), "reach2": int(o.act >= 1 or o.victory), "reach3": int(o.act >= 2 or o.victory),
            "decisions": n_dec, "changed": n_changed}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", type=int, default=256)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--envs", type=int, default=8, help="打前瞻战斗用的引擎数")
    ap.add_argument("--n", type=int, default=12, help="每个选项打几场")
    ap.add_argument("--z", type=float, default=2.0)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    torch.set_num_threads(2)
    blob = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    net = PolicyValueNet(ModelConfig(**blob["config"]["model"])).eval().to(a.device)
    load_state(net, blob["state"])
    vocab = Catalog().vocab
    k, n = map(int, a.shard.split("/"))
    seeds = [f"LA{i:04d}" for i in range(a.runs) if i % n == k]
    done = set()
    tag = Path(a.out).stem.rsplit("_s", 1)[0]
    for f in Path(a.out).parent.glob(f"{tag}_s*.jsonl"):
        done |= {json.loads(l)["seed"] for l in open(f, encoding="utf-8") if l.strip()}
    meta = Path(a.out).with_suffix(".meta.json")
    if not meta.exists():
        meta.write_text(json.dumps({"kind": "lookahead", "version": blob.get("version"), "items": len(seeds), "total": a.runs, "shard": a.shard,
                                    "start": round(time.time()), "params": {"n": a.n, "z": a.z, "envs": a.envs}}, ensure_ascii=False), encoding="utf-8")
    print(f"版本 {blob.get('version')}，{len(seeds)} 个种子（已做 {len(done & set(seeds))}）；每个选项 {a.n} 场，z={a.z}", flush=True)
    with Sts2Env() as main_env, VecEnv(a.envs) as env, open(a.out, "a", encoding="utf-8") as out:
        for s in seeds:
            if s in done:
                continue
            t0 = time.time()
            g = play(main_env, env, net, a.device, vocab, a, s, False)
            look = play(main_env, env, net, a.device, vocab, a, s, True)
            out.write(json.dumps({"seed": s, "greedy": g, "look": look, "t": round(time.time()), "secs": round(time.time() - t0)}, ensure_ascii=False) + "\n")
            out.flush()


if __name__ == "__main__":
    main()
