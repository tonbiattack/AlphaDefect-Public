"""“当场帮选”小规模试验（2026-11-03，决策记录 253 后续诊断）：整局贪心，只在选卡奖励界面让构筑价值模型帮着选。

做法：在策略概率最高的前 K 个选项里（拿某张卡，或跳过），各自算出“选完之后”的构筑——拿卡就是牌组加上这张卡，跳过就是牌组不变——
用冻结的构筑价值模型估通关率；最高的那个比策略原本的选择高出 MARGIN 以上才改选，否则照策略。其余决策（含战斗）一律贪心。
每次改选都记一行（层、幕、牌组张数、提供的卡、策略选的与概率、模型选的、两者的估计），对局结果另记，供和同种子的纯贪心配对比较。

    .venv/bin/python -I tools/bv_assist.py --ckpt 检查点.pt --bv 构筑价值模型.pt --seeds-file tools/remote/seed_pool.json --runs 200 \\
        --out 结果目录 [--topk 3] [--margin 0.005] [--envs 40]
输出：结果目录/runs.jsonl（每局：种子、输赢、层数、改选次数）、switches.jsonl（每次改选）、decisions.json（选卡界面总数等计数）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

from sts2env.env import VecEnv  # noqa: E402
from sts2env.protocol import ActionKind, DecisionKind  # noqa: E402
from sts2rl.actor import MODE_RUN, cand_name  # noqa: E402
from sts2rl.buildvalue_model import V, Estimator, obs_row  # noqa: E402
from sts2rl.catalog import Catalog  # noqa: E402
from sts2rl.features import Ragged, collate_ragged  # noqa: E402
from sts2rl.model import ModelConfig, PolicyValueNet, load_state  # noqa: E402

MAX_STEPS = 6000


def offered_token(o, k: int) -> int | None:
    """选卡候选 k 对应的牌组编码（编号 × 2 + 是否升级）；不是拿卡的候选返回 None。"""
    ci = o.cand_ints[k]
    if int(ci[0]) != ActionKind.PICK_CARD_REWARD:
        return None
    vid, up = int(ci[3]), 0
    if 0 <= int(ci[1]) < len(o.tok_ints):
        if vid == 0:
            vid = int(o.tok_ints[int(ci[1]), 1])
        up = int(o.tok_feats[int(ci[1]), 3] > 0)
    return min(vid, V - 1) * 2 + up


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--bv", required=True)
    ap.add_argument("--seeds-file", required=True); ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--out", required=True); ap.add_argument("--topk", type=int, default=3); ap.add_argument("--margin", type=float, default=0.005)
    ap.add_argument("--envs", type=int, default=40); ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    torch.set_num_threads(2)
    blob = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    net = PolicyValueNet(ModelConfig(**blob["config"]["model"])).eval().to(a.device)
    load_state(net, blob["state"])
    est = Estimator(a.bv)
    vocab = Catalog().vocab
    seeds = json.loads(Path(a.seeds_file).read_text(encoding="utf-8"))["seeds"][:a.runs]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    fr, fs = open(out / "runs.jsonl", "w", encoding="utf-8"), open(out / "switches.jsonl", "w", encoding="utf-8")
    cnt = {"card_reward": 0, "switched": 0}
    env = VecEnv(min(a.envs, len(seeds)))
    it = iter(seeds)
    slot: dict[int, dict] = {}

    def start(i):
        s = next(it, None)
        if s is None:
            slot.pop(i, None); return
        slot[i] = {"seed": s, "steps": 0, "sw": 0}
        env.send(i, {"op": "reset", "mode": "run", "character": "DEFECT", "ascension": 10, "seed": s, "snapshots": False, "route_summary": True})

    for i in range(len(env)):
        start(i)
    obs = {}
    while slot:
        for i, o in env.recv_ready(timeout=1.0):
            if i not in slot:
                continue
            st = slot[i]
            if isinstance(o, Exception) or o.done or o.stuck or st["steps"] > MAX_STEPS:
                win = (not isinstance(o, Exception)) and bool(o.done and o.victory)
                fr.write(json.dumps({"seed": st["seed"], "win": int(win), "floor": int(getattr(o, "floor", 0)), "switches": st["sw"]}) + "\n"); fr.flush()
                start(i); continue
            st["steps"] += 1
            obs[i] = o
        if not obs:
            continue
        ids = sorted(obs)
        with torch.no_grad():
            logits = net(collate_ragged(Ragged.from_obs([obs[i] for i in ids], np.full(len(ids), MODE_RUN))).to(a.device))[0]
        for k, i in enumerate(ids):
            o = obs[i]; na = o.num_actions
            pr = torch.softmax(logits[k, :na].float(), -1).cpu().numpy()
            act = int(pr.argmax())
            if o.decision == DecisionKind.CARD_REWARD and (row := obs_row(o)) is not None:
                cnt["card_reward"] += 1
                top = [int(x) for x in np.argsort(-pr)[:a.topk]]
                rows, opts = [], []
                for c in top:
                    tok = offered_token(o, c)
                    deck = row[6] + [tok] if tok is not None else row[6]
                    rows.append((*row[:6], deck, row[7], row[8]))
                    opts.append((c, cand_name(o, c, vocab).removeprefix("CARD.") if tok is not None else "跳过"))
                p = est(rows)
                j = int(np.argmax(p))
                if top[j] != act and p[j] - p[top.index(act)] > a.margin:
                    cnt["switched"] += 1; slot[i]["sw"] += 1
                    fs.write(json.dumps({"seed": slot[i]["seed"], "act": row[0], "floor": row[1], "deck": len(row[6]), "hp": round(row[2] / max(1, row[3]), 2),
                                         "offer": [n for _, n in opts], "policy": opts[top.index(act)][1], "policy_p": round(float(pr[act]), 3),
                                         "bv": opts[j][1], "bv_p": round(float(pr[top[j]]), 3), "phi_policy": round(float(p[top.index(act)]), 4),
                                         "phi_bv": round(float(p[j]), 4)}, ensure_ascii=False) + "\n"); fs.flush()
                    act = top[j]
            env.send_step(i, act)
        obs.clear()
    (out / "decisions.json").write_text(json.dumps(cnt, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(cnt, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
