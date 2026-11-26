"""局外决策的分枝标注（2026-11-03，决策记录 258）：同一局面下，策略前几名的选项各接着打几局，比较结果。

一、底局：用检查点贪心打一批整局（种子池给定），记下每一步的动作；在要标注的局外决策（默认只有选卡奖励）里每局抽至多 --per-game 个点。
二、分枝：每个点取策略概率前 --topk 的选项；每个选项配同一组 --salts 个 salt（共同随机数：同一 salt 下各选项面对同样的“新未来”，
    差别只来自选项本身）。每一枝 = 重放到这一步 → 公平重设未来（引擎 Branch.Apply，换掉玩家看不到的卡牌奖励、商店、遭遇、事件、后几幕）
    → 走这个选项 → 贪心打完。一局打完记输赢、层数。
三、输出：每个点一行（种子、动作带长度、幕、层、决策类型、各选项的动作号、牌名、策略概率、各 salt 的输赢与层数）。
    分析见 --report：各选项通关率、与策略首选按 salt 配对的胜负差、策略首选是否最好。

    .venv/bin/python -I tools/branch_label.py --ckpt 检查点.pt --seeds-file 种子池.json --games 50 --out 结果目录 [--topk 3] [--salts 8] [--envs 40]
    python3 tools/branch_label.py --report 结果目录/labels.jsonl
需要带 Branch 的引擎（engine/Sts2Headless/Engine/Branch.cs）。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import deque
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

KIND_NAMES = {2: "选牌", 3: "选卡奖励", 5: "奖励领取", 6: "事件", 7: "休息处", 8: "商店", 11: "选路线"}
RESET = {"op": "reset", "mode": "run", "character": "DEFECT", "ascension": 10, "route_summary": True}
MAX_STEPS = 6000


def run_batch(net, device, env, requests: list[dict], record: bool) -> list[dict]:
    """在 VecEnv 上并行打一批整局：每个请求是一个 reset（可带 replay、branch、first），之后贪心打完。
    record 为真时记下每一步的动作与决策信息（底局用）。返回与请求同序的结果。"""
    import numpy as np
    import torch
    from sts2rl.actor import MODE_RUN
    from sts2rl.features import Ragged, collate_ragged
    q = deque(enumerate(requests)); out: list = [None] * len(requests); slot: dict = {}; obs: dict = {}

    def start(i):
        if not q:
            slot.pop(i, None); return
        k, req = q.popleft()
        slot[i] = {"k": k, "steps": 0, "tape": [], "points": []}
        env.send(i, req)

    for i in range(len(env)):
        start(i)
    while slot:
        for i, o in env.recv_ready(timeout=1.0):
            if i not in slot:
                continue
            st = slot[i]
            if isinstance(o, Exception) or o.done or o.stuck or st["steps"] > MAX_STEPS:
                win = (not isinstance(o, Exception)) and bool(o.done and o.victory)
                out[st["k"]] = {"win": int(win), "floor": int(getattr(o, "floor", 0)),
                                "stuck": int(isinstance(o, Exception) or bool(getattr(o, "stuck", False)) or st["steps"] > MAX_STEPS),
                                **({"tape": st["tape"], "points": st["points"]} if record else {})}
                start(i); continue
            st["steps"] += 1; obs[i] = o
        if not obs:
            continue
        ids = sorted(obs)
        with torch.no_grad():
            lg = net(collate_ragged(Ragged.from_obs([obs[i] for i in ids], np.full(len(ids), MODE_RUN))).to(device))[0]
        for k, i in enumerate(ids):
            o = obs[i]; na = o.num_actions
            p = torch.softmax(lg[k, :na].float(), -1).cpu().numpy()
            act = int(p.argmax())
            if record:
                st = slot[i]
                if na > 1 and not o.in_combat:
                    st["points"].append({"t": len(st["tape"]), "d": int(o.decision), "act": int(o.act), "floor": int(o.floor), "deck": _deck_n(o),
                                         "p": [round(float(x), 4) for x in p], "names": [_name(o, j) for j in range(na)]})
                st["tape"].append(act)
            env.send_step(i, act)
        obs.clear()
    return out


_VOCAB = None


def _deck_n(o) -> int:
    from sts2env.protocol import TokenKind, Zone
    ti = o.tok_ints
    return int(((ti[:, 0] == TokenKind.CARD) & (ti[:, 2] == Zone.DECK)).sum())


def _name(o, j: int) -> str:
    global _VOCAB
    from sts2rl.actor import cand_name
    if _VOCAB is None:
        from sts2rl.catalog import Catalog
        _VOCAB = Catalog().vocab
    return cand_name(o, j, _VOCAB)


def label(a) -> None:
    import numpy as np
    import torch
    from sts2env.env import VecEnv
    from sts2rl.model import ModelConfig, PolicyValueNet, load_state
    torch.set_num_threads(2)
    blob = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    net = PolicyValueNet(ModelConfig(**blob["config"]["model"])).eval().to(a.device)
    load_state(net, blob["state"])
    seeds = json.loads(Path(a.seeds_file).read_text(encoding="utf-8"))["seeds"][a.seed_from:a.seed_from + a.games]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    env = VecEnv(a.envs, **({"dll": Path(a.dll)} if a.dll else {}))
    kinds = {int(x) for x in a.kinds.split(",")}
    rng = np.random.default_rng(a.rng)
    t0 = time.time()
    base = run_batch(net, a.device, env, [{**RESET, "seed": s} for s in seeds], record=True)
    print(f"底局 {len(base)} 局，通关 {np.mean([b['win'] for b in base]):.1%}，用时 {time.time() - t0:.0f} 秒", flush=True)
    jobs = []
    for s, b in zip(seeds, base):
        pts = [p for p in b["points"] if p["d"] in kinds and sorted(p["p"])[-1] < a.max_top]
        for p in (rng.choice(len(pts), size=min(a.per_game, len(pts)), replace=False) if pts else []):
            jobs.append((s, b, pts[int(p)]))
    salts = [int(x) for x in rng.integers(1, 2 ** 31, size=a.salts)]
    print(f"标注点 {len(jobs)} 个，每点 {a.topk} 个选项 × {a.salts} 个 salt", flush=True)
    reqs, index = [], []
    for j, (s, b, p) in enumerate(jobs):
        top = [int(x) for x in np.argsort(-np.asarray(p["p"]))[:a.topk]]
        if a.pairs:
            # 汇总设计：只比策略首选与一个备选（从其余前 topk 里随机挑一个），每点用各自的 salt（不同点之间不共用，配对只在点内）
            if len(top) < 2:
                continue
            top = [top[0], int(rng.choice(top[1:]))]
            pt_salts = [int(x) for x in rng.integers(1, 2 ** 31, size=a.salts)]
        else:
            pt_salts = salts
        for c in top:
            for salt in pt_salts:
                reqs.append({**RESET, "seed": s, "replay": b["tape"][:p["t"]], "branch": salt, "first": c}); index.append((j, c, salt))
    t1 = time.time()
    res = run_batch(net, a.device, env, reqs, record=False)
    print(f"分枝 {len(reqs)} 局，用时 {time.time() - t1:.0f} 秒（{len(reqs) / max(1, time.time() - t1):.1f} 局/秒），卡住 {sum(r['stuck'] for r in res)}", flush=True)
    agg: dict = {}
    for (j, c, salt), r in zip(index, res):
        agg.setdefault(j, {}).setdefault(c, []).append([salt, r["win"], r["floor"], r["stuck"]])
    with open(out / "labels.jsonl", "w", encoding="utf-8") as f:
        for j, (s, b, p) in enumerate(jobs):
            opts = agg.get(j, {})
            f.write(json.dumps({"seed": s, "t": p["t"], "d": p["d"], "act": p["act"], "floor": p["floor"], "base_win": b["win"], "deck": p.get("deck"),
                                "options": [{"a": c, "name": p["names"][c], "p": p["p"][c], "runs": opts[c]} for c in opts]},
                               ensure_ascii=False) + "\n")
    print(f"写入 {out / 'labels.jsonl'}", flush=True)


def report(path: str) -> None:
    """各点：策略首选对其他选项按 salt 配对的胜负差（其他 − 首选）。先看整体，再按类别汇总（备选是不是跳过、首选是不是跳过、幕、
    备选牌、首选牌），每类给配对数、平均胜负差与 95% 区间；另给同一 salt 下首选与备选结局的相关 ρ（决定配对能压掉多少噪声）。"""
    from collections import defaultdict
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    pairs = []      # (差, 首选赢, 备选赢, 类别字典)
    best_is_top, spread = 0, []
    for r in rows:
        opts = sorted(r["options"], key=lambda o: -o["p"])
        if len(opts) < 2:
            continue
        top = opts[0]; tw = {s: w for s, w, _, _ in top["runs"]}
        rates = [sum(w for _, w, _, _ in o["runs"]) / len(o["runs"]) for o in opts]
        spread.append(max(rates) - min(rates)); best_is_top += int(rates[0] >= max(rates))
        for o in opts[1:]:
            for s_, w, _, _ in o["runs"]:
                if s_ in tw:
                    pairs.append((w - tw[s_], tw[s_], w, {"整体": "全部", "决策类型": KIND_NAMES.get(r["d"], str(r["d"])), "备选": "跳过" if "SKIP" in o["name"] else "拿牌",
                                                       "首选": "跳过" if "SKIP" in top["name"] else "拿牌", "幕": f"第{r['act'] + 1}幕",
                                                       "备选牌": o["name"].removeprefix("CARD."), "首选牌": top["name"].removeprefix("CARD.")}))
    n = len(rows); m = len(pairs)
    print(f"{n} 个标注点、{m} 对配对续局；策略首选在本点各选项里最高（含并列）的占 {best_is_top / max(1, n):.0%}；"
          f"每点选项间通关率极差中位 {sorted(spread)[len(spread) // 2] if spread else 0:.2f}")
    if m > 2:
        a_ = [p[1] for p in pairs]; b_ = [p[2] for p in pairs]
        ma, mb = sum(a_) / m, sum(b_) / m
        va = sum((x - ma) ** 2 for x in a_) / m; vb = sum((x - mb) ** 2 for x in b_) / m
        cov = sum((x - ma) * (y - mb) for x, y in zip(a_, b_)) / m
        print(f"同一 salt 下首选与备选结局的相关 ρ = {cov / (va * vb) ** 0.5 if va > 0 and vb > 0 else float('nan'):.3f}（首选通关 {ma:.1%}、备选 {mb:.1%}）")
    for key in ("整体", "决策类型", "备选", "首选", "幕", "备选牌", "首选牌"):
        g = defaultdict(list)
        for d, _, _, c in pairs:
            g[c[key]].append(d)
        items = sorted(g.items(), key=lambda kv: -len(kv[1]))[: (12 if key.endswith("牌") else 99)]
        print(f"按{key}：" + "；".join(f"{k} {len(v)} 对 {sum(v) / len(v):+.3f}±{1.96 * (sum((x - sum(v) / len(v)) ** 2 for x in v) / max(1, len(v) - 1)) ** 0.5 / len(v) ** 0.5:.3f}"
                                    for k, v in items if len(v) >= (30 if key.endswith("牌") else 1)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default=None)
    ap.add_argument("--ckpt"); ap.add_argument("--seeds-file"); ap.add_argument("--games", type=int, default=50); ap.add_argument("--seed-from", type=int, default=0)
    ap.add_argument("--out"); ap.add_argument("--topk", type=int, default=3); ap.add_argument("--salts", type=int, default=8)
    ap.add_argument("--per-game", type=int, default=2); ap.add_argument("--kinds", default="3", help="要标注的决策类型（DecisionKind 编号，逗号分隔；3 为选卡奖励）")
    ap.add_argument("--max-top", type=float, default=1.01, help="策略首选概率不低于这个值的点不标（太确定的点不值得花算力）")
    ap.add_argument("--envs", type=int, default=40); ap.add_argument("--device", default="cuda"); ap.add_argument("--rng", type=int, default=258)
    ap.add_argument("--pairs", action="store_true", help="汇总设计：每点只比首选与一个备选，salt 每点各自抽（--salts 建议 1–2）；要汇总很多点看哪一类决策整体偏了")
    ap.add_argument("--dll", default=None, help="引擎程序集路径（默认仓库里的 Release 输出）")
    a = ap.parse_args()
    if a.report:
        report(a.report)
    else:
        label(a)


if __name__ == "__main__":
    main()
