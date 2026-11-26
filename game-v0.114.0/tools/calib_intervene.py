"""干预式标定（后续计划阶段 0 第 4 步）：在固定考卷的 Boss 战前局面上，一次只扰动一样资源，用同一个种子各打一遍（共同随机数），
按题配对比较胜率，估计每样资源在 Boss 战里值多少胜率。结果用来校对整局里的资源塑形系数（血量、药水、升级、遗物）。

    .venv/bin/python -I tools/calib_intervene.py [--exam 运行目录/exam.json] [--ckpt 检查点.pt] [--reps 2] [--workers 6] [--envs 4] [--out 结果.json]

扰动（每题各打一遍，与“原样”配对）：
    hp15     开场血量减去最大生命的 15%（至少留 1 点）
    potion   去掉一瓶药水（第一瓶；没有药水的题不计）
    upgrade  撤掉一张牌的一次升级（随机挑一张升级过的牌；没有升级牌的题不计）
    relic    去掉一件遗物（随机挑一件非初始、非 Boss、非先古的遗物；没有的题不计）
每种扰动 × 每个重复种子都和同种子的“原样”配对；报告胜率差（扰动 − 原样）及其标准误（配对：sqrt(b + c) / n）。
在本机跑（贪心、CPU 推理），不占租用机。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

import torch  # noqa: E402

from sts2rl import evaluator  # noqa: E402
from sts2rl.actor import MODE_COMBAT  # noqa: E402
from sts2rl.config import TrainConfig  # noqa: E402
from sts2rl.model import PolicyValueNet, load_state  # noqa: E402

TRANSFER = Path.home() / ".local/share/sls2/runs/remote/transfer"
CATALOG = REPO / "engine/catalog/v0.111.0/catalog.json"
KEEP_RARITY = {"Starter", "Boss", "Ancient"}       # 这几类遗物不拿来扰动


def relic_rarity() -> dict[str, str]:
    ents = json.loads(CATALOG.read_text(encoding="utf-8"))["entities"]
    return {e["key"]: e.get("rarity", "") for e in ents if e.get("category") == "RELIC"}


def perturb(snap_json: str, how: str, rng: random.Random, rarity: dict[str, str]) -> str | None:
    """返回扰动后的玩家快照；这道题做不了这种扰动时返回 None。"""
    p = json.loads(snap_json)
    if how == "hp15":
        p["current_hp"] = max(1, int(p["current_hp"]) - round(0.15 * int(p["max_hp"])))
    elif how == "potion":
        pots = [x for x in p.get("potions") or [] if x]
        if not pots:
            return None
        p["potions"] = [x for x in p["potions"] if x is not pots[0]]
    elif how == "upgrade":
        up = [c for c in p.get("deck") or [] if c.get("current_upgrade_level", 0) > 0]
        if not up:
            return None
        c = rng.choice(up)
        c["current_upgrade_level"] -= 1
        if c["current_upgrade_level"] == 0:
            del c["current_upgrade_level"]
    elif how == "relic":
        cand = [r for r in p.get("relics") or [] if rarity.get(r["id"], "") not in KEEP_RARITY]
        if not cand:
            return None
        drop = rng.choice(cand)
        p["relics"] = [r for r in p["relics"] if r is not drop]
    else:
        raise ValueError(how)
    return json.dumps(p, ensure_ascii=False)


def latest(name: str) -> Path | None:
    xs = sorted(TRANSFER.glob(f"*/{name}"), key=lambda q: q.stat().st_mtime)
    return xs[-1] if xs else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exam", default=None, help="exam.json（默认取本机最近一次迁移、分叉的中转目录里的）")
    ap.add_argument("--ckpt", default=None, help="检查点（默认同上的 latest.pt）")
    ap.add_argument("--reps", type=int, default=2, help="每题的重复种子数")
    ap.add_argument("--hows", default="hp15,potion,upgrade,relic")
    ap.add_argument("--limit", type=int, default=0, help="只用前 N 道题（试跑）")
    ap.add_argument("--workers", type=int, default=1, help="（保留）")
    ap.add_argument("--envs", type=int, default=24, help="并行引擎数")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    exam_p = Path(a.exam) if a.exam else latest("exam.json")
    ckpt_p = Path(a.ckpt) if a.ckpt else latest("latest.pt")
    if not exam_p or not ckpt_p:
        sys.exit("找不到 exam.json 或检查点，请用 --exam、--ckpt 指定")
    items = json.loads(exam_p.read_text(encoding="utf-8"))["items"]
    if a.limit:
        items = items[:a.limit]
    blob = torch.load(ckpt_p, map_location="cpu", weights_only=False)
    cfg = TrainConfig()
    for k, v in (blob.get("config") or {}).get("model", {}).items():
        setattr(cfg.model, k, v)
    net = PolicyValueNet(cfg.model).eval()
    load_state(net, blob["state"])
    rarity = relic_rarity()
    hows = a.hows.split(",")

    # 题目 × 重复种子 × （原样 + 各扰动）；扰动的随机挑选按题目与种子固定，可复现
    tasks, meta = [], []
    for qi, it in enumerate(items):
        spec0 = it["spec"]
        for r in range(a.reps):
            seed = f'{spec0.get("seed", "EXAM")}_cal{r}'
            for how in ["base"] + hows:
                snap = spec0["player_snapshot"] if how == "base" else perturb(spec0["player_snapshot"], how, random.Random(f"{qi}:{r}:{how}"), rarity)
                if snap is None:
                    continue
                tag = len(meta)
                meta.append((qi, r, how, int(spec0.get("act", 0))))
                tasks.append(({"mode": "combat", "spec": dict(spec0, seed=seed, player_snapshot=snap), "tag": tag}, MODE_COMBAT, "calib"))
    print(f"考卷 {exam_p}（{len(items)} 题），检查点 {ckpt_p}（第 {blob.get('version')} 版），共 {len(tasks)} 场", flush=True)
    it_ = iter(tasks)
    t0 = time.time()
    res = evaluator.play(net, torch.device("cpu"), lambda: next(it_, None), a.envs, threading.Event())
    print(f"打完 {len(res)} 场，用时 {time.time() - t0:.0f} 秒", flush=True)
    win = {}
    for r_ in res:
        if r_.get("tag") is not None and not r_["stuck"]:
            win[meta[r_["tag"]][:3]] = bool(r_["win"])
    report = {"exam": str(exam_p), "ckpt": str(ckpt_p), "version": blob.get("version"), "reps": a.reps, "results": {}}
    for act in (1, 2):
        for how in hows:
            pairs = [(win[(q, r, "base")], win[(q, r, how)]) for (q, r, h, ac) in meta
                     if h == how and ac == act and (q, r, "base") in win and (q, r, how) in win]
            n = len(pairs)
            if not n:
                continue
            b = sum(1 for x, y in pairs if x and not y)      # 原样赢、扰动输
            c = sum(1 for x, y in pairs if not x and y)      # 原样输、扰动赢
            base_w = sum(x for x, _ in pairs) / n
            d, se = (c - b) / n, (b + c) ** 0.5 / n
            report["results"][f"act{act + 1}:{how}"] = {"n": n, "base": round(base_w, 4), "diff": round(d, 4), "se": round(se, 4), "b": b, "c": c}
            print(f"第{'一二三'[act]}幕 Boss · {how:8s} {n:4d} 对：原样胜率 {base_w:.1%}，扰动后 {base_w + d:.1%}，"
                  f"差 {d * 100:+.1f} ± {se * 100:.1f} 个百分点（原样赢扰动输 {b}，反之 {c}）", flush=True)
    out = Path(a.out) if a.out else REPO.parent / f"calib_intervene_{time.strftime('%Y%m%d_%H%M')}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print("结果写到", out)


if __name__ == "__main__":
    main()
