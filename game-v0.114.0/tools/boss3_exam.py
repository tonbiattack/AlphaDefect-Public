"""双 Boss 诊断卷（2026-10-29 起，决策记录 197）：冻结千种子复测里打到第三幕第一个 Boss 的局面，连打两个 Boss，按题配对比较不同版本。

旧的固定考卷（exam.json，第二幕、第三幕 Boss 战前各 256 个单场题）分不清“攒下的资源更多”和“同样资源下更会打”，也没有连打，
已停用。这套题取自同一版本千种子复测的整局（tools/ability_test.py 记下的 boss3 项：第三幕第一个 Boss 开打前的快照与两个 Boss），
按六种 Boss 顺序 × 入场血量三档 × 药水两档分层抽题，每层最多 --per 题；每题换一个固定种子，连打两个 Boss（与训练的连打练习同一规格）。

  python3 tools/boss3_exam.py build 复测目录 [--per 12] [--out ~/.local/share/sls2/boss3_exam.json]
  .venv/bin/python -I tools/boss3_exam.py run --ckpt 检查点.pt --exam 考卷.json --out 结果.jsonl [--envs 40] [--device cuda]
  python3 tools/boss3_exam.py compare 结果A.jsonl 结果B.jsonl [--exam 考卷.json]

结果每题一行：题号、输赢（两场都赢）、有没有打到第二场；compare 按题配对（只一边赢的题数与符号检验），并按 Boss 顺序、血量档分开。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

DEFAULT = Path.home() / ".local/share/sls2/boss3_exam.json"
HP_BINS = ((0.0, 0.4, "血量 <40%"), (0.4, 0.7, "血量 40–70%"), (0.7, 9.0, "血量 ≥70%"))


def hp_bin(f: float) -> str:
    return next(name for lo, hi, name in HP_BINS if lo <= f < hi)


def cell(it: dict) -> tuple[str, str, str]:
    nxt = it["next_encounter"].removeprefix("ENCOUNTER.")     # 头部给的是词表键（带前缀），规格里照训练的连打练习原样用
    return f'{it["encounter"]} → {nxt}', hp_bin(it["hp_frac"]), "药水 0–1" if (it["potions"] or 0) <= 1 else "药水 ≥2"


def build(a) -> None:
    import random
    rng = random.Random(a.seed)
    src = []
    for f in sorted(Path(a.dir).glob("runs_s*.jsonl")):
        for line in open(f, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                b = r.get("boss3")
                if b and b.get("snap") and b.get("next_encounter"):
                    src.append({"from_seed": r["seed"], "version": r.get("version"), "encounter": b["encounter"], "next_encounter": b["next_encounter"],
                                "hp_frac": (b.get("hp") or 0) / max(1, b.get("max_hp") or 1), "potions": b.get("potions"), "upgrades": b.get("upgrades"),
                                "deck": b.get("deck"), "snap": b["snap"]})
    if not src:
        sys.exit(f"{a.dir} 里没有第三幕 Boss 的快照（要用记 boss3 的新版 ability_test 打的复测）")
    groups: dict[tuple, list] = defaultdict(list)
    for it in src:
        groups[cell(it)].append(it)
    items = []
    for key in sorted(groups):
        g = groups[key]
        items += rng.sample(g, min(a.per, len(g)))
    for k, it in enumerate(items):
        it["id"] = k
        it["spec"] = {"character": "DEFECT", "ascension": 10, "seed": f"BX{k:04d}", "act": 2, "encounter": it["encounter"], "room": "Boss",
                      "next_encounter": it["next_encounter"], "player_snapshot": it.pop("snap")}
    out = Path(a.out).expanduser()
    out.write_text(json.dumps({"built_from": str(a.dir), "per": a.per, "cells": len(groups), "items": items}, ensure_ascii=False), encoding="utf-8")
    print(f"{len(src)} 个局面、{len(groups)} 层，抽 {len(items)} 题 → {out}")


def run(a) -> None:
    import threading
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
    from sts2rl.actor import MODE_COMBAT, MODE_RUN
    from sts2rl.evaluator import play
    from sts2rl.model import ModelConfig, PolicyValueNet, load_state
    exam = json.loads(Path(a.exam).expanduser().read_text(encoding="utf-8"))["items"]
    blob = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    net = PolicyValueNet(ModelConfig(**blob["config"]["model"])).eval().to(a.device)
    load_state(net, blob["state"])
    if a.perturb > 0:      # 诊断 D4：进场快照里带着战斗随机数状态，换种子不起作用，改用权重扰动换出手
        g = torch.Generator().manual_seed(a.perturb_seed)
        with torch.no_grad():
            for prm in net.parameters():
                if prm.numel() > 1:
                    prm.add_(torch.randn(prm.shape, generator=g).to(prm.device) * prm.float().std() * a.perturb)
    it = iter(exam)

    def next_task():
        e = next(it, None)
        if e is None:
            return None
        spec = {**e["spec"], "seed": f"{e['spec']['seed']}R{a.reseed}"} if a.reseed is not None else e["spec"]
        return {"mode": "combat", "spec": spec, "tag": e["id"]}, MODE_COMBAT, "exam"
    # 模式一致性对照（决策记录 204）：--mode-as run 让网络把这些 Boss 战当成整局里的来看（引擎照常按战斗练习打），--mode-cmp 每步再用另一种模式算一遍做比较
    modes = {"combat": MODE_COMBAT, "run": MODE_RUN}
    pol = {**({"mode_as": modes[a.mode_as]} if a.mode_as else {}), **({"mode_cmp": modes[a.mode_cmp]} if a.mode_cmp else {}),
           **({"merge": True} if a.merge_equiv else {})}
    res = play(net, a.device, next_task, a.envs, threading.Event(), None, {"exam": 3000}, None, int(blob["version"]), policy=pol or None)
    if pol.get("_cmp"):
        n, same, kl = pol["_cmp"]
        print(f"模式对照：{n} 步，两种模式贪心动作相同 {same / max(n, 1):.1%}，平均 KL {kl / max(n, 1):.4f}")
    with open(a.out, "w", encoding="utf-8") as f:
        for r in sorted(res, key=lambda r: r["tag"]):
            f.write(json.dumps({"id": r["tag"], "version": int(blob["version"]), "win": int(r["win"]), "stage2": int(bool(r.get("stage2"))),
                                "stuck": int(r["stuck"])}, ensure_ascii=False) + "\n")
    print(f"第 {blob['version']} 版：{sum(r['win'] for r in res)}/{len(res)} 题两场都赢，{sum(bool(r.get('stage2')) for r in res)} 题打到第二场")


def load(p: str) -> dict[int, dict]:
    return {r["id"]: r for r in (json.loads(l) for l in open(p, encoding="utf-8") if l.strip())}


def sign_p(x: int, y: int) -> float:
    """双侧符号检验（只一边赢的题）。"""
    n = x + y
    if n == 0:
        return 1.0
    k = min(x, y)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def compare(a) -> None:
    A, B = load(a.a), load(a.b)
    exam = {e["id"]: e for e in json.loads(Path(a.exam).expanduser().read_text(encoding="utf-8"))["items"]} if a.exam else {}
    ids = [i for i in A if i in B and not A[i]["stuck"] and not B[i]["stuck"]]
    va, vb = next(iter(A.values()))["version"], next(iter(B.values()))["version"]
    x = sum(A[i]["win"] and not B[i]["win"] for i in ids); y = sum(B[i]["win"] and not A[i]["win"] for i in ids)
    out = {"a": va, "b": vb, "n": len(ids), "win_a": sum(A[i]["win"] for i in ids), "win_b": sum(B[i]["win"] for i in ids),
           "only_a": x, "only_b": y, "p": round(sign_p(x, y), 4), "by": {}}
    print(f"第 {va} 版 {out['win_a']}/{len(ids)}，第 {vb} 版 {out['win_b']}/{len(ids)}；只有前者赢 {x} 题、只有后者赢 {y} 题，p = {out['p']:.3f}")
    if exam:
        for dim, f in (("顺序", lambda e: cell(e)[0]), ("血量", lambda e: cell(e)[1]), ("药水", lambda e: cell(e)[2])):
            g: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
            for i in ids:
                c = g[f(exam[i])]; c[0] += 1; c[1] += A[i]["win"]; c[2] += B[i]["win"]
            out["by"][dim] = dict(g)
            print(f"  {dim}：" + "；".join(f"{k} {wa}/{n} 对 {wb}/{n}" for k, (n, wa, wb) in sorted(g.items())))
    if a.json:
        Path(a.json).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("dir"); b.add_argument("--per", type=int, default=12); b.add_argument("--seed", type=int, default=0)
    b.add_argument("--out", default=str(DEFAULT))
    r = sub.add_parser("run"); r.add_argument("--ckpt", required=True); r.add_argument("--exam", default=str(DEFAULT)); r.add_argument("--out", required=True)
    r.add_argument("--envs", type=int, default=40); r.add_argument("--device", default="cuda")
    r.add_argument("--mode-as", choices=("combat", "run"), default=None, help="网络输入的模式标记换成这个（默认按战斗练习）")
    r.add_argument("--mode-cmp", choices=("combat", "run"), default=None, help="每步另用这个模式再算一遍，统计贪心动作相同的比例与 KL")
    r.add_argument("--merge-equiv", action="store_true", help="贪心按等价动作合并（决策记录 205）")
    r.add_argument("--reseed", type=int, default=None, help="平台期诊断 D4（2026-11-03）：每题的战斗种子后面加 R<N>。实测不起作用：进场快照自带战斗随机数状态")
    r.add_argument("--perturb", type=float, default=0.0, help="诊断 D4：权重扰动系数（同 tools/ability_test.py --perturb）")
    r.add_argument("--perturb-seed", type=int, default=0)
    c = sub.add_parser("compare"); c.add_argument("a"); c.add_argument("b"); c.add_argument("--exam", default=None); c.add_argument("--json", default=None)
    a = ap.parse_args()
    {"build": build, "run": run, "compare": compare}[a.cmd](a)


if __name__ == "__main__":
    main()
