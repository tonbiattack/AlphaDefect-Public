"""SL 快照池万种级联测试的选题与记账（2026-11-02，决策记录 251，tools/remote/w10k.sh 调）。
用户：“暂停训练，我想对模型池跑万种测试：最新快照测 10000 个，接下来一个测上一个输的部分，以此类推。充分利用两机性能完成此事，
仅直播服务优先级高于此测试。”

种子：新生成的 1 万个随机种子（tools/seed_pool.py，池名 W1），没参与过任何挑选，测的是 SL 模式在新种子上的上限。
快照：SL 直播快照池（~/.local/share/sls2/live/sl_pool/）按版本从新到旧；第 1 个打满 1 万个种子，之后每个只打前面都没通关的种子。
记录：<目录>/cascade.jsonl 每个快照一行（打了几个、新通关几个、累计并集），<目录>/won.json 已通关的种子与首个通关的快照。

    python3 tools/w10k.py order                      # 打印快照顺序：每行 “版本 名字 检查点路径”
    python3 tools/w10k.py todo 目录 --out 种子表.json   # 写还没通关的种子表，打印个数
    python3 tools/w10k.py merge 目录 结果目录 版本 名字    # 记一个快照的结果，打印一行摘要
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

POOL_DIR = Path.home() / ".local/share/sls2/live/sl_pool"


def cmd_order(a) -> None:
    rows = []
    for f in POOL_DIR.iterdir():
        name = f.name
        if not (name.startswith("v") and name.endswith(".pt")):
            continue
        head = name[1:-3].split("-", 1)[0]
        if not head.isdigit():
            continue
        try:
            rows.append((int(head), name[:-3], str(f.resolve(strict=True))))
        except OSError:
            continue
    for v, n, p in sorted(rows, key=lambda x: (x[0], x[1]), reverse=True):
        print(v, n, p)


def won(d: Path) -> dict:
    p = d / "won.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def cmd_todo(a) -> None:
    d = Path(a.dir)
    seeds = json.loads((d / "seeds.json").read_text(encoding="utf-8"))["seeds"]
    w = won(d)
    todo = [s for s in seeds if s not in w]
    Path(a.out).write_text(json.dumps({"name": "W1", "seeds": todo}, ensure_ascii=False), encoding="utf-8")
    print(len(todo))


def cmd_merge(a) -> None:
    d = Path(a.dir)
    seeds = json.loads((d / "seeds.json").read_text(encoding="utf-8"))["seeds"]
    w = won(d)
    todo = [s for s in seeds if s not in w]
    res = {}
    for f in Path(a.res).glob("runs_s*.jsonl"):
        for line in open(f, encoding="utf-8"):
            if line.strip():
                r = json.loads(line); res[r["seed"]] = r
    played = [s for s in todo if s in res]
    new = [s for s in played if res[s]["win"]]
    for s in new:
        w[s] = a.name
    tmp = d / "won.json.tmp"; tmp.write_text(json.dumps(w, ensure_ascii=False), encoding="utf-8"); os.replace(tmp, d / "won.json")
    rec = {"time": round(time.time(), 1), "version": a.version, "name": a.name, "todo": len(todo), "played": len(played),
           "new": len(new), "stuck": sum(1 for s in played if res[s].get("stuck")), "union": len(w), "seeds": len(seeds)}
    with open(d / "cascade.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"第 {len(open(d / 'cascade.jsonl').readlines())} 个快照 {a.name}：打 {len(played)}/{len(todo)} 个种子，新通关 {len(new)}，"
          f"累计并集 {len(w)}/{len(seeds)} = {len(w) / len(seeds):.2%}" + (f"（卡住 {rec['stuck']}）" if rec["stuck"] else ""))


def _order() -> list[tuple[int, str]]:
    return [(int(l.split()[0]), l.split()[1]) for l in (Path.home() / ".local/share/sls2/ability_test/w10k/order.txt").read_text().split("\n") if l.strip()]


def _partial(d: Path, base: int) -> dict[str, int]:
    """正在打的那个快照（下标 base）已经出的结果：种子 → 0/1。"""
    order = _order()
    if base >= len(order):
        return {}
    res = {}
    for f in (d / f"s{base + 1}-{order[base][1]}").glob("**/runs_s*.jsonl"):
        for line in open(f, encoding="utf-8"):
            if line.strip():
                r = json.loads(line); res[r["seed"]] = int(r["win"])
    return res


def cmd_pipe_tasks(a) -> None:
    """逐种子级联（流水线）的起点：已测完的快照跳过；正在打的快照里已有结果的种子直接用结果（赢了不再打，输了从下一个起）。"""
    d = Path(a.dir)
    seeds = json.loads((d / "seeds.json").read_text(encoding="utf-8"))["seeds"]
    w, base = won(d), len(open(d / "cascade.jsonl").readlines())
    part = _partial(d, base)
    tasks = [[s, base + 1 if s in part else base] for s in seeds if s not in w and not part.get(s)]
    Path(a.out).write_text(json.dumps({"tasks": tasks, "base": base}, ensure_ascii=False), encoding="utf-8")
    print(len(tasks), base, sum(part.values()))


def cmd_pipe_merge(a) -> None:
    """把流水线结果（各进程的结果 jsonl）连同正在打的那个快照的部分结果，按快照汇总，接着写进 cascade.jsonl 与 won.json。"""
    d = Path(a.dir)
    seeds = json.loads((d / "seeds.json").read_text(encoding="utf-8"))["seeds"]
    order = _order()
    w, base = won(d), len(open(d / "cascade.jsonl").readlines())
    per: dict[int, dict[str, int]] = {}
    for s, v in _partial(d, base).items():
        if s not in w:
            per.setdefault(base, {})[s] = v
    for f in Path(a.res).glob("*.jsonl"):
        for line in open(f, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                if r["seed"] not in w:
                    per.setdefault(r["level"], {})[r["seed"]] = int(r["win"])
    with open(d / "cascade.jsonl", "a", encoding="utf-8") as fo:
        for lv in range(base, len(order)):
            res = per.get(lv, {})
            todo = len([s for s in seeds if s not in w])
            new = [s for s, v in res.items() if v and s not in w]
            for s in new:
                w[s] = order[lv][1]
            rec = {"time": round(time.time(), 1), "version": order[lv][0], "name": order[lv][1], "todo": todo, "played": len(res),
                   "new": len(new), "stuck": 0, "union": len(w), "seeds": len(seeds), "mode": "pipeline"}
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"第 {lv + 1} 个快照 {order[lv][1]}：打 {len(res)}/{todo} 个种子，新通关 {len(new)}，累计并集 {len(w)}/{len(seeds)} = {len(w) / len(seeds):.2%}")
    tmp = d / "won.json.tmp"; tmp.write_text(json.dumps(w, ensure_ascii=False), encoding="utf-8"); os.replace(tmp, d / "won.json")


def cmd_extra_models(a) -> None:
    """附加一轮的快照（用户：“把最近跑过首通测试的，不在模型池里的模型，再跑一遍首通测试”）：首通探测测过、但不在 SL 快照池里的，
    按探测时间从近到远；每行 “名字 本机检查点 A 上路径”。"""
    reg = json.loads((Path.home() / ".local/share/sls2/ability_test/retest/unsolved_probe.json").read_text(encoding="utf-8"))
    pool_targets, pool_names = set(), set()
    for f in POOL_DIR.iterdir():
        pool_names.add(f.name[:-3])
        try:
            pool_targets.add(str(f.resolve(strict=True)))
        except OSError:
            pass
    seen = set()
    for p in sorted(reg["probes"], key=lambda x: -x["time"]):
        ck = p.get("ckpt") or ""
        name = Path(p["dir"]).name.replace("-首通探测", "")
        if not ck or name in seen or name in pool_names or ck in pool_targets or not Path(ck).exists():
            continue
        seen.add(name)
        print(name, ck, f"/root/retest/{Path(p['dir']).name}/model.pt")


def cmd_extra_tasks(a) -> None:
    d = Path(a.dir)
    seeds = json.loads((d / "seeds.json").read_text(encoding="utf-8"))["seeds"]
    w = won(d)
    tasks = [[s, 0] for s in seeds if s not in w]
    Path(a.out).write_text(json.dumps({"tasks": tasks}, ensure_ascii=False), encoding="utf-8")
    print(len(tasks))


def cmd_extra_merge(a) -> None:
    """附加一轮的结果：按附加快照的顺序汇总，写 cascade_extra.jsonl，并把新通关的记进 won.json。"""
    d = Path(a.dir)
    seeds = json.loads((d / "seeds.json").read_text(encoding="utf-8"))["seeds"]
    names = [l.split()[0] for l in Path(a.models).read_text(encoding="utf-8").split("\n") if l.strip()]
    w = won(d)
    per: dict[int, dict[str, int]] = {}
    for f in Path(a.res).glob("*.jsonl"):
        for line in open(f, encoding="utf-8"):
            if line.strip():
                r = json.loads(line); per.setdefault(r["level"], {})[r["seed"]] = int(r["win"])
    with open(d / "cascade_extra.jsonl", "a", encoding="utf-8") as fo:
        for lv, name in enumerate(names):
            res = per.get(lv, {})
            todo = len([s for s in seeds if s not in w])
            new = [s for s, v in res.items() if v and s not in w]
            for s in new:
                w[s] = name
            rec = {"time": round(time.time(), 1), "name": name, "todo": todo, "played": len(res), "new": len(new), "union": len(w), "seeds": len(seeds)}
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if res:
                print(f"附加第 {lv + 1} 个 {name}：打 {len(res)}/{todo} 个种子，新通关 {len(new)}，累计并集 {len(w)}/{len(seeds)} = {len(w) / len(seeds):.2%}")
    tmp = d / "won.json.tmp"; tmp.write_text(json.dumps(w, ensure_ascii=False), encoding="utf-8"); os.replace(tmp, d / "won.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("order")
    t = sub.add_parser("todo"); t.add_argument("dir"); t.add_argument("--out", required=True)
    m = sub.add_parser("merge"); m.add_argument("dir"); m.add_argument("res"); m.add_argument("version", type=int); m.add_argument("name")
    pt = sub.add_parser("pipe-tasks"); pt.add_argument("dir"); pt.add_argument("--out", required=True)
    pm = sub.add_parser("pipe-merge"); pm.add_argument("dir"); pm.add_argument("res")
    sub.add_parser("extra-models")
    et = sub.add_parser("extra-tasks"); et.add_argument("dir"); et.add_argument("--out", required=True)
    em = sub.add_parser("extra-merge"); em.add_argument("dir"); em.add_argument("res"); em.add_argument("models")
    a = ap.parse_args()
    {"order": cmd_order, "todo": cmd_todo, "merge": cmd_merge, "pipe-tasks": cmd_pipe_tasks, "pipe-merge": cmd_pipe_merge,
     "extra-models": cmd_extra_models, "extra-tasks": cmd_extra_tasks, "extra-merge": cmd_extra_merge}[a.cmd](a)


if __name__ == "__main__":
    main()
