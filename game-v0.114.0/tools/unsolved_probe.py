"""首通探测（2026-11-02 起，决策记录 250）：只拿种子池 P1 里还没有任何快照通关过的种子，让一个个快照去打，
一旦有快照首次通关某个种子，立即把它链进 SL 直播快照池并记下来（看板显示）。tools/remote/unsolved_probe.sh 调它。
用户：“非定时复测更改策略，只测全局，只测复测池里的没有通关过的几个种子，这样大幅度加快速度，一旦有新的首通，马上拉入直播快照池
并记录入看板”“然后加测频率大大增加，保持一直有加测”。

全局未通关：P1 定时复测、并集补测纳入的快照都没赢过，首通探测也还没赢过的种子。同一快照打同一种子是确定的，所以每个快照只探测一次。
选快照：A 上常规评测最新留存的权重（eval_runs/，约每 6 分钟一版）里没测过的优先；没有新的时（用户：“间隔期间可以往前测格式符合的
从未测试过的留档，有贡献就拉到贡献池”），从新到旧取过往从没在 P1 上测过的留档——以前的定时复测（RT 种子，本机结果目录的 model.pt）、
A 上评测的前几名留档（best/）与偏高留档（high/）、B 的定时备份（backups/，取其中的滑动平均权重 ema）。只收观测格式 19（第 130,420 版）以后的。
回退过主线，备份的版本号可能与别的快照重复，所以备份按文件名认（id 为 B:文件名），链进快照池时名字带备份时间。
记录：retest/unsolved_probe.json（probes 每次探测、firsts 每个首通）。

    python3 tools/unsolved_probe.py seeds --out 种子表.json          # 写全局未通关的种子表（池名 P1），打印个数
    python3 tools/unsolved_probe.py next --eval-runs 列表 --best best.json --high high.json   # 打印 "EVAL 版本 子目录" 或 "LOCAL 路径 版本"
    python3 tools/unsolved_probe.py judge 版本 结果目录 检查点 [--tested 种子表.json]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

AT = Path.home() / ".local/share/sls2/ability_test"
REG = AT / "retest/unsolved_probe.json"
POOL_DIR = Path.home() / ".local/share/sls2/live/sl_pool"
W1_DIR = AT / "w10k"         # 万种测试的种子池 W1 与通关记录
MIN_V = 130420
POOL_AFTER = 3        # 首通探测做满几轮之后，快照池里没测过的快照优先
POOL = "P1"


def reg() -> dict:
    d = json.loads(REG.read_text(encoding="utf-8")) if REG.exists() else {}
    d.setdefault("probes", []); d.setdefault("firsts", [])
    return d


def save(d: dict) -> None:
    tmp = REG.with_suffix(".tmp"); tmp.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8"); os.replace(tmp, REG)


def wins(d) -> dict[str, int]:
    w = {}
    for f in Path(d).glob("runs_s*.jsonl"):
        for line in open(f, encoding="utf-8"):
            if line.strip():
                r = json.loads(line); w[r["seed"]] = int(r["win"])
    return w


def full_snapshots() -> list[tuple[int, str]]:
    """P1 上打满的快照：定时复测（同一版本只算一次）与并集补测纳入的。"""
    out, seen = [], set()
    for r in json.loads((AT / "retest/state.json").read_text(encoding="utf-8"))["runs"]:
        if r.get("seed_pool") == POOL and r["version"] not in seen:
            out.append((r["version"], r["dir"])); seen.add(r["version"])
    bf = AT / "retest/union_backfill.json"
    for x in (json.loads(bf.read_text(encoding="utf-8")).get("accepted", []) if bf.exists() else []):
        if x["version"] not in seen:
            out.append((x["version"], x["dir"])); seen.add(x["version"])
    return out


def unsolved() -> list[str]:
    seeds, won = None, set()
    for _, d in full_snapshots():
        w = wins(d)
        if len(w) < 1000:
            continue
        seeds = set(w) if seeds is None else seeds & set(w)
        won |= {s for s, v in w.items() if v}
    won |= {x["seed"] for x in reg()["firsts"]}
    pool = json.loads((Path(__file__).resolve().parent / "remote/seed_pool.json").read_text(encoding="utf-8"))["seeds"]
    out = [s for s in pool if (seeds is None or s in seeds) and s not in won]
    # 万种测试（种子池 W1，决策记录 251）里所有快照都没通关的种子也一起探测（11-02 用户：“把这 42 个剩下的种子加进首通探测里面”）
    w1 = W1_DIR / "seeds.json"
    if w1.exists():
        w1won = json.loads((W1_DIR / "won.json").read_text(encoding="utf-8")) if (W1_DIR / "won.json").exists() else {}
        out += [s for s in json.loads(w1.read_text(encoding="utf-8"))["seeds"] if s not in w1won and s not in won]
    return out


def tested_versions() -> set[int]:
    t = {v for v, _ in full_snapshots()}
    bf = AT / "retest/union_backfill.json"
    if bf.exists():
        b = json.loads(bf.read_text(encoding="utf-8"))
        t |= {x["version"] for k in ("accepted", "rejected", "failed") for x in b.get(k, [])}
    t |= {x["version"] for x in reg()["probes"] if not x.get("id")}
    return t


def tested_ids() -> set[str]:
    return {x["id"] for x in reg()["probes"] if x.get("id")}


def cmd_seeds(a) -> None:
    s = unsolved()
    Path(a.out).write_text(json.dumps({"name": POOL, "seeds": s, "note": "全局未通关的种子（tools/unsolved_probe.py）"}, ensure_ascii=False), encoding="utf-8")
    print(len(s))


def pool_untested(done: set[int]) -> list[tuple[int, str]]:
    """SL 直播快照池里还没在 P1 上测过的快照（按链接指向的检查点），版本从新到旧。备份链接（v<版本>-<日期>-…）按 id 判断。"""
    out, ids = [], tested_ids()
    if not POOL_DIR.is_dir():
        return out
    for f in POOL_DIR.iterdir():
        name = f.name
        if not (name.startswith("v") and name.endswith(".pt")):
            continue
        head = name[1:-3].split("-", 1)
        if not head[0].isdigit():
            continue
        v = int(head[0])
        if v < MIN_V or (len(head) == 1 and v in done) or (len(head) == 2 and f"B:{name}" in ids):
            continue
        try:
            out.append((v, str(f.resolve(strict=True))))
        except OSError:
            continue
    return sorted(out, reverse=True)


def cmd_next(a) -> None:
    done = tested_versions()
    # 前 POOL_AFTER 轮之后，SL 直播快照池里还没在 P1 上测过的快照优先（11-02 用户：“前几轮首通测试以后，把直播模型池未经测试的快照都跑一下首通测试”）
    if len(reg()["probes"]) >= POOL_AFTER:
        pu = pool_untested(done)
        if pu:
            print(f"LOCAL {pu[0][1]} {pu[0][0]}"); return
    er = []
    if a.eval_runs and Path(a.eval_runs).exists():
        er = [int(x[1:-3]) for x in Path(a.eval_runs).read_text(encoding="utf-8").split() if x.startswith("v") and x.endswith(".pt") and x[1:-3].isdigit()]
    new = [v for v in er if v not in done and v >= MIN_V]
    if new:
        print(f"EVAL {max(new)} eval_runs"); return
    cand: dict[int, str] = {}
    for idx, key in ((a.best, "best"), (a.high, "high")):
        if idx and Path(idx).exists():
            try:
                for x in json.loads(Path(idx).read_text(encoding="utf-8")).get(key, []):
                    if int(x["version"]) >= MIN_V:
                        cand[int(x["version"])] = f"EVAL {x['version']} {key}"
            except ValueError:
                pass
    for r in json.loads((AT / "retest/state.json").read_text(encoding="utf-8"))["runs"]:
        ck = Path(r["dir"]) / "model.pt"
        if r["version"] >= MIN_V and ck.exists():
            cand[r["version"]] = f"LOCAL {ck} {r['version']}"
    # B 的定时备份（文件名 v<版本>-<日期>-<时刻>[-标签].pt），按文件名去重
    ids = tested_ids()
    bk = []
    if a.backups and Path(a.backups).exists():
        for name in Path(a.backups).read_text(encoding="utf-8").split():
            if name.startswith("v") and name.endswith(".pt") and "-" in name and name[1:name.index("-")].isdigit():
                v = int(name[1:name.index("-")])
                if v >= MIN_V and f"B:{name}" not in ids:
                    bk.append((v, name))
    items = [(v, cand[v]) for v in cand if v not in done] + [(v, f"BACKUP {name} {v}") for v, name in bk]
    if items:
        print(max(items, key=lambda x: (x[0], x[1].startswith("BACKUP")))[1]); return
    print("")


def cmd_judge(a) -> None:
    d = reg()
    w = wins(a.dir)
    tested = json.loads(Path(a.tested).read_text(encoding="utf-8"))["seeds"] if a.tested else sorted(w)
    got = [s for s in tested if s in w]
    won = [s for s in got if w[s]]
    rec = {"version": a.version, "dir": a.dir, "ckpt": a.ckpt, "time": round(time.time(), 1), "tested": len(tested), "played": len(got), "wins": won,
           **({"id": a.id} if a.id else {})}
    d["probes"].append(rec)
    linked = None
    if won:
        for s in won:
            d["firsts"].append({"seed": s, "version": a.version, "time": rec["time"], **({"id": a.id} if a.id else {})})
        # W1 的种子首通时同步记进万种测试的通关记录
        if (W1_DIR / "won.json").exists():
            w1 = json.loads((W1_DIR / "won.json").read_text(encoding="utf-8"))
            w1s = set(json.loads((W1_DIR / "seeds.json").read_text(encoding="utf-8"))["seeds"])
            add = {s: Path(a.dir).name.replace("-首通探测", "") for s in won if s in w1s and s not in w1}
            if add:
                w1.update(add); tmp = W1_DIR / "won.json.tmp"; tmp.write_text(json.dumps(w1, ensure_ascii=False), encoding="utf-8"); os.replace(tmp, W1_DIR / "won.json")
        link = POOL_DIR / (a.link_name or f"v{a.version}.pt")
        if POOL_DIR.is_dir() and not link.exists() and not link.is_symlink():
            os.symlink(a.ckpt, link); linked = str(link)
        rec["pool_link"] = linked or (str(link) if link.exists() else None)
    save(d)
    if len(got) < len(tested):
        print(f"  首通探测：第 {a.version:,} 版只打完 {len(got)}/{len(tested)} 个种子")
    if won:
        print(f"  首通探测：第 {a.version:,} 版首通 {len(won)} 个种子 {won}"
              + (f"，已链进直播快照池（{linked}）" if linked else "，快照池里已有这一版"))
    else:
        print(f"  首通探测：第 {a.version:,} 版在 {len(got)} 个全局未通关的种子上没有首通")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("seeds"); s.add_argument("--out", required=True)
    n = sub.add_parser("next"); n.add_argument("--eval-runs"); n.add_argument("--best"); n.add_argument("--high"); n.add_argument("--backups", help="B 的 backups/ 文件列表")
    j = sub.add_parser("judge"); j.add_argument("version", type=int); j.add_argument("dir"); j.add_argument("ckpt"); j.add_argument("--tested")
    j.add_argument("--id", help="备份的 id（B:文件名）"); j.add_argument("--link-name", help="快照池里的链接名（默认 v<版本>.pt）")
    a = ap.parse_args()
    {"seeds": cmd_seeds, "next": cmd_next, "judge": cmd_judge}[a.cmd](a)


if __name__ == "__main__":
    main()
