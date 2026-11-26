"""实机导入保真：用直播录下的实机帧，检验无头引擎照实机动作重放时，每一帧是否与实机逐字节相同。

数据：
  - 录帧：~/.local/share/sls2/live/frames/<run_id>.frames.bin（u32 小端长度 + 帧字节，逐帧首尾相接）与 <run_id>.jsonl
    （每帧一行，chosen_action 为实机执行的动作；只录 num_actions > 0 的决策帧，结局帧不录，见 python/sts2env/record.py）。
  - 局：~/.local/share/sls2/live/runs.jsonl（局末写一行：time 为结束时间（本地时间）、seconds、steps、run_seed、character、
    ascension、game_mode、loads）。2026-11-14 15:30 起才记 run_seed，之前的局无法重放。
  - 录帧文件与局按时间与步数对上：录帧的帧数等于 steps，第一帧在“结束时间 − seconds”前后 90 秒内，最后一帧不晚于结束时间。

帧集固定（build 子命令）：把对上的局写进清单文件（默认 tools/clone/live_manifest.json，入库），之后 run 只按清单测，
直播继续录新帧也不会让两次测量不可比。清单记每局的帧数、frames.bin 的字节数与 SHA-1，run 时核对，变了就报错。
  - 排除：中途读档的局（loads 非空）、SL 支线的局（runs_sl.jsonl，时间区间与之重叠的录帧也排除）、对不上或对上多个录帧文件的局、
    正在进行的局（结束时间晚于建清单时最后一条 runs.jsonl）。
  - 分组：v0.113.1 = 2026-11-14 23:19（本机 Steam 更新）之后开始的局；v0.113.0 = 之前开始、有种子的局（11-14 15:16 起）。

重放（run 子命令）：reset 参数照直播无头搜索的镜像进程（sls2-live113 python/sts2rl/headless_search.py on_run_info）：
  {"op":"reset","mode":"run","character","ascension","seed":run_seed,"route_summary":true,"labels":true}，恐惧模式加 "game_mode":"dread"。
  之后按 jsonl 的 chosen_action 逐步 step，每一帧与实机帧比原始字节，跳过头部 12（本进程赢下的战斗数：实机的模组进程跨局累计）；
  头部 14（本进程累计的词表外键数）也跳过，两边不同的帧单独计数。
  - 模组丢掉动作（下一帧与上一帧逐字节相同）：同镜像的做法，引擎按历史 reset + replay 重建到走之前，记一次“丢弃”。
  - 第一处不一致：记下帧号、决策类型、第一处差异（段、条、项、两边的值）。之后照实机动作继续走（候选不够或引擎报错才停），
    另计“全程一致帧”：前缀一致帧数是直播镜像能跟住的帧数，全程一致帧数反映不一致是否只是个别帧的编码差异。
口径：一致率 = 一致帧 / 实机帧（每局实机帧 = jsonl 行数）。

  python -I tools/clone/live_fidelity.py build [--out tools/clone/live_manifest.json]
  python -I tools/clone/live_fidelity.py run --engine base=…/eng-base/Sts2Headless.dll --engine ref=…/ref-113/Sts2Headless.dll [--group v0.113.1]
  python -I tools/clone/live_fidelity.py run --engine ref=… --run 20261115-120000-p1-r9 --verbose     复现一局
默认 --workers 4（每片 1 个引擎进程），引擎进程以 nice 10 运行。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from common import EngineError  # noqa: E402

LIVE = Path.home() / ".local" / "share" / "sls2" / "live"
MANIFEST = Path(__file__).resolve().parent / "live_manifest.json"
V1131_FROM = "2026-11-14 23:19:00"     # 本机 Steam 更新到 v0.113.1 的时间（游戏更新v0.113.1/核对记录.md）
SKIP_LIVE = (C.H12, C.H14)


def _epoch(s: str) -> float:
    return time.mktime(datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timetuple())


def _local(t: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t))


def read_frames(path: Path) -> list[bytes]:
    data = path.read_bytes()
    out, off = [], 0
    while off + 4 <= len(data):
        (n,) = struct.unpack_from("<I", data, off)
        off += 4
        out.append(data[off:off + n])
        off += n
    return out


def sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- build：固定帧集

def build(a) -> None:
    frames_dir = Path(a.frames)
    runs = [json.loads(l) for l in open(a.runs, encoding="utf-8") if l.strip()]
    sl = [json.loads(l) for l in open(a.runs_sl, encoding="utf-8") if l.strip()] if Path(a.runs_sl).exists() else []
    sl_ranges = [(_epoch(r["time"]) - float(r.get("seconds") or 0), _epoch(r["time"])) for r in sl if r.get("time")]
    # 录帧文件的概况：第一帧、最后一帧的时间与帧数
    files = []
    for j in sorted(frames_dir.glob("*.jsonl")):
        if j.stem < a.since_file:
            continue
        lines = [l for l in open(j, encoding="utf-8") if l.strip()]
        if not lines:
            continue
        t0, t1 = json.loads(lines[0])["time"], json.loads(lines[-1])["time"]
        files.append({"run_id": j.stem, "n": len(lines), "t0": t0, "t1": t1})
    last_end = max(_epoch(r["time"]) for r in runs)
    out, skipped = [], Counter()
    used = set()
    for r in runs:
        if not r.get("run_seed"):
            continue
        end = _epoch(r["time"])
        start = end - float(r.get("seconds") or 0)
        if r.get("loads"):
            skipped["中途读档"] += 1
            continue
        if any(s - 60 <= end and start <= e + 60 for s, e in sl_ranges):
            skipped["与 SL 支线重叠"] += 1
            continue
        cand = [f for f in files if f["n"] == int(r["steps"]) and abs(f["t0"] - start) <= 90 and f["t1"] <= end + 5]
        if len(cand) != 1:
            skipped["录帧对不上" if not cand else "录帧对上多个"] += 1
            continue
        f = cand[0]
        if f["run_id"] in used:
            skipped["录帧已被别的局用"] += 1
            continue
        used.add(f["run_id"])
        fb = frames_dir / f"{f['run_id']}.frames.bin"
        group = "v0.113.1" if start >= _epoch(V1131_FROM) else "v0.113.0"
        out.append({"run_id": f["run_id"], "group": group, "run_seed": r["run_seed"], "character": r.get("character"),
                    "ascension": int(r.get("ascension") or 0), "game_mode": r.get("game_mode"), "steps": int(r["steps"]),
                    "start": _local(f["t0"]), "end": r["time"], "victory": bool(r.get("victory")), "act": r.get("act"), "floor": r.get("floor"),
                    "bytes": fb.stat().st_size, "sha1": sha1(fb)})
    man = {"built": time.strftime("%Y-%m-%d %H:%M:%S"), "frames_dir": str(frames_dir), "runs_log": str(a.runs), "v0.113.1_from": V1131_FROM,
           "last_run_end": _local(last_end), "skipped": dict(skipped),
           "groups": dict(Counter(x["group"] for x in out)), "runs": out}
    fr = Counter()
    for x in out:
        fr[x["group"]] += x["steps"]
    man["frames"] = dict(fr)
    Path(a.out).write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"清单 {a.out}：{len(out)} 局 {dict(Counter(x['group'] for x in out))}，帧 {dict(fr)}；排除 {dict(skipped)}")


# ---------------------------------------------------------------- run：重放比对

def reset_req(m: dict) -> dict:
    r = {"op": "reset", "mode": "run", "character": m["character"], "ascension": int(m["ascension"]), "seed": m["run_seed"],
         "route_summary": True, "labels": True}
    if str(m.get("game_mode") or "").lower() == "dread":
        r["game_mode"] = "dread"
    return r


def replay_run(eng: C.Eng, m: dict, frames_dir: Path, verbose: bool = False) -> dict:
    fb = frames_dir / f"{m['run_id']}.frames.bin"
    live = read_frames(fb)
    acts = [int(json.loads(l)["chosen_action"]) for l in open(frames_dir / f"{m['run_id']}.jsonl", encoding="utf-8") if l.strip()]
    res = {"run_id": m["run_id"], "group": m["group"], "frames": len(live), "prefix": 0, "matched": 0, "h14": 0, "dropped": 0,
           "first": None, "stop": None}
    if len(live) != m["steps"] or len(acts) != len(live):
        res["stop"] = f"录帧数变了：{len(live)} 帧 / {len(acts)} 行，清单 {m['steps']}"
        return res
    base = reset_req(m)
    hist: list[int] = []
    e = eng.req(base)
    diverged = False
    for i in range(len(live)):
        if isinstance(e, EngineError):
            res["stop"] = f"第 {i} 帧引擎报错：{str(e)[:160]}"
            break
        if C.h14_differs(e, live[i]):
            res["h14"] += 1
        same = C.same_frame(e, live[i], SKIP_LIVE)
        if not same and i > 0 and C.same_frame(live[i], live[i - 1], SKIP_LIVE) and not C.same_frame(e, live[i - 1], SKIP_LIVE):
            # 模组丢掉了上一个动作（同一局面又发了一次）：引擎按历史重建到走之前
            res["dropped"] += 1
            hist.pop()
            e = eng.req({**base, "replay": list(hist)})
            if isinstance(e, EngineError):
                res["stop"] = f"第 {i} 帧重建报错：{str(e)[:160]}"
                break
            same = C.same_frame(e, live[i], SKIP_LIVE)
        if same:
            res["matched"] += 1
            if not diverged:
                res["prefix"] += 1
        else:
            if not diverged:
                d = C.first_diff(e, live[i], SKIP_LIVE)
                res["first"] = {"frame": i, "live": C.frame_brief(live[i]), "engine": C.frame_brief(e), "diff": d, "key": C.diff_key(d)}
                if verbose:
                    print(f"{m['run_id']} 第 {i} 帧第一处不一致：{d['desc']}", flush=True)
            diverged = True
        a_ = acts[i]
        if a_ >= C.num_actions(e) or C.is_done(e):
            if i + 1 < len(live):
                res["stop"] = f"第 {i} 帧之后引擎候选不够（实机动作 {a_}，引擎 {C.num_actions(e)} 个候选）"
            break
        if i + 1 < len(live):
            hist.append(a_)
            e = eng.req({"op": "step", "action": a_})
    return res


def run_shard(job) -> list[dict]:
    items, name, dll, a = job
    C.be_nice(a.nice)
    log = open(Path(a.log_dir) / f"{name}_{items[0]['run_id'] if items else 0}.log", "ab")
    eng = C.Eng(dll, log, name)
    rows = []
    try:
        for m in items:
            r = replay_run(eng, m, Path(a.frames), a.verbose)
            r["engine"] = name
            rows.append(r)
            if a.verbose:
                print(json.dumps({k: v for k, v in r.items() if k != "first"}, ensure_ascii=False), flush=True)
    finally:
        eng.close()
    return rows


def summarize(rows: list[dict], name: str, group: str) -> dict:
    rs = [r for r in rows if r["engine"] == name and r["group"] == group]
    fr = sum(r["frames"] for r in rs)
    pre = sum(r["prefix"] for r in rs)
    mat = sum(r["matched"] for r in rs)
    full = sum(1 for r in rs if r["prefix"] == r["frames"])
    return {"engine": name, "group": group, "runs": len(rs), "frames": fr, "prefix": pre, "matched": mat, "full_runs": full,
            "prefix_rate": pre / max(1, fr), "matched_rate": mat / max(1, fr), "dropped": sum(r["dropped"] for r in rs),
            "h14": sum(r["h14"] for r in rs), "first_keys": dict(Counter(r["first"]["key"] for r in rs if r["first"]).most_common(12)),
            "stops": dict(Counter((r["stop"] or "")[:40] for r in rs if r["stop"]).most_common(6))}


def run(a) -> None:
    man = json.loads(Path(a.manifest).read_text(encoding="utf-8"))
    runs = [m for m in man["runs"] if (not a.group or m["group"] == a.group) and (not a.run or m["run_id"] in a.run)]
    if a.limit:
        by = {}
        for m in runs:
            by.setdefault(m["group"], []).append(m)
        runs = [m for g in by.values() for m in g[: a.limit]]
    frames_dir = Path(a.frames or man["frames_dir"])
    a.frames = str(frames_dir)
    bad = [m["run_id"] for m in runs if (frames_dir / f"{m['run_id']}.frames.bin").stat().st_size != m["bytes"]]
    if bad:
        sys.exit(f"录帧文件与清单不符（字节数变了）：{bad[:5]}")
    if a.check_sha:
        bad = [m["run_id"] for m in runs if sha1(frames_dir / f"{m['run_id']}.frames.bin") != m["sha1"]]
        if bad:
            sys.exit(f"录帧文件与清单不符（SHA-1 变了）：{bad[:5]}")
    Path(a.log_dir).mkdir(parents=True, exist_ok=True)
    engines = [e.split("=", 1) for e in a.engine]
    t0 = time.time()
    rows = []
    for name, dll in engines:
        w = max(1, min(a.workers, 4, len(runs)))
        order = sorted(runs, key=lambda m: -m["steps"])       # 长的先分，分片更均匀
        jobs = [(order[k::w], name, dll, a) for k in range(w)]
        t1 = time.time()
        if w == 1:
            res = [run_shard(jobs[0])]
        else:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(w) as ex:
                res = list(ex.map(run_shard, jobs))
        got = [r for rs in res for r in rs]
        rows += got
        print(f"[{name}] {len(got)} 局，{time.time() - t1:.0f} 秒", flush=True)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"清单 {a.manifest}（{man['built']} 建，录帧到 {man['last_run_end']}）；用时 {time.time() - t0:.0f} 秒")
    groups = sorted({m["group"] for m in runs})
    summ = []
    for g in groups:
        for name, _ in engines:
            s = summarize(rows, name, g)
            summ.append(s)
            print(f"  {g} 实机 × {name}：{s['runs']} 局 {s['frames']} 帧；前缀一致 {s['prefix']}（{s['prefix_rate']:.4%}），"
                  f"全程一致 {s['matched']}（{s['matched_rate']:.4%}），整局全对 {s['full_runs']} 局；丢弃 {s['dropped']}，头部14不同 {s['h14']}")
            if s["first_keys"]:
                print(f"      第一处不一致的分类：{s['first_keys']}")
            if s["stops"]:
                print(f"      提前停下：{s['stops']}")
    if a.summary:
        Path(a.summary).write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    for r in [r for r in rows if r["first"]][: a.show]:
        f = r["first"]
        print(f"  [{r['engine']}] {r['run_id']} 第 {f['frame']} 帧（决策 {f['live']['decision']}，第 {f['live']['act'] + 1} 幕第 {f['live']['floor']} 层）："
              f"{f['diff']['desc']}；前缀 {r['prefix']}/{r['frames']}，全程 {r['matched']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--frames", default=str(LIVE / "frames"))
    b.add_argument("--runs", default=str(LIVE / "runs.jsonl"))
    b.add_argument("--runs-sl", default=str(LIVE / "runs_sl.jsonl"))
    b.add_argument("--since-file", default="20261114", help="只看文件名不早于它的录帧（之前的局没有种子）")
    b.add_argument("--out", default=str(MANIFEST))
    r = sub.add_parser("run")
    r.add_argument("--manifest", default=str(MANIFEST))
    r.add_argument("--engine", action="append", default=[], help="名字=Sts2Headless.dll，可多次；默认 base、ref 两个")
    r.add_argument("--group", default="", help="只测这一组（v0.113.1 / v0.113.0）")
    r.add_argument("--run", action="append", default=[], help="只测这几局（run_id）")
    r.add_argument("--limit", type=int, default=0, help="每组只测前几局（试跑用）")
    r.add_argument("--frames", default="")
    r.add_argument("--check-sha", action="store_true", help="逐局核对 frames.bin 的 SHA-1（默认只核对字节数）")
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--nice", type=int, default=10)
    r.add_argument("--out", default=None, help="逐局结果 jsonl")
    r.add_argument("--summary", default=None, help="汇总 JSON")
    r.add_argument("--show", type=int, default=12, help="打印前几个不一致的局")
    r.add_argument("--log-dir", default=str(C.REPO / ".cache" / "clone_tools" / "live_fidelity"))
    r.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    if a.cmd == "run" and not a.engine:
        a.engine = [f"base={C.DLL_BASE}", f"ref={C.DLL_REF113}"]
    {"build": build, "run": run}[a.cmd](a)


if __name__ == "__main__":
    main()
