"""两个 C# 引擎逐字节对拍（取代 Rust 对 C# 的 rust-engine/tools/rustdiff/frame_diff.py，只用 C#）。

用途：引擎改动（加局面复制、换游戏版本、重构）前后，同样的请求、同样的随机动作发给两个引擎，逐帧比原始字节。
  - 比较时跳过头部 12（本进程赢下的战斗数）与 14（本进程累计的词表外键数）：两项随进程累计，与局面无关。
    头部 14 两边不同的帧另计一栏（“头部14不同”），不算不同。
  - 一场里第一次出现不同的帧就记下第一处差异（段、第几条、记录类型、第几项、两边的值）并结束这一场
    （之后两边的局面可能已经分开，再比没有意义，同 tools/engine_equal.py）。
  - 两边对同一请求都报错且报错相同，算一致（“同样报错”）；只一边报错或报错不同，算不同。

语料（同 frame_diff.py）：
  --decks starter|random：单场战斗（默认故障机器人，--char 可换），第 i 场种子 RD{i}，遭遇按序号轮转全部遭遇
    （内容目录四个幕条目引用的 80 个遭遇，见 common.encounters）；random 在初始牌组上加 5–15 张随机牌（三成升级）、
    0–4 个遗物、0–2 瓶药水，由 random.Random(f"deck{i}") 抽；动作由 random.Random(i) 在候选里均匀抽。
  --decks run：整局（种子 RUN{i}），reset 带训练采集的开关（privileged、route_summary、future_content、snapshots），
    动作默认偏向多打牌（--policy bias，均匀随机的整局多半死在第一场战斗，走不到战斗外的各种决策）。
口径：AGENTS.md 要求结论用大样本——random 200 场、starter 200 场。

  python -I tools/clone/pair_diff.py --a 旧.dll --b 新.dll --decks random --trials 200
  python -I tools/clone/pair_diff.py --a 旧.dll --b 新.dll --decks run --trials 20 --char FOOL
  python -I tools/clone/pair_diff.py --a 旧.dll --b 新.dll --decks random --trial 17      复现一场，打印第一处差异
默认 --workers 2（每个分片 2 个引擎进程，合计 4 个），引擎进程以 nice 10 运行。退出码：有不同为 1。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from common import EngineError  # noqa: E402


def reset_req(i: int, a) -> dict:
    if a.decks == "run":
        r = {"op": "reset", "mode": "run", "character": a.char, "ascension": 10, "seed": f"RUN{i}"}
        if a.flags != "none":
            r.update(C.RUN_FLAGS)
        return r
    r = {"op": "reset", "mode": "combat", "spec": C.combat_spec(i, a.decks, a.char, a._encs, a._pools)}
    if a.flags == "train":
        r.update(C.COMBAT_FLAGS)
    return r


def run_trial(A: C.Eng, B: C.Eng, i: int, a) -> dict:
    req = reset_req(i, a)
    rng = random.Random(i)
    max_steps = a.max_steps or (3000 if a.decks == "run" else 400)
    res = {"trial": i, "frames": 0, "h14": 0, "steps": 0, "result": "CLEAN"}
    if a.decks != "run":
        res["encounter"] = req["spec"]["encounter"]
    fa, fb = A.req(req), B.req({**req, **a._extra_b})
    step = 0
    kinds = Counter()
    while True:
        ea, eb = isinstance(fa, EngineError), isinstance(fb, EngineError)
        if ea or eb:
            if ea and eb and str(fa) == str(fb):
                res["result"] = "SETUP_ERROR" if step == 0 else "BOTH_ERROR"
                res["error"] = str(fa)[:200]
            else:
                res["result"] = "DIFF"
                res["diff"] = {"seg": "报错", "desc": f"A：{str(fa)[:160] if ea else '无'}；B：{str(fb)[:160] if eb else '无'}"}
                res["diff_step"] = step
            break
        res["frames"] += 1
        kinds[C.frame_brief(fa)["decision"]] += 1
        if C.h14_differs(fa, fb):
            res["h14"] += 1
        d = C.first_diff(fa, fb)
        if d is not None:
            res.update(result="DIFF", diff=d, diff_step=step, at=C.frame_brief(fa))
            break
        if C.is_done(fa) or step >= max_steps:
            res["result"] = "CLEAN" if C.is_done(fa) else "TRUNCATED"
            break
        k = C.pick_bias(rng, fa) if a.policy == "bias" else rng.randrange(C.num_actions(fa))
        fa, fb = A.req({"op": "step", "action": k}), B.req({"op": "step", "action": k})
        step += 1
    res["steps"] = step
    res["kinds"] = dict(kinds)
    return res


def run_shard(job) -> list[dict]:
    ids, a = job
    C.be_nice(a.nice)
    log_a = open(Path(a.log_dir) / f"a_{ids[0] if ids else 0}.log", "ab")
    log_b = open(Path(a.log_dir) / f"b_{ids[0] if ids else 0}.log", "ab")
    A, B = C.Eng(a.a, log_a, "A"), C.Eng(a.b, log_b, "B")
    rows = []
    try:
        for i in ids:
            r = run_trial(A, B, i, a)
            rows.append(r)
            if a.verbose:
                print(json.dumps(r, ensure_ascii=False), flush=True)
    finally:
        A.close(); B.close()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", default=str(C.DLL_REF113), help="引擎 A（基线）")
    ap.add_argument("--b", default=str(C.DLL_CLONE), help="引擎 B（新）")
    ap.add_argument("--decks", choices=["starter", "random", "run"], default="random")
    ap.add_argument("--char", default="DEFECT")
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--trial", type=int, default=None, help="只跑这一场（复现用）")
    ap.add_argument("--max-steps", type=int, default=0, help="每场最多走几步（默认战斗 400、整局 3000）")
    ap.add_argument("--flags", choices=["none", "train"], default="none",
                    help="reset 的开关：none 同 frame_diff（整局仍带训练采集开关）；train：战斗带 loop_cut + stall_turns 20")
    ap.add_argument("--policy", choices=["uniform", "bias"], default="",
                    help="动作：uniform 在候选里均匀抽（同 frame_diff，战斗默认）；bias 偏向多打牌（同 engine_equal.play_bias，整局默认，走得更远）")
    ap.add_argument("--extra-b", default="", help="只并进 B 的 reset 请求的 JSON（如 '{\"obs_format\":26}'：格式门控核对、反向核对用）")
    ap.add_argument("--workers", type=int, default=2, help="分片数（每片 2 个引擎进程；同时运行的引擎进程不超过 4 个）")
    ap.add_argument("--nice", type=int, default=10)
    ap.add_argument("--out", default=None, help="逐场结果写到这个 jsonl")
    ap.add_argument("--log-dir", default=str(C.REPO / ".cache" / "clone_tools" / "pair_diff"))
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    if a.decks == "run" and a.flags == "none":
        a.flags = "train"       # 整局默认带训练采集的开关（engine_equal 的 RUN_FLAGS）
    a.policy = a.policy or ("bias" if a.decks == "run" else "uniform")
    a._extra_b = json.loads(a.extra_b) if a.extra_b else {}
    Path(a.log_dir).mkdir(parents=True, exist_ok=True)
    a._encs = C.encounters()
    a._pools = C.load_pools(a.char)
    ids = [a.trial] if a.trial is not None else list(range(a.start, a.start + a.trials))
    w = max(1, min(a.workers, 2, len(ids)))
    jobs = [(ids[k::w], a) for k in range(w)]
    t0 = time.time()
    if w == 1:
        results = [run_shard(jobs[0])]
    else:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(w) as ex:
            results = list(ex.map(run_shard, jobs))
    rows = sorted((r for rs in results for r in rs), key=lambda r: r["trial"])
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    frames = sum(r["frames"] for r in rows)
    ndiff = sum(1 for r in rows if r["result"] == "DIFF")
    h14 = sum(r["h14"] for r in rows)
    tally = Counter(r["result"] for r in rows)
    cats = Counter(C.diff_key(r["diff"]) if r["diff"].get("seg") != "报错" else "报错" for r in rows if r.get("diff"))
    print(f"A = {C.dll_info(a.a)}\nB = {C.dll_info(a.b)}")
    print(f"语料 {a.decks}（{a.char}，开关 {a.flags}，动作 {a.policy}{'，B 另加 ' + a.extra_b if a.extra_b else ''}），{len(rows)} 场，"
          f"用时 {time.time() - t0:.0f} 秒，{w} 个分片")
    print(f"结果：{dict(tally)}")
    print(f"帧 {frames}，不同 {ndiff}，逐字节一致率 {(frames - ndiff) / max(1, frames):.4%}（跳过头部 12、14）；头部 14 不同的帧 {h14}")
    if cats:
        print("第一处差异的分类：", dict(cats.most_common(20)))
    for r in [r for r in rows if r["result"] == "DIFF"][:12]:
        print(f"  第 {r['trial']} 场第 {r['diff_step']} 步{(' ' + r.get('encounter', '')) if r.get('encounter') else ''}"
              f"{(' ' + json.dumps(r.get('at'), ensure_ascii=False)) if r.get('at') else ''}：{r['diff']['desc']}")
    errs = Counter(r.get("error", "")[:80] for r in rows if r["result"] in ("SETUP_ERROR", "BOTH_ERROR"))
    if errs:
        print("两边同样报错：", dict(errs.most_common(8)))
    sys.exit(1 if ndiff else 0)


if __name__ == "__main__":
    main()
