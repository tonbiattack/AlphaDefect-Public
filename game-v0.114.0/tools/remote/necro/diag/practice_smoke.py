"""亡灵契约师开训前的冒烟（2026-11-17）：随机出手，按采集进程（sts2rl.actor）同样的开局请求打整局与战斗练习的各类题，
查帧的观测格式、卡住、词表外键（头部 14）与引擎日志里的异常。

一、整局：reset mode=run（snapshots、route_summary、privileged、future_content、截断开关，同 actor），随机出手；
    选路之后紧接着开打的战斗按 actor 的规则收进快照池（SnapshotPools.add），记下各幕的普通战、精英、Boss。
二、战斗练习（reset mode=combat、privileged、截断开关），每类各打 --per 场：
    progress0 / progress05 / progress1  按进度出题（CombatCurriculum.sample 的最后一段，合成牌组）
    boss1 / boss2 / boss3               幕末 Boss（_boss，合成幕末规模牌组）
    mix                                 随机构筑（_mix，普通战与精英各半）
    real_monster / real_elite / real_boss  整局真实快照重打当时那场战斗（_real）
    chain3                              第三幕连打两个 Boss（真实快照 + act 2 + encounter + next_encounter；随机出手的整局到不了第三幕，
                                        快照借第一幕的，只为走到连打的代码路径）
    second3                             第三幕第二个 Boss 单打（second + prev_encounter，同上）
随机出手的战斗一场最多 --cap 步（截断开关会先判输，超过就记“超步数”）。

    .venv/bin/python -I tools/remote/necro/diag/practice_smoke.py --dll engine/Sts2Headless/bin/Release/net9.0/Sts2Headless.dll --runs 24 --per 40 --out 目录
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import threading
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "python"))

from sts2env.env import Sts2Env  # noqa: E402
from sts2env.protocol import DecisionKind, format_for, parse  # noqa: E402
from sts2rl.catalog import Catalog  # noqa: E402
from sts2rl.config import TrainConfig  # noqa: E402
from sts2rl.curriculum import CombatCurriculum, CurriculumConfig, SnapshotPools  # noqa: E402

ACT3_BOSSES = ["AEONGLASS_BOSS", "QUEEN_BOSS", "TEST_SUBJECT_BOSS"]
ACT1_BOSSES = ["CEREMONIAL_BEAST_BOSS", "VANTOM_BOSS", "THE_KIN_BOSS", "LAGAVULIN_MATRIARCH_BOSS", "SOUL_FYSH_BOSS", "WATERFALL_GIANT_BOSS"]
ACT2_BOSSES = ["KAISER_CRAB_BOSS", "KNOWLEDGE_DEMON_BOSS", "THE_INSATIABLE_BOSS"]


def play(env: Sts2Env, req: dict, rng: random.Random, cap: int, stats: Counter, fmt: int, pools: SnapshotPools | None, lock):
    env._proc.send(req)
    o = parse(env._proc.recv())
    steps, snap, snap_n, n_obs = 0, None, -1, 0
    while True:
        n_obs += 1
        stats["帧"] += 1
        if int(o.header[0]) != fmt:
            stats[f"帧的格式不是 {fmt}（{int(o.header[0])}）"] += 1
        if int(o.header[14]) > 0:
            stats["头部14（词表外键）>0 的帧"] += 1
        if pools is not None and o.extra:
            if "snapshot" in o.extra:
                snap, snap_n = o.extra["snapshot"], n_obs
            elif "combat" in o.extra and snap and n_obs == snap_n + 1:
                with lock:
                    pools.add(o.extra["combat"], snap, None)
                    stats[f"快照入池 {o.extra['combat']['act']}:{o.extra['combat']['room']}"] += 1
        if o.done:
            return o, steps
        if steps >= cap:
            stats["超步数"] += 1
            return o, steps
        o = env.step(rng.randrange(o.num_actions))
        steps += 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dll", required=True)
    ap.add_argument("--config", default=str(REPO / "tools/remote/necro/train_config.json"))
    ap.add_argument("--runs", type=int, default=24)
    ap.add_argument("--per", type=int, default=40)
    ap.add_argument("--cap", type=int, default=3000)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfg = TrainConfig.from_args(["--config", a.config, "--run-dir", str(out / "run")])
    fmt = format_for(cfg.character)
    cut = cfg.cut_flags()
    cur = CombatCurriculum(Catalog(), CurriculumConfig.for_character(cfg.character), seed=7)
    pools = SnapshotPools(256)
    lock = threading.Lock()
    stats: dict[str, Counter] = defaultdict(Counter)
    results: dict[str, Counter] = defaultdict(Counter)

    def worker(k: int, jobs: list):
        rng = random.Random(1000 + k)
        with open(out / f"engine{k}.log", "w") as log:
            env = Sts2Env(Path(a.dll), log=log)
            try:
                for name, req in jobs:
                    try:
                        o, steps = play(env, req, rng, a.cap, stats[name], fmt, pools if name == "run" else None, lock)
                        with lock:
                            results[name]["场"] += 1
                            results[name]["步"] += steps
                            if name == "run":
                                results[name]["层"] += int(o.floor)
                            if o.done and o.victory:
                                results[name]["赢"] += 1
                            if o.done and o.stuck:
                                results[name]["引擎判卡住"] += 1
                    except Exception as e:      # 引擎报错或通信断：记下来，换一个引擎接着打
                        with lock:
                            results[name]["报错"] += 1
                            (out / "errors.txt").open("a").write(f"{name} {json.dumps(req, ensure_ascii=False)[:400]} {e!r}\n")
                        env.close(); env = Sts2Env(Path(a.dll), log=log)
            finally:
                env.close()

    def run_jobs(jobs: list):
        parts = [jobs[i::a.threads] for i in range(a.threads)]
        ts = [threading.Thread(target=worker, args=(k, p)) for k, p in enumerate(parts) if p]
        for t in ts: t.start()
        for t in ts: t.join()

    # 一、整局
    run_req = {"op": "reset", "mode": "run", "snapshots": True, "route_summary": cfg.route_summary, "privileged": True,
               "character": cfg.character, "ascension": 10, **cut}
    if cfg.model.future_content:
        run_req["future_content"] = True
    run_jobs([("run", {**run_req, "seed": f"SMOKE{i:04d}"}) for i in range(a.runs)])

    # 二、战斗练习
    def creq(spec: dict) -> dict:
        spec = {k: v for k, v in spec.items() if not k.startswith("_")}
        return {"op": "reset", "mode": "combat", "spec": spec, "privileged": True, **cut}

    def progress_spec(p: float) -> dict:
        return cur.sample(p)        # 几个比例都为 0：只走最后一段按进度出题

    jobs = []
    for i in range(a.per):
        jobs += [("progress0", creq(progress_spec(0.0))), ("progress05", creq(progress_spec(0.5))), ("progress1", creq(progress_spec(1.0)))]
        jobs += [(f"boss{act + 1}", creq(cur._boss(act))) for act in range(3)]
        jobs.append(("mix", creq(cur._mix(0.5))))
        if pools.monster[0]:
            jobs.append(("real_monster", creq(cur._real(pools.monster[0], "act1"))))
        if pools.elite[0]:
            jobs.append(("real_elite", creq(cur._real(pools.elite[0], "act1"))))
        if pools.boss[0]:
            jobs.append(("real_boss", creq(cur._real(pools.boss[0], "boss"))))
        any_pool = list(pools.monster[0]) + list(pools.elite[0]) + list(pools.boss[0])
        if any_pool:
            e = any_pool[i % len(any_pool)]
            b1, b2 = ACT3_BOSSES[i % 3], ACT3_BOSSES[(i + 1) % 3]
            seed = f"SMK{i}"
            # 真实快照打第一、二幕 Boss（随机出手的整局很少走到 Boss，快照借普通战的；同 _real 的 probe 写法，指定遭遇）
            for act, bosses in ((0, ACT1_BOSSES), (1, ACT2_BOSSES)):
                jobs.append((f"real_boss_act{act + 1}", creq({"character": cfg.character, "ascension": 10, "seed": seed + f"b{act}", "act": act,
                                                              "encounter": bosses[i % len(bosses)], "room": "Boss", "player_snapshot": e["snap"], "probe": True})))
            jobs.append(("chain3", creq({"character": cfg.character, "ascension": 10, "seed": seed + "c", "act": 2, "encounter": b1, "room": "Boss",
                                         "player_snapshot": e["snap"], "next_encounter": b2})))
            jobs.append(("second3", creq({"character": cfg.character, "ascension": 10, "seed": seed + "s", "act": 2, "encounter": b2, "room": "Boss",
                                          "player_snapshot": e["snap"], "second": True, "prev_encounter": b1})))
    run_jobs(jobs)

    # 引擎日志里的异常行
    exc = Counter()
    for f in out.glob("engine*.log"):
        for line in f.read_text(errors="replace").splitlines():
            if "Exception" in line or "[error" in line or "[stuck" in line:
                exc[line[:160]] += 1
    rep = {"character": cfg.character, "格式": fmt, "整局与练习": {k: dict(v) for k, v in results.items()},
           "帧统计": {k: dict(v) for k, v in stats.items()},
           "快照池": {f"{kind}{act + 1}": len(getattr(pools, kind)[act]) for kind in ("monster", "elite", "boss") for act in range(3)},
           "引擎异常行": sum(exc.values()), "引擎异常行（前 20 种）": exc.most_common(20),
           "报错": (out / "errors.txt").read_text()[:4000] if (out / "errors.txt").exists() else ""}
    (out / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    print(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
