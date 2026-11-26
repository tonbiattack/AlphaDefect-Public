"""固定种子评测进程：每隔 eval_every_s 秒（0 为上一轮结束就开始下一轮）载入当前权重，用贪心策略打一组固定种子的
A10 Defect 整局，外加一组固定的战斗课程（进度 0.5），结果追加到 run_dir/eval.jsonl。评测不参与训练，只用于看板和比较。
评测进行中把进度写到 run_dir/eval_progress.json，供看板显示。

评测只用训练闲置的资源：推理在 CPU 上做（不碰显卡），本进程及其引擎以 SCHED_IDLE 调度，训练需要 CPU 时评测自动让路。
CPU 推理单线程较慢，所以开 eval_workers 个工作进程并行（线程会被 GIL 卡住），每个进程管 eval_envs 个引擎；
工作进程继承空闲调度，常驻复用，只在权重版本变化时重新载入（从本进程写出的快照载入，保证一轮评测用同一个版本）。
SCHED_IDLE 只保证不抢时间片；这台机器 36 个线程只有 18 个物理核，评测占满超线程仍会拖慢训练，
所以再用 cgroup 把评测（含工作进程和引擎）限制在 eval_cpu_cores 个核以内。

一轮的全部对局（整局在前、战斗在后）放进一个共享队列，哪个引擎空了就取下一局：
不会出现某个进程还在打长局、其余进程干等的情况，整局和战斗之间也没有关卡。

每轮评测的整局平均层数进入前 best_keep 名时，把这一版权重（只有网络参数，没有优化器状态）留到 run_dir/best/，
索引在 best/index.json：训练跑偏时可以从评测最好的版本重新开始，不必只能用跑偏期间的定时备份。"""
from __future__ import annotations

import contextlib
import gzip
import json
import multiprocessing as mp
import os
import queue
import threading
import time
import shutil
import subprocess
import traceback
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor, wait
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from sts2env import EngineError, VecEnv
from sts2env.protocol import ActionKind, DecisionKind, TokenKind

from .greedy import merged_argmax
from .actor import DECK_BINS, MODE_COMBAT, MODE_RUN, ancient_offer, cand_name, card_offer, choice_key, deck_bin, explore_choice, is_macro, load_weights, relic_offer, relic_skip_detail, skipped_free
from .catalog import Catalog
from .curriculum import CombatCurriculum, CurriculumConfig
from .replay import describe
from .features import Ragged, collate_ragged
from .model import PolicyValueNet


def run_evaluator(cfg, weights_path: str, run_dir: str, stop_event) -> None:
    # 报错文件只代表当前这次运行：启动时删掉上次留下的，看板据此判断评测是否正在出错（2026-10-26 评测停摆一小时没人发现后加）
    try:
        (Path(run_dir) / "eval_error.txt").unlink()
    except OSError:
        pass
    try:
        _loop(cfg, weights_path, Path(run_dir), stop_event)
    except Exception:
        (Path(run_dir) / "eval_error.txt").write_text(traceback.format_exc(), encoding="utf-8")


def _cpu_cap(cores: float) -> str:
    """把本进程（及之后启动的子进程）放进限额 cgroup：最多用 cores 个核。成功返回空串，否则返回原因。"""
    root = Path("/sys/fs/cgroup")
    try:
        ctl = root / "cgroup.subtree_control"
        if "cpu" not in ctl.read_text().split():
            ctl.write_text("+cpu")
        g = root / "sls2-eval"
        g.mkdir(exist_ok=True)
        (g / "cpu.max").write_text(f"{int(cores * 100_000)} 100000")
        (g / "cgroup.procs").write_text(str(os.getpid()))
        return ""
    except OSError as e:
        return str(e)


def _idle_priority(nice: int | None = None) -> None:
    """本进程（及之后启动的引擎子进程）只在 CPU 空闲时运行；给了 nice 时改为普通调度、降这么多优先级
    （评测放在协同采集机上时用，那台 CPU 一直被采集占满，空闲调度会排不上，决策记录 198）。"""
    if nice is not None:
        os.nice(nice)
        return
    try:
        os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    except (AttributeError, OSError):
        os.nice(19)


class _Alive:
    """对局循环的停止条件：收到停止信号，或父进程已经不在。"""
    def __init__(self, stop_event, parent: int):
        self.stop_event, self.parent = stop_event, parent

    def is_set(self) -> bool:
        return self.stop_event.is_set() or os.getppid() != self.parent


# ---------------------------------------------------------------- 工作进程

_W: dict = {}


def _watch_parent(parent: int) -> None:
    """评测进程不在了就立即退出（否则空闲的工作进程会一直等任务，成为孤儿）。"""
    while True:
        time.sleep(2)
        if os.getppid() != parent:
            os._exit(0)


def _worker_init(model_cfg, tasks, counters, stop_event, caps, loop_log, device: str = "cpu", all_dir: str | None = None, merge: bool = False) -> None:
    torch.set_num_threads(1)
    threading.Thread(target=_watch_parent, args=(os.getppid(),), daemon=True).start()
    dev = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        # 评测用显卡推理时关掉 TF32，与 CPU 推理的结果尽量一致（贪心取最大，接近平手的决策才可能翻转）
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    _W.update(net=PolicyValueNet(model_cfg).eval().to(dev), device=dev, version=None, tasks=tasks, counters=counters,
              stop=_Alive(stop_event, os.getppid()), caps=caps, loop_log=loop_log, all_dir=all_dir, merge=merge)


def _worker_drain(snapshot: str, version: int, envs: int) -> list[dict]:
    """从共享队列取对局打，直到队列取空。"""
    if _W["version"] != version:
        _W["net"].load_state_dict(torch.load(snapshot, map_location="cpu"))
        _W["version"] = version

    def next_task():
        try:
            return _W["tasks"].get(timeout=1.0)
        except queue.Empty:
            return None

    def done_one(kind: str) -> None:
        c = _W["counters"][kind]
        with c.get_lock():
            c.value += 1
    return play(_W["net"], _W["device"], next_task, envs, _W["stop"], done_one, _W["caps"], _W["loop_log"], version, all_dir=_W["all_dir"],
                policy={"merge": True} if _W.get("merge") else None, time_cap=_W["caps"].get("_time", 600.0))


# ---------------------------------------------------------------- 评测主循环

def _loop(cfg, weights_path: str, run_dir: Path, stop_event) -> None:
    _idle_priority(cfg.eval_nice)
    why = _cpu_cap(cfg.eval_cpu_cores)
    if why:
        print(f"评测：无法设置 CPU 上限（{why}），只靠空闲调度", flush=True)
    torch.set_num_threads(1)
    alive = _Alive(stop_event, os.getppid())
    net = PolicyValueNet(cfg.model).eval()
    snapshot = str((Path("/dev/shm") if Path("/dev/shm").exists() else run_dir) / f"sts2rl_eval_{os.getpid()}.pt")
    ctx = mp.get_context("spawn")
    tasks = ctx.Queue()
    counters = {"run": ctx.Value("i", 0), "combat": ctx.Value("i", 0), "exam": ctx.Value("i", 0)}
    caps = {"run": cfg.max_steps_run, "combat": cfg.max_steps_combat, "exam": cfg.max_steps_combat,   # 考卷也是单场战斗
            "_time": cfg.eval_time_cap,     # 单局时长上限（play 的 time_cap）
            "fight": cfg.max_steps_fight}   # 整局里单场战斗的步数上限（None 不限；亡灵契约师决策记录第 10 条）
    pool = ProcessPoolExecutor(cfg.eval_workers, mp_context=ctx, initializer=_worker_init,
                               initargs=(cfg.model, tasks, counters, stop_event, caps, str(run_dir / "eval_loops.txt"), cfg.eval_device,
                                         str(run_dir / "eval_replays") if cfg.eval_replay_all else None, cfg.eval_merge_equiv))
    try:
        _eval_loop(cfg, weights_path, run_dir, alive, net, snapshot, pool, tasks, counters)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        try:
            os.unlink(snapshot)
        except OSError:
            pass


def _eval_loop(cfg, weights_path, run_dir, alive, net, snapshot, pool, tasks, counters) -> None:
    version = -1
    last = 0.0
    cur = CombatCurriculum(Catalog(), CurriculumConfig.for_character(cfg.character), seed=4242)        # 只建一次：每次新建会让 128 场都是同一场
    combat_specs = [cur.sample(0.5) for _ in range(cfg.eval_combats)]
    eval_seeds = list(cfg.eval_seeds) if cfg.eval_seeds else [f"EVAL{k:03d}" for k in range(cfg.eval_runs)]
    done_versions = {r.get("version") for r in _read_jsonl(run_dir / "eval.jsonl")}   # 续训后不重复评测同一版本
    # 固定考卷（tools/exam_build.py 生成 exam.json，2026-10-26 起）：每 exam_every 版用贪心把冻结的 Boss 局面各打一遍，结果写 exam.jsonl
    exam = (_read_json(run_dir / "exam.json") or {}).get("items") or []
    last_exam = max((r.get("version", -10**9) for r in _read_jsonl(run_dir / "exam.jsonl")), default=-10**9)
    while not alive.is_set():
        if time.time() - last < cfg.eval_every_s:
            time.sleep(2)
            continue
        # 运行目录里有 PAUSE 时不开新一轮（千种子复测在同一台机器上跑时由 retest.sh 放上、测完删掉，两者不互相拖慢，决策记录 199）
        if (run_dir / "PAUSE").exists():
            time.sleep(5)
            continue
        v, _, _ = load_weights(net, weights_path, version, use_ema=cfg.eval_use_ema)
        if v < 0 or v in done_versions:     # 还没有权重，或在等新版本
            time.sleep(2)
            continue
        version = v
        done_versions.add(v)
        torch.save(net.state_dict(), snapshot)       # 工作进程从快照载入，一轮评测里所有对局用同一个版本
        last = time.time()
        status = _read_json(run_dir / "status.json")
        prog = Progress(run_dir / "eval_progress.json", {"version": v, "start": last, "runs": cfg.eval_runs, "combats": len(combat_specs)})
        for c in counters.values():
            with c.get_lock():
                c.value = 0
        # 金币由策略自己领（决策记录 160）：过渡期内评测照旧自动领，曲线前后可比；过渡结束（版本 ≥ start + span）后改为由策略领
        manual_gold = cfg.manual_gold_start is not None and v >= cfg.manual_gold_start + cfg.manual_gold_span
        lc = cfg.cut_flags(evaluating=True)     # 局面重复截断、跨回合无进展判输（判输线用 eval_stall_turns）
        for k, seed in enumerate(eval_seeds):
            tasks.put(({"mode": "run", "character": cfg.character, "ascension": 10, "seed": seed, "snapshots": True, "tag": k,
                       "route_summary": cfg.route_summary, **({"auto_gold": False} if manual_gold else {}), **lc}, MODE_RUN, "run"))
        for s in combat_specs:
            tasks.put(({"mode": "combat", "spec": s, **lc}, MODE_COMBAT, "combat"))
        do_exam = bool(exam) and cfg.exam_every > 0 and v - last_exam >= cfg.exam_every
        if do_exam:
            last_exam = v
            for k, it in enumerate(exam):
                tasks.put(({"mode": "combat", "spec": it["spec"], "tag": k}, MODE_COMBAT, "exam"))
        futs = [pool.submit(_worker_drain, snapshot, v, cfg.eval_envs) for _ in range(cfg.eval_workers)]
        while True:
            _, pending = wait(futs, timeout=1.0)
            prog.update(counters["run"].value, counters["combat"].value)
            if not pending or alive.is_set():
                break
        if alive.is_set():
            break
        results = [r for f in futs for r in f.result()]
        if do_exam:   # 考卷结果：按题目顺序的 0/1 串，便于不同版本按题配对
            res = ["-"] * len(exam)
            for r in results:
                if r["kind"] == "exam" and r.get("tag") is not None:
                    res[r["tag"]] = "1" if r["win"] else ("x" if r["stuck"] else "0")
            agg = {f"act{a + 1}": [sum(1 for k, it in enumerate(exam) if it["act"] == a and res[k] == "1"),
                                   sum(1 for k, it in enumerate(exam) if it["act"] == a and res[k] in "01")] for a in (1, 2)}
            with open(run_dir / "exam.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"time": time.time(), "version": v, **agg, "res": "".join(res)}) + "\n")
        # 标定记录（2026-10-26）：整局里每场 Boss 战、精英战开打时的状态与输赢，一行一场写到 calib.jsonl（看板不读，离线拟合各资源的边际价值）
        try:
            with open(run_dir / "calib.jsonl", "a", encoding="utf-8") if cfg.calib_log else contextlib.nullcontext() as f:
                for r in (x for x in results if x["kind"] == "run" and cfg.calib_log):
                    cs = r.get("combats") or []
                    for j, c in enumerate(cs):
                        if c.get("room") not in ("Boss", "Elite"):
                            continue
                        lost = j == len(cs) - 1 and c["floor"] == r["floor"] and not r["win"]
                        if lost and r["stuck"]:
                            continue
                        f.write(json.dumps({"v": v, "act": c["act"], "room": c["room"], "enc": c["encounter"], "floor": c["floor"],
                                            "hp": c["hp"], "max_hp": c["max_hp"], "deck": c["deck"], "relics": c["relics"],
                                            "potions": c.get("potions"), "upgrades": c.get("upgrades"), "win": int(not lost)},
                                           ensure_ascii=False, separators=(",", ":")) + "\n")
        except Exception as e:
            print(f"标定记录写入失败：{e}", flush=True)
        runs = [r for r in results if r["kind"] == "run"]
        combats = [r for r in results if r["kind"] == "combat"]
        try:        # 主动跳过遗物的现场，每次一行（2026-10-28 起，决策记录 159）
            with open(run_dir / "relic_skip.jsonl", "a", encoding="utf-8") as f:
                for r in runs:
                    for d in r.get("relic_skips") or []:
                        f.write(json.dumps({"v": version, **d}, ensure_ascii=False, separators=(",", ":")) + "\n")
        except Exception as e:
            print(f"跳过遗物记录写入失败：{e}", flush=True)
        prog.update(len(runs), len(combats), force=True)
        floors = [r["floor"] for r in runs]
        by_seed = {r["tag"]: r for r in runs if isinstance(r.get("tag"), int)}
        f192 = [by_seed[k] for k in range(192) if k in by_seed]
        rec = {
            "time": time.time(), "version": version, "total_steps": status.get("total_steps"), "seconds": round(time.time() - last, 1),
            "runs": len(runs), "win_rate": float(np.mean([r["win"] for r in runs])) if runs else 0.0, **({"weights": "ema"} if cfg.eval_use_ema else {}),
            "floor_mean": float(np.mean(floors)) if floors else 0.0, "floor_median": float(np.median(floors)) if floors else 0.0,
            "floor_max": int(max(floors)) if floors else 0, "floor_p90": float(np.percentile(floors, 90)) if floors else 0.0,
            "reach_act2": float(np.mean([r["act"] >= 1 or r["win"] for r in runs])) if runs else 0.0,
            "reach_act3": float(np.mean([r["act"] >= 2 or r["win"] for r in runs])) if runs else 0.0,
            "floors": floors,
            "pos": [run_pos(r) for r in runs],
            # 逐种子的输赢（第 k 个字符为种子表里第 k 个种子这一局：1 赢、0 输、- 没打完），不同版本可以按种子配对（决策记录 198）
            "res": "".join("1" if by_seed[k]["win"] else "0" if k in by_seed else "-" for k in range(len(eval_seeds))) if by_seed else None,
            "seed_set": _seed_set(cfg, eval_seeds),
            # 2026-10-29 局数 192 → 384：多出来的 EVAL192–383 以前没测过、更难，前 192 个种子另记一份，与以前的评测前后可比
            **({"first192": {"runs": len(f192), "win_rate": float(np.mean([r["win"] for r in f192])), "floor_mean": float(np.mean([r["floor"] for r in f192])),
                             "reach_act2": float(np.mean([r["act"] >= 1 or r["win"] for r in f192])),
                             "reach_act3": float(np.mean([r["act"] >= 2 or r["win"] for r in f192]))}} if len(by_seed) > 192 and f192 and not cfg.eval_seeds else {}),
            "combat_win_rate": float(np.mean([c["win"] for c in combats])) if combats else None,
            "combat_hp_delta": float(np.mean([c["hp_delta"] for c in combats])) if combats else None,
            "combat_eval": 2 if combats else 0,     # 2 起为 128 场不同的战斗（之前误为同一场）；0 为这一轮没打（eval_combats = 0）
            "stuck": int(sum(r["stuck"] for r in runs + combats)),
            **_run_diagnostics(runs),
            # 局外行为：各类局外选择在全部整局里的次数合计（actor.choice_key），看板按局数折算
            "choices": dict(sum((Counter(r["choices"]) for r in runs), Counter())),
            # 选牌奖励：每张牌在第 1/2/3 幕被提供、被选中的次数（card_counts），与社区数据的选取率对照
            "cards": merge_cards(r.get("cards") for r in runs),
            # 古神遗物：每件遗物在第 1/2/3 幕被古神提供、被选中的次数（格式同 cards；"_SCREEN" 为各幕界面数与没选遗物的次数）
            "ancient": merge_cards(r.get("ancient") for r in runs),
            # 奖励界面与宝箱的遗物（跳过遗物的探针，2026-10-28 起）：每件遗物在第 1/2/3 幕出现、被主动跳过的次数；"_REWARDS"、"_TREASURE" 为两种界面的合计
            "relic": merge_cards(r.get("relic") for r in runs),
            # 路线（只在打开路线摘要时有）：各幕走到 Boss 的局里，实际走的精英、休息处、前面有火堆的精英，与这一幕开头能做到的最多值
            "route": merge_route(r.get("route") for r in runs),
        }
        prev = _recent_rates(run_dir / "eval.jsonl", rec["runs"], cfg.eval_high_window, rec["seed_set"])
        with open(run_dir / "eval.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        _archive_best(run_dir, snapshot, rec, cfg.best_keep)
        try:
            _archive_high(run_dir, snapshot, rec, prev, cfg.eval_high_margin, cfg.eval_high_keep)
        except Exception as e:
            print(f"偏高评测的权重留档失败：{e}", flush=True)
        if cfg.eval_keep_runs > 0 and cfg.eval_seeds:
            try:
                _keep_runs(run_dir / "eval_runs", snapshot, version, eval_seeds, runs, cfg.eval_keep_runs)
            except Exception as e:
                print(f"逐局结果留存失败：{e}", flush=True)


def _keep_runs(d: Path, snapshot: str, version: int, seeds: list, runs: list[dict], keep: int) -> None:
    """一轮评测的逐局结果按 tools/ability_test.py 的格式写成 d/v<版本>.jsonl，权重另存 d/v<版本>.pt（决策记录 203）。
    评测与千种子复测用同一个 play、同一批 RT 种子，同一份权重逐局完全相同（决策记录 199 补记），复测这一版本时把这个文件放进结果目录，
    ability_test 跳过已有的种子，只打其余的。逐局结果留最近 keep 轮，权重留最近 2 轮（每份约 72 MB）。先写临时文件再改名。"""
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f"v{version}.jsonl.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in sorted((r for r in runs if isinstance(r.get("tag"), int)), key=lambda r: r["tag"]):
            row = {k: v for k, v in r.items() if k not in ("kind", "tag")}
            f.write(json.dumps({"asc": 10, "seed": seeds[r["tag"]], "version": version, "pos": run_pos(r), **row}, ensure_ascii=False) + "\n")
    tmp.rename(d / f"v{version}.jsonl")
    shutil.copyfile(snapshot, d / f"v{version}.pt.tmp")
    (d / f"v{version}.pt.tmp").rename(d / f"v{version}.pt")
    for pat, n in (("v*.jsonl", keep), ("v*.pt", 2)):
        for old in sorted(d.glob(pat), key=lambda x: x.stat().st_mtime)[:-n]:
            old.unlink(missing_ok=True)


def _seed_set(cfg, eval_seeds: list) -> str:
    """这一轮评测的种子口径：EVAL000…、384 个分层 RT 种子（决策记录 198），或种子池（remote_eval.py 按池名设 eval_seed_set，决策记录 240）。"""
    if not cfg.eval_seeds:
        return "EVAL"
    return getattr(cfg, "eval_seed_set", None) or f"{len(eval_seeds)} 个分层 RT 种子"


def _recent_rates(path: Path, runs: int, window: int, seed_set: str | None = None) -> list[float]:
    """eval.jsonl 里最近 window 轮、局数与种子口径都与这一轮相同的通关率（写入这一轮之前读）。"""
    try:
        with open(path, "rb") as f:
            f.seek(0, 2); f.seek(max(0, f.tell() - 3_000_000))
            lines = f.read().decode("utf-8", errors="ignore").splitlines()[1:]
    except OSError:
        return []
    out = []
    for l in reversed(lines):
        try:
            r = json.loads(l)
        except ValueError:
            continue
        if r.get("runs") == runs and r.get("win_rate") is not None and (seed_set is None or r.get("seed_set") == seed_set):
            out.append(r["win_rate"])
            if len(out) >= window:
                break
    return out


def _archive_high(run_dir: Path, snapshot: str, rec: dict, prev: list[float], margin: float, keep: int) -> None:
    """评测偏高的权重留档到 high/（决策记录 208）：通关率不低于之前若干轮平均 + margin 就留，按时间留最近 keep 份，
    index.json 记版本、通关率、当时的平均。best/ 只留历来前几名，一阵偏高但不进前几名的会被挤掉，这里另外留。"""
    if keep <= 0 or len(prev) < 5 or rec.get("win_rate") is None:
        return
    base = sum(prev) / len(prev)
    if rec["win_rate"] < base + margin:
        return
    d = run_dir / "high"; d.mkdir(exist_ok=True)
    name = f"v{rec['version']}.pt"
    shutil.copyfile(snapshot, d / (name + ".part")); os.replace(d / (name + ".part"), d / name)
    idx = [x for x in _read_json(d / "index.json").get("high", []) if x["file"] != name]
    idx.append({"version": rec["version"], "file": name, "win_rate": round(rec["win_rate"], 4), "base": round(base, 4),
                "window": len(prev), "floor_mean": round(rec["floor_mean"], 3), "time": round(rec["time"], 1)})
    idx.sort(key=lambda x: x["time"])
    for x in idx[:-keep]:
        try:
            os.unlink(d / x["file"])
        except OSError:
            pass
    (d / "index.json").write_text(json.dumps({"high": idx[-keep:]}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"评测偏高：第 {rec['version']} 版 {rec['win_rate']:.1%}（之前 {len(prev)} 轮平均 {base:.1%}），权重留档到 high/", flush=True)


def _archive_best(run_dir: Path, snapshot: str, rec: dict, keep: int) -> None:
    """评测成绩进前 keep 名的权重留档到 best/。2026-10-29 起按通关率排、平均层数只在通关率相同时比（决策记录 196）：
    平均层数几乎一样的两版，千种子复测的通关率可以差 10 个百分点（第 66,040 与 82,160 版）。没有通关率的旧条目排在最后。"""
    if keep <= 0 or not rec["runs"]:
        return
    rank = lambda x: (x.get("win_rate", -1.0), x["floor_mean"])
    d = run_dir / "best"; d.mkdir(exist_ok=True)
    # 换种子口径后通关率不可比（决策记录 240）：只与同口径的条目比、在同口径里留前 keep 名，旧口径的条目不动（没记口径的算分层 RT）
    ss = rec.get("seed_set")
    same = lambda x: x.get("seed_set", "384 个分层 RT 种子") == ss
    all_idx = _read_json(d / "index.json").get("best", [])
    other, idx = [x for x in all_idx if not same(x)], [x for x in all_idx if same(x)]
    if len(idx) >= keep and rank(rec) <= min(map(rank, idx)):
        return
    name = f"v{rec['version']}.pt"
    shutil.copyfile(snapshot, d / (name + ".part")); os.replace(d / (name + ".part"), d / name)
    idx = [x for x in idx if x["file"] != name]     # 回退后同一版本号再进前几名：文件已覆盖，旧条目去掉
    idx.append({"version": rec["version"], "file": name, "win_rate": round(rec["win_rate"], 4), "floor_mean": round(rec["floor_mean"], 3),
                "reach_act2": round(rec["reach_act2"], 3), "time": round(rec["time"], 1), **({"seed_set": ss} if ss else {})})
    idx.sort(key=rank, reverse=True)
    for x in idx[keep:]:
        if any(o["file"] == x["file"] for o in other):
            continue
        try:
            os.unlink(d / x["file"])
        except OSError:
            pass
    other = [o for o in other if o["file"] not in {x["file"] for x in idx[:keep]}]
    (d / "index.json").write_text(json.dumps({"best": idx[:keep] + other}, ensure_ascii=False, indent=1), encoding="utf-8")


# 各幕常规层数（含先古和 Boss）：第一幕 17、第二幕 16；地图加长时实际层数更多，所以幕的起点按上一幕 Boss 所在层算
ACT_FLOORS = (17, 16)


def run_pos(r: dict) -> list[int]:
    """一局停在哪：[幕, 本幕第几层（先古为第 1 层）, 第几个 Boss（不在 Boss 战为 0；第三幕连打两个为 1、2）]。
    按幕分开算层数，第二幕地图加长多出的两层算在第二幕里，看板上 Boss 总在各幕末尾。"""
    cs = r.get("combats") or []
    act, f = int(r["act"]), int(r["floor"])
    start = 1
    if act > 0:
        prev = [c["floor"] for c in cs if c["act"] == act - 1 and c["room"] == "Boss"]
        start = max(prev) + 1 if prev else 1 + sum(ACT_FLOORS[:act])
    last = cs[-1] if cs else None
    boss = 0
    if last and last["room"] == "Boss" and last["act"] == act and last["floor"] == f:
        boss = sum(1 for c in cs if c["act"] == act and c["room"] == "Boss")
    return [act, f - start + 1, boss]


def _run_diagnostics(runs: list[dict]) -> dict:
    """死因与牌组规模（引擎的战斗开始信息，见 Snapshot.Extra）。
    deaths：输掉的局死在哪场战斗（"幕:房间类型:遭遇"，幕从 0 起，按局数从多到少；死在战斗之外记“非战斗”）；
    deck_at：几个节点上的 [局数, 平均牌数, 平均遗物数, 平均开场血量比例]——第一幕 Boss、第二幕第一场、第二幕 Boss，
    以及第三幕第一场、第三幕第一个 Boss、第二个 Boss（2026-10-26 起）。
    deck_bins：同样几个节点按牌组张数分档（actor.DECK_BINS）的分布；两个 Boss 记 [打过的局数, 局数]，第二幕第一场只记局数。
    bosses：按 Boss 的 [打赢场数, 场数]（"幕:遭遇"，第三幕连打的两个 Boss 各记一场）；卡住的局里最后那场不计。
    bosses3：第三幕的两个 Boss 按第几个分开记（"1:遭遇"、"2:遭遇"，同样是 [打赢场数, 场数]），场数减打赢就是死在这一场的局数
    （2026-11-07，用户：“能在看板上加上死因吗？第三幕BOSS分第一个和第二个显示。”deaths 的键不分第几个，原样保留）。"""
    deaths = Counter()
    bosses: dict[str, list[int]] = {}
    bosses3: dict[str, list[int]] = {}
    at = {"boss1": [], "act2_first": [], "boss2": [], "act3_first": [], "boss3": [], "boss3b": []}
    nb = len(DECK_BINS) + 1
    bins = {"boss1": [[0, 0] for _ in range(nb)], "act2_first": [0] * nb, "boss2": [[0, 0] for _ in range(nb)]}
    for r in runs:
        cs = r.get("combats") or []
        if not r["win"] and not r["stuck"]:
            last = cs[-1] if cs else None
            deaths[f"{last['act']}:{last['room']}:{last['encounter']}" if last and last["floor"] == r["floor"] else "非战斗"] += 1
        k3 = 0      # 这一局打到第三幕的第几个 Boss
        for j, c in enumerate(cs):
            if c["room"] != "Boss":
                continue
            k3 += c["act"] == 2
            final = j == len(cs) - 1 and c["floor"] == r["floor"] and not r["win"]
            if final and r["stuck"]:
                continue
            b = bosses.setdefault(f"{c['act']}:{c['encounter']}", [0, 0])
            b[0] += int(not final); b[1] += 1
            if c["act"] == 2:
                b = bosses3.setdefault(f"{k3}:{c['encounter']}", [0, 0])
                b[0] += int(not final); b[1] += 1
        first2 = next((c for c in cs if c["act"] == 1), None)
        for key, c in (("boss1", next((c for c in cs if c["room"] == "Boss" and c["act"] == 0), None)), ("act2_first", first2),
                       ("boss2", next((c for c in cs if c["room"] == "Boss" and c["act"] == 1), None))):
            if c:
                at[key].append((c["deck"], c["relics"], c["hp"] / max(1, c["max_hp"])))
        b3 = [c for c in cs if c["room"] == "Boss" and c["act"] == 2]
        for key, c in (("act3_first", next((c for c in cs if c["act"] == 2), None)), ("boss3", b3[0] if b3 else None),
                       ("boss3b", b3[1] if len(b3) > 1 else None)):
            if c:
                at[key].append((c["deck"], c["relics"], c["hp"] / max(1, c["max_hp"])))
        for key, c in (("boss1", next((c for c in cs if c["room"] == "Boss" and c["act"] == 0), None)), ("act2_first", first2),
                       ("boss2", next((c for c in cs if c["room"] == "Boss" and c["act"] == 1), None))):
            if c:
                k = deck_bin(c["deck"])
                if key == "act2_first":
                    bins[key][k] += 1
                else:
                    bins[key][k][0] += int(r["act"] >= (1 if key == "boss1" else 2) or r["win"]); bins[key][k][1] += 1
    return {"deaths": dict(deaths.most_common()), "deck_bins": bins, "bosses": dict(sorted(bosses.items())), "bosses3": dict(sorted(bosses3.items())),
            "deck_at": {k: [len(v), *[round(float(np.mean(x)), 2) for x in zip(*v)]] if v else [0] for k, v in at.items()}}


class Progress:
    """评测进度（整局、战斗各完成多少），至多每 2 秒写一次文件。"""
    def __init__(self, path: Path, base: dict):
        self.path, self.base, self.t = path, base, 0.0
        self.update(0, 0, force=True)

    def update(self, runs_done: int, combats_done: int, force: bool = False) -> None:
        if not force and time.time() - self.t < 2:
            return
        self.t = time.time()
        rec = dict(self.base, done_runs=runs_done, done_combats=combats_done, done=runs_done + combats_done,
                   total=self.base["runs"] + self.base["combats"], time=self.t)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec), encoding="utf-8")
        os.replace(tmp, self.path)


def _read_jsonl(p: Path) -> list[dict]:
    try:
        return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    except Exception:
        return []


def _read_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


# ---------------------------------------------------------------- 对局循环

def save_replay(rec_dir: str, version, seed, result: dict, steps: list[str]) -> None:
    """一局回放存成 gzip 压缩的 JSON Lines：第一行是这局的概要，之后每行一步（replay.describe）。先写临时文件再改名。"""
    try:
        os.makedirs(rec_dir, exist_ok=True)
        head = {"version": version, "seed": seed, "time": time.time(), "win": result["win"], "floor": result["floor"], "act": result["act"],
                "combats": result["combats"], "steps": len(steps)}
        name = f"v{version}_{seed}"
        tmp = os.path.join(rec_dir, f".{name}.tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as f:
            f.write(json.dumps(head, ensure_ascii=False) + "\n")
            for s in steps:
                f.write(s + "\n")
        os.replace(tmp, os.path.join(rec_dir, f"{name}.jsonl.gz"))
    except Exception as e:      # 回放存不下不影响评测
        print(f"回放保存失败：{e}", flush=True)



def card_counts(offers: dict) -> dict[str, list[int]]:
    """一局的选牌记录 → {牌: [第1/2/3幕提供次数, 第1/2/3幕选中次数]}；"_SCREEN" 记各幕选牌界面数与跳过数。
    offers 以 (层数, 提供的牌) 为键，同一次奖励跳过后又回来再选，只按最后一次算。"""
    out: dict[str, list[int]] = {}
    for (_, names), (act, picked) in offers.items():
        if not names:
            continue
        act = min(max(act, 0), 2)
        scr = out.setdefault("_SCREEN", [0] * 6)
        scr[act] += 1
        scr[3 + act] += picked is None
        for n in names:
            out.setdefault(n, [0] * 6)[act] += 1
        if picked is not None:
            out[picked][3 + act] += 1
    return out


def relic_counts(relics: dict) -> dict[str, list[int]]:
    """一局的遗物记录 → {遗物: [第1/2/3幕出现次数, 第1/2/3幕主动跳过次数]}；"_REWARDS"、"_TREASURE" 记两种界面的合计。
    同一界面同一件遗物（奖励界面领完卡牌回来还在）只算一次；拿了就不算跳过。"""
    out: dict[str, list[int]] = {}
    for (_, dec, name), (act, st) in relics.items():
        act = min(max(act, 0), 2)
        for key in (name, "_" + dec):
            v = out.setdefault(key, [0] * 6)
            v[act] += 1
            v[3 + act] += st == 2
    return out


def merge_cards(items) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for c in items:
        for n, v in (c or {}).items():
            acc = out.setdefault(n, [0] * len(v))
            for j, x in enumerate(v):
                acc[j] += x
    return out


class RouteTrack:
    """一局里按幕记录路线：每幕第一次地图决策时，从各个下一步的路线摘要取这一幕能做到的最多值；之后累计实际走过的节点。
    只统计走到了 Boss 的幕（中途死掉的幕走的节点偏少，不可比）。"""

    def __init__(self):
        self.done: list[list[int]] = []
        self.cur = None

    def step(self, o, a: int, vocab: list[str]) -> None:
        rs = np.flatnonzero(o.tok_ints[:, 0] == TokenKind.ROUTE_SUMMARY)
        if not len(rs):
            return
        if self.cur is None or self.cur["act"] != o.act:
            self.cur = None
            if (o.tok_feats[np.flatnonzero(o.tok_ints[:, 0] == TokenKind.MAP_NODE), 2] > 0).sum() > 1:
                return      # 不是这一幕的第一次地图决策（已走过起点以外的节点，如中途开局），这一幕不记
            f = o.tok_feats[rs]
            self.cur = dict(act=int(o.act), e=0, r=0, cov=0, fire=0, best=[int(f[:, 3].max()), int(f[:, 6].max()), int(f[:, 10].max())],
                            tab=[int(x) for x in f[:, 12:16].max(0)])
        if int(o.cand_ints[a][0]) != ActionKind.MAP_MOVE:
            return
        t = cand_name(o, a, vocab).removeprefix("MAPNODE:")
        c = self.cur
        if t == "Elite":
            c["e"] += 1; c["cov"] += c["fire"]; c["fire"] = 0
        elif t == "RestSite":
            c["r"] += 1; c["fire"] = 1
        elif t == "Boss":
            self.done.append([c["act"], c["e"], c["r"], c["cov"], *c["best"], c["tab"][min(c["e"], 3)]])
            self.cur = None

    def result(self) -> list[list[int]]:
        return self.done


def merge_route(items) -> dict[str, list[int]]:
    """各局的路线记录 → {幕: [幕数, 精英, 休息处, 有火堆的精英, 最多精英, 最多休息处, 最多有火堆的精英, 同样精英数下最多休息处] 的合计}。"""
    out: dict[str, list[int]] = {}
    for rows in items:
        for act, *v in rows or []:
            acc = out.setdefault(str(act), [0] * (len(v) + 1))
            acc[0] += 1
            for j, x in enumerate(v):
                acc[j + 1] += x
    return out


def play(net, device, next_task, envs: int, stop_event, on_done=None, caps: dict | None = None, loop_log: str | None = None,
         version: int | None = None, replay_all: bool = False, on_result=None, all_dir: str | None = None, policy: dict | None = None,
         time_cap: float | None = 600.0) -> list[dict]:
    """用贪心策略打对局：next_task() 返回 (请求, 对局模式, 类别) 或 None（没有了）。
    引擎空出来就取下一局；返回每局的结果（含类别 kind），on_done(类别) 在每局结束时调用。
    caps：各类别的单局步数上限，超过按卡住结束（贪心策略可能来回打转）；最后几十步的决策类型记到 loop_log。
    time_cap：单局时长上限（秒），超过按卡住结束；引擎在途超过上限再加 60 秒还没回话就结束这个引擎进程（换新进程接着打下一局）。
      10-31 第 122,770 版的评测里一局在第三幕战斗中反复出牌、选牌，每步越来越慢，31 分钟走了 4,699 步还没到步数上限，整轮评测等它一个（决策记录 212）。
    整局逐步记下回放（replay.describe，每步一行 JSON 文本），通关的局存到 run_dir/replays/（run_dir 取 loop_log 所在目录）。
    on_result(结果)：每局结束时立即调用（能力测试用它逐局写结果，tools/ability_test.py）。
    all_dir：给定时每局整局（不论输赢）另存一份回放到这里（评测的 eval_replay_all，给胜率模型攒数据，决策记录 175）。
    policy：不用贪心时的出手方式（采样拆分评测，决策记录 197）：{"temp": 温度（>0 时按 softmax(logits / 温度) 采样）,
    "explore": 强制探索的 explore_probs（整局的局外决策照训练的办法改选）, "seed": 随机数种子,
    "boss3_snap": 为真时整局打到第三幕第一个 Boss 时记下开打前的快照与两个 Boss（结果的 boss3 项，双 Boss 诊断卷取题用，tools/boss3_exam.py），
    "boss_snaps": 为真时另记第一、二幕 Boss 开打前的快照（结果的 boss1、boss2 项，取法同 boss3；树搜索在前两幕 Boss 上评测取题用），
    "mode_as": 网络输入的对局模式一律换成这个（引擎照常按原模式打；模式一致性对照，决策记录 204），
    "mode_cmp": 每步另用这个模式再算一遍，累计到 policy["_cmp"]：[步数, 两种模式贪心动作相同的步数, KL(原模式 ‖ 对照模式) 之和],
    "merge": 为真时贪心按等价动作合并（sts2rl.greedy.merged_argmax，决策记录 205），不按单个候选取最大，
    "bias": {"cards": 卡牌词表名集合, "delta": 加分} 成套偏离测试（决策记录 265）：整局的选卡奖励与商店里，候选涉及这些牌（商店只算买）的对数几率加 delta 再取最大，
    其余决策照原样}。"""
    out = []
    vocab = Catalog().vocab
    pol = policy or {}
    rng = np.random.default_rng(pol.get("seed", 0))
    explore = {k: float(v) for k, v in (pol.get("explore") or {}).items() if float(v) > 0}
    snap_of: dict[int, str] = {}; boss3: dict[int, dict] = {}
    boss_snaps: dict[int, dict] = {}     # boss_snaps：第一、二幕 Boss 开打前的快照（结果的 boss1、boss2 项）
    stage2: set[int] = set()     # 连打（spec 带 next_encounter）打到了第二场的引擎（头部 23 为本幕已打赢的 Boss 数）
    log = open(os.path.join(os.path.dirname(loop_log), "engine.log"), "ab") if loop_log else subprocess.DEVNULL
    with VecEnv(envs, log=log, engine="csharp") as env:     # 评测始终用游戏本体（准绳），不随采集换引擎
        start_hp, mode_of, kind_of, steps, recent, req_of, last, combats, choices, offers, routes = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
        ancients: dict[int, dict] = {}
        relics: dict[int, dict] = {}        # 跳过遗物的探针：(层, 界面, 遗物) → [幕, 0 提供 / 1 拿了 / 2 跳过]
        relic_skips: dict[int, list] = {}   # 主动跳过时的现场（relic_skip_detail）
        seed_of, tape, tag_of = {}, {}, {}
        sent: dict[int, list[int]] = {}       # 这一局依次走过的动作、reset 请求原文（卡住时写进 stuck.jsonl，照着重放可复现）
        req_full: dict[int, dict] = {}
        rec_dir = os.path.join(os.path.dirname(loop_log), "replays") if loop_log else None
        obs: dict[int, object] = {}
        t0: dict[int, float] = {}            # 每局开局时刻（单局时长上限）
        fight0: dict[int, int] = {}          # 整局：最近一次不在战斗中时的步数（单场战斗步数上限 caps["fight"]：连续在战斗中的步数）
        killed: dict[int, str] = {}          # 因在途超时被结束进程的引擎 → 卡住原因
        next_check = time.time() + 5.0
        exhausted = False

        def launch(i: int) -> None:
            nonlocal exhausted
            item = None if exhausted else next_task()
            if item is None:
                exhausted = True
                return
            req, mode, kind = item
            mode_of[i], kind_of[i], steps[i], recent[i], combats[i], choices[i] = mode, kind, 0, deque(maxlen=40), [], Counter()
            offers[i] = {}
            ancients[i] = {}
            relics[i] = {}; relic_skips[i] = []
            routes[i] = RouteTrack()
            req_of[i] = json.dumps(req, ensure_ascii=False)[:120]
            req_full[i] = {k: x for k, x in req.items() if k != "tag"}
            sent[i] = []
            seed_of[i] = req.get("seed")
            tag_of[i] = req.get("tag")
            t0[i] = time.time()
            fight0[i] = 0
            killed.pop(i, None)
            tape[i] = [] if (rec_dir or all_dir) and mode == MODE_RUN else None
            env.send(i, dict({k: x for k, x in req.items() if k != "tag"}, op="reset"))

        def finish(i: int, o, stuck: bool, why: str = "") -> None:
            if stuck and loop_log:      # 现场记录（2026-10-27 起）：与 loop_log 同目录的 stuck.jsonl
                try:
                    with open(os.path.join(os.path.dirname(loop_log), "stuck.jsonl"), "a", encoding="utf-8") as f:
                        f.write(json.dumps({"time": round(time.time(), 1), "version": version, "kind": kind_of[i], "reason": why or "引擎判卡住",
                                            "act": int(o.act), "floor": int(o.floor), "steps": steps.get(i, 0), "recent": list(recent.get(i, [])),
                                            "req": {"op": "reset", **req_full.get(i, {})}, "actions": sent.get(i, [])},
                                           ensure_ascii=False, separators=(",", ":")) + "\n")
                except Exception as e:
                    print(f"卡住记录写入失败：{e}", flush=True)
            hp0, mhp = start_hp.pop(i)
            last.pop(i, None)
            out.append({"kind": kind_of[i], "win": bool(o.victory) and not stuck, "floor": int(o.floor), "act": int(o.act),
                        "stuck": stuck, "hp_delta": (o.hp - hp0) / mhp, "combats": combats.pop(i, []), "choices": dict(choices.pop(i, {})),
                        "cards": card_counts(offers.pop(i, {})), "ancient": card_counts(ancients.pop(i, {})), "relic": relic_counts(relics.pop(i, {})), "relic_skips": relic_skips.pop(i, []), "route": routes.pop(i).result() if i in routes else [],
                        "tag": tag_of.pop(i, None), **({"boss3": boss3.pop(i)} if i in boss3 else {}), **boss_snaps.pop(i, {}),
                        **({"stage2": i in stage2 or int(getattr(o, "header", [0] * 24)[23]) >= 1} if mode_of.get(i) == MODE_COMBAT else {})})
            snap_of.pop(i, None); stage2.discard(i)
            steps_rec = tape.pop(i, None)
            if steps_rec and rec_dir and (out[-1]["win"] or replay_all):
                save_replay(rec_dir, version, seed_of.get(i), out[-1], steps_rec)
            if steps_rec and all_dir:
                save_replay(all_dir, version, seed_of.get(i), out[-1], steps_rec)
            if on_done:
                on_done(kind_of[i])
            if on_result:
                on_result(out[-1])
            launch(i)

        def receive(i: int, o) -> None:
            if isinstance(o, EngineError):   # 引擎报错：按卡住结束这一局，层数按最后一次观测
                start_hp.setdefault(i, (0, 1))
                p = last.pop(i, None)
                finish(i, SimpleNamespace(victory=False, floor=p.floor if p else 0, act=p.act if p else 0, hp=p.hp if p else 0), True,
                       killed.pop(i, None) or "引擎报错：" + str(o)[:300])
                return
            last[i] = o
            if mode_of.get(i) == MODE_COMBAT and len(o.header) > 23 and int(o.header[23]) >= 1:
                stage2.add(i)
            if o.extra and "combat" in o.extra:
                combats[i].append(o.extra["combat"])
                c = o.extra["combat"]
                # 第三幕第一个 Boss：本幕还没打过 Boss（第二个 Boss 开打时这一局已经记过了）；头部 17 为紧接着的第二个 Boss（观测格式 15）
                if pol.get("boss3_snap") and c.get("room") == "Boss" and c.get("act") == 2 and i not in boss3 and snap_of.get(i) and 0 < int(o.header[17]) < len(vocab):
                    boss3[i] = {"encounter": c["encounter"], "next_encounter": vocab[int(o.header[17])], "hp": c.get("hp"), "max_hp": c.get("max_hp"),
                                "potions": c.get("potions"), "upgrades": c.get("upgrades"), "deck": c.get("deck"), "snap": snap_of[i]}
                # 第一、二幕 Boss（boss_snaps，2026-11-17 树搜索在前两幕 Boss 上评测取题用；默认关）：同 boss3 的取法，每幕只记第一次
                if pol.get("boss_snaps") and c.get("room") == "Boss" and c.get("act") in (0, 1) and snap_of.get(i):
                    bs = boss_snaps.setdefault(i, {})
                    key = f"boss{int(c['act']) + 1}"
                    if key not in bs:
                        bs[key] = {"encounter": c["encounter"], "hp": c.get("hp"), "max_hp": c.get("max_hp"), "potions": c.get("potions"),
                                   "upgrades": c.get("upgrades"), "deck": c.get("deck"), "snap": snap_of[i]}
            if o.extra and "snapshot" in o.extra:
                snap_of[i] = o.extra["snapshot"]
            if i not in start_hp:
                start_hp[i] = (o.hp, max(1, o.max_hp))
            if o.done or o.stuck or o.num_actions == 0:
                finish(i, o, bool(o.stuck))
                return
            steps[i] += 1
            recent[i].append(f"{o.decision.name}:{o.num_actions}")
            # 整局里单场战斗打转（亡灵契约师决策记录第 10 条：十足蜈蚣节段反复接回，无进展判输判不下来，原来要拖到 20,000 步或单局时限）：
            # 连续在战斗中超过 caps["fight"] 步按卡住结束（结局本来也是“超过步数”卡住、判负，只是早停）；离开战斗就重新数
            if kind_of[i] == "run" and caps and caps.get("fight"):
                if not o.in_combat:
                    fight0[i] = steps[i]
                elif steps[i] - fight0.get(i, 0) > caps["fight"]:
                    if loop_log:
                        with open(loop_log, "a", encoding="utf-8") as f:
                            f.write(f"{time.strftime('%F %T')} {kind_of[i]} 单场战斗超过 {caps['fight']} 步（整局第 {steps[i]} 步），第 {o.floor} 层；开局 {req_of[i]}；"
                                    f"最近决策 {dict(Counter(recent[i]))}\n")
                    finish(i, o, True, f"单场战斗超过 {caps['fight']} 步")
                    return
            if caps and steps[i] > caps.get(kind_of[i], float("inf")):
                if loop_log:
                    seen = Counter(recent[i])
                    with open(loop_log, "a", encoding="utf-8") as f:
                        f.write(f"{time.strftime('%F %T')} {kind_of[i]} 超过 {caps[kind_of[i]]} 步，第 {o.floor} 层；开局 {req_of[i]}；"
                                f"最近决策 {dict(seen)}\n")
                finish(i, o, True, f"超过 {caps[kind_of[i]]} 步")
                return
            if time_cap and time.time() - t0[i] > time_cap:
                if loop_log:
                    with open(loop_log, "a", encoding="utf-8") as f:
                        f.write(f"{time.strftime('%F %T')} {kind_of[i]} 超过 {time_cap:.0f} 秒（{steps[i]} 步），第 {o.floor} 层；开局 {req_of[i]}；"
                                f"最近决策 {dict(Counter(recent[i]))}\n")
                finish(i, o, True, f"超过 {time_cap:.0f} 秒（{steps[i]} 步）")
                return
            obs[i] = o

        for i in range(len(env)):
            launch(i)
        while (env.pending or obs) and not stop_event.is_set():
            for i, o in env.recv_ready(timeout=1.0):
                receive(i, o)
            if time_cap and time.time() >= next_check:
                next_check = time.time() + 5.0
                for i in list(env.pending):
                    if i not in killed and i in t0 and time.time() - t0[i] > time_cap + 60:
                        killed[i] = f"超过 {time_cap:.0f} 秒（{steps.get(i, 0)} 步），引擎无响应，结束进程"
                        try:
                            env.envs[i]._proc.p.kill()
                        except Exception as e:
                            print(f"结束超时引擎失败：{e}", flush=True)
            # 凑批：在途的引擎稍等 2 毫秒，减少推理次数
            deadline = time.time() + 0.002
            while obs and env.pending and time.time() < deadline:
                for i, o in env.recv_ready(timeout=max(0.0, deadline - time.time())):
                    receive(i, o)
            if not obs:
                continue
            ids = list(obs)
            modes = np.array([pol.get("mode_as", mode_of[i]) for i in ids])
            with torch.no_grad():
                rg = Ragged.from_obs([obs[i] for i in ids], modes)
                # 成员结构不同的集成（tools/ability_test.py 的 EnsembleNet）自己按各成员的配置组批
                logits, _ = net.from_ragged(rg, device) if hasattr(net, "from_ragged") else net(collate_ragged(rg).to(device))
                if pol.get("mode_cmp") is not None:
                    l2, _ = net(collate_ragged(Ragged.from_obs([obs[i] for i in ids], np.full(len(ids), pol["mode_cmp"]))).to(device))
                    lp1, lp2 = torch.log_softmax(logits.float(), -1), torch.log_softmax(l2.float(), -1)
                    ok = torch.isfinite(lp1)
                    kl = torch.where(ok, lp1.exp() * (lp1 - lp2), torch.zeros_like(lp1)).sum(-1)
                    c = pol.setdefault("_cmp", [0, 0, 0.0])
                    c[0] += len(ids); c[1] += int((logits.argmax(-1) == l2.argmax(-1)).sum()); c[2] += float(kl.sum())
            probs = torch.softmax(logits.float(), -1).cpu().numpy() if any(tape.get(i) is not None for i in ids) else None
            acts = logits.argmax(-1).tolist()
            if pol.get("merge"):
                pm = torch.softmax(logits.float(), -1).cpu().numpy()
                acts = [merged_argmax(obs[i], pm[k]) for k, i in enumerate(ids)]
            if pol.get("bias"):         # 成套偏离测试（决策记录 265）：选卡奖励、商店买牌时偏向指定流派的牌
                bset, bd = pol["bias"]["cards"], float(pol["bias"]["delta"])
                lg = logits.float().cpu().numpy()
                for k, i in enumerate(ids):
                    o = obs[i]
                    if mode_of[i] != MODE_RUN or not is_macro(o) or int(o.decision) not in (int(DecisionKind.CARD_REWARD), int(DecisionKind.SHOP)):
                        continue
                    row = lg[k, :o.num_actions].copy(); hit = False
                    for c in range(o.num_actions):
                        if int(o.decision) == int(DecisionKind.SHOP) and int(o.cand_ints[c, 0]) != int(ActionKind.BUY):
                            continue
                        if cand_name(o, c, vocab) in bset:
                            row[c] += bd; hit = True
                    if hit:
                        acts[k] = int(np.argmax(row))
                        pol.setdefault("_bias_n", [0, 0])[0] += 1
                        pol["_bias_n"][1] += int(acts[k] != int(np.argmax(lg[k, :o.num_actions])))
            if pol.get("temp") or explore:
                if pol.get("temp"):
                    pr = torch.softmax(logits.float() / float(pol["temp"]), -1).cpu().numpy().astype(np.float64)
                    acts = [int(rng.choice(len(row), p=row / row.sum())) for row in pr]
                for k, i in enumerate(ids):
                    if explore and mode_of[i] == MODE_RUN and is_macro(obs[i]) and (ex := explore_choice(obs[i], acts[k], vocab, explore, rng)) is not None:
                        acts[k] = ex[1]
            for k, (i, a) in enumerate(zip(ids, acts)):
                if mode_of[i] == MODE_RUN and is_macro(obs[i]):
                    choices[i][choice_key(obs[i], a, vocab)] += 1
                    nr, ng = skipped_free(obs[i], a, vocab)      # 放弃的无负面遗物、金币（决策记录 160）
                    if nr or ng:
                        choices[i]["FREE_SKIP:RELIC"] += nr; choices[i]["FREE_SKIP:GOLD"] += ng
                    if obs[i].decision.name == "EVENT" and (anc := ancient_offer(obs[i], a, vocab)) is not None:
                        ancients[i][(int(obs[i].floor), anc[0])] = (int(obs[i].act), anc[1])
                    if (ro := relic_offer(obs[i], a, vocab)) is not None:
                        names, ks, taken, skipped = ro
                        for n in names:
                            st = relics[i].setdefault((int(obs[i].floor), obs[i].decision.name, n), [int(obs[i].act), 0])
                            if n == taken:
                                st[1] = 1
                            elif skipped and st[1] == 0:
                                st[1] = 2
                        if skipped:
                            p = torch.softmax(logits[k].float(), -1).cpu().numpy()
                            relic_skips[i].append(dict(relic_skip_detail(obs[i], a, vocab, names, ks, p), seed=seed_of.get(i)))
                    if obs[i].decision.name == "CARD_REWARD":
                        names, picked = card_offer(obs[i], a, vocab)
                        offers[i][(int(obs[i].floor), names)] = (int(obs[i].act), picked)
                    elif obs[i].decision.name == "MAP":
                        routes[i].step(obs[i], a, vocab)
                if tape.get(i) is not None:
                    tape[i].append(json.dumps(describe(obs[i], a, probs[k]), separators=(",", ":")))
                sent[i].append(int(a))
                env.send_step(i, a)
            obs.clear()
    return out
