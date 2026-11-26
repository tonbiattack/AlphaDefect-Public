"""协同采集机（A）上的常规评测（2026-10-29 起，决策记录 198）：原来跑在主线学习机 B 上（6 个进程 × 4 个引擎、CPU 推理，
一轮 192 局约 5.3 分钟，占 B 约 8.5 个核），改到 A 上用显卡推理，B 的学习进程不再起评测（train_config 的 eval_runs 为 0）。

评测代码就是训练里的 sts2rl.evaluator.run_evaluator，不改：权重读本机采集进程用的那份（remote_actors.py 从 B 取来写进
/dev/shm/sts2rl_remote.pt，每次更新都有），配置取 B 的 train_config.json 副本，再按命令行改评测的局数、进程数、引擎数、推理设备、
CPU 上限与调度优先级。结果写 --run-dir（eval.jsonl、eval_progress.json、eval_error.txt、best/、eval_replays/ 等），本机拉取到
~/.local/share/sls2/runs/remote/defect-eval/，看板从切换时刻起改读这里的评测。评测进程退出就 30 秒后重启。

    .venv/bin/python -I tools/remote/remote_eval.py --config /root/b_train_config.json --run-dir ~/.local/share/sls2/runs/defect-eval \\
        --seeds-file tools/remote/seed_pool.json --workers 8 --envs 12 --device cuda --cores 20 --nice 5
停止：kill 本进程。
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import signal
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

from sts2rl.config import TrainConfig  # noqa: E402
from sts2rl.evaluator import run_evaluator  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--weights", default="/dev/shm/sts2rl_remote.pt")
    ap.add_argument("--runs", type=int, default=192)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--envs", type=int, default=12)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cores", type=float, default=10.0)
    ap.add_argument("--nice", type=int, default=5)
    ap.add_argument("--combats", type=int, default=0)
    ap.add_argument("--seeds-file", default=None, help="种子表（tools/eval_seeds.py 生成的 JSON），给了就用它、局数取种子数")
    ap.add_argument("--merge-equiv", action="store_true", help="贪心按等价动作合并（决策记录 205）")
    ap.add_argument("--ema", action="store_true", help="用权重文件里学习端的滑动平均权重（决策记录 223）")
    ap.add_argument("--no-replay-all", action="store_true", help="不另存每局的整局回放（eval_replays/，给胜率小模型攒数据用；储君暂不需要）")
    ap.add_argument("--keep-runs", type=int, default=0, help="逐局结果与权重留存到 run-dir/eval_runs/（复测沿用，决策记录 203）：逐局结果留最近几轮，0 为不留")
    a = ap.parse_args()
    run_dir = Path(a.run_dir).expanduser(); run_dir.mkdir(parents=True, exist_ok=True)
    cfg = TrainConfig.from_args(["--config", a.config, "--run-dir", str(run_dir)])
    cfg.eval_runs, cfg.eval_workers, cfg.eval_envs, cfg.eval_device = a.runs, a.workers, a.envs, a.device
    cfg.eval_cpu_cores, cfg.eval_nice, cfg.eval_combats, cfg.eval_every_s = a.cores, a.nice, a.combats, 0.0
    if a.seeds_file:
        # 种子池（tools/seed_pool.py，决策记录 240）带 eval（评测用的那部分）与 name（口径名）；旧的 eval_seeds.json 只有 seeds
        sf = json.loads(Path(a.seeds_file).read_text(encoding="utf-8"))
        cfg.eval_seeds = sf.get("eval") or sf["seeds"]; cfg.eval_runs = len(cfg.eval_seeds)
        if sf.get("name"):
            cfg.eval_seed_set = f"种子池 {sf['name']}（{cfg.eval_runs} 个）"
    cfg.eval_keep_runs = a.keep_runs; cfg.eval_merge_equiv = a.merge_equiv; cfg.eval_use_ema = a.ema
    cfg.eval_replay_all = not a.no_replay_all      # 每局整局另存回放（eval_replays/，直播的胜率小模型用，决策记录 175）；A 上的配置副本比这一项旧
    while not Path(a.weights).exists():
        print("等权重文件…", flush=True)
        time.sleep(10)
    ctx = mp.get_context("spawn")
    stop = ctx.Event()
    quit_ = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: quit_.set())
    signal.signal(signal.SIGINT, lambda *_: quit_.set())
    while not quit_.is_set():
        p = ctx.Process(target=run_evaluator, args=(cfg, a.weights, str(run_dir), stop), daemon=False)
        p.start()
        print(f"{time.strftime('%F %T')} 评测进程 {p.pid}：{cfg.eval_runs} 局，{a.workers} 个进程 × {a.envs} 个引擎，{a.device} 推理", flush=True)
        while p.is_alive() and not quit_.wait(5):
            pass
        if quit_.is_set():
            stop.set(); p.join(60)
            break
        print(f"{time.strftime('%F %T')} 评测进程退出（{p.exitcode}），30 秒后重启", flush=True)
        quit_.wait(30)


if __name__ == "__main__":
    main()
