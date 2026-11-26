"""搜索整局流（2026-11-17，训练中用搜索的试点）：跟着一条训练线的最新检查点，一轮轮用 run_search.py 打整局（局外贪心，Boss 与精英战 32 次模拟的树搜索）。

用户：“开始试点，两台机器还是给你，充分利用”。
  - 搜索组：--labels 打开，搜过的根写进 labels_s0.jsonl，学习端（search_kl_coef > 0，sts2rl/search_kl.py）读去做 KL 辅助损失；
  - 对照组：同样跑（两组算力对称），只当“带搜索的整局胜率”的走势看，学习端不读。
每轮开始前看训练目录的 latest.pt 有没有更新：更新了就把滑动平均（ema）换进 state、只留网络权重，存成本进程自己的检查点（与 68.4% 那组测试同一种权重）。
种子：--seeds 池里从 --start 起，第 r 轮、第 k 个进程（共 n 个）打 [start + (r·n + k)·chunk, +chunk)；各进程、各轮不重叠，中断后按已有的轮次目录接着打。
结果在 <out>/w<k>/r<轮>/（runs_s0.jsonl、labels_s0.jsonl，另记这一轮用的网络版本 version.txt）；<out>/STOP 存在时打完这一轮就退出。

  .venv/bin/python -I tools/future/search_stream.py --run-dir 训练目录 --out 输出目录 --seeds 种子池.json --worker 0 --workers 6 [--labels]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_search as RS  # noqa: E402


def refresh(run_dir: Path, mine: Path, last: list) -> int | None:
    """训练目录的 latest.pt 更新了就重做本进程的检查点；返回网络版本（没有可用检查点时 None）。"""
    import torch
    src = run_dir / "latest.pt"
    if not src.exists():
        return last[1]
    mt = src.stat().st_mtime
    if last[0] == mt and mine.exists():
        return last[1]
    try:
        b = torch.load(src, map_location="cpu", weights_only=False)
    except Exception as e:      # 正好在写（学习端写临时文件再改名，一般不会），下一轮再试
        print(f"{time.strftime('%H:%M:%S')} 读 {src} 失败：{e}", flush=True)
        return last[1]
    state = dict(b["state"])
    if b.get("ema"):
        state.update({k: v for k, v in b["ema"].items() if k in state})
    out = {k: b[k] for k in ("version", "config", "vocab_n", "obs_format", "content_names", "catalog_version") if k in b}
    out["state"] = state
    out["note"] = f"search_stream：{src} 第 {b.get('version')} 版，ema 换进 state"
    tmp = mine.with_suffix(".tmp")
    torch.save(out, tmp)
    os.replace(tmp, mine)
    last[0], last[1] = mt, int(b.get("version", 0))
    print(f"{time.strftime('%H:%M:%S')} 换上第 {last[1]} 版", flush=True)
    return last[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True); ap.add_argument("--out", required=True); ap.add_argument("--seeds", required=True)
    ap.add_argument("--start", type=int, default=0); ap.add_argument("--worker", type=int, required=True); ap.add_argument("--workers", type=int, required=True)
    ap.add_argument("--chunk", type=int, default=8); ap.add_argument("--labels", action="store_true")
    ap.add_argument("--sims", type=int, default=32); ap.add_argument("--rooms", default="Boss,Elite"); ap.add_argument("--lam", type=float, default=0.06)
    ap.add_argument("--procs", type=int, default=8); ap.add_argument("--device", default="cuda")
    ap.add_argument("--dll", default="", help="推演与实打用的引擎程序集（默认部署目录里的；2026-11-17 起用带战后奖励确定化补丁的那份）")
    a = ap.parse_args()
    run_dir, out = Path(a.run_dir), Path(a.out)
    wdir = out / f"w{a.worker:02d}"; wdir.mkdir(parents=True, exist_ok=True)
    pool = json.loads(Path(a.seeds).read_text(encoding="utf-8"))["seeds"]
    mine = wdir / "ck.pt"
    last = [None, None]
    r = 0
    while (wdir / f"r{r:05d}" / "done").exists():
        r += 1
    while not (out / "STOP").exists():
        ver = refresh(run_dir, mine, last)
        if ver is None:
            print("还没有检查点，等 60 秒", flush=True); time.sleep(60); continue
        lo = a.start + (r * a.workers + a.worker) * a.chunk
        seeds = pool[lo: lo + a.chunk]
        if not seeds:
            print("种子池用完", flush=True); break
        rd = wdir / f"r{r:05d}"; rd.mkdir(exist_ok=True)
        (rd / "seeds.json").write_text(json.dumps({"seeds": seeds}), encoding="utf-8")
        (rd / "version.txt").write_text(str(ver), encoding="utf-8")
        t0 = time.time()
        argv = ["--seeds", str(rd / "seeds.json"), "--runs", str(len(seeds)), "--ckpt", str(mine), "--out", str(rd), "--shard", "0/1",
                "--sims", str(a.sims), "--rooms", a.rooms, "--lam", str(a.lam), "--procs", str(a.procs), "--device", a.device,
                "--policy-seed", str(r)] + (["--labels"] if a.labels else []) + (["--dll", a.dll] if a.dll else [])
        RS.run(RS.parse(argv))
        (rd / "done").write_text(time.strftime("%F %T"), encoding="utf-8")
        print(f"{time.strftime('%H:%M:%S')} 第 {r} 轮（第 {ver} 版，种子 {lo}–{lo + len(seeds) - 1}）{time.time() - t0:.0f} 秒", flush=True)
        r += 1
    print("退出", flush=True)


if __name__ == "__main__":
    main()
