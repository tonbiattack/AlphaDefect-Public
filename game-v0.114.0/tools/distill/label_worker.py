"""搜索蒸馏的标签产出（2026-10-27 起在 A 上，用户要求：act3v4 跑完后 A 换最新引擎和检查点产标签，B 加辅助损失，用考卷衡量）。

从 ~/distill/problems/ 取题（本机中转 tools/distill/relay.py 放进来的 Boss 战开打局面，取自主线导出的真实快照，不含考卷的题），
用 tools/mcts_eval.py 的确定化搜索（价值估值叶子版，即 act3v5 的参数）把这场 Boss 战打完。每个搜索过的决策记一条标签：
当时的观测、网络原本的选择、搜索的选择、是否改选。每做完 --batch 道题写一个标签文件到 ~/distill/out/（pickle 再 gzip，先写临时文件再改名），
由本机中转取走、转给主线训练（python/sts2rl/distill.py）。

检查点：~/distill/ckpt/latest.pt（本机中转定期从主线换新），每批开始前看一眼，变了就重新载入。
多进程：同时开几个进程（--worker 编号不同），各自从题目目录里认领文件（改名到 claimed/<编号>/），互不重复。

    .venv/bin/python -I tools/distill/label_worker.py --worker 0 --envs 6 --live 4 --device cuda
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import pickle
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "tools"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import mcts_eval  # noqa: E402
from sts2env.protocol import FORMAT_VERSION  # noqa: E402
from sts2rl.actor import MODE_COMBAT  # noqa: E402
from sts2rl.features import Ragged  # noqa: E402
from sts2rl.model import ModelConfig, PolicyValueNet, load_state  # noqa: E402

ROOT = Path(os.environ.get("DISTILL_ROOT") or Path.home() / "distill")


def claim(worker: int, n: int) -> list[Path]:
    """从题目目录认领最多 n 道题（改名到 claimed/<worker>/，改名是原子的，多个进程不会认领到同一道）。"""
    mine = ROOT / "problems" / "claimed" / str(worker)
    mine.mkdir(parents=True, exist_ok=True)
    got = sorted(mine.glob("*.json"))           # 上次没做完（进程被停）的先做
    for f in sorted((ROOT / "problems").glob("*.json")):
        if len(got) >= n:
            break
        try:
            os.rename(f, mine / f.name)
            got.append(mine / f.name)
        except OSError:
            continue
    return got[:n]


def load_net(path: Path, device: str):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if blob.get("obs_format", FORMAT_VERSION) != FORMAT_VERSION:
        raise SystemExit(f"检查点的观测格式 {blob.get('obs_format')} 与本机代码 {FORMAT_VERSION} 不同：先同步引擎与代码")
    net = PolicyValueNet(ModelConfig(**blob["config"]["model"])).eval().to(device)
    load_state(net, blob["state"])
    return net, int(blob["version"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--batch", type=int, default=24, help="每做完这么多道题写一个标签文件")
    ap.add_argument("--envs", type=int, default=6)
    ap.add_argument("--live", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    # 搜索参数：默认是 act3v5（价值估值叶子）的，第三幕考卷 +3.5±1.7、第二幕的第三版 +6.2±1.7
    ap.add_argument("--tau", type=float, default=0.9)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--pmin", type=float, default=0.02)
    ap.add_argument("--sims0", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--z", type=float, default=2.0)
    ap.add_argument("--leaf", type=int, default=12)
    ap.add_argument("--once", action="store_true", help="只做一批就退出（测试用）")
    a = ap.parse_args()
    a.paired = True
    mcts_eval.SCORE, mcts_eval.LEAF = "win", a.leaf
    torch.set_num_threads(2)
    out_dir = ROOT / "out"; out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ROOT / "ckpt" / "latest.pt"
    net, version, mtime = None, -1, None
    seq = 0
    while True:
        if not ckpt.exists():
            print("等检查点 ~/distill/ckpt/latest.pt", flush=True); time.sleep(60); continue
        m = ckpt.stat().st_mtime
        if m != mtime:
            net, version = load_net(ckpt, a.device); mtime = m
            print(f"载入检查点第 {version} 版", flush=True)
        files = claim(a.worker, a.batch)
        if not files:
            time.sleep(30); continue
        items, meta = [], {}
        for j, f in enumerate(files):
            try:
                it = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                f.unlink(missing_ok=True); continue
            items.append((j, it)); meta[j] = (f, it)
        labels: dict[int, list] = {}
        results: dict[int, dict] = {}
        t0 = time.time()
        mcts_eval.run(a, items, net, a.device, on_item=lambda i, r: results.__setitem__(i, r), labels=labels)
        views, lab, greedy, changed, pg, act, pid, win = [], [], [], [], [], [], [], []
        for j, rs in labels.items():
            it = meta[j][1]
            for o, g, b, p in rs:
                views.append(o); greedy.append(g); lab.append(b); changed.append(b != g); pg.append(p)
                act.append(it["act"]); pid.append(it["id"])
        for j in results:
            win.append(bool(results[j].get("win")))
        if views:
            rec = {"version": version, "obs_format": FORMAT_VERSION, "t": time.time(), "worker": a.worker,
                   "view": Ragged.from_obs(views, np.full(len(views), MODE_COMBAT)),
                   "label": np.array(lab, np.int64), "greedy": np.array(greedy, np.int64), "changed": np.array(changed, bool),
                   "p_greedy": np.array(pg, np.float32), "act": np.array(act, np.int8), "pid": pid,
                   # 这批题：道数、搜索打赢的道数、各题的幕
                   "problems": len(results), "wins": int(sum(win)), "problem_acts": [meta[j][1]["act"] for j in results]}
            name = f"w{a.worker}_{int(time.time())}_{seq}.pkl.gz"; seq += 1      # gzip 后约为原来的 1/40
            tmp = out_dir / ("." + name + ".tmp")
            with gzip.open(tmp, "wb", compresslevel=6) as fh:
                pickle.dump(rec, fh, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, out_dir / name)
            # 摘要（本机中转收集到 ~/.local/share/sls2/distill/labels.jsonl，看板用）：标签文件不用解开就能统计
            summ = {"file": name, "t": round(time.time()), "worker": a.worker, "version": version, "problems": len(results), "wins": int(sum(win)),
                    "labels": len(views), "changed": int(sum(changed)), "by_act": {str(k): [sum(1 for j in results if meta[j][1]["act"] == k),
                                                                                            sum(1 for j in results if meta[j][1]["act"] == k and results[j].get("win"))]
                                                                                        for k in (1, 2)}}
            (out_dir / (name[: -len(".pkl.gz")] + ".json")).write_text(json.dumps(summ, ensure_ascii=False), encoding="utf-8")
        for f, _ in meta.values():
            f.unlink(missing_ok=True)
        print(f"{time.strftime('%H:%M:%S')} 第 {version} 版：{len(results)} 道题，搜索赢 {sum(win)}；标签 {len(views)} 条，改选 {sum(changed)}；"
              f"用时 {time.time() - t0:.0f} 秒", flush=True)
        if a.once:
            break


if __name__ == "__main__":
    main()
