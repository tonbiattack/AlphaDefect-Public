"""训练停下时用整台机器评测若干检查点，挑回退的起点（贪心、固定种子 EVAL000 起，与在线评测同一套种子；30 个进程 × 4 个引擎，CPU 推理）。
检查点可以是训练的完整检查点（backups/、latest.pt），也可以是评测留档的网络参数（best/）。在租用机上：
  cd /root/sls2-deploy/sls2/python && ../.venv/bin/python ../tools/remote/eval_ckpts.py $PWD 384 检查点...
每份 384 局约 1.5 分钟；会占满 CPU，只在训练停下时用。"""
import json, os, sys, time, queue
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch

sys.path.insert(0, sys.argv[1])
from sts2rl.actor import MODE_RUN
from sts2rl.evaluator import play, _run_diagnostics
from sts2rl.model import ModelConfig, PolicyValueNet, load_state

W = {}


def init(path, tasks):
    torch.set_num_threads(1)
    net = PolicyValueNet(ModelConfig(d=256, layers=4)).eval()
    blob = torch.load(path, map_location="cpu", weights_only=False)
    load_state(net, blob.get("state", blob))
    W.update(net=net, tasks=tasks)


def drain(envs):
    class Never:
        def is_set(self): return False

    def nxt():
        try:
            return W["tasks"].get(timeout=1.0)
        except queue.Empty:
            return None
    return play(W["net"], torch.device("cpu"), nxt, envs, Never(), caps={"run": 6000, "combat": 400})


if __name__ == "__main__":
    n, ckpts = int(sys.argv[2]), sys.argv[3:]
    ctx = mp.get_context("spawn")
    for p in ckpts:
        t0 = time.time()
        tasks = ctx.Queue()
        for k in range(n):
            tasks.put(({"mode": "run", "character": "DEFECT", "ascension": 10, "seed": f"EVAL{k:03d}", "snapshots": True}, MODE_RUN, "run"))
        with ProcessPoolExecutor(30, mp_context=ctx, initializer=init, initargs=(p, tasks)) as pool:
            runs = [r for f in [pool.submit(drain, 4) for _ in range(30)] for r in f.result()]
        fl = np.array([r["floor"] for r in runs])
        d = _run_diagnostics(runs)["deck_at"]
        print(json.dumps({"ckpt": os.path.basename(p), "runs": len(runs), "floor_mean": round(float(fl.mean()), 2),
                          "floor_se": round(float(fl.std() / np.sqrt(len(fl))), 2), "floor_median": float(np.median(fl)),
                          "reach_act2": round(float(np.mean([r["act"] >= 1 or r["win"] for r in runs])), 3),
                          "reach_act3": round(float(np.mean([r["act"] >= 2 or r["win"] for r in runs])), 3),
                          "win": round(float(np.mean([r["win"] for r in runs])), 3), "stuck": int(sum(r["stuck"] for r in runs)),
                          "boss1": d["boss1"], "sec": round(time.time() - t0)}, ensure_ascii=False), flush=True)
