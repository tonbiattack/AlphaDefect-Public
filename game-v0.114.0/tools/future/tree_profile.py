"""树搜索单分片的耗时拆分（2026-11-17）：等引擎回复（recv）、网络前向（ev.forward）、其余（Python 树逻辑、编码）。留出题前 4 道。
  cd 部署目录 && .venv/bin/python -I tools/future/tree_profile.py 模拟次数 [tree_search 的其余参数]"""
import json, sys, time
from pathlib import Path
sys.argv = [sys.argv[0], "run", "--items", "/root/i2a/data/items_p2.json", "--ckpt", "/root/i2a/data/L1000.pt", "--out", "/root/i2a/tree/prof",
            "--sims", sys.argv[1], "--decide", "kl", "--lam", "0.03"] + sys.argv[2:]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import tree_search as T
import boss_pimc as B
ap_args = None
_orig_parse = T.argparse.ArgumentParser.parse_args
def grab(self, *x, **k):
    global ap_args
    ap_args = _orig_parse(self, *x, **k); return ap_args
T.argparse.ArgumentParser.parse_args = grab
T.run = lambda a: None
T.main()
a = ap_args; a.k_shard = 0
Path(a.out).mkdir(parents=True, exist_ok=True)
acc = {"recv": 0.0, "n": 0}
orig_recv = B.Runner.recv
def recv(self, w):
    t = time.perf_counter(); r = orig_recv(self, w); acc["recv"] += time.perf_counter() - t; acc["n"] += 1; return r
B.Runner.recv = recv
items = [it for it in json.loads(Path(a.items).read_text())["items"] if it["id"] % 4 == 0][:4]
R = T.TreeRunner(a)
t0 = time.perf_counter()
for it in items:
    R.acc = {}
    res = R.play(it, None, None)
    print(it["id"], res["win"], res["searches"], res["s"], flush=True)
tt = time.perf_counter() - t0
print(f"总 {tt:.1f} 秒；等引擎 {acc['recv']:.1f} 秒（{acc['n']} 次回复）；网络前向 {R.ev.t_fwd:.1f} 秒（{R.ev.n_fwd} 次、{R.ev.n_rows} 行，每次平均 {R.ev.n_rows / max(1, R.ev.n_fwd):.1f} 行）；其余 {tt - acc['recv'] - R.ev.t_fwd:.1f} 秒")
R.close()
