"""离线比价值头的准度（2026-11-17）：同策略采样对局（boss_pimc --temp 1 --save-frames，默认 /root/future/t2/samp）的每 STRIDE 步，
价值头读数与这局实际终局得分（sim_pimc.terminal 同尺度）比均方差、相关、偏差，分第一 / 二个 Boss 与局内进度。
  cd 部署目录 && .venv/bin/python -I tools/future/value_quality.py [STRIDE]（检查点路径写在下面）"""
import glob, json, sys, time
import numpy as np, torch
sys.path.insert(0, "python"); sys.path.insert(0, "tools")
from sts2env.protocol import parse
from sts2rl.model import ModelConfig, PolicyValueNet, load_state
import sim_pimc as X
D = "/root/future/t2/samp"; STRIDE = int(sys.argv[1]) if len(sys.argv) > 1 else 4
nets = {}
for name, p in (("L1000", "/root/i2a/data/L1000.pt"), ("v24181", "/root/i2a/data/v24181_ema.pt")):
    b = torch.load(p, map_location="cpu", weights_only=False)
    net = PolicyValueNet(ModelConfig(**b["config"]["model"])).eval().to("cuda"); load_state(net, b["state"]); nets[name] = net
ev = {k: X.make_evaluator(None, n, "cuda", policy_mode=0, value_mode=0, strip_priv=True) for k, n in nets.items()}
rows = []
for res in sorted(glob.glob(f"{D}/res_s*.jsonl")):
    k = res.split("res_s")[1].split(".")[0]; fb = open(f"{D}/frames_s{k}.bin", "rb")
    for l in open(res):
        r = json.loads(l); tr = r.get("traj") or []
        if r.get("stuck") or not tr: continue
        win = bool(r["win"]); left_end = 0.0 if win else tr[-1][5]
        term = 1.5 if win else -1.0 + 0.25 * (tr[-1][6] + 1.0 - left_end)
        for t in range(0, len(tr), STRIDE):
            off, ln, act, n, hp, left, stage, v, pa = tr[t]
            fb.seek(off); rows.append((parse(fb.read(ln)), term, stage, t / len(tr), v, r["id"]))
print(len(rows), "帧", flush=True)
V = {k: [] for k in ev}
t0 = time.time()
for i in range(0, len(rows), 256):
    fs = [x[0] for x in rows[i:i + 256]]
    for k, e in ev.items():
        _, v = e.forward(fs, [0] * len(fs)); V[k].extend(float(x) for x in v)
print(f"{time.time() - t0:.0f} 秒", flush=True)
Y = np.array([x[1] for x in rows]); ST = np.array([x[2] for x in rows]); FR = np.array([x[3] for x in rows]); V0 = np.array([x[4] for x in rows])
V = {k: np.array(v) for k, v in V.items()}; V["平均"] = (V["L1000"] + V["v24181"]) / 2
print("L1000 与记录里的读数最大差", float(np.abs(V["L1000"] - V0).max()))
for k, v in V.items():
    out = [f"{k:8s} 均方差 {np.mean((v - Y) ** 2):.4f} 相关 {np.corrcoef(v, Y)[0, 1]:.4f} 偏差 {np.mean(v - Y):+.4f}"]
    for s in (0, 1):
        m = ST == s; out.append(f"第{s + 1}个 Boss 均方差 {np.mean((v[m] - Y[m]) ** 2):.4f}")
    for lo, hi in ((0, .33), (.33, .67), (.67, 1.01)):
        m = (FR >= lo) & (FR < hi); out.append(f"进度 {lo:.2f}+ {np.mean((v[m] - Y[m]) ** 2):.4f}")
    print("  ".join(out))
print("两网读数相关", np.corrcoef(V["L1000"], V["v24181"])[0, 1])
