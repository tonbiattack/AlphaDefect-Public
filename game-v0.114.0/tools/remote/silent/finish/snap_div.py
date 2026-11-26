# 猎人快照之间的首选不同（照储君第 24 条补记“首选不同”的量法）：同一批 4,096 个多选训练局面（第 24,599 版的批次存档），
# 各快照贪心首选两两比较；另算窄池 / 宽池集成（每步概率平均）与各成员、滑动平均权重的差异。CPU 推理（显卡被训练占着）。
import json, pickle, sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, "/root/silent-deploy/sls2/python")
from sts2rl.features import collate_ragged
from sts2rl.model import ModelConfig, PolicyValueNet, load_state
from sts2env.protocol import DecisionKind
torch.set_num_threads(4)
CK = Path("/root/ens_silent/ck"); KEEP = Path("/root/.local/share/sls2/runs/silent/keep"); OUT = Path("/root/ens_silent/div")
OUT.mkdir(exist_ok=True)
sub = OUT / "states.pkl"
if sub.exists():
    data = pickle.load(open(sub, "rb"))
else:
    r = pickle.load(open("/root/distill_silent/d00_v24599.pkl", "rb"))
    multi = np.flatnonzero(np.diff(r.cand_off) > 1)
    print("批内", len(r.cand_off) - 1, "步，多选", len(multi), flush=True)
    data = r.select(np.sort(np.random.default_rng(0).choice(multi, 4096, replace=False)))
    pickle.dump(data, open(sub, "wb")); del r
combat = (data.header[:, 11] != 0) | (data.header[:, 1] == int(DecisionKind.COMBAT))
print("局面", len(data), "战斗", int(combat.sum()), "局外", int((~combat).sum()), flush=True)

def load_blob(name):
    p = CK / f"{name}.pt"
    if p.exists():
        return torch.load(p, map_location="cpu", weights_only=False)
    v = int(name[1:]); f = sorted(KEEP.glob(f"v{v}-*.pt"))
    f = [x for x in f if "滑动" not in x.name]
    b = torch.load(f[0], map_location="cpu", weights_only=False)
    if name[0] == "m":
        ema = b.get("ema") or {}; b["state"] = {k: (ema[k].to(t.dtype) if k in ema else t) for k, t in b["state"].items()}
    return b

def probs(name):
    f = OUT / f"{name}.npy"
    if f.exists(): return np.load(f)
    t = time.time(); b = load_blob(name); mc = b["config"]["model"]
    net = PolicyValueNet(ModelConfig(**mc)).eval(); load_state(net, b["state"]); del b
    out = []
    with torch.no_grad():
        for j in range(0, len(data), 256):
            bt = collate_ragged(data.select(np.arange(j, min(j + 256, len(data)))), attend=bool(mc.get("macro_attend")))
            p = torch.softmax(net(bt.to(torch.device("cpu")))[0].float(), -1).numpy()
            out.append(np.pad(p, ((0, 0), (0, 64 - p.shape[1]))) if p.shape[1] < 64 else p[:, :64])
    x = np.concatenate(out).astype(np.float32); np.save(f, x)
    print(f"{name} {time.time() - t:.0f} 秒", flush=True)
    return x

def dis(x, y):
    d = x.argmax(-1) != y.argmax(-1)
    return f"{d.mean():.3f}（战斗 {d[combat].mean():.3f} 局外 {d[~combat].mean():.3f}）"

narrow = ["v24460", "v24220", "v24043", "v23880", "v23720", "v23540", "v23360"]
wide = ["v24460", "v23880", "v23360", "v22860", "v22320", "v21760", "v21240", "v20700"]
emas = ["m24460", "m23880", "m23360", "m22320"]
P = {n: probs(n) for n in dict.fromkeys(narrow + wide + emas)}
print("\n对第 24,460 版原始权重：")
for n in narrow[1:] + wide[3:]:
    print(f"  {n}（相隔 {24460 - int(n[1:])} 版）：{dis(P['v24460'], P[n])}")
def pair(names):
    ds = [(P[a].argmax(-1) != P[b].argmax(-1)).mean() for i, a in enumerate(names) for b in names[i + 1:]]
    return f"两两平均 {np.mean(ds):.3f}（{min(ds):.3f}–{max(ds):.3f}）"
print("窄池 7 个（第 23,360–24,460 版）：", pair(narrow))
print("宽池 8 个（第 20,700–24,460 版）：", pair(wide))
print("滑动平均 4 个：", pair(emas))
En = np.mean([P[n] for n in narrow], 0); Ew = np.mean([P[n] for n in wide], 0)
print("\n窄池集成 对 第 24,460 版原始：", dis(En, P["v24460"]), "；对滑动平均 m24460：", dis(En, P["m24460"]))
print("宽池集成 对 第 24,460 版原始：", dis(Ew, P["v24460"]), "；对滑动平均 m24460：", dis(Ew, P["m24460"]), "；对窄池集成：", dis(Ew, En))
print("滑动平均 m24460 对 第 24,460 版原始：", dis(P["m24460"], P["v24460"]))
for n in ["m23880", "m23360", "m22320"]:
    print(f"滑动平均 m24460 对 {n}：{dis(P['m24460'], P[n])}")
# 成员对集成的平均差异：成员偏差平均掉多少
print("窄池成员对窄池集成平均：", f"{np.mean([(P[n].argmax(-1) != En.argmax(-1)).mean() for n in narrow]):.3f}",
      "；宽池成员对宽池集成平均：", f"{np.mean([(P[n].argmax(-1) != Ew.argmax(-1)).mean() for n in wide]):.3f}")
