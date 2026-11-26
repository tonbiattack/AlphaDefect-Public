"""胜率模型重新标定：每个头的输出对数几率做 a·z + c（训练种子上按加权对数损失拟合），折进最后一层，结构与载入方式不变。
在按种子留出的测试种子上比较原模型与标定后的模型（决策记录 234）。
    python tools/winprob/recal.py 特征缓存目录（train.py --cache-dir 写的 *.pkl） 模型.pt 输出.pt
默认用显卡；环境变量 WINPROB_DEVICE=cpu 改用 CPU。
环境变量 WINPROB_SHIFT_ONLY=1 时只平移截距（a 固定为 1、只拟合 c）：储君第 30 条的做法，a·z + c 在带搜索的局上两头过于自信、对数损失反而变差。"""
import copy, glob, os, pickle, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import bucket, calib, concat, scores  # noqa
from winprob import HEADS, N_SCALAR, WinProb, select, to_torch  # noqa

d = concat([pickle.load(open(f, "rb")) for f in sorted(glob.glob(f"{sys.argv[1]}/*.pkl"))])
seeds = d.pop("seed"); d.pop("games")
useed, inv = np.unique(seeds, return_inverse=True)
u = np.asarray([bucket(str(x)) for x in useed])[inv]
dev = torch.device(os.environ.get("WINPROB_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu"))   # 显存被占满的机器上设 WINPROB_DEVICE=cpu
b = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
am = b["model"]["mlp.0.weight"].shape[1] == 9 * b["args"].get("d", 16) + N_SCALAR
net = WinProb(**{**b["args"], "amounts": am}).to(dev).eval(); net.load_state_dict(b["model"])

@torch.no_grad()
def logits(D, bs=65536):
    out = []
    for i in range(0, D["n"], bs):
        out.append(net(to_torch(select(D, np.arange(i, min(i + bs, D["n"]))), dev)).float().cpu())
    return torch.cat(out).numpy()

SHIFT = os.environ.get("WINPROB_SHIFT_ONLY") == "1"
rng = np.random.default_rng(0)
fit_i = np.flatnonzero(u >= 0.3); fit_i = np.sort(rng.choice(fit_i, min(len(fit_i), 800000), replace=False))
F, T = select(d, fit_i), select(d, np.flatnonzero(u < 0.2))
zF, zT = logits(F), logits(T)
ac = []
for h in range(3):
    m = F["y"][:, h] >= 0
    z, y, w = (torch.tensor(x, dtype=torch.float64) for x in (zF[m, h], F["y"][m, h].astype(float), F["w"][m]))
    a, c = torch.ones(1, dtype=torch.float64, requires_grad=not SHIFT), torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([c] if SHIFT else [a, c], max_iter=200, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad(); l = (w * torch.nn.functional.binary_cross_entropy_with_logits(a * z + c, y, reduction="none")).sum() / w.sum(); l.backward(); return l
    opt.step(closure); ac.append((a.item(), c.item()))
print("标定参数（a, c）：", {HEADS[h]: (round(a, 4), round(c, 4)) for h, (a, c) in enumerate(ac)})
sig = lambda x: 1 / (1 + np.exp(-x))
print(f"测试种子 {T['n']} 步、{len(set(T['game']))} 局")
for h, head in enumerate(HEADS):
    m = T["y"][:, h] >= 0
    y, w, z = T["y"][m, h].astype(float), T["w"][m], zT[m, h]
    for name, p in (("原", sig(z)), ("标定", sig(ac[h][0] * z + ac[h][1]))):
        s = scores(p, y, w)
        print(f"{head} {name}：对数损失 {s['logloss']}  Brier {s['brier']}  AUC {s['auc']}  平均预测 {np.average(p, weights=w):.3f}  实际 {s['rate']}")
        for x in range(3):
            mx = T["act"][m] == x
            print(f"    第{x + 1}幕：平均预测 {np.average(p[mx], weights=w[mx]):.3f}  实际 {np.average(y[mx], weights=w[mx]):.3f}")
        print("    校准：", [(c[0], c[1], c[2]) for c in calib(p, y, w)])
sd = copy.deepcopy(b["model"])
A = torch.tensor([a for a, _ in ac], dtype=sd["mlp.6.weight"].dtype); C = torch.tensor([c for _, c in ac], dtype=sd["mlp.6.bias"].dtype)
sd["mlp.6.weight"] = sd["mlp.6.weight"] * A[:, None]; sd["mlp.6.bias"] = sd["mlp.6.bias"] * A + C
out = {**b, "model": {k: v.cpu() for k, v in sd.items()}, "recal": {"from": sys.argv[2], "a_c": ac, "data": sys.argv[1], "shift_only": SHIFT}}
torch.save(out, sys.argv[3])
net.load_state_dict(out["model"]); chk = logits(select(T, np.arange(2000)))
print("折进最后一层后核对：最大偏差", float(np.abs(chk - (np.array([a for a, _ in ac]) * zT[:2000] + np.array([c for _, c in ac]))).max()))
