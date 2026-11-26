"""格式 29 上线前的本机核对（2026-11-20，铁甲战士决策记录第 27 条；R2-编码第七节第 10 步）：
  1. 同种子、同动作锁步跑格式 28 与 29：记录整数、记录特征 0–21、候选整数、候选特征 0–13 逐位相同（只在登记的 V29 修正处允许不同，
     --allow-fix 时只报告不同的比例），新列 22–29、14–19 的取值概况；
  2. 同一份权重，ModelConfig.obs_v29 关与开在格式 29 的帧上输出逐位相同（新层清零初始化），旧检查点能载入；
  3. 新列给随机值时 feat_v29 / cand_v29 有梯度、输出随之变化；
  4. 上线冲击（--impact）：同一模型（打开 obs_v29，清零初始化）在格式 28 与格式 29 的配对帧上，按决策类型报 KL(28‖29)、首选一致率、价值差，
     对照 R2-编码第二节的门槛（各决策类型 KL 平均 ≤ 0.005、首选一致 ≥ 98%）。--greedy 时动作取模型在格式 29 帧上的首选（走到更真实的局面），
     否则偏随机。
用法（仓库根，CPU 即可）：.venv/bin/python -I tools/v29_launch_check.py <检查点或评测留档的裸权重> [--char IRONCLAD] [--steps 200] [--allow-fix] [--impact] [--greedy]
"""
import argparse, json, sys
sys.path.insert(0, "python")
import numpy as np, torch
from sts2env import VecEnv
from sts2rl.model import ModelConfig, PolicyValueNet, load_state
from sts2rl.features import Ragged, collate_ragged
from sts2rl.actor import MODE_RUN

ap = argparse.ArgumentParser()
ap.add_argument("weights"); ap.add_argument("--char", default="IRONCLAD"); ap.add_argument("--steps", type=int, default=200)
ap.add_argument("--envs", type=int, default=4); ap.add_argument("--allow-fix", action="store_true")
ap.add_argument("--impact", action="store_true"); ap.add_argument("--greedy", action="store_true"); ap.add_argument("--threads", type=int, default=4)
a = ap.parse_args()
torch.manual_seed(0); rng = np.random.default_rng(3); torch.set_num_threads(a.threads)
blob = torch.load(a.weights, map_location="cpu", weights_only=False)
if "config" in blob:
    mc, st = dict(blob["config"]["model"]), blob["state"]
else:
    mc, st = dict(json.load(open("tools/remote/ironclad/train_config.json"))["model"]), blob
mc.pop("obs_v29", None)
pol = None
if a.greedy:        # 贪心走局用的模型（格式 29 帧上取首选）
    pol = PolicyValueNet(ModelConfig(**{**mc, "obs_v29": True})).eval(); load_state(pol, st)

# 1. 锁步对拍
E28, E29 = VecEnv(a.envs), VecEnv(a.envs)
base = {"op": "reset", "mode": "run", "character": a.char, "ascension": 10, "privileged": True, "route_summary": True, "future_content": True}
req = lambda i, k, f: {**base, "seed": f"V29C{i}_{k}", "obs_format": f}
o28 = E28.reset([req(i, 0, 28) for i in range(a.envs)]); o29 = E29.reset([req(i, 0, 29) for i in range(a.envs)])
diff = {"tok": 0, "cand": 0, "frames": 0}; new_tok = []; new_cand = []; frames29 = []; frames28 = []
for t in range(a.steps):
    acts = []
    for i, (x, y) in enumerate(zip(o28, o29)):
        diff["frames"] += 1
        assert int(y.header[0]) == 29 and int(x.header[0]) == 28, (x.header[0], y.header[0])
        same_t = np.array_equal(x.tok_ints, y.tok_ints) and np.array_equal(x.tok_feats[:, :22], y.tok_feats[:, :22])
        same_c = np.array_equal(x.cand_ints, y.cand_ints) and np.array_equal(x.cand_feats[:, :14], y.cand_feats[:, :14])
        diff["tok"] += int(not same_t); diff["cand"] += int(not same_c)
        if not (same_t and same_c) and not a.allow_fix:
            raise SystemExit(f"第 {t} 步引擎 {i}：格式 28 与 29 的旧列不同（tok {same_t} cand {same_c}）")
        new_tok.append(y.tok_feats[:, 22:30]); new_cand.append(y.cand_feats[:, 14:20]); frames29.append(y); frames28.append(x)
        acts.append(int(rng.integers(y.num_actions)) if y.num_actions > 0 and not y.done else None)
    if pol is not None:     # 贪心：同一批格式 29 帧一次前向，取首选（候选列表两边相同）
        live = [i for i, y in enumerate(o29) if y.num_actions > 0 and not y.done]
        if live:
            bb = collate_ragged(Ragged.from_obs([o29[i] for i in live], np.full(len(live), MODE_RUN))).to("cpu")
            with torch.no_grad():
                lg = pol.forward_infer(bb)[0]
            for k, i in enumerate(live):
                acts[i] = int(lg[k, :o29[i].num_actions].argmax())
    n28, n29 = E28.step(acts), E29.step(acts)
    for i in range(a.envs):
        x, y = n28[i] if n28[i] is not None else o28[i], n29[i] if n29[i] is not None else o29[i]
        if y.done or y.num_actions == 0:
            x = E28.reset_some({i: req(i, t + 1, 28)})[i]; y = E29.reset_some({i: req(i, t + 1, 29)})[i]
        o28[i], o29[i] = x, y
nt, nc = np.concatenate(new_tok), np.concatenate(new_cand)
print(f"1. 锁步 {diff['frames']} 帧：旧列不同的帧 记录 {diff['tok']}、候选 {diff['cand']}；新列非零比例 记录 {np.mean(nt != 0):.4f}、候选 {np.mean(nc != 0):.4f}")

# 2. obs_v29 关与开
off = PolicyValueNet(ModelConfig(**mc)).eval(); load_state(off, st)
on = PolicyValueNet(ModelConfig(**{**mc, "obs_v29": True})).eval(); load_state(on, st)
pool = frames29[: 600]
b = collate_ragged(Ragged.from_obs(pool, np.full(len(pool), MODE_RUN))).to("cpu")
with torch.no_grad():
    l0, v0, _, _ = off.forward_all(b); l1, v1, _, _ = on.forward_all(b)
    i0, i1 = off.forward_infer(b), on.forward_infer(b)
eq = lambda x, y: bool(torch.equal(torch.nan_to_num(x, neginf=-9e9), torch.nan_to_num(y, neginf=-9e9)))
print(f"2. {len(pool)} 帧：obs_v29 关与开 logits {eq(l0, l1)} 价值 {eq(v0, v1)} 推理 logits {eq(i0[0], i1[0])} 推理价值 {eq(i0[1], i1[1])}")

# 3. 新列给随机值：梯度与输出变化
b2 = dict(b); tf = b["tok_feats"].clone(); cf = b["cand_feats"].clone()
tf[:, 22:30] = torch.randn_like(tf[:, 22:30]); cf[:, 14:20] = torch.randn_like(cf[:, 14:20]); b2["tok_feats"] = tf; b2["cand_feats"] = cf
on.train()
l, v, _, _ = on.forward_all(b2)
(l.float().log_softmax(-1).nan_to_num(neginf=0).sum() + v.float().sum()).backward()
gf, gc = on.feat_v29.weight.grad, on.cand_v29.weight.grad
print(f"3. 梯度 feat_v29 范数 {float(gf.norm()):.4g}（有限 {bool(torch.isfinite(gf).all())}）cand_v29 {float(gc.norm()):.4g}（有限 {bool(torch.isfinite(gc).all())}）")
with torch.no_grad():
    on.feat_v29.weight.normal_(0, 0.02); on.cand_v29.weight.normal_(0, 0.02); on.eval()
    l2, v2, _, _ = on.forward_all(b2)
print(f"   加权重后 logits 变化 {float((l2 - l0).nan_to_num().abs().max()):.4f}、价值变化 {float((v2 - v0).abs().max()):.4f}（应大于 0）")

# 4. 上线冲击：同一模型在格式 28 / 29 的配对帧上的差别
if a.impact:
    from collections import defaultdict
    from sts2env.protocol import DecisionKind
    net = PolicyValueNet(ModelConfig(**{**mc, "obs_v29": True})).eval(); load_state(net, st)
    pairs = [(x, y) for x, y in zip(frames28, frames29) if y.num_actions > 1 and not y.done]
    agg = defaultdict(lambda: [0, 0.0, 0, 0.0, 0.0])      # 帧数、KL 和、首选相同、价值差绝对值和、KL 最大
    for i in range(0, len(pairs), 256):
        ch = pairs[i:i + 256]
        b28 = collate_ragged(Ragged.from_obs([x for x, _ in ch], np.full(len(ch), MODE_RUN))).to("cpu")
        b29 = collate_ragged(Ragged.from_obs([y for _, y in ch], np.full(len(ch), MODE_RUN))).to("cpu")
        with torch.no_grad():
            l28, v28 = net.forward_infer(b28); l29, v29 = net.forward_infer(b29)
        lp28, lp29 = torch.log_softmax(l28.float(), -1), torch.log_softmax(l29.float(), -1)
        p28 = lp28.exp()
        kl = torch.where(p28 > 0, p28 * (lp28 - lp29), torch.zeros_like(p28)).sum(-1)
        same = (l28.argmax(-1) == l29.argmax(-1))
        for k, (x, y) in enumerate(ch):
            key = ("战斗" if y.in_combat else "局外") + ":" + DecisionKind(int(y.header[1])).name
            g = agg[key]; g[0] += 1; g[1] += float(kl[k]); g[2] += int(same[k]); g[3] += abs(float(v28[k] - v29[k])); g[4] = max(g[4], float(kl[k]))
    print(f"4. 上线冲击（{len(pairs)} 个多选帧；门槛：KL 平均 ≤ 0.005、首选一致 ≥ 98%）")
    tot = [0, 0.0, 0, 0.0]
    for key, (n, kls, sm, dv, kmax) in sorted(agg.items(), key=lambda kv: -kv[1][0]):
        ok = kls / n <= 0.005 and sm / n >= 0.98
        print(f"   {key:<28} {n:6d} 帧  KL 平均 {kls / n:.5f}（最大 {kmax:.3f}）  首选一致 {sm / n:.4f}  价值差 {dv / n:.4f}  {'过' if ok else '超门槛'}")
        tot[0] += n; tot[1] += kls; tot[2] += sm; tot[3] += dv
    if tot[0]:
        print(f"   合计 {tot[0]} 帧  KL 平均 {tot[1] / tot[0]:.5f}  首选一致 {tot[2] / tot[0]:.4f}  价值差 {tot[3] / tot[0]:.4f}")
