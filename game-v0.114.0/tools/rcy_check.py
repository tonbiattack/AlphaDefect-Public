# 用法：.venv/bin/python -I tools/rcy_check.py <检查点或评测留档的权重>（在仓库根目录跑，CPU 即可）
# 回灌（第 26 条）本机核对：①用真实权重，打开回灌后 forward_all / forward_infer 与不回灌逐位相同；②反向时 rcy_ 各层梯度有限且非零；
# ③rcy_ 各层随便给一点权重后输出确实变化，第二遍对第一遍的 KL > 0。观测由本机引擎随机走出（整局 + 战斗练习各一些）
import sys, copy, json
sys.path.insert(0, "python")
import numpy as np, torch
from sts2env import VecEnv
from sts2rl.model import ModelConfig, PolicyValueNet, load_state
from sts2rl.features import Ragged, collate_ragged
from sts2rl.actor import MODE_RUN
torch.manual_seed(0)
W = sys.argv[1]
blob = torch.load(W, map_location="cpu", weights_only=False)
if "config" in blob:
    mc, st = dict(blob["config"]["model"]), blob["state"]
else:       # 评测留档的裸权重：网络配置取训练配置
    mc, st = dict(json.load(open("tools/remote/ironclad/train_config.json"))["model"]), blob
base = PolicyValueNet(ModelConfig(**mc)).eval(); load_state(base, st)
rc = PolicyValueNet(ModelConfig(**{**mc, "recycle": True})).eval(); load_state(rc, st)
# 观测
env = VecEnv(6); rng = np.random.default_rng(1)
reqs = [{"op": "reset", "mode": "run", "character": "IRONCLAD", "ascension": 10, "seed": f"RC{i}", "privileged": True, "route_summary": True, "future_content": True} for i in range(6)]
obs = env.reset(reqs); pool = []
for t in range(160):
    acts = [int(rng.integers(o.num_actions)) if o.num_actions > 0 and not o.done else None for o in obs]
    pool += [o for o in obs if o.num_actions > 0 and not o.done]
    new = env.step(acts)
    obs = [n if n is not None else o for n, o in zip(new, obs)]
    for i, o in enumerate(obs):
        if o.done or o.num_actions == 0:
            obs[i] = env.reset_some({i: {**reqs[i], "seed": f"RC{i}_{t}"}})[i]
print("观测帧", len(pool), "其中战斗", sum(o.in_combat for o in pool))
b = collate_ragged(Ragged.from_obs(pool, np.full(len(pool), MODE_RUN))).to("cpu")
with torch.no_grad():
    l0, v0, p0, x0 = base.forward_all(b); l1, v1, p1, x1 = rc.forward_all(b)
    i0 = base.forward_infer(b); i1 = rc.forward_infer(b)
eq = lambda a, c: bool(torch.equal(torch.nan_to_num(a, neginf=-9e9), torch.nan_to_num(c, neginf=-9e9)))
print("① 逐位相同：logits", eq(l0, l1), "价值", eq(v0, v1), "打分", eq(p0, p1), "推理 logits", eq(i0[0], i1[0]), "推理价值", eq(i0[1], i1[1]),
      "第一遍=第二遍", eq(x1["rcy_l1"], l1))
# ② 反向
rc.train()
l, v, _, ex = rc.forward_all(b)
dist = torch.distributions.Categorical(logits=l.float(), validate_args=False)
loss = -dist.entropy().mean() + v.float().pow(2).mean() - l.float().log_softmax(-1).gather(1, dist.sample()[:, None]).mean()
loss.backward()
for n in ("rcy_a", "rcy_b", "rcy_c", "rcy_p"):
    g = getattr(rc, n).weight.grad
    print("② 梯度", n, "有限" if torch.isfinite(g).all() else "非有限", f"范数 {float(g.norm()):.4g}")
print("   主干梯度有限", all(torch.isfinite(p.grad).all() for p in rc.parameters() if p.grad is not None))
# ③ 给 rcy_ 一点权重
rc.eval()
with torch.no_grad():
    for n in ("rcy_a", "rcy_b", "rcy_c", "rcy_p"):
        getattr(rc, n).weight.normal_(0, 0.05)
    l2, v2, _, x2 = rc.forward_all(b)
    lp2, lp1 = l2.float().log_softmax(-1), x2["rcy_l1"].float().log_softmax(-1); p2 = lp2.exp()
    kl = torch.where(p2 > 0, p2 * (lp2 - lp1), torch.zeros_like(p2)).sum(-1).mean()
    i2 = rc.forward_infer(b)
print("③ 加权重后：KL(第二遍‖第一遍)", f"{float(kl):.4f}", "首选相同", f"{float((l2.argmax(-1) == x2['rcy_l1'].argmax(-1)).float().mean()):.3f}",
      "价值变化", f"{float((v2 - v0).abs().mean()):.4f}", "推理与训练前向一致", eq(i2[0], l2), f"{float((i2[1] - v2).abs().max()):.2e}")
