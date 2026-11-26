"""（2026-10-28，决策记录 161；用法：.venv/bin/python -I tools/check_obs_v11.py 检查点.pt）观测格式 11 的本机核对：新特征、特权闸门、抽签推演准确率、模型的上线等价与策略不依赖特权。"""
import sys, collections
import numpy as np
import torch
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "python"))
from sts2env import Sts2Env, TokenKind, Zone
from sts2env.env import parse
from sts2rl.features import Ragged, collate_ragged
from sts2rl.model import ModelConfig, PolicyValueNet, load_state

rng = np.random.default_rng(0)
env = Sts2Env()
vocab = env.vocab()


def reset(req):
    env._proc.send(dict(req, op="reset"))
    return parse(env._proc.recv())


def pick(o):
    """随手打：优先出牌，其次结束回合；局外随机。"""
    k = o.cand_ints[: o.num_actions, 0]
    plays = np.flatnonzero(k == 2)
    if len(plays) and rng.random() < 0.8:
        return int(rng.choice(plays))
    return int(rng.integers(o.num_actions))


def play(req, steps=400):
    out = []
    o = reset(req)
    for _ in range(steps):
        out.append(o)
        if o.done or o.num_actions == 0:
            break
        o = parse_step(pick(o))
    return out


def parse_step(a):
    env._proc.send({"op": "step", "action": int(a)})
    return parse(env._proc.recv())


def summary(obs, tag):
    ti = lambda o: o.tok_ints
    pw2 = collections.Counter(); stats = []; forms = []; status = 0; selfdmg = []; priv_nz = 0
    for o in obs:
        priv_nz += int(np.abs(o.tok_priv).sum() > 0)
        for t in range(len(o.tok_ints)):
            kd, vid = int(o.tok_ints[t, 0]), int(o.tok_ints[t, 1])
            f = o.tok_feats[t]
            if kd == TokenKind.POWER and f[2] != f[0]:
                pw2[vocab[vid]] += 1
            if kd == TokenKind.COMBAT_STAT:
                (stats if f[0] == 0 else forms).append(f[:13].round(1).tolist())
            if kd == TokenKind.MOVE and f[14] > 0:
                status += 1
            if kd == TokenKind.CREATURE and f[5] > 0 and f[18] > 0:
                selfdmg.append(f[16:20].round(1).tolist())
    print(f"[{tag}] 帧 {len(obs)}，特权非零的帧 {priv_nz}；界面数值≠层数的能力 {dict(pw2)}；带塞牌张数的招式记录 {status}")
    print(f"   战斗计数样例 {stats[-1] if stats else None}；形态记录 {forms[:1]} … {forms[-1:] if forms else None}；玩家手牌自伤（16–19）样例 {selfdmg[:3]}")


# 1. 永恒玻璃：不带 / 带特权
spec = {"act": 2, "encounter": "AEONGLASS_BOSS", "hp": 300, "max_hp": 300}
a0 = play({"mode": "combat", "spec": dict(spec, seed="C1")}); summary(a0, "永恒玻璃 不带特权")
assert all(np.abs(o.tok_priv).sum() == 0 for o in a0), "不带特权时出现了非零特权特征！"
a1 = play({"mode": "combat", "spec": dict(spec, seed="C1"), "privileged": True}); summary(a1, "永恒玻璃 带特权")
# 2. 实验体
t1 = play({"mode": "combat", "spec": {"act": 2, "encounter": "TEST_SUBJECT_BOSS", "hp": 400, "max_hp": 400, "seed": "C2"}, "privileged": True}, 600); summary(t1, "实验体")
# 3. 整局：预抽遭遇、抽签推演准确率
hit = tot = 0; enc = collections.Counter(); runs_obs = []
for k in range(6):
    obs = play({"mode": "run", "character": "DEFECT", "ascension": 10, "seed": f"V11_{k}", "privileged": True, "route_summary": True}, 1500)
    runs_obs += obs
    for o in obs:
        g = np.flatnonzero(o.tok_ints[:, 0] == TokenKind.GLOBAL)
        if len(g):
            p = o.tok_priv[g[0]]
            enc[(vocab[int(p[0])] if p[0] else "-", vocab[int(p[1])] if p[1] else "-")] += 1
    # 推演：某帧标出的第 1 步抽签结果 vs 下一个玩家回合的当前招式（同一只怪 = 同一个所属记录下标不可靠，按怪物词表 ID + 站位近似）
    for i, o in enumerate(obs[:-1]):
        mv = np.flatnonzero((o.tok_ints[:, 0] == TokenKind.MOVE) & (o.tok_priv[:, 1] > 0) & (o.tok_priv[:, 0] > 0.5))
        if not len(mv):
            continue
        turn = o.tok_feats[np.flatnonzero(o.tok_ints[:, 0] == TokenKind.GLOBAL)[0], 9]
        for j in range(i + 1, len(obs)):
            q = obs[j]
            if not q.in_combat:
                break
            gq = np.flatnonzero(q.tok_ints[:, 0] == TokenKind.GLOBAL)[0]
            if q.tok_feats[gq, 9] == turn + 1:
                cur = {(int(q.tok_ints[q.tok_ints[t, 3], 1]), int(q.tok_ints[q.tok_ints[t, 3], 4])): int(q.tok_ints[t, 1])
                       for t in np.flatnonzero((q.tok_ints[:, 0] == TokenKind.MOVE) & (q.tok_feats[:, 0] == 0))}
                for t in mv:
                    key = (int(o.tok_ints[o.tok_ints[t, 3], 1]), int(o.tok_ints[o.tok_ints[t, 3], 4]))
                    if key in cur:
                        tot += 1; hit += cur[key] == int(o.tok_ints[t, 1])
                break
        # 每个回合只比一次
print(f"[整局] 帧 {len(runs_obs)}；预抽遭遇样例 {enc.most_common(3)}；抽签推演 {hit}/{tot} 命中")
summary(runs_obs, "整局")

# 4. 模型：载入现有检查点（版本 10 训练的），核对上线等价与策略不依赖特权
blob = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
net = PolicyValueNet(ModelConfig(**blob["config"]["model"])).eval()
load_state(net, blob["state"])
with torch.no_grad():
    net.kind_emb.weight[int(TokenKind.COMBAT_STAT)].zero_()
sample = [o for o in a1 + t1 + runs_obs if o.num_actions > 0][::7][:256]
modes = np.array([0 if any(o is x for x in a1 + t1) else 1 for o in sample])
def run_net(obs_list):
    with torch.no_grad():
        lg, v, _ = net.forward_phi(collate_ragged(Ragged.from_obs(obs_list, modes)).to("cpu"))
    return lg, v
lg0, v0 = run_net(sample)
# 特权打乱
import copy
shuf = []
for o in sample:
    q = copy.copy(o); q.tok_priv = rng.normal(size=o.tok_priv.shape).astype(np.float32) * 5; shuf.append(q)
lg1, v1 = run_net(shuf)
fin = torch.isfinite(lg0)
print(f"[模型] 特权打乱后策略对数几率最大差 {(lg0 - lg1)[fin].abs().max().item():.2e}（应为 0），价值最大差 {(v0 - v1).abs().max().item():.2e}（价值头新层清零，应为 0）")
# 新位置清零（模拟版本 10 的输入）
old = []
for o in sample:
    q = copy.copy(o); tf = o.tok_feats.copy(); k = o.tok_ints[:, 0]
    tf[k == TokenKind.POWER, 2] = 0; tf[k == TokenKind.MOVE, 14] = 0
    pl = (k == TokenKind.CREATURE) & (tf[:, 5] > 0); tf[pl, 18] = 0; tf[pl, 19] = 0
    keep = k != TokenKind.COMBAT_STAT
    q.tok_feats = tf[keep]; q.tok_ints = o.tok_ints[keep].copy(); q.tok_priv = o.tok_priv[keep]
    # 删掉记录后下标要重排：候选与所属记录引用的下标
    remap = np.cumsum(keep) - 1
    own = q.tok_ints[:, 3]; q.tok_ints[:, 3] = np.where(own >= 0, remap[np.clip(own, 0, None)], -1)
    ci = o.cand_ints.copy()
    for c in (1, 2):
        ci[:, c] = np.where(ci[:, c] >= 0, remap[np.clip(ci[:, c], 0, None)], -1)
    q.cand_ints = ci; old.append(q)
lg2, v2 = run_net(old)
print(f"[模型] 版本 10 输入 vs 版本 11 输入：策略对数几率最大差 {(lg0 - lg2)[fin].abs().max().item():.2e}，价值最大差 {(v0 - v2).abs().max().item():.2e}（玩家特征 17 含义改了，有手牌自伤的帧会不同）")
env.close()
