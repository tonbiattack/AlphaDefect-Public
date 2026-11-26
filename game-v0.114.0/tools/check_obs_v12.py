"""（2026-10-28，决策记录 164；用法：.venv/bin/python -I tools/check_obs_v12.py [检查点.pt]）观测格式 12 的本机核对：
1. 回合结束预演补上的能力、遗物效果：按固定场景打，每次结束回合时拿预演和实际结算比（敌人剩余生命、玩家掉血）；
2. 手牌计算型数值（弹幕的段数、编译驱动的抽牌数、同步的集中、伏特的引导数）与当前充能球对照；
3. 招式记录 15–17（Boss 给自己、给队友的力量和获得的格挡）与敌人回合实际的变化对照；
4. 给了检查点时：招式新列对输出的影响。用升级前（观测格式 11）的检查点时 feat_v11 这几列为 0，差应为 0；之后的检查点已在学这几列，差不为 0。"""
import collections
import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "python"))
from sts2env import Sts2Env, TokenKind  # noqa: E402
from sts2env.env import parse  # noqa: E402

rng = np.random.default_rng(0)
env = Sts2Env()
vocab = env.vocab()
vid = {k: i for i, k in enumerate(vocab)}
END, PLAY = 1, 2


def reset(req):
    env._proc.send(dict(req, op="reset"))
    return parse(env._proc.recv())


def step(a):
    env._proc.send({"op": "step", "action": int(a)})
    return parse(env._proc.recv())


def pick(o, prefer=(), p_play=0.7):
    """优先打 prefer 里的牌（按顺序），其次以 p_play 的概率随手打一张，否则结束回合；非战斗决策随机。"""
    k = o.cand_ints[: o.num_actions, 0]
    plays = np.flatnonzero(k == PLAY)
    for name in prefer:
        hit = [i for i in plays if o.tok_ints[o.cand_ints[i, 1], 1] == vid["CARD." + name]]
        if hit:
            return int(hit[0])
    if len(plays) and rng.random() < p_play:
        return int(rng.choice(plays))
    ends = np.flatnonzero(k == END)
    return int(ends[0]) if len(ends) else int(rng.integers(o.num_actions))


def creatures(o):
    """(词表 ID, 站位) → (记录下标, 特征)。"""
    t = np.flatnonzero(o.tok_ints[:, 0] == TokenKind.CREATURE)
    return {(int(o.tok_ints[i, 1]), int(o.tok_ints[i, 4])): (i, o.tok_feats[i]) for i in t}


def power(o, tok, name):
    m = (o.tok_ints[:, 0] == TokenKind.POWER) & (o.tok_ints[:, 3] == tok) & (o.tok_ints[:, 1] == vid.get("POWER." + name, -9))
    return float(o.tok_feats[m, 0].sum())


def turn(o):
    g = np.flatnonzero(o.tok_ints[:, 0] == TokenKind.GLOBAL)
    return o.tok_feats[g[0], 9] if len(g) else -1


def end_turns(spec, prefer=(), p_play=0.7, steps=500, privileged=False):
    """打一场，返回每次结束回合的（结束回合前一帧，下一个玩家回合的第一帧或结束帧）。"""
    o = reset({"mode": "combat", "spec": spec, **({"privileged": True} if privileged else {})})
    pairs, pending = [], None
    for _ in range(steps):
        if pending is not None and (o.done or (o.in_combat and turn(o) > turn(pending))):
            pairs.append((pending, o)); pending = None
        if o.done or o.num_actions == 0:
            break
        a = pick(o, prefer, p_play) if o.in_combat else int(rng.integers(o.num_actions))
        if o.in_combat and o.cand_ints[a, 0] == END:
            pending = o
        o = step(a)
    return pairs


def check_end_turn(tag, specs, prefer=(), p_play=0.7):
    """敌人：预计剩余生命 = 生命 − max(0, 被动伤害 − 格挡)；玩家：预计掉血（特征 17）。和下个玩家回合的实际值比。"""
    e_err, p_err, n_new = [], [], 0
    for spec in specs:
        for o, q in end_turns(spec, prefer, p_play):
            co, cq = creatures(o), creatures(q)
            for key, (i, f) in co.items():
                if f[5] > 0:
                    p_err.append(min(f[17], o.hp) - (o.hp - q.hp))     # 打死时实际掉血被剩余生命封顶
                    continue
                if q.done and not q.victory:
                    continue        # 玩家死了，看不到敌人的结果
                mv = (o.tok_ints[:, 0] == TokenKind.MOVE) & (o.tok_ints[:, 3] == i) & (o.tok_feats[:, 0] == 0)
                if (o.tok_feats[mv, 11] > 0).any():
                    continue        # 敌人这一招回血（知识恶魔的沉思等），生命变化不只来自回合结束
                pred = max(0.0, f[0] - max(0.0, f[16] - f[2]))
                _, g = cq.get(key, (None, None))
                actual = g[0] if g is not None else 0.0
                e_err.append(pred - actual); n_new += f[16] > 0
    e, p = np.abs(np.array(e_err)), np.abs(np.array(p_err))
    print(f"[{tag}] 敌人 {len(e)} 次（其中被动伤害 > 0 的 {n_new} 次）：剩余生命预演完全准确 {np.mean(e < 0.5):.1%}，平均误差 {e.mean() if len(e) else 0:.2f}；"
          f"玩家 {len(p)} 次：掉血预演完全准确 {np.mean(p < 0.5):.1%}，平均误差 {p.mean() if len(p) else 0:.2f}")
    return e, p


base = {"character": "DEFECT", "act": 2, "encounter": "AEONGLASS_BOSS", "hp": 999, "max_hp": 999}
deck = lambda *cs: [{"id": c} for c, n in cs for _ in range(n)]
seeds = [f"V12_{k}" for k in range(4)]

# 1. 回合结束预演
check_end_turn("对照：初始牌组", [dict(base, seed=s) for s in seeds])
check_end_turn("冰雹风暴", [dict(base, seed=s, deck=deck(("HAILSTORM", 2), ("COOLHEADED", 4), ("DEFEND_DEFECT", 4))) for s in seeds], ("HAILSTORM", "COOLHEADED"))
check_end_turn("吞噬暗影 + 雷霆", [dict(base, seed=s, deck=deck(("CONSUMING_SHADOW", 2), ("THUNDER", 1), ("DARKNESS", 3), ("ZAP", 3), ("COOLHEADED", 2))) for s in seeds],
               ("CONSUMING_SHADOW", "THUNDER", "DARKNESS", "ZAP", "COOLHEADED"))
check_end_turn("遗物（石历、斗篷扣、涟漪盆、尖叫酒壶、格挡盾、奥利哈钢）",
               [dict(base, seed=s, relics=["STONE_CALENDAR", "CLOAK_CLASP", "RIPPLE_BASIN", "SCREAMING_FLAGON", "PARRYING_SHIELD", "ORICHALCUM"]) for s in seeds * 2], p_play=0.6)
check_end_turn("知识恶魔（瓦解）", [dict(base, seed=s, act=1, encounter="KNOWLEDGE_DEMON_BOSS") for s in seeds * 2])

# 2. 手牌计算型数值
o = reset({"mode": "combat", "spec": dict(base, seed="V12_C", deck=deck(("BARRAGE", 3), ("COMPILE_DRIVER", 2), ("SYNCHRONIZE", 2), ("VOLTAIC", 2), ("ZAP", 3), ("COOLHEADED", 3)))})
seen = collections.defaultdict(set)
for _ in range(300):
    if o.done or o.num_actions == 0:
        break
    if o.in_combat:
        orbs = o.tok_ints[o.tok_ints[:, 0] == TokenKind.ORB, 1]
        hand = np.flatnonzero((o.tok_ints[:, 0] == TokenKind.CARD) & (o.tok_ints[:, 2] == 3))
        for i in hand:
            n = vocab[o.tok_ints[i, 1]].split(".")[1]
            if n in ("BARRAGE", "COMPILE_DRIVER", "SYNCHRONIZE", "VOLTAIC"):
                seen[n].add((len(orbs), len(set(orbs.tolist())), float(o.tok_feats[i, 7]), float(o.tok_feats[i, 8])))
    o = step(pick(o, ("ZAP", "COOLHEADED"), 0.5) if o.in_combat else int(rng.integers(o.num_actions)))
for n, v in seen.items():
    print(f"[计算型数值] {n}：（球数, 球种类数, 特征 7, 特征 8）{sorted(v)[:8]}")

# 3. 招式记录 15–17
stat = collections.defaultdict(lambda: [0, 0, []])
for enc, act in (("AEONGLASS_BOSS", 2), ("QUEEN_BOSS", 2), ("TEST_SUBJECT_BOSS", 2), ("KNOWLEDGE_DEMON_BOSS", 1), ("THE_INSATIABLE_BOSS", 1), ("KAISER_CRAB_BOSS", 1)):
    for s in seeds:
        for o, q in end_turns(dict(base, seed=s, act=act, encounter=enc), p_play=0.5, steps=800):
            if q.done:
                continue
            co, cq = creatures(o), creatures(q)
            mv = np.flatnonzero((o.tok_ints[:, 0] == TokenKind.MOVE) & (o.tok_feats[:, 0] == 0))
            # 每只怪这回合实际涨的力量 = 自己这一招给自己的 + 其他怪这一招给队友的
            own_of = {t: int(o.tok_ints[t, 3]) for t in mv}
            ally_from = {own: o.tok_feats[t, 16] for t, own in own_of.items()}
            for t, own in own_of.items():
                key = (int(o.tok_ints[own, 1]), int(o.tok_ints[own, 4]))
                if key not in cq:
                    continue
                f = o.tok_feats[t]; qi, g = cq[key]
                ds = power(q, qi, "STRENGTH_POWER") - power(o, own, "STRENGTH_POWER")
                pred = f[15] + sum(v for k, v in ally_from.items() if k != own)
                name = vocab[o.tok_ints[t, 1]]
                st = stat[name]; st[0] += 1
                ok = abs(pred - ds) < 0.5 and abs(f[17] - g[2]) < 0.5
                st[1] += ok
                if not ok and len(st[2]) < 2:
                    st[2].append(f"预演 力量 {pred:.0f}（自己 {f[15]:.0f}）格挡 {f[17]:.0f}；实际 力量 {ds:.0f} 格挡 {g[2]:.0f}")
for name, (n, ok, bad) in sorted(stat.items()):
    print(f"[招式增益] {name:60s} {ok}/{n} 一致 {bad}")

# 4. 模型：招式新列不影响现有检查点的输出
if len(sys.argv) > 1:
    import copy
    import torch
    from sts2rl.features import Ragged, collate_ragged
    from sts2rl.model import ModelConfig, PolicyValueNet, load_state
    blob = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
    net = PolicyValueNet(ModelConfig(**blob["config"]["model"])).eval()
    load_state(net, blob["state"])
    print(f"[模型] 第 {blob.get('version')} 版（观测格式 {blob.get('obs_format')}），feat_v11 第 15–17 列范数 {net.feat_v11.weight[:, 15:18].norm().item():.2e}")
    sample = [p for o, _ in end_turns(dict(base, seed="V12_M", encounter="QUEEN_BOSS"), p_play=0.5, steps=600) for p in [o]][:128]
    zero = []
    for o in sample:
        q = copy.copy(o); tf = o.tok_feats.copy(); tf[o.tok_ints[:, 0] == TokenKind.MOVE, 15:18] = 0; q.tok_feats = tf; zero.append(q)
    with torch.no_grad():
        modes = np.zeros(len(sample), np.int64)
        l0, v0, _ = net.forward_phi(collate_ragged(Ragged.from_obs(sample, modes)).to("cpu"))
        l1, v1, _ = net.forward_phi(collate_ragged(Ragged.from_obs(zero, modes)).to("cpu"))
    fin = torch.isfinite(l0)
    print(f"[模型] 招式 15–17 清零与否：策略对数几率最大差 {(l0 - l1)[fin].abs().max().item():.2e}，价值最大差 {(v0 - v1).abs().max().item():.2e}（升级前的检查点应为 0）")
env.close()
