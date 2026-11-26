"""分类价值头（HL-Gauss）与战斗短程辅助头（决策记录 261）的上线前检验，本机 CPU。

一、标签（actor.combat_aux_labels）：用检查点贪心打几局整局（带 privileged 与 future_content），逐局核对
  1. 所有战斗中的步都有标签，打输只出现在输掉的那一局的最后一场；
  2. 每场战斗最后一步的“剩余掉血”等于（这一步血量 − 下一帧血量）/ 最大生命，输掉那场按结束血量 0；
  3. 按采集端的规则在选路线帧处切段（每段至少 --seg 步），各段分别算出的标签拼起来与整局一次算出的完全相同。
二、软标签（train.hl_target）：每行和为 1；范围内的回报，软标签的期望与回报相差不到半个桶宽；范围外的截到两端。
三、模型：同一检查点分别载入原配置与打开 future_content、value_hl、combat_aux 的配置，
  1. hl_beta 为 0 时策略对数几率与价值逐位相同；
  2. hl_beta 为 1 时策略不变，价值等于分类头的期望（输出层清零时为各桶中心的平均）；
  3. 分类价值头的隐层从数值价值头复制，反向时分类头、辅助头和编码器都有梯度。

    .venv/bin/python -I tools/hl_aux_check.py --dll 新引擎/Sts2Headless.dll --ckpt 检查点.pt [--games 6] [--steps 3000]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

from sts2env.env import _Process  # noqa: E402
from sts2env.protocol import DecisionKind, parse  # noqa: E402

fails: list[str] = []


def check(ok: bool, msg: str) -> bool:
    if not ok:
        fails.append(msg)
        print("  不通过：", msg, flush=True)
    return ok


def load(ckpt: str, **over):
    import torch
    from sts2rl.model import ModelConfig, PolicyValueNet, load_state
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    net = PolicyValueNet(ModelConfig(**{**blob["config"]["model"], **over})).eval()
    load_state(net, blob["state"])
    return net


def play(proc: _Process, seed: str, steps: int, pol) -> tuple[list, object]:
    """贪心打一局，返回（决策帧列表，终局帧）。"""
    proc.send({"op": "reset", "mode": "run", "character": "DEFECT", "ascension": 10, "seed": seed, "privileged": True,
               "route_summary": True, "future_content": True})
    o = parse(proc.recv()); obs = []
    for _ in range(steps):
        if o.done or o.stuck or o.num_actions == 0:
            break
        obs.append(o)
        proc.send({"op": "step", "action": pol(o)}); o = parse(proc.recv())
    return obs, o


def label_check(games: list[tuple[list, object]], seg: int) -> None:
    from sts2rl.actor import combat_aux_labels
    n_steps = n_comb = n_lab = n_lost = 0
    for gi, (obs, final) in enumerate(games):
        m, win, hp = combat_aux_labels(obs, final)
        inc = np.array([bool(o.in_combat) for o in obs])
        n_steps += len(obs); n_comb += int(inc.sum()); n_lab += int(m.sum())
        lost_game = bool(final.done) and not bool(final.victory) and not final.stuck
        check(not final.stuck and bool((m == inc).all()), f"第 {gi} 局：有战斗步没有标签或非战斗步有标签（卡住={final.stuck}）")
        # 场次：连续战斗步、层数或第几场不变
        blocks, i = [], 0
        while i < len(obs):
            if not inc[i]:
                i += 1; continue
            j = i
            while j + 1 < len(obs) and inc[j + 1] and (obs[j + 1].floor, int(obs[j + 1].header[23])) == (obs[i].floor, int(obs[i].header[23])):
                j += 1
            blocks.append((i, j)); i = j + 1
        lost = [b for b in blocks if win[b[0]] == 0]
        n_lost += len(lost)
        died_in_combat = lost_game and bool(inc[-1])
        check(len(lost) == int(died_in_combat) and (not lost or lost[0] == blocks[-1]),
              f"第 {gi} 局：打输的场次 {lost}，应只有最后一场且仅当死在战斗里（共 {len(blocks)} 场）")
        for i, j in blocks:
            nxt = obs[j + 1] if j + 1 < len(obs) else final
            end = 0 if (win[j] == 0) else nxt.hp
            exp = np.clip((obs[j].hp - end) / max(obs[j].max_hp, 1), -1, 1)
            check(abs(hp[j] - exp) < 1e-6, f"第 {gi} 局第 {i}–{j} 步：剩余掉血 {hp[j]:.4f}，应为 {exp:.4f}")
            check(bool((win[i:j + 1] == win[i]).all()), f"第 {gi} 局第 {i}–{j} 步：同一场的输赢标签不一致")
        # 分段：采集端在选路线帧、段长至少 seg 步时送出前面的部分，after 为这个选路线帧
        cuts, s0 = [], 0
        for k, o in enumerate(obs):
            if o.decision == DecisionKind.MAP and k - s0 >= seg:
                cuts.append(k); s0 = k
        parts, s0 = [], 0
        for k in cuts + [len(obs)]:
            parts.append(combat_aux_labels(obs[s0:k], obs[k] if k < len(obs) else final)); s0 = k
        cat = [np.concatenate([p[x] for p in parts]) for x in range(3)]
        check(all(np.array_equal(a, b) for a, b in zip(cat, (m, win, hp))), f"第 {gi} 局：分 {len(parts)} 段算的标签与整局不同")
    print(f"标签：{len(games)} 局 {n_steps} 步，战斗步 {n_comb}，有标签 {n_lab}，打输的场 {n_lost}", flush=True)


def target_check() -> None:
    import torch
    from sts2rl.train import hl_target
    edges = torch.linspace(-2.5, 8.0, 129)
    r = torch.tensor([-5.0, -2.5, -1.0, 0.0, 0.37, 1.0, 2.01, 4.5, 7.9, 8.0, 20.0])
    t = hl_target(r, edges, 0.75)
    centers = (edges[:-1] + edges[1:]) / 2
    check(bool(torch.allclose(t.sum(-1), torch.ones(len(r)), atol=1e-5)), "软标签每行和不为 1")
    mean = t @ centers
    inside = (r > -2.3) & (r < 7.8)
    w = float(edges[1] - edges[0])
    check(bool(((mean - r).abs()[inside] < w / 2).all()), f"范围内软标签的期望偏离回报超过半桶：{(mean - r)[inside].tolist()}")
    check(bool((mean[r <= -2.5] < -2.3).all() and (mean[r >= 8.0] > 7.8).all()), "范围外的回报没有截到两端")
    print(f"软标签：桶宽 {w:.4f}，范围内期望最大偏差 {float((mean - r).abs()[inside].max()):.5f}", flush=True)


def model_check(ckpt: str, games: list[tuple[list, object]]) -> None:
    import torch
    import torch.nn.functional as F
    from sts2rl.actor import MODE_RUN
    from sts2rl.features import Ragged, collate_ragged
    from sts2rl.train import hl_target
    base = load(ckpt)
    new = load(ckpt, future_content=True, value_hl=True, combat_aux=True)
    obs = [o for g, _ in games for o in g]
    rng = np.random.default_rng(0)
    pick = sorted(rng.choice(len(obs), min(600, len(obs)), replace=False))
    sample = [obs[i] for i in pick]
    b = collate_ragged(Ragged.from_obs(sample, np.full(len(sample), MODE_RUN))).to("cpu")
    with torch.no_grad():
        l0, v0 = base(b)
        l1, v1 = new(b)
        same = lambda x, y: bool(torch.equal(torch.nan_to_num(x, neginf=-1e30), torch.nan_to_num(y, neginf=-1e30)))
        check(same(l0, l1), f"hl_beta 0：策略输出不同，最大差 {float((l0 - l1).nan_to_num().abs().max())}")
        check(torch.equal(v0, v1), f"hl_beta 0：价值不同，最大差 {float((v0 - v1).abs().max())}")
        new.hl_beta.fill_(1.0)
        l2, v2 = new(b)
        check(same(l0, l2), "hl_beta 1：策略输出变了")
        exp = float(new.hl_centers.mean())
        check(bool(torch.allclose(v2, torch.full_like(v2, exp), atol=1e-5)), f"hl_beta 1：价值应为桶中心平均 {exp:.4f}，实际 {v2[:4].tolist()}")
        new.hl_beta.fill_(0.0)
        for a, c in (("value_hl.0", "value_mlp.0"), ("value_hl_m.0", "value_mlp_m.0")):
            pa = dict(new.named_parameters())
            check(torch.equal(pa[a + ".weight"], pa[c + ".weight"]) and torch.equal(pa[a + ".bias"], pa[c + ".bias"]), f"{a} 没有从 {c} 复制")
    # 反向：分类头、辅助头、编码器都有梯度
    new.train()
    _, v, _, ex = new.forward_all(b)
    R = torch.as_tensor(rng.normal(1.0, 1.0, len(sample)), dtype=torch.float32)
    l_hl = -(hl_target(R, new.hl_edges, 0.75) * F.log_softmax(ex["hl_logits"], -1)).sum(-1).mean()
    a = ex["aux"]
    loss = l_hl + F.binary_cross_entropy_with_logits(a[:, 0], torch.ones(len(sample))) + (a[:, 1] ** 2).mean()
    loss.backward()
    g = {n: (p.grad is not None and float(p.grad.abs().sum()) > 0) for n, p in new.named_parameters()}
    for n in ("value_hl.2.weight", "value_hl_m.2.weight", "value_aux.2.weight", "value_aux.0.weight", "encoder.layers.0.linear1.weight"):
        check(g.get(n, False), f"{n} 没有梯度")
    check(not g.get("cand_mlp.0.weight", False), "策略打分头不该从这几项损失拿到梯度")
    # 只读主干（hl_detach，决策记录 261 补记二）：只有交叉熵时分类头有梯度、编码器没有；辅助头仍传回编码器
    det = load(ckpt, future_content=True, value_hl=True, combat_aux=True, hl_detach=True).train()
    _, _, _, ex = det.forward_all(b)
    (-(hl_target(R, det.hl_edges, 0.75) * F.log_softmax(ex["hl_logits"], -1)).sum(-1).mean()).backward(retain_graph=True)
    gd = {n: (p.grad is not None and float(p.grad.abs().sum()) > 0) for n, p in det.named_parameters()}
    check(gd.get("value_hl.2.weight", False), "只读主干：分类头输出层没有梯度（隐层第一步没有梯度是因为输出层清零）")
    check(not any(v for n, v in gd.items() if n.startswith(("encoder.", "id_emb", "value_mlp", "priv_", "fut_"))), "只读主干：交叉熵传回了编码器或数值头")
    det.zero_grad(set_to_none=True)
    F.binary_cross_entropy_with_logits(ex["aux"][:, 0], torch.ones(len(sample))).backward()
    check(any(p.grad is not None and float(p.grad.abs().sum()) > 0 for n, p in det.named_parameters() if n.startswith("encoder.")), "只读主干：辅助头的梯度也没传回编码器")
    n_macro = int(b["macro"].sum())
    print(f"模型：{len(sample)} 帧（局外 {n_macro}），新参数 {sum(p.numel() for n, p in new.named_parameters() if n.startswith(('value_hl', 'value_aux'))):,} 个", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dll", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--games", type=int, default=6)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seg", type=int, default=64)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(4)
    from sts2rl.actor import MODE_RUN
    from sts2rl.features import Ragged, collate_ragged
    pol_net = load(a.ckpt)

    def pol(o) -> int:
        with torch.no_grad():
            lg, _ = pol_net(collate_ragged(Ragged.from_obs([o], np.full(1, MODE_RUN))).to("cpu"))
        return int(lg[0, :o.num_actions].argmax())
    proc = _Process(Path(a.dll), open("/dev/null", "wb"))
    games = []
    try:
        for i in range(a.games):
            obs, final = play(proc, f"HLAUX{i:04d}", a.steps, pol)
            games.append((obs, final))
            print(f"  第 {i} 局：{len(obs)} 步，第 {final.floor} 层，{'通关' if final.victory else '卡住' if final.stuck else '输'}", flush=True)
    finally:
        proc.close() if hasattr(proc, "close") else None
    label_check(games, a.seg)
    target_check()
    model_check(a.ckpt, games)
    print("全部通过" if not fails else f"共 {len(fails)} 项不通过", flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
