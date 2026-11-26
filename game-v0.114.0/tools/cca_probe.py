"""反事实信用分配（Counterfactual Credit Assignment，Mesnard 等 2021）的离线验证（决策记录 284）：
能不能从“这一步之后实际发生的未来”里学出一个与这一步的选择无关的运气摘要 Φ，拿它当基线把运气从局外优势里减掉。

一、采集（collect）：用给定权重按概率出手打整局，记下每一步主干的全局表示 g（半精度）、是否局外、是否战斗后的第一步；
    局外多选步另记候选（cand_ints）、策略的对数概率与选的动作；每局的结局（是否通关）。每 --chunk 局一份 pickle。
二、拟合（fit），按局分 80% / 20%：
    1. 普通价值 V(g_t)：只看当前，拟合结局（±1）——对照；
    2. 事后摘要 Φ_t：对每局“局外步与战斗后第一步”组成的短序列反向跑一层 GRU，Φ_t 为 t 之后（不含 t）的摘要；
       事后价值 V(g_t, Φ_t) 拟合结局；
    3. 独立性：分类器 q(a | g_t, Φ_t) = 策略对数几率 + 候选嵌入 · MLP(g_t, Φ_t)，按最大似然训练（Φ 不回传）；
       Φ 的编码器同时最小化事后价值误差与 −β × 分类器的交叉熵（梯度反转：让 Φ 帮不了分类器猜动作）。
    报告留出对局：两种价值的解释方差、优势方差（结局 − 基线）之比、分类器比策略多出的对数似然（每步，接近 0 说明 Φ 与动作无关、基线无偏）。

    python -I tools/cca_probe.py collect --weights 权重.pt --config train_config.json --runs 3000 --out 目录 [--envs 48] [--dll 引擎]
    python -I tools/cca_probe.py fit --data 目录 [--beta 1.0] [--epochs 8]
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

RESET = {"op": "reset", "mode": "run", "character": "DEFECT", "ascension": 10, "route_summary": True, "snapshots": True, "privileged": True}


def collect(a) -> None:
    import torch
    from sts2env.env import VecEnv
    from sts2rl.actor import MODE_RUN
    from sts2rl.features import Ragged, collate_ragged
    from sts2rl.model import ModelConfig, PolicyValueNet, load_state
    dev = torch.device(a.device)
    mc = dict(json.loads(Path(a.config).read_text(encoding="utf-8"))["model"]); mc["oracle"] = False
    net = PolicyValueNet(ModelConfig(**mc)).to(dev).eval()
    blob = torch.load(a.weights, map_location="cpu", weights_only=False); load_state(net, blob["state"])
    if mc.get("future_content"):
        RESET["future_content"] = True
    grab = {}
    enc = net._encode
    def _enc(b):        # 截下主干的全局表示 g，不必再算一遍编码器
        out = enc(b); grab["g"] = out[2]; return out
    net._encode = _enc
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    env = VecEnv(a.envs, **({"dll": Path(a.dll)} if a.dll else {}))
    rng = np.random.default_rng(a.seed)
    started = done = 0
    slot: dict[int, dict] = {}
    chunk: list[dict] = []
    t0 = time.time()

    def start(i: int) -> None:
        nonlocal started
        if started >= a.runs:
            slot.pop(i, None); return
        slot[i] = {"g": [], "macro": [], "post": [], "act": [], "dec": [], "cands": [], "logits": [], "in_combat_prev": False, "key_prev": None}
        env.send(i, {**RESET, "seed": f"{a.prefix}{started:06d}"})
        started += 1

    def flush() -> None:
        if chunk:
            k = len(list(out.glob("chunk_*.pkl")))
            pickle.dump(chunk, open(out / f"chunk_{k:03d}.pkl", "wb"))
            chunk.clear()

    for i in range(a.envs):
        start(i)
    obs: dict[int, object] = {}
    while slot:
        for i, o in env.recv_ready(timeout=1.0):
            st = slot.get(i)
            if st is None:
                continue
            if isinstance(o, Exception) or o.done or o.stuck or len(st["g"]) > a.max_steps:      # 超过步数上限按卡住丢掉（原地打转的局）
                if not isinstance(o, Exception) and o.done and not o.stuck and st["g"]:
                    chunk.append({"g": np.stack(st["g"]).astype(np.float16), "macro": np.array(st["macro"]), "post": np.array(st["post"]),
                                  "act": np.array(st["act"]), "dec": st["dec"], "cands": st["cands"], "logits": st["logits"],
                                  "win": int(bool(o.victory)), "floor": int(o.floor)})
                    done += 1
                    if done % a.chunk == 0:
                        flush(); print(f"{time.strftime('%H:%M:%S')} {done} 局，{(time.time() - t0) / 60:.1f} 分钟", flush=True)
                start(i); continue
            obs[i] = o
        if not obs:
            continue
        ids = sorted(obs)
        with torch.no_grad():
            lg = net(collate_ragged(Ragged.from_obs([obs[i] for i in ids], np.full(len(ids), MODE_RUN))).to(dev))[0].float()
            g = grab["g"].float().cpu().numpy()
        for k, i in enumerate(ids):
            o, st = obs[i], slot[i]
            na = o.num_actions
            x = lg[k, :na]
            p = torch.softmax(x / a.temp, -1).cpu().numpy().astype(np.float64)
            act = int(rng.choice(na, p=p / p.sum()))
            inc = bool(o.in_combat)
            key = (int(o.floor), int(o.header[23]) if len(o.header) > 23 else 0)
            post = (not inc) and st["in_combat_prev"] or (inc and st["in_combat_prev"] and key != st["key_prev"])
            macro = not inc
            st["g"].append(g[k]); st["macro"].append(macro); st["post"].append(bool(post)); st["act"].append(act)
            st["in_combat_prev"], st["key_prev"] = inc, key
            if macro and na > 1:        # 局外多选步：候选、策略的对数几率
                st["dec"].append(len(st["g"]) - 1)
                st["cands"].append(np.asarray(o.cand_ints[:na], dtype=np.int32))
                st["logits"].append(torch.log_softmax(x, -1).cpu().numpy().astype(np.float32))
            env.send_step(i, act)
        obs.clear()
    flush()
    print(f"采集完：{done} 局", flush=True)


def fit(a) -> None:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    dev = torch.device(a.device)
    runs = [r for f in sorted(Path(a.data).glob("chunk_*.pkl")) for r in pickle.load(open(f, "rb"))]
    rr = np.random.default_rng(0); order = rr.permutation(len(runs))
    test_set = set(order[: len(runs) // 5].tolist())
    d = runs[0]["g"].shape[1]
    print(f"局 {len(runs):,}（留出 {len(test_set):,}），通关比例 {np.mean([r['win'] for r in runs]):.3f}，"
          f"平均每局 {np.mean([len(r['g']) for r in runs]):.0f} 步、局外多选步 {np.mean([len(r['dec']) for r in runs]):.0f} 个", flush=True)

    # 每局的“关键序列”：局外步与战斗后第一步（事后摘要只看它们，序列短）
    for r in runs:
        r["key"] = np.flatnonzero(r["macro"] | r["post"])

    H = a.hidden
    v_base = nn.Sequential(nn.Linear(d, H), nn.GELU(), nn.Linear(H, 1)).to(dev)
    enc = nn.GRU(d, H, batch_first=True).to(dev)
    v_hind = nn.Sequential(nn.Linear(d + H, H), nn.GELU(), nn.Linear(H, 1)).to(dev)
    VOC = 4096
    c_emb = nn.Embedding(VOC, 32).to(dev); k_emb = nn.Embedding(32, 32).to(dev)
    q_mlp = nn.Sequential(nn.Linear(d + H, H), nn.GELU(), nn.Linear(H, 32)).to(dev)
    opt_v = torch.optim.Adam(list(v_base.parameters()), lr=a.lr)
    opt_h = torch.optim.Adam(list(enc.parameters()) + list(v_hind.parameters()), lr=a.lr)
    opt_q = torch.optim.Adam(list(c_emb.parameters()) + list(k_emb.parameters()) + list(q_mlp.parameters()), lr=a.lr)

    def run_tensors(r):
        g = torch.from_numpy(r["g"].astype(np.float32)).to(dev)
        y = torch.full((len(r["g"]),), 1.0 if r["win"] else -1.0, device=dev)
        return g, y

    def hindsight(g: torch.Tensor, key: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """Φ_j：关键步 j 之后紧接着的 --window 个关键步（不含 j）的摘要（反向 GRU 读这一小段，取最后的隐状态）。
        第一版读 j 之后的整段未来，网络直接从中读出了结局（第 2、3 轮留出解释方差 0.986、0.992），所以只给近处一小段；
        这一小段碰到对局结尾（窗口没装满，或装到最后一个关键步）的位置不用 Φ（mask 为 False，改用普通价值），免得从“快结束了”读出输赢。"""
        K = len(key); W = a.window
        mask = torch.tensor([j + W < K - 1 for j in range(K)], device=dev)
        phi = g.new_zeros(K, H)
        idx = [j for j in range(K) if j + W < K - 1]
        if idx:
            gk = g[torch.from_numpy(key).to(dev)]
            win = torch.stack([gk[j + 1:j + 1 + W].flip(0) for j in idx])       # [n, W, d]，反向读
            _, h = enc(win)
            phi[torch.tensor(idx, device=dev)] = h[0]
        return phi, mask

    def q_logits(gt, phi, r, j_dec):
        """分类器：策略对数几率 + 候选嵌入 · MLP(g, Φ)。j_dec 为该局第几个局外多选步。"""
        ci = torch.from_numpy(r["cands"][j_dec]).to(dev).long()
        e = c_emb(ci[:, 3].clamp(0, VOC - 1)) + k_emb(ci[:, 0].clamp(0, 31))
        w = q_mlp(torch.cat([gt, phi], -1))
        return torch.from_numpy(r["logits"][j_dec]).to(dev) + e @ w

    class GradRev(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, beta):
            ctx.beta = beta; return x.view_as(x)
        @staticmethod
        def backward(ctx, gout):
            return -ctx.beta * gout, None

    train_ids = [i for i in range(len(runs)) if i not in test_set]
    for ep in range(a.epochs):
        rr.shuffle(train_ids)
        tot = {"vb": 0.0, "vh": 0.0, "q": 0.0, "n": 0}
        for bi in range(0, len(train_ids), a.batch_runs):
            lb = lh = lq = 0.0; nq = 0
            for i in train_ids[bi:bi + a.batch_runs]:
                r = runs[i]; g, y = run_tensors(r)
                key = r["key"]
                phi, pm = hindsight(g, key)                               # [K, H]、[K]
                gk, yk = g[torch.from_numpy(key).to(dev)], y[torch.from_numpy(key).to(dev)]
                lb = lb + ((v_base(gk).squeeze(-1) - yk) ** 2).mean()
                if pm.any():
                    lh = lh + ((v_hind(torch.cat([gk, phi], -1)).squeeze(-1)[pm] - yk[pm]) ** 2).mean()
                # 独立性：只在局外多选步上（它们都在关键序列里）
                pos = {int(s): j for j, s in enumerate(key)}
                for jd, s in enumerate(r["dec"]):
                    j = pos.get(int(s))
                    if j is None or not bool(pm[j]):
                        continue
                    ql = q_logits(gk[j], GradRev.apply(phi[j], a.beta), r, jd)
                    lq = lq + F.cross_entropy(ql[None], torch.tensor([int(r["act"][s])], device=dev)); nq += 1
            nr = len(train_ids[bi:bi + a.batch_runs])
            opt_v.zero_grad(); opt_h.zero_grad(); opt_q.zero_grad()
            (lb / nr + lh / nr + (lq / max(nq, 1) if nq else 0.0)).backward()
            opt_v.step(); opt_h.step(); opt_q.step()
            tot["vb"] += float(lb); tot["vh"] += float(lh); tot["q"] += float(lq); tot["n"] += nr
        res = evaluate(runs, sorted(test_set), v_base, v_hind, hindsight, q_logits, dev, run_tensors)
        print(f"第 {ep + 1} 轮：训练 价值误差 普通 {tot['vb'] / tot['n']:.4f} 事后 {tot['vh'] / tot['n']:.4f}；留出 {res}", flush=True)


def evaluate(runs, test_ids, v_base, v_hind, hindsight, q_logits, dev, run_tensors) -> str:
    import torch
    import torch.nn.functional as F
    yb, pb, ph, gain, n = [], [], [], 0.0, 0
    with torch.no_grad():
        for i in test_ids:
            r = runs[i]; g, y = run_tensors(r)
            key = r["key"]; phi, pm = hindsight(g, key)
            kk = torch.from_numpy(key).to(dev)
            gk, yk = g[kk], y[kk]
            mk = torch.from_numpy(r["macro"][key]).to(dev)       # 只统计局外步（战斗后第一步也是局外步）
            vb = v_base(gk).squeeze(-1)
            vh = torch.where(pm, v_hind(torch.cat([gk, phi], -1)).squeeze(-1), vb)     # 窗口碰到结尾的位置用普通价值
            yb.append(yk[mk].cpu()); pb.append(vb[mk].cpu()); ph.append(vh[mk].cpu())
            pos = {int(s): j for j, s in enumerate(key)}
            for jd, s in enumerate(r["dec"]):
                j = pos.get(int(s))
                if j is None or not bool(pm[j]):
                    continue
                a_ = int(r["act"][s])
                lq = F.log_softmax(q_logits(gk[j], phi[j], r, jd), -1)[a_]
                lp = torch.from_numpy(r["logits"][jd])[a_]
                gain += float(lq.cpu() - lp); n += 1
    y, b, h = torch.cat(yb), torch.cat(pb), torch.cat(ph)
    ev = lambda p: float(1 - ((y - p) ** 2).mean() / y.var())
    var_ratio = float(((y - h) ** 2).mean() / ((y - b) ** 2).mean())
    return (f"解释方差 普通 {ev(b):.4f} 事后 {ev(h):.4f}；优势方差之比（事后 / 普通）{var_ratio:.3f}；"
            f"分类器比策略多出的对数似然 {gain / max(n, 1):+.4f}/步（{n:,} 步）")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["collect", "fit"])
    ap.add_argument("--weights"); ap.add_argument("--config"); ap.add_argument("--dll")
    ap.add_argument("--runs", type=int, default=3000); ap.add_argument("--envs", type=int, default=48); ap.add_argument("--chunk", type=int, default=200)
    ap.add_argument("--out"); ap.add_argument("--data")
    ap.add_argument("--temp", type=float, default=1.0); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--prefix", default="CCA")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=8); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden", type=int, default=128); ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--batch-runs", type=int, default=32)
    ap.add_argument("--max-steps", type=int, default=6000)
    ap.add_argument("--window", type=int, default=8, help="Φ 只读之后紧接着的几个关键步（局外步与战斗后第一步）")
    a = ap.parse_args()
    collect(a) if a.mode == "collect" else fit(a)


if __name__ == "__main__":
    main()
