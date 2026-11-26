"""预知记录（决策记录 280）对价值预测有多大帮助的离线对照：冻结主干，只训练局外价值头，比较“看 / 不看预知记录”在留出对局上的解释方差。

一、采集：用给定权重按概率出手（温度 1，同训练采集）打整局，reset 带 privileged、future_content、oracle；记下每个战斗外决策帧
    （带预知记录的帧）与这一局的结局（是否通关、最后的层数）。每 --chunk 局存一份 pickle。
二、拟合：主干冻结，先把每帧的全局表示 g 与已有的特权汇总（抽牌顺序、预抽遭遇、整局预生成内容）算好缓存；
    对照组只训练局外价值头 value_mlp_m（输入 g 与特权汇总），实验组同样的头再加预知小网络（orc_*，输出映射从 0 起）。
    两组同样的初始化、数据、步数，目标是通关与否（±1），按局分 80% / 20%，报告留出对局的解释方差（全部、按幕）。

    python -I tools/oracle_value_probe.py collect --dll 新引擎/Sts2Headless.dll --weights 权重.pt --config train_config.json --runs 3000 --out 目录 [--envs 48]
    python -I tools/oracle_value_probe.py fit --data 目录 --weights 权重.pt --config train_config.json [--epochs 6]
"""
from __future__ import annotations

import argparse
import copy
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

RESET = {"op": "reset", "mode": "run", "character": "DEFECT", "ascension": 10, "route_summary": True, "snapshots": True,
         "privileged": True, "future_content": True, "oracle": True}


def load_net(weights: str, config: str, oracle: bool, device):
    import torch
    from sts2rl.model import ModelConfig, PolicyValueNet, load_state
    mc = dict(json.loads(Path(config).read_text(encoding="utf-8"))["model"])
    mc["oracle"] = oracle
    net = PolicyValueNet(ModelConfig(**mc)).to(device).eval()
    blob = torch.load(weights, map_location="cpu", weights_only=False)
    load_state(net, blob["state"])
    return net, int(blob.get("version") or 0)


def collect(a) -> None:
    import torch
    from sts2env.env import VecEnv
    from sts2env.protocol import TokenKind
    from sts2rl.actor import MODE_RUN
    from sts2rl.features import Ragged, collate_ragged
    dev = torch.device(a.device)
    net, ver = load_net(a.weights, a.config, False, dev)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    env = VecEnv(a.envs, dll=Path(a.dll))
    rng = np.random.default_rng(a.seed)
    started = done = 0
    slot: dict[int, dict] = {}
    chunk: list[dict] = []
    t0 = time.time()

    def start(i: int) -> None:
        nonlocal started
        if started >= a.runs:
            slot.pop(i, None); return
        slot[i] = {"run": started, "frames": []}
        env.send(i, {**RESET, "seed": f"{a.prefix}{started:06d}"})
        started += 1

    def flush() -> None:
        if not chunk:
            return
        frames = [f for c in chunk for f in c["frames"]]
        meta = {"run": np.concatenate([np.full(len(c["frames"]), c["run"]) for c in chunk]),
                "win": np.concatenate([np.full(len(c["frames"]), c["win"]) for c in chunk]),
                "floor": np.concatenate([np.full(len(c["frames"]), c["floor"]) for c in chunk]),
                "act": np.array([int(f.act) for f in frames]), "at_floor": np.array([int(f.floor) for f in frames])}
        r = Ragged.from_obs(frames, np.full(len(frames), MODE_RUN))
        k = len(list(out.glob("chunk_*.pkl")))
        pickle.dump({"ragged": r, **meta, "version": ver}, open(out / f"chunk_{k:03d}.pkl", "wb"))
        chunk.clear()

    for i in range(a.envs):
        start(i)
    obs: dict[int, object] = {}
    while slot:
        for i, o in env.recv_ready(timeout=1.0):
            st = slot.get(i)
            if st is None:
                continue
            if isinstance(o, Exception) or o.done or o.stuck:
                if not isinstance(o, Exception) and not o.stuck and st["frames"]:
                    chunk.append({"run": st["run"], "frames": st["frames"], "win": int(bool(o.victory)), "floor": int(o.floor)})
                    done += 1
                    if done % a.chunk == 0:
                        flush()
                        print(f"{time.strftime('%H:%M:%S')} {done} 局，{(time.time() - t0) / 60:.1f} 分钟", flush=True)
                start(i); continue
            if (o.tok_ints[:, 0] == TokenKind.ORACLE).any():
                st["frames"].append(o)
            obs[i] = o
        if not obs:
            continue
        ids = sorted(obs)
        with torch.no_grad():
            lg = net(collate_ragged(Ragged.from_obs([obs[i] for i in ids], np.full(len(ids), MODE_RUN))).to(dev))[0].float()
        for k, i in enumerate(ids):
            na = obs[i].num_actions
            p = torch.softmax(lg[k, :na] / a.temp, -1).cpu().numpy().astype(np.float64)
            env.send_step(i, int(rng.choice(na, p=p / p.sum())))
        obs.clear()
    flush()
    print(f"采集完：{done} 局", flush=True)


def fit(a) -> None:
    import torch
    import torch.nn.functional as F
    from sts2rl.features import Ragged, collate_ragged
    dev = torch.device(a.device)
    net, ver = load_net(a.weights, a.config, True, dev)
    for p in net.parameters():
        p.requires_grad_(False)
    data = [pickle.load(open(f, "rb")) for f in sorted(Path(a.data).glob("chunk_*.pkl"))]
    print(f"数据 {len(data)} 份，帧 {sum(len(d['win']) for d in data):,}，局 {len({(j, r) for j, d in enumerate(data) for r in set(d['run'].tolist())}):,}", flush=True)
    G, P, OI, OS, OB, OF = [], [], [], [], [], []      # 全局表示、特权汇总；预知记录的词表 ID、类别 × 先后、子类、所属帧
    meta = {k: [] for k in ("run", "win", "floor", "act")}
    base = 0
    with torch.no_grad():
        for d in data:
            r: Ragged = d["ragged"]
            for j in range(0, len(r), 512):
                idx = np.arange(j, min(j + 512, len(r)))
                b = collate_ragged(r.select(idx)).to(dev)
                e, ctx, g = net._encode(b)
                B, dd = g.shape
                top = e.new_zeros(B, dd).index_add_(0, b["draw_step"], net._take(e, b["draw_tok"]) * b["draw_w"][:, None]) / b["draw_wsum"][:, None]
                priv = net.priv_proj(top)
                vs = net.cfg.vocab_size
                if b.get("priv_enc") is not None:
                    pe = net.id_emb(b["priv_enc"].clamp(0, vs - 1)).reshape(-1, 2 * dd)
                    priv = priv.index_add(0, b["priv_enc_step"], net.priv_enc(pe).to(priv.dtype))
                    pr = net.id_emb(net._take(b["tok_ints"][:, 1], b["roll_tok"]).clamp(0, vs - 1))
                    priv = priv.index_add(0, b["roll_step"], net.priv_roll(pr).to(priv.dtype))
                if net.cfg.future_content and b.get("fut_tok") is not None and b["fut_tok"].numel():
                    from sts2rl.model import FUT_CODES
                    fid = net._take(b["tok_ints"][:, 1], b["fut_tok"]).clamp(0, vs - 1)
                    h = F.gelu(net.fut_in(torch.cat([net.fut_emb(fid), net.fut_slot(b["fut_code"].clamp(0, FUT_CODES - 1))], -1)))
                    pooled = h.new_zeros(B, h.shape[1]).index_add_(0, b["fut_step"], h * b["fut_w"][:, None].to(h.dtype))
                    priv = priv + net.fut_out(pooled).to(priv.dtype)
                G.append(g.float().cpu()); P.append(priv.float().cpu())
                OI.append(net._take(b["tok_ints"][:, 1], b["orc_tok"]).cpu()); OS.append(b["orc_slot"].cpu()); OB.append(b["orc_sub"].cpu())
                OF.append(b["orc_step"].cpu() + base); base += B
            for k in meta:
                meta[k].append(d[k])
    G, P = torch.cat(G), torch.cat(P)
    OI, OS, OB, OF = torch.cat(OI), torch.cat(OS), torch.cat(OB), torch.cat(OF)
    m = {k: np.concatenate(v) for k, v in meta.items()}
    # 不同份数据里的局号各自从 0 编起：按份重新编号
    run_key = np.concatenate([np.full(len(d["win"]), j) * 10_000_000 + d["run"] for j, d in enumerate(data)])
    y = torch.tensor(np.where(m["win"] > 0, 1.0, -1.0), dtype=torch.float32)
    runs = np.unique(run_key); rr = np.random.default_rng(0); rr.shuffle(runs)
    test_runs = set(runs[: len(runs) // 5].tolist())
    test = torch.tensor([k in test_runs for k in run_key])
    N = len(y)
    order_of = torch.argsort(OF)
    OI, OS, OB, OF = OI[order_of], OS[order_of], OB[order_of], OF[order_of]
    starts = torch.searchsorted(OF, torch.arange(N + 1))
    print(f"帧 {N:,}（留出 {int(test.sum()):,}），通关比例 {float((y > 0).float().mean()):.3f}，预知记录 {len(OI):,}", flush=True)

    def run_arm(use_oracle: bool) -> dict:
        torch.manual_seed(0)
        head = copy.deepcopy(net.value_mlp_m).to(dev).float()
        mods = [head]
        if use_oracle:
            orc = {k: copy.deepcopy(getattr(net, k)).to(dev).float() for k in ("orc_emb", "orc_slot", "orc_sub", "orc_in", "orc_out")}
            if a.share_emb:
                # 借用主干学好的词表嵌入（冻结，d 维），只训练一层投影到 64 维，替代从零学的 orc_emb（数据少时从零学会过拟合）
                orc["orc_emb"] = torch.nn.Sequential(copy.deepcopy(net.id_emb).float().requires_grad_(False), torch.nn.Linear(net.cfg.d, 64)).to(dev)
                mods += [orc["orc_emb"][1]] + [orc[k] for k in ("orc_slot", "orc_sub", "orc_in", "orc_out")]
            else:
                mods += list(orc.values())
        for mm in mods:
            for p in mm.parameters():
                p.requires_grad_(True)
        opt = torch.optim.AdamW([p for mm in mods for p in mm.parameters() if p.requires_grad], lr=a.lr, weight_decay=a.wd)

        def forward(idx: torch.Tensor) -> torch.Tensor:
            g, pv = G[idx].to(dev), P[idx].to(dev)
            if use_oracle:
                lo, hi = starts[idx], starts[idx + 1]
                cnt = (hi - lo)
                sel = torch.cat([torch.arange(int(l), int(h_)) for l, h_ in zip(lo.tolist(), hi.tolist())]) if int(cnt.sum()) else torch.zeros(0, dtype=torch.long)
                step = torch.repeat_interleave(torch.arange(len(idx)), cnt)
                if len(sel):
                    hh = F.gelu(orc["orc_in"](torch.cat([orc["orc_emb"](OI[sel].to(dev).clamp(0, net.cfg.vocab_size - 1)),
                                                        orc["orc_slot"](OS[sel].to(dev)), orc["orc_sub"](OB[sel].to(dev))], -1)))
                    w = (1.0 / cnt.clamp(min=1).float())[step].to(dev)
                    pooled = hh.new_zeros(len(idx), hh.shape[1]).index_add_(0, step.to(dev), hh * w[:, None])
                    pv = pv + orc["orc_out"](pooled)
            return head(torch.cat([g, pv], -1)).squeeze(-1)

        train_idx = torch.nonzero(~test).squeeze(1)
        test_idx = torch.nonzero(test).squeeze(1)
        curve = []
        for ep in range(a.epochs):
            perm = train_idx[torch.randperm(len(train_idx))]
            for j in range(0, len(perm), a.batch):
                idx = perm[j:j + a.batch]
                loss = ((forward(idx) - y[idx].to(dev)) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                pt = torch.cat([forward(test_idx[j:j + 4096]) for j in range(0, len(test_idx), 4096)]).cpu()
            yt = y[test_idx]
            curve.append(round(float(1 - ((yt - pt) ** 2).mean() / yt.var()), 4))
        print(f"  {'看预知' if use_oracle else '不看预知'}：每轮留出解释方差 {curve}", flush=True)
        with torch.no_grad():
            pred = torch.cat([forward(torch.arange(j, min(j + 4096, N))) for j in range(0, N, 4096)]).cpu()
        out = {}
        for name, sel in (("训练", ~test), ("留出", test)):
            for act in (None, 0, 1, 2):
                s = sel & (torch.ones(N, dtype=torch.bool) if act is None else torch.tensor(m["act"] == act))
                if int(s.sum()) < 50:
                    continue
                yy, pp = y[s], pred[s]
                out[f"{name}·{'全部' if act is None else f'第{act + 1}幕'}"] = (round(float(1 - ((yy - pp) ** 2).mean() / yy.var()), 4), int(s.sum()))
        return out

    res0 = run_arm(False)
    res1 = run_arm(True)
    print("解释方差（目标：是否通关 ±1；冻结主干，只训局外价值头）：", flush=True)
    for k in res0:
        print(f"  {k}：不看预知 {res0[k][0]:.4f}  看预知 {res1.get(k, (float('nan'),))[0]:.4f}  （{res0[k][1]:,} 帧）", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["collect", "fit"])
    ap.add_argument("--dll")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--runs", type=int, default=3000)
    ap.add_argument("--envs", type=int, default=48)
    ap.add_argument("--chunk", type=int, default=200)
    ap.add_argument("--out")
    ap.add_argument("--data")
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prefix", default="ORCV")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--share-emb", type=int, default=1, help="1：预知记录借用主干冻结的词表嵌入加一层投影；0：从零学 orc_emb")
    a = ap.parse_args()
    collect(a) if a.mode == "collect" else fit(a)


if __name__ == "__main__":
    main()
