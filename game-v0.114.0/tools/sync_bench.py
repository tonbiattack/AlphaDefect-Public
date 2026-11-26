"""（2026-10-28，决策记录 158）学习端去同步的核对与测速：同一批导出数据、同一份权重，比较
old：改前的 PPO 小批循环（每小批 ~10 次 .item()/.any()、Categorical 校验、前向里 nonzero/取总长、默认流上拷贝）；
new：train.py 里的新 minibatches + 统计量留在显卡上（每遍一次同步）+ 预先算好的打包偏移。
    python sync_bench.py --dump batch_dump.pkl --ckpt ckpt.pt --passes 4
"""
import argparse, pickle, queue, sys, threading, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import numpy as np
import torch
import torch.nn.functional as F

from sts2rl.features import collate_ragged
from sts2rl.model import ModelConfig, PolicyValueNet, load_state, packed_meta
from sts2rl.train import minibatches
from sts2env.protocol import DecisionKind


def old_minibatches(data, perm, size, device):
    q = queue.Queue(maxsize=2)
    def work():
        for s in range(0, len(perm), size):
            idx = perm[s:s + size]
            q.put((idx, collate_ragged(data.select(idx)).to(device)))
        q.put(None)
    threading.Thread(target=work, daemon=True).start()
    while True:
        it = q.get()
        if it is None:
            break
        yield it


WORKERS = 1
WAIT: list = []


def run(mode, net, fwd, opt, data, arrays, passes, mb, device, seed):
    actions, old_logp, adv_n, ret, w_pi = arrays
    rng = np.random.default_rng(seed)
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(passes):
        perm = rng.permutation(len(data))
        if mode == "old":
            for idx, b in old_minibatches(data, perm, mb, device):
                a = torch.from_numpy(actions[idx]).to(device); olp = torch.from_numpy(old_logp[idx]).to(device)
                A = torch.from_numpy(adv_n[idx]).to(device); R = torch.from_numpy(ret[idx]).to(device); W = torch.from_numpy(w_pi[idx]).to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits, v = fwd(b)
                logits, v = logits.float(), v.float()
                dist = torch.distributions.Categorical(logits=logits)
                lp = dist.log_prob(a); ratio = torch.exp(lp - olp)
                l_pi = -(W * torch.min(ratio * A, ratio.clamp(0.8, 1.2) * A)).sum() / W.sum()
                l_v = F.mse_loss(v, R); ent_all = dist.entropy(); ent = ent_all.mean()
                is_map = b["decision"] == int(DecisionKind.MAP)
                ent_w = torch.where(is_map, 0.05, torch.where(b["macro"], 0.03, 0.01))
                loss = l_pi + 0.5 * l_v - (ent_w * ent_all).mean()
                opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
                with torch.no_grad():
                    s = [l_pi.item(), l_v.item(), ent.item(), ((ratio - 1) - (lp - olp)).mean().item(), ((ratio - 1).abs() > 0.2).float().mean().item()]
                    mbm = b["macro"]
                    if mbm.any():
                        s += [ent_all[mbm].mean().item(), F.mse_loss(v[mbm], R[mbm]).item()]
                    if is_map.any():
                        s.append(ent_all[is_map].mean().item())
        else:
            per = {"ppo_a": actions, "ppo_olp": old_logp, "ppo_adv": adv_n, "ppo_ret": ret, "ppo_w": w_pi}
            acc = torch.zeros(10, device=device); timing = {}
            for idx, b in minibatches(data, perm, mb, device, timing, per, packed=True, workers=WORKERS):
                a, olp, A, R, W = b["ppo_a"], b["ppo_olp"], b["ppo_adv"], b["ppo_ret"], b["ppo_w"]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits, v = fwd(b)
                logits, v = logits.float(), v.float()
                dist = torch.distributions.Categorical(logits=logits, validate_args=False)
                lp = dist.log_prob(a); ratio = torch.exp(lp - olp)
                l_pi = -(W * torch.min(ratio * A, ratio.clamp(0.8, 1.2) * A)).sum() / W.sum()
                l_v = F.mse_loss(v, R); ent_all = dist.entropy(); ent = ent_all.mean()
                is_map = b["decision"] == int(DecisionKind.MAP)
                ent_w = torch.where(is_map, 0.05, torch.where(b["macro"], 0.03, 0.01))
                loss = l_pi + 0.5 * l_v - (ent_w * ent_all).mean()
                opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
                with torch.no_grad():
                    kl = ((ratio - 1) - (lp - olp)).mean(); mbm = b["macro"].float(); mm = is_map.float()
                    acc += torch.stack([l_pi.detach(), l_v.detach(), ent.detach(), kl, ((ratio - 1).abs() > 0.2).float().mean(),
                                        (ent_all * mbm).sum() / mbm.sum().clamp(min=1), ((v - R) ** 2 * mbm).sum() / mbm.sum().clamp(min=1), (mbm.sum() > 0).float(),
                                        (ent_all * mm).sum() / mm.sum().clamp(min=1), (mm.sum() > 0).float()])
            acc.tolist(); WAIT.append(timing.get("wait", 0.0))
    torch.cuda.synchronize()
    return (time.time() - t0) / passes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--passes", type=int, default=4); ap.add_argument("--mb", type=int, default=4096)
    ap.add_argument("--workers", default="1")
    ap.add_argument("--rounds", type=int, default=3); ap.add_argument("--compile", type=int, default=1)
    a = ap.parse_args()
    device = torch.device("cuda")
    data = pickle.load(open(a.dump, "rb"))
    blob = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    net = PolicyValueNet(ModelConfig(**blob["config"]["model"])).to(device); load_state(net, blob["state"]); net.fast = True
    N = len(data); rng = np.random.default_rng(0)
    arrays = (rng.integers(0, 2, N).astype(np.int64), np.full(N, -0.7, np.float32), rng.standard_normal(N).astype(np.float32),
              rng.standard_normal(N).astype(np.float32), np.ones(N, np.float32))
    # ---- 数值核对（fp32、关 TF32）：带预算偏移与不带，同一小批
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    net.eval()
    idx = np.arange(min(4096, N)); bt = collate_ragged(data.select(idx))
    with torch.no_grad():
        l0, v0 = net(bt.to(device))
        for k, v in packed_meta(bt.core_idx, bt.core_mask).items():
            setattr(bt, k, v)
        l1, v1 = net(bt.to(device))
    fin = torch.isfinite(l0) & torch.isfinite(l1)
    print(f"核对：对数几率最大差 {(l0 - l1)[fin].abs().max().item():.2e}，价值最大差 {(v0 - v1).abs().max().item():.2e}，"
          f"-inf 位置一致 {bool(((~torch.isfinite(l0)) == (~torch.isfinite(l1))).all())}", flush=True)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    net.train()
    fwd = torch.compile(net, dynamic=True) if a.compile else net
    opt = torch.optim.Adam(net.parameters(), lr=1e-6, fused=True)
    global WORKERS
    modes = ["old"] + [f"new{w}" for w in a.workers.split(",")]
    def go(m, passes, seed):
        global WORKERS
        if m.startswith("new"):
            WORKERS = int(m[3:])
        WAIT.clear()
        t = run("old" if m == "old" else "new", net, fwd, opt, data, arrays, passes, a.mb, device, seed)
        return t, (sum(WAIT) / max(1, len(WAIT)))
    for m in modes:        # 预热（含编译）
        t, w = go(m, 2, 1)
        print(f"预热 {m} 每遍 {t:.2f} 秒", flush=True)
    for r in range(a.rounds):
        res = {m: go(m, a.passes, 100 + r) for m in modes}
        print(f"第 {r + 1} 轮：" + "，".join(f"{m} {t:.3f} 秒/遍" + (f"（等组批 {w:.2f}）" if m != "old" else "") for m, (t, w) in res.items()), flush=True)


if __name__ == "__main__":
    main()
