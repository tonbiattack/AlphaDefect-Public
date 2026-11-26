"""公开信息战斗价值头（2026-11-17）：给搜索的叶子估值单独训一个只看公开信息的价值头，主干与策略头不动。

为什么：战斗价值头训练时看得到特权特征（采集端开局 privileged: True：抽牌真实顺序、预抽的遭遇、怪物抽签推演，只进价值网络），
评测、实机、搜索都把这些通道清零（决策记录 161、214），叶子估值是在训练里没见过的输入上算的（reports/搜索算法文献调研.md）。
冒烟里带特权与清零的读数之差，个别局面到 0.5，与搜索里候选 Q 差（标准差约 0.04）同一量级。

做法：同一批帧（boss_pimc run --privileged --save-frames，非留出题、温度 1）——
  老师 = 带特权的价值读数 V_priv(s)；学生输入 = 清零特权后价值头的输入 x（主干全局表示 + 特权编码器在零输入上的输出，用钩子取）；
  新价值头（结构、初值同原 value_mlp）拟合 E[V_priv | 公开信息]（均方差回归到 V_priv，老师读数的噪声远小于终局结果）。
  只换 value_mlp（战斗的价值头）的参数：策略头、主干、局外价值头逐位不变，实机贪心与搜索的先验不变，只有叶子估值变。
按题号 id % 5 == 0 留出，报告对老师读数、对实际终局得分（sim_pimc.terminal 同尺度）的均方差：原清零读数、新头、带特权读数。

  .venv/bin/python -I tools/future/value_public.py --data /root/i2a/vpub/data --ckpt L1000.pt --out 目录
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="逗号分隔的几个目录（res_s*.jsonl + frames_s*.bin）")
    ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True); ap.add_argument("--device", default="cuda")
    ap.add_argument("--stride", type=int, default=1); ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--bs", type=int, default=2048); ap.add_argument("--target", choices=["priv", "mc"], default="priv")
    ap.add_argument("--full", action="store_true", help="整网微调（主干 + 价值头）当单独的战斗价值网络：只用它的价值头（tree_search --value-ckpts … --value-replace）")
    ap.add_argument("--ft-lr", type=float, default=2e-5); ap.add_argument("--ft-epochs", type=int, default=2); ap.add_argument("--ft-bs", type=int, default=128)
    a = ap.parse_args()
    from sts2env.protocol import parse
    from sts2rl.features import Ragged, collate_ragged
    from sts2rl.model import ModelConfig, PolicyValueNet, load_state
    import sim_pimc as X
    dev = a.device
    torch.manual_seed(0); np.random.seed(0)
    blob = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    net = PolicyValueNet(ModelConfig(**blob["config"]["model"])).eval().to(dev)
    load_state(net, blob["state"])
    cap = {}
    hk = net.value_mlp.register_forward_hook(lambda m, i, o: cap.update(x=i[0].detach(), y=o.detach()))

    # ---- 读帧
    rows = []      # (题号, 原始帧, 终局得分, 局内进度)
    t0 = time.time()
    for d in a.data.split(","):
        for res in sorted(glob.glob(f"{d}/res_s*.jsonl")):
            k = res.split("res_s")[1].split(".")[0]
            fb = open(f"{d}/frames_s{k}.bin", "rb")
            for l in open(res, encoding="utf-8"):
                if not l.strip():
                    continue
                r = json.loads(l)
                tr = r.get("traj") or []
                if r.get("stuck") or not tr:
                    continue
                win = bool(r["win"]); left_end = 0.0 if win else tr[-1][5]
                term = 1.5 if win else -1.0 + 0.25 * (tr[-1][6] + 1.0 - left_end)
                for t in range(0, len(tr), a.stride):
                    off, ln = tr[t][0], tr[t][1]
                    fb.seek(off)
                    rows.append((int(r["id"]), fb.read(ln), term, t / len(tr)))
    print(f"{len(rows)} 帧，读入 {time.time() - t0:.0f} 秒", flush=True)

    # ---- 老师读数（带特权）与学生输入（清零特权后 value_mlp 的输入）
    XS, VP, VO, Y, ID, FR = [], [], [], [], [], []
    with torch.no_grad():
        for i in range(0, len(rows), 256):
            chunk = rows[i:i + 256]
            fs = [parse(x[1]) for x in chunk]
            b = collate_ragged(Ragged.from_obs(fs, np.zeros(len(fs), np.int64))).to(dev)
            _, vp = net(b)
            bs = collate_ragged(Ragged.from_obs([X._no_priv(o) for o in fs], np.zeros(len(fs), np.int64))).to(dev)
            _, vo = net(bs)
            c = (~bs["macro"]).nonzero().squeeze(-1).cpu().numpy()
            if len(c) != cap["x"].shape[0]:
                raise RuntimeError("钩子取到的行数与战斗行数不一致")
            XS.append(cap["x"].float().cpu()); VP.append(vp.float().cpu().numpy()[c]); VO.append(vo.float().cpu().numpy()[c])
            Y.append(np.array([chunk[j][2] for j in c])); ID.append(np.array([chunk[j][0] for j in c])); FR.append(np.array([chunk[j][3] for j in c]))
    hk.remove()
    Xs = torch.cat(XS); VP, VO, Y, ID, FR = map(np.concatenate, (VP, VO, Y, ID, FR))
    print(f"战斗行 {len(VP)}，取读数 {time.time() - t0:.0f} 秒；带特权与清零读数之差：标准差 {np.std(VP - VO):.4f}，|差| 的 99 分位 {np.quantile(np.abs(VP - VO), 0.99):.4f}", flush=True)
    va = ID % 5 == 0
    tr_idx = np.flatnonzero(~va); va_idx = np.flatnonzero(va)
    tgt = torch.tensor(VP if a.target == "priv" else Y, dtype=torch.float32)

    def report(v_new: np.ndarray) -> str:
        m = va
        out = []
        for name, v in (("带特权", VP), ("原清零", VO), ("新头", v_new)):
            out.append(f"{name} 对终局 均方差 {np.mean((v[m] - Y[m]) ** 2):.4f} 相关 {np.corrcoef(v[m], Y[m])[0, 1]:.4f}")
        out.append(f"对老师读数 均方差：原清零 {np.mean((VO[m] - VP[m]) ** 2):.5f} 新头 {np.mean((v_new[m] - VP[m]) ** 2):.5f}")
        return "；".join(out)

    if a.full:
        return full_finetune(a, net, blob, rows, VP, VO, Y, ID, FR, va, report)
    head = copy.deepcopy(net.value_mlp).float().to(dev).train()
    opt = torch.optim.Adam(head.parameters(), lr=a.lr)
    Xd = Xs.to(dev); Td = tgt.to(dev)

    def predict() -> np.ndarray:
        head.eval()
        with torch.no_grad():
            out = torch.cat([head(Xd[i:i + 8192]).squeeze(-1) for i in range(0, len(Xd), 8192)]).cpu().numpy()
        head.train()
        return out

    v0 = predict()
    print(f"起点（新头 = 原 value_mlp，应与原清零读数相同，最大差 {np.abs(v0 - VO).max():.2e}）：{report(v0)}", flush=True)
    best, best_state = None, None
    for ep in range(a.epochs):
        perm = np.random.permutation(tr_idx)
        for s in range(0, len(perm), a.bs):
            idx = torch.from_numpy(perm[s:s + a.bs]).to(dev)
            loss = ((head(Xd[idx]).squeeze(-1) - Td[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        v = predict()
        val = float(np.mean((v[va] - (VP if a.target == "priv" else Y)[va]) ** 2))
        if best is None or val < best:
            best, best_state = val, copy.deepcopy(head.state_dict())
        if ep % 5 == 4 or ep == a.epochs - 1:
            print(f"第 {ep + 1} 遍：留出对目标均方差 {val:.5f}；{report(v)}", flush=True)
    head.load_state_dict(best_state)
    v = predict()
    print(f"取留出最好的一遍（对目标均方差 {best:.5f}）：{report(v)}", flush=True)
    # 按局内进度看
    for lo, hi in ((0, .33), (.33, .67), (.67, 1.01)):
        m = va & (FR >= lo) & (FR < hi)
        print(f"  进度 {lo:.2f}+：原清零 {np.mean((VO[m] - Y[m]) ** 2):.4f}  新头 {np.mean((v[m] - Y[m]) ** 2):.4f}  带特权 {np.mean((VP[m] - Y[m]) ** 2):.4f}", flush=True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    blob2 = {k: x for k, x in blob.items() if k != "state"}
    st = {k: x.detach().cpu().clone() for k, x in blob["state"].items()}
    for k, x in head.state_dict().items():
        st[f"value_mlp.{k}"] = x.detach().cpu().to(st[f"value_mlp.{k}"].dtype)
    blob2["state"] = st
    blob2["value_public"] = {"from": a.ckpt, "data": a.data, "target": a.target, "epochs": a.epochs, "lr": a.lr, "best_val": best}
    torch.save(blob2, out / "vpub.pt")
    print("存", out / "vpub.pt", flush=True)


def full_finetune(a, net, blob, rows, VP, VO, Y, ID, FR, va, report) -> None:
    """整网微调：清零特权的帧 → 回归到带特权的读数（--target priv）或终局得分。所有帧都是战斗行（上面已核对行数）。"""
    from sts2env.protocol import parse
    from sts2rl.features import Ragged, collate_ragged
    import sim_pimc as X
    dev = a.device
    if len(VP) != len(rows):
        raise RuntimeError("整网微调要求所有帧都是战斗行")
    tgt = torch.tensor(VP if a.target == "priv" else Y, dtype=torch.float32, device=dev)
    tr_idx, va_idx = np.flatnonzero(~va), np.flatnonzero(va)
    opt = torch.optim.AdamW(net.parameters(), lr=a.ft_lr, weight_decay=0.0)

    def fwd(idx, grad):
        fs = [X._no_priv(parse(rows[i][1])) for i in idx]
        b = collate_ragged(Ragged.from_obs(fs, np.zeros(len(fs), np.int64))).to(dev)
        with torch.set_grad_enabled(grad):
            _, v = net(b)
        return v.float().reshape(-1)

    def predict() -> np.ndarray:
        out = np.full(len(rows), np.nan)
        with torch.no_grad():
            for s in range(0, len(va_idx), 256):
                idx = va_idx[s:s + 256]
                out[idx] = fwd(idx, False).cpu().numpy()
        return out

    t0 = time.time()
    net.eval()      # 不开 dropout；梯度照常
    for ep in range(a.ft_epochs):
        perm = np.random.permutation(tr_idx)
        tot, nb = 0.0, 0
        for s in range(0, len(perm), a.ft_bs):
            idx = perm[s:s + a.ft_bs]
            loss = ((fwd(idx, True) - tgt[torch.from_numpy(idx).to(dev)]) ** 2).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach()); nb += 1
        v = predict()
        print(f"整网第 {ep + 1} 遍：训练均方差 {tot / nb:.5f}，{time.time() - t0:.0f} 秒；{report(v)}", flush=True)
    for lo, hi in ((0, .33), (.33, .67), (.67, 1.01)):
        m = va & (FR >= lo) & (FR < hi)
        print(f"  进度 {lo:.2f}+：原清零 {np.mean((VO[m] - Y[m]) ** 2):.4f}  整网 {np.mean((v[m] - Y[m]) ** 2):.4f}  带特权 {np.mean((VP[m] - Y[m]) ** 2):.4f}", flush=True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    blob2 = {k: x for k, x in blob.items() if k != "state"}
    blob2["state"] = {k: x.detach().cpu() for k, x in net.state_dict().items()}
    blob2["value_public"] = {"from": a.ckpt, "data": a.data, "target": a.target, "full": True, "ft_lr": a.ft_lr, "ft_epochs": a.ft_epochs}
    torch.save(blob2, out / "vnet.pt")
    print("存", out / "vnet.pt（只用它的价值头：tree_search --value-ckpts 它 --value-replace）", flush=True)


if __name__ == "__main__":
    main()
