"""模型池蒸馏（决策记录 273）：把几个快照的集成（老师，每步概率平均）蒸馏进单个网络（学生，从其中一个快照起步）。
快照集成实测约 +3 个百分点（决策记录 272），这一步看单个网络能保住多少，也检验“网络学不学得进更好的目标”——
第 217 条搜索蒸馏只学进了改选的 6%，如果连这种平滑、无搜索噪声的老师都教不会，搜索蒸馏也难成。

  损失 = KL(老师 ‖ 学生)（只算候选不少于 2 个的步）+ w_v × 学生价值与它起步时价值的均方误差（价值头不需要改，拴住即可）。
  局面来自学习端的批次存档（dump），留一份作留存：每 --eval-every 步报 KL、学生与老师的首选一致率、学生与起步网络的首选一致率（战斗、局外分开）。
  输出：学生检查点原有的全部字段（版本、进度、步数、观测格式、词表大小、配置等）照抄，只换掉权重，去掉优化器状态与滑动平均权重——
  既能给 tools/ability_test.py 评测，也能放成学习端的 latest.pt 直接续训（学习端见到没有优化器状态会从一成学习率逐步升）。
  老师与学生的结构可以不同（决策记录 273 补记：老结构的快照池蒸馏进新结构的主线），按各自的 macro_attend 分别组批。

    .venv/bin/python -I tools/pool_distill.py --student 起步.pt --teachers 快照列表.txt --dumps d1.pkl ... --holdout d5.pkl --out 学生.pt [--steps 4000 --lr 3e-5]
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--teachers", required=True, help="老师快照路径，一行一个")
    ap.add_argument("--dumps", nargs="+", required=True)
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--mb", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--w-v", type=float, default=0.5)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-n", type=int, default=8192)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set-model", action="append", default=[], help="改学生的网络结构，如 macro_attend=false（决策记录 274：撤掉局外帧扩展注意力）；起步权重照常载入，多出来的层丢掉")
    a = ap.parse_args()
    import torch
    import torch.nn.functional as F
    from sts2rl.features import Ragged, collate_ragged
    from sts2rl.model import ModelConfig, PolicyValueNet, load_state, packed_meta, use_tf32
    from sts2env.protocol import DecisionKind
    use_tf32(); torch.manual_seed(a.seed)
    dev = torch.device(a.device)

    def build(path, train=False, override=None):
        blob = torch.load(path, map_location="cpu", weights_only=False)
        if override:
            blob["config"] = json.loads(json.dumps(blob["config"])); blob["config"]["model"].update(override)
        net = PolicyValueNet(ModelConfig(**blob["config"]["model"])).to(dev)
        load_state(net, blob["state"]); net.fast = dev.type == "cuda"
        net.train(train)
        if not train:
            for p in net.parameters():
                p.requires_grad_(False)
        return net, blob
    ov = {}
    for kv in a.set_model:
        k, v = kv.split("=", 1)
        ov[k] = {"true": True, "false": False}.get(v.lower(), v)
    student, sblob = build(a.student, train=True, override=ov)
    origin, _ = build(a.student, override=ov)     # 起步网络（同样的结构改动；拴价值、算“与起步一致”）
    attend = bool(sblob["config"]["model"].get("macro_attend"))
    teachers = []
    for l in open(a.teachers):
        if l.strip():
            t, tb = build(l.strip())
            teachers.append((t, bool(tb["config"]["model"].get("macro_attend"))))
    print(f"学生起步：第 {sblob['version']} 版；老师：{len(teachers)} 个快照的概率平均", flush=True)

    def load(path):
        r = pickle.load(open(path, "rb"))
        return r, np.flatnonzero(np.diff(r.cand_off) > 1)
    parts = [load(p) for p in a.dumps]
    data = Ragged.concat([p[0] for p in parts])
    multi = np.flatnonzero(np.diff(data.cand_off) > 1)
    hold, hmulti = load(a.holdout)
    print(f"训练局面 {len(data)}（多选 {len(multi)}），留存 {len(hold)}（多选 {len(hmulti)}）", flush=True)

    def batch(r, idx):
        """同一批局面按需要的 attend 各组一份：{attend: 组好的批}，学生用自己的，老师各用各的。"""
        sel = r.select(idx); out = {}
        for att in {attend} | {ta for _, ta in teachers}:
            bt = collate_ragged(sel, attend=att)
            d = bt.to(dev)
            if dev.type == "cuda":
                d.update({k: (torch.from_numpy(v).to(dev) if isinstance(v, np.ndarray) else v) for k, v in packed_meta(bt.core_idx, bt.core_mask).items()})
            out[att] = d
        return out

    def teacher_lp(bs):
        with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            ps = [F.softmax(t(bs[ta])[0].float(), -1) for t, ta in teachers]
        W = max(p.shape[1] for p in ps)
        ps = [F.pad(p, (0, W - p.shape[1])) for p in ps]
        return torch.log(torch.stack(ps).mean(0))

    @torch.no_grad()
    def evaluate(step):
        student.eval()
        idx = np.sort(np.random.default_rng(0).choice(hmulti, min(a.eval_n, len(hmulti)), replace=False))
        res = {"KL": [], "师": [], "原": [], "combat": []}
        for j in range(0, len(idx), a.mb):
            bs = batch(hold, idx[j:j + a.mb]); b = bs[attend]
            tl = teacher_lp(bs)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                sl = F.log_softmax(student(b)[0].float(), -1); ol = F.log_softmax(origin(b)[0].float(), -1)
            res["KL"].append((tl.exp() * (tl - sl)).nan_to_num(0.0, 0.0, 0.0).sum(-1).cpu())
            res["师"].append((sl.argmax(-1) == tl.argmax(-1)).float().cpu()); res["原"].append((sl.argmax(-1) == ol.argmax(-1)).float().cpu())
            hd = b["decision"].cpu(); res["combat"].append(((hd == int(DecisionKind.COMBAT)) | ~b["macro"].cpu()).cpu())
        kl, t_, o_, c = (torch.cat(res[k]) for k in ("KL", "师", "原", "combat"))
        out = {"KL": round(float(kl.mean()), 5), "与老师一致": round(float(t_.mean()), 4), "与老师一致_战斗": round(float(t_[c].mean()), 4),
               "与老师一致_局外": round(float(t_[~c].mean()), 4) if (~c).any() else None, "与起步一致": round(float(o_.mean()), 4)}
        student.train()
        print(f"第 {step} 步 留存：{json.dumps(out, ensure_ascii=False)}", flush=True)
        return out

    hist = [evaluate(0)]
    opt = torch.optim.Adam(student.parameters(), lr=a.lr, eps=1e-5)
    rng = np.random.default_rng(a.seed + 1); t0 = time.time()
    for step in range(1, a.steps + 1):
        bs = batch(data, np.sort(rng.choice(multi, a.mb, replace=False))); b = bs[attend]
        tl = teacher_lp(bs)
        with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            v0 = origin(b)[1].float()
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            slog, sv = student(b)
        sl = F.log_softmax(slog.float(), -1)
        kl = (tl.exp() * (tl - sl)).nan_to_num(0.0, 0.0, 0.0).sum(-1).mean()
        lv = ((sv.float() - v0) ** 2).mean()
        loss = kl + a.w_v * lv
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0); opt.step()
        if step % 100 == 0:
            print(f"第 {step} 步：KL {float(kl):.4f}，价值 {float(lv):.5f}，{time.time() - t0:.0f} 秒", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            hist.append(evaluate(step))
    out = {k: v for k, v in sblob.items() if k not in ("opt", "ema", "ema_state")}
    out["state"] = {k: v.detach().cpu() for k, v in student.state_dict().items()}
    out["pool_distill"] = {"decision": 273, "teachers": a.teachers, "steps": a.steps, "lr": a.lr, "eval": hist}
    torch.save(out, a.out)
    print(f"已存 {a.out}", flush=True)


if __name__ == "__main__":
    main()
