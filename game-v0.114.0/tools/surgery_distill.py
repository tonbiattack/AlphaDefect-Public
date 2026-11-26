"""网络手术后的蒸馏（决策记录 262）：局外帧扩展注意力（macro_attend）会稀释原有记录的注意力权重，打开时策略与价值输出必然改变。
上线前先让新结构（学生）在真实学习数据上模仿原结构（老师，同一份权重、不扩展）的输出，把冲击压下去（OpenAI Five 换结构时的“手术”做法）。

  损失 = 策略 KL(老师 ‖ 学生) + w_v ×（数值价值头均方误差 + 分类价值头 KL + 辅助头均方误差，有的才算）
  每个小批按 --changed-frac 从“扩展后会变的帧”（局外帧、候选引用了牌堆的选牌帧）里抽，其余从不变的帧里抽（压住主干漂移）。
  留一份数据作留存：每 --eval-every 步报局外多选帧的 KL、首选一致率、价值差，以及不变帧的 KL。
  输出：学生的检查点（配置打开 macro_attend；优化器状态沿用老师的，新加的 kind_x 一组由学习端补；去掉滑动平均权重，学习端从当前参数重新起算）。

    .venv/bin/python -I tools/surgery_distill.py --ckpt latest.pt --dumps d1.pkl d2.pkl ... --holdout d5.pkl --out student.pt [--steps 2000]
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
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dumps", nargs="+", required=True)
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--mb", type=int, default=1024)
    ap.add_argument("--changed-frac", type=float, default=0.6)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--w-v", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--eval-n", type=int, default=4096)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    import torch
    import torch.nn.functional as F
    from sts2rl.features import Ragged, collate_ragged, macro_rows
    from sts2rl.model import ModelConfig, PolicyValueNet, load_state, packed_meta, use_tf32
    use_tf32()
    dev = torch.device(a.device)
    blob = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    mc = dict(blob["config"]["model"]); mc.pop("macro_attend", None)
    teacher = PolicyValueNet(ModelConfig(**mc)).to(dev).eval()
    load_state(teacher, blob["state"])
    student = PolicyValueNet(ModelConfig(**mc, macro_attend=True)).to(dev)
    load_state(student, blob["state"])
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.fast = student.fast = dev.type == "cuda"
    print(f"老师/学生：第 {blob['version']} 版，{json.dumps(mc, ensure_ascii=False)}", flush=True)

    def load_dump(path) -> tuple[Ragged, np.ndarray]:
        r = pickle.load(open(path, "rb"))
        mac = macro_rows(r.header, r.modes())
        # 候选引用了牌堆里的牌的帧（选牌）：扩展后也会变
        sc = np.repeat(np.arange(len(r)), np.diff(r.cand_off))
        ref = np.zeros(len(r), bool)
        for col in (1, 2):
            x = r.cand_ints[:, col].astype(np.int64); ok = (x >= 0) & (x < np.diff(r.tok_off)[sc])
            t = r.tok_off[sc[ok]] + x[ok]
            pooled = (r.tok_ints[t, 0] == 4) & np.isin(r.tok_ints[t, 2], [4, 5, 6, 7])       # 牌，在抽牌堆、弃牌堆、消耗堆、牌组
            ref[sc[ok][pooled]] = True
        return r, mac | ref
    parts = [load_dump(p) for p in a.dumps]
    data = Ragged.concat([p[0] for p in parts]); chg = np.concatenate([p[1] for p in parts])
    hold, hchg = load_dump(a.holdout)
    ci, ui = np.flatnonzero(chg), np.flatnonzero(~chg)
    print(f"训练数据 {len(data)} 步（会变的 {len(ci)}），留存 {len(hold)} 步（会变的 {int(hchg.sum())}）", flush=True)

    def batches(r: Ragged, idx: np.ndarray):
        s = r.select(idx)
        b0, b1 = collate_ragged(s, attend=False), collate_ragged(s, attend=True)
        out = []
        for bt in (b0, b1):
            d = bt.to(dev)
            if dev.type == "cuda":
                d.update({k: (torch.from_numpy(v).to(dev) if isinstance(v, np.ndarray) else v) for k, v in packed_meta(bt.core_idx, bt.core_mask).items()})
            out.append(d)
        return out

    def losses(b0, b1, train: bool):
        with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            tl, tv, _, tex = teacher.forward_all(b0)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"), (torch.enable_grad() if train else torch.no_grad()):
            sl, sv, _, sex = student.forward_all(b1)
        tl, sl = tl.float(), sl.float()
        tp = F.log_softmax(tl, -1); sp = F.log_softmax(sl, -1)
        kl = (tp.exp() * (tp - sp)).nan_to_num(0.0, 0.0, 0.0).sum(-1)
        lv = (sv.float() - tv.float()) ** 2
        extra = torch.zeros_like(lv)
        if "hl_logits" in sex:
            th, sh = F.log_softmax(tex["hl_logits"].float(), -1), F.log_softmax(sex["hl_logits"].float(), -1)
            extra = extra + (th.exp() * (th - sh)).sum(-1) + (sex["v_scalar"].float() - tex["v_scalar"].float()) ** 2
        if "aux" in sex:
            extra = extra + ((sex["aux"].float() - tex["aux"].float()) ** 2).sum(-1)
        return kl, lv, extra, tl, sl, tv, sv

    @torch.no_grad()
    def evaluate(step: int) -> dict:
        student.eval()
        rng = np.random.default_rng(0)
        hc, hu = np.flatnonzero(hchg), np.flatnonzero(~hchg)
        res = {}
        for name, idx in (("会变的帧", rng.choice(hc, min(a.eval_n, len(hc)), replace=False)), ("不变的帧", rng.choice(hu, min(a.eval_n, len(hu)), replace=False))):
            kls, agr, vd, n_multi = [], [], [], 0
            for j in range(0, len(idx), a.mb):
                b0, b1 = batches(hold, np.sort(idx[j:j + a.mb]))
                kl, lv, _, tl, sl, tv, sv = losses(b0, b1, False)
                multi = torch.isfinite(tl).sum(-1) > 1
                kls.append(kl[multi].cpu()); agr.append((tl.argmax(-1) == sl.argmax(-1))[multi].float().cpu()); vd.append((sv - tv).abs().float().cpu())
            kl_all, ag, v = torch.cat(kls), torch.cat(agr), torch.cat(vd)
            res[name] = {"多选帧": int(len(kl_all)), "KL平均": round(float(kl_all.mean()), 5), "KL中位": round(float(kl_all.median()), 5),
                         "KL_p99": round(float(kl_all.quantile(0.99)), 4), "首选一致": round(float(ag.mean()), 4), "价值差平均": round(float(v.mean()), 4)}
        student.train()
        print(f"第 {step} 步 留存：{json.dumps(res, ensure_ascii=False)}", flush=True)
        return res

    hist = [evaluate(0)]
    opt = torch.optim.Adam([p for p in student.parameters() if p.requires_grad], lr=a.lr, eps=1e-5)
    rng = np.random.default_rng(1)
    student.train(); t0 = time.time()
    for step in range(1, a.steps + 1):
        nc = int(a.mb * a.changed_frac)
        idx = np.sort(np.concatenate([rng.choice(ci, nc), rng.choice(ui, a.mb - nc)]))
        b0, b1 = batches(data, idx)
        kl, lv, extra, *_ = losses(b0, b1, True)
        loss = kl.mean() + a.w_v * (lv.mean() + extra.mean())
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0); opt.step()
        if step % 50 == 0:
            print(f"第 {step} 步：KL {float(kl.mean()):.4f}，价值 {float(lv.mean()):.4f}，其余 {float(extra.mean()):.4f}，{time.time() - t0:.0f} 秒", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            hist.append(evaluate(step))
    out = {k: v for k, v in blob.items() if k not in ("ema",)}
    out["state"] = {k: v.detach().cpu() for k, v in student.state_dict().items()}
    cfg = json.loads(json.dumps(blob["config"])); cfg["model"]["macro_attend"] = True; out["config"] = cfg
    out["surgery"] = {"decision": 262, "from_version": blob["version"], "steps": a.steps, "eval": hist}
    torch.save(out, a.out)
    print(f"已存 {a.out}（第 {blob['version']} 版的手术蒸馏，滑动平均权重已去掉）", flush=True)


if __name__ == "__main__":
    main()
