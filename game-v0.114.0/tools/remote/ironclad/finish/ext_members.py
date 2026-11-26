# 从铁甲战士训练的完整检查点抽出测试与集成成员（2026-11-19，铁甲战士决策记录第 12 条；照 tools/remote/fool/finish/ext_members.py）：
#   v<版本>.pt 原始权重（state，常规评测评的就是它）；m<版本>.pt 学习端的滑动平均权重（ema，衰减 0.999，时间常数约 1,000 版）
# 本线没开免调度，没有 sf_x，不抽 x。只留评测要的字段，不带优化器
#   python -I ext_members.py <输出目录> <完整检查点> ...
import sys, torch
out = sys.argv[1]
for p in sys.argv[2:]:
    b = torch.load(p, map_location="cpu", weights_only=False)
    v = b["version"]
    keep = {k: b[k] for k in ("version", "progress", "config", "obs_format", "vocab_n", "total_steps", "content_names", "catalog_version") if k in b}
    raw = b["state"]
    torch.save({**keep, "state": raw, "src": p, "weights": "y"}, f"{out}/v{v}.pt")
    ema = b.get("ema") or {}
    sm = {k: (ema[k].to(t.dtype) if k in ema else t) for k, t in raw.items()}
    torch.save({**keep, "state": sm, "src": p, "weights": "ema"}, f"{out}/m{v}.pt")
    d = lambda a: sum(float((a[k].float() - raw[k].float()).norm() ** 2) for k in raw if torch.is_floating_point(raw[k])) ** 0.5
    print(v, p.split("/")[-1], f"ema 覆盖 {sum(k in ema for k in raw)}/{len(raw)}；ema 与原始权重的差（范数）{d(sm):.3f}", flush=True)
