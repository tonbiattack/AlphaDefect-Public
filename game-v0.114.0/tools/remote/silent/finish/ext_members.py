# 从训练的完整检查点抽出集成成员：v<版本>.pt 为原始权重，m<版本>.pt 为学习端的滑动平均权重（衰减 0.999）；
# 只留评测要的字段（config、state、version、progress、obs_format、vocab_n、total_steps），不带优化器
import sys, torch
out = sys.argv[1]
for p in sys.argv[2:]:
    b = torch.load(p, map_location="cpu", weights_only=False)
    v = b["version"]
    keep = {k: b[k] for k in ("version", "progress", "config", "obs_format", "vocab_n", "total_steps") if k in b}
    torch.save({**keep, "state": b["state"], "src": p}, f"{out}/v{v}.pt")
    ema = b.get("ema") or {}
    st = {k: (ema[k].to(t.dtype) if k in ema else t) for k, t in b["state"].items()}
    torch.save({**keep, "state": st, "src": p, "weights": "ema"}, f"{out}/m{v}.pt")
    print(v, p, "滑动平均覆盖", sum(k in ema for k in b["state"]), "/", len(b["state"]), flush=True)
