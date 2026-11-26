# 宽窗口权重平均（2026-11-20，铁甲战士决策记录第 19 条；故障机器人第 222 条、第 252 条 D3 的检查点权重平均）：
# 把若干份成员权重（ext_members.py 抽出的 v<版本>.pt）的 state 逐参数取算术平均，存成一个单网络，供 ability_test 当 --ckpt 用。
# 浮点参数按 float32 求平均后转回原类型；非浮点的（若有）取最后一个成员的。元数据（config、obs_format 等）取最后一个成员的。
#   python -I avg_members.py <输出文件> <成员列表文件（一行一个）>
import sys, torch
out, lst = sys.argv[1], sys.argv[2]
paths = [l.strip() for l in open(lst) if l.strip()]
acc, last = None, None
for p in paths:
    b = torch.load(p, map_location="cpu", weights_only=False)
    s = b["state"]
    if acc is None:
        acc = {k: (t.float().clone() if torch.is_floating_point(t) else t) for k, t in s.items()}
    else:
        assert s.keys() == acc.keys(), f"{p} 的参数名与第一个成员不同"
        for k, t in s.items():
            if torch.is_floating_point(t):
                acc[k] += t.float()
    last = b
n = len(paths)
avg = {k: ((acc[k] / n).to(last["state"][k].dtype) if torch.is_floating_point(last["state"][k]) else last["state"][k]) for k in acc}
keep = {k: last[k] for k in ("version", "progress", "config", "obs_format", "vocab_n", "total_steps", "content_names", "catalog_version") if k in last}
torch.save({**keep, "state": avg, "src": paths, "weights": f"avg{n}"}, out)
d = sum(float((avg[k].float() - last["state"][k].float()).norm() ** 2) for k in avg if torch.is_floating_point(avg[k])) ** 0.5
print(f"{n} 个成员的权重平均 → {out}；与最后一个成员（第 {last.get('version')} 版）的差（范数）{d:.3f}", flush=True)
