# 从发布的滑动平均权重续训（猎人决策记录第 17 条；用户：“胜率小模型练完以后，从蒸馏点续训吧。”）：
# 学习端检查点 latest.pt 的 state 换成它自己的 ema（缺的项照旧），优化器（Adam 一二阶矩）、Muon 动量、结果头与 ema 本身都留着，
# 版本号不变（第 24,721 版，接着从第 24,722 版走）。先写临时文件再改名；换完核对 state 与发布的 m24721.pt 逐项相同。
#   .venv/bin/python -I swap_ema.py latest.pt /root/ens_silent/ck/m24721.pt
import os, sys, torch
src, ref = sys.argv[1], sys.argv[2]
b = torch.load(src, map_location="cpu", weights_only=False)
ema = b.get("ema") or {}
st = b["state"]
n_swap = sum(k in ema for k in st)
b["state"] = {k: (ema[k].to(t.dtype) if k in ema else t) for k, t in st.items()}
b["resumed_from"] = {"weights": "ema", "version": b["version"], "note": "猎人决策记录第 17 条：从发布的滑动平均权重续训"}
r = torch.load(ref, map_location="cpu", weights_only=False)["state"]
same = all(torch.equal(b["state"][k], r[k]) for k in r) and set(r) == set(b["state"])
print(f"版本 {b['version']}：{n_swap}/{len(st)} 项换成滑动平均；与 {os.path.basename(ref)} 逐项相同：{same}；保留 {sorted(k for k in b if k not in ('state',))}", flush=True)
if not same:
    sys.exit("和发布的权重对不上，不写")
tmp = src + ".tmp"
torch.save(b, tmp); os.replace(tmp, src)
print("已写", src)
