"""平台期诊断 D3（2026-11-03）：把几个检查点的权重逐项取平均，存成一个能直接给 tools/ability_test.py 用的检查点。
    python -I tools/diag_soup.py --out soup.pt a.pt b.pt c.pt
版本号记最后一个检查点的；只平均浮点张量，其余（如计数）取最后一个的。"""
import argparse

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("ckpts", nargs="+")
a = ap.parse_args()
blobs = [torch.load(p, map_location="cpu", weights_only=False) for p in a.ckpts]
out = dict(blobs[-1])
st = {}
for k, v in blobs[-1]["state"].items():
    st[k] = torch.stack([b["state"][k].float() for b in blobs]).mean(0).to(v.dtype) if torch.is_floating_point(v) else v
out["state"] = st
out["note"] = f"权重平均：{[int(b['version']) for b in blobs]}"
torch.save(out, a.out)
print(out["note"], "→", a.out)
