"""S3（真实所选动作的优势式读数头）的数据量缩放检查：同一批老师留出根上，训练数据取 1/8、1/4、1/2、全部，看候选差相关是否随数据增长。"""
import json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import heads as Hd
from future_heads import ResidualHead
dev = "cuda"
D = torch.load(sys.argv[1], weights_only=False); T = torch.load(sys.argv[2], weights_only=False)
R = Hd.root_tables(D); te = [t for t in R if t["id"] % 5 == 0]
keep = (T["ids"] % 5 != 0)
Z, Y, ids = T["Z"][keep], T["Y"][keep], T["ids"][keep]
probs = torch.unique(ids)
out = {}
for hidden, epochs in ((128, 20), (256, 30)):
    for frac in (0.125, 0.25, 0.5, 1.0):
        torch.manual_seed(0)
        pick = probs[torch.randperm(len(probs))[: max(1, int(len(probs) * frac))]]       # 按题取子集（同一题的步不拆开）
        m = torch.isin(ids, pick)
        Z3, Y3 = Z[m].to(dev), Y[m].to(dev)
        h = ResidualHead(hidden=hidden, out=Hd.M, kind="pred", zero=False).to(dev)
        o = torch.optim.AdamW(h.parameters(), lr=1e-3, weight_decay=1e-4)
        mu, sd = Y3.mean(0), Y3.std(0) + 1e-3
        for ep in range(epochs):
            perm = torch.randperm(len(Y3), device=dev)
            for s in range(0, len(perm), 1024):
                k = perm[s:s + 1024]
                loss = ((h(Z3[k]) - (Y3[k] - mu) / sd) ** 2).mean()
                o.zero_grad(); loss.backward(); o.step()
        f = Hd.diff_fit(lambda z: h(z)[:, 0], te, D, dev)
        out[f"h{hidden}_f{frac}"] = {"steps": int(m.sum()), **f}
        print(hidden, frac, int(m.sum()), f, flush=True)
Path(sys.argv[3]).write_text(json.dumps(out, ensure_ascii=False, indent=1))
