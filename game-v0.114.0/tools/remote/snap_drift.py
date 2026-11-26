"""学习端（B）上运行：快照漂移的轨迹（决策记录 274 补记三，看板“快照漂移”两张图的数据）。

在一批固定的真实局面（学习端批次存档里抽的多选步）上，给每份定时备份的快照算一遍首选，缓存起来（每份只算一次）；
再加两个参照：学生 e（续训起点，锚定的老师）与 SL 快照池均匀 16 个的集成（模型池蒸馏的老师）。
输出 snap_drift.json：
  - points：每份快照的版本、时间、与学生 e / 集成老师 / 上一份快照首选不同的比例（全部、战斗、局外）；
  - map：最近 MAP_N 份快照加两个参照，两两首选不同比例的经典多维缩放二维坐标（点距 ≈ 决策差异），按时间连线就是漂移轨迹。
由 lite_stats.py 在有新备份时以低优先级在后台调起；单独运行：

    .venv/bin/python -I tools/remote/snap_drift.py [--since 165018]
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
RUN = Path.home() / ".local/share/sls2/runs/defect"
OUT = RUN / "snap_drift.json"
CACHE = RUN / "snap_drift"
DUMP = Path("/root/sd262/dump_5.pkl")
STUDENT_E = Path("/root/diag/pd3/student_e.pt")
TEACHER_LIST = Path("/root/diag/slpool_spread16.txt")
N_STATES = 4096
MAP_N = 40


def ver_of(p: Path) -> int | None:
    m = re.match(r"v(\d+)", p.name)
    return int(m.group(1)) if m else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", type=int, default=169360, help="只看这一版及以后的备份（决策记录 281 起：从学生 e 重起、按场自举的这一段；第 169,359 版有同号的旧备份，不取）")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    import torch
    from sts2rl.features import collate_ragged
    from sts2rl.model import ModelConfig, PolicyValueNet, load_state, packed_meta
    from sts2env.protocol import DecisionKind
    CACHE.mkdir(exist_ok=True)
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    sub = CACHE / "states.pkl"       # 抽好的 4,096 个局面另存一份小文件：以后每次不必再读 600 MB 的批次存档
    if sub.exists():
        data = pickle.load(open(sub, "rb"))
    else:
        r = pickle.load(open(DUMP, "rb"))
        multi = np.flatnonzero(np.diff(r.cand_off) > 1)
        data = r.select(np.sort(np.random.default_rng(0).choice(multi, min(N_STATES, len(multi)), replace=False)))
        pickle.dump(data, open(sub, "wb")); del r
    combat = (data.header[:, 11] != 0) | (data.header[:, 1] == int(DecisionKind.COMBAT))

    def probs(path: Path) -> np.ndarray:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        mc = blob["config"]["model"]
        net = PolicyValueNet(ModelConfig(**mc)).to(dev).eval(); load_state(net, blob["state"]); net.fast = dev.type == "cuda"
        out = []
        with torch.no_grad():
            for j in range(0, len(data), 512):
                bt = collate_ragged(data.select(np.arange(j, min(j + 512, len(data)))), attend=bool(mc.get("macro_attend"))); b = bt.to(dev)
                if net.fast:
                    b.update({k: (torch.from_numpy(v).to(dev) if isinstance(v, np.ndarray) else v) for k, v in packed_meta(bt.core_idx, bt.core_mask).items()})
                p = torch.softmax(net(b)[0].float(), -1).cpu().numpy()
                out.append(np.pad(p, ((0, 0), (0, 64 - p.shape[1]))) if p.shape[1] < 64 else p[:, :64])
        del net
        return np.concatenate(out)

    def cached(key: str, fn) -> np.ndarray:
        f = CACHE / f"{key}.npy"
        if f.exists():
            return np.load(f)
        x = fn(); np.save(f, x); return x

    ref_e = cached("student_e", lambda: probs(STUDENT_E).argmax(-1).astype(np.int16)) if STUDENT_E.exists() else None
    def teacher():
        ps = [probs(Path(l.strip())) for l in open(TEACHER_LIST) if l.strip()]
        return np.mean(ps, 0).argmax(-1).astype(np.int16)
    ref_t = cached("teacher16", teacher) if TEACHER_LIST.exists() else None
    snaps = {}
    for p in sorted((RUN / "backups").glob("v*.pt"), key=lambda q: q.stat().st_mtime):
        v = ver_of(p)
        if v is None or v < a.since or v in snaps:
            continue
        snaps[v] = (p, p.stat().st_mtime)
    acts = {}
    for v, (p, _) in sorted(snaps.items()):
        try:
            acts[v] = cached(f"v{v}", lambda p=p: probs(p).argmax(-1).astype(np.int16))
        except Exception as e:     # 半截的备份等：跳过，下次再试
            print(f"第 {v} 版算不了：{e}", flush=True)
    vs = sorted(acts)
    # “当前”点：学习端正在发布的最新权重（/dev/shm/sts2rl_*.pt），每次现算、不缓存，让曲线最右端不必等下一份定时备份
    live = None
    shm = sorted(Path("/dev/shm").glob("sts2rl_*.pt"), key=lambda q: q.stat().st_mtime)
    if shm:
        try:
            lb = torch.load(shm[-1], map_location="cpu", weights_only=False)
            lv = int(lb.get("version") or 0)
            if lv > (vs[-1] if vs else 0):
                # 发布的权重文件不带配置：取运行目录 train_config.json 里的网络配置
                mc = json.loads((RUN / "train_config.json").read_text(encoding="utf-8"))["model"]
                tmpf = CACHE / "_live.pt"; torch.save({"version": lv, "config": {"model": mc}, "state": lb["state"]}, tmpf)
                live = (lv, probs(tmpf).argmax(-1).astype(np.int16), shm[-1].stat().st_mtime)
                tmpf.unlink()
        except Exception as e:
            print(f"当前权重算不了：{e}", flush=True)

    def dis(x, y):
        d = x != y
        return [round(float(d.mean()), 4), round(float(d[combat].mean()), 4), round(float(d[~combat].mean()), 4)]
    points = []
    for i, v in enumerate(vs):
        points.append({"version": v, "time": round(snaps[v][1], 1),
                       "vs_e": dis(acts[v], ref_e) if ref_e is not None else None,
                       "vs_teacher": dis(acts[v], ref_t) if ref_t is not None else None,
                       "vs_prev": dis(acts[v], acts[vs[i - 1]]) if i else None})
    if live is not None:
        lv, la, lt = live
        points.append({"version": lv, "time": round(lt, 1), "live": True, "vs_e": dis(la, ref_e) if ref_e is not None else None,
                       "vs_teacher": dis(la, ref_t) if ref_t is not None else None, "vs_prev": dis(la, acts[vs[-1]]) if vs else None})
    # 轨迹地图：最近 MAP_N 份快照 + 两个参照，两两首选不同比例 → 经典多维缩放
    keys = [("e", ref_e)]        # 2026-11-06 起地图不再放集成老师（模型池蒸馏已不在关注范围）
    # 地图只画续训以后的快照：--since 那一份（如第 165,018 版）是蒸馏前的旧主线，离得太远会把其余点挤成一团（曲线上仍保留）
    keys = [(k, x) for k, x in keys if x is not None] + [(f"v{v}", acts[v]) for v in [v for v in vs if v > a.since][-MAP_N:]]
    if live is not None:
        keys.append((f"v{live[0]}", live[1]))
    mp = None
    if len(keys) >= 3:
        X = np.stack([x for _, x in keys])
        D = (X[:, None, :] != X[None, :, :]).mean(-1)
        n = len(D); J = np.eye(n) - 1.0 / n
        Bm = -0.5 * J @ (D ** 2) @ J
        w, U = np.linalg.eigh(Bm)
        o = np.argsort(w)[::-1][:2]
        xy = U[:, o] * np.sqrt(np.maximum(w[o], 0))
        mp = {"labels": [k for k, _ in keys], "xy": [[round(float(a_), 5), round(float(b_), 5)] for a_, b_ in xy],
              "explained": round(float(np.maximum(w[o], 0).sum() / max(np.maximum(w, 0).sum(), 1e-12)), 3)}
    res = {"time": time.time(), "n_states": int(len(data)), "n_combat": int(combat.sum()), "points": points, "map": mp,
           "refs": {"e": "学生 e（第 165,018 版续训起点）", "teacher": "SL 快照池均匀 16 个的集成"}}
    tmp = OUT.with_suffix(".tmp"); tmp.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8"); os.replace(tmp, OUT)
    print(f"快照 {len(vs)} 份：{vs[:2]}…{vs[-2:]}", flush=True)


if __name__ == "__main__":
    main()
