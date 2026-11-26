"""PIMC 推理侧提速的测量与核对（2026-11-16）。不改 tools/future/boss_pimc.py：借它的 Runner，只把评估器（self.ev）换掉。

  record：跑 boss_pimc 的题，把每次过网络的帧（推演帧与根帧）原样记下来，作为离线对照的固定输入。
      .venv/bin/python -I tools/pimc_infer.py record --frames 帧.pkl -- --items items.json --ckpt L1000.pt --out 目录 --limit 2 --device cpu
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "tools" / "future"))
sys.path.insert(0, str(REPO / "python"))


def _obs_tuple(o) -> tuple:
    return (o.header, o.tok_ints, o.tok_feats, o.tok_priv, o.cand_ints, o.cand_feats)


class Recorder:
    """包一层评估器：每次 forward 把帧与模式记进 self.batches（根帧是单帧的 forward，推演帧是 rollout 合成的一批）。"""

    def __init__(self, ev):
        self.ev = ev
        self.batches: list[dict] = []
        self.kind = "root"

    def forward(self, frames, modes):
        self.batches.append({"kind": self.kind, "modes": list(map(int, modes)), "frames": [_obs_tuple(o) for o in frames]})
        return self.ev.forward(frames, modes)

    def policy(self, o, mode=None):
        return self.ev.policy(o, mode)

    def rollout(self, frames, leaf, keys):
        self.kind = "roll"
        try:
            # 与 ModelEvaluator.rollout 相同的模式规则，只是经过 self.forward 记下来
            if not frames:
                return [], []
            logits, values = self.forward(frames, [self.ev.value_mode if l else self.ev.policy_mode for l in leaf])
            self.batches[-1]["leaf"] = list(map(bool, leaf))
            acts = [0 if l else int(logits[i, : o.num_actions].argmax()) for i, (o, l) in enumerate(zip(frames, leaf))]
            return acts, [float(values[i]) for i in range(len(frames))]
        finally:
            self.kind = "root"

    def __getattr__(self, k):
        return getattr(self.ev, k)


def _boss_pimc_main(argv: list[str], patch) -> None:
    """用 boss_pimc 自己的命令行跑一遍 run，Runner 换成 patch(原 Runner) 返回的子类。"""
    import boss_pimc as BP
    BP.Runner = patch(BP.Runner)
    sys.argv = ["boss_pimc.py", "run", *argv]
    BP.main()


def cmd_record(a, rest) -> None:
    holder = {}

    def patch(Base):
        class R(Base):
            def __init__(self, ba):
                super().__init__(ba)
                self.ev = Recorder(self.ev)
                holder["r"] = self
        return R

    t0 = time.time()
    _boss_pimc_main(rest, patch)
    rec = holder["r"].ev
    roll = [b for b in rec.batches if b["kind"] == "roll"]
    n_roll = sum(len(b["frames"]) for b in roll)
    n_root = sum(len(b["frames"]) for b in rec.batches if b["kind"] == "root")
    with open(a.frames, "wb") as f:
        pickle.dump({"batches": rec.batches, "argv": rest}, f, protocol=5)
    print(f"记下 {len(rec.batches)} 次前向：推演 {len(roll)} 批 {n_roll} 帧（每批平均 {n_roll / max(1, len(roll)):.2f}），根帧 {n_root}；"
          f"{time.time() - t0:.0f} 秒 → {a.frames}（{os.path.getsize(a.frames) / 1e6:.1f} MB）")


# ---------------------------------------------------------------- 在 boss_pimc 的题上跑：换推理路径（play）与逐根双跑核对（agree）

MODE_COMBAT = 0


def _ev(spec, net, device):
    import sim_pimc as X
    return X.make_evaluator(spec, net, device, policy_mode=MODE_COMBAT, value_mode=MODE_COMBAT, strip_priv=True)


def cmd_play(a, rest) -> None:
    """照 boss_pimc run 打题，推理换成 --ev（例 bf16、unix:/tmp/x.sock）；结果文件同 boss_pimc，每题另记推理的行数、调用次数与耗时。
    用共用推理进程时 boss_pimc 的 --device 给 cpu（分片里不建显卡上下文，载入的网络不用）。"""

    def patch(Base):
        class R(Base):
            def __init__(self, ba):
                super().__init__(ba)
                self.ev = _ev(a.ev, self.net, ba.device)

            def play(self, item, lab_fh, frame_fh):
                n0, c0, t0 = self.ev.n_rows, self.ev.n_fwd, self.ev.t_fwd
                out = super().play(item, lab_fh, frame_fh)
                out["infer"] = {"ev": a.ev, "rows": self.ev.n_rows - n0, "calls": self.ev.n_fwd - c0, "s": round(self.ev.t_fwd - t0, 3)}
                return out
        return R

    _boss_pimc_main(rest, patch)


class _Dual:
    """agree 用：rollout 与 forward 都交给当前选中的那一个（ref 或 new）；单帧的根 forward 两个都算，记下 new 的根概率。"""

    def __init__(self, ref, new):
        self.ref, self.new, self.cur = ref, new, ref
        self.root_new = None

    def forward(self, frames, modes):
        if len(frames) == 1 and self.cur is self.ref:
            lg, v = self.new.forward(frames, modes)
            self.root_new = (lg, v)
        return self.cur.forward(frames, modes)

    def rollout(self, frames, leaf, keys):
        return self.cur.rollout(frames, leaf, keys)

    def __getattr__(self, k):
        return getattr(self.cur, k)


def cmd_agree(a, rest) -> None:
    """逐根双跑：实机照 --ref（默认原 fp32 路径）的决策走；每个要决策的根上，new（--ev）用自己的根概率决定搜不搜、
    搜就用同样的世界种子再跑一遍搜索，记下两边最终选的动作。输出 agree_s{分片}.jsonl（每根一行）。"""
    import json
    import torch

    def patch(Base):
        class R(Base):
            def __init__(self, ba):
                super().__init__(ba)
                self.dual = _Dual(_ev(a.ref, self.net, ba.device), _ev(a.ev, self.net, ba.device))
                self.ev = self.dual
                self.afh = open(Path(ba.out) / f"agree_s{ba.k_shard}.jsonl", "a", encoding="utf-8")
                self.item = None
                self.t = 0

            def play(self, item, lab_fh, frame_fh):
                self.item, self.t = item, 0
                out = super().play(item, lab_fh, frame_fh)
                self.afh.flush()
                return out

            def gated(self, p):
                # 每一步实机决策都会走到这里（单候选的步除外）：p 是 ref 的根概率，new 的在 self.dual.root_new
                lg = self.dual.root_new[0][0, : len(p)].double()
                pn = torch.softmax(lg, -1).numpy()
                g_ref, g_new = Base.gated(self, p), Base.gated(self, pn)
                self._pending = (p, pn, g_ref, g_new)
                return g_ref or g_new

            def search(self, base, hist, root, p, dec_seed):
                p_ref, pn, g_ref, g_new = self._pending
                rec = {"id": self.item["id"], "t": len(hist), "n": int(len(p)), "g_ref": g_ref, "g_new": g_new,
                       "greedy_ref": int(np.argmax(p_ref)), "greedy_new": int(np.argmax(pn)),
                       "dp_max": float(np.max(np.abs(p_ref - pn)))}
                res_ref = res_new = None
                if g_ref:
                    self.dual.cur = self.dual.ref
                    res_ref = Base.search(self, base, hist, root, p_ref, dec_seed)
                if g_new:
                    self.dual.cur = self.dual.new
                    try:
                        res_new = Base.search(self, base, hist, root, pn, dec_seed)
                    finally:
                        self.dual.cur = self.dual.ref
                # 贪心项以搜索自己的为准（boss_pimc 用 argsort，并列时可能与 argmax 取不同的下标）
                if res_ref:
                    rec["greedy_ref"] = int(res_ref["greedy"])
                if res_new:
                    rec["greedy_new"] = int(res_new["greedy"])
                b_ref = res_ref["best"] if res_ref else rec["greedy_ref"]
                b_new = res_new["best"] if res_new else rec["greedy_new"]
                # 两个下标在两边的根概率都逐位相同，视为同一个动作（两张一样的牌，输入相同、打分逐位相同）
                twin = bool(b_ref != b_new and p_ref[b_ref] == p_ref[b_new] and pn[b_ref] == pn[b_new])
                rec.update(best_ref=int(b_ref), best_new=int(b_new), same=int(b_ref == b_new or twin), twin=int(twin))
                for k, r in (("ref", res_ref), ("new", res_new)):
                    if r:
                        rec[f"diff_{k}"] = r["diff"]; rec[f"ms_{k}"] = r["ms"]; rec[f"cands_{k}"] = r["cands"]
                        rec[f"scores_{k}"] = [[m[0], m[1], m[2]] for m in r["meas"]]
                self.afh.write(json.dumps(rec) + "\n")
                if res_ref is None:     # 只有 new 要搜：实机照 ref 的贪心走，标签里也只记 ref
                    return {"best": int(b_ref), "greedy": rec["greedy_ref"], "cands": [], "diff": {}, "meas": [], "errors": 0, "ms": 0.0}
                return res_ref
        return R

    _boss_pimc_main(rest, patch)


def _load_arm(d: str) -> dict:
    """一臂（c_online 式目录：res_s*.jsonl、labels_s*.jsonl、WALL、server_stats.json）的结果。"""
    import glob
    import json
    res, lab = {}, {}
    for f in glob.glob(os.path.join(d, "res_s*.jsonl")):
        for l in open(f, encoding="utf-8"):
            if l.strip():
                r = json.loads(l); res[(r["id"], r.get("rep", 0))] = r
    for f in glob.glob(os.path.join(d, "labels_s*.jsonl")):
        for l in open(f, encoding="utf-8"):
            if l.strip():
                r = json.loads(l); lab[(r["id"], r["t"])] = r
    wall = None
    if os.path.exists(os.path.join(d, "WALL")):
        for l in open(os.path.join(d, "WALL")):
            if l.startswith("wall"):
                wall = float(l.split()[1])
    st = None
    if os.path.exists(os.path.join(d, "server_stats.json")):
        import json as _j
        st = _j.load(open(os.path.join(d, "server_stats.json")))
    return {"res": res, "lab": lab, "wall": wall, "server": st}


def cmd_report(a, rest) -> None:
    """汇总在线对比：每臂的吞吐（推理帧 / 墙钟秒）、单次搜索耗时、胜率；每臂与第一臂（参照）配对：
    同一题的胜负、根帧逐字节相同的根上改选是否一致（只有两边走到同一局面才可比）。"""
    import json
    arms = [(p.split("=", 1) if "=" in p else (os.path.basename(p.rstrip("/")), p)) for p in a.arms]
    data = {n: _load_arm(d) for n, d in arms}
    ref_name = arms[0][0]
    out = {}
    for n, _ in arms:
        D = data[n]
        res, lab = D["res"], D["lab"]
        ms = np.array([r["ms"] for r in lab.values()]) if lab else np.zeros(0)
        rows = sum(r.get("infer", {}).get("rows", 0) for r in res.values())
        calls = sum(r.get("infer", {}).get("calls", 0) for r in res.values())
        wins = sum(r["win"] for r in res.values())
        o = {"items": len(res), "wins": wins, "win_rate": wins / max(1, len(res)), "searches": len(lab),
             "changed": sum(int(r["best"] != r["greedy"]) for r in lab.values()),
             "search_ms_mean": float(ms.mean()) if len(ms) else None, "search_ms_p50": float(np.median(ms)) if len(ms) else None,
             "rows": rows, "calls": calls, "rows_per_call": rows / max(1, calls), "wall": D["wall"],
             "fps": None,
             "item_s_mean": float(np.mean([r["s"] for r in res.values()])) if res else None}
        # 吞吐：各分片依次打题，分片 k 的帧 / 秒 = 它已完成各题的推理帧数之和 ÷ 用时之和；全机 = 各分片相加（跑满窗口就停也不受影响）
        per = {}
        for (iid, _), r in res.items():
            k = iid % a.shards
            pr = per.setdefault(k, [0, 0.0, 0.0])
            pr[0] += r.get("infer", {}).get("rows", 0); pr[1] += r["s"]; pr[2] += r.get("infer", {}).get("s", 0.0)
        o["fps"] = sum(v[0] / v[1] for v in per.values() if v[1] > 0)
        o["infer_share"] = sum(v[2] for v in per.values()) / max(1e-9, sum(v[1] for v in per.values()))
        o["infer_ms_per_call"] = 1000 * sum(r.get("infer", {}).get("s", 0.0) for r in res.values()) / max(1, calls)
        if D["server"]:
            s = D["server"]; b = max(1, s["batches"])
            o["server"] = {"batches": s["batches"], "rows_per_batch": s["rows"] / b, "busy": (s["t_collate"] + s["t_fwd"] + s["t_send"] + s["t_recv"]) / max(1e-9, s["uptime"]),
                           "collate_ms": 1000 * s["t_collate"] / b, "fwd_ms": 1000 * s["t_fwd"] / b}
        if n != ref_name:
            R = data[ref_name]
            common = sorted(set(res) & set(R["res"]))
            wa = sum(R["res"][k]["win"] for k in common); wb = sum(res[k]["win"] for k in common)
            only_a = sum(1 for k in common if R["res"][k]["win"] and not res[k]["win"])
            only_b = sum(1 for k in common if res[k]["win"] and not R["res"][k]["win"])
            same_traj = sum(1 for k in common if R["res"][k].get("steps") == res[k].get("steps") and R["res"][k].get("hp") == res[k].get("hp"))
            keys = sorted(set(lab) & set(R["lab"]))
            same_root = [k for k in keys if lab[k]["frame"] == R["lab"][k]["frame"]]
            def _same(x, y):
                # 下标相同，或两边的根概率（标签里保留 5 位小数）在这两个下标上都相等：两张一样的牌，并列时取了不同的下标
                bx, by = x["best"], y["best"]
                return bx == by or (x["p"][bx] == x["p"][by] and y["p"][bx] == y["p"][by])
            def _tied(x, y):
                sc: dict = {}
                for m in y.get("meas", []):
                    sc.setdefault(m[0], {})[m[1]] = m[2]
                a, b = sc.get(x["best"], {}), sc.get(y["best"], {})
                common = set(a) & set(b)
                return len(common) >= 2 and all(abs(a[k] - b[k]) <= 1e-3 for k in common)
            agree = sum(1 for k in same_root if _same(lab[k], R["lab"][k]))
            tie = sum(1 for k in same_root if not _same(lab[k], R["lab"][k]) and _tied(lab[k], R["lab"][k]))
            chg_ref = sum(1 for k in same_root if R["lab"][k]["best"] != R["lab"][k]["greedy"])
            chg_both = sum(1 for k in same_root if R["lab"][k]["best"] != R["lab"][k]["greedy"] and _same(lab[k], R["lab"][k]))
            o["vs_ref"] = {"common_items": len(common), "wins_ref": wa, "wins": wb, "only_ref_wins": only_a, "only_this_wins": only_b,
                           "same_end": same_traj, "roots_same_frame": len(same_root), "best_agree": agree / max(1, len(same_root)),
                           "ref_changed": chg_ref, "ref_changed_kept": chg_both, "tied": tie, "agree_or_tied": (agree + tie) / max(1, len(same_root))}
        out[n] = o
    for n, o in out.items():
        line = (f"{n:12s} 题 {o['items']:4d}  胜 {o['wins']:4d}（{100 * o['win_rate']:.1f}%）  搜索 {o['searches']:6d}（改选 {o['changed']}）"
                f"  单次搜索 平均 {o['search_ms_mean'] or 0:8.1f} / 中位 {o['search_ms_p50'] or 0:8.1f} 毫秒  推理 {o['rows']} 帧 / {o['calls']} 次"
                f"（{o['rows_per_call']:.2f} 帧 / 次，{o['infer_ms_per_call']:.2f} 毫秒 / 次，占题时 {100 * o['infer_share']:.0f}%）  全机 {o['fps']:.0f} 帧/秒")
        print(line)
        if "server" in o:
            s = o["server"]
            print(f"{'':12s} 推理进程：{s['batches']} 批、{s['rows_per_batch']:.1f} 帧 / 批，组批 {s['collate_ms']:.2f} + 前向 {s['fwd_ms']:.2f} 毫秒 / 批，忙 {100 * s['busy']:.0f}%")
        if "vs_ref" in o:
            v = o["vs_ref"]
            print(f"{'':12s} 对 {ref_name}：共同 {v['common_items']} 题，胜 {v['wins_ref']} → {v['wins']}（只参照赢 {v['only_ref_wins']}、只这臂赢 {v['only_this_wins']}），"
                  f"结局相同 {v['same_end']} 题；根帧相同的根 {v['roots_same_frame']} 个，改选一致 {100 * v['best_agree']:.2f}%，加上结局等价的 {v['tied']} 个为 {100 * v['agree_or_tied']:.2f}%"
                  f"（参照改选的 {v['ref_changed']} 个里保留 {v['ref_changed_kept']}）")
    if a.out:
        Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return max(0.0, c - h), min(1.0, c + h)


def cmd_agree_report(a, rest) -> None:
    """汇总逐根双跑（agree）：每个实机决策的根上，参照与新路径最终选的动作是否相同；分开看两边都搜了的根、只一边搜了的根（门限两侧），
    以及参照改了选的根新路径是否也改成同一个。区间为 Wilson 95%。"""
    import glob
    import json
    out = {}
    for spec in a.dirs:
        name, d = spec.split("=", 1) if "=" in spec else (os.path.basename(spec.rstrip("/")), spec)
        rows = [json.loads(l) for f in sorted(glob.glob(os.path.join(d, "agree_s*.jsonl"))) for l in open(f, encoding="utf-8") if l.strip()]
        def tied(r):
            """两边选的动作不同，但在参照的搜索里这两个候选在每个共同世界的得分都相同（差 ≤ 1e-3）：结局等价。
            boss_pimc 的改选判据是 m > z·se，两个候选结局相同时 se 为 0，平均差只剩 ±1e-8 的浮点噪声，选哪个就看噪声的正负。"""
            if r["same"] or "scores_ref" not in r:
                return False
            sc: dict = {}
            for c, sd, v in r["scores_ref"]:
                sc.setdefault(c, {})[sd] = v
            a, b = sc.get(r["best_ref"], {}), sc.get(r["best_new"], {})
            common = set(a) & set(b)
            return len(common) >= 2 and all(abs(a[k] - b[k]) <= 1e-3 for k in common)
        for r in rows:
            r["tied"] = int(tied(r))
        both = [r for r in rows if r["g_ref"] and r["g_new"]]
        one = [r for r in rows if r["g_ref"] != r["g_new"]]
        chg = [r for r in both if r["best_ref"] != r["greedy_ref"]]
        same = sum(r["same"] for r in rows)
        sb = sum(r["same"] for r in both)
        items = len({r["id"] for r in rows})
        dpm = np.array([r["dp_max"] for r in rows]) if rows else np.zeros(1)
        tie = sum(r["tied"] for r in rows)
        eq = same + tie
        o = {"items": items, "roots": len(rows), "agree": same / max(1, len(rows)), "agree_ci": _wilson(same, len(rows)),
             "tied": tie, "agree_or_tied": eq / max(1, len(rows)), "agree_or_tied_ci": _wilson(eq, len(rows)),
             "both_searched": len(both), "agree_searched": sb / max(1, len(both)), "agree_searched_ci": _wilson(sb, len(both)),
             "gate_differs": len(one), "greedy_differs": sum(r["greedy_ref"] != r["greedy_new"] for r in rows),
             "ref_changed": len(chg), "ref_changed_kept": sum(r["same"] for r in chg),
             "new_changed": sum(r["best_new"] != r["greedy_new"] for r in both),
             "root_dp_max": {"p50": float(np.median(dpm)), "p99": float(np.quantile(dpm, 0.99)), "max": float(dpm.max())}}
        # 世界级：两边都推演过的同一（候选、种子）世界，得分是否相同（差 ≤ 1e-3 记为相同：叶子估值的浮点差在 1e-5 量级，
        # 差得多说明这个世界的推演中途某一步最高项翻了、后面整条走岔）
        wd = []
        for r in both:
            sr = {(m[0], m[1]): m[2] for m in r.get("scores_ref", [])}
            for m in r.get("scores_new", []):
                if (m[0], m[1]) in sr:
                    wd.append(abs(m[2] - sr[(m[0], m[1])]))
        wd = np.array(wd) if wd else np.zeros(0)
        o["worlds"] = int(len(wd)); o["worlds_diverged"] = float((wd > 1e-3).mean()) if len(wd) else None
        o["world_dscore_mean"] = float(wd.mean()) if len(wd) else None
        ms_r = [r["ms_ref"] for r in both if "ms_ref" in r]; ms_n = [r["ms_new"] for r in both if "ms_new" in r]
        o["ms_ref_mean"] = float(np.mean(ms_r)) if ms_r else None; o["ms_new_mean"] = float(np.mean(ms_n)) if ms_n else None
        out[name] = o
        lo, hi = o["agree_searched_ci"]
        el, eh = o["agree_or_tied_ci"]
        print(f"{name:10s} {items} 题 {len(rows)} 个根：最终动作一致 {100 * o['agree']:.2f}%，加上结局等价的 {tie} 个为 {100 * o['agree_or_tied']:.2f}%"
              f"（{100 * el:.2f}–{100 * eh:.2f}%）；两边都搜的 {len(both)} 个根一致 {100 * o['agree_searched']:.2f}%"
              f"（95% 区间 {100 * lo:.2f}–{100 * hi:.2f}%）；门限两侧各搜一边 {len(one)} 个、贪心项不同 {o['greedy_differs']} 个；"
              f"参照改选 {len(chg)} 个、新路径同样改成它 {o['ref_changed_kept']} 个，新路径改选 {o['new_changed']} 个；根概率最大差 p99 {o['root_dp_max']['p99']:.1e}；"
              f"同一世界 {o['worlds']} 个、得分不同 {100 * (o['worlds_diverged'] or 0):.1f}%（平均差 {o['world_dscore_mean'] or 0:.3f}）")
    if a.out:
        Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


# ---------------------------------------------------------------- 离线：同一批帧上的差别与测速

def load_frames(path: str) -> list[dict]:
    with open(path, "rb") as f:
        return pickle.load(f)["batches"]


def ragged_of(batch: dict):
    """记下的一批帧 → Ragged（特权特征已按 strip_priv 清零，与 ModelEvaluator 送进网络的相同）。"""
    from sts2rl.features import Ragged
    fr = batch["frames"]
    tl = np.array([0] + [f[1].shape[0] for f in fr]); cl = np.array([0] + [f[4].shape[0] for f in fr])
    r = Ragged(np.stack([f[0] for f in fr]), np.concatenate([f[1] for f in fr]), np.concatenate([f[2] for f in fr]),
               np.concatenate([f[3] for f in fr]), np.cumsum(tl), np.concatenate([f[4] for f in fr]), np.concatenate([f[5] for f in fr]), np.cumsum(cl))
    r.mode = np.asarray(batch["modes"], np.int64)
    r.tok_priv = np.zeros_like(r.tok_priv)
    return r


def rows_of(batches: list[dict]) -> list[dict]:
    """逐帧展开：每帧一行（所在批、批内序号、种类 root / roll、是否叶子、候选数）。"""
    out = []
    for bi, b in enumerate(batches):
        leaf = b.get("leaf") or [False] * len(b["frames"])
        for i, f in enumerate(b["frames"]):
            out.append({"b": bi, "i": i, "kind": b["kind"], "leaf": bool(leaf[i]), "n": int(f[0][3])})
    return out


def reference(net, batches, device="cpu"):
    """原来的推理路径（ModelEvaluator 默认：net(batch) + no_grad，fp32），按记下时的批次。返回逐帧的 (对数几率列表, 价值)。"""
    import torch
    from sts2rl.features import collate_ragged
    lg_all, v_all = [], []
    with torch.no_grad():
        for b in batches:
            lg, v = net(collate_ragged(ragged_of(b)).to(device))
            lg, v = lg.float().cpu().numpy(), v.float().cpu().numpy()
            for i, f in enumerate(b["frames"]):
                lg_all.append(lg[i, : int(f[0][3])].copy()); v_all.append(float(v[i]))
    return lg_all, v_all


def run_config(spec: str, net, batches, device, rebatch: int = 0):
    """按 spec（sim_pimc.make_evaluator 的写法，"unix:" 除外）跑同一批帧。rebatch > 0 时把所有帧按顺序重新切成每批 rebatch 帧
    （模拟共用推理进程里的大批；同批里帧的组合与原来不同）。返回逐帧结果（顺序同 rows_of）与耗时。"""
    import sim_pimc as X
    from sts2rl.features import Ragged
    ev = X.make_evaluator(spec, net, device)
    rs = [ragged_of(b) for b in batches]
    if rebatch:
        allr = Ragged.concat(rs)
        idx = np.arange(len(allr))
        rs = [allr.select(idx[k:k + rebatch]) for k in range(0, len(allr), rebatch)]
    lg_all, v_all = [], []
    t0 = time.perf_counter()
    for r in rs:
        lg, v = ev._run(r)
        lg, v = lg.numpy(), v.numpy()
        n = np.diff(r.cand_off)
        for i in range(len(r)):
            lg_all.append(lg[i, : int(r.header[i, 3])].copy()); v_all.append(float(v[i]))
    return lg_all, v_all, time.perf_counter() - t0


def compare(ref, new, rows) -> dict:
    """逐帧比较：对数几率（只看有效候选）与概率的最大差、价值差、最高项是否相同（推演里要出手的帧与根帧；叶子帧只看价值）。"""
    rl, rv = ref; nl, nv = new
    dl, dp, dv, flips, gaps, n_pol, ties = [], [], [], [], [], 0, 0
    for k, row in enumerate(rows):
        a, b = rl[k], nl[k]
        dv.append(abs(rv[k] - nv[k]))
        if len(a) == 0:
            continue
        dl.append(float(np.max(np.abs(a - b))))
        pa = np.exp(a - a.max()); pa /= pa.sum(); pb = np.exp(b - b.max()); pb /= pb.sum()
        dp.append(float(np.max(np.abs(pa - pb))))
        if row["leaf"] or len(a) <= 1:
            continue
        n_pol += 1
        ia, ib = int(a.argmax()), int(b.argmax())
        if ia != ib:
            if a[ia] == a[ib]:      # 原路径里这两项本就逐位相同（两张一样的牌）：只是并列时取了不同的下标，不算翻转
                ties += 1
                continue
            s = np.sort(a)
            flips.append(k); gaps.append(float(s[-1] - s[-2]))
    q = lambda x: {"p50": float(np.quantile(x, 0.5)), "p99": float(np.quantile(x, 0.99)), "max": float(np.max(x)), "mean": float(np.mean(x))} if len(x) else {}
    leaf_dv = [dv[k] for k, r in enumerate(rows) if r["leaf"]]
    return {"frames": len(rows), "policy_frames": n_pol, "argmax_flips": len(flips), "argmax_ties": ties, "agree": 1 - len(flips) / max(1, n_pol),
            "flip_gap_ref": gaps[:20], "dlogit": q(dl), "dprob": q(dp), "dvalue": q(dv), "dvalue_leaf": q(leaf_dv)}


def cmd_diff(a, rest) -> None:
    """同一批帧：各推理路径与原来的 fp32 路径的差别。"""
    import json
    import torch
    from sts2rl.infer import load_net
    torch.manual_seed(0)
    batches = load_frames(a.frames)
    if a.max_batches:
        batches = batches[: a.max_batches]
    rows = rows_of(batches)
    net = load_net(a.ckpt, a.device, ema=a.ema)
    t0 = time.perf_counter()
    ref = reference(net, batches, a.device)
    print(f"{len(rows)} 帧（{len(batches)} 批）；原路径 {time.perf_counter() - t0:.1f} 秒", flush=True)
    out = {}
    for spec in a.specs.split(";"):
        for rb in [0] + [int(x) for x in a.rebatch.split(",") if x]:
            lg, v, dt = run_config(spec, net, batches, a.device, rb)
            res = compare(ref, (lg, v), rows)
            res["seconds"] = round(dt, 2)
            key = f"{spec} 批{'同原' if not rb else rb}"
            out[key] = res
            print(f"{key:28s} 一致 {res['agree']:.5f}（翻 {res['argmax_flips']}/{res['policy_frames']}，并列换下标 {res['argmax_ties']}） "
                  f"Δ对数几率 p99 {res['dlogit'].get('p99', 0):.2e} 最大 {res['dlogit'].get('max', 0):.2e}  "
                  f"Δ概率 最大 {res['dprob'].get('max', 0):.2e}  Δ价值 p99 {res['dvalue']['p99']:.2e} 最大 {res['dvalue']['max']:.2e}  {dt:.1f} 秒", flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


def cmd_bench(a, rest) -> None:
    """离线测速：所有推演帧按顺序切成每批 B 帧，测组批（numpy）与前向（含拷到显卡、拷回）的耗时与帧 / 秒。"""
    import json
    import sim_pimc as X
    from sts2rl.features import Ragged, collate_ragged
    from sts2rl.infer import load_net
    batches = [b for b in load_frames(a.frames) if b["kind"] == "roll"]
    allr = Ragged.concat([ragged_of(b) for b in batches])
    N = len(allr)
    net = load_net(a.ckpt, a.device, ema=a.ema)
    out = {}
    for spec in a.specs.split(";"):
        ev = X.make_evaluator(spec, net, a.device)
        for B in [int(x) for x in a.sizes.split(",")]:
            n_b = max(1, min(a.batches, N // B))
            starts = [(k * B) % max(1, N - B) for k in range(n_b + a.warm)]
            rs = [allr.select(np.arange(s, s + B)) for s in starts]
            for r in rs[: a.warm]:
                ev._run(r)
            tc = tf = 0.0
            for r in rs[a.warm:]:
                t0 = time.perf_counter()
                bt = collate_ragged(r)
                t1 = time.perf_counter()
                if ev.engine is not None:
                    ev.engine.run_batch(bt)
                else:
                    import torch
                    with torch.no_grad():
                        lg, v = ev.net(bt.to(a.device))
                        lg.float().cpu(); v.float().cpu()
                t2 = time.perf_counter()
                tc += t1 - t0; tf += t2 - t1
            n = n_b * B
            res = {"B": B, "batches": n_b, "collate_ms": 1000 * tc / n_b, "fwd_ms": 1000 * tf / n_b, "fps": n / (tc + tf), "fps_fwd": n / tf}
            out[f"{spec}|{B}"] = res
            print(f"{spec:24s} B={B:4d}  组批 {res['collate_ms']:7.2f} 毫秒  前向 {res['fwd_ms']:7.2f} 毫秒  {res['fps']:8.0f} 帧/秒（只算前向 {res['fps_fwd']:8.0f}）", flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


# ---------------------------------------------------------------- 共用推理进程与压测

def cmd_serve(a, rest) -> None:
    """起共用推理进程（一台机器一个），搜索分片用 --ev unix:套接字 接上。"""
    import json
    from sts2rl.infer import InferEngine, load_net, parse_spec, serve
    net = load_net(a.ckpt, a.device, ema=a.ema)
    eng = InferEngine(net, a.device, **parse_spec(a.ev))
    st = serve(eng, a.sock, wait_ms=a.wait_ms, max_rows=a.max_rows, log=lambda s: print(s, flush=True), depth=a.depth)
    print(json.dumps(st, ensure_ascii=False), flush=True)


def _load_worker(k, a, q):
    """压测的一个客户端（模拟一个搜索分片）：按记下的推演批次依次请求，每次之间可加“想”的时间（模拟推演进程走一步）。"""
    import torch
    torch.set_num_threads(a.client_threads)
    import sim_pimc as X
    from sts2rl.infer import load_net
    batches = [b for b in load_frames(a.frames) if b["kind"] == "roll"]
    rs = [ragged_of(b) for b in batches]
    rs = rs[(k * 997) % len(rs):] + rs[: (k * 997) % len(rs)]
    if a.ev.startswith("unix:"):
        ev = X.make_evaluator(a.ev, None)
    else:
        ev = X.make_evaluator(a.ev, load_net(a.ckpt, a.device, ema=a.ema), a.device)
    for r in rs[:3]:
        ev._run(r)
    q.put(("ready", k))
    t_end = time.time() + a.seconds
    lat, rows, i = [], 0, 0
    t0 = time.time()
    while time.time() < t_end:
        r = rs[i % len(rs)]; i += 1
        s = time.perf_counter()
        ev._run(r)
        lat.append(time.perf_counter() - s)
        rows += len(r)
        if a.think_ms:
            time.sleep(a.think_ms / 1000.0)
    q.put(("done", k, rows, time.time() - t0, lat))


def cmd_load(a, rest) -> None:
    """K 个客户端同时请求（每个客户端模拟一个搜索分片），测总帧 / 秒与每次请求的延迟。
    --ev unix:套接字 时所有客户端共用推理进程；否则每个客户端各载一份模型（改前的做法：每个分片各自前向）。"""
    import json
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    ps = [ctx.Process(target=_load_worker, args=(k, a, q), daemon=True) for k in range(a.clients)]
    for p in ps:
        p.start()
    ready = 0
    while ready < a.clients:
        m = q.get()
        ready += m[0] == "ready"
    res = [q.get() for _ in range(a.clients)]
    for p in ps:
        p.join()
    rows = sum(r[2] for r in res)
    sec = max(r[3] for r in res)
    lat = np.concatenate([np.array(r[4]) for r in res]) * 1000
    out = {"ev": a.ev, "clients": a.clients, "think_ms": a.think_ms, "fps": rows / sec, "reqs": int(len(lat)),
           "lat_ms": {"p50": float(np.median(lat)), "p90": float(np.quantile(lat, 0.9)), "p99": float(np.quantile(lat, 0.99)), "mean": float(lat.mean())}}
    print(f"{a.ev:30s} 客户端 {a.clients:3d}  想 {a.think_ms:4.1f} 毫秒  {out['fps']:8.0f} 帧/秒  请求 {out['reqs']}  "
          f"延迟 中位 {out['lat_ms']['p50']:.2f} / p90 {out['lat_ms']['p90']:.2f} / p99 {out['lat_ms']['p99']:.2f} 毫秒", flush=True)
    if a.ev.startswith("unix:"):
        from sts2rl.infer import InferClient
        c = InferClient(a.ev[5:]); s = c.stats(); c.close()
        out["server"] = s
        b = max(1, s["batches"])
        print(f"  推理进程：{s['batches']} 批，平均 {s['rows'] / b:.1f} 帧 / 批，每批 组批 {1000 * s['t_collate'] / b:.2f} + 前向 {1000 * s['t_fwd'] / b:.2f} "
              f"+ 回复 {1000 * s['t_send'] / b:.2f} 毫秒，收 {1000 * s['t_recv'] / max(1, s['reqs']):.3f} 毫秒 / 请求", flush=True)
    if a.out:
        with open(a.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(out, ensure_ascii=False) + "\n")


def main() -> None:
    argv = sys.argv[1:]
    rest = []
    if "--" in argv:
        k = argv.index("--")
        argv, rest = argv[:k], argv[k + 1:]
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record"); r.add_argument("--frames", required=True)
    for name in ("diff", "bench"):
        p = sub.add_parser(name)
        p.add_argument("--frames", required=True); p.add_argument("--ckpt", required=True); p.add_argument("--ema", action="store_true")
        p.add_argument("--device", default="cpu"); p.add_argument("--specs", default="fp32"); p.add_argument("--out", default="")
        if name == "diff":
            p.add_argument("--rebatch", default="128"); p.add_argument("--max-batches", type=int, default=0)
        else:
            p.add_argument("--sizes", default="1,8,16,32,64,128,256"); p.add_argument("--batches", type=int, default=50); p.add_argument("--warm", type=int, default=5)
    pl = sub.add_parser("play"); pl.add_argument("--ev", default="torch")
    ag = sub.add_parser("agree"); ag.add_argument("--ev", required=True); ag.add_argument("--ref", default="torch")
    rp = sub.add_parser("report"); rp.add_argument("arms", nargs="+", help="名字=目录（第一个为参照）"); rp.add_argument("--out", default="")
    rp.add_argument("--shards", type=int, default=16)
    ar = sub.add_parser("agree-report"); ar.add_argument("dirs", nargs="+", help="名字=目录（agree 的输出）"); ar.add_argument("--out", default="")
    s = sub.add_parser("serve")
    s.add_argument("--ckpt", required=True); s.add_argument("--ema", action="store_true"); s.add_argument("--device", default="cuda")
    s.add_argument("--ev", default="fp32"); s.add_argument("--sock", required=True)
    s.add_argument("--wait-ms", type=float, default=2.0); s.add_argument("--max-rows", type=int, default=1024)
    s.add_argument("--depth", type=int, default=1, help="1：依次做；2：组批与前向重叠（前向在工作线程）")
    l = sub.add_parser("load")
    l.add_argument("--frames", required=True); l.add_argument("--ckpt", default=""); l.add_argument("--ema", action="store_true")
    l.add_argument("--device", default="cpu"); l.add_argument("--ev", default="fp32"); l.add_argument("--clients", type=int, default=4)
    l.add_argument("--seconds", type=float, default=20); l.add_argument("--think-ms", type=float, default=0.0)
    l.add_argument("--client-threads", type=int, default=1); l.add_argument("--out", default="")
    a = ap.parse_args(argv)
    {"record": cmd_record, "diff": cmd_diff, "bench": cmd_bench, "serve": cmd_serve, "load": cmd_load,
     "play": cmd_play, "agree": cmd_agree, "report": cmd_report, "agree-report": cmd_agree_report}[a.cmd](a, rest)


if __name__ == "__main__":
    main()
