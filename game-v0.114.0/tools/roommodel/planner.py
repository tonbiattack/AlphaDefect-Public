"""路线风险展开（2026-10-27 起，决策记录 152）：在地图决策上，用房间结果模型（roommodel.py）按血量做动态规划，
估计“从这个节点走下去、之后每步都选最安全的路，活着打过本幕 Boss 的概率”。

- 地图：观测里的地图节点记录（行、列、连到下一行的子节点列号；第 0 行古神，第 1–15 行房间，最后一行 Boss）。
- 状态是血量 h ∈ {0, 1, …, 最大生命}（0 为死亡）。每种房间是一个转移矩阵 T[h, h']：
  普通战 / 精英战：按（幕、房间类型、行）的遭遇分布（从房间记录统计，相邻行平滑）混合各遭遇，每个遭遇在每个血量上问一次房间结果模型
  （死亡概率 + 存活掉血的 2 点一档分布）；问号房：按观测里的几率混合普通战与“不变”（宝箱、商店、事件都当作不变）；
  休息处：回 30% 最大生命；商店、宝箱、古神：不变。
- Boss：V(h) = 打赢本幕 Boss 的概率；第三幕连打两个 Boss（表头第 17 项非 0）时，按第一场打完的血量分布再乘第二场打赢的概率。
- 倒推：V_节点(h) = Σ_h' T_节点[h, h'] · max_子节点 V_子节点(h')（到了节点先结算房间，再在知道血量后选下一步）。
只看战斗风险，不看遗物、选牌、商店的收益，所以只拿来修正策略的选路，不单独拍板（见 route_eval.py）。
牌组在路上会变强，这里一律用当前快照，估得偏保守。
"""
from __future__ import annotations

import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from roommodel import DMG_LO, DMG_W, N_BINS, ROOMS, featurize, to_torch

MAP_NODE = 8
ACT_BASE = {0: 1, 1: 18, 2: 34}      # 各幕古神所在的总层数（行 = 总层数 − 它）
REST_HEAL = 0.30


def build_prior(log_dir: Path, max_rows: int = 400_000) -> dict:
    """整局真打的战斗里，（幕、房间类型、行）→ 遭遇计数。"""
    cnt: dict = defaultdict(Counter)
    n = 0
    for f in sorted(log_dir.glob("*.jsonl.gz"), reverse=True):
        try:
            with gzip.open(f, "rt", encoding="utf-8") as fh:
                for l in fh:
                    r = json.loads(l)
                    if r["src"] != "run" or r.get("room") not in ("Monster", "Elite"):
                        continue
                    row = int(r["floor"]) - ACT_BASE.get(int(r["act"]), 0)
                    cnt[(int(r["act"]), r["room"], row)][r["enc"]] += 1
                    n += 1
        except (EOFError, OSError):
            pass
        if n >= max_rows:
            break
    return {f"{a}|{rm}|{rw}": dict(c) for (a, rm, rw), c in cnt.items()}


class Planner:
    def __init__(self, model, vocab: dict, prior: dict, device: str):
        self.model, self.vocab, self.device = model.eval(), vocab, device
        self.names = {i: k for k, i in vocab.items()}
        self.prior = defaultdict(Counter)
        for k, v in prior.items():
            a, rm, rw = k.split("|")
            self.prior[(int(a), rm, int(rw))] = Counter(v)

    def enc_dist(self, act: int, room: str, row: int) -> dict:
        """这一行的遭遇分布：本行 + 相邻两行（权重减半），再不行就整幕。"""
        c = Counter()
        for dr, w in ((0, 1.0), (-1, 0.5), (1, 0.5)):
            for e, n in self.prior.get((act, room, row + dr), {}).items():
                c[e] += w * n
        if not c:
            for (a, rm, _), cc in self.prior.items():
                if a == act and rm == room:
                    c.update(cc)
        tot = sum(c.values())
        return {e: n / tot for e, n in c.items()} if tot else {}

    # ---- 房间结果模型：一份快照、若干（遭遇、房间、血量）组合一次算完
    def outcome_probs(self, snap: str, act: int, floor: int, H: int, encs: list[tuple[str, str]]) -> np.ndarray:
        """返回 [遭遇数, H, 61]：开打血量为 1..H 时的结局分布。"""
        base = featurize([{"snap": snap, "enc": encs[0][0], "room": encs[0][1], "act": act, "floor": floor, "src": "run",
                           "hp0": H, "max0": H, "hp1": H, "win": True}], self.vocab)
        n = len(encs) * H
        nc, nr, npo = len(base["c_id"]), len(base["r_id"]), len(base["p_id"])
        rows = np.arange(n)
        d = {"n": n,
             "c_id": np.tile(base["c_id"], n), "c_up": np.tile(base["c_up"], n), "c_en": np.tile(base["c_en"], n), "c_row": np.repeat(rows, nc),
             "r_id": np.tile(base["r_id"], n), "r_row": np.repeat(rows, nr), "p_id": np.tile(base["p_id"], n), "p_row": np.repeat(rows, npo)}
        d["enc"] = np.repeat(np.array([self.vocab.get("ENCOUNTER." + e, 1) for e, _ in encs], np.int64), H)
        d["room"] = np.repeat(np.array([ROOMS.get(rm, 0) for _, rm in encs], np.int64), H)
        d["act"] = np.full(n, act % 3, np.int64); d["src"] = np.zeros(n, np.int64)
        sc = np.repeat(base["scal"], n, 0)
        hp = np.tile(np.arange(1, H + 1, dtype=np.float32), len(encs))
        sc[:, 0] = hp / H; sc[:, 1] = hp / 100
        d["scal"] = sc
        with torch.no_grad():
            lg = self.model(to_torch(d, self.device))
        return torch.softmax(lg.float(), -1).cpu().numpy().reshape(len(encs), H, -1)

    @staticmethod
    def trans_from_probs(P: np.ndarray, H: int) -> np.ndarray:
        """[H, 61] 的结局分布 → 转移矩阵 [H+1, H+1]（行 0 是死亡的吸收态）。每档掉血 [lo, lo+2) 平分到两个整数上。"""
        T = np.zeros((H + 1, H + 1))
        T[0, 0] = 1.0
        h = np.arange(1, H + 1)
        T[h, 0] = P[:, 0]
        for k in range(N_BINS):
            for j in range(DMG_W):
                dmg = DMG_LO + DMG_W * k + j
                h2 = np.clip(h - dmg, 0, H)
                # 掉到 0 及以下但模型说存活的，记为 1 血（存活就不会是 0）
                h2 = np.where(h2 <= 0, 1, h2)
                np.add.at(T, (h, h2), P[:, 1 + k] / DMG_W)
        return T

    def node_values(self, o, snap: str, act: int) -> dict[int, float]:
        """当前地图决策：每个可前往节点的记录下标 → 从它走下去打过本幕 Boss 的概率（当前血量下）。"""
        ti, tf = o.tok_ints, o.tok_feats
        idx = np.flatnonzero(ti[:, 0] == MAP_NODE)
        nodes, cur_row, tok_of = {}, 0, {}
        for t in idx:
            name = self.names.get(int(ti[t, 1]), "")
            kind = name.split(":", 1)[1] if ":" in name else name
            r, c = int(tf[t, 0]), int(tf[t, 1])
            ch = [int(x) - 1 for x in tf[t, 6:9] if x > 0]
            nodes[(r, c)] = (kind, ch)
            tok_of[(r, c)] = int(t)
            if tf[t, 4] > 0:
                cur_row = r
        if not nodes:
            return {}
        H = max(1, int(o.max_hp))
        h_now = int(np.clip(o.hp, 1, H))
        floor0 = int(o.floor) - cur_row          # 本幕古神所在的总层数
        boss_row = max(r for r, _ in nodes)
        # 需要问模型的（遭遇、房间）
        need: dict = {}
        fight_rows = set()
        for (r, c), (kind, _) in nodes.items():
            if r <= cur_row:
                continue
            if kind in ("Monster", "Elite", "Unknown"):
                rm = "Elite" if kind == "Elite" else "Monster"
                fight_rows.add((rm, r))
                for e in self.enc_dist(act, rm, r):
                    need[(e, rm)] = None
        boss1 = self.names.get(int(o.header[16]), "").replace("ENCOUNTER.", "")
        boss2 = self.names.get(int(o.header[17]), "").replace("ENCOUNTER.", "") if int(o.header[17]) else ""
        for b in (boss1, boss2):
            if b:
                need[(b, "Boss")] = None
        encs = list(need)
        if not encs:
            return {}
        P = self.outcome_probs(snap, act, int(o.floor), H, encs)
        Tenc = {k: self.trans_from_probs(P[i], H) for i, k in enumerate(encs)}
        I = np.eye(H + 1)
        Tfight = {}
        for rm, r in fight_rows:
            dist = self.enc_dist(act, rm, r)
            Tfight[(rm, r)] = sum(w * Tenc[(e, rm)] for e, w in dist.items()) if dist else I
        g = np.flatnonzero(ti[:, 0] == 1)
        p_monster = float(tf[g[0], 16]) if len(g) and tf.shape[1] > 16 else 0.1
        heal = int(round(REST_HEAL * H))
        Trest = np.zeros((H + 1, H + 1)); Trest[0, 0] = 1
        for h in range(1, H + 1):
            Trest[h, min(H, h + heal)] = 1
        # Boss 的价值
        pwin1 = 1 - P[encs.index((boss1, "Boss")), :, 0] if boss1 else np.ones(H)
        if boss2:
            T1 = Tenc[(boss1, "Boss")]
            pwin2 = np.concatenate([[0.0], 1 - P[encs.index((boss2, "Boss")), :, 0]])
            Vb = T1 @ pwin2
        else:
            Vb = np.concatenate([[0.0], pwin1])
        # 倒推
        V: dict = {}
        for r in range(boss_row, cur_row, -1):
            for (rr, c), (kind, ch) in nodes.items():
                if rr != r:
                    continue
                if kind == "Boss" or r == boss_row:
                    V[(r, c)] = Vb
                    continue
                nxt = [V[(r + 1, x)] for x in ch if (r + 1, x) in V]
                M = np.max(np.stack(nxt), 0) if nxt else Vb
                if kind in ("Monster", "Elite"):
                    T = Tfight[("Elite" if kind == "Elite" else "Monster", r)]
                elif kind == "Unknown":
                    T = p_monster * Tfight[("Monster", r)] + (1 - p_monster) * I
                elif kind == "RestSite":
                    T = Trest
                else:
                    T = I
                V[(r, c)] = T @ M
        return {tok_of[k]: float(v[h_now]) for k, v in V.items() if k in tok_of}
