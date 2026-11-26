"""构筑价值模型（2026-11-03，决策记录 252，情况二）：每到战斗外的决策，用当时的构筑估计这一局最后通关的概率。
离线训练与检验在 tools/buildvalue.py；这里放两边共用的部分：输入编码（expand）、网络（make_model）、
从观测直接取输入（obs_row，与回放 replay.describe 取的是同一批记录），以及采集端用的冻结估计器（Estimator）。

输入只看构筑相关的：牌组（每张卡分升级与否：编号 × 2 + 是否升级）、遗物、药水，加幕、层、血量、上限、金币、决策类型；不看战斗里的局面。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from sts2env.protocol import TokenKind, Zone

V = 2304                    # 词表大小（同 tools/winprob）
N_KIND = 13


def expand(r) -> tuple:
    """紧凑行（幕, 层, 血量, 上限, 金币, 决策类型, 牌组, 遗物, 药水）→ 训练用的行（0, 幕, 层, 血量比例, 数值输入, 牌组, 遗物, 药水）。"""
    act, floor, hp, mhp, gold, d, deck, rel, pot = r
    mhp = max(1, mhp)
    n_up = sum(x & 1 for x in deck)
    kind = [0.0] * N_KIND; kind[min(d, N_KIND - 1)] = 1.0
    scal = [act == 0, act == 1, act == 2, floor / 50, hp / mhp, hp / 100, mhp / 100, gold / 300,
            len(deck) / 40, n_up / max(1, len(deck)), len(rel) / 20, len(pot) / 5] + kind
    return (0, act, floor, hp / mhp, [float(x) for x in scal], list(deck), list(rel), list(pot))


def obs_row(o) -> tuple | None:
    """一个观测 → 紧凑行；战斗中返回 None（只在战斗外的决策用）。
    取法与 replay.describe 相同：全局记录的 g[0] 幕、g[1] 层、g[3] 金币、g[4] 血量、g[5] 上限、g[13] 是否战斗中；
    牌组为牌组区的卡（特征 3 为升级次数），遗物、药水为己方区的遗物、药水；决策类型为 header[1]。"""
    ti, tf = o.tok_ints, o.tok_feats
    kind, vid, zone = ti[:, 0], ti[:, 1], ti[:, 2]
    g = tf[int(np.argmax(kind == TokenKind.GLOBAL))]
    if g[13]:
        return None
    deck = [min(int(vid[j]), V - 1) * 2 + (tf[j, 3] > 0) for j in np.flatnonzero((kind == TokenKind.CARD) & (zone == Zone.DECK))]
    rel = [min(int(vid[j]), V - 1) for j in np.flatnonzero((kind == TokenKind.RELIC) & (zone == Zone.ALLY))]
    pot = [min(int(vid[j]), V - 1) for j in np.flatnonzero((kind == TokenKind.POTION) & (zone == Zone.ALLY))]
    return (int(g[0]), int(g[1]), int(round(float(g[4]))), int(round(float(g[5]))), int(round(float(g[3]))), int(o.header[1]),
            [int(x) for x in deck], rel, pot)


def make_model(n_scal: int, bags: bool, d: int, hidden: int, dropout: float) -> nn.Module:
    class BuildValue(nn.Module):
        def __init__(self):
            super().__init__()
            self.bags = bags
            extra = 0
            if bags:
                self.deck = nn.EmbeddingBag(2 * V, d, mode="sum")
                self.rel = nn.EmbeddingBag(V, d, mode="sum")
                self.pot = nn.EmbeddingBag(V, d, mode="sum")
                extra = 4 * d
            self.mlp = nn.Sequential(nn.Linear(n_scal + extra, hidden), nn.ReLU(), nn.Dropout(dropout),
                                     nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

        def forward(self, scal, deck=None, rel=None, pot=None):
            x = [scal]
            if self.bags:
                ds = self.deck(deck[0], deck[1])
                x += [ds / deck[2].clamp(min=1).unsqueeze(1), ds / 20, self.rel(rel[0], rel[1]) / 5, self.pot(pot[0], pot[1]) / 2]
            return self.mlp(torch.cat(x, 1)).squeeze(1)
    return BuildValue()


def bag_tensors(lists: list[list[int]], device) -> tuple:
    """若干个袋子 → EmbeddingBag 的（拼接的编号、每个袋子的起点、每个袋子的件数）。"""
    off = np.zeros(len(lists), np.int64); n = 0
    for k, l in enumerate(lists):
        off[k] = n; n += len(l)
    flat = np.fromiter((x for l in lists for x in l), np.int64, count=n)
    cnt = np.asarray([len(l) for l in lists], np.float32)
    return torch.from_numpy(flat).to(device), torch.from_numpy(off).to(device), torch.from_numpy(cnt).to(device)


class Estimator:
    """冻结的构筑价值模型：紧凑行 → 通关概率。采集端用（CPU 即可：约 10 万参数，一批几十行不到 1 毫秒）。"""

    def __init__(self, path: str, device="cpu"):
        blob = torch.load(path, map_location="cpu", weights_only=False)
        c = blob["config"]
        self.net = make_model(c["n_scal"], True, c["dim"], c["hidden"], 0.0)
        self.net.load_state_dict(blob["state"])
        self.net.to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, rows: list[tuple]) -> np.ndarray:
        if not rows:
            return np.zeros(0, np.float32)
        ex = [expand(r) for r in rows]
        sc = torch.from_numpy(np.asarray([e[4] for e in ex], np.float32)).to(self.device)
        z = self.net(sc, bag_tensors([e[5] for e in ex], self.device), bag_tensors([e[6] for e in ex], self.device),
                     bag_tensors([e[7] for e in ex], self.device))
        return torch.sigmoid(z).cpu().numpy()
