"""训练里的胜率势能（2026-10-29 起，决策记录 189）：胜率小模型与主模型一起训练，给整局做势能塑形。

势能 Φ(s) = 系数 × 胜率模型预测的通关概率；整局里每一步的奖励加 Φ(下一步) − Φ(这一步)，对局结束时 Φ = 0，
整局合计 = −Φ(开局)，与策略无关，所以不改变最优策略（基于势能的塑形），只把“这一步让通关更可能还是更不可能”提前告诉策略。

模型与特征同 tools/winprob/winprob.py（离线训练的实时胜率模型，直播侧栏用），离线训好的权重可以直接载入当初始值；
特征直接从观测算（不经回放的 describe），牌组在战斗中沿用最近一次战斗外的（carry）。在线训练只用通关头：
标签是这一局最后有没有通关（训练整局按概率采样，比贪心评测的胜率低）。
"""
from __future__ import annotations

import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sts2env.protocol import TokenKind, Zone

ROOM_MONSTER, ROOM_ELITE, ROOM_BOSS = 1, 2, 3
BAGS = ("deck", "rel", "pot", "pow", "en")
N_SCALAR = 29
WIN_HEAD = 2        # 三个输出头里的通关头（另两个为赢下当前战斗、打过下一个 Boss，在线训练不用）


def checkpoint(o) -> tuple[int, int, int]:
    """算势能的时刻：战斗外的每个决策（层、0、决策类型）与战斗中每回合的第一个决策（层、回合、−1）。
    只在这些时刻更新 Φ，中间的步不加塑形：每局合计仍是 −Φ(开局)，不改变最优策略，算力约为每步都算的三分之一。"""
    g = o.tok_feats[int(np.argmax(o.tok_ints[:, 0] == TokenKind.GLOBAL))]
    return (int(g[1]), int(g[9]), -1) if g[13] else (int(g[1]), 0, int(o.header[1]))


def deck_of(o) -> list[int] | None:
    """战斗外的牌组（词表编号）；战斗中返回 None（沿用最近一次战斗外的）。"""
    ti = o.tok_ints
    g = o.tok_feats[np.flatnonzero(ti[:, 0] == TokenKind.GLOBAL)[0]]
    if g[13]:
        return None
    return [int(x) for x in ti[(ti[:, 0] == TokenKind.CARD) & (ti[:, 2] == Zone.DECK), 1]]


def features(o, deck: list[int], n_up: int = 0) -> tuple[list[float], dict[str, list[int]]]:
    """一个观测 → (数值特征, 各袋子的词表编号)，与 tools/winprob/winprob.py 的 featurize_step 同义。
    n_up 为牌组里升级过的张数（与 deck 一起在战斗外记下）。"""
    ti, tf = o.tok_ints, o.tok_feats
    kind, vid, zone, owner = ti[:, 0], ti[:, 1], ti[:, 2], ti[:, 3]
    g = tf[int(np.argmax(kind == TokenKind.GLOBAL))].tolist()
    act, floor, hp, mhp, gold = int(g[0]), int(g[1]), g[4], max(1.0, g[5]), g[3]
    en_, en_max, turn, fight, room = g[6], g[7], g[9], bool(g[13]), int(g[14])
    en_ids, e_hp, e_m, e_b, dmg, blk, pows, hand, draw = [], 0.0, 0.0, 0.0, 0.0, 0.0, [], 0, 0
    if fight:
        cr = kind == TokenKind.CREATURE
        en_mask = cr & (zone == Zone.ENEMY) & (tf[:, 3] > 0)
        idx = np.flatnonzero(en_mask)
        if len(idx):
            en_ids = vid[idx].tolist(); e_hp = float(tf[idx, 0].sum()); e_m = float(np.maximum(1.0, tf[idx, 1]).sum()); e_b = float(tf[idx, 2].sum())
            mv = (kind == TokenKind.MOVE) & (tf[:, 0] == 0)                 # 来袭伤害：招式预告第 0 步按概率加权（单次伤害 × 次数）
            mv &= np.isin(owner, idx)
            if mv.any():
                f = tf[mv]
                dmg = float((f[:, 1] * np.maximum(0.0, f[:, 5]) * np.maximum(1.0, f[:, 4])).sum())
        me = np.flatnonzero(cr & (tf[:, 5] > 0))
        if len(me):
            blk = float(tf[me[0], 2])
            pows = vid[(kind == TokenKind.POWER) & (owner == me[0])].tolist()
        cards = kind == TokenKind.CARD
        hand, draw = int((cards & (zone == Zone.HAND)).sum()), int((cards & ((zone == Zone.DRAW) | ((zone >= Zone.DRAW_TOP1) & (zone <= Zone.DRAW_TOP3)))).sum())
    ally = zone == Zone.ALLY
    rel = vid[(kind == TokenKind.RELIC) & ally].tolist()
    pot = vid[(kind == TokenKind.POTION) & ally].tolist()
    ne = len(en_ids)
    scal = [act == 0, act == 1, act == 2, floor / 50, hp / mhp, hp / 100, mhp / 100, gold / 300, en_ / 5, en_max / 5,
            room == ROOM_MONSTER, room == ROOM_ELITE, room == ROOM_BOSS, room not in (ROOM_MONSTER, ROOM_ELITE, ROOM_BOSS), fight,
            turn / 10 if fight else 0.0, blk / 50 if fight else 0.0, ne / 5, e_hp / e_m if ne else 0.0, e_hp / 300, e_b / 50,
            dmg / 50, min(3.0, dmg / max(1.0, hp)), hand / 10 if fight else 0.0, draw / 30 if fight else 0.0,
            len(deck) / 40, n_up / max(1, len(deck)), len(rel) / 20, len(pot) / 5]
    return [float(x) for x in scal], {"deck": deck, "rel": rel, "pot": pot, "pow": pows, "en": en_ids}


def upgraded(o) -> int:
    ti, tf = o.tok_ints, o.tok_feats
    return int(((ti[:, 0] == TokenKind.CARD) & (ti[:, 2] == Zone.DECK) & (tf[:, 3] > 0)).sum())


def collate(rows: list[tuple[list[float], dict]], device) -> dict:
    """若干行 → 模型输入（各袋子拼成一维，另给所属样本下标）。"""
    out = {"scal": torch.from_numpy(np.asarray([r[0] for r in rows], np.float32).reshape(len(rows), N_SCALAR)).to(device)}
    for b in BAGS:
        lists = [r[1][b] for r in rows]
        lens = np.fromiter(map(len, lists), np.int64, len(lists))
        ids = np.fromiter(itertools.chain.from_iterable(lists), np.int64, int(lens.sum()))
        out[f"{b}_id"] = torch.from_numpy(ids).to(device)
        out[f"{b}_row"] = torch.from_numpy(np.repeat(np.arange(len(rows)), lens)).to(device)
    return out


class WinProb(nn.Module):
    """同 tools/winprob/winprob.py 的 WinProb（参数名一致，离线权重可直接载入）。"""

    def __init__(self, vocab_size: int = 2304, d: int = 16, hidden: int = 64, dropout: float = 0.0):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d)
        self.drop = nn.Dropout(dropout)
        self.mlp = nn.Sequential(nn.Linear(6 * d + N_SCALAR, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, 3))

    def forward(self, b: dict) -> torch.Tensor:
        B, d = b["scal"].shape[0], self.emb.embedding_dim
        s = {}
        for k in BAGS:
            e = self.emb(b[f"{k}_id"].clamp(0, self.emb.num_embeddings - 1))
            s[k] = e.new_zeros(B, d).index_add_(0, b[f"{k}_row"], e)
        cnt = torch.bincount(b["deck_row"], minlength=B).clamp(min=1).to(s["deck"].dtype)[:, None]
        x = torch.cat([self.drop(torch.cat([s["deck"] / cnt, s["deck"] / 20, s["rel"] / 10, s["pot"], s["pow"] / 5, s["en"]], -1)), b["scal"]], -1)
        return self.mlp(x)


def build(args: dict | None = None) -> WinProb:
    a = dict(vocab_size=2304, d=32, hidden=128, dropout=0.2) | (args or {})
    return WinProb(**a)


@torch.no_grad()
def predict(net: WinProb, rows: list, device) -> np.ndarray:
    """通关概率。"""
    if not rows:
        return np.zeros(0, np.float32)
    net.eval()
    return torch.sigmoid(net(collate(rows, device))[:, WIN_HEAD].float()).cpu().numpy()


def train_step(net: WinProb, opt, rows: list, y: np.ndarray, w: np.ndarray, device) -> float:
    """在线训练一步：只训通关头，按样本权重（每局每层合计为 1）加权的对数损失。"""
    net.train()
    z = net(collate(rows, device))[:, WIN_HEAD]
    yt, wt = torch.tensor(y, dtype=torch.float32, device=device), torch.tensor(w, dtype=torch.float32, device=device)
    loss = (F.binary_cross_entropy_with_logits(z, yt, reduction="none") * wt).sum() / wt.sum().clamp(min=1e-6)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    opt.step()
    net.eval()
    return float(loss.item())
