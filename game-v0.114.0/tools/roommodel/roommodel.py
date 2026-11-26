"""房间结果模型（2026-10-27 起，决策记录 147）：给定开打时的局面与遭遇，预测这场战斗的结局——死亡，或存活时掉多少血。

数据是 B 的采集进程记下的房间结果（python/sts2rl/room_log.py，A 从 B 直连拉到 ~/roomlog/），每场一行：玩家快照（游戏的
SerializablePlayer）、遭遇、房间类型、幕、开打与结束时的血量、输赢，src 区分整局真打（run）与真实快照练习战（practice）。

输入特征（featurize）：
- 牌组：每张牌的词表编号（engine/catalog/vocab.txt）+ 升级次数 + 有无附魔，嵌入后求平均与求和（求和除以 20）；
- 遗物、药水：词表编号的嵌入求和；
- 遭遇、房间类型、幕、数据来源：各一个嵌入；
- 数值：开打血量 / 最大生命、开打血量、最大生命、金币、能量上限、充能球栏位、药水数、层数、牌组张数、遗物数（都做了缩放）。
输出：61 类——第 0 类死亡；第 1–60 类存活时的掉血 hp0 − hp1，按 2 点一档从 −20 到 99（两端截断；负数是回血）。
死亡概率 = 第 0 类的概率；存活时的期望掉血 = 其余各档中心按概率加权后除以存活概率。
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOMS = {"Monster": 0, "Elite": 1, "Boss": 2}
SRCS = {"run": 0, "practice": 1}
DMG_LO, DMG_W, N_BINS = -20, 2, 60
N_CLS = 1 + N_BINS
CENTERS = DMG_LO + DMG_W * (np.arange(N_BINS) + 0.5)
N_SCALAR = 10


def load_vocab(path: Path) -> dict:
    return {l.rstrip("\n"): i for i, l in enumerate(open(path, encoding="utf-8"))}


def target_class(win: bool, hp0: int, hp1: int) -> int:
    if not win:
        return 0
    d = hp0 - hp1
    return 1 + int(np.clip((d - DMG_LO) // DMG_W, 0, N_BINS - 1))


def featurize(rows: list[dict], vocab: dict) -> dict:
    """一组记录 → 数组：牌、遗物、药水按“所属样本下标”拼成一维（便于 index_add 求和），其余每样本一行。"""
    unk = vocab.get("<unk>", 1)
    vid = lambda k: vocab.get(k, unk)
    c_id, c_up, c_en, c_row, r_id, r_row, p_id, p_row = [], [], [], [], [], [], [], []
    enc, room, act, src, scal, y, dmg, win, n_deck = [], [], [], [], [], [], [], [], []
    for i, r in enumerate(rows):
        s = json.loads(r["snap"])
        deck = s.get("deck") or []
        for c in deck:
            c_id.append(vid(c["id"])); c_up.append(min(3, int(c.get("current_upgrade_level") or 0))); c_en.append(1 if c.get("enchantment") else 0); c_row.append(i)
        for c in s.get("relics") or []:
            r_id.append(vid(c["id"])); r_row.append(i)
        pots = s.get("potions") or []
        for c in pots:
            p_id.append(vid(c["id"])); p_row.append(i)
        enc.append(vid("ENCOUNTER." + r["enc"])); room.append(ROOMS.get(r.get("room") or "", 0)); act.append(int(r.get("act") or 0) % 3)
        src.append(SRCS.get(r["src"], 0))
        hp0, mx = int(r["hp0"]), max(1, int(r["max0"]))
        scal.append([hp0 / mx, hp0 / 100, mx / 100, (s.get("gold") or 0) / 500, (s.get("max_energy") or 3) / 5, (s.get("base_orb_slot_count") or 3) / 5,
                     len(pots) / 3, (r.get("floor") or 0) / 50, len(deck) / 40, len(s.get("relics") or []) / 30])
        y.append(target_class(bool(r["win"]), hp0, int(r["hp1"]))); dmg.append(hp0 - int(r["hp1"])); win.append(bool(r["win"])); n_deck.append(len(deck))
    a = lambda x, t=np.int64: np.asarray(x, dtype=t)
    return {"c_id": a(c_id), "c_up": a(c_up), "c_en": a(c_en), "c_row": a(c_row), "r_id": a(r_id), "r_row": a(r_row), "p_id": a(p_id), "p_row": a(p_row),
            "enc": a(enc), "room": a(room), "act": a(act), "src": a(src), "scal": a(scal, np.float32), "y": a(y), "dmg": a(dmg, np.float32),
            "win": a(win, bool), "n": len(rows)}


def read_rows(files: list[Path]) -> list[dict]:
    rows = []
    for f in files:
        try:
            with gzip.open(f, "rt", encoding="utf-8") as fh:
                for l in fh:
                    if l.strip():
                        rows.append(json.loads(l))
        except (EOFError, OSError):       # 正在追加的文件末尾可能不完整
            pass
    return rows


def select(d: dict, idx: np.ndarray) -> dict:
    """按样本下标取子集（拼接数组重新编号）。"""
    idx = np.asarray(idx)
    remap = np.full(d["n"], -1, np.int64); remap[idx] = np.arange(len(idx))
    out = {"n": len(idx)}
    for k in ("enc", "room", "act", "src", "scal", "y", "dmg", "win"):
        out[k] = d[k][idx]
    for pre, keys in (("c", ("id", "up", "en")), ("r", ("id",)), ("p", ("id",))):
        m = remap[d[f"{pre}_row"]] >= 0
        out[f"{pre}_row"] = remap[d[f"{pre}_row"][m]]
        for k in keys:
            out[f"{pre}_{k}"] = d[f"{pre}_{k}"][m]
    return out


def concat(parts: list[dict]) -> dict:
    out, base = {"n": sum(p["n"] for p in parts)}, 0
    for k in ("enc", "room", "act", "src", "scal", "y", "dmg", "win", "c_id", "c_up", "c_en", "r_id", "p_id"):
        out[k] = np.concatenate([p[k] for p in parts])
    for k in ("c_row", "r_row", "p_row"):
        rows = []
        base = 0
        for p in parts:
            rows.append(p[k] + base); base += p["n"]
        out[k] = np.concatenate(rows)
    return out


class RoomModel(nn.Module):
    def __init__(self, vocab_size: int = 4096, d: int = 128, hidden: int = 512):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d)           # 牌、遗物、药水、遭遇共用词表
        self.up = nn.Embedding(4, d); self.en = nn.Embedding(2, d)
        self.room = nn.Embedding(3, d); self.act = nn.Embedding(3, d); self.src = nn.Embedding(2, d)
        self.scal = nn.Linear(N_SCALAR, d)
        self.mlp = nn.Sequential(nn.Linear(7 * d, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, N_CLS))

    def forward(self, b: dict) -> torch.Tensor:
        B = b["enc"].shape[0]
        d = self.emb.embedding_dim
        ce = self.emb(b["c_id"]) + self.up(b["c_up"]) + self.en(b["c_en"])
        csum = ce.new_zeros(B, d).index_add_(0, b["c_row"], ce)
        cnt = torch.bincount(b["c_row"], minlength=B).clamp(min=1).to(ce.dtype)[:, None]
        rsum = ce.new_zeros(B, d).index_add_(0, b["r_row"], self.emb(b["r_id"]))
        psum = ce.new_zeros(B, d).index_add_(0, b["p_row"], self.emb(b["p_id"]))
        ctx = self.emb(b["enc"]) + self.room(b["room"]) + self.act(b["act"]) + self.src(b["src"])
        x = torch.cat([csum / cnt, csum / 20, rsum / 10, psum, ctx, self.scal(b["scal"]), ctx * (csum / cnt)], -1)
        return self.mlp(x)


def to_torch(d: dict, device) -> dict:
    return {k: (torch.from_numpy(v).to(device) if isinstance(v, np.ndarray) else v) for k, v in d.items()}


def predict(logits: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """死亡概率与存活时的期望掉血。"""
    p = torch.softmax(logits.float(), -1).cpu().numpy()
    pdie = p[:, 0]
    surv = np.clip(1 - pdie, 1e-6, None)
    edmg = (p[:, 1:] * CENTERS).sum(1) / surv
    return pdie, edmg
