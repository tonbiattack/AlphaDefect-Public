"""实时胜率模型（2026-10-29 起，决策记录 175）：给定对局里任一决策点的局面，预测本局主模型（贪心）
赢下当前这场战斗、打过下一个 Boss、通关的概率。监督学习，模型很小（约 5 万参数），每步都能实时算。

输入是回放的一步（python/sts2rl/replay.py 的 describe：g、pl、en、hand、rel、pot、deck 等），实机对局里对观测调 describe 就能得到同样的格式。
牌组只在战斗外出现，战斗中沿用最近一次战斗外的牌组（carry）。

特征（featurize_step）：
- 牌组：词表编号的嵌入求平均与求和（求和除以 20）；遗物、药水、玩家身上的能力、活着的敌人：嵌入求和；
  带层数的模型（amounts，2026-10-31 起，第十四份外部评审，决策记录 207）另加：玩家能力按层数加权的嵌入和（权重 sign·log1p|层数|，
  原来缓冲 1 层与 3 层、力量 2 与 8 输入完全相同），活着的敌人身上的能力（嵌入和、按层数加权的嵌入和）。旧模型（不带层数）照常能载入；
- 数值：幕（独热）、层、血量比例、血量、最大生命、金币、能量与上限、房间类型（独热）、是否战斗中、回合、
  格挡、敌人数与剩余血量、敌人这回合的来袭伤害（招式预告第 0 步按概率加权）及其与血量之比、手牌与抽牌堆张数、牌组张数、
  升级比例、遗物数、药水数。第三幕两个 Boss 靠层数区分（第 48、49 层）。
输出 3 个对数几率：赢下当前战斗（只在战斗中有标签）、打过下一个 Boss（打过最后一个 Boss 之后没有标签）、通关。

标签（labels）：回放结束于对局结束；输的局最后一步在致死的那场战斗里。一段连续的战斗步（同一层）后面还有步、或这局赢了，就算打赢。
下一个 Boss = 此前已打完的 Boss 战段数（在 Boss 战中时就是这一场）；打过的 Boss 段数多于它就算打过。
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOM_MONSTER, ROOM_ELITE, ROOM_BOSS = 1, 2, 3
N_BOSS = 4                  # 进阶 10：第三幕有两个 Boss
BAGS = ("deck", "rel", "pot", "pow", "en", "epow")
WEIGHTED = ("pow", "epow")      # 带层数权重（<袋子>_w，与词表编号一一对应）的袋子
N_SCALAR = 29
HEADS = ("combat", "boss", "win")


def read_replay(path: Path) -> tuple[dict, list[dict]] | None:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            lines = [json.loads(l) for l in f if l.strip()]
    except (EOFError, OSError, json.JSONDecodeError):
        return None
    return (lines[0], lines[1:]) if len(lines) > 1 else None


def incoming(en: list[dict]) -> float:
    """活着的敌人这一回合的来袭伤害：招式预告里第 0 步的各候选按概率加权（单次伤害 × 次数）。"""
    tot = 0.0
    for e in en:
        if not e.get("al"):
            continue
        for mv in e.get("mv") or []:
            if mv[1] == 0:
                tot += float(mv[2]) * max(0, mv[3]) * max(1, mv[4])
    return tot


def featurize_step(s: dict, deck: list) -> tuple[list[float], dict[str, list[int]]]:
    """一步 → (数值特征, 各袋子的词表编号)。deck 为 [[卡, 升级], ...]（战斗中传最近一次战斗外的牌组）。"""
    g = s["g"]
    act, floor, hp, mhp, gold, en_, en_max, turn, fight, room = g
    mhp = max(1, mhp)
    en = [e for e in s.get("en") or [] if e.get("al")]
    pl = s.get("pl") or {}
    e_hp, e_m, e_b = sum(e["hp"] for e in en), sum(max(1, e["m"]) for e in en), sum(e["b"] for e in en)
    dmg = incoming(en) if fight else 0.0
    n_up = sum(1 for c in deck if c[1] > 0)
    scal = [act == 0, act == 1, act == 2, floor / 50, hp / mhp, hp / 100, mhp / 100, gold / 300, en_ / 5, en_max / 5,
            room == ROOM_MONSTER, room == ROOM_ELITE, room == ROOM_BOSS, room not in (ROOM_MONSTER, ROOM_ELITE, ROOM_BOSS), fight,
            turn / 10 if fight else 0.0, (pl.get("b") or 0) / 50 if fight else 0.0, len(en) / 5, e_hp / e_m if en else 0.0, e_hp / 300, e_b / 50,
            dmg / 50, min(3.0, dmg / max(1, hp)), len(s.get("hand") or []) / 10 if fight else 0.0, (s.get("draw") or 0) / 30 if fight else 0.0,
            len(deck) / 40, n_up / max(1, len(deck)), len(s.get("rel") or []) / 20, len(s.get("pot") or []) / 5]
    pw = (pl.get("p") or []) if fight else []
    ep = [p for e in en for p in e.get("p") or []]
    amt = lambda ps: [float(np.sign(p[1]) * np.log1p(abs(p[1]))) if len(p) > 1 else 0.0 for p in ps]
    bags = {"deck": [c[0] for c in deck], "rel": [r[0] for r in s.get("rel") or []], "pot": [p[0] for p in s.get("pot") or []],
            "pow": [p[0] for p in pw], "en": [e["n"] for e in en], "epow": [p[0] for p in ep], "pow_w": amt(pw), "epow_w": amt(ep)}
    return [float(x) for x in scal], bags


def labels(head: dict, steps: list[dict]) -> np.ndarray:
    """每步三个标签（−1 为没有标签）：赢下当前战斗、打过下一个 Boss、通关。"""
    n, win = len(steps), bool(head.get("win"))
    seg = np.full(n, -1)           # 战斗段编号
    segs: list[tuple[int, int, bool]] = []      # (开始, 结束, 是否 Boss)
    i = 0
    while i < n:
        g = steps[i]["g"]
        if g[8]:
            j = i
            while j + 1 < n and steps[j + 1]["g"][8] and steps[j + 1]["g"][1] == g[1]:
                j += 1
            seg[i:j + 1] = len(segs); segs.append((i, j, g[9] == ROOM_BOSS)); i = j + 1
        else:
            i += 1
    won = [win or e + 1 < n for _, e, _ in segs]
    boss_beaten = sum(1 for k, (_, _, b) in enumerate(segs) if b and won[k])
    y = np.full((n, 3), -1, np.int8)
    y[:, 2] = win
    done = 0                       # 已打完的 Boss 段数
    k_boss = [k for k, (_, _, b) in enumerate(segs) if b]
    ends = {segs[k][1]: k for k in k_boss}
    for t in range(n):
        if seg[t] >= 0:
            y[t, 0] = won[seg[t]]
        if done < N_BOSS:
            y[t, 1] = boss_beaten > done
        if t in ends:
            done += 1
    return y


def game_rows(head: dict, steps: list[dict]):
    """一局 → 每步的 (数值特征, 袋子, 标签, 层)。"""
    y = labels(head, steps)
    deck: list = []
    for t, s in enumerate(steps):
        if "deck" in s:
            deck = s["deck"]
        sc, bags = featurize_step(s, deck)
        yield sc, bags, y[t], s["g"][1]


def pack(rows: list[tuple[list[float], dict]]) -> dict:
    """若干步 → 数组：各袋子的词表编号拼成一维（int16，词表不到 3 万），每个样本在其中的起止位置记在 <袋子>_off（n + 1 个）。"""
    out = {"scal": np.asarray([r[0] for r in rows], np.float32).reshape(len(rows), N_SCALAR), "n": len(rows)}
    for b in BAGS:
        ids, lens = [], []
        for r in rows:
            ids += r[1][b]; lens.append(len(r[1][b]))
        out[f"{b}_id"] = np.asarray(ids, np.int16)
        out[f"{b}_off"] = np.concatenate([[0], np.cumsum(lens, dtype=np.int64)])
        if b in WEIGHTED:
            out[f"{b}_w"] = np.asarray([w for r in rows for w in r[1][f"{b}_w"]], np.float32)
    return out


def _offsets(d: dict, b: str) -> np.ndarray:
    """每个样本在袋子一维数组里的起止位置；只有“所属样本下标”（<袋子>_row，按样本排好序）时由它算出。"""
    k = f"{b}_off"
    if k not in d:
        d[k] = np.searchsorted(d[f"{b}_row"], np.arange(d["n"] + 1)).astype(np.int64)
    return d[k]


def select(d: dict, idx: np.ndarray) -> dict:
    idx = np.asarray(idx, np.int64)
    out = {"n": len(idx)}
    for k, v in d.items():
        if k != "n" and not k.endswith(("_id", "_row", "_off")) and k not in {f"{b}_w" for b in WEIGHTED}:
            out[k] = v[idx]
    for b in BAGS:
        if f"{b}_off" not in d and f"{b}_row" not in d:       # 旧缓存没有这个袋子（敌人能力是 amounts 起才有的）
            continue
        off = _offsets(d, b)
        st, ln = off[idx], off[idx + 1] - off[idx]
        tot = int(ln.sum())
        pos = np.repeat(st - np.concatenate([[0], np.cumsum(ln)[:-1]]), ln) + np.arange(tot)
        out[f"{b}_id"], out[f"{b}_row"] = d[f"{b}_id"][pos], np.repeat(np.arange(len(idx), dtype=np.int32), ln)
        if f"{b}_w" in d:
            out[f"{b}_w"] = d[f"{b}_w"][pos]
    return out


class WinProb(nn.Module):
    def __init__(self, vocab_size: int = 2304, d: int = 16, hidden: int = 64, dropout: float = 0.0, amounts: bool = True,
                 tie_final: bool = False, tie_mode: str = "mean"):
        super().__init__()
        self.amounts = amounts                          # 带层数（决策记录 207）：多 3 段输入，见文件头；旧模型为 False
        self.tie_final, self.tie_mode = tie_final, tie_mode     # 最后一个 Boss 三个数统一（见 FinalStage），只在 Estimator 里用，前向不变
        self.emb = nn.Embedding(vocab_size, d)          # 牌、遗物、药水、能力、怪物共用词表（engine/catalog/vocab.txt）
        self.drop = nn.Dropout(dropout)
        self.mlp = nn.Sequential(nn.Linear((9 if amounts else 6) * d + N_SCALAR, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, len(HEADS)))

    def forward(self, b: dict) -> torch.Tensor:
        B, d = b["scal"].shape[0], self.emb.embedding_dim
        s = {}
        for k in BAGS if self.amounts else BAGS[:5]:
            e = self.emb(b[f"{k}_id"].long().clamp(0, self.emb.num_embeddings - 1))
            s[k] = e.new_zeros(B, d).index_add_(0, b[f"{k}_row"].long(), e)
            if self.amounts and k in WEIGHTED:
                s[k + "_w"] = e.new_zeros(B, d).index_add_(0, b[f"{k}_row"].long(), e * b[f"{k}_w"].to(e.dtype)[:, None])
        cnt = torch.bincount(b["deck_row"].long(), minlength=B).clamp(min=1).to(s["deck"].dtype)[:, None]
        parts = [s["deck"] / cnt, s["deck"] / 20, s["rel"] / 10, s["pot"], s["pow"] / 5, s["en"]]
        if self.amounts:
            parts += [s["pow_w"] / 5, s["epow"] / 5, s["epow_w"] / 5]
        x = torch.cat([self.drop(torch.cat(parts, -1)), b["scal"]], -1)
        return self.mlp(x)


def to_torch(d: dict, device) -> dict:
    return {k: (torch.from_numpy(v).to(device) if isinstance(v, np.ndarray) else v) for k, v in d.items()}


class FinalStage:
    """最后一个 Boss（储君版胜率小模型，储君决策记录第 30 条）：按一局里走过的步数第三幕已经打完几个 Boss。
    第三幕的两个 Boss 不一定在第 48、49 层（储君约 5% 的局在第 50、51 层），只看一步分不出来，要按顺序数：一场第三幕 Boss 战结束
    （离开战斗、或换了层）且游戏还在继续，就记打过一个。打过第一个之后，下一个 Boss 就是最后一个——这时“打过下一个 Boss”与“通关”
    是同一件事（返回 1）；在最后一个 Boss 的战斗里，“赢下这一场”也是同一件事（返回 2）；其余为 0。新一局（第一幕第 0、1 层）自动清零。"""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.done3, self.boss_at = 0, None      # 第三幕打完的 Boss 数；上一步在打第三幕 Boss 时为它所在的层

    def __call__(self, g) -> int:
        act, floor, fight, room = g[0], g[1], g[8], g[9]
        if act == 0 and floor <= 1:
            self.reset()
        in_boss = act == 2 and bool(fight) and room == ROOM_BOSS
        if self.boss_at is not None and (not in_boss or floor != self.boss_at):
            self.done3 += 1
        self.boss_at = floor if in_boss else None
        if act != 2 or self.done3 < 1:
            return 0
        return 2 if in_boss else 1


def final_stages(steps: list[dict]) -> np.ndarray:
    """一局各步的 FinalStage（0 / 1 / 2），检验三个数在最后一个 Boss 前后是否接近时用。"""
    fs = FinalStage()
    return np.asarray([fs(s["g"]) for s in steps], np.int8)


def unify(p: dict, stage: int, mode: str = "mean") -> dict:
    """stage 1：打过下一个 Boss 与通关取同一个值；stage 2：再加上赢下这一场。mode 为 mean（几个概率的平均）或某一个头的名字（照抄它）。"""
    keys = ("boss", "win") if stage == 1 else ("combat", "boss", "win")
    vals = [p[k] for k in keys if p.get(k) is not None]
    if not vals:
        return p
    v = sum(vals) / len(vals) if mode == "mean" else (p.get(mode) if p.get(mode) is not None else sum(vals) / len(vals))
    return {**p, **{k: v for k in keys}}


class Estimator:
    """实时用：逐步喂回放格式的一步（describe 的结果），返回三个概率（不在战斗中时第一个为 None）。"""

    def __init__(self, path: str | Path, device: str = "cpu"):
        ck = torch.load(path, map_location=device)
        d = ck["args"].get("d", 16)          # 带不带层数看第一层的输入宽度（train.py 存的 args 里没有这一项）
        amounts = ck["model"]["mlp.0.weight"].shape[1] == 9 * d + N_SCALAR
        self.net = WinProb(**{**ck["args"], "amounts": amounts}).to(device).eval()
        self.net.load_state_dict(ck["model"])
        self.device, self.deck = device, []
        self.tie, self.tie_mode, self.stage = bool(ck["args"].get("tie_final")), ck["args"].get("tie_mode", "mean"), FinalStage()

    @torch.no_grad()
    def __call__(self, step: dict) -> dict:
        if "deck" in step:
            self.deck = step["deck"]
        sc, bags = featurize_step(step, self.deck)
        b = select(pack([(sc, bags)]), np.arange(1))      # pack 只给起止位置，select 补上模型要的“所属样本下标”
        p = torch.sigmoid(self.net(to_torch(b, self.device)))[0].tolist()
        out = {"combat": p[0] if step["g"][8] else None, "boss": p[1], "win": p[2]}
        st = self.stage(step["g"])
        return unify(out, st, self.tie_mode) if self.tie and st else out
