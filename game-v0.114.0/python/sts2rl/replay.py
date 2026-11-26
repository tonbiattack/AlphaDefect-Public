"""对局回放的记录：把一个决策点的观测压成人能看懂的紧凑记录（JSON 可序列化，名字用词表 ID，由本机的播放器换成游戏里的中文名）。

每步一条：
  d 决策类型，a 选了第几个候选，pr 概率最高的几个候选 [[下标, 概率], ...]
  g [幕, 层数, 生命, 最大生命, 金币, 能量, 能量上限, 回合, 是否战斗中, 房间类型]
  pl 玩家 {hp, m, b, p:[[能力, 层数], ...]}，en 敌人 [{n, s, hp, m, b, al, p, mv:[[招式, 第几步, 概率, 单次伤害, 次数], ...]}, ...]
  orb [[球, 被动, 激发], ...]，球位数 os
  hand [[卡, 升级, 费用（X 费为 −1）, 能打出, 伤害, 格挡], ...]，draw 抽牌堆张数，disc / exh 弃牌堆、消耗堆 [[卡, 升级], ...]
  rel [[遗物, 计数], ...]，pot [[药水, 栏位], ...]，deck [[卡, 升级], ...]（只在战斗外）
  map [[行, 列, 类型, 已到过, 当前, 可达, 子节点列...], ...]（只在选路线时）
  sel 选牌说明 [用途下标, 最少, 最多, 已选, 发起者]，sp 水晶球已揭开的格子数与剩余次数
  c 候选 [[动作类型, A, B, 键, f0, f1, f2], ...]，A / B 为对象描述 [记录类型, 编号, 区域, 升级或数值]，没有为 null
  版本 13 起补记（只供复盘，不进训练）：pl / en 的 pv 与 p 对齐的能力界面数值，et 回合结束预演（玩家 [冰霜格挡, 预计掉血, 自伤, 额外格挡,
  随机目标都没打到攻击者时的预计掉血, 靠随机目标才打得死的敌人数]，敌人 [被动伤害, 剩余生命加格挡, 本回合最多还能打多少, 能否杀掉]），
  orbx [球位已满, 被挤出的球的激发值, 循环给的格挡, 循环给的其他数值]，drawl 抽牌堆明细 [[卡, 升级], ...]（按 ID 排序，不是真实顺序），
  cx 带附魔或苦痛的牌 [[区域, 卡, 升级, 附魔, 附魔层数, 苦痛, 苦痛层数], ...]，rt 路线摘要 [[节点类型, 到 Boss 还剩几行, 到休息处, 到商店, 下个休息处前最少精英, 最少战斗], ...]
  shop 商店货架 {g 金币, full 药水栏满, it [[类别 0 牌 1 遗物 2 药水 3 删牌, 编号, 价格, 有货, 买得起, 升级, 删牌已用], ...]}（只在商店，引擎附加信息）
"""
from __future__ import annotations

import numpy as np

from sts2env.protocol import TokenKind, Zone

_DECK_ZONES = (Zone.DECK,)


def _i(x) -> int:
    return int(round(float(x)))


def _obj(o, j: int):
    """记录 j 的简短描述：[类型, 编号, 区域, 升级次数（卡牌）/ 数值（选项）/ 行列（地图节点）]。"""
    if j < 0:
        return None
    k, vid, z = (int(v) for v in o.tok_ints[j, :3])
    f = o.tok_feats[j]
    extra = 0
    if k == TokenKind.CARD:
        extra = _i(f[3])
    elif k == TokenKind.OPTION:
        extra = [_i(f[0]), int(o.tok_ints[j, 4])]     # 数值、附带的遗物（Neow 等古神的遗物选项靠它认出是哪件）
    elif k == TokenKind.MAP_NODE:
        extra = [_i(f[0]), _i(f[1])]
    elif k == TokenKind.CREATURE:
        extra = _i(o.tok_ints[j, 4])
    return [k, vid, z, extra]


_SHOP_KIND = {"CARD": 0, "RELIC": 1, "POTION": 2, "REMOVAL": 3}
_VOCAB: dict | None = None


def _vocab_index() -> dict:
    """词表键 → 编号（商店货架的商品由引擎按键给出，回放里统一存编号）。"""
    global _VOCAB
    if _VOCAB is None:
        from .catalog import Catalog
        _VOCAB = {k: i for i, k in enumerate(Catalog().vocab)}
    return _VOCAB


def describe(o, action: int, probs: np.ndarray) -> dict:
    ti, tf = o.tok_ints, o.tok_feats
    kind, vid, zone, owner = ti[:, 0], ti[:, 1], ti[:, 2], ti[:, 3]
    g = tf[np.flatnonzero(kind == TokenKind.GLOBAL)[0]]
    in_combat = bool(g[13])
    r: dict = {"d": int(o.header[1]), "a": int(action),
               "g": [_i(g[0]), _i(g[1]), _i(g[4]), _i(g[5]), _i(g[3]), _i(g[6]), _i(g[7]), _i(g[9]), int(in_combat), _i(g[14])]}
    top = np.argsort(-probs)[:5]
    r["pr"] = [[int(k), round(float(probs[k]), 3)] for k in top if probs[k] >= 0.005]

    powers: dict[int, list] = {}
    pvals: dict[int, list] = {}
    for j in np.flatnonzero(kind == TokenKind.POWER):
        powers.setdefault(int(owner[j]), []).append([int(vid[j]), _i(tf[j, 0])])
        pvals.setdefault(int(owner[j]), []).append(_i(tf[j, 2]))
    moves: dict[int, list] = {}
    for j in np.flatnonzero(kind == TokenKind.MOVE):
        moves.setdefault(int(owner[j]), []).append([int(vid[j]), _i(tf[j, 0]), round(float(tf[j, 1]), 2), _i(tf[j, 5]), _i(tf[j, 4])])
    if in_combat:
        en = []
        for j in np.flatnonzero(kind == TokenKind.CREATURE):
            f = tf[j]
            if f[5]:
                r["pl"] = {"hp": _i(f[0]), "m": _i(f[1]), "b": _i(f[2]), "p": powers.get(int(j), []), "pv": pvals.get(int(j), []),
                           "et": [round(float(x), 1) for x in f[16:20]]}
                st = np.flatnonzero((kind == TokenKind.COMBAT_STAT) & (owner == j))
                if len(st):
                    fs = tf[st[0]]
                    r["pl"]["et"] += [round(float(fs[13]), 1), _i(fs[14])]
                    r["orbx"] = [_i(fs[15]), _i(fs[16]), _i(fs[17]), _i(fs[18])]
            elif zone[j] == Zone.ENEMY:
                en.append({"n": int(vid[j]), "s": int(ti[j, 4]), "hp": _i(f[0]), "m": _i(f[1]), "b": _i(f[2]), "al": int(f[3]),
                           "p": powers.get(int(j), []), "pv": pvals.get(int(j), []), "mv": moves.get(int(j), []),
                           "et": [round(float(x), 1) for x in f[16:20]],
                           "it": [_i(f[6]), _i(f[7]), [int(x) for x in f[9:16]]]})
        r["en"] = en
        r["orb"] = [[int(vid[j]), _i(tf[j, 1]), _i(tf[j, 2])] for j in np.flatnonzero(kind == TokenKind.ORB)]
        r["os"] = _i(g[10])
        cards = kind == TokenKind.CARD
        r["hand"] = [[int(vid[j]), _i(tf[j, 3]), _i(tf[j, 0]) if not tf[j, 1] else -1, _i(tf[j, 4]), _i(tf[j, 5]), _i(tf[j, 6])]
                     for j in np.flatnonzero(cards & (zone == Zone.HAND))]
        top = cards & (zone >= Zone.DRAW_TOP1) & (zone <= Zone.DRAW_TOP3)
        r["draw"] = int((cards & (zone == Zone.DRAW)).sum() + top.sum())
        r["drawl"] = [[int(vid[j]), _i(tf[j, 3])] for j in np.flatnonzero(cards & (zone == Zone.DRAW))]
        # 观测格式 20（储君）：星、手牌星费（没有星费为 −1，星 X 费为 −2），玩家知道位置的抽牌堆顶部（从顶部起，[卡, 升级]）；没有时不记
        if int(o.header[0]) >= 20:
            r["star"] = _i(g[8])
            r["hstar"] = [-2 if tf[j, 1] == 2 else _i(tf[j, 2]) for j in np.flatnonzero(cards & (zone == Zone.HAND))]
        if top.any():
            r["top"] = [[int(vid[j]), _i(tf[j, 3])] for j in np.flatnonzero(top)]
        r["disc"] = [[int(vid[j]), _i(tf[j, 3])] for j in np.flatnonzero(cards & (zone == Zone.DISCARD))]
        r["exh"] = [[int(vid[j]), _i(tf[j, 3])] for j in np.flatnonzero(cards & (zone == Zone.EXHAUST))]
    else:
        r["deck"] = [[int(vid[j]), _i(tf[j, 3])] for j in np.flatnonzero((kind == TokenKind.CARD) & (zone == Zone.DECK))]
    cx = []
    for j in np.flatnonzero((kind == TokenKind.CARD) | (kind == TokenKind.SELECTED_CARD)):
        aff = powers.get(int(j))
        if ti[j, 4] or aff:
            cx.append([int(zone[j]), int(vid[j]), _i(tf[j, 3]), int(ti[j, 4]), _i(tf[j, 13]), aff[0][0] if aff else 0, aff[0][1] if aff else 0])
    if cx:
        r["cx"] = cx
    rt = np.flatnonzero(kind == TokenKind.ROUTE_SUMMARY)
    if len(rt):
        r["rt"] = [[int(vid[j]), _i(tf[j, 0]), _i(tf[j, 16]), _i(tf[j, 17]), _i(tf[j, 18]), _i(tf[j, 19])] for j in rt]
    r["rel"] = [[int(vid[j]), _i(tf[j, 0]) if tf[j, 1] else -1] for j in np.flatnonzero((kind == TokenKind.RELIC) & (zone == Zone.ALLY))]
    # 补全（第十一份外部评审，决策记录 204）：rel、cx 保持原样（胜率模型读它们），另记遗物的全部特征（计数、蜡制、失效、激活、本场已触发、
    # 融化顺序等，记录特征 0–7）与每张牌的全部附加记录（附魔失效、苦痛、费用有效期、重放、副本、疯狂科学等，每条带能力记录特征 0–6），可以完整复核观测
    r["relx"] = [[int(vid[j])] + [_i(x) for x in tf[j, :8]] for j in np.flatnonzero((kind == TokenKind.RELIC) & (zone == Zone.ALLY))]
    cxa = []
    for j in np.flatnonzero((kind == TokenKind.CARD) | (kind == TokenKind.SELECTED_CARD)):
        att = [[int(vid[k])] + [_i(x) for x in tf[k, :7]] for k in np.flatnonzero((kind == TokenKind.POWER) & (owner == j))]
        if att:
            cxa.append([int(zone[j]), int(vid[j]), _i(tf[j, 3]), att])
    if cxa:
        r["cxa"] = cxa
    r["pot"] = [[int(vid[j]), _i(tf[j, 0])] for j in np.flatnonzero((kind == TokenKind.POTION) & (zone == Zone.ALLY))]
    mp = np.flatnonzero(kind == TokenKind.MAP_NODE)
    if len(mp) and int(o.header[1]) == 11:
        r["map"] = [[_i(tf[j, 0]), _i(tf[j, 1]), int(vid[j]), _i(tf[j, 2]), _i(tf[j, 4]), _i(tf[j, 5]), _i(tf[j, 6]), _i(tf[j, 7]), _i(tf[j, 8])] for j in mp]
    sp = np.flatnonzero(kind == TokenKind.SELECT_PROMPT)
    if len(sp):
        f = tf[sp[0]]
        purpose = next((k for k in range(7) if f[k]), -1)
        r["sel"] = [purpose, _i(f[7]), _i(f[8]), _i(f[9]), int(vid[sp[0]])]
    cells = np.flatnonzero((kind == TokenKind.SPHERE_CELL) & (tf[:, 19] == 0))     # 版本 15 起另有还没露面的物品清单（19 为 1），不算揭开的格子
    if len(cells):
        r["sp"] = [len(cells), _i(tf[cells[0], 15])]
    # 商店：整个货架（引擎附加信息，含买不起、栏满不能买、已售出的），只供回放显示
    shop = (o.extra or {}).get("shop") if getattr(o, "extra", None) else None
    if shop:
        idx = _vocab_index()
        r["shop"] = {"g": int(shop.get("gold", 0)), "full": int(bool(shop.get("slots_full"))),
                     "it": [[_SHOP_KIND.get(x["k"], -1), idx.get(x.get("id") or "", -1), int(x["cost"]), int(bool(x["stock"])), int(bool(x["gold"])),
                             int(x.get("up", 0)), int(bool(x.get("used")))] for x in shop.get("items", [])]}
    ci, cf = o.cand_ints, o.cand_feats
    r["c"] = [[int(ci[k, 0]), _obj(o, int(ci[k, 1])), _obj(o, int(ci[k, 2])), int(ci[k, 3]), _i(cf[k, 0]), _i(cf[k, 1]), _i(cf[k, 2])]
              for k in range(o.num_actions)]
    return r
