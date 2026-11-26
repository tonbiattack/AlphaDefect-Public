"""格式 29 包 5（生物与局外）的核对（研究报告/2026-11-20_全角色观测缺口调研/R3-格式29规格.md 5.5）。

两组引擎进程锁步走同一局：一组 reset 带 obs_format 28，一组带 29；动作由格式 28 的帧决定（检查点按格式 28 训练），
两边局面逐步相同。每一条按规格的“核对”在两种格式上各算一遍：新列只在 29 有，28 一栏给旧方法（如招式 5）或“无此列”。

  python -I tools/obs29/creatures.py combat --dll <新引擎> --n 40          # 单场战斗：G24、G26、睡眠类、G03、G44、G21、G04、Boss 的 G05
  python -I tools/obs29/creatures.py run --dll <新引擎> --ckpt <检查点> --runs 40   # 铁甲整局（检查点贪心）：G05、G21、J5、J3、J4、J12
  python -I tools/obs29/creatures.py events --dll <新引擎> --runs 200     # 随机整局（低血量进事件）：J4、J12、J3

各条的口径：
  G05  第一个决策帧（每个玩家回合开头）的第 k 步预告（概率 1），若 k 回合后实际出的是同一招，比较推演（29：招式 22；28：招式 5）与那时的招式 5；
       另比“那时的招式 22”（含同回合先出手的队友加成）。按怪物分类，“期间玩家改变过它”（上减益、改力量、打死过）的样本单列。
  睡眠 实际醒来的那一步（前一步还在睡，期间没掉血），比较第 k 步预告的招式 ID 与实际。
  G24  复活中（生物 24 = 1）的每一帧：有没有招式记录；25 与“下个玩家回合第一帧它活着”；活着时 26 与那时的生命。
  G26  瀑布巨兽：27 = 1 与格式 28 的生命 999,999,999 一一对应；倒地那一回合第 1 步“爆炸”的预告与下个玩家回合第 0 步的爆炸（招式 5）。
  G04  出牌、用药水之前 23 = 1 的敌人，这一步之后死了：战斗当场结束（胜）。
  G03  律动残余：遗物 22 与 max(0, 20 − 战斗计数 6)。
  G44  回声形态：能力 22 > 0 与“这张牌打了两次”（战斗计数 1 加 2）。
  G21  结束回合的那一帧：全局 22 与下个玩家回合第一帧的能量（全局 6）；28 一栏为“下回合能量 = 能量上限”。
  J5   战斗外的最后一帧：全局 23 与下一场战斗第 2 回合的能量上限（全局 7）。
  J3   奖励页的特殊卡牌奖励：候选 A 的记录（29 为卡牌）与领取后牌组里多出的牌。
  J4   选了的事件选项：选项 22 与“选完就死”。
  J12  审判事件：选项 8–11、16–19 里出现 101–999 的帧。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

import numpy as np  # noqa: E402

from sts2env import VecEnv  # noqa: E402
from sts2env.protocol import ActionKind, DecisionKind, TokenKind, Zone  # noqa: E402

DLL = REPO / "engine/Sts2Headless/bin/Release/net9.0/Sts2Headless.dll"
WAKE = {"MONSTER.LAGAVULIN_MATRIARCH": ("SLEEP_MOVE", "SLASH_MOVE"), "MONSTER.SLUMBERING_BEETLE": ("SNORE_MOVE", "ROLL_OUT_MOVE")}
BACK = ("POWER.BACK_ATTACK_LEFT_POWER", "POWER.BACK_ATTACK_RIGHT_POWER")
CATS = {"MONSTER.TORCH_HEAD_AMALGAM": "女王火炬头", "MONSTER.AEONGLASS": "永世沙漏", "MONSTER.BYGONE_EFFIGY": "往日雕像"}


# 包 5 会改到的旧列与记录结构（给 tools/obs_v29_check.py 的登记用，R3 规格第三节 3.3 与第七节第 2 步）：
# (条目, 位置, 什么时候可能起作用)。位置同规格："rec" 为（记录类型, 列）；"struct" 为记录条数或记录整数不同。
FIXES = [
    ("G05", ("rec-id", TokenKind.MOVE), "有乐加维林族母（沉睡）或沉睡甲虫（酣睡）的战斗：第 k ≥ 1 步、层数 ≤ k 时招式记录的词表 ID 改为醒来的招式，之后几步随之改"),
    ("G24", ("struct", "复活中的怪多出招式记录、实验体的形态记录"), "敌人身上有重新接上 / 适者生存 / 幻影，且已死"),
    ("G24", ("rec", TokenKind.CREATURE, tuple(range(6, 16))), "同上：复活中的怪输出意图标志与意图伤害"),
    ("G26", ("rec", TokenKind.CREATURE, (0, 1, 17, 18, 19)), "瀑布巨兽倒地（生命 999,999,999）"),
    ("G34", ("struct", "特殊卡牌奖励的选项记录换成卡牌记录，候选对象 A 随之改"), "奖励页上有 REWARD:SpecialCardReward"),
    ("G62", ("rec", TokenKind.OPTION, (8, 9, 10, 11, 16, 17, 18, 19)), "审判（TRIAL）事件的选项"),
]


# ---------------------------------------------------------------- 帧的摘要

def extract(o, V: list[str]) -> dict:
    """一帧里核对要用的东西（只留小数据，整局帧不全存）。"""
    ti, tf = o.tok_ints, o.tok_feats
    n = ti.shape[0]
    d = {"dec": int(o.header[1]), "done": bool(o.done), "vic": bool(o.victory), "na": int(o.num_actions), "hp": int(o.hp)}
    gl = [i for i in range(n) if ti[i, 0] == TokenKind.GLOBAL]
    g = gl[0] if gl else None
    d["inc"] = bool(g is not None and tf[g, 13] > 0)
    d["turn"] = int(tf[g, 9]) if g is not None else 0
    d["energy"], d["maxE"], d["g22"], d["g23"] = (float(tf[g, 6]), float(tf[g, 7]), float(tf[g, 22]), float(tf[g, 23])) if g is not None else (0, 0, 0, 0)
    cre, cidx = [], {}
    seen = Counter()
    for i in range(n):
        if ti[i, 0] != TokenKind.CREATURE:
            continue
        vid = int(ti[i, 1]); zone = int(ti[i, 2])
        key = None
        if zone == Zone.ENEMY:
            key = (vid, seen[vid]); seen[vid] += 1
        c = {"vid": vid, "name": V[vid], "zone": zone, "key": key, "alive": tf[i, 3] > 0, "hp": float(tf[i, 0]), "maxhp": float(tf[i, 1]),
             "block": float(tf[i, 2]), "y": tf[i, 22:30].copy(), "f17": float(tf[i, 17]), "moves": [], "pw": {}, "deb": 0.0, "str": 0.0}
        cidx[i] = len(cre); cre.append(c)
    d["cnt"] = dict(seen)
    for i in range(n):
        k, vid, zone, own = (int(x) for x in ti[i, :4])
        if own in cidx:
            c = cre[cidx[own]]
            if k == TokenKind.MOVE:
                f = tf[i]
                c["moves"].append({"step": int(f[0]), "p": float(f[1]), "vid": vid, "name": V[vid].split(".", 1)[-1], "f4": float(f[4]), "f5": float(f[5]),
                                   "f6": float(f[6]), "atk": bool(f[7] > 0), "y": (float(f[22]), float(f[23]), float(f[24]))})
            elif k == TokenKind.POWER:
                nm = V[vid]
                c["pw"][nm] = (float(tf[i, 0]), float(tf[i, 22]), float(tf[i, 8]))
                if tf[i, 1] > 0: c["deb"] += float(tf[i, 0])
                if nm == "POWER.STRENGTH_POWER": c["str"] = float(tf[i, 0])
            elif k == TokenKind.COMBAT_STAT and tf[i, 0] == 0 and c["zone"] == Zone.ALLY:
                d["cs1"], d["cs6"] = float(tf[i, 1]), float(tf[i, 6])
    d["cre"] = cre
    d["relics"] = {V[ti[i, 1]] for i in range(n) if ti[i, 0] == TokenKind.RELIC and ti[i, 2] == Zone.ALLY}
    d["remnant"] = next((float(tf[i, 22]) for i in range(n) if ti[i, 0] == TokenKind.RELIC and V[ti[i, 1]] == "RELIC.BEATING_REMNANT" and ti[i, 2] == Zone.ALLY), None)
    d["opts"] = {i: {"name": V[ti[i, 1]], "y": float(tf[i, 22]), "dv": np.concatenate([tf[i, 8:12], tf[i, 16:20]]).copy()}
                 for i in range(n) if ti[i, 0] == TokenKind.OPTION}
    d["deck"] = Counter(int(ti[i, 1]) for i in range(n) if ti[i, 0] == TokenKind.CARD and ti[i, 2] == Zone.DECK)
    m = o.num_actions
    d["cand"] = [(int(o.cand_ints[j, 0]), int(o.cand_ints[j, 1])) for j in range(m)]
    d["candA"] = [(int(ti[a, 0]), V[ti[a, 1]]) if 0 <= a < n else (None, None) for _, a in d["cand"]]
    return d


def find(fr: dict, key) -> dict | None:
    return next((c for c in fr["cre"] if c["key"] == key), None)


# ---------------------------------------------------------------- 统计

class Stat:
    def __init__(self):
        self.c = Counter()
        self.err = defaultdict(list)
        self.ex = defaultdict(list)

    def add(self, item: str, fmt: str, ok: bool, err: float | None = None, ex=None):
        self.c[(item, fmt, "n")] += 1
        self.c[(item, fmt, "ok")] += int(ok)
        if err is not None: self.err[(item, fmt)].append(err)
        if not ok and ex is not None and len(self.ex[(item, fmt)]) < 6: self.ex[(item, fmt)].append(ex)

    def rows(self):
        items = sorted({(k[0]) for k in self.c})
        for it in items:
            r = {"item": it}
            for fmt in ("28", "29"):
                n = self.c[(it, fmt, "n")]
                if n:
                    r[fmt] = {"n": n, "rate": self.c[(it, fmt, "ok")] / n,
                              "mae": float(np.mean(self.err[(it, fmt)])) if self.err[(it, fmt)] else None}
            yield r


def turn_firsts(F: list[dict]) -> list[tuple[int, int, int]]:
    """（帧下标，战斗序号，回合）：每个玩家回合的第一个战斗决策帧。"""
    out, cid, last_t, was = [], 0, None, False
    for i, fr in enumerate(F):
        if not fr["inc"]:
            if was: cid += 1
            was = False; last_t = None
            continue
        if not was: was = True
        if fr["dec"] == DecisionKind.COMBAT and fr["turn"] != last_t:
            if last_t is not None and fr["turn"] < last_t: cid += 1
            out.append((i, cid, fr["turn"])); last_t = fr["turn"]
    return out


def flagged(F29: list[dict], i0: int, i1: int, key) -> bool:
    """i0..i1 之间，玩家回合里（同一回合的相邻帧）这个怪物的减益层数涨了、力量变了（虚弱、减力量，或激怒这类因玩家出牌涨力量），
    或者它被打死过（千足虫节段、实验体复活后力量清零），或者凯撒蟹战里玩家转了身（背刺变了）：期间玩家改变过它，推演不负责这一部分。"""
    for a in range(i0, i1):
        x, y = F29[a], F29[a + 1]
        if not (x["inc"] and y["inc"]): continue
        cx, cy = find(x, key), find(y, key)
        if cx and cy and cx["alive"] and not cy["alive"]:
            return True
        if cx and cy and any(cx["pw"].get(p, (0, 0, 0))[2] != cy["pw"].get(p, (0, 0, 0))[2] for p in BACK):
            return True         # 凯撒蟹：玩家转身，钳子从背后打（背刺 ×1.5）变了
        if x["turn"] != y["turn"]: continue
        if cx and cy and (cy["deb"] > cx["deb"] + 1e-6 or abs(cy["str"] - cx["str"]) > 1e-6):
            return True
    return False


def analyze(ep: list[tuple[dict, dict, int | None]], S: Stat):
    F28 = [x[0] for x in ep]; F29 = [x[1] for x in ep]; acts = [x[2] for x in ep]
    firsts = turn_firsts(F29)
    byturn = {(cid, t): i for i, cid, t in firsts}
    # ---- G05 招式推演、睡眠类
    for i, cid, t in firsts:
        for c29 in F29[i]["cre"]:
            if c29["key"] is None or not c29["moves"]: continue
            key = c29["key"]
            c28 = find(F28[i], key)
            for k in (1, 2, 3):
                j = byturn.get((cid, t + k))
                if j is None: continue
                a29 = find(F29[j], key); a28 = find(F28[j], key)
                # 同名怪的个数在 i..j 之间变过（死了被移出、又召唤了新的）：按（ID，序号）认不准是不是同一只，跳过
                if a29 is None or any(F29[b]["inc"] and F29[b]["cnt"].get(key[0]) != F29[i]["cnt"].get(key[0]) for b in range(i, j + 1)): continue
                act0 = [m for m in a29["moves"] if m["step"] == 0]
                if not act0: continue
                act0 = act0[0]
                fl = flagged(F29, i, j, key)
                cat = CATS.get(c29["name"], "其他")
                for fmt, cc, ac in (("28", c28, a28), ("29", c29, a29)):
                    if cc is None or ac is None: continue
                    pk = [m for m in cc["moves"] if m["step"] == k]
                    if len(pk) != 1 or pk[0]["p"] < 0.999: continue
                    pm = pk[0]
                    a0 = [m for m in ac["moves"] if m["step"] == 0]
                    if not a0: continue
                    a0 = a0[0]
                    if pm["vid"] != a0["vid"] or not a0["atk"]: continue
                    pred = pm["y"][0] if fmt == "29" else pm["f5"]
                    truth = a0["f5"]
                    tag = f"G05 第{k}步 {cat}" + ("（期间玩家改变过它）" if fl else "")
                    S.add(tag, fmt, abs(pred - truth) < 0.5, abs(pred - truth), ex=(c29["name"], pm["name"], pred, truth))
                    if not fl:
                        S.add("G05 第1–3步 全部（期间没改变）", fmt, abs(pred - truth) < 0.5, abs(pred - truth))
                        if fmt == "29":
                            t2 = a0["y"][0]
                            S.add("G05 第1–3步 全部（期间没改变，对那时的招式 22）", fmt, abs(pred - t2) < 0.5, abs(pred - t2))
                        else:
                            S.add("G05 第1–3步 全部（期间没改变，对那时的招式 22）", fmt, abs(pred - truth) < 0.5, abs(pred - truth))
                        hp_, ht = (pm["y"][1] if fmt == "29" else pm["f4"]), a0["f4"]
                        S.add("G05 段数（期间没改变）", fmt, abs(hp_ - ht) < 0.5, abs(hp_ - ht))
                # 睡眠类：实际醒来的那一步
                if c29["name"] in WAKE and k >= 1:
                    sleep, wake = WAKE[c29["name"]]
                    jp = byturn.get((cid, t + k - 1))
                    if jp is None: continue
                    prev = find(F29[jp], key)
                    if prev is None: continue
                    p0 = [m for m in prev["moves"] if m["step"] == 0]
                    if not p0 or p0[0]["name"] != sleep or act0["name"] != wake: continue
                    lost = any((find(F29[a], key) or {"hp": 0})["hp"] < c29["hp"] for a in range(i, j + 1) if F29[a]["inc"])
                    if lost: continue
                    for fmt, cc in (("28", c28), ("29", c29)):
                        pk = [m for m in (cc or {"moves": []})["moves"] if m["step"] == k]
                        ok = len(pk) == 1 and pk[0]["name"] == wake
                        S.add("睡眠类 醒来那一步的招式 ID", fmt, ok, ex=(c29["name"], k, [m["name"] for m in pk]))
    # ---- 逐帧：G24、G26、G03、G21、J5、G04、G44、J3、J4、J12
    n = len(ep)
    combat_of = {}
    cid = 0; was = False
    for i, fr in enumerate(F29):
        if not fr["inc"]:
            if was: cid += 1
            was = False
        else:
            was = True; combat_of[i] = cid
    def next_first(i: int) -> int | None:
        if i not in combat_of: return None
        return byturn.get((combat_of[i], F29[i]["turn"] + 1))
    for i in range(n):
        x29, x28, a = F29[i], F28[i], acts[i]
        if x29["inc"]:
            j = next_first(i)
            for c in x29["cre"]:
                if c["key"] is None: continue
                y = c["y"]
                c28 = find(x28, c["key"])
                # G24
                if y[2] > 0:
                    S.add("G24 复活窗口有招式记录", "29", len(c["moves"]) > 0, ex=(c["name"],))
                    S.add("G24 复活窗口有招式记录", "28", c28 is not None and len(c28["moves"]) > 0)
                    if j is not None and F29[j]["cnt"].get(c["key"][0]) == x29["cnt"].get(c["key"][0]):
                        nx = find(F29[j], c["key"])
                        if nx is not None:
                            S.add(f"G24 25 与下回合活着 {c['name']}", "29", (y[3] > 0) == nx["alive"], ex=(c["name"], float(y[3]), nx["alive"], [m["name"] for m in c["moves"]]))
                            if nx["alive"]:
                                # 实验体换形态、幻影回满：复活后的生命 = 那时的最大生命（当前生命可能已被回合开始的伤害扣掉，如水银沙漏）；千足虫节段回 25 血，比当前生命
                                tv = nx["maxhp"] if c["name"] == "MONSTER.TEST_SUBJECT" or "POWER.ILLUSION_POWER" in c["pw"] else nx["hp"]
                                S.add(f"G24 26 与复活后的生命 {c['name']}", "29", abs(y[4] - tv) < 0.5, abs(y[4] - tv), ex=(c["name"], float(y[4]), nx["hp"], nx["maxhp"]))
                                S.add("G24 26 不小于下回合第一帧的生命（回合开始的伤害会先扣掉）", "29", y[4] >= nx["hp"] - 0.5)
                # G26
                if c["name"] == "MONSTER.WATERFALL_GIANT" and c["alive"] and c28 is not None:
                    down28 = c28["hp"] > 1e8
                    S.add("G26 27 只在倒地窗口为 1", "29", (y[5] > 0) == down28 and (y[5] == 0 or (c["hp"] == 0 and c["maxhp"] == 0)), ex=(float(y[5]), c28["hp"], c["hp"]))
                    S.add("G26 倒地时生命不出 999,999,999", "28", not down28 or c28["hp"] < 1e8)
                    if y[5] > 0: S.add("G26 倒地时生命不出 999,999,999", "29", c["hp"] < 1e8)
                    m0 = [m for m in c["moves"] if m["step"] == 0]
                    if m0 and m0[0]["name"] == "ABOUT_TO_BLOW_MOVE" and j is not None:
                        nx = find(F29[j], c["key"])
                        e0 = [m for m in (nx or {"moves": []})["moves"] if m["step"] == 0 and m["name"] == "EXPLODE_MOVE"]
                        if e0:
                            truth = e0[0]["f5"]
                            for fmt, cc in (("28", c28), ("29", c)):
                                e1 = [m for m in cc["moves"] if m["step"] == 1 and m["name"] == "EXPLODE_MOVE"]
                                pred = (e1[0]["y"][0] if fmt == "29" else e1[0]["f5"]) if e1 else -1
                                S.add("G26 第1步爆炸 对下回合的爆炸伤害", fmt, abs(pred - truth) < 0.5, abs(pred - truth), ex=(pred, truth))
            # G03
            if x29.get("remnant") is not None and "cs6" in x29:
                truth = max(0.0, 20 - x29["cs6"])
                S.add("G03 遗物 22 与 20 − 本回合已失去的生命", "29", abs(x29["remnant"] - truth) < 0.5, abs(x29["remnant"] - truth), ex=(x29["remnant"], truth))
            # G21
            if x29["dec"] == DecisionKind.COMBAT and a is not None and x29["cand"][a][0] == ActionKind.END_TURN and j is not None:
                truth = F29[j]["energy"]
                me_ = next((c for c in x29["cre"] if c["zone"] == Zone.ALLY and c["name"].startswith("CHARACTER.")), None)
                S.add("G21 全局 22 与下回合开头的能量", "29", abs(x29["g22"] - truth) < 0.5, abs(x29["g22"] - truth),
                      ex=(x29["g22"], truth, x29["energy"], x29["maxE"], x29["turn"], sorted(me_["pw"]) if me_ else None, sorted(x29["relics"])))
                S.add("G21 全局 22 与下回合开头的能量", "28", abs(x28["maxE"] - truth) < 0.5, abs(x28["maxE"] - truth))
            # G04、G44：玩家的一步
            if x29["dec"] == DecisionKind.COMBAT and a is not None and x29["cand"][a][0] in (ActionKind.PLAY_CARD, ActionKind.USE_POTION) and i + 1 < n:
                nx = F29[i + 1]
                for c in x29["cre"]:
                    if c["key"] is None or not c["alive"] or c["y"][1] <= 0: continue
                    if not nx["inc"] or nx["done"]:
                        if nx["done"] and not nx["vic"]: continue        # 玩家死了，不是打死它
                        killed, ended = True, True
                    else:
                        vid = c["key"][0]; after = find(nx, c["key"])
                        killed = nx["cnt"].get(vid, 0) < x29["cnt"].get(vid, 0) or after is None or not after["alive"]
                        ended = False
                    if killed:
                        S.add("G04 23=1 的敌人被打死时战斗当场结束", "29", ended, ex=(c["name"], nx["inc"], nx["done"], nx["vic"]))
                me = next((c for c in x29["cre"] if c["zone"] == Zone.ALLY and c["name"].startswith("CHARACTER.")), None)
                if me and "POWER.ECHO_FORM_POWER" in me["pw"] and x29["cand"][a][0] == ActionKind.PLAY_CARD and nx["inc"] and nx["turn"] == x29["turn"] and "cs1" in nx and "cs1" in x29:
                    left = me["pw"]["POWER.ECHO_FORM_POWER"][1]
                    d1 = nx["cs1"] - x29["cs1"]
                    S.add("G44 能力 22 > 0 与这张牌打了两次", "29", (left > 0) == (d1 >= 2), ex=(left, d1))
        # J5：战斗外的最后一帧 → 下一场战斗第 2 回合的能量上限
        if not x29["inc"] and i + 1 < n and F29[i + 1]["inc"]:
            k1 = next((b for b in range(i + 1, min(n, i + 40)) if F29[b]["inc"] and F29[b]["dec"] == DecisionKind.COMBAT), None)
            if k1 is not None and F29[k1]["turn"] == 1:
                # 第 1 回合的第一帧（还没出牌）：面包第 1 回合不加，按第 2 回合起的上限补回 1；佩尔之肉第 3 回合起才加，全局 23 本来就不算
                truth = F29[k1]["maxE"] + (1 if "RELIC.BREAD" in F29[k1]["relics"] else 0)
                S.add("J5 全局 23 与下一场开局的能量上限", "29", abs(x29["g23"] - truth) < 0.5, abs(x29["g23"] - truth), ex=(x29["g23"], truth, sorted(F29[k1]["relics"])[:12]))
            k2 = next((b for b in range(i + 1, min(n, i + 400)) if F29[b]["inc"] and F29[b]["turn"] == 2), None)
            if k2 is not None and all(F29[b]["inc"] for b in range(i + 1, k2 + 1)):
                S.add("J5 全局 23 与下一场第 2 回合的能量上限（含第 1 回合里打出的加上限的牌）", "29", abs(x29["g23"] - F29[k2]["maxE"]) < 0.5, abs(x29["g23"] - F29[k2]["maxE"]), ex=(x29["g23"], F29[k2]["maxE"]))
        # J3：特殊卡牌奖励
        if x29["dec"] == DecisionKind.REWARDS:
            for jj, (kind, _) in enumerate(x29["cand"]):
                if kind != ActionKind.TAKE_REWARD: continue
                k29, n29 = x29["candA"][jj]; k28, n28 = x28["candA"][jj]
                if n28 != "REWARD:SpecialCardReward": continue
                S.add("J3 特殊卡牌奖励的对象 A 是卡牌记录", "29", k29 == TokenKind.CARD, ex=(k29, n29))
                S.add("J3 特殊卡牌奖励的对象 A 是卡牌记录", "28", k28 == TokenKind.CARD)
                if a == jj:
                    after = next((b for b in range(i + 1, min(n, i + 6)) if F29[b]["deck"]), None)
                    if after is not None:
                        got = [VOCAB[k] for k, v in (F29[after]["deck"] - x29["deck"]).items() if v > 0]
                        S.add("J3 卡牌记录的词表 ID 与领回的牌", "29", k29 == TokenKind.CARD and n29 in got, ex=(n29, got))
        # J4：选了的事件选项
        if x29["dec"] == DecisionKind.EVENT and a is not None and x29["cand"][a][0] == ActionKind.EVENT_OPTION:
            o = x29["opts"].get(x29["cand"][a][1])
            if o is not None and i + 1 < n:
                nx = F29[i + 1]
                died = nx["done"] and not nx["vic"] and nx["hp"] <= 0
                S.add("J4 选项 22 与选完就死", "29", (o["y"] > 0) == died, ex=(o["name"], o["y"], died, nx["hp"]))
                if o["y"] > 0 or died:
                    S.add("J4 选项 22 与选完就死（其中 22 = 1 或选完就死的）", "29", (o["y"] > 0) == died, ex=(o["name"], o["y"], died, nx["hp"]))
        if x29["dec"] == DecisionKind.EVENT:
            S.c[("J4 事件帧里有选项 22 = 1", "29", "n")] += 1
            S.c[("J4 事件帧里有选项 22 = 1", "29", "ok")] += int(any(o["y"] > 0 for o in x29["opts"].values()))
            for fmt, xx in (("28", x28), ("29", x29)):
                tri = [o for o in xx["opts"].values() if o["name"].startswith("EVENTOPT:TRIAL.")]
                if tri:
                    S.add("J12 审判选项 8–11、16–19 没有 101–999 的装饰数", fmt, not any(((o["dv"] >= 101) & (o["dv"] <= 999)).any() for o in tri), ex=[o["dv"].tolist() for o in tri][:1])


VOCAB: list[str] = []


# ---------------------------------------------------------------- 驱动

def drive(dll: Path, envs: int, reqs: list[dict], policy, S: Stat, max_steps: int, log=None, partial=None):
    """reqs：每局一个 reset 请求（不含 obs_format）。两组进程锁步，动作由格式 28 的帧决定。"""
    global VOCAB
    E28, E29 = VecEnv(envs, dll), VecEnv(envs, dll)
    VOCAB = E29.envs[0].vocab()
    todo = list(range(len(reqs)))
    slot: dict[int, int | None] = {}
    eps: dict[int, list] = {}
    o28, o29, steps = {}, {}, {}
    def start(i):
        if not todo:
            slot[i] = None; return
        r = todo.pop(0); slot[i] = r; eps[i] = []; steps[i] = 0
        a = E28.reset_some({i: {**reqs[r], "obs_format": 28}})[i]; b = E29.reset_some({i: {**reqs[r], "obs_format": 29}})[i]
        o28[i], o29[i] = a, b
    for i in range(envs): start(i)
    t0 = time.time(); done_n = 0
    T = Counter()
    while any(s is not None for s in slot.values()):
        live = [i for i in range(envs) if slot[i] is not None]
        tp = time.time()
        acts = policy([o28[i] for i in live])
        T["policy"] += time.time() - tp
        step = [None] * envs
        for i, a in zip(live, acts):
            x, y = o28[i], o29[i]
            fin = x.done or x.num_actions == 0 or steps[i] >= max_steps
            if not fin and (y.num_actions != x.num_actions):
                S.c[("锁步：候选数不同", "29", "n")] += 1
                fin = True
            te = time.time()
            eps[i].append((extract(x, VOCAB), extract(y, VOCAB), None if fin else a))
            T["extract"] += time.time() - te
            # 帧级的旧列对比（同 v29_launch_check 的口径，另把记录结构不同单列）
            struct = x.tok_ints.shape != y.tok_ints.shape or not np.array_equal(x.tok_ints, y.tok_ints)
            old = not struct and not np.array_equal(x.tok_feats[:, :22], y.tok_feats[:, :22])
            cint = x.cand_ints.shape != y.cand_ints.shape or not np.array_equal(x.cand_ints, y.cand_ints)
            cand = x.cand_feats.shape != y.cand_feats.shape or not np.array_equal(x.cand_feats[:, :14], y.cand_feats[:, :14])
            for nm, v in (("帧比例：记录结构不同（条数或记录整数）", struct), ("帧比例：记录结构相同、旧列 0–21 不同", old),
                          ("帧比例：候选整数不同（记录多了，下标平移）", cint), ("帧比例：候选旧列 0–13 不同", cand)):
                S.c[(nm, "29", "n")] += 1; S.c[(nm, "29", "ok")] += int(v)
                if v:       # 差异出在哪类帧：规格登记的几种（特殊卡牌奖励、复活中、瀑布巨兽倒地、睡眠类、审判）与其余
                    f28, f29 = eps[i][-1][0], eps[i][-1][1]
                    why = ("特殊卡牌奖励" if any(f28_o["name"] == "REWARD:SpecialCardReward" for f28_o in f28["opts"].values()) else
                           "复活中" if any(c_["y"][2] > 0 for c_ in f29["cre"]) else
                           "瀑布巨兽倒地" if any(c_["y"][5] > 0 for c_ in f29["cre"]) else
                           "睡眠类" if any(c_["name"] in WAKE for c_ in f29["cre"]) else
                           "审判" if any(o_["name"].startswith("EVENTOPT:TRIAL.") for o_ in f29["opts"].values()) else "其他")
                    S.c[(f"{nm} 其中：{why}", "29", "n")] += 1; S.c[(f"{nm} 其中：{why}", "29", "ok")] += 1
                    if why == "其他" and len(S.ex[(nm, "其他")]) < 6:
                        k28 = Counter((int(t[0]), VOCAB[t[1]]) for t in x.tok_ints); k29 = Counter((int(t[0]), VOCAB[t[1]]) for t in y.tok_ints)
                        S.ex[(nm, "其他")].append({"dec": int(x.header[1]), "多": list((k29 - k28).items())[:8], "少": list((k28 - k29).items())[:8], "n": [len(x.tok_ints), len(y.tok_ints)]})
            if fin:
                analyze(eps[i], S); done_n += 1; del eps[i]
                if log: print(f"  {done_n}/{len(reqs)} 局（{time.time() - t0:.0f} 秒）", file=log, flush=True)
                if partial and done_n % 4 == 0: partial(done_n)
                start(i)
                continue
            step[i] = a; steps[i] += 1
        ts = time.time()
        n28, n29 = E28.step(step), E29.step(step)
        T["engine"] += time.time() - ts
        for i in range(envs):
            if step[i] is not None:
                o28[i], o29[i] = n28[i], n29[i]
    E28.close(); E29.close()
    if log: print(f"  耗时：策略 {T['policy']:.0f} 秒，帧摘要 {T['extract']:.0f} 秒，引擎 {T['engine']:.0f} 秒", file=log, flush=True)


def random_policy(rng: random.Random, p_play=0.85):
    def act(obs):
        out = []
        for o in obs:
            n = o.num_actions
            if n <= 0: out.append(0); continue
            if o.decision == DecisionKind.COMBAT:
                kinds = o.cand_ints[:n, 0]
                play = [k for k in range(n) if kinds[k] in (ActionKind.PLAY_CARD, ActionKind.USE_POTION)]
                if play and rng.random() < p_play:
                    out.append(rng.choice(play)); continue
            out.append(rng.randrange(n))
        return out
    return act


def ckpt_policy(path: Path):
    import torch
    from sts2rl.actor import MODE_RUN
    from sts2rl.features import Ragged, collate_ragged
    from sts2rl.model import ModelConfig, PolicyValueNet, load_state
    torch.set_num_threads(4)
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if "config" in blob:
        mc, st = dict(blob["config"]["model"]), blob["state"]
    else:
        mc, st = dict(json.load(open(REPO / "tools/remote/ironclad/train_config.json"))["model"]), blob
    mc.pop("obs_v29", None)
    net = PolicyValueNet(ModelConfig(**mc)).eval(); load_state(net, st)

    def act(obs):
        if not obs: return []
        with torch.no_grad():
            b = collate_ragged(Ragged.from_obs(obs, np.full(len(obs), MODE_RUN))).to("cpu")
            logits = net.forward_infer(b)[0]
        return [int(logits[k, :o.num_actions].argmax()) if o.num_actions > 0 else 0 for k, o in enumerate(obs)]
    return act


STARTER = [{"id": "STRIKE_IRONCLAD"}] * 5 + [{"id": "DEFEND_IRONCLAD"}] * 4 + [{"id": "BASH"}]
STRONG = [{"id": "BLUDGEON", "upgrades": 1}] * 6 + [{"id": "WHIRLWIND", "upgrades": 1}] * 2 + [{"id": "OFFERING"}] * 2 + [{"id": "DEFEND_IRONCLAD"}] * 3
DEFEND = [{"id": "DEFEND_IRONCLAD", "upgrades": 1}] * 10
ENERGY_RELICS = [["HAPPY_FLOWER", "CANDELABRA", "CHANDELIER"], ["ICE_CREAM", "ART_OF_WAR"], ["PAELS_TEARS", "SEAL_OF_GOLD"], ["BREAD", "PAELS_FLESH", "HAPPY_FLOWER"], ["ICE_CREAM", "CANDELABRA", "PAELS_TEARS"]]


def combat_specs(n: int) -> list[tuple[str, dict]]:
    G = []
    def add(group, k, **spec):
        base = {"character": "IRONCLAD", "ascension": 10, "seed": f"OBS29{group}{k}", "hp": 300, "max_hp": 300}
        G.append((group, {"op": "reset", "mode": "combat", "spec": {**base, **spec}, "loop_cut": True, "stall_turns": 12}))
    for k in range(n):
        add("G24 千足虫", k, act=1, encounter="DECIMILLIPEDE_ELITE", deck=STRONG)
        add("G24 实验体", k, act=2, encounter="TEST_SUBJECT_BOSS", deck=STRONG)
        add("G24 黯影者", k, act=1, encounter="THE_OBSCURA_NORMAL", deck=STRONG)
        add("G26 瀑布巨兽", k, act=0, encounter="WATERFALL_GIANT_BOSS", deck=STRONG)
        add("睡眠 族母", k, act=0, encounter="LAGAVULIN_MATRIARCH_BOSS", deck=DEFEND if k % 2 == 0 else STARTER)
        add("睡眠 甲虫", k, act=1, encounter="SLUMBERING_BEETLE_NORMAL", deck=DEFEND if k % 2 == 0 else STARTER)
        add("G05 女王", k, act=2, encounter="QUEEN_BOSS", deck=STARTER)
        add("G05 永世沙漏", k, act=2, encounter="AEONGLASS_BOSS", deck=STARTER)
        add("G05 往日雕像", k, act=0, encounter="BYGONE_EFFIGY_ELITE", deck=STARTER)
        add("G03 律动残余", k, act=k % 3, encounter=["elite", "boss"][k % 2], deck=STARTER, relics=["BEATING_REMNANT"], hp=200, max_hp=200)
        add("G21 能量遗物", k, act=k % 3, encounter=["regular", "elite", "boss"][k % 3], deck=STARTER, relics=ENERGY_RELICS[k % len(ENERGY_RELICS)])
        add("G04 同族", k, act=0, encounter="THE_KIN_BOSS", deck=STRONG)
        G.append(("G44 回声形态", {"op": "reset", "mode": "combat", "loop_cut": True, "stall_turns": 12, "spec": {
            "character": "DEFECT", "ascension": 10, "seed": f"OBS29ECHO{k}", "act": k % 3, "encounter": ["regular", "elite", "boss"][k % 3], "hp": 300, "max_hp": 300,
            "deck": [{"id": "ECHO_FORM"}] * 3 + [{"id": "STRIKE_DEFECT"}] * 5 + [{"id": "DEFEND_DEFECT"}] * 4, "relics": ["ICE_CREAM"]}}))
    return G


def report(S: Stat, out: Path | None, title: str):
    rows = list(S.rows())
    print(f"## {title}")
    print("| 条目 | 28 帧数 | 28 一致率 | 28 平均误差 | 29 帧数 | 29 一致率 | 29 平均误差 |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        def cell(f):
            if f not in r: return ["—", "—", "—"]
            x = r[f]
            return [str(x["n"]), f"{100 * x['rate']:.1f}%", "—" if x["mae"] is None else f"{x['mae']:.3f}"]
        print("| " + " | ".join([r["item"]] + cell("28") + cell("29")) + " |")
    if out:
        out.write_text(json.dumps({"rows": rows, "examples": {f"{k[0]}@{k[1]}": v for k, v in S.ex.items()}}, ensure_ascii=False, indent=1, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["combat", "run", "events"])
    ap.add_argument("--dll", type=Path, default=DLL)
    ap.add_argument("--envs", type=int, default=4)
    ap.add_argument("--n", type=int, default=40, help="combat：每组场数")
    ap.add_argument("--runs", type=int, default=40)
    ap.add_argument("--ckpt", type=Path)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    S = Stat()
    rng = random.Random(a.seed)
    if a.mode == "combat":
        specs = combat_specs(a.n)
        drive(a.dll, a.envs, [r for _, r in specs], random_policy(rng), S, 600, log=sys.stderr)
    elif a.mode == "run":
        reqs = [{"op": "reset", "mode": "run", "character": "IRONCLAD", "ascension": 10, "seed": f"OBS29RUN{a.seed}_{k}", "route_summary": True, "privileged": True}
                for k in range(a.runs)]
        def partial(k):     # 整局慢：每打完 4 局把到目前为止的表写一份，中途停下也有数
            if a.out:
                with open(str(a.out) + ".partial.md", "w") as f:
                    old_out = sys.stdout; sys.stdout = f
                    try: report(S, None, f"run 前 {k} 局")
                    finally: sys.stdout = old_out
        drive(a.dll, a.envs, reqs, ckpt_policy(a.ckpt), S, 4000, log=sys.stderr, partial=partial)
    else:
        reqs = [{"op": "reset", "mode": "run", "character": ["IRONCLAD", "SILENT", "DEFECT", "REGENT", "NECROBINDER", "FOOL"][k % 6], "ascension": 10,
                 "seed": f"OBS29EV{a.seed}_{k}", "route_summary": True} for k in range(a.runs)]
        drive(a.dll, a.envs, reqs, random_policy(rng, 0.6), S, 3000, log=sys.stderr)
    report(S, a.out, f"{a.mode}（{a.dll}）")


if __name__ == "__main__":
    main()
