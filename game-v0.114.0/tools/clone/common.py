"""复制局面工具（tools/clone/）的公用部分：引擎进程、逐字节比帧、第一处差异的定位、对拍语料。

- Eng：一个 C# 引擎进程，按原始字节收发（不解析），引擎报错时返回 EngineError 而不是抛出，便于两边对照。
- same_frame / first_diff：逐字节比两帧，跳过头部里与进程有关的项（12 本进程赢下的战斗数、14 本进程累计的词表外键数）；
  不同时按帧布局（protocol.py）定位到段、第几条、第几项、两边的值。
- encounters / combat_spec：单场战斗语料，照 rust-engine/tools/rustdiff/frame_diff.py 的做法（种子 RD{i}、按序号轮转全部遭遇、
  random 用 random_loadout(random.Random(f"deck{i}"))），只是遭遇列表改从内容目录（engine/catalog/<版本>/catalog.json）的
  幕条目取，不再问 Rust 引擎；卡池、遗物池、药水池也取同一份内容目录。

本模块不改任何引擎行为，只供 pair_diff.py、live_fidelity.py、clone_fidelity.py 共用。
"""
from __future__ import annotations

import json
import os
import random
import struct
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "tools"))

from sts2env.env import EngineError, _Process  # noqa: E402
from sts2env.protocol import (CAND_FEATS, CAND_FEATS_V18, CAND_FEATS_V25, CAND_INTS, HEADER_INTS, HEADER_INTS_V13,  # noqa: E402
                              HEADER_INTS_V14, HEADER_INTS_V15, HEADER_INTS_V16, HEADER_INTS_V17, HEADER_INTS_V18,
                              TOK_INTS, TOK_PRIV, TokenKind, tok_feats_n)

# 主会话编好的引擎（2026-11-16）：复制版、v0.113.1 适配后无复制代码的参照、v0.113.0 原样。命令行一般都显式给 dll；
# 环境变量 STS2_CLONE_ROOT 可换目录（v0.114.0 起局面复制已在主线引擎里，不再是单独的分支）
CLONE_ROOT = Path(os.environ.get("STS2_CLONE_ROOT", Path.home() / "dev" / "sls2-clone113" / ".cache"))
DLL_CLONE = CLONE_ROOT / "eng-clone" / "Sts2Headless.dll"
DLL_REF113 = CLONE_ROOT / "ref-113" / "Sts2Headless.dll"
DLL_BASE = CLONE_ROOT / "eng-base" / "Sts2Headless.dll"
from sts2env.gamever import GAME_VERSION as CATALOG_VERSION  # noqa: E402  # 原来写死 v0.113.1：在 v0.114.0 上会从旧目录抽遭遇和牌组（含已删掉的 RAPID_SHOT）

# 训练采集的整局开关（同 tools/engine_equal.py 的 RUN_FLAGS）
RUN_FLAGS = {"snapshots": True, "route_summary": True, "privileged": True, "future_content": True}
# 战斗练习的开关（同 tools/future/boss_pimc.py 的 EXTRA、评测的 eval_stall_turns 20）
COMBAT_FLAGS = {"loop_cut": True, "stall_turns": 20}

H12 = 12        # 本进程赢下的战斗数（UiFlows.CombatsWon，跨局累计，各进程各算各的）
H14 = 14        # 本进程累计遇到的词表外键数
SKIP = (H12, H14)


def be_nice(n: int) -> None:
    """降低本进程优先级（子进程，即引擎，继承）。主会话和别的代理也在用这台机器。"""
    if n:
        try:
            os.nice(n)
        except OSError:
            pass


def dll_info(path: Path | str) -> str:
    """引擎的出处（报告用）：路径、修改时间、SHA-1 前 10 位。主会话可能重编，结果要能对上是哪一份构建。"""
    import hashlib
    import time as _t
    p = Path(path)
    try:
        return f"{p}（{_t.strftime('%m-%d %H:%M:%S', _t.localtime(p.stat().st_mtime))}，sha1 {hashlib.sha1(p.read_bytes()).hexdigest()[:10]}）"
    except OSError:
        return f"{p}（读不到）"


class Eng:
    """一个 C# 引擎进程，按原始字节收发。req() 引擎报错时返回 EngineError（进程仍可用；进程退出了则自动换一个新的）。
    env 为这个进程额外的环境变量（如反向核对的 STS2_CLONE_WEAK_LEAF=1）。"""

    def __init__(self, dll: Path | str, log=None, name: str = "", env: dict | None = None):
        self.dll = Path(dll)
        self.log = log if log is not None else open(os.devnull, "wb")
        self.name = name
        self.env = dict(env or {})
        self.p = self._spawn()
        self.n_req = 0

    def _spawn(self) -> _Process:
        old = {k: os.environ.get(k) for k in self.env}
        os.environ.update(self.env)
        try:
            return _Process(self.dll, self.log)
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def send(self, r: dict) -> None:
        self.n_req += 1
        self.p.send(r)

    def recv(self) -> bytes | EngineError:
        try:
            return self.p.recv()
        except EngineError as e:
            if self.p.p.poll() is not None:        # 进程退出：换一个新的，这次按引擎报错返回
                self.p = self._spawn()
                return EngineError(f"进程退出：{e}")
            return e

    def req(self, r: dict) -> bytes | EngineError:
        try:
            self.send(r)
        except OSError as e:
            self.p = self._spawn()
            return EngineError(f"请求发不出去：{type(e).__name__}")
        return self.recv()

    def json(self, r: dict) -> dict:
        b = self.req(r)
        if isinstance(b, EngineError):
            raise b
        return json.loads(b.decode("utf-8"))

    def close(self) -> None:
        self.p.close()


# ---------------------------------------------------------------- 帧的比较与定位

def _hdr(buf: bytes, i: int) -> int:
    return struct.unpack_from("<i", buf, 4 * i)[0] if len(buf) >= 4 * (i + 1) else 0


def same_frame(a: bytes | None, b: bytes | None, skip=SKIP) -> bool:
    """逐字节相同（跳过头部 skip 里的项，默认 12、14）。"""
    if a is None or b is None or len(a) != len(b):
        return False
    pos = 0
    for i in sorted(skip):
        s = 4 * i
        if a[pos:s] != b[pos:s]:
            return False
        pos = s + 4
    return a[pos:] == b[pos:]


def h14_differs(a: bytes, b: bytes) -> bool:
    return _hdr(a, H14) != _hdr(b, H14)


def layout(buf: bytes) -> dict:
    """按帧布局切段：[(段名, 起始字节, 条数, 每条项数, 类型)]，以及标签与附加段的起始位置。"""
    ver = _hdr(buf, 0)
    nh = HEADER_INTS if ver >= 19 else HEADER_INTS_V18 if ver == 18 else HEADER_INTS_V17 if ver == 17 else \
        HEADER_INTS_V16 if ver == 16 else HEADER_INTS_V15 if ver == 15 else HEADER_INTS_V14 if ver == 14 else HEADER_INTS_V13
    n, m = _hdr(buf, 2), _hdr(buf, 3)
    ntf = tok_feats_n(ver)
    ncf = CAND_FEATS if ver >= 26 else CAND_FEATS_V25 if ver >= 19 else CAND_FEATS_V18
    segs, off = [], 0
    for name, cnt, w, t in (("头部", 1, nh, "i"), ("记录整数", n, TOK_INTS, "i"), ("记录特征", n, ntf, "f"), ("特权特征", n, TOK_PRIV, "f"),
                            ("候选整数", m, CAND_INTS, "i"), ("候选特征", m, ncf, "f")):
        segs.append((name, off, cnt, w, t))
        off += 4 * cnt * w
    return {"ver": ver, "n": n, "m": m, "segs": segs, "labels": off}


TOK_NAMES = {int(k): k.name for k in TokenKind}


def first_diff(a: bytes, b: bytes, skip=SKIP) -> dict | None:
    """两帧第一处不同（跳过头部 skip）；相同返回 None。返回 {段, 条, 项, a, b, 记录类型, 说明}。"""
    if same_frame(a, b, skip):
        return None
    la, lb = layout(a), layout(b)
    if (la["ver"], la["n"], la["m"]) != (lb["ver"], lb["n"], lb["m"]):
        for i in (0, 2, 3):
            if _hdr(a, i) != _hdr(b, i):
                return {"seg": "头部", "row": 0, "col": i, "a": _hdr(a, i), "b": _hdr(b, i),
                        "desc": f"头部第 {i} 项（{'版本' if i == 0 else '记录数' if i == 2 else '候选数'}）：{_hdr(a, i)} / {_hdr(b, i)}"}
    for name, off, cnt, w, t in la["segs"]:
        for r in range(cnt):
            for c in range(w):
                p = off + 4 * (r * w + c)
                if name == "头部" and c in skip:
                    continue
                if p + 4 > len(a) or p + 4 > len(b):
                    return {"seg": "长度", "desc": f"帧长 {len(a)} / {len(b)}"}
                if a[p:p + 4] != b[p:p + 4]:
                    va, vb = struct.unpack_from("<" + t, a, p)[0], struct.unpack_from("<" + t, b, p)[0]
                    kind = ""
                    if name in ("记录整数", "记录特征", "特权特征"):
                        k0 = struct.unpack_from("<i", a, la["segs"][1][1] + 4 * r * TOK_INTS)[0]
                        kind = TOK_NAMES.get(k0, str(k0))
                    elif name in ("候选整数", "候选特征"):
                        kind = "动作" + str(struct.unpack_from("<i", a, la["segs"][4][1] + 4 * r * CAND_INTS)[0])
                    return {"seg": name, "row": r, "col": c, "a": va, "b": vb, "kind": kind,
                            "desc": f"{name}第 {r} 条{('（' + kind + '）') if kind else ''}第 {c} 项：{va} / {vb}"}
    p = la["labels"]
    if a[p:] != b[p:]:
        ta, tb = a[p:], b[p:]
        k = next((i for i in range(min(len(ta), len(tb))) if ta[i] != tb[i]), min(len(ta), len(tb)))
        return {"seg": "标签与附加", "row": 0, "col": k, "desc": f"标签/附加段第 {k} 字节起不同：{ta[k:k + 60]!r} / {tb[k:k + 60]!r}"}
    return {"seg": "长度", "desc": f"帧长 {len(a)} / {len(b)}"}


def diff_key(d: dict | None) -> str:
    """差异的分类名（汇总用）：段/记录类型/项。"""
    if d is None:
        return ""
    if d["seg"] in ("头部", "长度", "标签与附加"):
        return f"{d['seg']}/{d.get('col', '')}"
    return f"{d['seg']}/{d.get('kind', '')}/{d['col']}"


def frame_brief(buf: bytes) -> dict:
    """帧的概况（报告用）：决策类型、幕、层、生命、候选数。"""
    return {"ver": _hdr(buf, 0), "decision": _hdr(buf, 1), "n": _hdr(buf, 3), "act": _hdr(buf, 6), "floor": _hdr(buf, 7), "hp": _hdr(buf, 8),
            "done": _hdr(buf, 4)}


def num_actions(buf: bytes) -> int:
    return _hdr(buf, 3)


def is_done(buf: bytes) -> bool:
    return bool(_hdr(buf, 4)) or _hdr(buf, 3) == 0


# ---------------------------------------------------------------- 语料

def catalog(version: str = CATALOG_VERSION) -> list[dict]:
    return json.load(open(REPO / "engine" / "catalog" / version / "catalog.json", encoding="utf-8"))["entities"]


ACT_INDEX = {"ACT.OVERGROWTH": 0, "ACT.UNDERDOCKS": 0, "ACT.HIVE": 1, "ACT.GLORY": 2}


def encounters(version: str = CATALOG_VERSION) -> list[tuple[str, int, str]]:
    """全部遭遇 [(ID 不带前缀, 幕, 档位)]：取内容目录里四个幕条目引用的遭遇（第一幕两种地图、第二幕、第三幕），
    顺序同 Rust 引擎的 all_encounters（幕表顺序 OVERGROWTH、UNDERDOCKS、HIVE、GLORY，幕内按 ID 排序）。
    事件专用的遭遇（*_EVENT_ENCOUNTER）、测试用的 MOCK_* 不在任何幕条目里，不收。"""
    ents = catalog(version)
    enc = {e["key"]: e for e in ents if e["category"] == "ENCOUNTER"}
    out = []
    for act_key in ("ACT.OVERGROWTH", "ACT.UNDERDOCKS", "ACT.HIVE", "ACT.GLORY"):
        act = next(e for e in ents if e["key"] == act_key)
        for ref in sorted(s[4:] for s in act["signature"] if s.startswith("ref:ENCOUNTER.")):
            e = enc[ref]
            tier = "Weak" if e.get("weak") else e.get("room", "Monster")
            out.append((ref[len("ENCOUNTER."):], ACT_INDEX[act_key], tier))
    return out


# 故障机器人的初始配置（同 combat_diff.STARTER：A10 带进阶之灾）；其余角色取 sts2rl.curriculum 与 engine_equal 的初始牌组
STARTER = {"DEFECT": ["STRIKE_DEFECT"] * 4 + ["DEFEND_DEFECT"] * 4 + ["ZAP", "DUALCAST", "ASCENDERS_BANE"]}


def starter(char: str) -> list[str]:
    if char in STARTER:
        return STARTER[char]
    from sts2rl.curriculum import STARTING_DECK
    from engine_equal import EXTRA_DECKS
    deck = STARTING_DECK.get(char) or EXTRA_DECKS[char]
    return list(deck) + ["ASCENDERS_BANE"]


def load_pools(char: str = "DEFECT", version: str = CATALOG_VERSION) -> dict:
    """随机牌组的候选（同 combat_diff.load_pools，按角色）：角色卡池与无色卡池的非基础、非先古、非衍生牌，共享与角色遗物池，共享与角色药水池。"""
    ents = catalog(version)
    cards = [e["key"][5:] for e in ents if e["category"] == "CARD" and e.get("pool") in (f"{char}_CARD_POOL", "COLORLESS_CARD_POOL")
             and e.get("rarity") not in ("Basic", "Ancient", "Token")]
    relics = [e["key"][6:] for e in ents if e["category"] == "RELIC" and e.get("pool") in ("SHARED_RELIC_POOL", f"{char}_RELIC_POOL")]
    potions = [e["key"][7:] for e in ents if e["category"] == "POTION" and set(e.get("pools", [])) & {"SHARED_POTION_POOL", f"{char}_POTION_POOL"}]
    return {"cards": sorted(cards), "relics": sorted(relics), "potions": sorted(potions), "char": char}


def random_loadout(rng: random.Random, pools: dict) -> dict:
    """同 combat_diff.random_loadout：初始牌组加 5–15 张随机牌（三成升级）、0–4 个随机遗物、0–2 瓶药水。"""
    deck = [{"id": c, "upgrades": 0} for c in starter(pools.get("char", "DEFECT"))]
    for _ in range(rng.randint(5, 15)):
        deck.append({"id": rng.choice(pools["cards"]), "upgrades": 1 if rng.random() < 0.3 else 0})
    return {"deck": deck, "relics": rng.sample(pools["relics"], rng.randint(0, 4)),
            "potions": [rng.choice(pools["potions"]) for _ in range(rng.randint(0, 2))]}


def combat_spec(i: int, decks: str, char: str = "DEFECT", encs=None, pools=None) -> dict:
    """第 i 场的 CombatSpec（同 frame_diff.run_range）：种子 RD{i}，遭遇按序号轮转；random 时牌组遗物药水由 random.Random(f"deck{i}") 抽。"""
    encs = encs or encounters()
    enc, act, _tier = encs[i % len(encs)]
    spec = {"character": char, "ascension": 10, "seed": f"RD{i}", "act": act, "encounter": enc}
    if decks == "random":
        spec.update(random_loadout(random.Random(f"deck{i}"), pools or load_pools(char)))
    # starter：不给牌组，引擎用角色自己的初始配置（同 frame_diff）
    return spec


def boss_items(path: Path | str, limit: int = 0) -> list[dict]:
    """第三幕 Boss 题（tools/future/boss_pimc.py build 的产物，含连打第二个 Boss）。"""
    items = json.loads(Path(path).read_text(encoding="utf-8"))["items"]
    return items[:limit] if limit else items


# ---------------------------------------------------------------- 直接从字节读的几项（不整帧解析）

def decision(buf: bytes) -> int:
    return _hdr(buf, 1)


def in_combat(buf: bytes) -> bool:
    return bool(_hdr(buf, 11))


def cand_kinds(buf: bytes) -> list[int]:
    """每个候选的动作类型（候选整数第 0 项）。"""
    L = layout(buf)
    _, off, cnt, w, _t = L["segs"][4]
    return [struct.unpack_from("<i", buf, off + 4 * r * w)[0] for r in range(cnt)]


def draw_top_cards(buf: bytes) -> int:
    """帧里已知的抽牌堆顶部有几张（卡牌记录、区域 16–18；格式 20 起才有）。"""
    L = layout(buf)
    _, off, cnt, w, _t = L["segs"][1]
    k = 0
    for r in range(cnt):
        kind, _vid, zone = struct.unpack_from("<3i", buf, off + 4 * r * w)
        if kind == int(TokenKind.CARD) and 16 <= zone <= 18:
            k += 1
    return k


PLAY_KINDS = (2, 3)     # ActionKind.PLAY_CARD、USE_POTION


def pick_bias(rng: random.Random, buf: bytes, p_play: float = 0.9) -> int:
    """随机出牌，偏向多打牌（同 engine_equal.play_bias）：战斗决策里以 p_play 的概率在出牌与用药水里选，否则在全部候选里均匀选。"""
    n = num_actions(buf)
    if decision(buf) == 1:
        kinds = cand_kinds(buf)
        play = [k for k in range(n) if kinds[k] in PLAY_KINDS]
        if play and rng.random() < p_play:
            return rng.choice(play)
    return rng.randrange(n)


def split_extra(buf: bytes) -> tuple[bytes, dict | None]:
    """帧 → (附加段之前的部分, 附加段的 JSON 或 None)。附加段是整局 reset 带 snapshots 时 Snapshot.Extra 给的（开战信息、玩家快照、开幕快照、货架）。"""
    p = layout(buf)["labels"]
    if p + 4 > len(buf):
        return buf, None
    (ln,) = struct.unpack_from("<i", buf, p)
    q = p + 4 + ln
    if q + 4 > len(buf):
        return buf[:q], None
    (en,) = struct.unpack_from("<i", buf, q)
    try:
        return buf[:q], json.loads(buf[q + 4:q + 4 + en].decode("utf-8"))
    except ValueError:
        return buf[:q], {"_raw": buf[q:].hex()}


REPLAY_ONLY_EXTRA = ("combat", "act_start")


def extra_only(less: bytes, more: bytes, skip=SKIP) -> bool:
    """两帧只差在附加段：more 比 less 多出开战信息（combat）或开幕快照（act_start），附加段之外逐字节相同（跳过头部 skip）。"""
    bl, el = split_extra(less)
    bm, em = split_extra(more)
    if not em or not (set(REPLAY_ONLY_EXTRA) & set(em)) or not same_frame(bl, bm, skip):
        return False
    return (el or {}) == {k: v for k, v in em.items() if k not in REPLAY_ONLY_EXTRA}


def replay_extra_only(fc: bytes, fr: bytes) -> bool:
    """R 的帧（reset + replay 之后的第一帧）只比 C 多出开战信息（combat）或开幕快照（act_start），其余逐字节相同（跳过头部 12、14）。
    这是重放的已知差异，与复制无关：replay 不编码中间帧，服务端的“上一帧是否在战斗中”还停在 reset 时（不在战斗中），
    换幕时记下的开幕快照（ActStart.Pending）也还没发出去，于是重放后的第一帧被当成“刚开战”或“刚换幕”（Snapshot.Extra）；
    逐步走（以及复制，它恢复的是逐步走过来的服务端状态）不会。只在整局 reset 带 snapshots 时出现。"""
    return extra_only(fc, fr)
