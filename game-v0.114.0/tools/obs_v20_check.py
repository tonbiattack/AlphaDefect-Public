"""观测格式 20（储君）的核对工具。

  defect：故障机器人的帧逐字节不变。同一组请求、同样的动作同时发给两个引擎（改动前的基线、改动后的新引擎），逐帧比较返回的字节。
          动作来自评测回放（runs/remote/replays/v{版本}_{种子}.jsonl.gz 里每步的 "a"，学生的真实对局，能打到深层），
          另加战斗课程的单场战斗（按课程抽题、随机出牌）。reset 带训练采集的全部开关（特权、路线摘要、整局预生成内容、快照）。
  state：储君整局与专门牌组，每一帧另向引擎要一次真实状态（op state，StateProbe：星、能量、手牌与费用、抽牌堆真实顺序），与观测逐项比对；
         另摘录君王之剑（锻造、寻锋、招架）与已知牌堆顶的帧，写进核对报告。
  surround：凯撒蟹的被包围（决策记录第 14 条）：改动前后的引擎跑同一批储君对局，只有带被包围记录的帧允许不同。
  sandpit：贪食者的流沙（决策记录第 16 条）：改动前后的引擎跑同一批储君对局，只有带流沙记录的帧允许不同。
  stall：跨回合无进展判输第二版（决策记录第 20 条）：改动前后的引擎跑同一批储君对局，只允许一边先判输、另一边没判这一种不同。
  aoe：储君专门牌组（打全体、打随机的攻击，寻锋），不选目标的攻击候选 4–7（伤害、击杀数）对打出去之后的真实结果。
  regent：储君整局与单场战斗，只跑新引擎：统计格式 20 新字段出现的次数（已知牌堆顶、星费、星 X 费、X 费段数、遗物本回合已触发、
          能力内部数据、星统计），已知牌堆顶与引擎内部记录不一致的次数（引擎日志里的 [drawtop]），并把有代表性的帧写成文字供逐项核对。

用法：
  python -I tools/obs_v20_check.py defect --base .cache/eng-base/Sts2Headless.dll --new .cache/eng-v20/Sts2Headless.dll --replays 40 --combats 60
  python -I tools/obs_v20_check.py regent --new .cache/eng-v20/Sts2Headless.dll --runs 20 --combats 40 --dump 报告目录
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

import numpy as np  # noqa: E402

from sts2env.env import Sts2Env  # noqa: E402
from sts2env.protocol import ActionKind, DecisionKind, TokenKind, Zone, parse  # noqa: E402

REPLAYS = Path.home() / ".local/share/sls2/runs/remote/replays"
# 评测回放录制时金币自动领（auto_gold 不带），这里照样，否则奖励界面的候选不同、动作对不上
RUN_FLAGS = {"snapshots": True, "route_summary": True, "privileged": True, "future_content": True}
# 实机桥接的帧：没有特权与整局预生成内容，路线摘要按模组配置（默认开）。--live 时按这组比，核对直播模组换新源码后故障机器人的帧不变
LIVE_FLAGS = {"route_summary": True}


class Raw:
    """引擎进程，按原始字节收发（不解析），便于逐字节比较。"""

    def __init__(self, dll: Path, log):
        self.env = Sts2Env(dll=dll, log=log)

    def req(self, r: dict) -> bytes:
        self.env._proc.send(r)
        return self.env._proc.recv()

    def close(self):
        self.env.close()


def pick_replays(n: int, min_version: int, seed: int) -> list[Path]:
    names = []
    with os.scandir(REPLAYS) as it:
        for e in it:
            if not e.name.endswith(".jsonl.gz") or not e.name.startswith("v"):
                continue
            try:
                v = int(e.name[1:].split("_", 1)[0])
            except ValueError:
                continue
            if v >= min_version:
                names.append(e.name)
    names.sort()
    random.Random(seed).shuffle(names)
    return [REPLAYS / x for x in names[:n]]


def replay_actions(p: Path) -> tuple[str, list[int]]:
    seed = p.name.split("_", 1)[1].removesuffix(".jsonl.gz")
    acts = []
    with gzip.open(p, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if "a" in r and r["a"] is not None:
                acts.append(int(r["a"]))
    return seed, acts


def compare(base: Raw, new: Raw, reset: dict, actions, max_steps: int, stats: dict, tag: str) -> None:
    """两边发同样的 reset 与动作，逐帧比较。actions(step, obs) → 候选下标。"""
    a, b = base.req(reset), new.req(reset)
    step = 0
    while True:
        stats["frames"] += 1
        if a != b:
            stats["diff"] += 1
            if len(stats["examples"]) < 10:
                oa, ob = parse(a), parse(b)
                stats["examples"].append({"tag": tag, "step": step, "len": [len(a), len(b)], "ver": [int(oa.header[0]), int(ob.header[0])],
                                          "tok": [oa.tok_ints.shape[0], ob.tok_ints.shape[0]]})
            return      # 分歧之后两边的局面不同，不再往下比
        o = parse(a)
        if o.done or o.num_actions == 0 or step >= max_steps:
            return
        k = actions(step, o)
        if k >= o.num_actions:
            stats["desync"] += 1
            k %= o.num_actions
        a, b = base.req({"op": "step", "action": k}), new.req({"op": "step", "action": k})
        step += 1


def cmd_defect(args) -> None:
    base_log = open(args.log_dir / "base.log", "w"); new_log = open(args.log_dir / "new.log", "w")
    base, new = Raw(Path(args.base), base_log), Raw(Path(args.new), new_log)
    stats = {"frames": 0, "diff": 0, "desync": 0, "games": 0, "examples": []}
    by_kind: dict[str, int] = {}
    t0 = time.time()
    for p in pick_replays(args.replays, args.min_version, args.seed):
        seed, acts = replay_actions(p)
        f0 = stats["frames"]
        compare(base, new, {"op": "reset", "mode": "run", "character": "DEFECT", "ascension": 10, "seed": seed, **(LIVE_FLAGS if args.live else RUN_FLAGS)},
                lambda i, o: acts[i] if i < len(acts) else 0, 6000, stats, p.name)
        stats["games"] += 1
        by_kind["replay"] = by_kind.get("replay", 0) + stats["frames"] - f0
        print(f"  回放 {p.name}：{stats['frames'] - f0} 帧（累计 {stats['frames']}，不同 {stats['diff']}，动作越界 {stats['desync']}）", flush=True)
    from sts2rl.catalog import Catalog
    from sts2rl.curriculum import CombatCurriculum
    cur = CombatCurriculum(Catalog(), seed=args.seed)
    rng = random.Random(args.seed)
    for i in range(args.combats):
        spec = cur.sample(rng.random(), boss_fraction=0.3, mix_fraction=0.3)
        f0 = stats["frames"]
        r = random.Random(1000 + i)
        compare(base, new, {"op": "reset", "mode": "combat", "spec": spec, **({} if args.live else {"privileged": True})},
                lambda s, o: r.randrange(o.num_actions), 600, stats, f"combat{i}")
        stats["games"] += 1
        by_kind["combat"] = by_kind.get("combat", 0) + stats["frames"] - f0
    base.close(); new.close()
    print(json.dumps({**stats, "by_kind": by_kind, "seconds": round(time.time() - t0, 1)}, ensure_ascii=False, indent=1))
    if stats["diff"]:
        sys.exit(1)


# ---------------------------------------------------------------- 储君

def regent_policy(rng: random.Random):
    """随机出牌，但偏向多打牌、少提前结束回合，能多碰到星费牌、锻造、放回牌堆顶这些情况。"""
    def act(step: int, o) -> int:
        n = o.num_actions
        if o.decision == DecisionKind.COMBAT:
            kinds = o.cand_ints[:n, 0]
            play = [k for k in range(n) if kinds[k] in (ActionKind.PLAY_CARD, ActionKind.USE_POTION)]
            if play and rng.random() < 0.9:
                return rng.choice(play)
        return rng.randrange(n)
    return act


def summarize(o, stats: dict, vocab: list[str], dump: list, tag: str, step: int) -> None:
    ti, tf = o.tok_ints, o.tok_feats
    kind, zone = ti[:, 0], ti[:, 2]
    stats["frames"] += 1
    stats["v20"] += int(o.header[0] == 20)
    card = kind == TokenKind.CARD
    top = card & (zone >= 16) & (zone <= 18)
    if top.any():
        stats["known_top_frames"] += 1
        stats["known_top_cards"] += int(top.sum())
        for j in np.flatnonzero(top):
            stats["known_top_ids"][vocab[ti[j, 1]]] = stats["known_top_ids"].get(vocab[ti[j, 1]], 0) + 1
    hand = card & (zone == Zone.HAND)
    stats["star_cost_cards"] += int((hand & (tf[:, 2] >= 0)).sum())
    stats["star_x_cards"] += int((card & (tf[:, 1] == 2)).sum())
    rel = (kind == TokenKind.RELIC) & (tf[:, 8] > 0)
    stats["relic_turn_used"] += int(rel.sum())
    pw = kind == TokenKind.POWER
    st = pw & (ti[:, 1] == vocab.index("STAT:STARS"))
    pin = pw & ~st & (np.abs(tf[:, 8:12]).sum(1) > 0)
    stats["power_internal"] += int(pin.sum())
    for j in np.flatnonzero(pin):
        stats["power_internal_ids"][vocab[ti[j, 1]]] = stats["power_internal_ids"].get(vocab[ti[j, 1]], 0) + 1
    stats["star_stat"] += int(st.sum())
    if o.in_combat and not st.any():
        stats["combat_no_star_stat"] += 1
    m = o.num_actions
    for k in range(m):
        if o.cand_ints[k, 0] == ActionKind.PLAY_CARD and o.cand_ints[k, 1] >= 0:
            cid = vocab[ti[o.cand_ints[k, 1], 1]]
            if cid in ("CARD.STARDUST", "CARD.HEAVENLY_DRILL") and o.cand_feats[k, 5] > 0:
                stats["x_hits"].append([cid, float(o.cand_feats[k, 5]), int(o.header[0])])
    interesting = top.any() or (st.any() and tf[st][0, 0] > 0) or rel.any() or (hand & (tf[:, 1] == 2)).any()
    if interesting and len(dump) < 400:
        dump.append(describe(o, vocab, tag, step))


def describe(o, vocab: list[str], tag: str, step: int) -> str:
    """一帧的文字摘要：全局（能量、星）、玩家能力、遗物、手牌（费用、星费、标记）、已知牌堆顶、星统计、候选的伤害预览。"""
    ti, tf = o.tok_ints, o.tok_feats
    lines = [f"## {tag} 第 {step} 步  决策 {o.decision.name}  格式 {int(o.header[0])}  第 {o.act + 1} 幕 第 {o.floor} 层"]
    g = np.flatnonzero(ti[:, 0] == TokenKind.GLOBAL)
    if len(g):
        f = tf[g[0]]
        lines.append(f"全局：能量 {f[6]:g}/{f[7]:g}  星 {f[8]:g}  回合 {f[9]:g}  生命 {f[4]:g}/{f[5]:g}")
    def name(j): return vocab[ti[j, 1]] if 0 <= ti[j, 1] < len(vocab) else str(ti[j, 1])
    for j in range(len(ti)):
        k, z = ti[j, 0], ti[j, 2]
        f = tf[j]
        if k == TokenKind.CARD and z == Zone.HAND:
            lines.append(f"  手牌 {name(j)} 升级{f[3]:g} 费{f[0]:g} X标记{f[1]:g} 星费{f[2]:g} 能打{f[4]:g} 伤害{f[16]:g} 格挡{f[17]:g}")
        elif k == TokenKind.CARD and 16 <= z <= 18:
            lines.append(f"  已知牌堆顶（区域 {z}）{name(j)} 升级{f[3]:g}  真实位置(特权) {o.tok_priv[j, 0]:.3f}")
        elif k == TokenKind.RELIC and z == Zone.ALLY:
            lines.append(f"  遗物 {name(j)} 计数{f[0]:g} 本场已触发{f[6]:g} 本回合已触发{f[8]:g}")
        elif k == TokenKind.POWER and name(j) == "STAT:STARS":
            lines.append(f"  星统计：本回合得 {f[0]:g} 失 {f[1]:g}，本场得 {f[8]:g} 失 {f[9]:g}")
        elif k == TokenKind.POWER and z == Zone.ALLY and ti[j, 3] >= 0 and ti[ti[j, 3], 0] == TokenKind.CREATURE:
            lines.append(f"  能力 {name(j)} 层数{f[0]:g} 界面{f[2]:g} 已触发{f[6]:g} 内部[{f[8]:g},{f[9]:g},{f[10]:g}] 剩免费{f[11]:g}")
    draw = int(((ti[:, 0] == TokenKind.CARD) & (ti[:, 2] == Zone.DRAW)).sum())
    lines.append(f"  抽牌堆汇总里 {draw} 张")
    for k in range(o.num_actions):
        if o.cand_ints[k, 0] == ActionKind.PLAY_CARD and o.cand_feats[k, 5] > 0:
            a = o.cand_ints[k, 1]
            lines.append(f"  候选 出 {name(a) if a >= 0 else '-'} → 单次 {o.cand_feats[k, 4]:g} 段数 {o.cand_feats[k, 5]:g} 总 {o.cand_feats[k, 6]:g} 能打死 {o.cand_feats[k, 7]:g}")
    p = np.flatnonzero((ti[:, 0] == TokenKind.CREATURE) & (tf[:, 5] > 0))
    if len(p):
        f = tf[p[0]]
        lines.append(f"  玩家回合结束预演：冰霜 {f[16]:g} 预计掉血 {f[17]:g} 自伤 {f[18]:g} 额外格挡 {f[19]:g}")
    return "\n".join(lines)


def cmd_regent(args) -> None:
    log_path = args.log_dir / "regent.log"
    log = open(log_path, "w")
    eng = Sts2Env(dll=Path(args.new), log=log)
    vocab = eng.vocab()
    stats = {"frames": 0, "v20": 0, "known_top_frames": 0, "known_top_cards": 0, "known_top_ids": {}, "star_cost_cards": 0,
             "star_x_cards": 0, "relic_turn_used": 0, "power_internal": 0, "power_internal_ids": {}, "star_stat": 0, "combat_no_star_stat": 0, "x_hits": [],
             "games": 0, "floors": [], "stuck": 0}
    dump: list[str] = []
    t0 = time.time()
    for i in range(args.runs):
        rng = random.Random(i)
        pol = regent_policy(rng)
        o = eng.reset_run("REGENT", 10, f"RG{i}", snapshots=True)
        step = 0
        while True:
            summarize(o, stats, vocab, dump, f"整局 RG{i}", step)
            if o.done or o.num_actions == 0 or step >= 6000:
                break
            o = eng.step(pol(step, o)); step += 1
        stats["games"] += 1; stats["floors"].append(o.floor); stats["stuck"] += int(o.stuck)
        print(f"  整局 RG{i}：{step} 步，到第 {o.floor} 层{'（卡住）' if o.stuck else ''}", flush=True)
    from sts2rl.catalog import Catalog
    from sts2rl.curriculum import CombatCurriculum, CurriculumConfig
    cur = CombatCurriculum(Catalog(), CurriculumConfig.for_character("REGENT"), seed=args.seed)
    for i in range(args.combats):
        rng = random.Random(100 + i)
        pol = regent_policy(rng)
        spec = cur.sample(1.0, boss_fraction=0.3, mix_fraction=0.5)
        o = eng.reset_combat(spec)
        step = 0
        while True:
            summarize(o, stats, vocab, dump, f"战斗 {spec['seed']}", step)
            if o.done or o.num_actions == 0 or step >= 600:
                break
            o = eng.step(pol(step, o)); step += 1
        stats["games"] += 1; stats["stuck"] += int(o.stuck)
    # 专门的牌组：塞满放回牌堆顶（闪光、光子切割、宇宙冷漠、闪耀打击）、我无敌、从抽牌堆变化（冲锋）、星 X 费与能量 X 费攻击、
    # 虚空形态、暗淡蓝点，带小储君、君王石、银河尘埃、化学物 X，逐条走一遍跟踪与新字段
    focus = (["GLIMMER", "PHOTON_CUT", "COSMIC_INDIFFERENCE", "SHINING_STRIKE", "I_AM_INVINCIBLE", "CHARGE", "STARDUST", "HEAVENLY_DRILL",
              "VOID_FORM", "PALE_BLUE_DOT", "TUTOR", "BEGONE"])
    for i in range(args.focus):
        rng = random.Random(500 + i)
        pol = regent_policy(rng)
        deck = [{"id": c, "upgrades": rng.randrange(2)} for c in focus for _ in range(2)] + [{"id": "STRIKE_REGENT", "upgrades": 0}] * 3
        spec = {"character": "REGENT", "ascension": 10, "seed": f"F{i}", "act": i % 3, "encounter": ["regular", "elite", "boss"][i % 3],
                "deck": deck, "relics": ["MINI_REGENT", "REGALITE", "GALACTIC_DUST"] + (["CHEMICAL_X"] if i % 2 else []), "potions": []}
        o = eng.reset_combat(spec)
        step = 0
        while True:
            summarize(o, stats, vocab, dump, f"专门牌组 F{i}", step)
            if o.done or o.num_actions == 0 or step >= 600:
                break
            o = eng.step(pol(step, o)); step += 1
        stats["games"] += 1; stats["stuck"] += int(o.stuck)
    eng.close(); log.close()
    mism = sum(1 for line in open(log_path, encoding="utf-8", errors="replace") if "[drawtop]" in line)
    stats["drawtop_mismatch_log_lines"] = mism
    stats["seconds"] = round(time.time() - t0, 1)
    stats["x_hits"] = stats["x_hits"][:40]
    print(json.dumps(stats, ensure_ascii=False, indent=1))
    if args.dump:
        out = Path(args.dump)
        out.mkdir(parents=True, exist_ok=True)
        (out / "储君帧摘录.md").write_text("\n\n".join(dump), encoding="utf-8")
        (out / "储君核对统计.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")


def state_of(eng) -> dict:
    eng._proc.send({"op": "state"})
    return json.loads(eng._proc.recv().decode("utf-8"))


def cross_check(o, st: dict, vocab: list[str], c: dict, ex: list, tag: str, step: int) -> None:
    """一帧观测对真实状态：星、能量、手牌（ID 与能量费）、抽牌堆张数、已知牌堆顶与真实顶部。不一致记例子。"""
    if not st.get("in_combat") or not o.in_combat:
        return
    ti, tf = o.tok_ints, o.tok_feats
    pl = st["player"]
    g = tf[np.flatnonzero(ti[:, 0] == TokenKind.GLOBAL)[0]]
    def chk(name, ok, detail):
        c[name] = c.get(name, 0) + 1
        if not ok:
            c[name + "_bad"] = c.get(name + "_bad", 0) + 1
            if len(ex) < 20:
                ex.append(f"{tag} 第 {step} 步 {name}：{detail}")
    chk("星", int(g[8]) == pl["stars"], f"观测 {g[8]} 真实 {pl['stars']}")
    chk("能量", int(g[6]) == pl["energy"], f"观测 {g[6]} 真实 {pl['energy']}")
    card = ti[:, 0] == TokenKind.CARD
    hand = [(vocab[ti[j, 1]].split(".", 1)[1], -1 if tf[j, 1] == 1 else int(tf[j, 0])) for j in np.flatnonzero(card & (ti[:, 2] == Zone.HAND))]
    real = [(x["id"], x.get("cost", -9)) for x in st["hand"]]
    chk("手牌与能量费", sorted(hand) == sorted(real), f"观测 {sorted(hand)} 真实 {sorted(real)}")
    top = np.flatnonzero(card & (ti[:, 2] >= Zone.DRAW_TOP1) & (ti[:, 2] <= Zone.DRAW_TOP3))
    n_draw = int((card & (ti[:, 2] == Zone.DRAW)).sum()) + len(top)
    chk("抽牌堆张数", n_draw == len(st["draw"]), f"观测 {n_draw} 真实 {len(st['draw'])}")
    if len(top):
        ids = [vocab[ti[j, 1]].split(".", 1)[1] for j in top]
        chk("已知牌堆顶与真实顶部", ids == [x["id"] for x in st["draw"][:len(ids)]], f"观测 {ids} 真实前 {len(ids)} 张 {[x['id'] for x in st['draw'][:len(ids)]]}")


def cmd_state(args) -> None:
    log = open(args.log_dir / "state.log", "w")
    eng = Sts2Env(dll=Path(args.new), log=log)
    vocab = eng.vocab()
    c: dict = {}; ex: list = []; blade: list[str] = []; tops: list[str] = []
    SB = vocab.index("CARD.SOVEREIGN_BLADE")
    def run_one(o, pol, tag, limit):
        step = 0
        while True:
            cross_check(o, state_of(eng), vocab, c, ex, tag, step)
            ti, tf = o.tok_ints, o.tok_feats
            hb = np.flatnonzero((ti[:, 0] == TokenKind.CARD) & (ti[:, 1] == SB) & (ti[:, 2] == Zone.HAND))
            if len(hb) and len(blade) < 12 and (step % 3 == 0):
                blade.append(describe(o, vocab, tag, step))
            if ((ti[:, 0] == TokenKind.CARD) & (ti[:, 2] >= 16)).any() and len(tops) < 8 and step % 2 == 0:
                tops.append(describe(o, vocab, tag, step))
            if o.done or o.num_actions == 0 or step >= limit:
                return o
            o = eng.step(pol(step, o)); step += 1
    for i in range(args.runs):
        o = run_one(eng.reset_run("REGENT", 10, f"RS{i}"), regent_policy(random.Random(i)), f"整局 RS{i}", 6000)
        print(f"  整局 RS{i}：到第 {o.floor} 层", flush=True)
    forge = ["REFINE_BLADE", "SUMMON_FORTH", "BULWARK", "SEEKING_EDGE", "PARRY", "CONQUEROR", "WROUGHT_IN_WAR", "BEAT_INTO_SHAPE"]
    top = ["GLIMMER", "PHOTON_CUT", "COSMIC_INDIFFERENCE", "SHINING_STRIKE", "I_AM_INVINCIBLE", "CHARGE", "STARDUST", "VOID_FORM"]
    for i in range(args.focus):
        rng = random.Random(900 + i)
        deck = [{"id": x, "upgrades": rng.randrange(2)} for x in (forge if i % 2 == 0 else top) for _ in range(2)] + [{"id": "STRIKE_REGENT", "upgrades": 0}] * 2
        spec = {"character": "REGENT", "ascension": 10, "seed": f"FS{i}", "act": i % 3, "encounter": ["regular", "elite", "boss"][i % 3],
                "deck": deck, "relics": ["MINI_REGENT", "REGALITE"], "potions": ["KINGS_COURAGE"] if i % 4 == 0 else []}
        run_one(eng.reset_combat(spec), regent_policy(rng), f"专门牌组 FS{i}（{'锻造' if i % 2 == 0 else '牌堆顶'}）", 600)
    eng.close(); log.close()
    mism = sum(1 for line in open(args.log_dir / "state.log", encoding="utf-8", errors="replace") if "[drawtop]" in line)
    out = {"检查": c, "例子": ex, "drawtop_mismatch_log_lines": mism}
    print(json.dumps(out, ensure_ascii=False, indent=1))
    if args.dump:
        d = Path(args.dump); d.mkdir(parents=True, exist_ok=True)
        (d / "逐帧对照.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        (d / "君王之剑帧摘录.md").write_text("\n\n".join(blade), encoding="utf-8")
        (d / "牌堆顶帧摘录.md").write_text("\n\n".join(tops), encoding="utf-8")


AOE = ["SEVEN_STARS", "CRASH_LANDING", "ASTRAL_PULSE", "DYING_STAR", "CRUSH_UNDER", "METEOR_SHOWER", "RADIATE", "STARDUST",
       "SEEKING_EDGE", "REFINE_BLADE", "GATHER_LIGHT", "GLOW"]


def cmd_aoe(args) -> None:
    """不选目标的攻击（打全体、只剩一个敌人时的打随机）：候选 4–7 的预测与真的打出去之后的结果比对。
    打之前记下每个活着的敌人（op state），打出这张牌，再看哪些死了、生命加格挡少了多少（同 ID 的敌人按先后对应；
    已从战斗里移除的算死了，战斗结束算全死了）。"""
    log = open(args.log_dir / "aoe.log", "w")
    eng = Sts2Env(dll=Path(args.new), log=log)
    vocab = eng.vocab()
    c = {"比对": 0, "击杀数一致": 0, "击杀数不一致": 0, "无击杀时总伤害一致": 0, "无击杀时总伤害不一致": 0}
    by_card: dict[str, list[int]] = {}
    ex: list[str] = []
    t0 = time.time()
    for i in range(args.combats):
        rng = random.Random(700 + i)
        deck = [{"id": x, "upgrades": rng.randrange(2)} for x in AOE for _ in range(2)] + [{"id": "DEFEND_REGENT", "upgrades": 0}] * 2
        spec = {"character": "REGENT", "ascension": 10, "seed": f"AO{i}", "act": i % 3, "encounter": ["regular", "elite", "boss"][i % 3],
                "deck": deck, "relics": ["CHEMICAL_X"] if i % 4 == 3 else [], "potions": []}
        o = eng.reset_combat(spec)
        base = regent_policy(rng)
        step = 0
        while not (o.done or o.num_actions == 0 or step >= 600):
            n = o.num_actions
            cands = [k for k in range(n) if o.cand_ints[k, 0] == ActionKind.PLAY_CARD and o.cand_ints[k, 2] < 0 and o.cand_feats[k, 5] > 0]
            if not (cands and o.decision == DecisionKind.COMBAT and rng.random() < 0.7):
                o = eng.step(base(step, o)); step += 1
                continue
            k = rng.choice(cands)
            cid = vocab[o.tok_ints[o.cand_ints[k, 1], 1]].split(".", 1)[1]
            cf = o.cand_feats[k].copy()
            # 候选 10（做完这一步就结束回合预计掉多少血）对结束回合候选的 10：打死了敌人时应少掉它们的攻击
            end = [j for j in range(n) if o.cand_ints[j, 0] == ActionKind.END_TURN]
            if end and cf[7] >= 1:
                e10 = float(o.cand_feats[end[0], 10])
                c["击杀时 10 比结束回合少"] = c.get("击杀时 10 比结束回合少", 0) + int(cf[10] < e10)
                c["击杀时 10 与结束回合相同"] = c.get("击杀时 10 与结束回合相同", 0) + int(cf[10] == e10)
                c["击杀时 10 比结束回合多"] = c.get("击杀时 10 比结束回合多", 0) + int(cf[10] > e10)
            pre = state_of(eng)
            before = [(e["id"], e["hp"], e["block"]) for e in pre["enemies"] if e["alive"]]
            o = eng.step(k); step += 1
            post = state_of(eng)
            pool: dict[str, list] = {}
            for e in post.get("enemies", []):
                pool.setdefault(e["id"], []).append(e)
            kills, dmg = 0, 0
            for eid, hp, bl in before:
                e = pool[eid].pop(0) if pool.get(eid) else None
                if e is None or not e["alive"] or not post.get("in_combat"):
                    kills += 1; dmg += hp + bl
                else:
                    dmg += hp + bl - e["hp"] - e["block"]
            pred_k, pred_total = int(round(float(cf[7]))), float(cf[6])
            c["比对"] += 1
            bc = by_card.setdefault(cid, [0, 0]); bc[0] += 1
            after_s = [(e["id"], e["hp"], e["block"], e["alive"]) for e in post.get("enemies", [])]
            if pred_k == kills:
                c["击杀数一致"] += 1; bc[1] += 1
            else:
                c["击杀数不一致"] += 1
                if len(ex) < 25:
                    ex.append(f"AO{i} 第 {step} 步 {cid}：预测打死 {pred_k}、实际 {kills}；单次 {cf[4]:g} 段数 {cf[5]:g} 总 {pred_total:g}；敌人 {before} → {after_s}")
            if pred_k == 0 and kills == 0:
                if abs(pred_total - dmg) < 0.5:
                    c["无击杀时总伤害一致"] += 1
                else:
                    c["无击杀时总伤害不一致"] += 1
                    if len(ex) < 25:
                        ex.append(f"AO{i} 第 {step} 步 {cid}：无击杀，预测总伤害 {pred_total:g}（单次 {cf[4]:g} × {cf[5]:g} 段）、实际生命加格挡少了 {dmg}；敌人 {before} → {after_s}")
    eng.close(); log.close()
    print(json.dumps({"统计": c, "按牌（比对次数，击杀数一致）": by_card, "例子": ex, "seconds": round(time.time() - t0, 1)}, ensure_ascii=False, indent=1))


def cmd_surround(args) -> None:
    """凯撒蟹的被包围（储君决策记录第 14 条）：改动前后的引擎跑同一批储君对局逐帧比较，只有带被包围（SURROUNDED_POWER）记录的帧允许不同。
    另摘几帧凯撒蟹的朝向、背刺与候选 10。"""
    base_log = open(args.log_dir / "sur_base.log", "w"); new_log = open(args.log_dir / "sur_new.log", "w")
    base, new = Raw(Path(args.base), base_log), Raw(Path(args.new), new_log)
    vocab = new.env.vocab()
    SUR, BL, BR = vocab.index("POWER.SURROUNDED_POWER"), vocab.index("POWER.BACK_ATTACK_LEFT_POWER"), vocab.index("POWER.BACK_ATTACK_RIGHT_POWER")
    st = {"frames": 0, "same": 0, "diff_surround": 0, "diff_other": 0, "games": 0}
    ex: list[str] = []
    def run(reset, pol, limit, tag):
        a, b = base.req(reset), new.req(reset); step = 0
        while True:
            st["frames"] += 1
            ob = parse(b)
            if a == b: st["same"] += 1
            elif (ob.tok_ints[:, 1] == SUR).any(): st["diff_surround"] += 1
            else:
                st["diff_other"] += 1; ex.append(f"{tag} 第 {step} 步：不带被包围的帧不同"); return
            if (ob.tok_ints[:, 1] == SUR).any() and len(ex) < 6 and step % 3 == 1:
                ti, tf = ob.tok_ints, ob.tok_feats
                pw = [(vocab[ti[j, 1]].split(".")[-1], float(tf[j, 8]), int(ti[j, 3])) for j in range(len(ti)) if ti[j, 1] in (SUR, BL, BR)]
                cands = [(ob.labels[k] if ob.labels else k, float(ob.cand_feats[k, 10])) for k in range(ob.num_actions)]
                ex.append(f"{tag} 第 {step} 步：{pw}；候选 10 {cands}")
            o = parse(a)
            if o.done or o.num_actions == 0 or step >= limit: return
            k = pol(step, o)
            a, b = base.req({"op": "step", "action": k}), new.req({"op": "step", "action": k}); step += 1
    for i in range(args.runs):
        run({"op": "reset", "mode": "run", "character": "REGENT", "ascension": 10, "seed": f"SR{i}", **RUN_FLAGS, "loop_cut": True, "stall_turns": 20},
            regent_policy(random.Random(i)), 3000, f"整局 SR{i}"); st["games"] += 1
    from sts2rl.catalog import Catalog
    from sts2rl.curriculum import CombatCurriculum, CurriculumConfig
    cur = CombatCurriculum(Catalog(), CurriculumConfig.for_character("REGENT"), seed=args.seed)
    for i in range(args.combats):
        spec = cur.sample(1.0, boss_fraction=0.3, mix_fraction=0.5)
        run({"op": "reset", "mode": "combat", "spec": spec, "privileged": True}, regent_policy(random.Random(300 + i)), 600, f"战斗 {i}"); st["games"] += 1
    for i in range(args.crab):
        spec = {"character": "REGENT", "ascension": 10, "seed": f"CRAB{i}", "act": 1, "encounter": "KAISER_CRAB_BOSS", "hp": 300, "max_hp": 300}
        run({"op": "reset", "mode": "combat", "spec": spec, "privileged": True, "labels": True}, regent_policy(random.Random(800 + i)), 600, f"凯撒蟹 {i}"); st["games"] += 1
    base.close(); new.close()
    print(json.dumps({**st, "例子": ex}, ensure_ascii=False, indent=1))
    if st["diff_other"]: sys.exit(1)


def cmd_sandpit(args) -> None:
    """贪食者的流沙（储君决策记录第 16 条）：改动前后的引擎跑同一批储君对局逐帧比较，只有带流沙（SANDPIT_POWER）记录的帧允许不同；
    另摘流沙计数为 1 的帧：玩家生物 17（现在结束回合预计掉血）与各候选的 10。"""
    base_log = open(args.log_dir / "pit_base.log", "w"); new_log = open(args.log_dir / "pit_new.log", "w")
    base, new = Raw(Path(args.base), base_log), Raw(Path(args.new), new_log)
    vocab = new.env.vocab()
    PIT = vocab.index("POWER.SANDPIT_POWER")
    st = {"frames": 0, "same": 0, "diff_pit": 0, "diff_other": 0, "games": 0, "pit1_frames": 0}
    ex: list[str] = []
    def run(reset, pol, limit, tag):
        a, b = base.req(reset), new.req(reset); step = 0
        while True:
            st["frames"] += 1
            ob = parse(b)
            has = (ob.tok_ints[:, 1] == PIT).any()
            if a == b: st["same"] += 1
            elif has: st["diff_pit"] += 1
            else:
                st["diff_other"] += 1; ex.append(f"{tag} 第 {step} 步：不带流沙的帧不同"); return
            if has and ob.decision == DecisionKind.COMBAT:
                ti, tf = ob.tok_ints, ob.tok_feats
                amt = [float(tf[j, 0]) for j in range(len(ti)) if ti[j, 1] == PIT]
                if amt and amt[0] <= 1:
                    st["pit1_frames"] += 1
                    if len(ex) < 8:
                        pl = [j for j in range(len(ti)) if ti[j, 0] == TokenKind.CREATURE and tf[j, 5] > 0][0]
                        cands = [(ob.labels[k] if ob.labels else k, float(ob.cand_feats[k, 10])) for k in range(ob.num_actions)]
                        ex.append(f"{tag} 第 {step} 步：流沙 {amt[0]:g}，玩家生命 {tf[pl, 0]:g}、预计掉血(17) {tf[pl, 17]:g}；候选 10 {cands}")
            o = parse(a)
            if o.done or o.num_actions == 0 or step >= limit: return
            k = pol(step, o)
            a, b = base.req({"op": "step", "action": k}), new.req({"op": "step", "action": k}); step += 1
    for i in range(args.runs):
        run({"op": "reset", "mode": "run", "character": "REGENT", "ascension": 10, "seed": f"SP{i}", **RUN_FLAGS, "loop_cut": True, "stall_turns": 8},
            regent_policy(random.Random(i)), 3000, f"整局 SP{i}"); st["games"] += 1
    from sts2rl.catalog import Catalog
    from sts2rl.curriculum import CombatCurriculum, CurriculumConfig
    cur = CombatCurriculum(Catalog(), CurriculumConfig.for_character("REGENT"), seed=args.seed + 7)
    for i in range(args.combats):
        spec = cur.sample(1.0, boss_fraction=0.3, mix_fraction=0.5)
        run({"op": "reset", "mode": "combat", "spec": spec, "privileged": True}, regent_policy(random.Random(400 + i)), 600, f"战斗 {i}"); st["games"] += 1
    for i in range(args.pit):
        # 贪食者战：随机出牌偏防守（多结束回合），好让流沙数到 1
        spec = {"character": "REGENT", "ascension": 10, "seed": f"PIT{i}", "act": 1, "encounter": "THE_INSATIABLE_BOSS", "hp": 400, "max_hp": 400}
        rng = random.Random(900 + i)
        def pol(step, o, rng=rng):
            n = o.num_actions
            ends = [k for k in range(n) if o.cand_ints[k, 0] == ActionKind.END_TURN]
            return ends[0] if ends and rng.random() < 0.5 else rng.randrange(n)
        run({"op": "reset", "mode": "combat", "spec": spec, "privileged": True, "labels": True}, pol, 600, f"贪食者 {i}"); st["games"] += 1
    base.close(); new.close()
    print(json.dumps({**st, "例子": ex}, ensure_ascii=False, indent=1))
    if st["diff_other"]: sys.exit(1)


def cmd_stall(args) -> None:
    """跨回合无进展判输第二版（储君决策记录第 20 条）：改动前后的引擎跑同一批储君对局逐帧比较。允许的不同只有一种：同一帧一边整局结束（判输）、
    另一边没结束，这局就此停止比较；分别记新引擎先判输、旧引擎先判输的局数和例子，其余任何不同都算错。
    整局分判输线 20（评测）与 8（训练）两组；随机出牌偏防守（常结束回合），另加瀑布巨兽单场（荆棘 + 只防守，会吸血回满）。"""
    base_log = open(args.log_dir / "stall_base.log", "w"); new_log = open(args.log_dir / "stall_new.log", "w")
    base, new = Raw(Path(args.base), base_log), Raw(Path(args.new), new_log)
    st = {"frames": 0, "same": 0, "games": 0, "new_first": 0, "base_first": 0, "diff_other": 0}
    ex: list[str] = []
    def run(reset, pol, limit, tag):
        a, b = base.req(reset), new.req(reset); step = 0
        while True:
            st["frames"] += 1
            if a != b:
                oa, ob = parse(a), parse(b)
                if ob.done and not oa.done:
                    st["new_first"] += 1; ex.append(f"{tag} 第 {step} 步：新引擎先判输（第 {ob.floor} 层，生命 {ob.hp}）")
                elif oa.done and not ob.done:
                    st["base_first"] += 1; ex.append(f"{tag} 第 {step} 步：旧引擎先判输（第 {oa.floor} 层，生命 {oa.hp}）")
                else:
                    st["diff_other"] += 1; ex.append(f"{tag} 第 {step} 步：帧不同，但不是一边判输")
                return
            st["same"] += 1
            o = parse(a)
            if o.done or o.num_actions == 0 or step >= limit: return
            k = pol(step, o)
            a, b = base.req({"op": "step", "action": k}), new.req({"op": "step", "action": k}); step += 1
    def guarded(rng: random.Random, p_end: float):
        play = regent_policy(rng)
        def act(step, o):
            if o.decision == DecisionKind.COMBAT:
                n = o.num_actions
                ends = [k for k in range(n) if o.cand_ints[k, 0] == ActionKind.END_TURN]
                if ends and rng.random() < p_end: return ends[0]
            return play(step, o)
        return act
    for i in range(args.runs):
        for turns in (20, 8):
            run({"op": "reset", "mode": "run", "character": "REGENT", "ascension": 10, "seed": f"ST{i}", **RUN_FLAGS, "loop_cut": True, "stall_turns": turns},
                guarded(random.Random(i * 2 + (turns == 8)), 0.35), 4000, f"整局 ST{i}（判输线 {turns}）"); st["games"] += 1
    from sts2rl.catalog import Catalog
    from sts2rl.curriculum import CombatCurriculum, CurriculumConfig
    cur = CombatCurriculum(Catalog(), CurriculumConfig.for_character("REGENT"), seed=args.seed + 11)
    for i in range(args.combats):
        spec = cur.sample(1.0, boss_fraction=0.3, mix_fraction=0.5)
        run({"op": "reset", "mode": "combat", "spec": spec, "privileged": True, "loop_cut": True, "stall_turns": 8},
            guarded(random.Random(600 + i), 0.5), 1500, f"战斗 {i}"); st["games"] += 1
    for i in range(args.giant):
        spec = {"character": "REGENT", "ascension": 10, "seed": f"WG{i}", "act": 0, "encounter": "WATERFALL_GIANT_BOSS", "hp": 300, "max_hp": 300}
        run({"op": "reset", "mode": "combat", "spec": spec, "privileged": True, "loop_cut": True, "stall_turns": 20},
            guarded(random.Random(800 + i), 0.6), 3000, f"瀑布巨兽 {i}"); st["games"] += 1
    base.close(); new.close()
    for name, log in (("旧", args.log_dir / "stall_base.log"), ("新", args.log_dir / "stall_new.log")):
        cuts = [l for l in open(log, encoding="utf-8", errors="replace") if l.startswith("[stallcut] 第")]
        st[f"{name}引擎判输次数"] = len(cuts)
    print(json.dumps({**st, "例子": ex[:40]}, ensure_ascii=False, indent=1))
    if st["diff_other"]: sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("defect")
    d.add_argument("--base", required=True); d.add_argument("--new", required=True)
    d.add_argument("--replays", type=int, default=40); d.add_argument("--combats", type=int, default=60)
    d.add_argument("--min-version", type=int, default=160000); d.add_argument("--seed", type=int, default=0)
    d.add_argument("--live", action="store_true", help="按实机桥接的开关比（不带特权、整局预生成内容）")
    r = sub.add_parser("regent")
    r.add_argument("--new", required=True); r.add_argument("--runs", type=int, default=20); r.add_argument("--combats", type=int, default=40)
    r.add_argument("--seed", type=int, default=0); r.add_argument("--dump", default="")
    r.add_argument("--focus", type=int, default=30, help="专门牌组的战斗场数")
    t = sub.add_parser("state")
    t.add_argument("--new", required=True); t.add_argument("--runs", type=int, default=20); t.add_argument("--focus", type=int, default=40)
    t.add_argument("--dump", default="")
    e = sub.add_parser("aoe")
    e.add_argument("--new", required=True); e.add_argument("--combats", type=int, default=60)
    u = sub.add_parser("surround")
    u.add_argument("--base", required=True); u.add_argument("--new", required=True)
    u.add_argument("--runs", type=int, default=12); u.add_argument("--combats", type=int, default=60); u.add_argument("--crab", type=int, default=6)
    u.add_argument("--seed", type=int, default=0)
    q = sub.add_parser("sandpit")
    q.add_argument("--base", required=True); q.add_argument("--new", required=True)
    q.add_argument("--runs", type=int, default=12); q.add_argument("--combats", type=int, default=60); q.add_argument("--pit", type=int, default=8)
    q.add_argument("--seed", type=int, default=0)
    w = sub.add_parser("stall")
    w.add_argument("--base", required=True); w.add_argument("--new", required=True)
    w.add_argument("--runs", type=int, default=20); w.add_argument("--combats", type=int, default=60); w.add_argument("--giant", type=int, default=8)
    w.add_argument("--seed", type=int, default=0)
    for p in (d, r, t, e, u, q, w):
        p.add_argument("--log-dir", type=Path, default=REPO / ".cache" / "obs_v20_check")
    args = ap.parse_args()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    {"defect": cmd_defect, "regent": cmd_regent, "state": cmd_state, "aoe": cmd_aoe, "surround": cmd_surround, "sandpit": cmd_sandpit, "stall": cmd_stall}[args.cmd](args)


if __name__ == "__main__":
    main()
