"""观测格式 21（静默猎手）的核对工具（猎人观测调研.md 第三节、猎人训练交接.md 第四节第 1 步）。

  regent：储君的帧逐字节不变。同一组请求、同样的动作同时发给两个引擎（改动前的基线、改动后的新引擎），逐帧比较返回的字节。
          动作来源：储君检查点贪心打整局（本机没有储君的评测回放，用检查点现打，能打到第三幕）、随机出牌的整局、按储君课程抽题的单场战斗、
          储君专门牌组（放回牌堆顶、星 X 费、虚空形态等，同 obs_v20_check.py regent）。reset 带训练采集的开关（特权、路线摘要、整局预生成内容、快照）。
  gate：猎人的帧强制成格式 19、20（reset 带 obs_format）时，两个引擎逐字节相同：格式 21 的新内容只在格式 21 出现，不会漏到别的格式里。
  diff21：同一个新引擎、同一批猎人对局，格式 20 与 21 逐帧结构对照：去掉格式 21 多出的战斗计数子类型 2 那条记录（其后记录的下标前移）之后，
          只允许中毒记录的 12–13、出牌候选的 4–8 与 10–11 不同，其余逐项相同。
  effects：猎人专门牌组，格式 21 的新字段对真实结果：刀刃陷阱、毒性爆发、出牌附带效果（紧勒、群蛇形态、余像）的候选 4–8 对打出后的真实变化（op state）；
           中毒 12–13 对敌方回合开始后真实掉的血与死亡；战斗计数子类型 2 对谋杀、铭记死亡的卡面预览。
  故障机器人的帧不变用 tools/obs_v20_check.py defect（同一个基线与新引擎）。

用法：
  python -I tools/obs_v21_check.py regent --base .cache/eng-base-v21/Sts2Headless.dll --new .cache/eng-v21/Sts2Headless.dll
  python -I tools/obs_v21_check.py gate --base .cache/eng-base-v21/Sts2Headless.dll --new .cache/eng-v21/Sts2Headless.dll
  python -I tools/obs_v21_check.py diff21 --new .cache/eng-v21/Sts2Headless.dll
  python -I tools/obs_v21_check.py effects --new .cache/eng-v21/Sts2Headless.dll --dump .cache/obs_v21_check/effects
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

import numpy as np  # noqa: E402

from sts2env.env import Sts2Env  # noqa: E402
from sts2env.protocol import ActionKind, DecisionKind, TokenKind, Zone, parse  # noqa: E402

RUN_FLAGS = {"snapshots": True, "route_summary": True, "privileged": True, "future_content": True}
REGENT_CKPT = Path.home() / ".local/share/sls2/live/ckpt-regent/v45900-20261108-储君.pt"
MODE_RUN = 1


class Raw:
    """引擎进程，按原始字节收发（不解析），便于逐字节比较。"""

    def __init__(self, dll: Path, log):
        self.env = Sts2Env(dll=dll, log=log)

    def req(self, r: dict) -> bytes:
        self.env._proc.send(r)
        return self.env._proc.recv()

    def state(self) -> dict:
        return json.loads(self.req({"op": "state"}).decode("utf-8"))

    def close(self):
        self.env.close()


def play_bias(rng: random.Random, p_play: float = 0.9):
    """随机出牌，偏向多打牌、少提前结束回合。"""
    def act(step: int, o) -> int:
        n = o.num_actions
        if o.decision == DecisionKind.COMBAT:
            kinds = o.cand_ints[:n, 0]
            play = [k for k in range(n) if kinds[k] in (ActionKind.PLAY_CARD, ActionKind.USE_POTION)]
            if play and rng.random() < p_play:
                return rng.choice(play)
        return rng.randrange(n)
    return act


def greedy_policy(ckpt: Path):
    """检查点贪心（同 sim_corpus.py：整局模式、逐候选取最大）。"""
    import torch
    from sts2rl.features import Ragged, collate_ragged
    from sts2rl.model import ModelConfig, PolicyValueNet, load_state
    torch.set_num_threads(2)
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    net = PolicyValueNet(ModelConfig(**blob["config"]["model"])).eval()
    load_state(net, blob["state"], blob.get("content_names"))

    def act(step: int, o) -> int:
        with torch.no_grad():
            logits, _ = net(collate_ragged(Ragged.from_obs([o], np.array([MODE_RUN]))).to("cpu"))
        return int(logits[0, :o.num_actions].argmax())
    return act


def compare(base: Raw, new: Raw, reset_a: dict, reset_b: dict, actions, max_steps: int, st: dict, tag: str, same=None) -> None:
    """两边发 reset（可以不同）与同样的动作，逐帧比较。actions(step, obs) → 候选下标；same(a, b) → (相同?, 说明)，默认逐字节。"""
    a, b = base.req(reset_a), new.req(reset_b)
    step = 0
    while True:
        st["frames"] += 1
        oa = parse(a)
        st["versions"][int(oa.header[0])] += 1
        ok, why = (a == b, "") if same is None else same(a, b)
        if not ok:
            st["diff"] += 1
            if len(st["examples"]) < 12:
                ob = parse(b)
                st["examples"].append({"tag": tag, "step": step, "len": [len(a), len(b)], "ver": [int(oa.header[0]), int(ob.header[0])],
                                       "tok": [oa.tok_ints.shape[0], ob.tok_ints.shape[0]], "why": why})
            return      # 分歧之后两边的局面不同，不再往下比
        if oa.done or oa.num_actions == 0 or step >= max_steps:
            st["floors"].append(int(oa.floor))
            return
        k = actions(step, oa)
        if k >= oa.num_actions:
            st["desync"] += 1
            k %= oa.num_actions
        a, b = base.req({"op": "step", "action": k}), new.req({"op": "step", "action": k})
        step += 1


def new_stats() -> dict:
    return {"frames": 0, "diff": 0, "desync": 0, "games": 0, "versions": Counter(), "floors": [], "examples": []}


def finish(st: dict, by_kind: dict, t0: float) -> None:
    st["versions"] = dict(st["versions"])
    fl = st.pop("floors")
    out = {**st, "by_kind": by_kind, "deepest_floor": max(fl, default=0), "floors_ge_34": sum(f >= 34 for f in fl), "seconds": round(time.time() - t0, 1)}
    print(json.dumps(out, ensure_ascii=False, indent=1))
    if st["diff"]:
        sys.exit(1)


# ---------------------------------------------------------------- 储君逐字节

REGENT_FOCUS = ["GLIMMER", "PHOTON_CUT", "COSMIC_INDIFFERENCE", "SHINING_STRIKE", "I_AM_INVINCIBLE", "CHARGE", "STARDUST", "HEAVENLY_DRILL",
                "VOID_FORM", "PALE_BLUE_DOT", "TUTOR", "BEGONE", "SEEKING_EDGE", "PARRY", "REFINE_BLADE", "SEVEN_STARS"]


def cmd_regent(args) -> None:
    base, new = Raw(Path(args.base), open(args.log_dir / "rg_base.log", "w")), Raw(Path(args.new), open(args.log_dir / "rg_new.log", "w"))
    st, by_kind, t0 = new_stats(), {}, time.time()

    def run(kind, reset, pol, limit, tag):
        f0 = st["frames"]
        compare(base, new, reset, reset, pol, limit, st, tag)
        st["games"] += 1
        by_kind[kind] = by_kind.get(kind, 0) + st["frames"] - f0
        return st["frames"] - f0

    if args.greedy:
        pol = greedy_policy(Path(args.ckpt))
        for i in range(args.greedy):
            n = run("greedy", {"op": "reset", "mode": "run", "character": "REGENT", "ascension": 10, "seed": f"V21RG{i}", **RUN_FLAGS,
                               "loop_cut": True, "stall_turns": 20}, pol, 6000, f"贪心 V21RG{i}")
            print(f"  贪心整局 V21RG{i}：{n} 帧，到第 {st['floors'][-1] if st['floors'] else '?'} 层（累计 {st['frames']}，不同 {st['diff']}）", flush=True)
    for i in range(args.runs):
        run("random_run", {"op": "reset", "mode": "run", "character": "REGENT", "ascension": 10, "seed": f"V21RR{i}", **RUN_FLAGS,
                           "loop_cut": True, "stall_turns": 8}, play_bias(random.Random(i)), 4000, f"随机整局 V21RR{i}")
    from sts2rl.catalog import Catalog
    from sts2rl.curriculum import CombatCurriculum, CurriculumConfig
    cur = CombatCurriculum(Catalog(), CurriculumConfig.for_character("REGENT"), seed=args.seed)
    for i in range(args.combats):
        spec = cur.sample(1.0, boss_fraction=0.3, mix_fraction=0.5)
        run("combat", {"op": "reset", "mode": "combat", "spec": spec, "privileged": True, "loop_cut": True, "stall_turns": 8},
            play_bias(random.Random(100 + i)), 800, f"战斗 {i}")
    for i in range(args.focus):
        rng = random.Random(500 + i)
        deck = [{"id": c, "upgrades": rng.randrange(2)} for c in rng.sample(REGENT_FOCUS, 10) for _ in range(2)] + [{"id": "STRIKE_REGENT", "upgrades": 0}] * 3
        spec = {"character": "REGENT", "ascension": 10, "seed": f"V21RF{i}", "act": i % 3, "encounter": ["regular", "elite", "boss"][i % 3],
                "deck": deck, "relics": ["MINI_REGENT", "REGALITE", "GALACTIC_DUST"] + (["CHEMICAL_X"] if i % 2 else []), "potions": []}
        run("focus", {"op": "reset", "mode": "combat", "spec": spec, "privileged": True}, play_bias(rng), 800, f"专门牌组 {i}")
    base.close(); new.close()
    finish(st, by_kind, t0)


# ---------------------------------------------------------------- 猎人：门控与结构对照

SILENT_FOCUS = ["KNIFE_TRAP", "BLADE_DANCE", "CLOAK_AND_DAGGER", "UP_MY_SLEEVE", "HIDDEN_DAGGERS", "STORM_OF_STEEL", "INFINITE_BLADES",
                "ACCURACY", "PHANTOM_BLADES", "FAN_OF_KNIVES", "OUTBREAK", "ACCELERANT", "DEADLY_POISON", "POISONED_STAB", "BOUNCING_FLASK",
                "NOXIOUS_FUMES", "CORROSIVE_WAVE", "BUBBLE_BUBBLE", "SNAKEBITE", "STRANGLE", "SERPENT_FORM", "AFTERIMAGE", "BURST",
                "MEMENTO_MORI", "MURDER", "CALCULATED_GAMBLE", "SURVIVOR", "ACROBATICS", "PREPARED", "DAGGER_THROW", "TOOLS_OF_THE_TRADE",
                "MIRAGE", "ENVENOM", "SKEWER", "FINISHER", "FLECHETTES", "REFLEX", "TACTICIAN", "WRAITH_FORM", "BLUR", "EXPOSE"]
SILENT_RELICS = [["RING_OF_THE_SNAKE", "TINGSHA", "TOUGH_BANDAGES"], ["SNECKO_SKULL", "TWISTED_FUNNEL", "NINJA_SCROLL"],
                 ["HELICAL_DART", "PAPER_KRANE"], ["RING_OF_THE_SNAKE", "SNECKO_SKULL", "NINJA_SCROLL"]]


def silent_focus_spec(i: int, rng: random.Random, pool=SILENT_FOCUS, k: int = 10, hp: int | None = None) -> dict:
    deck = [{"id": c, "upgrades": rng.randrange(2)} for c in rng.sample(pool, k) for _ in range(2)] \
        + [{"id": "STRIKE_SILENT", "upgrades": 0}] * 2 + [{"id": "DEFEND_SILENT", "upgrades": 0}] * 2
    spec = {"character": "SILENT", "ascension": 10, "seed": f"V21SF{i}", "act": i % 3, "encounter": ["regular", "elite", "boss"][i % 3],
            "deck": deck, "relics": SILENT_RELICS[i % len(SILENT_RELICS)], "potions": ["POISON_POTION", "GHOST_IN_A_JAR", "CUNNING_POTION"][: 1 + i % 3]}
    if hp:
        spec["hp"] = hp; spec["max_hp"] = hp
    return spec


def silent_games(args):
    """猎人对局的来源：随机整局，按 5 种难度的专门牌组单场战斗。返回 [(种类, reset 请求, 策略, 步数上限, 标签)]。"""
    out = []
    for i in range(args.runs):
        out.append(("run", {"op": "reset", "mode": "run", "character": "SILENT", "ascension": 10, "seed": f"V21SR{i}", **RUN_FLAGS,
                            "loop_cut": True, "stall_turns": 8}, play_bias(random.Random(i), 0.85), 4000, f"整局 V21SR{i}"))
    for i in range(args.focus):
        rng = random.Random(300 + i)
        out.append(("focus", {"op": "reset", "mode": "combat", "spec": silent_focus_spec(i, rng, hp=200), "privileged": True, "loop_cut": True,
                              "stall_turns": 8}, play_bias(rng, 0.85), 800, f"专门牌组 {i}"))
    return out


def cmd_gate(args) -> None:
    base, new = Raw(Path(args.base), open(args.log_dir / "gate_base.log", "w")), Raw(Path(args.new), open(args.log_dir / "gate_new.log", "w"))
    st, by_kind, t0 = new_stats(), {}, time.time()
    for fmt in (19, 20):
        for kind, reset, pol, limit, tag in silent_games(args):
            f0 = st["frames"]
            r = {**reset, "obs_format": fmt}
            compare(base, new, r, r, pol, limit, st, f"格式 {fmt} {tag}")
            st["games"] += 1
            by_kind[f"{fmt}_{kind}"] = by_kind.get(f"{fmt}_{kind}", 0) + st["frames"] - f0
        print(f"  强制格式 {fmt}：累计 {st['frames']} 帧，不同 {st['diff']}", flush=True)
    # 另外：两个引擎不带 obs_format 跑储君与故障机器人的随机整局（按角色查表的默认值不变）
    for ch, fmt in (("DEFECT", 19), ("REGENT", 20), ("IRONCLAD", 19)):
        for i in range(args.runs):
            f0 = st["frames"]
            r = {"op": "reset", "mode": "run", "character": ch, "ascension": 10, "seed": f"V21G{ch[:2]}{i}", **RUN_FLAGS}
            compare(base, new, r, r, play_bias(random.Random(50 + i)), 4000, st, f"{ch} 默认格式 {i}")
            st["games"] += 1
            by_kind[f"default_{ch}"] = by_kind.get(f"default_{ch}", 0) + st["frames"] - f0
    base.close(); new.close()
    finish(st, by_kind, t0)


def strip21(o21):
    """格式 21 的帧去掉战斗计数子类型 2 那条记录，下标前移（所属记录、候选的 A/B）。返回（记录整数、特征、特权、候选整数、去掉了几条）。"""
    ti, tf = o21.tok_ints, o21.tok_feats
    extra = (ti[:, 0] == TokenKind.COMBAT_STAT) & (tf[:, 0] == 2)
    keep = ~extra
    idx = np.cumsum(keep) - 1
    ti2 = ti[keep].copy()
    own = ti2[:, 3]
    assert not np.isin(own, np.flatnonzero(extra)).any(), "有记录挂在子类型 2 的战斗计数上"
    m = own >= 0
    ti2[m, 3] = idx[own[m]]
    ci = o21.cand_ints.copy()
    for col in (1, 2):
        m = ci[:, col] >= 0
        ci[m, col] = idx[ci[m, col]]
    return ti2, tf[keep], o21.tok_priv[keep], ci, int(extra.sum())


def cmd_diff21(args) -> None:
    """同一个新引擎开两个进程：一个强制格式 20，一个默认（猎人为 21），同样的动作，逐帧结构对照。"""
    e20, e21 = Raw(Path(args.new), open(args.log_dir / "d20.log", "w")), Raw(Path(args.new), open(args.log_dir / "d21.log", "w"))
    vocab = e21.env.vocab()
    POISON = vocab.index("POWER.POISON_POWER")
    c = Counter(); by_card = Counter(); ex: list = []

    def same(a: bytes, b: bytes):
        o20, o21 = parse(a), parse(b)
        if int(o20.header[0]) != 20 or int(o21.header[0]) != 21:
            return False, f"版本 {int(o20.header[0])} / {int(o21.header[0])}"
        ti, tf, tp, ci, n_extra = strip21(o21)
        c["帧"] += 1
        if o21.in_combat:
            c["战斗帧"] += 1
            c["战斗帧里子类型2恰好一条"] += int(n_extra == 1)
        else:
            c["战斗外帧里子类型2为0"] += int(n_extra == 0)
        h20, h21 = o20.header.copy(), o21.header.copy()
        h21[2] -= n_extra
        h20[0] = h21[0] = 0
        if not np.array_equal(h20, h21): return False, "头部"
        if not np.array_equal(ti, o20.tok_ints): return False, "记录整数"
        if not np.array_equal(tp, o20.tok_priv): return False, "特权特征"
        if not np.array_equal(ci, o20.cand_ints): return False, "候选整数"
        allow = np.zeros_like(tf, bool)
        allow[(ti[:, 0] == TokenKind.POWER) & (ti[:, 1] == POISON), 12:14] = True
        d = (tf != o20.tok_feats) & ~(np.isnan(tf) & np.isnan(o20.tok_feats))
        if (d & ~allow).any():
            j, k = np.argwhere(d & ~allow)[0]
            return False, f"记录特征 记录 {j}（类型 {ti[j, 0]}，{vocab[ti[j, 1]]}）第 {k} 列 {o20.tok_feats[j, k]} → {tf[j, k]}"
        if d.any(): c["中毒 12–13 有值的帧"] += 1
        pois = (ti[:, 0] == TokenKind.POWER) & (ti[:, 1] == POISON)
        if (pois & (o20.tok_feats[:, 12] != 0)).any(): return False, "格式 20 的中毒 12 不为 0"
        callow = np.zeros_like(o21.cand_feats, bool)
        play = o21.cand_ints[:, 0] == ActionKind.PLAY_CARD
        callow[play, 4:9] = True; callow[play, 10:12] = True
        cd = o21.cand_feats != o20.cand_feats
        if (cd & ~callow).any():
            j, k = np.argwhere(cd & ~callow)[0]
            return False, f"候选特征 候选 {j}（动作 {o21.cand_ints[j, 0]}）第 {k} 列 {o20.cand_feats[j, k]} → {o21.cand_feats[j, k]}"
        for j in np.flatnonzero(cd.any(1)):
            a = o21.cand_ints[j, 1]
            name = vocab[o21.tok_ints[a, 1]] if a >= 0 else "?"
            by_card[name] += 1
            if len(ex) < 30:
                ex.append(f"{name}：{np.round(o20.cand_feats[j, 4:12], 2).tolist()} → {np.round(o21.cand_feats[j, 4:12], 2).tolist()}")
        if cd.any(): c["候选有不同的帧"] += 1
        return True, ""

    st, by_kind, t0 = new_stats(), {}, time.time()
    for kind, reset, pol, limit, tag in silent_games(args):
        f0 = st["frames"]
        compare(e20, e21, {**reset, "obs_format": 20}, reset, pol, limit, st, tag, same=same)
        st["games"] += 1
        by_kind[kind] = by_kind.get(kind, 0) + st["frames"] - f0
    e20.close(); e21.close()
    st["结构对照"] = dict(c); st["候选不同（按牌）"] = dict(by_card.most_common()); st["候选不同的例子（4–11 列，格式 20 → 21）"] = ex
    finish(st, by_kind, t0)


# ---------------------------------------------------------------- 猎人：新字段对真实结果

def enemy_records(o) -> list[int]:
    """敌人的生物记录，按战斗里的先后（附加编号为在 Creatures 里的位置），与 op state 的 enemies 一一对应。"""
    ti = o.tok_ints
    js = [j for j in np.flatnonzero((ti[:, 0] == TokenKind.CREATURE) & (ti[:, 2] == Zone.ENEMY)) if ti[j, 4] >= 0]
    return sorted(js, key=lambda j: ti[j, 4])


def match_enemy(pre: dict, post: dict, idx: int):
    """打出或结束回合之后，pre 里第 idx 个敌人在 post 里是哪一个：按“同 ID 的第几个”对应（召唤会把新怪插到前面，按下标对应会错人）。"""
    eid = pre["enemies"][idx]["id"]
    k = sum(1 for e in pre["enemies"][:idx] if e["id"] == eid)
    same = [e for e in post.get("enemies", []) if e["id"] == eid]
    return same[k] if k < len(same) else None


def powers_of(o, j: int, vocab) -> dict[str, np.ndarray]:
    ti = o.tok_ints
    return {vocab[ti[k, 1]]: o.tok_feats[k] for k in np.flatnonzero((ti[:, 0] == TokenKind.POWER) & (ti[:, 3] == j))}


def cmd_effects(args) -> None:
    eng = Raw(Path(args.new), open(args.log_dir / "effects.log", "w"))
    vocab = eng.env.vocab()
    c = Counter(); ex: dict[str, list] = {}

    def note(kind: str, ok: bool, msg: str):
        c[f"{kind}：比对"] += 1
        c[f"{kind}：{'一致' if ok else '不一致'}"] += 1
        if not ok and len(ex.setdefault(kind, [])) < 12:
            ex[kind].append(msg)

    pools = {
        "knife": ["KNIFE_TRAP", "BLADE_DANCE", "CLOAK_AND_DAGGER", "UP_MY_SLEEVE", "HIDDEN_DAGGERS", "ACCURACY", "PHANTOM_BLADES", "FAN_OF_KNIVES",
                  "STORM_OF_STEEL", "INFINITE_BLADES", "BURST", "AFTERIMAGE"],
        "poison": ["OUTBREAK", "ACCELERANT", "DEADLY_POISON", "POISONED_STAB", "BOUNCING_FLASK", "NOXIOUS_FUMES", "CORROSIVE_WAVE",
                   "BUBBLE_BUBBLE", "SNAKEBITE", "BURST", "MIRAGE", "ENVENOM"],
        "rider": ["STRANGLE", "SERPENT_FORM", "AFTERIMAGE", "BLADE_DANCE", "DEADLY_POISON", "BACKFLIP", "NEUTRALIZE", "SURVIVOR", "ACROBATICS",
                  "DAGGER_SPRAY", "PREPARED", "BURST"],
        "stats": ["MEMENTO_MORI", "MURDER", "CALCULATED_GAMBLE", "SURVIVOR", "ACROBATICS", "PREPARED", "DAGGER_THROW", "TOOLS_OF_THE_TRADE",
                  "ACROBATICS", "BACKFLIP", "STORM_OF_STEEL", "EXPERTISE"],
    }
    focus_cards = {"KNIFE_TRAP", "OUTBREAK", "STRANGLE", "SERPENT_FORM", "AFTERIMAGE", "MEMENTO_MORI", "MURDER"}
    t0 = time.time()
    games = 0
    for i in range(args.combats):
        rng = random.Random(1000 + i)
        group = list(pools)[i % len(pools)]
        pool = [x for x in pools[group] if x in {v.split(".", 1)[1] for v in vocab if v.startswith("CARD.")}]
        spec = silent_focus_spec(i, rng, pool=sorted(set(pool)), k=min(9, len(set(pool))), hp=300)
        spec["seed"] = f"V21E{i}"
        # 遗物按场次除 4 轮换（与按模 4 选的牌组分开，免得某组牌总配同一组遗物）
        spec["relics"] = [["SNECKO_SKULL", "NINJA_SCROLL"], ["RING_OF_THE_SNAKE"], ["TINGSHA", "TOUGH_BANDAGES"], []][(i // 4) % 4]
        o = parse(eng.req({"op": "reset", "mode": "combat", "spec": spec, "labels": True, "loop_cut": True, "stall_turns": 8}))
        games += 1
        step = 0
        while not (o.done or o.num_actions == 0 or step >= 900):
            n = o.num_actions
            ti, tf = o.tok_ints, o.tok_feats
            # 战斗计数子类型 2 对谋杀、铭记死亡的手牌预览（目标为空的预览：基础 + 每次加多少 × 次数，再加力量；有虚弱等别的修正时不比）
            if o.in_combat:
                stat = np.flatnonzero((ti[:, 0] == TokenKind.COMBAT_STAT) & (tf[:, 0] == 2))
                pl = [j for j in np.flatnonzero((ti[:, 0] == TokenKind.CREATURE) & (tf[:, 5] > 0))]
                c["战斗帧"] += 1
                c["战斗帧有子类型 2"] += int(len(stat) == 1)
                if len(stat) == 1 and pl:
                    pw = powers_of(o, pl[0], vocab)
                    plain = set(pw) <= {"STAT:STARS", "POWER.STRENGTH_POWER", "POWER.DEXTERITY_POWER", "POWER.ACCURACY_POWER", "POWER.PHANTOM_BLADES_POWER",
                                        "POWER.NOXIOUS_FUMES_POWER", "POWER.ACCELERANT_POWER", "POWER.AFTERIMAGE_POWER", "POWER.SERPENT_FORM_POWER",
                                        "POWER.INFINITE_BLADES_POWER", "POWER.TOOLS_OF_THE_TRADE_POWER", "POWER.ENVENOM_POWER", "POWER.FAN_OF_KNIVES_POWER"}
                    strength = float(pw.get("POWER.STRENGTH_POWER", np.zeros(20))[0])
                    for j in np.flatnonzero((ti[:, 0] == TokenKind.CARD) & (ti[:, 2] == Zone.HAND)):
                        name = vocab[ti[j, 1]]
                        if name not in ("CARD.MURDER", "CARD.MEMENTO_MORI") or not plain or ti[j, 4] != 0:
                            continue
                        up = int(tf[j, 3])
                        if name == "CARD.MURDER":
                            want = 1 + 1 * tf[stat[0], 2] + strength
                            note("谋杀：手牌预览 = 1 + 本场抽牌数 + 力量", abs(tf[j, 16] - want) < 0.5, f"V21E{i} 第 {step} 步：预览 {tf[j, 16]:g}，按计数 {want:g}")
                        else:
                            want = (11 if up else 9) + (5 if up else 4) * tf[stat[0], 1] + strength
                            note("铭记死亡：手牌预览 = 基础 + 每张 × 本回合弃牌数 + 力量", abs(tf[j, 16] - want) < 0.5,
                                 f"V21E{i} 第 {step} 步：预览 {tf[j, 16]:g}，按计数 {want:g}（升级 {up}）")
            if o.decision != DecisionKind.COMBAT:
                o = parse(eng.req({"op": "step", "action": rng.randrange(n)})); step += 1
                continue
            kinds = o.cand_ints[:n, 0]
            plays = [k for k in range(n) if kinds[k] == ActionKind.PLAY_CARD]
            focus = [k for k in plays if vocab[ti[o.cand_ints[k, 1], 1]].split(".", 1)[1] in focus_cards]
            end = [k for k in range(n) if kinds[k] == ActionKind.END_TURN]
            pre = eng.state()
            enemies = enemy_records(o)
            if focus and rng.random() < 0.6: k = rng.choice(focus)
            elif plays and rng.random() < 0.8: k = rng.choice(plays)
            else: k = rng.randrange(n)
            cf = o.cand_feats[k].copy()
            # ---- 出牌：候选 4–8 对打出后的真实变化
            if kinds[k] == ActionKind.PLAY_CARD:
                card = vocab[ti[o.cand_ints[k, 1], 1]].split(".", 1)[1]
                tgt_rec = int(o.cand_ints[k, 2])
                tgt = enemies.index(tgt_rec) if tgt_rec in enemies else None
                ppw = {p[0]: p[1] for p in pre["player"]["powers"]}
                strangled = [e for e in pre["enemies"] if e["alive"] and any(p[0] == "STRANGLE_POWER" for p in e["powers"])]
                serpent, image = ppw.get("SERPENT_FORM_POWER", 0), ppw.get("AFTERIMAGE_POWER", 0)
                hittable = [e for e in pre["enemies"] if e["alive"]]
                post_o = parse(eng.req({"op": "step", "action": k})); step += 1
                # 打出途中要选牌（生存者、杂技、早有准备等从手牌选牌弃掉）：这时附带效果还没结算（AfterCardPlayed 在选完之后），随机选完再量
                picked = 0
                while post_o.decision not in (DecisionKind.COMBAT, DecisionKind.GAME_OVER) and not post_o.done and post_o.num_actions > 0 and picked < 30:
                    post_o = parse(eng.req({"op": "step", "action": rng.randrange(post_o.num_actions)})); step += 1; picked += 1
                post = eng.state()
                alive_pre = [(e["id"], e["hp"], e["block"]) if e["alive"] else None for e in pre["enemies"]]
                over = not post.get("in_combat") or post.get("over")
                if over and not post.get("won"):       # 玩家死了（荆棘、自伤等），不比
                    o = post_o
                    continue
                dmg, dead = [], []
                for idx, a in enumerate(alive_pre):
                    if a is None:
                        dmg.append(0); dead.append(False); continue
                    e = match_enemy(pre, post, idx)
                    # 实验体这类多条命的怪：打死一条命换下一形态，最大生命变了，算打死
                    if over or e is None or not e["alive"] or e["max_hp"] != pre["enemies"][idx]["max_hp"]:
                        dmg.append(a[1] + a[2]); dead.append(True)
                    else:
                        dmg.append(a[1] + a[2] - e["hp"] - e["block"]); dead.append(False)
                serpent_random = serpent > 0 and len(hittable) > 1
                fan = "FAN_OF_KNIVES_POWER" in ppw
                card_rec = int(o.cand_ints[k, 1])
                own = card == "KNIFE_TRAP" or tf[card_rec, 5] > 0       # 这张牌自己有伤害（卡牌特征 5：伤害数值）
                kind = None
                if card == "KNIFE_TRAP": kind = "刀刃陷阱" + ("（刀扇）" if fan else "")
                elif card == "OUTBREAK": kind = "毒性爆发"
                elif strangled or serpent:
                    kind = "附带伤害（" + "、".join(x for x, y in (("紧勒", strangled), ("群蛇单敌", serpent and not serpent_random), ("群蛇多敌", serpent_random)) if y) + "）"
                # 打随机、多个敌人的攻击：SilentPreview 不算它自己的伤害，真实结果里却有，不比
                if kind and own and tgt is None and card not in ("OUTBREAK",) and len(hittable) > 1 and cf[5] == 0:
                    kind = None
                if kind and picked:
                    kind += "（经过选牌）"
                if kind and "TINGSHA" in pre["player"]["relics"]:
                    kind += "（有铜钹：弃牌伤害不算）"
                if kind:
                    kills = sum(dead)
                    if tgt is not None and own and not (card == "KNIFE_TRAP" and fan):
                        # 有目标、自己有伤害：7 能不能打死目标，6 对目标的总伤害（不扣格挡；打死时只比击杀）
                        note(f"{kind}：能否打死目标", int(round(cf[7])) == int(dead[tgt]),
                             f"V21E{i} 第 {step} 步 {card}：预测 {cf[7]:g}，实际 {'死' if dead[tgt] else '活'}；4–7 {np.round(cf[4:8], 1).tolist()}；目标 {alive_pre[tgt]}")
                        if not dead[tgt] and not serpent_random:
                            note(f"{kind}：目标总伤害（没打死时，生命加格挡的减少）", abs(cf[6] - dmg[tgt]) < 0.5,
                                 f"V21E{i} 第 {step} 步 {card}：预测 {cf[6]:g}，实际 {dmg[tgt]}；4–7 {np.round(cf[4:8], 1).tolist()}；目标 {alive_pre[tgt]}")
                    else:
                        # 打全体的口径：7 打得死几个，6 各敌人总伤害之和
                        note(f"{kind}：打死几个", int(round(cf[7])) == kills,
                             f"V21E{i} 第 {step} 步 {card}：预测 {cf[7]:g}，实际 {kills}；4–7 {np.round(cf[4:8], 1).tolist()}；敌人 {alive_pre} 掉 {dmg}")
                        if kills == 0 and not serpent_random:
                            note(f"{kind}：总伤害（没打死时）", abs(cf[6] - sum(dmg)) < 0.5,
                                 f"V21E{i} 第 {step} 步 {card}：预测 {cf[6]:g}，实际 {sum(dmg)}；4–7 {np.round(cf[4:8], 1).tolist()}；敌人 {alive_pre} 掉 {dmg}")
                if image and not over:
                    gain = post["player"]["block"] - pre["player"]["block"]
                    # 只比打出后格挡不会被别的东西改的情况：这张牌自己的格挡预览在 8 里，其余（结实绷带等）不算
                    if "TOUGH_BANDAGES" not in pre["player"]["relics"]:
                        note("余像：候选 8 = 打出后格挡的增加", abs(cf[8] - gain) < 0.5, f"V21E{i} 第 {step} 步 {card}：预测 {cf[8]:g}，实际 {gain}（余像 {image}）")
                o = post_o
                continue
            # ---- 结束回合：中毒 12–13 对敌方回合开始后的真实结果
            if kinds[k] == ActionKind.END_TURN:
                pred = []
                for idx, j in enumerate(enemies):
                    pw = powers_of(o, j, vocab)
                    p = pw.get("POWER.POISON_POWER")
                    if p is not None and idx < len(pre["enemies"]) and pre["enemies"][idx]["alive"]:
                        pred.append((idx, float(p[0]), float(p[12]), int(p[13]), pre["enemies"][idx]))
                post_o = parse(eng.req({"op": "step", "action": k})); step += 1
                post = eng.state()
                for idx, amt, dmg_p, lethal, e0 in pred:
                    over = not post.get("in_combat") or post.get("over")
                    if over and not post.get("won"):       # 玩家在敌方回合被打死，整场结束，不比
                        c["中毒：玩家在敌方回合死了，不比"] += 1
                        continue
                    e1 = None if over else match_enemy(pre, post, idx)
                    died = over or e1 is None or not e1["alive"] or e1["max_hp"] != e0["max_hp"]
                    if died and not lethal:
                        c["中毒：没标毒得死、但敌方回合后死了（别的伤害）"] += 1
                        if len(ex.setdefault("中毒：没标毒得死、但敌方回合后死了（看是不是别的伤害）", [])) < 12:
                            ex["中毒：没标毒得死、但敌方回合后死了（看是不是别的伤害）"].append(
                                f"V21E{i} 第 {step} 步 {e0['id']}（第 {idx} 个敌人）：毒 {amt:g} 预测 {dmg_p:g}，生命 {e0['hp']}，能力 {[q[0] for q in e0['powers']]}；"
                                f"玩家能力 {[q[0] for q in pre['player']['powers']]}；后 {[(e['id'], e['hp'], e['alive']) for e in post.get('enemies', [])]}，战斗中 {post.get('in_combat')}")
                    note("中毒 13：标了毒得死的都死了", (not lethal) or died,
                         f"V21E{i} 第 {step} 步 {e0['id']}：毒 {amt:g} 预测 {dmg_p:g}/{'死' if lethal else '活'}，生命 {e0['hp']} → {'死' if died else e1['hp']}")
                    if lethal:
                        c["中毒 13 为 1"] += 1
                    if not died:
                        others = [p[0] for p in e0["powers"] if p[0] not in ("POISON_POWER", "WEAK_POWER", "VULNERABLE_POWER", "STRENGTH_POWER")]
                        lost = e0["hp"] - e1["hp"]
                        note("中毒 12：活下来的敌人这轮掉的血 = 预测毒伤", abs(lost - dmg_p) < 0.5,
                             f"V21E{i} 第 {step} 步 {e0['id']}（第 {idx} 个敌人）：毒 {amt:g}，预测 {dmg_p:g}，实际掉 {lost}（生命 {e0['hp']} → {e1['hp']}，其他能力 {others}）；"
                             f"前 {[(e['id'], e['hp'], e['alive']) for e in pre['enemies']]} 后 {[(e['id'], e['hp'], e['alive'], [q[0] for q in e['powers']]) for e in post['enemies']]}")
                o = post_o
                continue
            o = parse(eng.req({"op": "step", "action": k})); step += 1
    eng.close()
    out = {"统计": dict(sorted(c.items())), "不一致的例子": ex, "场数": games, "seconds": round(time.time() - t0, 1)}
    print(json.dumps(out, ensure_ascii=False, indent=1))
    if args.dump:
        d = Path(args.dump); d.mkdir(parents=True, exist_ok=True)
        (d / "新字段对真实结果.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("regent")
    r.add_argument("--base", required=True); r.add_argument("--new", required=True)
    r.add_argument("--ckpt", default=str(REGENT_CKPT)); r.add_argument("--greedy", type=int, default=12)
    r.add_argument("--runs", type=int, default=12); r.add_argument("--combats", type=int, default=60); r.add_argument("--focus", type=int, default=30)
    r.add_argument("--seed", type=int, default=0)
    g = sub.add_parser("gate")
    g.add_argument("--base", required=True); g.add_argument("--new", required=True)
    g.add_argument("--runs", type=int, default=8); g.add_argument("--focus", type=int, default=40)
    d = sub.add_parser("diff21")
    d.add_argument("--new", required=True); d.add_argument("--runs", type=int, default=8); d.add_argument("--focus", type=int, default=60)
    e = sub.add_parser("effects")
    e.add_argument("--new", required=True); e.add_argument("--combats", type=int, default=120); e.add_argument("--dump", default="")
    for p in (r, g, d, e):
        p.add_argument("--log-dir", type=Path, default=REPO / ".cache" / "obs_v21_check")
    args = ap.parse_args()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    {"regent": cmd_regent, "gate": cmd_gate, "diff21": cmd_diff21, "effects": cmd_effects}[args.cmd](args)


if __name__ == "__main__":
    main()
