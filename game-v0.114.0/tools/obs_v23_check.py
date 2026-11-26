"""观测格式 23（静默猎手第二版）的核对：候选 10 的新算法对真实结果（猎人观测调研.md 第七节第一档）。

两个新引擎进程同步走同一局：一个 reset 带 obs_format 23，另一个带 obs_format 21（对照）。观测格式不影响游戏，两边的局面逐步相同，
所以同一个局面上能比“格式 23 的预测”与“格式 21 的预测”哪个更接近真实结果。真实结果来自 op state。

  出牌、用药水：候选 10 的自洽——打出前预测的“做完这一步就结束回合掉多少血” = 这一步实际掉的血 + 打出后结束回合候选的 10。
        打出后虚弱、力量、毒、激怒、感染、凋萎都已经真实发生，结束回合的预演按真实状态算，所以这条能一起检验格式 23 的几项：
        给敌人上虚弱与减力量（中和、扫腿、烟雾、萎靡、钩拳、尖啸、虚弱药水、束缚药水），实验体的激怒、感染棱柱的感染，
        加毒、打伤之后毒得死（致命毒药、蛇咬、带毒刺击、咕嘟冒泡、烟雾、弹跳药瓶、触媒、毒性爆发、毒药水），凋萎、荆棘。
        按牌与场面分类统计，格式 23 与 21 各算一遍；能抽牌的牌单列（抽到的牌可能带回合结束效果，两个格式都不算）。
  结束回合：结束回合候选的 10（玩家预计掉的血）对敌方回合里实际掉的血，按“有荆棘 / 中毒的敌人带滑溜或缓冲 / 都没有”分开，格式 23 与 21 各算一遍。
  另外统计每个战斗帧里格式 23 与 21 的候选 10 不同的候选数（按类别）。

  python -I tools/obs_v23_check.py effects --new .cache/eng-v23/Sts2Headless.dll --combats 200 --dump 报告目录
  python -I tools/obs_v23_check.py replays --new .cache/eng-v23/Sts2Headless.dll --dir 回放目录 --n 60   # 训练好的模型真实打的局上，候选 10 改了多少
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
sys.path.insert(0, str(REPO / "tools"))

import numpy as np  # noqa: E402

from sts2env.protocol import ActionKind, DecisionKind, TokenKind, parse  # noqa: E402
from obs_v21_check import Raw, enemy_records  # noqa: E402

DEBUFF = {"NEUTRALIZE", "LEG_SWEEP", "HAZE", "MALAISE", "SUCKER_PUNCH", "PIERCING_WAIL"}
POISON = {"DEADLY_POISON", "SNAKEBITE", "POISONED_STAB", "BUBBLE_BUBBLE", "BOUNCING_FLASK", "ACCELERANT", "OUTBREAK", "HAZE"}
DRAW = {"BACKFLIP", "ESCAPE_PLAN", "ACROBATICS", "ADRENALINE", "PREPARED", "CALCULATED_GAMBLE", "EXPERTISE", "DAGGER_THROW", "TOOLS_OF_THE_TRADE",
        "BURST", "SURVIVOR", "PREDATOR", "HIDDEN_DAGGERS"}
POTIONS = {"WEAK_POTION": "虚弱药水", "POISON_POTION": "毒药水", "POTION_OF_BINDING": "束缚药水"}
GROUPS = {
    "减攻击": ["NEUTRALIZE", "LEG_SWEEP", "HAZE", "MALAISE", "SUCKER_PUNCH", "PIERCING_WAIL", "DEFLECT", "BLUR", "FOOTWORK", "DODGE_AND_ROLL"],
    "加毒": ["DEADLY_POISON", "SNAKEBITE", "POISONED_STAB", "BUBBLE_BUBBLE", "BOUNCING_FLASK", "ACCELERANT", "OUTBREAK", "HAZE", "NOXIOUS_FUMES", "BURST"],
    "常用": ["DEFLECT", "SNAKEBITE", "DODGE_AND_ROLL", "FOOTWORK", "DEADLY_POISON", "NOXIOUS_FUMES", "BUBBLE_BUBBLE", "LEG_SWEEP",
             "ACCELERANT", "BLUR", "HAZE", "AFTERIMAGE", "MALAISE", "ABRASIVE", "POISONED_STAB", "MIRAGE"],
    "荆棘": ["ABRASIVE", "DEFLECT", "BLUR", "FOOTWORK", "NEUTRALIZE", "DEADLY_POISON", "SNAKEBITE", "LEG_SWEEP", "PIERCING_WAIL"],
}
ENCOUNTERS = [(2, "TEST_SUBJECT_BOSS"), (2, "AEONGLASS_BOSS"), (2, "QUEEN_BOSS"), (1, "INFESTED_PRISMS_ELITE"),
              (0, "elite"), (1, "elite"), (2, "elite"), (1, "regular"), (2, "regular"), (0, "boss"), (1, "boss")]
RELICS = [["RING_OF_THE_SNAKE"], ["PAPER_KRANE"], ["SNECKO_SKULL"], ["BRONZE_SCALES"], ["PAPER_KRANE", "SNECKO_SKULL", "RING_OF_THE_SNAKE"]]


def spec_for(i: int, group: str, pool: list[str], rng: random.Random) -> dict:
    act, enc = ENCOUNTERS[i % len(ENCOUNTERS)]
    k = min(8, len(pool))
    deck = [{"id": c, "upgrades": rng.randrange(2)} for c in rng.sample(pool, k) for _ in range(2)]
    deck += [{"id": "STRIKE_SILENT", "upgrades": 0}] * 2 + [{"id": "DEFEND_SILENT", "upgrades": 0}] * 2 + [{"id": "NEUTRALIZE", "upgrades": 0}]
    relics = RELICS[(i // len(ENCOUNTERS)) % len(RELICS)]
    if group == "荆棘" and "BRONZE_SCALES" not in relics: relics = relics + ["BRONZE_SCALES"]
    pots = ["WEAK_POTION", "POISON_POTION", "POTION_OF_BINDING"]
    return {"character": "SILENT", "ascension": 10, "seed": f"V23{group}{i}", "act": act, "encounter": enc, "deck": deck,
            "relics": relics, "potions": pots[: 1 + i % 3], "hp": 250, "max_hp": 250}


def end_turn_c10(o) -> float | None:
    n = o.num_actions
    ks = [k for k in range(n) if o.cand_ints[k, 0] == ActionKind.END_TURN]
    return float(o.cand_feats[ks[0], 10]) if ks else None


def foe_powers(st: dict) -> set[str]:
    return {p[0] for e in st.get("enemies", []) if e.get("alive") for p in e.get("powers", [])}


def cmd_effects(args) -> None:
    a = Raw(Path(args.new), open(args.log_dir / "v23.log", "w"))
    b = Raw(Path(args.new), open(args.log_dir / "v21.log", "w"))
    vocab = a.env.vocab()
    cards = {v.split(".", 1)[1] for v in vocab if v.startswith("CARD.")}
    c = Counter(); ex: dict[str, list] = {}

    def note(kind: str, ok: bool, msg: str = ""):
        c[f"{kind}：比对"] += 1
        c[f"{kind}：{'一致' if ok else '不一致'}"] += 1
        if not ok and msg and len(ex.setdefault(kind, [])) < args.examples:
            ex[kind].append(msg)

    groups = list(GROUPS.items())
    t0 = time.time()
    for i in range(args.combats):
        group, pool = groups[i % len(groups)]
        pool = sorted({x for x in pool if x in cards})
        rng = random.Random(9000 + i)
        spec = spec_for(i, group, pool, rng)
        req = {"op": "reset", "mode": "combat", "spec": spec, "labels": True, "loop_cut": True, "stall_turns": 8}
        o = parse(a.req({**req, "obs_format": 23})); o21 = parse(b.req({**req, "obs_format": 21}))
        tag = f"V23E{i}（{group} {spec['encounter']}）"
        step = 0
        while not (o.done or o.num_actions == 0 or step >= 700):
            n = o.num_actions
            if o.decision != DecisionKind.COMBAT:
                k = rng.randrange(n)
                o = parse(a.req({"op": "step", "action": k})); o21 = parse(b.req({"op": "step", "action": k})); step += 1
                continue
            c["战斗帧"] += 1
            if int(o.header[0]) != 23: c["不是格式 23 的帧"] += 1
            if int(o21.header[0]) != 21: c["对照不是格式 21 的帧"] += 1
            if o.tok_ints.shape != o21.tok_ints.shape or not np.array_equal(o.tok_ints, o21.tok_ints): c["记录整数与格式 21 不同的帧（应为 0）"] += 1
            if o.cand_ints.shape != o21.cand_ints.shape or not np.array_equal(o.cand_ints, o21.cand_ints): c["候选整数与格式 21 不同的帧（应为 0）"] += 1
            kinds = o.cand_ints[:n, 0]
            ti = o.tok_ints
            pre = a.state()
            fp = foe_powers(pre)
            ppw = {p[0]: p[1] for p in pre["player"]["powers"]}
            # 每个候选格式 23 与 21 的候选 10 差多少（按类别）
            for k in range(n):
                d = float(o.cand_feats[k, 10] - o21.cand_feats[k, 10])
                if kinds[k] == ActionKind.PLAY_CARD:
                    nm = vocab[ti[o.cand_ints[k, 1], 1]].split(".", 1)[1]
                    cat = "减攻击牌" if nm in DEBUFF else "加毒牌" if nm in POISON else "其他牌"
                elif kinds[k] == ActionKind.USE_POTION:
                    cat = "药水"
                else:
                    cat = "结束回合等"
                c[f"候选：{cat}"] += 1
                if abs(d) > 0.5: c[f"候选：{cat} 候选 10 与格式 21 不同"] += 1
                if d > 0.5: c[f"候选：{cat} 格式 23 预计掉得更多"] += 1
            # 其余特征（候选 0–9、记录特征）应与格式 21 相同，只有候选 10–11 与回合结束预演相关的记录可以不同
            cf_other = np.delete(o.cand_feats[:n], [10, 11], axis=1); cf21_other = np.delete(o21.cand_feats[:n], [10, 11], axis=1)
            if not np.allclose(cf_other, cf21_other, equal_nan=True): c["候选 0–9 与格式 21 不同的帧（应为 0）"] += 1
            plays = [k for k in range(n) if kinds[k] in (ActionKind.PLAY_CARD, ActionKind.USE_POTION)]
            ends = [k for k in range(n) if kinds[k] == ActionKind.END_TURN]
            k = rng.choice(plays) if plays and rng.random() < 0.78 else (ends[0] if ends and rng.random() < 0.8 else rng.randrange(n))
            cf, cf21 = o.cand_feats[k].copy(), o21.cand_feats[k].copy()
            if kinds[k] in (ActionKind.PLAY_CARD, ActionKind.USE_POTION):
                is_card = kinds[k] == ActionKind.PLAY_CARD
                name = vocab[ti[o.cand_ints[k, 1], 1]].split(".", 1)[1]
                post_o = parse(a.req({"op": "step", "action": k})); post21 = parse(b.req({"op": "step", "action": k})); step += 1
                picked = 0
                while post_o.decision not in (DecisionKind.COMBAT, DecisionKind.GAME_OVER) and not post_o.done and post_o.num_actions > 0 and picked < 30:
                    kk = rng.randrange(post_o.num_actions)
                    post_o = parse(a.req({"op": "step", "action": kk})); post21 = parse(b.req({"op": "step", "action": kk})); step += 1; picked += 1
                post = a.state()
                over = not post.get("in_combat") or post.get("over")
                if not over and not picked and post_o.decision == DecisionKind.COMBAT:
                    e23, e21 = end_turn_c10(post_o), end_turn_c10(post21)
                    if e23 is not None and e21 is not None:
                        lost = pre["player"]["hp"] - post["player"]["hp"]
                        if is_card and name in DRAW: cat = "能抽牌的牌（不算）"
                        elif not is_card: cat = POTIONS.get(name, "其他药水")
                        elif name in DEBUFF and name in POISON: cat = "烟雾（减攻击 + 加毒）"
                        elif name in DEBUFF: cat = "减攻击牌"
                        elif name in POISON: cat = "加毒牌"
                        else: cat = "其他牌"
                        scene = []
                        if "ENRAGE_POWER" in fp: scene.append("激怒")
                        if "VITAL_SPARK_POWER" in fp: scene.append("感染")
                        if "WITHERING_PRESENCE_POWER" in fp or "WITHERING_PRESENCE_POWER" in ppw: scene.append("凋萎")
                        if "THORNS_POWER" in ppw: scene.append("荆棘")
                        if "ARTIFACT_POWER" in fp: scene.append("人工制品")
                        if "POISON_POWER" in fp: scene.append("敌人中毒")
                        lab = f"候选 10 自洽（{cat}" + ("；" + "、".join(scene) if scene else "") + "）"
                        msg = f"{tag} 第 {step} 步 {name}：格式 23 预测 {cf[10]:g}，实际掉 {lost} + 打出后预计 {e23:g}（21：{cf21[10]:g} / {e21:g}）；敌人 {[(e['id'], e['hp'], [p[0] for p in e['powers']]) for e in pre['enemies'] if e['alive']]}"
                        note(lab + " — 格式 23", abs(cf[10] - (lost + e23)) < 0.5, msg)
                        note(lab + " — 格式 21", abs(cf21[10] - (lost + e21)) < 0.5)
                        # 汇总：只按牌的类别
                        note(f"候选 10 自洽汇总（{cat}） — 格式 23", abs(cf[10] - (lost + e23)) < 0.5)
                        note(f"候选 10 自洽汇总（{cat}） — 格式 21", abs(cf21[10] - (lost + e21)) < 0.5)
                o, o21 = post_o, post21
                continue
            if kinds[k] == ActionKind.END_TURN:
                pred23, pred21 = float(cf[10]), float(cf21[10])
                slip = any(p[0] in ("SLIPPERY_POWER", "BUFFER_POWER", "HARDENED_SHELL_POWER") for e in pre["enemies"] if e["alive"] for p in e["powers"]) and "POISON_POWER" in fp
                thorns = "THORNS_POWER" in ppw
                post_o = parse(a.req({"op": "step", "action": k})); post21 = parse(b.req({"op": "step", "action": k})); step += 1
                post = a.state()
                over = not post.get("in_combat") or post.get("over")
                if over:
                    c["结束回合后战斗结束，不比"] += 1
                    o, o21 = post_o, post21
                    continue
                start_fx = {"REGEN_POWER", "DEMISE_POWER"} & set(ppw)
                lost = pre["player"]["hp"] - post["player"]["hp"]
                cat = "有荆棘" if thorns else "中毒的敌人带滑溜、缓冲或硬化外壳" if slip else "都没有"
                if not start_fx:
                    note(f"结束回合：玩家预计掉血（{cat}） — 格式 23", abs(pred23 - lost) < 0.5, f"{tag} 第 {step} 步：预测 {pred23:g}（21：{pred21:g}），实际 {lost}；敌人 {[(e['id'], e['hp'], [p[0] for p in e['powers']]) for e in pre['enemies'] if e['alive']]}")
                    note(f"结束回合：玩家预计掉血（{cat}） — 格式 21", abs(pred21 - lost) < 0.5)
                o, o21 = post_o, post21
                continue
            o = parse(a.req({"op": "step", "action": k})); o21 = parse(b.req({"op": "step", "action": k})); step += 1
    a.close(); b.close()
    for name in ("v23.log", "v21.log"):
        log = (args.log_dir / name).read_text(encoding="utf-8", errors="replace")
        c[f"引擎日志里的异常行（{name}）"] = sum(1 for line in log.splitlines() if "Exception" in line or "[error]" in line.lower())
    out = {"统计": dict(sorted(c.items())), "不一致的例子": ex, "场数": args.combats, "seconds": round(time.time() - t0, 1)}
    print(json.dumps(out, ensure_ascii=False, indent=1))
    if args.dump:
        d = Path(args.dump); d.mkdir(parents=True, exist_ok=True)
        (d / "格式23对真实结果.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


BOSS_GROUPS = {"MONSTER.TEST_SUBJECT": "实验体", "MONSTER.AEONGLASS": "永世沙漏", "MONSTER.QUEEN": "女王", "MONSTER.CRUSHER": "凯撒蟹",
               "MONSTER.ROCKET": "凯撒蟹", "MONSTER.INFESTED_PRISM": "感染棱柱"}


def cmd_replays(args) -> None:
    """猎人评测回放（训练好的模型真实打的局）上，格式 23 与 21 的候选 10 差多少、会不会改变“哪一步之后掉血最少”和“能不能活过本回合”。"""
    import gzip
    a = Raw(Path(args.new), open(args.log_dir / "rp23.log", "w"))
    b = Raw(Path(args.new), open(args.log_dir / "rp21.log", "w"))
    vocab = a.env.vocab()
    c = Counter(); diffs: dict[str, list[float]] = {}
    files = sorted(Path(args.dir).glob("v*_*.jsonl.gz"))[: args.n]
    t0 = time.time()
    for p in files:
        seed = p.name.split("_", 1)[1].removesuffix(".jsonl.gz")
        with gzip.open(p, "rt", encoding="utf-8") as f:
            lines = [json.loads(x) for x in f]
        head, acts = lines[0], [int(r["a"]) for r in lines[1:] if r.get("a") is not None]
        req = {"op": "reset", "mode": "run", "character": "SILENT", "ascension": 10, "seed": seed, "snapshots": True, "route_summary": True,
               "loop_cut": True, "stall_turns": 20}
        o = parse(a.req({**req, "obs_format": 23})); o21 = parse(b.req({**req, "obs_format": 21}))
        ok = True
        for step, k in enumerate(acts):
            if o.done or o.num_actions == 0:
                break
            n = o.num_actions
            if k >= n:
                c["动作越界（回放与引擎对不上）"] += 1; ok = False; break
            if o.decision == DecisionKind.COMBAT:
                ti = o.tok_ints
                names = {vocab[ti[j, 1]] for j in enemy_records(o)}
                grp = next((g for m, g in BOSS_GROUPS.items() if m in names), "其他战斗")
                kinds = o.cand_ints[:n, 0]
                acts_k = [j for j in range(n) if kinds[j] in (ActionKind.PLAY_CARD, ActionKind.USE_POTION, ActionKind.END_TURN)]
                c10a, c10b = o.cand_feats[:n, 10], o21.cand_feats[:n, 10]
                hp = float(o.cand_feats[0, 10] + o.cand_feats[0, 11]) if n else 0.0     # 10 + 11 = 当前生命
                c[f"{grp}：战斗帧"] += 1
                d = [float(c10a[j] - c10b[j]) for j in acts_k if kinds[j] != ActionKind.END_TURN]
                if any(abs(x) > 0.5 for x in d): c[f"{grp}：有出牌或药水候选的候选 10 变了的帧"] += 1
                diffs.setdefault(grp, []).extend(x for x in d if abs(x) > 0.5)
                c[f"{grp}：出牌与药水候选"] += len(d); c[f"{grp}：其中候选 10 变了"] += sum(1 for x in d if abs(x) > 0.5)
                if acts_k:
                    best23 = {j for j in acts_k if c10a[j] <= min(c10a[jj] for jj in acts_k) + 0.5}
                    best21 = {j for j in acts_k if c10b[j] <= min(c10b[jj] for jj in acts_k) + 0.5}
                    if best23 != best21: c[f"{grp}：“做完就结束回合掉血最少”的候选集合变了的帧"] += 1
                    live23 = {j for j in acts_k if c10a[j] < hp}; live21 = {j for j in acts_k if c10b[j] < hp}
                    if live23 != live21: c[f"{grp}：“做完就结束回合还活着”的候选集合变了的帧"] += 1
                    if not live21 and live23: c[f"{grp}：格式 21 看来哪步都活不过、格式 23 看出有活路的帧"] += 1
                if kinds[k] in (ActionKind.PLAY_CARD, ActionKind.USE_POTION):
                    c[f"{grp}：模型选的出牌与药水"] += 1
                    if abs(float(c10a[k] - c10b[k])) > 0.5: c[f"{grp}：模型选的那一步候选 10 变了"] += 1
            o = parse(a.req({"op": "step", "action": k})); o21 = parse(b.req({"op": "step", "action": k}))
        c["回放局数"] += 1
        fl = int(o.header[7])
        if ok and fl == int(head["floor"]): c["结束层数与回放一致的局"] += 1
        else: c["结束层数与回放不一致的局"] += 1
    a.close(); b.close()
    stat = {}
    for g, xs in diffs.items():
        if xs:
            arr = np.array(xs)
            stat[g] = {"变了的候选数": len(xs), "平均差（23 − 21）": round(float(arr.mean()), 2), "格式 23 更少的比例": round(float((arr < 0).mean()), 3),
                       "差的绝对值中位数": float(np.median(np.abs(arr))), "差的绝对值 90 分位": float(np.percentile(np.abs(arr), 90))}
    out = {"统计": dict(sorted(c.items())), "候选 10 的差": stat, "回放": [p.name for p in files], "seconds": round(time.time() - t0, 1)}
    print(json.dumps(out, ensure_ascii=False, indent=1))
    if args.dump:
        d = Path(args.dump); d.mkdir(parents=True, exist_ok=True)
        (d / "格式23评测回放.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("effects")
    e.add_argument("--new", required=True); e.add_argument("--combats", type=int, default=200); e.add_argument("--dump", default="")
    e.add_argument("--examples", type=int, default=8)
    r = sub.add_parser("replays", help="猎人评测回放上量格式 23 改了多少候选 10")
    r.add_argument("--new", required=True); r.add_argument("--dir", required=True); r.add_argument("--n", type=int, default=60); r.add_argument("--dump", default="")
    for p in (e, r):
        p.add_argument("--log-dir", type=Path, default=REPO / ".cache" / "obs_v23_check")
    args = ap.parse_args()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    {"effects": cmd_effects, "replays": cmd_replays}[args.cmd](args)


if __name__ == "__main__":
    main()
