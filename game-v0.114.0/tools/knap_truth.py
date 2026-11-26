"""背包（敌人记录 18“本回合最多还能打多少”、19“打不打得死”）的真值核对：格式 28 与格式 29 各比一遍（R3-格式29规格.md 5.2，包 2）。

每个工作进程开两个引擎（同一程序集）：A 按格式 28、B 按格式 29，同一个 CombatSpec、同样的动作锁步走（动作按 B 的帧随机挑，偏向出牌）。
战斗中的出牌决策、手牌 ≤ 7、能量 ≤ 5 的帧，在 B 上用局面复制暴力搜索（tools/obs29/knap.py 的 Searcher），得出本回合实际能对每个敌人
打掉的生命 + 格挡（真值），再与两边帧里的敌人 18、19 比较：18 对 min(真值, 17)，19 对“真值 ≥ 17”。结果按条件分组（规格 5.2 表里每一条一类）。

  python -I tools/knap_truth.py --out out.json                       # 全部场景
  python -I tools/knap_truth.py --only WHIRLWIND,MAUL --combats 4 --out out.json
  python -I tools/knap_truth.py --report out.json                    # 只打印已有结果的表

场景见 SCENARIOS：铁甲战士的旋风斩、撕咬、放血 + 祭品、燃烧、FEEDING_FRENZY、凡庸、恶魔之焰、飞剑回旋镖（多敌）、卡戎之灰 + 消耗牌、狱火 + 自伤牌、
ONE_TWO_PUNCH、势不可当、ENTHRALLED，仪式兽（鸣响）、往日雕像（迟缓）、猎杀者（软弱）、幽灵园丁（胆小）；猎人小刀 + 幻影之刃；亡灵致死性；
储君虚空形态、星尘；愚者暴力和弦、爆炸箭矢；不含任何条件的对照组。
同时运行的引擎进程 = 2 × --workers（默认 4 个工作进程，共 8 个引擎）。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "tools"))

from sts2env.env import DEFAULT_DLL, EngineError, Sts2Env  # noqa: E402
from sts2env.protocol import parse  # noqa: E402
from obs29 import knap  # noqa: E402

I_BASE = ["STRIKE_IRONCLAD"] * 4 + ["DEFEND_IRONCLAD"] * 3 + ["BASH"]
S_BASE = ["STRIKE_SILENT"] * 4 + ["DEFEND_SILENT"] * 3 + ["NEUTRALIZE"]
N_BASE = ["STRIKE_NECROBINDER"] * 4 + ["DEFEND_NECROBINDER"] * 3
R_BASE = ["STRIKE_REGENT"] * 4 + ["DEFEND_REGENT"] * 3 + ["FALLING_STAR", "VENERATE"]
F_BASE = ["STRIKE_FOOL"] * 4 + ["DEFEND_FOOL"] * 3 + ["FOOLISH_SHOT"]


def D(*names, up=()):
    """牌组：名字列表，up 为要升级的名字（每个名字第一张升级）。"""
    out, done = [], set()
    for n in names:
        u = 1 if (n in up and n not in done) else 0
        if u:
            done.add(n)
        out.append({"id": n, "upgrades": u})
    return out


# 场景：名字 → 角色、牌组、遗物、遭遇（每场轮流）、幕、额外的 CombatSpec 字段、优先打出的牌（走步时 80% 先打它们，好让能力挂上）、
# 重点条件（帧里出现这些条件时一定算真值，每场最多 --frames 帧；其余帧按 --other 的概率抽，每场最多 --frames 帧）
def S(ch, deck, relics=(), encs=("regular", "elite"), act=0, extra=None, prefer=(), focus=()):
    return {"char": ch, "deck": deck, "relics": list(relics), "encs": list(encs), "act": act, "extra": extra or {}, "prefer": set(prefer), "focus": set(focus)}


BIG = {"max_hp": 300, "hp": 300}
SCENARIOS: dict[str, dict] = {
    "无条件": S("IRONCLAD", D(*I_BASE, "TWIN_STRIKE", "POMMEL_STRIKE", "SHRUG_IT_OFF", "IRON_WAVE"), encs=("regular", "SLIMES_NORMAL")),
    "WHIRLWIND": S("IRONCLAD", D(*I_BASE, "WHIRLWIND", "WHIRLWIND", "TWIN_STRIKE", up=("WHIRLWIND",)), encs=("SLIMES_NORMAL", "regular"), focus=("G18-X费",)),
    "MAUL": S("IRONCLAD", D("STRIKE_IRONCLAD", "DEFEND_IRONCLAD", "DEFEND_IRONCLAD", "BASH", "MAUL", "MAUL", "MAUL", "MAUL", "MAUL", up=("MAUL",)),
              act=1, focus=("G64-撕咬",)),
    "BLOODLETTING+OFFERING": S("IRONCLAD", D(*I_BASE, "BLOODLETTING", "OFFERING", "BLUDGEON", "HEMOKINESIS", "TWIN_STRIKE"), ["RED_SKULL"], act=1,
                               extra={"hp": 46}, focus=("G76-得能量得星", "G76-加力量")),
    "INFLAME": S("IRONCLAD", D(*I_BASE, "INFLAME", "INFLAME", "TWIN_STRIKE", "TWIN_STRIKE", "POMMEL_STRIKE"), focus=("G76-加力量",)),
    "FEEDING_FRENZY": S("IRONCLAD", D(*I_BASE, "FEEDING_FRENZY", "FEEDING_FRENZY", "TWIN_STRIKE", "SETUP_STRIKE", up=("FEEDING_FRENZY",)),
                        focus=("G76-加力量",)),
    "NORMALITY": S("IRONCLAD", D(*I_BASE, "NORMALITY", "ANGER", "ANGER", "FLASH_OF_STEEL", "FLASH_OF_STEEL", "BLOODLETTING"), encs=("regular",),
                   focus=("G16-凡庸",)),
    "FIEND_FIRE": S("IRONCLAD", D(*I_BASE, "FIEND_FIRE", "FIEND_FIRE", "TWIN_STRIKE", up=("FIEND_FIRE",)), act=1, focus=("G19-恶魔之焰",)),
    "SWORD_BOOMERANG": S("IRONCLAD", D(*I_BASE, "SWORD_BOOMERANG", "SWORD_BOOMERANG", "SWORD_BOOMERANG", up=("SWORD_BOOMERANG",)),
                         encs=("SLIMES_NORMAL", "CULTISTS_NORMAL", "NIBBITS_NORMAL"), focus=("G20-随机目标",)),
    "CHARON": S("IRONCLAD", D(*I_BASE, "TRUE_GRIT", "SECOND_WIND", "OFFERING", "IMPERVIOUS", "PACTS_END"), ["CHARONS_ASHES"], act=1, focus=("G43-铁甲",)),
    "INFERNO": S("IRONCLAD", D(*I_BASE, "INFERNO", "INFERNO", "BLOODLETTING", "HEMOKINESIS", "BREAKTHROUGH", "BLOOD_WALL"), act=1,
                 prefer=("INFERNO",), focus=("G43-铁甲",)),
    "ONE_TWO_PUNCH": S("IRONCLAD", D(*I_BASE, "ONE_TWO_PUNCH", "ONE_TWO_PUNCH", "TWIN_STRIKE", "POMMEL_STRIKE", up=("ONE_TWO_PUNCH",)),
                       prefer=("ONE_TWO_PUNCH",), focus=("G17-ONE_TWO_PUNCH",)),
    "JUGGERNAUT": S("IRONCLAD", D(*I_BASE, "JUGGERNAUT", "JUGGERNAUT", "SHRUG_IT_OFF", "IRON_WAVE"), encs=("regular", "SLIMES_NORMAL"), act=1,
                    prefer=("JUGGERNAUT",), focus=("G43-铁甲",)),
    "ENTHRALLED": S("IRONCLAD", D(*I_BASE, "ENTHRALLED", "TWIN_STRIKE"), encs=("regular",), focus=("G16-ENTHRALLED",)),
    "仪式兽": S("IRONCLAD", D(*I_BASE, "TWIN_STRIKE", "POMMEL_STRIKE", "ANGER", "SHRUG_IT_OFF", up=("STRIKE_IRONCLAD",)), encs=("CEREMONIAL_BEAST_BOSS",),
             extra=BIG, focus=("G16-鸣响",)),
    "往日雕像": S("IRONCLAD", D(*I_BASE, "TWIN_STRIKE", "POMMEL_STRIKE", "ANGER"), encs=("BYGONE_EFFIGY_ELITE",), focus=("G19-迟缓",)),
    "猎杀者": S("IRONCLAD", D(*I_BASE, "TWIN_STRIKE", "POMMEL_STRIKE", "ANGER"), encs=("HUNTER_KILLER_NORMAL",), act=1, focus=("G19-软弱",)),
    "幽灵园丁": S("IRONCLAD", D(*I_BASE, "TWIN_STRIKE", "POMMEL_STRIKE", "ANGER"), encs=("PHANTASMAL_GARDENERS_ELITE",), focus=("G19-胆小",)),
    "猎人幻影之刃": S("SILENT", D(*S_BASE, "PHANTOM_BLADES", "PHANTOM_BLADES", "BLADE_DANCE", "BLADE_DANCE", "CLOAK_AND_DAGGER", up=("PHANTOM_BLADES",)),
                prefer=("PHANTOM_BLADES", "BLADE_DANCE", "CLOAK_AND_DAGGER"), extra=BIG, focus=("G17-幻影之刃",)),
    "亡灵致死性": S("NECROBINDER", D(*N_BASE, "LETHALITY", "LETHALITY", "POKE", "SNAP", up=("LETHALITY",)), prefer=("LETHALITY",), extra=BIG,
               focus=("G17-致死性",)),
    "储君虚空形态": S("REGENT", D(*R_BASE, "VOID_FORM", "VOID_FORM", "GLOW", "GATHER_LIGHT", "SHINING_STRIKE", up=("VOID_FORM",)), act=1,
                prefer=("VOID_FORM",), extra=BIG, focus=("G17-虚空形态",)),
    "储君星尘": S("REGENT", D(*R_BASE, "STARDUST", "STARDUST", "GATHER_LIGHT", "GLOW", "ROYAL_GAMBLE"), encs=("SLIMES_NORMAL", "regular"),
              focus=("G18-X费", "G20-随机目标")),
    "愚者暴力和弦": S("FOOL", D(*F_BASE, "VIOLENT_CHORD", "VIOLENT_CHORD", "WAH_HAMMER", "WAH_TOUCH", "WAH_KI"), encs=("regular", "SLIMES_NORMAL"),
                prefer=("VIOLENT_CHORD",), extra=BIG, focus=("G43-愚者",)),
    "愚者爆炸箭矢": S("FOOL", D(*F_BASE, "EXPLOSIVE_ARROWS", "EXPLOSIVE_ARROWS", "QUICK_SHOT", "RAPID_SHOT", "HEAVY_SHOT"), encs=("regular", "SLIMES_NORMAL"),
                prefer=("EXPLOSIVE_ARROWS",), extra=BIG, focus=("G43-愚者",)),
}


class Eng:
    def __init__(self, dll: Path, fmt: int):
        self.env = Sts2Env(dll=dll)
        self.fmt = fmt

    def req(self, r: dict):
        self.env._proc.send(r)
        return parse(self.env._proc.recv())

    def close(self):
        self.env.close()


def run_scenario(args) -> dict:
    name, combats, dll, seed0, max_frames, budget, p_other = args
    sc = SCENARIOS[name]
    ch = sc["char"]
    A, B = Eng(Path(dll), 28), Eng(Path(dll), 29)
    voc = A.env.vocab()
    rng = np.random.default_rng(seed0)
    srch = knap.Searcher(B.env._proc, voc, parse, budget=budget)
    rows, stats = [], {"frames": 0, "evaluated": 0, "skipped_big": 0, "budget": 0, "desync": 0, "errors": 0, "seconds": 0.0}
    t0 = time.time()
    for ci in range(combats):
        spec = {"character": ch, "seed": f"KT{seed0}_{ci}", "act": sc["act"], "encounter": sc["encs"][ci % len(sc["encs"])], "deck": sc["deck"],
                "relics": sc["relics"], **sc["extra"]}
        try:
            oa = A.req({"op": "reset", "mode": "combat", "spec": spec, "obs_format": 28, "loop_cut": True, "stall_turns": 20})
            ob = B.req({"op": "reset", "mode": "combat", "spec": spec, "obs_format": 29, "loop_cut": True, "stall_turns": 20})
        except EngineError as e:
            stats["errors"] += 1
            print(f"[{name}] reset 出错：{str(e)[:200]}", flush=True)
            continue
        ev_focus = ev_other = 0
        acts: list[int] = []
        for step in range(600):
            if ob.done or ob.num_actions == 0:
                break
            if oa.num_actions != ob.num_actions:
                stats["desync"] += 1
                break
            stats["frames"] += 1
            if ob.decision == 1:
                fb, fa = knap.Frame(ob, voc), knap.Frame(oa, voc)
                conds_all = {c for f in fb.foes for c in knap.conditions(fb, f)}
                focus = bool(conds_all & sc["focus"])
                take = (focus and ev_focus < max_frames) or (not focus and ev_other < max_frames and rng.random() < p_other)
                if take and (len(fb.hand) > 7 or fb.energy > 5):
                    stats["skipped_big"] += 1
                elif take and fb.hand and len(fa.foes) == len(fb.foes):
                    try:
                        r = srch.truth(ob, rng)
                    except EngineError:
                        stats["errors"] += 1
                        r = None
                    if r is not None:
                        if focus:
                            ev_focus += 1
                        else:
                            ev_other += 1
                        stats["evaluated"] += 1
                        for k, tv in r["truth"].items():
                            if tv is None:
                                stats["budget"] += 1
                                continue
                            conds = knap.conditions(fb, fb.foes[k])
                            row = knap.compare_row(fa, fb, k, tv, conds, r["random"])
                            row.update({"scen": name, "char": ch, "combat": ci, "step": step, "hand": fb.hand_names(), "energy": fb.energy,
                                        "spec": spec, "acts": list(acts), "k": k})
                            rows.append(row)
            # 走一步：优先打出 prefer 里的牌，否则偏向出牌的随机
            n = ob.num_actions
            kinds = ob.cand_ints[:n, 0]
            play = [k for k in range(n) if kinds[k] in (2, 3)]
            pref = [k for k in range(n) if kinds[k] == 2 and knap._name(voc, int(ob.tok_ints[int(ob.cand_ints[k, 1]), 1])) in sc["prefer"]]
            if ob.decision == 1 and pref and rng.random() < 0.8:
                a = int(rng.choice(pref))
            else:
                a = int(rng.choice(play)) if (ob.decision == 1 and play and rng.random() < 0.85) else int(rng.integers(n))
            if ob.decision == 1 and kinds[a] == 3 and rng.random() < 0.7:      # 少用药水
                a = int(next((k for k in range(n) if kinds[k] == 1), a))
            oa, ob = A.req({"op": "step", "action": a}), B.req({"op": "step", "action": a})
            acts.append(a)
    stats["seconds"] = round(time.time() - t0, 1)
    A.close(); B.close()
    return {"name": name, "rows": rows, "stats": stats}


def fmt_pct(x):
    return "—" if x is None else f"{100 * x:.1f}%"


def table(rows: list[dict], title: str) -> list[str]:
    """每格写“格式 28 → 格式 29”。一致率、差 ≤ 2、偏差、MAE 只算非随机帧；随机帧（真值为 6 个 reseed 的平均）另给 MAE。"""
    sm = knap.summarize(rows)
    out = [f"\n## {title}（{len(rows)} 个帧 × 敌人）\n",
           "| 条件 | 帧×敌人 | 非随机 | 18 一致 | 差≤2 | 偏差 | MAE | 随机帧 MAE | 19 一致 |",
           "|---|---|---|---|---|---|---|---|---|"]
    def arrow(a, b, f):
        return "—" if a is None else f"{f(a)} → {f(b)}"
    num = lambda x: f"{x}"
    for g, d in sm.items():
        a, b = d["28"], d["29"]
        out.append(f"| {g} | {d['n']} | {a['n18']} | {arrow(a['exact'], b['exact'], fmt_pct)} | {arrow(a['within2'], b['within2'], fmt_pct)} | "
                   f"{arrow(a['bias'], b['bias'], num)} | {arrow(a['mae'], b['mae'], num)} | {arrow(a['mae_rand'], b['mae_rand'], num)} | "
                   f"{arrow(a['agree19'], b['agree19'], fmt_pct)} |")
    return out


# 规格 5.2“核对”的场景 → 条目（表里的名字）
SCEN_ITEM = {"WHIRLWIND": "G18 旋风斩", "MAUL": "G64 撕咬", "BLOODLETTING+OFFERING": "G76 放血 + 祭品（红骷髅）", "INFLAME": "G76 燃烧",
             "FEEDING_FRENZY": "G69 FEEDING_FRENZY", "NORMALITY": "G16 凡庸", "FIEND_FIRE": "G19 恶魔之焰", "SWORD_BOOMERANG": "G20 飞剑回旋镖（多敌）",
             "CHARON": "G43 卡戎之灰 + 消耗牌", "INFERNO": "G43 狱火 + 自伤牌", "ONE_TWO_PUNCH": "G17 ONE_TWO_PUNCH", "JUGGERNAUT": "G43 势不可当",
             "ENTHRALLED": "G16 ENTHRALLED", "仪式兽": "G16 鸣响（仪式兽）", "往日雕像": "G19 迟缓（往日雕像）", "猎杀者": "G19 软弱（猎杀者）",
             "幽灵园丁": "G19 胆小（幽灵园丁）", "猎人幻影之刃": "G17 幻影之刃（猎人）", "亡灵致死性": "G17 致死性（亡灵）",
             "储君虚空形态": "G17 虚空形态（储君）", "储君星尘": "G18 星尘（储君）", "愚者暴力和弦": "G43 暴力和弦（愚者）", "愚者爆炸箭矢": "G43 爆炸箭矢（愚者）"}


# 条目本身就是随机目标的场景（随机帧算进目标）；其余场景的随机帧（如多个敌人时的势不可当，规格只做单个敌人）只列 MAE、不算目标
RANDOM_SCEN = {"SWORD_BOOMERANG", "储君星尘"}


def scenario_table(res: dict) -> list[str]:
    """规格 5.2 的核对场景逐条：只看帧里出现了这个场景的重点条件、且不含规格不做的项的（帧 × 敌人）。
    非随机帧比 18 完全一致率，目标 ≥ 90% 且不比格式 28 差（28 已是 100% 时持平也算）；随机帧比平均绝对误差，目标下降。"""
    out = ["\n## 规格 5.2 的核对场景（帧里有该场景的重点条件，去掉规格不做的项）\n",
           "| 条目 | 场景 | 涉及（帧×敌人） | 非随机 | 18 一致 28 | 18 一致 29 | 随机帧 MAE 28 → 29 | 目标 | 通过 |",
           "|---|---|---|---|---|---|---|---|---|"]
    for sc in res["scenarios"]:
        name = sc["name"]
        focus = SCENARIOS[name]["focus"]
        if not focus:
            continue
        rs = [r for r in sc["rows"] if set(r["conds"]) & focus and not r.get("oos")]
        det = [r for r in rs if not r["random"] and r["target"][1] is not None]
        rnd = [r for r in rs if r["random"] and r["target"][1] is not None]
        ex = [sum(abs(r["e18c"][j] - r["target"][j]) <= 0.5 for r in det) / len(det) if det else None for j in (0, 1)]
        mae = [float(np.mean([abs(r["e18c"][j] - r["target"][j]) for r in rnd])) if rnd else None for j in (0, 1)]
        ok = True
        goal = []
        if det:
            goal.append("非随机一致 ≥ 90%")
            ok &= ex[1] >= 0.9 and ex[1] >= ex[0]
        if rnd and name in RANDOM_SCEN:
            goal.append("随机帧 MAE 下降")
            ok &= mae[1] < mae[0] or mae[0] == 0
        out.append(f"| {SCEN_ITEM.get(name, name)} | {name} | {len(rs)} | {len(det)} | {fmt_pct(ex[0])} | {fmt_pct(ex[1])} | "
                   f"{'—' if mae[0] is None else f'{mae[0]:.2f} → {mae[1]:.2f}'} | {'；'.join(goal) or '—'} | {'是' if ok and rs else ('无帧' if not rs else '否')} |")
    return out


def report(res: dict) -> str:
    rows = [r for s in res["scenarios"] for r in s["rows"]]
    lines = []
    for title, sel in [("全部", rows), ("全部，去掉规格不做的项（先打增益：上易伤、ONE_TWO_PUNCH 牌、降费、先打能力牌；G48 打下箭再射）与归包 4 的 PACTS_END 的帧", [r for r in rows if not r.get("oos")])] \
            + [(c, [r for r in rows if r["char"] == c]) for c in ("IRONCLAD", "SILENT", "NECROBINDER", "REGENT", "FOOL")] \
            + [(c + "，去掉规格不做的项与 PACTS_END 的帧", [r for r in rows if r["char"] == c and not r.get("oos")]) for c in ("IRONCLAD", "SILENT", "NECROBINDER", "REGENT", "FOOL")]:
        if sel:
            lines += table(sel, title)
    lines += scenario_table(res)
    # 不含任何条件的帧：格式 29 的 18、19 与 28 逐位相同
    none = [r for r in rows if not r["conds"]]
    same = sum(r["e18"][0] == r["e18"][1] and r["e19"][0] == r["e19"][1] for r in none)
    lines.append(f"\n不含任何条件的（帧 × 敌人）{len(none)} 个，格式 29 的 18、19 与 28 相同 {same} 个。")
    st = {k: sum(s["stats"][k] for s in res["scenarios"]) for k in res["scenarios"][0]["stats"]} if res["scenarios"] else {}
    lines.append(f"统计：{json.dumps(st, ensure_ascii=False)}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dll", default=str(DEFAULT_DLL))
    ap.add_argument("--only", default="", help="逗号分隔的场景名（默认全部）")
    ap.add_argument("--combats", type=int, default=3, help="每个场景打几场")
    ap.add_argument("--frames", type=int, default=12, help="每场最多算几帧的真值（重点条件的帧与其余帧各算这么多）")
    ap.add_argument("--other", type=float, default=0.3, help="不含重点条件的帧按这个概率抽来算真值")
    ap.add_argument("--budget", type=int, default=400, help="每个（帧，敌人，seed）最多展开几个节点，超出不计")
    ap.add_argument("--workers", type=int, default=4, help="工作进程数（每个开两个引擎）")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="")
    ap.add_argument("--report", default="", help="只打印这个结果文件的表")
    a = ap.parse_args()
    if a.report:
        print(report(json.loads(Path(a.report).read_text(encoding="utf-8"))))
        return
    names = [x for x in a.only.split(",") if x] or list(SCENARIOS)
    jobs = [(n, a.combats, a.dll, a.seed * 1000 + i, a.frames, a.budget, a.other) for i, n in enumerate(names)]
    t0 = time.time()
    out = []
    with Pool(min(a.workers, len(jobs))) as pool:
        for r in pool.imap_unordered(run_scenario, jobs):
            out.append(r)
            print(f"{r['name']}: {len(r['rows'])} 行 {json.dumps(r['stats'], ensure_ascii=False)}", flush=True)
    out.sort(key=lambda r: names.index(r["name"]))
    res = {"dll": a.dll, "seconds": round(time.time() - t0, 1), "args": vars(a), "scenarios": out}
    if a.out:
        Path(a.out).write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    print(report(res))


if __name__ == "__main__":
    main()
