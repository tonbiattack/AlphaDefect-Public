"""猎人（SILENT）在现有引擎里的冒烟：整局与专门牌组的单场战斗，随机出牌（偏向多打牌）。
统计：帧数、帧版本、词表缺失（头部 14）、卡住、决策类型分布、选牌决策（弃牌等）次数、手牌里带奇巧标记的帧、引擎日志里的异常行。
用法：python -I tools/silent_smoke.py <Sts2Headless.dll> <输出目录> [整局数] [专门牌组场数]（猎人观测调研.md 第五节的读数是 4 局、24 场）"""
import collections
import json
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))
from sts2env.env import Sts2Env  # noqa: E402
from sts2env.protocol import ActionKind, DecisionKind, TokenKind, Zone  # noqa: E402

dll, out = Path(sys.argv[1]), Path(sys.argv[2])
runs = int(sys.argv[3]) if len(sys.argv) > 3 else 4
combats = int(sys.argv[4]) if len(sys.argv) > 4 else 24
out.mkdir(parents=True, exist_ok=True)
log_path = out / "engine.log"
log = open(log_path, "w")
eng = Sts2Env(dll=dll, log=log)
vocab = eng.vocab()
silent_cards = [v for v in vocab if v.startswith("CARD.")]


def policy(rng):
    def act(o):
        n = o.num_actions
        if o.decision == DecisionKind.COMBAT:
            kinds = o.cand_ints[:n, 0]
            play = [k for k in range(n) if kinds[k] in (ActionKind.PLAY_CARD, ActionKind.USE_POTION)]
            if play and rng.random() < 0.85:
                return rng.choice(play)
        return rng.randrange(n)
    return act


st = {"frames": 0, "versions": collections.Counter(), "vocab_missing_frames": 0, "vocab_missing_max": 0, "decisions": collections.Counter(),
      "card_select_frames": 0, "sly_hand_frames": 0, "retain_hand_frames": 0, "games": 0, "stuck": 0, "floors": [], "steps": [],
      "unknown_ids": collections.Counter()}


def look(o):
    st["frames"] += 1
    h = o.header
    st["versions"][int(h[0])] += 1
    miss = int(h[14])
    if miss:
        st["vocab_missing_frames"] += 1
        st["vocab_missing_max"] = max(st["vocab_missing_max"], miss)
    st["decisions"][DecisionKind(int(o.decision)).name] += 1
    if o.decision == DecisionKind.CARD_SELECT:
        st["card_select_frames"] += 1
    ti, tf = o.tok_ints, o.tok_feats
    hand = (ti[:, 0] == TokenKind.CARD) & (ti[:, 2] == Zone.HAND)
    if (hand & (tf[:, 18] > 0)).any():
        st["sly_hand_frames"] += 1
    if (hand & (tf[:, 10] > 0)).any():
        st["retain_hand_frames"] += 1
    for j in range(ti.shape[0]):
        if ti[j, 1] == 0 and ti[j, 0] in (TokenKind.CARD, TokenKind.RELIC, TokenKind.POWER, TokenKind.POTION):
            st["unknown_ids"][int(ti[j, 0])] += 1


t0 = time.time()
for i in range(runs):
    rng = random.Random(i); pol = policy(rng)
    o = eng.reset_run("SILENT", 10, f"SL{i}", snapshots=True)
    step = 0
    while True:
        look(o)
        if o.done or o.num_actions == 0 or step >= 3000:
            break
        o = eng.step(pol(o)); step += 1
    st["games"] += 1; st["stuck"] += int(o.stuck); st["floors"].append(int(o.floor)); st["steps"].append(step)
    print(f"整局 SL{i}：{step} 步，第 {o.floor} 层{'，卡住' if o.stuck else ''}", flush=True)

focus = ["NIGHTMARE", "WELL_LAID_PLANS", "KNIFE_TRAP", "HAND_TRICK", "TOOLS_OF_THE_TRADE", "BURST", "STRANGLE", "SERPENT_FORM",
         "CALCULATED_GAMBLE", "PREPARED", "ACROBATICS", "SURVIVOR", "STORM_OF_STEEL", "PHANTOM_BLADES", "ACCELERANT", "BOUNCING_FLASK",
         "MALAISE", "SKEWER", "GRAND_FINALE", "MEMENTO_MORI", "MURDER", "EXPERTISE", "MASTER_PLANNER", "UNTOUCHABLE", "REFLEX", "TACTICIAN",
         "SNEAKY", "FLICK_FLACK", "RICOCHET", "ABRASIVE", "BLADE_DANCE", "INFINITE_BLADES", "ENVENOM", "CORROSIVE_WAVE", "NOXIOUS_FUMES",
         "WRAITH_FORM", "AFTERIMAGE", "BLUR", "FINISHER", "PINPOINT", "THE_HUNT", "MIRAGE", "EXPOSE", "OUTBREAK", "SPEEDSTER", "TRACKING"]
relsets = [["RING_OF_THE_SNAKE", "TINGSHA", "TOUGH_BANDAGES"], ["SNECKO_SKULL", "TWISTED_FUNNEL", "NINJA_SCROLL"], ["HELICAL_DART", "PAPER_KRANE"]]
bad_specs = []
for i in range(combats):
    rng = random.Random(500 + i); pol = policy(rng)
    picks = rng.sample(focus, 9)
    deck = [{"id": c, "upgrades": rng.randrange(2)} for c in picks for _ in range(2)] + [{"id": "STRIKE_SILENT", "upgrades": 0}] * 3 + [{"id": "DEFEND_SILENT", "upgrades": 0}] * 2
    spec = {"character": "SILENT", "ascension": 10, "seed": f"SF{i}", "act": i % 3, "encounter": ["regular", "elite", "boss"][i % 3],
            "deck": deck, "relics": relsets[i % 3], "potions": ["POISON_POTION", "GHOST_IN_A_JAR", "CUNNING_POTION"][: 1 + i % 3]}
    try:
        o = eng.reset_combat(spec)
    except Exception as e:  # noqa: BLE001
        bad_specs.append((spec["seed"], repr(e)[:200])); continue
    step = 0
    while True:
        look(o)
        if o.done or o.num_actions == 0 or step >= 600:
            break
        o = eng.step(pol(o)); step += 1
    st["games"] += 1; st["stuck"] += int(o.stuck)
eng.close(); log.close()
errs = [ln.strip()[:240] for ln in open(log_path, encoding="utf-8", errors="replace") if any(k in ln for k in ("Exception", "异常", "错误", "Error", "[warn]"))]
st["engine_error_lines"] = len(errs); st["engine_error_examples"] = errs[:15]; st["bad_specs"] = bad_specs
st["seconds"] = round(time.time() - t0, 1)
for k in ("versions", "decisions", "unknown_ids"):
    st[k] = dict(st[k])
print(json.dumps(st, ensure_ascii=False, indent=1))
(out / "silent_smoke.json").write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
