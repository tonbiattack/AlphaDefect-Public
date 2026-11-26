"""局面复制 × 钩子监听者列表池的定向核对（2026-11-16，引擎提速）。

游戏的钩子迭代可能停在等玩家选择的地方：赌博筹码（GAMBLING_CHIP，AfterPlayerTurnStart 里让你选牌弃掉）、工具箱（TOOLBOX，BeforeHandDraw 里选一张牌）
等，外层 foreach 还没走完就出了选牌决策。在这样的决策点拍快照、往下走、再恢复，恢复后的时间线要与一口气走下来的完全相同。
PerfPatches.HookListenerPool 把监听者列表借给迭代器：池若不随快照恢复，恢复后迭代器拿着的列表可能已在池里、被别人借走清空。

每题：新引擎开战 → 走到第一个选牌决策 → snapshot → 按种子随机走 N 步记帧（A）→ restore → 同样的动作再走（B），A 与 B 逐字节比；
另用参照引擎（没有局面复制，--base）开同一题、一口气走同样的动作，与 A 逐字节比（选牌决策之前的动作一并重放）。

  python -I tools/perf/clone_pool_check.py --new .cache/eng-perf/Sts2Headless.dll --base 参照.dll --n 12
退出码：有不同为 1。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from boss_equal import Raw  # noqa: E402
from sts2env.protocol import ActionKind, DecisionKind, parse  # noqa: E402

CHARS = ["IRONCLAD", "SILENT", "DEFECT", "REGENT", "NECROBINDER", "FOOL"]
RELICS = [["GAMBLING_CHIP"], ["TOOLBOX"], ["GAMBLING_CHIP", "TOOLBOX"]]


def pick(rng: random.Random, o) -> int:
    n = o.num_actions
    if o.decision == DecisionKind.COMBAT:
        kinds = o.cand_ints[:n, 0]
        play = [k for k in range(n) if kinds[k] in (ActionKind.PLAY_CARD, ActionKind.USE_POTION)]
        if play and rng.random() < 0.85:
            return rng.choice(play)
    return rng.randrange(n)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--new", required=True)
    ap.add_argument("--base", default="")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--log-dir", type=Path, default=REPO / ".cache" / "clone_pool_check")
    a = ap.parse_args()
    a.log_dir.mkdir(parents=True, exist_ok=True)
    new = Raw(Path(a.new), open(a.log_dir / "new.log", "w"))
    base = Raw(Path(a.base), open(a.log_dir / "base.log", "w")) if a.base else None
    st = {"cases": 0, "hit_select": 0, "frames": 0, "diff_restore": 0, "diff_base": 0, "errors": 0, "examples": []}
    for i in range(a.n):
        spec = {"character": CHARS[i % len(CHARS)], "seed": f"POOL{i}", "act": i % 3, "encounter": ["weak", "regular", "elite", "boss"][i % 4],
                "relics": RELICS[i % len(RELICS)]}
        reset = {"op": "reset", "mode": "combat", "spec": spec, "privileged": True}
        rng = random.Random(i)
        pre: list[int] = []
        try:
            x = new.req(reset)
            o = parse(x)
            # 走到第一个选牌决策（停在钩子迭代中间的那种）
            while o.decision != DecisionKind.CARD_SELECT and not o.done and o.num_actions > 0 and len(pre) < 200:
                k = pick(rng, o); pre.append(k)
                x = new.req({"op": "step", "action": k}); o = parse(x)
            st["cases"] += 1
            if o.decision != DecisionKind.CARD_SELECT:
                continue
            st["hit_select"] += 1
            root = x
            new.req({"op": "snapshot", "slot": 0})
            acts, fa = [], []
            for _ in range(a.steps):
                if o.done or o.num_actions == 0:
                    break
                k = pick(rng, o); acts.append(k)
                x = new.req({"op": "step", "action": k}); o = parse(x); fa.append(x)
            y = new.req({"op": "restore", "slot": 0})
            fb = []
            same = y == root
            for k in acts:
                y = new.req({"op": "step", "action": k}); fb.append(y)
            st["frames"] += len(fa)
            if not same or fa != fb:
                st["diff_restore"] += 1
                if len(st["examples"]) < 8:
                    j = next((t for t, (p, q) in enumerate(zip(fa, fb)) if p != q), None)
                    st["examples"].append({"case": i, "spec": spec, "root_same": same, "first_diff_step": j})
            if base is not None:
                z = base.req(reset)
                for k in pre:
                    z = base.req({"op": "step", "action": k})
                ok = z == root
                fz = []
                for k in acts:
                    z = base.req({"op": "step", "action": k}); fz.append(z)
                if not ok or fz != fa:
                    st["diff_base"] += 1
                    if len(st["examples"]) < 8:
                        j = next((t for t, (p, q) in enumerate(zip(fa, fz)) if p != q), None)
                        st["examples"].append({"case": i, "vs": "base", "root_same": ok, "first_diff_step": j})
            new.req({"op": "drop"})
        except Exception as e:  # noqa: BLE001
            st["errors"] += 1
            if len(st["examples"]) < 8:
                st["examples"].append({"case": i, "spec": spec, "error": str(e)[:200]})
            new.close()
            new = Raw(Path(a.new), open(a.log_dir / "new.log", "a"))
    new.close()
    if base:
        base.close()
    print(json.dumps(st, ensure_ascii=False, indent=1))
    sys.exit(1 if st["diff_restore"] or st["diff_base"] or st["errors"] else 0)


if __name__ == "__main__":
    main()
