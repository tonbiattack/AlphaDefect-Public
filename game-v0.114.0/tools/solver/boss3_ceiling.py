"""第三幕双 Boss 的“理论上能不能赢”（决策记录 286）：评测里死在第三幕第一个或第二个 Boss 的局，从第一个 Boss 开打前的快照起，
用 Combat Solver 模组（完全信息：知道这场的抽牌顺序和怪物行动）连解两场。

每题：第一个 Boss 用评测记下的战前快照（ability_test 的 boss3 项）开打；求解器能赢，就把快照的血量、最大生命改成它最好路线打完时的值，
去掉路线上用掉的药水，接着解第二个 Boss（同 A10 连打：两场之间没有别的事件）。两场都赢 = 理论上能通关。
口径：每场战斗的随机数由题目自己的种子（B3C_<种子>_1 / _2）生成，不是评测那一局原来的随机数；开打时的选择（如开局选牌的遗物）
求解器不接，选第一项；打完第一场后的遗物回血等没有算进去。所以是“这副牌、这个血量、这些药水，在一组随机数下完美打法能不能赢”。

    python3 tools/solver/boss3_ceiling.py 死局.json 结果.jsonl [--workers 4] [--dop 4] [--timeout 120]
死局.json：[{seed, died_at, encounter, next_encounter, hp, max_hp, potions, deck, snap}, ...]。已解过的种子跳过（可续跑）。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HOME = Path.home()
REPO = Path(__file__).resolve().parents[2]
DLL = REPO / "engine/Sts2Headless/bin/Release/net9.0/Sts2Headless.dll"
MODS = HOME / ".local/share/sls2/mods"


def solve(spec: dict, timeout: float, dop: int) -> dict:
    env = dict(os.environ, DOTNET_ROOT=str(HOME / ".dotnet"), PATH=f"{HOME / '.dotnet'}:{os.environ['PATH']}", DOTNET_CLI_TELEMETRY_OPTOUT="1",
               COMBAT_SOLVER_DIR=str(MODS / "CombatSolver"), RITSULIB_DIR=str(MODS / "RitsuLib"), SOLVER_DOP=str(dop))
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        f.write(json.dumps({"label": {}, "spec": spec}, ensure_ascii=False) + "\n")
    try:
        p = subprocess.run(["dotnet", str(DLL), "solve-test", f.name, "0", "--timeout-s", str(timeout)], env=env, capture_output=True, text=True,
                           timeout=timeout + 180)
        out = next((json.loads(l) for l in reversed(p.stdout.splitlines()) if l.startswith('{"id"')), None)
        if out is None:
            return {"error": (p.stdout + p.stderr)[-400:]}
        out.pop("policy", None)
        return out
    except subprocess.TimeoutExpired:
        return {"error": "进程超时"}
    finally:
        os.unlink(f.name)


def won(r: dict) -> bool:
    # 限时内没有结论时返回的是未分胜负的最好路线：不算能赢
    return bool(r.get("win")) and not r.get("timed_out") and not r.get("error")


def after_first(snap: str, r: dict) -> str:
    """第一场打完的快照：血量、最大生命取求解器路线的结局，去掉用掉的药水。"""
    s = json.loads(snap)
    s["current_hp"] = int(r["end_hp"]); s["max_hp"] = int(r.get("end_max_hp") or s["max_hp"])
    pots = list(s.get("potions") or [])
    for pid in r.get("potions_used") or []:
        key = str(pid).removeprefix("POTION.")
        k = next((i for i, p in enumerate(pots) if p["id"].removeprefix("POTION.") == key), None)
        if k is not None:
            pots.pop(k)
    s["potions"] = pots
    return json.dumps(s, ensure_ascii=False)


def one(it: dict, a) -> dict:
    t0 = time.time()
    spec1 = {"character": "DEFECT", "ascension": 10, "seed": f"B3C_{it['seed']}_1", "act": 2, "encounter": it["encounter"], "player_snapshot": it["snap"]}
    r1 = solve(spec1, a.timeout, a.dop)
    rec = {"seed": it["seed"], "died_at": it["died_at"], "encounter": it["encounter"], "next_encounter": it["next_encounter"],
           "hp": it["hp"], "max_hp": it["max_hp"], "potions": it["potions"], "deck": it["deck"], "b1": r1}
    if won(r1) and it.get("next_encounter"):
        spec2 = {"character": "DEFECT", "ascension": 10, "seed": f"B3C_{it['seed']}_2", "act": 2,
                 "encounter": it["next_encounter"].removeprefix("ENCOUNTER."), "prev_encounter": it["encounter"],
                 "player_snapshot": after_first(it["snap"], r1)}
        rec["b2"] = solve(spec2, a.timeout, a.dop)
    rec["win"] = won(r1) and won(rec.get("b2") or {})
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("items")
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dop", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=120)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    items = json.loads(Path(a.items).read_text(encoding="utf-8"))
    out = Path(a.out)
    done = {json.loads(l)["seed"] for l in open(out, encoding="utf-8")} if out.exists() else set()
    todo = [it for it in items if it["seed"] not in done][: a.limit or None]
    print(f"{time.strftime('%T')} {len(items)} 题，已解 {len(done)}，这次 {len(todo)} 题；{a.workers} 路 × {a.dop} 线程，每场限时 {a.timeout:.0f} 秒", flush=True)
    n = w = 0
    with ThreadPoolExecutor(a.workers) as pool:
        for fu in as_completed([pool.submit(one, it, a) for it in todo]):
            rec = fu.result(); n += 1; w += rec["win"]
            with open(out, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            b1, b2 = rec["b1"], rec.get("b2") or {}
            s1 = "出错" if b1.get("error") else "超时" if b1.get("timed_out") else f"赢（剩 {b1.get('end_hp')}）" if b1.get("win") else "输"
            s2 = "" if not b2 else " → 第二个 " + ("出错" if b2.get("error") else "超时" if b2.get("timed_out") else f"赢（剩 {b2.get('end_hp')}）" if b2.get("win") else "输")
            print(f"{time.strftime('%T')} [{n}/{len(todo)}] {rec['seed']}（评测死在第 {rec['died_at']} 个，入场 {rec['hp']}/{rec['max_hp']}）第一个 {s1}{s2}；"
                  f"累计能通关 {w}/{n}，{rec['seconds']:.0f} 秒", flush=True)


if __name__ == "__main__":
    sys.exit(main())
