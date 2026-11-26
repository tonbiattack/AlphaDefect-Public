"""sts2-sim 的“存读档 + 编码”对拍（2026-11-06）：Check.cs 只按语料动作重放、比内部状态，不经过搜索用到的两条路——
每个决策前存档（save）、读档（load，回一帧，要调观测编码器）。编码器给手牌算预览时会调用伤害、格挡的修正钩子，
钩子里有状态副作用就只在这条路上出错（护臂曾把被预览的牌记成触发牌，SIMX0001：格挡 26 对原版 39）。

这里照搜索评测的做法：sts2-sim 服务（移植版编码器）导入语料里每场战斗前的存档，每个顶层战斗决策前 save → load，
再用 tools/sim_pimc.py 的 compare（血量、格挡、能量、手牌、敌人、决策类型）与真实帧比，候选按语义对应后走语料里的动作。

    python -I tools/sim_roundtrip_check.py --dll Sts2SimServe.dll --vocab vocab.txt --tables sim_tables.json [--rooms boss] [--acts 2] 语料1.jsonl [语料2.jsonl ...]
"""
from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "tools"))

from sts2env.env import EngineError, _Process  # noqa: E402
from sts2env.protocol import DecisionKind, parse  # noqa: E402
import sim_pimc as S  # noqa: E402


class _Pipe(_Process):
    def __init__(self, p):      # 借用 _Process 的帧读写
        self.p = p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dll", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--tables", required=True)
    ap.add_argument("--rooms", default="boss")
    ap.add_argument("--acts", default="2")
    ap.add_argument("--no-roundtrip", action="store_true", help="不存读档，只走步（每步照样编码）：区分是复制状态还是编码出的问题")
    ap.add_argument("--seeds", default=None, help="只查这些种子（逗号分隔）")
    ap.add_argument("corpus", nargs="+")
    a = ap.parse_args()
    rooms, acts = set(a.rooms.split(",")), {int(x) for x in a.acts.split(",")}
    p = subprocess.Popen(["dotnet", a.dll, "serve", a.vocab, a.tables], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, bufsize=0)
    sp = _Pipe(p)

    def req(d: dict, raw: bool = False):
        sp.send(d); b = sp.recv(); return b if raw else parse(b)

    n = ok = 0
    bad: Counter = Counter()
    for fn in a.corpus:
        for line in open(fn, encoding="utf-8"):
            r = json.loads(line)
            if r["room"] not in rooms or int(r["act"]) not in acts or (a.seeds and r["seed"] not in a.seeds.split(",")):
                continue
            n += 1
            frames = [parse(base64.b64decode(f)) for f in r["frames"]]
            why = None
            try:
                m = req({"op": "reset", "mode": "transplant", "save": r["save"], "next_room_id": r["next_room_id"], "target": r["target"],
                         "room": r["room"], "root_frame": r["frames"][0]})
                for i, act in enumerate(r["actions"]):
                    o = frames[i]
                    if not a.no_roundtrip and o.decision == DecisionKind.COMBAT and not S.combat_over(m):
                        req({"op": "save", "slot": 0}, raw=True)
                        m = req({"op": "load", "slot": 0})
                    why = S.compare(o, m)
                    if why:
                        why = f"第 {i} 步 {why}"
                        break
                    cmap, how = S.cand_map(o, m)
                    if cmap is None:
                        why = f"第 {i} 步 {how}"
                        break
                    m = req({"op": "step", "action": cmap[act]})
            except EngineError as e:
                why = f"出错：{str(e)[:120]}"
                p.kill()
                p = subprocess.Popen(["dotnet", a.dll, "serve", a.vocab, a.tables], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, bufsize=0)
                sp = _Pipe(p)
            if why is None:
                ok += 1
            else:
                bad[why.split("：")[0].split(" ")[-1] if "出错" not in why else "出错"] += 1
                print(f"{r['seed']} 第{int(r['act']) + 1}幕 第{r['floor']}层 {r['room']}：{why}", flush=True)
    p.kill()
    print(f"汇总：{n} 场，存读档 + 编码全程一致 {ok}；不一致 {dict(bad)}")


if __name__ == "__main__":
    main()
