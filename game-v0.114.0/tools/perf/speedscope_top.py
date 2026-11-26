"""dotnet-trace 的 speedscope 文件 → 某个函数之下的耗时排行（引擎提速剖析用，2026-11-16）。

  dotnet-trace collect --profile dotnet-sampled-thread-time --format Speedscope -o x.nettrace -- dotnet Sts2Headless.dll bench-encode ...
  python3 tools/perf/speedscope_top.py x.speedscope.json --root 'Observation.Build' --top 60

只统计栈里含 --root 的时间：inclusive 为栈里出现过这一帧的时间（同一栈里递归只算一次），self 为它在栈顶的时间。
--under 再限定只看栈里含某一帧的部分（如 'Observation.AddChoices'）。--callers 列出某帧的直接调用者。
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict


def short(name: str) -> str:
    # 去掉参数表与程序集名，留 类型.方法
    name = re.sub(r"\(.*$", "", name)
    name = name.split("!")[-1]
    parts = name.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--root", default="Observation.Build")
    ap.add_argument("--under", default="")
    ap.add_argument("--top", type=int, default=50)
    ap.add_argument("--callers", default="")
    ap.add_argument("--full", action="store_true", help="不缩短帧名")
    a = ap.parse_args()
    d = json.load(open(a.file))
    frames = d["shared"]["frames"]
    names = [f["name"] if a.full else short(f["name"]) for f in frames]
    incl = defaultdict(float); self_ = defaultdict(float); callers = defaultdict(float)
    root_total = 0.0; all_total = 0.0
    for prof in d["profiles"]:
        if prof.get("type") != "evented":
            continue
        stack: list[int] = []
        last = None
        for ev in prof["events"]:
            t = ev["at"]
            if last is not None and stack:
                dt = t - last
                all_total += dt
                ns = [names[i] for i in stack]
                if any(a.root in n for n in ns) and (not a.under or any(a.under in n for n in ns)):
                    root_total += dt
                    # 只看 root 之下的部分
                    k = max(i for i, n in enumerate(ns) if a.root in n)
                    sub = ns[k:]
                    for n in set(sub):
                        incl[n] += dt
                    self_[sub[-1]] += dt
                    if a.callers:
                        for i in range(1, len(sub)):
                            if a.callers in sub[i]:
                                callers[sub[i - 1]] += dt
                                break
            last = t
            if ev["type"] == "O":
                stack.append(ev["frame"])
            elif ev["type"] == "C":
                # 关闭的应是栈顶
                if stack and stack[-1] == ev["frame"]:
                    stack.pop()
                elif ev["frame"] in stack:
                    stack = stack[:len(stack) - 1 - stack[::-1].index(ev["frame"])]
    unit = d["profiles"][0].get("unit", "?")
    print(f"全部 {all_total:.0f} {unit}，含 {a.root}{' 且含 ' + a.under if a.under else ''} 的 {root_total:.0f} {unit}")
    print(f"\n== inclusive（占 root 的比例） ==")
    for n, v in sorted(incl.items(), key=lambda kv: -kv[1])[:a.top]:
        print(f"{v / root_total * 100:6.1f}%  {v:9.0f}  {n}")
    print(f"\n== self ==")
    for n, v in sorted(self_.items(), key=lambda kv: -kv[1])[:a.top]:
        print(f"{v / root_total * 100:6.1f}%  {v:9.0f}  {n}")
    if a.callers:
        print(f"\n== {a.callers} 的直接调用者 ==")
        for n, v in sorted(callers.items(), key=lambda kv: -kv[1])[:a.top]:
            print(f"{v / root_total * 100:6.1f}%  {v:9.0f}  {n}")


if __name__ == "__main__":
    main()
