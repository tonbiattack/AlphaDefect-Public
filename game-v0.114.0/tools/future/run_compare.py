"""整局评测的逐种子配对比较（2026-11-17）：run_search.py 的 runs_s*.jsonl，各组对参照组（默认第一组）。

  python3 tools/future/run_compare.py 名字=目录[,目录…] 名字=目录 …
每组：局数、通关率、对参照的配对差（± 一个标准误，只赢 / 只输，符号检验 p）、死因分布、各幕 Boss 的局内胜率、每局秒数。
"""
import glob
import json
import math
import sys
from collections import Counter


def load(spec: str) -> dict:
    out = {}
    for d in spec.split(","):
        for f in glob.glob(f"{d}/runs_s*.jsonl"):
            for l in open(f, encoding="utf-8"):
                if l.strip():
                    r = json.loads(l)
                    out[r["seed"]] = r
    return out


def sign_p(a: int, b: int) -> float:
    n = a + b
    if n == 0:
        return 1.0
    k = min(a, b)
    s = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * s)


def main() -> None:
    arms = [(x.split("=", 1)[0], load(x.split("=", 1)[1])) for x in sys.argv[1:]]
    ref_name, ref = arms[0]
    for name, R in arms:
        n = len(R)
        if not n:
            print(f"{name}: 0 局"); continue
        w = sum(r["win"] for r in R.values())
        secs = sum(r.get("s", 0) for r in R.values()) / n
        line = f"{name:10s} {n:5d} 局 通关 {w / n:.4f}  每局 {secs:5.1f} 秒"
        if name != ref_name:
            com = [s for s in R if s in ref]
            d = [R[s]["win"] - ref[s]["win"] for s in com]
            if com:
                m = sum(d) / len(d)
                sd = math.sqrt(sum((x - m) ** 2 for x in d) / max(1, len(d) - 1))
                up, dn = sum(1 for x in d if x > 0), sum(1 for x in d if x < 0)
                line += f" | 对 {ref_name} {len(com)} 局：{100 * m:+.2f} ± {100 * sd / math.sqrt(len(d)):.2f}（+{up} −{dn}，p ≈ {sign_p(up, dn):.3g}）"
        print(line)
        deaths = Counter(r.get("death") for r in R.values() if not r["win"])
        print("    死因：" + "，".join(f"{k} {v / n:.3f}" for k, v in deaths.most_common(6)))
        boss = {}
        for r in R.values():
            k3 = 0
            for f in r.get("fights", []):
                if f["room"] == "Boss":
                    if f["act"] == 2:
                        k3 += 1
                    key = f"第{f['act'] + 1}幕" + ("" if f["act"] < 2 else ("·一" if k3 == 1 else "·二"))
                    st = boss.setdefault(key, [0, 0])
                    st[0] += 1; st[1] += int((f.get("hp1") or 0) > 0)
        print("    Boss 局内胜率：" + "，".join(f"{k} {v[1] / v[0]:.4f}（{v[0]}）" for k, v in sorted(boss.items())))
        srch = sum(r.get("searches", 0) for r in R.values()) / n
        mac = sum((r.get("macro") or {}).get("searches", 0) for r in R.values()) / n
        if srch or mac:
            print(f"    每局搜索：战斗 {srch:.1f} 次（改选 {sum(r.get('changed', 0) for r in R.values()) / n:.1f}），局外 {mac:.1f} 次")


if __name__ == "__main__":
    main()
