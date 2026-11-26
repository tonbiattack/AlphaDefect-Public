"""评测与复测的种子池（2026-11-01 起，决策记录 240）：用系统的密码学随机源（secrets，即 /dev/urandom）抽种子字符串，取代按序号起名的
RT0000–RT0999（复测）与从中按难度分层抽出的 384 个（评测，tools/eval_seeds.py）。

游戏把种子字符串哈希成 32 位整数（StringHelper.GetDeterministicHashCode）再起随机数发生器；只差末几位的序号种子哈希值也只在
有规律的几位上不同，不能保证各局之间互不相关。这里每个种子是 10 位独立均匀抽取的字符（字母表去掉易混的 I、O，共 34 个，约 51 位熵），
1,000 个互不相同；评测种子是从池里不放回均匀抽出的 384 个（简单随机抽样，不再按难度分层），所以评测与复测照旧有 384 局重叠，
复测沿用评测留存的那 384 局、只打其余 616 个。

    python3 tools/seed_pool.py --name P1 --out tools/remote/seed_pool.json
池一经启用就不再改动；要换池就起一个新名字（P2…），评测与复测按池名分口径。
"""
from __future__ import annotations

import argparse
import json
import secrets
import time
from pathlib import Path

ALPHABET = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ"
OLD_PREFIXES = ("RT", "HO", "TEST", "EVAL")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="池名，评测与复测按它分口径")
    ap.add_argument("--n", type=int, default=1000, help="复测种子数")
    ap.add_argument("--eval", type=int, default=384, help="其中给常规评测的种子数")
    ap.add_argument("--length", type=int, default=10)
    ap.add_argument("--out", default="tools/remote/seed_pool.json")
    a = ap.parse_args()
    rng = secrets.SystemRandom()
    seeds: list[str] = []
    while len(seeds) < a.n:
        s = "".join(secrets.choice(ALPHABET) for _ in range(a.length))
        if s not in seeds and not s.startswith(OLD_PREFIXES):     # 与旧的序号种子不会撞名，以防万一仍排除
            seeds.append(s)
    pick = set(rng.sample(range(a.n), a.eval))
    out = {"name": a.name, "created": time.strftime("%F %T %z"), "seeds": seeds, "eval": [s for i, s in enumerate(seeds) if i in pick],
           "note": f"secrets.choice 独立均匀抽 {a.length} 位（字母表 {ALPHABET}），{a.n} 个互不相同；eval 为其中不放回均匀抽出的 {a.eval} 个"
                   "（tools/seed_pool.py，决策记录 240）"}
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"种子池 {a.name}：{a.n} 个，其中评测 {a.eval} 个 → {a.out}")


if __name__ == "__main__":
    main()
