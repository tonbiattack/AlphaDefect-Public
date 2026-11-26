"""游戏更新的影响评估（引擎模块与游戏更新适配.md 第六节第 3 步）：新旧两个版本之间改了什么，哪些改动碰到了引擎的代码。

  1. 内容目录（engine/catalog/<版本>/catalog.json）逐条比较：新增、删掉、改了的卡牌、遗物、能力、药水、怪物、遭遇、事件等，
     改了的列出变了哪些字段（费用、类型、稀有度、目标、关键词、数值、升级后的数值、效果签名……）。
  2. 引擎依赖的游戏类：扫描引擎源码里写全名引用的游戏类（MegaCrit.Sts2.Core.…），在两个版本的反编译源码里各找这个类的文件，
     比较内容：删掉的、改了的类连同引擎里引用它的位置（文件:行）一起列出——这些是要逐个复核的地方（观测里按游戏代码写死的算法、补丁的目标）。
  3. 内容表命中：内容目录里改了或删掉的条目，引擎源码里按类名引用到的（内容表、按卡牌写死的规则），单独列出。

  python3 tools/game_update/update_impact.py --old v0.111.0 --new v0.112.0 [--out 报告.md]

只读：不改任何文件（--out 只写报告）。反编译源码在 ~/.local/share/sls2/decompiled/<版本>/（按命名空间分目录，一个类一个文件）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DECOMP = Path.home() / ".local/share/sls2/decompiled"
ENGINE_SRC = [REPO / "engine" / "Sts2Headless", REPO / "engine" / "Sts2Bridge"]
SKIP_FIELDS = {"id"}       # 词表 ID 只追加，不算改动


def snake(name: str) -> str:
    """类名 → 内容目录的条目名（StrikeIronclad → STRIKE_IRONCLAD，同游戏的 ModelId 规则）。"""
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_", name).upper()


def load_catalog(v: str) -> dict[str, dict]:
    doc = json.loads((REPO / "engine" / "catalog" / v / "catalog.json").read_text(encoding="utf-8"))
    return {e["key"]: e for e in doc["entities"]}


def diff_entity(a: dict, b: dict, skip: set[str] = frozenset()) -> list[str]:
    out = []
    for k in sorted(set(a) | set(b)):
        va, vb = a.get(k), b.get(k)
        if k in SKIP_FIELDS or k in skip or va == vb or (not va and not vb and va != 0 and vb != 0):     # None、{}、[] 都当作没有
            continue
        if isinstance(va, dict) and isinstance(vb, dict):
            for kk in sorted(set(va) | set(vb)):
                if va.get(kk) != vb.get(kk):
                    out.append(f"{k}.{kk}: {va.get(kk)} → {vb.get(kk)}")
        elif isinstance(va, list) and isinstance(vb, list):
            add, rm = sorted(set(map(str, vb)) - set(map(str, va))), sorted(set(map(str, va)) - set(map(str, vb)))
            out.append(f"{k}: " + "，".join(x for x in (f"加 {add}" if add else "", f"去 {rm}" if rm else "") if x))
        else:
            out.append(f"{k}: {va} → {vb}")
    return out


def engine_refs() -> dict[str, list[tuple[str, int]]]:
    """引擎源码里写全名引用的游戏类：全名 → [(文件, 行)]。"""
    pat = re.compile(r"\bMegaCrit\.Sts2\.Core(?:\.[A-Z][A-Za-z0-9]*)+")
    refs: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for root in ENGINE_SRC:
        for p in root.rglob("*.cs"):
            if "/obj/" in str(p) or "/bin/" in str(p):
                continue
            for i, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if line.lstrip().startswith("using "):
                    continue
                for m in pat.finditer(line):
                    refs[m.group(0)].append((str(p.relative_to(REPO)), i))
    return refs


def class_file(version: str, fqn: str) -> Path | None:
    """全名 → 反编译源码里的文件（命名空间目录/类名.cs）；嵌套类、枚举值等取最外层能找到的类。"""
    parts = fqn.split(".")
    for cut in range(len(parts) - 1, 3, -1):
        ns, name = ".".join(parts[:cut]), parts[cut]
        p = DECOMP / version / ns / f"{name}.cs"
        if p.exists():
            return p
    return None


def digest(p: Path) -> str:
    # 反编译输出里的行尾空白、编译器生成的名字顺序可能变，只比较去掉空白的内容
    txt = re.sub(r"\s+", " ", p.read_text(encoding="utf-8", errors="replace"))
    return hashlib.sha1(txt.encode()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--max", type=int, default=400, help="每类最多列多少条")
    a = ap.parse_args()
    L: list[str] = [f"# 游戏更新影响评估：{a.old} → {a.new}", ""]

    # 1. 内容目录
    old_c, new_c = load_catalog(a.old), load_catalog(a.new)
    added = sorted(set(new_c) - set(old_c)); removed = sorted(set(old_c) - set(new_c))
    # 内容目录本身的格式也会扩充（如 v0.111.0 起加了升级后的数值）：只有一边有的字段不算内容的改动，单独说明
    fo = {k for e in old_c.values() for k in e}; fn = {k for e in new_c.values() for k in e}
    schema = (fo ^ fn) - SKIP_FIELDS
    changed = {k: d for k in sorted(set(old_c) & set(new_c)) if (d := diff_entity(old_c[k], new_c[k], schema))}
    by_cat = lambda keys: defaultdict(list, {c: [k for k in keys if k.split(".", 1)[0] == c] for c in sorted({k.split(".", 1)[0] for k in keys})})
    L += ["## 一、内容目录", "", f"新增 {len(added)}，删掉 {len(removed)}，改了 {len(changed)}（不含只是词表 ID 不同的）。", ""]
    if schema:
        L += [f"目录格式不同：只有旧版有的字段 {sorted(fo - fn)}，只有新版有的字段 {sorted(fn - fo)}（导出目录的代码改过，这些字段不参与比较）。", ""]
    for title, keys in (("新增", added), ("删掉", removed)):
        if keys:
            L.append(f"### {title}")
            for c, ks in by_cat(keys).items():
                L.append(f"- {c}（{len(ks)}）：" + "、".join(k.split(".", 1)[1] for k in ks[: a.max]))
            L.append("")
    if changed:
        L.append("### 改了")
        for c, ks in by_cat(list(changed)).items():
            L.append(f"- **{c}**（{len(ks)}）")
            for k in ks[: a.max]:
                L.append(f"  - {k.split('.', 1)[1]}：" + "；".join(changed[k]))
        L.append("")

    # 2. 引擎依赖的游戏类
    refs = engine_refs()
    files = {}
    for fqn in refs:
        fo, fn = class_file(a.old, fqn), class_file(a.new, fqn)
        if fo is None and fn is None:
            continue
        key = str((fo or fn).relative_to(DECOMP / (a.old if fo else a.new)))
        files.setdefault(key, {"old": fo, "new": fn, "refs": []})["refs"] += refs[fqn]
    gone = {k: v for k, v in files.items() if v["new"] is None}
    diff = {k: v for k, v in files.items() if v["old"] is not None and v["new"] is not None and digest(v["old"]) != digest(v["new"])}
    L += ["## 二、引擎依赖的游戏类", "",
          f"引擎源码里写全名引用的游戏类对应 {len(files)} 个文件：新版里没有了 {len(gone)} 个，内容变了 {len(diff)} 个。"
          "没有了的要改引擎（编译也会报错）；变了的要对照新旧反编译源码复核引擎里按它写的逻辑（观测的算法、补丁的目标与参数）。", ""]
    for title, group in (("新版里没有了", gone), ("内容变了", diff)):
        if not group:
            continue
        L.append(f"### {title}")
        for k in sorted(group):
            locs = sorted(set(group[k]["refs"]))
            L.append(f"- `{k}`：引擎里 {len(locs)} 处 — " + "，".join(f"{f}:{n}" for f, n in locs[:8]) + (" ……" if len(locs) > 8 else ""))
        L.append("")

    # 3. 内容表命中：内容目录里改了或删掉的条目，引擎按类名引用到的
    touched = set(changed) | set(removed)
    hits = []
    for fqn, locs in refs.items():
        name = fqn.rsplit(".", 1)[1]
        kind = fqn.split(".")[4] if len(fqn.split(".")) > 5 else ""
        cat = {"Cards": "CARD", "Relics": "RELIC", "Powers": "POWER", "Potions": "POTION", "Monsters": "MONSTER", "Events": "EVENT", "Encounters": "ENCOUNTER",
               "Orbs": "ORB", "Enchantments": "ENCHANTMENT", "Afflictions": "AFFLICTION"}.get(kind)
        if cat and f"{cat}.{snake(name)}" in touched:
            hits.append((f"{cat}.{snake(name)}", sorted(set(locs))))
    # 按类名字符串写的内容表（如 OrbChannels）
    for root in ENGINE_SRC:
        for p in root.rglob("*.cs"):
            if "/obj/" in str(p):
                continue
            for i, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                for m in re.finditer(r'"([A-Z][A-Za-z]+)"', line):
                    k = f"CARD.{snake(m.group(1))}"
                    if k in touched and ("case" in line or "{" in line):
                        hits.append((k, [(str(p.relative_to(REPO)), i)]))
    merged: dict[str, set] = defaultdict(set)
    for k, locs in hits: merged[k] |= set(locs)
    hits = [(k, sorted(v)) for k, v in merged.items()]
    L += ["## 三、内容表命中", "", "内容目录里改了或删掉、引擎又按名字用到的条目（内容表、按卡牌写死的规则），要逐条看引擎的处理还对不对：", ""]
    for k, locs in sorted(hits):
        L.append(f"- {k}（{'删掉' if k in removed else '；'.join(changed.get(k, []))[:200]}）：" + "，".join(f"{f}:{n}" for f, n in locs[:6]))
    if not hits:
        L.append("（没有）")
    L.append("")
    text = "\n".join(L)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
        print(f"报告写入 {a.out}（内容目录：新增 {len(added)}、删掉 {len(removed)}、改了 {len(changed)}；引擎依赖的类：没有了 {len(gone)}、变了 {len(diff)}；内容表命中 {len(hits)}）")
    else:
        print(text)


if __name__ == "__main__":
    main()
