"""把一局回放（ability_test / 评测存的 <版本>_<种子>.jsonl.gz）写成中文的逐步行动记录（Markdown），方便照着手打。
2026-11-05 用户要求：模型池在种子上找到通关解后，“把解给我（中文文件）”。

    python3 tools/replay_text.py 回放.jsonl.gz -o 解.md [--title 说明]

名字取自本机游戏文件的中文本地化（与 tools/replay_viewer.py 同一套名表）。战斗按回合列出出牌、用药水与目标；
局外列出每次选择与当时的其他选项（按模型给的概率排序）。
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay_viewer import REPO, names  # noqa: E402

DK = ["卡住", "战斗", "选牌", "卡牌奖励", "选卡包", "领取奖励", "事件", "休息处", "商店", "宝箱", "占卜", "选路线", "结束"]
SEL = ["变换", "消耗", "删除", "附魔", "弃牌", "升级"]
T_CREATURE, T_CARD, T_MAP, T_OPTION = 2, 4, 8, 9


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("replay")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--title", default="")
    a = ap.parse_args()
    vocab = (REPO / "engine/catalog/vocab.txt").read_text(encoding="utf-8").split("\n")
    N = names(vocab)
    enc = {k.split(".", 1)[1]: N[i][0] for i, k in enumerate(vocab) if k.startswith("ENCOUNTER.")}
    nm = lambda i: N[i][0] if 0 <= i < len(N) else f"#{i}"
    up = lambda n: "+" * n if n and n > 0 else ""
    lines = [l for l in gzip.open(a.replay, "rt", encoding="utf-8") if l.strip()]
    head, steps = json.loads(lines[0]), [json.loads(l) for l in lines[1:]]
    combats = {c["floor"]: c for c in head.get("combats") or []}

    def obj(o, st) -> str:
        if not o:
            return ""
        k, i, z, ex = o
        if k == T_CARD:
            return nm(i) + up(ex)
        if k == T_CREATURE:
            if z == 1:
                return "自己"
            same = sum(1 for e in st.get("en") or [] if e.get("n") == i)
            return nm(i) + (f" #{ex}" if same > 1 else "")
        if k == T_OPTION:
            v, aux = ex
            s = nm(aux) if i <= 1 and aux else nm(i)
            return s + (f" {v}" if s == "金币" and v else "")
        if k == T_MAP:
            return nm(i) + f"（第 {ex[0] + 1} 行第 {ex[1] + 1} 列）"
        return nm(i)

    def cand(c, st) -> str:
        k, A, B, kk, f0, f1, f2 = c
        x, y = obj(A, st), obj(B, st)
        return {1: "结束回合", 2: f"打出【{x}】" + (f" → {y}" if B else ""), 3: f"使用药水【{x}】" + (f" → {y}" if B else ""),
                4: f"丢弃药水 {x}", 5: f"选【{x}】", 6: "确认选择", 7: f"【{x}】", 8: nm(kk), 9: f"选第 {f0 + 1} 包", 10: f"领取 {x}",
                11: "离开 / 跳过", 12: x + (f"（{y}）" if B else ""), 13: x, 14: f"购买【{x}】" + (f"（{f0} 金）" if f0 else ""),
                15: f"前往 {x}", 16: f"拿遗物【{x}】", 17: "不拿遗物", 18: f"占卜格子 ({f0}, {f1}) · {'大' if f2 else '小'}"}.get(k, f"动作 {k}")

    out = [f"# 种子 {head['seed']} 的{'通关' if head.get('win') else '对局'}记录（故障机器人 · 进阶 10）", ""]
    if a.title:
        out += [a.title, ""]
    out += [f"- 结果：{'通关' if head.get('win') else '第 ' + str(head.get('floor')) + ' 层结束'}；共 {len(steps)} 步决策；模型版本 {head.get('version')}",
            "- 记法：战斗里按回合列出每一步（打出的牌 → 目标）；局外的选择后面括号里是当时其他选项，按模型概率从高到低。"
            "同名敌人后面的 #1、#2 是引擎里的编号（一般是从左数的顺序，对不上时以敌人当时的血量为准：每层开头列了开打时各敌人的血量）。"
            "同名的牌（如几张打击）选哪一张都一样。",
            "- 引擎是游戏规则的无头复刻，随机数由种子决定；手打时只要每一步都一样，后续抽牌、敌人行动、奖励应当一致。若某一步对不上（如抽到的牌不同），说明之前有一步不同。", ""]
    floor, turn, buf = None, None, []
    for st in steps:
        g = st["g"]
        if g[1] != floor:
            floor, turn = g[1], None
            c = combats.get(floor)
            room = (enc.get(c["encounter"], c["encounter"]) + {"Elite": "（精英）", "Boss": "（Boss）"}.get(c["room"], "")) if c else ""
            out += ["", f"## 第 {g[0] + 1} 幕 · 第 {floor} 层" + (f" · {room}" if room else ""), f"进入时生命 {g[2]}/{g[3]}，金币 {g[4]}", ""]
        ch = st["c"][st["a"]] if st.get("c") else None
        if not ch:
            continue
        if st["d"] == 1:
            if turn is None and floor not in combats and st.get("en"):     # 回放头没记遭遇时，用开打时的敌人列名字
                foes = "、".join(f"{nm(e['n'])}（{e['hp']}）" for e in st["en"])
                out.append(f"- 战斗：{foes}")
            if g[7] != turn:
                turn = g[7]
                hand = "、".join(nm(h[0]) + up(h[1]) for h in st.get("hand") or [])
                out.append(f"- **第 {turn} 回合**（能量 {g[5]}/{g[6]}，生命 {g[2]}；手牌：{hand}）")
            out.append(f"  - {cand(ch, st)}")
            continue
        pr = {k: p for k, p in st.get("pr") or []}
        others = sorted((k for k in range(len(st["c"])) if k != st["a"]), key=lambda k: -pr.get(k, 0))
        what = DK[st["d"]] if st["d"] < len(DK) else "决策"
        if st.get("sel"):
            what = "选牌：" + (SEL[st["sel"][0]] if st["sel"][0] < len(SEL) else "选择")
        alt = "、".join(cand(st["c"][k], st) for k in others[:6]) + ("……" if len(others) > 6 else "")
        out.append(f"- {what}：{cand(ch, st)}" + (f"（其他：{alt}）" if alt else ""))
    last = steps[-1] if steps else None
    if last and last.get("deck"):
        cnt: dict[str, int] = {}
        for i, u in last["deck"]:
            s = nm(i) + up(u); cnt[s] = cnt.get(s, 0) + 1
        out += ["", "## 最终牌组", "、".join(s + (f" ×{n}" if n > 1 else "") for s, n in sorted(cnt.items())),
                "", "## 最终遗物", "、".join(nm(i) for i, _ in last.get("rel") or [])]
    Path(a.out).write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"写好 {a.out}：{len(steps)} 步", flush=True)


if __name__ == "__main__":
    main()
