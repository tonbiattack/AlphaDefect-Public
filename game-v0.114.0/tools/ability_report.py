"""模型能力测试报告（tools/ability_test.py 的结果 → 一页静态 HTML + 汇总 JSON）。

    python3 tools/ability_report.py 结果目录 输出.html [--notes 要点.txt] [--target 1000 --asc 0-10]

页面分两部分：
  · 跨进阶对比：指标卡、各进阶成绩（折线 + 明细表）、止步阶段、各进阶打法对比——每个数都只来自一个进阶，按进阶并排。
  · 分进阶细看：死因、Boss 战、构筑、局外选择、抓牌、古神遗物。每个进阶预先渲染一份，用页面上的进阶切换只显示所选的一份
    （选择记在浏览器 localStorage；默认最高进阶，进行中且最高进阶还没有数据时默认已有数据的最高进阶）。
测试进行中（局数不到 --target × 进阶数）时页面标为“进行中”、显示进度，并每 2 分钟自动重新载入（本机 tools/ability_live.sh 定时重新生成），
重新载入后保留所选进阶与滚动位置。

要点文件：每行一条，写在报告开头的“要点”里（数字由人核对后写入；不给就不显示这一节）。
中文名（遭遇、遗物、卡牌、古神）沿用看板 tools/dashboard.py 的读法；只展示本次测试的结果，不引用社区数据。
"""
from __future__ import annotations

import argparse
import html
import json
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.argv, _argv = sys.argv[:1], sys.argv          # dashboard 模块导入时不解析参数
import dashboard as D  # noqa: E402
sys.argv = _argv

esc = html.escape
ACT_CN = "一二三"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if not n:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def pct(x: float | None, d: int = 1) -> str:
    return "—" if x is None else f"{100 * x:.{d}f}%"


def ratio(k: float, n: float) -> float | None:
    return k / n if n else None


def load(dirp: Path) -> list[dict]:
    runs = []
    for f in sorted(dirp.glob("runs_s*.jsonl")):
        runs += [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
    return runs


def boss_fights(r: dict):
    """这一局的 Boss 战：(幕, 本幕第几个 Boss, 遭遇, 打赢没有, 进场血量比例)；卡住的局里最后那场不计（同 evaluator._run_diagnostics）。"""
    cs = r.get("combats") or []
    seen = Counter()
    for j, c in enumerate(cs):
        if c["room"] != "Boss":
            continue
        k = seen[c["act"]]; seen[c["act"]] += 1
        final = j == len(cs) - 1 and c["floor"] == r["floor"] and not r["win"]
        if final and r["stuck"]:
            continue
        yield c["act"], k, c["encounter"], not final, c["hp"] / max(1, c["max_hp"])


def ending(r: dict) -> str:
    """一局停在哪个阶段。"""
    if r["win"]:
        return "通关"
    if r["stuck"]:
        return "卡住"
    cs = r.get("combats") or []
    last = cs[-1] if cs and cs[-1]["floor"] == r["floor"] else None
    act = last["act"] if last else int(r["act"])
    if act >= 2:
        return "第三幕 Boss" if last and last["room"] == "Boss" else "第三幕其他"
    return f"第{ACT_CN[act]}幕"


ENDINGS = ["第一幕", "第二幕", "第三幕其他", "第三幕 Boss", "卡住", "通关"]


def summarize(runs: list[dict]) -> dict:
    by = defaultdict(list)
    for r in runs:
        by[r["asc"]].append(r)
    out = {"ascensions": {}}
    for asc in sorted(by):
        rs = by[asc]
        n = len(rs)
        w = sum(r["win"] for r in rs)
        b3 = [[c for c in (r.get("combats") or []) if c["room"] == "Boss" and c["act"] == 2] for r in rs]
        reach_b3 = sum(1 for x in b3 if x)
        # 第三幕第一个 Boss 打过：进阶 10 以下只有一个 Boss，打过即通关；进阶 10 连打两个，打过第一个即到了第二个
        pass_b3 = sum(1 for r, x in zip(rs, b3) if x and (len(x) >= 2 or (r["win"] and asc < 10)))
        reach_b3b = sum(1 for x in b3 if len(x) >= 2)
        ends = Counter(ending(r) for r in rs)
        floors = [r["floor"] for r in rs]
        out["ascensions"][asc] = {
            "n": n, "wins": w, "win_rate": w / n, "win_ci": wilson(w, n),
            "floor_mean": statistics.mean(floors), "floor_median": statistics.median(floors),
            "reach2": sum(1 for r in rs if r["act"] >= 1 or r["win"]) / n, "reach3": sum(1 for r in rs if r["act"] >= 2 or r["win"]) / n,
            "boss3_reach": reach_b3 / n, "boss3_pass": pass_b3 / reach_b3 if reach_b3 else None,
            "boss3b_reach": reach_b3b, "boss3b_pass": (w / reach_b3b if reach_b3b else None) if asc >= 10 else None,
            "stuck": sum(r["stuck"] for r in rs), "endings": {k: ends.get(k, 0) for k in ENDINGS},
        }
    return out


def merge(items) -> dict[str, list[int]]:
    tot: dict[str, list[int]] = {}
    for d in items:
        for k, v in (d or {}).items():
            acc = tot.setdefault(k, [0] * len(v))
            for j, x in enumerate(v):
                acc[j] += x
    return tot


def build_at(rs: list[dict], asc: int) -> dict:
    """打到各个 Boss 时的牌组、遗物、升级、药水与开场血量（平均）。rs 只含这一进阶的局。"""
    nodes = {"第一幕 Boss": (0, 0), "第二幕 Boss": (1, 0), "第三幕 Boss": (2, 0)}
    if asc >= 10:
        nodes["第三幕第二个 Boss"] = (2, 1)
    out = {}
    for name, (act, k) in nodes.items():
        xs = []
        for r in rs:
            bs = [c for c in (r.get("combats") or []) if c["room"] == "Boss" and c["act"] == act]
            if len(bs) > k:
                xs.append(bs[k])
        if xs:
            out[name] = {"n": len(xs), "deck": statistics.mean(c["deck"] for c in xs), "relics": statistics.mean(c["relics"] for c in xs),
                         "upgrades": statistics.mean(c.get("upgrades") or 0 for c in xs), "potions": statistics.mean(c.get("potions") or 0 for c in xs),
                         "hp": statistics.mean(c["hp"] / max(1, c["max_hp"]) for c in xs)}
    return out


def habits(rs: list[dict]) -> dict:
    """局外选择（rs 只含一个进阶）。比例的分母见各项；“每局”按全部对局平均，死得早的局次数自然少。"""
    ch = Counter()
    for r in rs:
        ch.update(r.get("choices") or {})
    n = len(rs)
    g = lambda k: ch.get(k, 0)
    rest = {k.rsplit(":", 1)[1]: v for k, v in ch.items() if k.startswith("REST_SITE:REST_OPTION:")}
    rest_all = sum(rest.values())
    scr = merge(r.get("cards") for r in rs).get("_SCREEN", [0] * 6)
    shops = g("MAP:MAP_MOVE:MAPNODE:Shop")
    past1 = [r for r in rs if r["act"] >= 1 or r["win"]]
    return {"n": n,
            "elites": sum(1 for r in rs for c in (r.get("combats") or []) if c["room"] == "Elite") / n,
            "elites1": ratio(sum(1 for r in past1 for c in (r.get("combats") or []) if c["room"] == "Elite" and c["act"] == 0), len(past1)),
            "shops": shops / n, "unknown": g("MAP:MAP_MOVE:MAPNODE:Unknown") / n, "rests": g("MAP:MAP_MOVE:MAPNODE:RestSite") / n,
            "card_skip": ratio(sum(scr[3:]), sum(scr[:3])),
            "smith": ratio(rest.get("SMITH", 0), rest_all), "heal": ratio(rest.get("HEAL", 0), rest_all),
            "rest_other": ratio(rest_all - rest.get("SMITH", 0) - rest.get("HEAL", 0), rest_all),
            "shop_card": g("SHOP:BUY:CARD") / n, "shop_relic": g("SHOP:BUY:RELIC") / n, "shop_potion": g("SHOP:BUY:POTION") / n,
            "shop_remove": g("SHOP:BUY:SHOP:CARD_REMOVAL") / n, "remove_per_shop": ratio(g("SHOP:BUY:SHOP:CARD_REMOVAL"), shops)}


_CARD_INFO: dict[str, tuple[str, str]] | None = None
RARITY_CN = {"Basic": "基础", "Common": "普通", "Uncommon": "罕见", "Rare": "稀有", "Ancient": "先古"}


def card_info() -> dict[str, tuple[str, str]]:
    """{牌: (卡池, 稀有度)}，来自内容目录；读不到为空（此时不按卡池过滤）。"""
    global _CARD_INFO
    if _CARD_INFO is None:
        _CARD_INFO = {}
        try:
            cat = json.loads((Path(D.__file__).resolve().parents[1] / "engine/catalog/v0.111.0/catalog.json").read_text(encoding="utf-8"))["entities"]
            _CARD_INFO = {e["key"].removeprefix("CARD."): (e.get("pool") or "", e.get("rarity") or "") for e in cat if e["category"] == "CARD"}
        except (OSError, ValueError, KeyError):
            pass
    return _CARD_INFO


def card_cn() -> dict[str, str]:
    try:
        return D.card_ref()[2] or {}          # 只取中文名
    except Exception:
        return {}


# ---------------------------------------------------------------- 图

PAL = ["var(--s1)", "var(--s2)", "var(--s3)"]


def line_chart(summ: dict, full: list[int], target: int = 0) -> str:
    """各进阶：过第一幕、过第二幕、通关率（一个纵轴，0–100%），通关率带 95% 置信区间。横轴固定为计划的进阶范围（进行中时只画已有的点）。"""
    A = [a for a in sorted(summ["ascensions"]) if a in full]
    pos = {a: i for i, a in enumerate(full)}
    part = lambda a: bool(target) and summ["ascensions"][a]["n"] < target
    W, H, L, R, T, B = 760, 280, 46, 18, 16, 36       # 右端不放数值标签（2026-10-27 用户要求），数值看图例、悬停提示和下面的表
    xs = lambda i: L + (W - L - R) * i / max(1, len(full) - 1)
    ys = lambda v: T + (H - T - B) * (1 - v)
    series = [("过第一幕", "reach2", PAL[0]), ("过第二幕", "reach3", PAL[1]), ("通关", "win_rate", PAL[2])]
    g = "".join(f'<line x1="{L}" x2="{W - R}" y1="{ys(t):.1f}" y2="{ys(t):.1f}" class="gl"/><text x="{L - 8}" y="{ys(t) + 4:.1f}" class="ax" text-anchor="end">{int(t * 100)}%</text>'
                for t in (0, .25, .5, .75, 1))
    g += "".join(f'<text x="{xs(i):.1f}" y="{H - B + 18}" class="ax" text-anchor="middle">{a}</text>' for i, a in enumerate(full))
    g += f'<text x="{(L + W - R) / 2:.0f}" y="{H - 4}" class="ax" text-anchor="middle">进阶</text>'
    if A:
        band = " ".join(f"{xs(pos[a]):.1f},{ys(summ['ascensions'][a]['win_ci'][1]):.1f}" for a in A) + " " + \
            " ".join(f"{xs(pos[a]):.1f},{ys(summ['ascensions'][a]['win_ci'][0]):.1f}" for a in reversed(A))
        g += f'<polygon points="{band}" fill="var(--s3)" opacity=".14"/>'
        for name, key, col in series:
            pts = [(xs(pos[a]), ys(summ["ascensions"][a][key])) for a in A]
            g += f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}" fill="none" stroke="{col}" stroke-width="1.5" stroke-linejoin="round"/>'
            g += "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="var(--surface)" stroke="{col}" stroke-width="1.5"/>' if part(a) else
                         f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{col}" stroke="var(--surface)" stroke-width="1.5"/>' for a, (x, y) in zip(A, pts))
    # 悬停：每个进阶一条竖带
    data = {str(a): [summ["ascensions"][a][k] for _, k, _ in series] + list(summ["ascensions"][a]["win_ci"]) + [summ["ascensions"][a]["n"]] for a in A}
    step = (W - L - R) / max(1, len(full) - 1)
    hits = "".join(f'<rect class="hit" data-a="{a}" x="{xs(pos[a]) - step / 2:.1f}" y="{T}" width="{step:.1f}" height="{H - T - B}"/>' for a in A)
    legend = "".join(f'<span class="lg"><i style="background:{c}"></i>{esc(n_)}</span>' for n_, _, c in series) + '<span class="lg"><i class="band"></i>通关率 95% 置信区间</span>'
    if any(part(a) for a in A):
        legend += '<span class="lg"><i class="hollow"></i>空心点：未打满</span>'
    return (f'<div class="legend">{legend}</div><div class="chart" data-series=\'{json.dumps(data)}\'>'
            f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="各进阶的过第一幕、过第二幕与通关率">{g}<line class="xh" y1="{T}" y2="{H - B}"/>{hits}</svg><div class="tip"></div></div>')


END_COL = {"第一幕": "var(--e1)", "第二幕": "var(--e2)", "第三幕其他": "var(--e3)", "第三幕 Boss": "var(--e4)", "卡住": "var(--s2)", "通关": "var(--s3)"}


def stack_chart(summ: dict, full: list[int], target: int, empty: str) -> str:
    """各进阶对局停在哪个阶段（占比，横向堆叠）。计划内还没有数据的进阶也占一行。"""
    S = summ["ascensions"]
    used = [k for k in ENDINGS if any(S[a]["endings"][k] for a in S)]
    rows = []
    for a in full:
        s = S.get(a)
        if not s:
            rows.append(f'<div class="srow na"><span class="sl">{a}</span><div class="sbar e">{empty}</div><span class="sn">—</span><span class="sv">—</span></div>')
            continue
        n = s["n"]
        segs = "".join(f'<i style="width:{100 * s["endings"][k] / n:.2f}%;background:{END_COL[k]}" title="进阶 {a} · {k}：{s["endings"][k]:,} 局（{pct(s["endings"][k] / n)}）"></i>'
                       for k in used if s["endings"][k])
        part = target and n < target
        rows.append(f'<div class="srow"><span class="sl">{a}</span><div class="sbar">{segs}</div>'
                    f'<span class="sn{" pt" if part else ""}">{n:,}</span><span class="sv">{pct(s["endings"]["通关"] / n)}</span></div>')
    legend = "".join(f'<span class="lg"><i style="background:{END_COL[k]}"></i>{"死在" + k if k not in ("通关", "卡住") else k}</span>' for k in used)
    return (f'<div class="legend">{legend}</div><div class="stack"><div class="srow sh"><span class="sl">进阶</span><div></div><span class="sn">局数</span><span class="sv">通关</span></div>'
            f'{"".join(rows)}</div>' + ('<p class="note">* 未打满计划局数，占比还会变化。</p>' if any(target and S[a]["n"] < target for a in S) else ""))


def tbl(head: str, body: str, cls: str = "cmp") -> str:
    """表格放进可横向滚动的容器（手机上不撑宽整页）。"""
    return f'<div class="tw"><table class="{cls}"><thead>{head}</thead><tbody>{body}</tbody></table></div>'


# ---------------------------------------------------------------- 单个进阶的各节

def sec_deaths(rs: list[dict], top: int = 10) -> str:
    n = len(rs)
    c = Counter()
    for r in rs:
        if r["win"] or r["stuck"]:
            continue
        cs = r.get("combats") or []
        last = cs[-1] if cs and cs[-1]["floor"] == r["floor"] else None
        c[f"{last['act']}:{last['room']}:{last['encounter']}" if last else "非战斗"] += 1
    dead = sum(c.values())
    rows = []
    for k, v in c.most_common(top):
        name, where = D.enc_name(k) if k != "非战斗" else ("非战斗", "")
        rows.append(f'<tr><td class="t">{esc(name)}</td><td class="t m">{esc(where)}</td><td>{v:,}</td><td>{pct(v / n)}</td>'
                    f'<td class="bc"><div class="bar"><i style="width:{100 * v / dead:.1f}%"></i></div></td></tr>')
    rest = dead - sum(v for _, v in c.most_common(top))
    if rest:
        rows.append(f'<tr class="sub"><td class="t">其余 {len(c) - top} 种</td><td></td><td>{rest:,}</td><td>{pct(rest / n)}</td><td></td></tr>')
    wins, stuck = sum(r["win"] for r in rs), sum(r["stuck"] for r in rs)
    if not dead:
        return f'<p class="empty">这个进阶没有死亡的对局（通关 {wins:,}、卡住 {stuck:,}）。</p>'
    return (tbl('<tr><th class="t">遭遇</th><th class="t">位置</th><th>局数</th><th>占全部对局</th><th class="t">占死亡局</th></tr>', "".join(rows))
            + f'<p class="note">共 {n:,} 局：死亡 {dead:,}、卡住 {stuck:,}、通关 {wins:,}。死因记为死时所在的那场战斗。</p>')


def sec_boss(rs: list[dict]) -> str:
    t: dict[tuple, list] = defaultdict(lambda: [0, 0, 0.0])
    for r in rs:
        for act, k, enc, won, hp in boss_fights(r):
            x = t[(act, k, enc)]; x[0] += int(won); x[1] += 1; x[2] += hp
    if not t:
        return '<p class="empty">这个进阶还没有打到 Boss 的记录。</p>'
    groups = sorted({(a, k) for a, k, _ in t})
    rows = []
    for a, k in groups:
        label = f"第{ACT_CN[a]}幕" + ("第二个 Boss" if k else "")
        keys = sorted((x for x in t if x[:2] == (a, k)), key=lambda x: -t[x][1])
        fights = sum(t[x][1] for x in keys); wins = sum(t[x][0] for x in keys)
        rows.append(f'<tr class="grp"><td class="t" colspan="2">{label}</td><td>{fights:,}</td><td>{pct(wins / fights, 0)}</td><td></td></tr>')
        for x in keys:
            w, f, hp = t[x]
            rows.append(f'<tr><td class="t">{esc(D.enc_name(f"{a}:Boss:{x[2]}")[0])}</td><td class="bc"><div class="bar"><i style="width:{100 * w / f:.1f}%"></i></div></td>'
                        f'<td>{f:,}</td><td><b>{pct(w / f, 0)}</b></td><td>{pct(hp / f, 0)}</td></tr>')
    return (tbl('<tr><th class="t">Boss</th><th></th><th>场数</th><th>胜率</th><th>进场血量</th></tr>', "".join(rows))
            + '<p class="note">场数为打到这个 Boss 的次数；胜率为其中打赢的比例；进场血量为开打时占最大生命的比例（平均）。灰色行是这一幕各 Boss 合计。卡住的局里最后那场不计。</p>')


def sec_build(rs: list[dict], asc: int) -> str:
    bd = build_at(rs, asc)
    if not bd:
        return '<p class="empty">还没有打到 Boss 的记录。</p>'
    rows = "".join(f'<tr><td class="t">{esc(k)}</td><td>{x["n"]:,}</td><td>{x["deck"]:.1f}</td><td>{x["relics"]:.1f}</td>'
                   f'<td>{x["upgrades"]:.1f}</td><td>{x["potions"]:.1f}</td><td>{pct(x["hp"], 0)}</td></tr>' for k, x in bd.items())
    return (tbl('<tr><th class="t">打到</th><th>局数</th><th>牌数</th><th>遗物</th><th>升级</th><th>药水</th><th>进场血量</th></tr>', rows)
            + '<p class="note">打到这个 Boss 的局，开打时的平均值。</p>')


def sec_habits(rs: list[dict]) -> str:
    h = habits(rs)
    f2 = lambda v: "—" if v is None else f"{v:.2f}"
    groups = [("路线（每局）", [("精英战", f2(h["elites"])), ("商店", f2(h["shops"])), ("问号房", f2(h["unknown"])), ("休息处", f2(h["rests"]))]),
              ("休息处的选择", [("升级", pct(h["smith"], 0)), ("休息", pct(h["heal"], 0)), ("其他", pct(h["rest_other"], 0))]),
              ("商店（每局）", [("买牌", f2(h["shop_card"])), ("买遗物", f2(h["shop_relic"])), ("买药水", f2(h["shop_potion"])), ("删牌", f2(h["shop_remove"]))]),
              ("选牌", [("整个界面跳过", pct(h["card_skip"], 0))])]
    blocks = "".join(f'<div class="hb"><div class="ah"><b>{g}</b></div>' + "".join(f'<div class="hr"><span>{k}</span><b>{v}</b></div>' for k, v in items) + "</div>"
                     for g, items in groups)
    return f'<div class="hgrid">{blocks}</div><p class="note">“每局”按这一进阶的全部对局平均，死得早的局次数自然少；休息处的比例按在休息处做的选择计。</p>'


def sec_cards(rs: list[dict], info: dict, cn: dict, top_n: int = 15) -> str:
    cards = merge(r.get("cards") for r in rs)
    scr = cards.pop("_SCREEN", [0] * 6)
    screens = sum(scr[:3])
    if not screens:
        return '<p class="empty">还没有选牌记录。</p>'
    own = lambda k: not info or info.get(k, ("", ""))[0] in ("DEFECT_CARD_POOL", "COLORLESS_CARD_POOL")
    floor_n = max(20, round(0.005 * screens))
    act_floor = 10
    items = []
    for k, v in cards.items():
        off, pk = sum(v[:3]), sum(v[3:])
        if off >= floor_n and own(k):
            items.append((pk / off, off, k, v))
    items.sort(key=lambda x: (-x[0], -x[1]))
    m = min(top_n, len(items) // 2)

    def rar(k):
        pool, r = info.get(k, ("", ""))
        return ("无色 · " if pool == "COLORLESS_CARD_POOL" else "") + RARITY_CN.get(r, r)

    def rows(sel):
        out = []
        for p, off, k, v in sel:
            acts = "".join(f'<td>{pct(v[3 + a] / v[a], 0)}</td>' if v[a] >= act_floor else '<td class="m">—</td>' for a in range(3))
            out.append(f'<tr><td class="t">{esc(cn.get(k) or k.replace("_", " ").title())}</td><td class="t m">{esc(rar(k))}</td>'
                       f'<td>{off:,}</td><td><b>{pct(p, 0)}</b></td>{acts}</tr>')
        return "".join(out)
    head = ('<tr><th class="t" rowspan="2">牌</th><th class="t" rowspan="2">稀有度</th><th rowspan="2">出现次数</th><th rowspan="2">选取率</th>'
            '<th colspan="3" class="c">各幕选取率</th></tr><tr><th>一</th><th>二</th><th>三</th></tr>')
    scr_rows = "".join(f'<tr><td class="t">第{ACT_CN[a]}幕</td><td>{scr[a]:,}</td><td>{scr[a] - scr[3 + a]:,}</td><td>{scr[3 + a]:,}</td><td>{pct(ratio(scr[3 + a], scr[a]), 0)}</td></tr>'
                       for a in range(3) if scr[a])
    scr_rows += f'<tr class="tot"><td class="t">合计</td><td>{screens:,}</td><td>{screens - sum(scr[3:]):,}</td><td>{sum(scr[3:]):,}</td><td>{pct(ratio(sum(scr[3:]), screens), 0)}</td></tr>'
    lists = (f'<div class="two"><div><h4>最常拿的 {m} 张</h4>{tbl(head, rows(items[:m]))}</div>'
             f'<div><h4>最少拿的 {m} 张</h4>{tbl(head, rows(sorted(items, key=lambda x: (x[0], -x[1]))[:m]))}</div></div>') if m else '<p class="empty">出现次数够多的牌还太少，暂不列表。</p>'
    return (f'<h4>选牌界面</h4>{tbl("<tr><th class=t>幕</th><th>界面数</th><th>拿了一张</th><th>整个跳过</th><th>跳过比例</th></tr>", scr_rows, "cmp narrow")}'
            f'{lists}<p class="note">出现次数：这张牌在选牌界面里出现的次数；选取率：出现后被选中的比例。只列出现不少于 {floor_n:,} 次的牌'
            f'（这一进阶选牌界面数的 0.5%，至少 20 次），不含其他角色的牌；某一幕出现不到 {act_floor} 次时该幕记为“—”。</p>')


def sec_ancient(rs: list[dict]) -> str:
    anc = merge(r.get("ancient") for r in rs)
    ascr = anc.pop("_SCREEN", [0] * 6)
    if not sum(ascr[:3]) and not anc:
        return '<p class="empty">还没有古神遗物的记录。</p>'
    groups: dict[str, list] = defaultdict(list)
    for k, v in anc.items():
        groups[D.ancient_of(k)].append((k, sum(v[:3]), sum(v[3:])))
    order = sorted(groups, key=lambda g: (0 if g == "NEOW" else 1, -sum(o for _, o, _ in groups[g])))
    blocks = []
    for g in order:
        its = sorted(groups[g], key=lambda x: (-(x[2] / x[1] if x[1] else 0), -x[1]))
        rows = "".join(f'<div class="arow" title="{esc(D.relic_tip(k))}"><span class="an">{esc(D.relic_cn(k))}</span><div class="bar"><i style="width:{100 * p_ / o if o else 0:.0f}%"></i></div>'
                       f'<span class="av">{pct(p_ / o if o else None, 0)}<span class="m"> {p_:,}/{o:,}</span></span></div>' for k, o, p_ in its)
        blocks.append(f'<div class="ablk"><div class="ah"><b>{esc(D.ancient_cn(g))}</b><span class="m">出现 {sum(o for _, o, _ in its):,} 次</span></div>{rows}</div>')
    return (f'<div class="ancgrid">{"".join(blocks)}</div><p class="note">这一进阶共 {sum(ascr[:3]):,} 个古神三选一界面，'
            f'其中 {sum(ascr[3:]):,} 次没选遗物（选了其他选项）。每件遗物：选取率 = 被选次数 / 出现次数（右侧小字），三选一时平均约 33%。'
            f'组内按选取率从高到低排列；电脑上鼠标停在遗物上可看效果。</p>')


# ---------------------------------------------------------------- 页面

def build(dirp: Path, notes: list[str], target: int = 0, asc_range: tuple[int, int] = (0, 10)) -> tuple[str, dict]:
    runs = load(dirp)
    total = target * (asc_range[1] - asc_range[0] + 1)
    live = bool(target) and len(runs) < total
    summ = summarize(runs)
    version = runs[0]["version"]
    meta = {}
    try:
        meta = json.loads((dirp / "meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    a_lo, a_hi = asc_range
    full = list(range(a_lo, a_hi + 1))
    S = summ["ascensions"]
    n_per = target or max(s["n"] for s in S.values())
    empty = "尚未开始" if live else "暂无数据"
    by = defaultdict(list)
    for r in runs:
        by[r["asc"]].append(r)
    partial = lambda a: bool(target) and a in S and S[a]["n"] < target

    # 1. 指标卡
    def kpi(label, val, sub):
        return f'<div class="kpi"><div class="kl">{esc(label)}</div><div class="kv">{val}</div><div class="ks">{esc(sub)}</div></div>'
    def nlab(a):
        return f"{S[a]['n']:,} 局" + (f"（进行中，计划 {target:,}）" if partial(a) else "")
    def kpi_win(a):
        s = S.get(a)
        return kpi(f"进阶 {a} 通关率", pct(s["win_rate"]) if s else "—",
                   f"95% 区间 {pct(s['win_ci'][0])}–{pct(s['win_ci'][1])}，{nlab(a)}" if s else empty)
    sh = S.get(a_hi)
    kpis = (kpi_win(a_hi) + kpi_win(a_lo)
            + kpi(f"进阶 {a_hi} 平均层数", f'{sh["floor_mean"]:.1f}' if sh else "—", f"中位 {sh['floor_median']:.0f} 层（满层 49）" if sh else empty)
            + kpi(f"进阶 {a_hi} 过第二幕", pct(sh["reach3"]) if sh else "—", f"过第一幕 {pct(sh['reach2'])}" if sh else empty))

    # 2. 各进阶成绩表（计划内的每个进阶一行）
    trs = []
    for a in full:
        s = S.get(a)
        if not s:
            trs.append(f'<tr class="na"><td>{a}</td><td class="t m" colspan="10">{empty}</td></tr>')
            continue
        ncell = f'{s["n"]:,}' + (f'<span class="m"> / {target:,}</span>' if partial(a) else "")
        trs.append(f'<tr><td>{a}</td><td>{ncell}</td><td><b>{pct(s["win_rate"])}</b></td><td class="m">{pct(s["win_ci"][0])}–{pct(s["win_ci"][1])}</td>'
                   f'<td>{s["floor_mean"]:.1f}</td><td>{s["floor_median"]:.0f}</td><td>{pct(s["reach2"])}</td><td>{pct(s["reach3"])}</td>'
                   f'<td>{pct(s["boss3_pass"])}</td><td>{pct(s["boss3b_pass"]) if s["boss3b_pass"] is not None else "—"}</td><td>{s["stuck"] or ""}</td></tr>')
    table1 = tbl('<tr><th>进阶</th><th>局数</th><th>通关率</th><th>95% 区间</th><th>平均层数</th><th>中位层数</th><th>过第一幕</th><th>过第二幕</th>'
                 '<th>第三幕 Boss<br><span>打到后打赢</span></th><th>第二个 Boss<br><span>打到后打赢</span></th><th>卡住</th></tr>', "".join(trs), "")

    # 3. 各进阶打法对比（只放比例、按进阶各自计算，不合并）
    hrows = []
    for a in full:
        if a not in S:
            hrows.append(f'<tr class="na"><td>{a}</td><td class="t m" colspan="6">{empty}</td></tr>')
            continue
        h = habits(by[a]); b1 = build_at(by[a], a).get("第一幕 Boss")
        ncell = f'{h["n"]:,}' + (f'<span class="m"> / {target:,}</span>' if partial(a) else "")
        el1 = "—" if h["elites1"] is None else f'{h["elites1"]:.2f}'
        deck1 = f'{b1["deck"]:.1f}' if b1 else "—"
        hrows.append(f'<tr><td>{a}</td><td>{ncell}</td><td>{pct(h["card_skip"], 0)}</td><td>{pct(h["smith"], 0)}</td>'
                     f'<td>{el1}</td><td>{pct(h["remove_per_shop"], 0)}</td><td>{deck1}</td></tr>')
    trend = tbl('<tr><th>进阶</th><th>局数</th><th>选牌界面<br><span>整个跳过</span></th><th>休息处<br><span>选升级</span></th>'
                '<th>第一幕精英战<br><span>场 / 局</span></th><th>进商店<br><span>删了牌</span></th><th>第一幕 Boss<br><span>开打时牌数</span></th></tr>', "".join(hrows), "")

    # 4. 分进阶细看：每个进阶预先渲染，按所选进阶显示
    info, cn = card_info(), card_cn()
    secs = {"deaths": {}, "boss": {}, "build": {}, "habits": {}, "cards": {}, "ancient": {}}
    for a in full:
        rs = by.get(a)
        if not rs:
            for k in secs:
                secs[k][a] = f'<p class="empty">进阶 {a} {empty}。</p>'
            continue
        secs["deaths"][a] = sec_deaths(rs); secs["boss"][a] = sec_boss(rs); secs["build"][a] = sec_build(rs, a)
        secs["habits"][a] = sec_habits(rs); secs["cards"][a] = sec_cards(rs, info, cn); secs["ancient"][a] = sec_ancient(rs)

    def status(a):
        if a not in S:
            return f"进阶 {a} {empty}"
        s = S[a]
        base = f"进阶 {a} · {s['n']:,} 局 · 通关 {pct(s['win_rate'])}"
        return base + (f"<em>未打满（计划 {target:,} 局），数字还会变</em>" if partial(a) else "")

    def panels(key):
        warn = lambda a: (f'<p class="warn">进阶 {a} 目前只有 {S[a]["n"]:,} 局（计划 {target:,}），以下数字还会变化。</p>' if partial(a) else "")
        return "".join(f'<div class="ap" data-a="{a}">{warn(a) if key in ("deaths", "cards") else ""}{secs[key][a]}</div>' for a in full)

    has = [a for a in full if a in S]
    default = a_hi if a_hi in S else (has[-1] if has else a_hi)
    btns = "".join(f'<button type="button" data-a="{a}"{" class=nd" if a not in S else ""} aria-pressed="{str(a == default).lower()}" '
                   f'title="进阶 {a}：{(str(S[a]["n"]) + " 局") if a in S else empty}">{a}</button>' for a in full)
    sw = (f'<nav class="sw" aria-label="选择进阶"><div class="swr"><span class="swl">进阶</span><div class="seg">{btns}</div></div>'
          f'<div class="sws">{"".join(f"<div class=ap data-a={a}>{status(a)}</div>" for a in full)}</div></nav>')
    sel_css = "".join(f'main[data-sel="{a}"] .ap[data-a="{a}"]{{display:block}}main[data-sel="{a}"] .seg button[data-a="{a}"]{{background:var(--ink);color:var(--surface);border-color:var(--ink)}}'
                      for a in full)

    notes_html = ("<section><h2>要点</h2><ul class=\"notes\">" + "".join(f"<li>{esc(x)}</li>" for x in notes) + "</ul></section>") if notes else ""
    when = time.strftime("%Y-%m-%d", time.localtime(meta.get("finished") or time.time()))
    prog = ""
    if live:
        per = Counter(r["asc"] for r in runs)
        cells = "".join(f'<span class="pc" title="进阶 {a}：{per.get(a, 0)} / {target} 局"><i style="height:{100 * per.get(a, 0) / target:.0f}%"></i><b>{a}</b></span>'
                        for a in full)
        prog = (f'<div class="live"><span class="dot"></span><b>测试进行中</b>：已完成 {len(runs):,} / {total:,} 局（{pct(len(runs) / total, 0)}），'
                f'{time.strftime("%H:%M", time.localtime())} 更新，页面每 2 分钟自动刷新；数字会随局数增加而变化，未打满的进阶区间更宽。<span class="pcs">{cells}</span></div>')
    body = f"""
<header><div><h1>Defect 模型能力测试报告</h1>
<p class="sub">模型第 {version:,} 版 · 纯贪心 · 进阶 {a_lo}–{a_hi} 各 {target or n_per:,} 局 · {when}</p></div></header>{prog}
<div class="kpis">{kpis}</div>
{notes_html}
<div class="part"><h3>跨进阶对比</h3><p>每个数只来自一个进阶，按进阶并排比较。</p></div>
<section><h2>各进阶成绩</h2><p class="lead">每个进阶用同一组 {n_per:,} 个种子（地图、遭遇的随机数相同，只有进阶不同）。进阶 10 起第三幕连打两个 Boss，进阶 10 以下只有一个。</p>
{line_chart(summ, full, target)}{table1}</section>
<section><h2>止步何处</h2><p class="lead">各进阶对局结束在哪个阶段，按占比堆叠；悬停色块可看局数。</p>{stack_chart(summ, full, target, empty)}</section>
<section><h2>各进阶打法对比</h2><p class="lead">几项能看出打法随难度变化的比例，各进阶分别计算。</p>{trend}
<p class="note">整个跳过 = 选牌界面一张不拿的比例；选升级 = 在休息处选“升级”的比例；第一幕精英战只算打完第一幕的局；删了牌 = 进商店时花钱删牌的比例；牌数为打到第一幕 Boss 的局的平均。</p></section>
<div class="part"><h3>分进阶细看</h3><p>以下各节只含所选进阶的数据。</p></div>
{sw}
<section><h2>死因</h2>{panels("deaths")}</section>
<section><h2>Boss 战</h2>{panels("boss")}</section>
<section><h2>构筑与局外选择</h2><h4>打到各 Boss 时</h4>{panels("build")}<h4>局外选择</h4>{panels("habits")}</section>
<section><h2>抓牌</h2>{panels("cards")}</section>
<section><h2>古神遗物</h2><p class="lead">按古神分组，组内按选取率从高到低。</p>{panels("ancient")}</section>
<section class="method"><h2>方法与数据</h2><dl>
<dt>模型</dt><dd>主线（4090 机）第 {version:,} 版，6 层 Transformer、约 1,105 万个参数；测试时的最新版（这一版所在的 12 次评测是训练以来最好的一段）。</dd>
<dt>打法</dt><dd>纯贪心：每一步都选网络概率最高的动作，不搜索、不随机，同一种子同一进阶的结果可复现。单局超过 6,000 步记为卡住。</dd>
<dt>种子</dt><dd>TEST0000–TEST{n_per - 1:04d}，与训练评测（EVAL000–191）不重叠，各进阶共用。通关率的区间为 Wilson 95% 置信区间。</dd>
<dt>记录</dt><dd>每局一行明细（各场战斗开打时的血量、牌数、遗物、药水，局外选择，选牌、古神遗物，路线）与逐步回放（与对局回放页同一格式），见 {esc(meta.get("archive", "测试目录"))}。</dd>
</dl></section>"""
    page = PAGE.replace("__SELCSS__", sel_css).replace("__ASCS__", json.dumps([str(a) for a in full])).replace(
        "<main>", f'<main data-sel="{default}">').replace("__BODY__", body)
    if live:      # 进行中：每 2 分钟重新载入，保留所选进阶与滚动位置
        page = page.replace("</body>", "<script>try{var y=sessionStorage.getItem('ablY');if(y)scrollTo(0,+y)}catch(e){}"
                            "setTimeout(function(){try{sessionStorage.setItem('ablY',scrollY);sessionStorage.setItem('ablSel',document.querySelector('main').dataset.sel)}catch(e){}location.reload()},120000)</script></body>")
    return page, summ


PAGE = """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>模型能力测试报告</title><style>
:root{color-scheme:light;--page:#f6f6f3;--surface:#fff;--ink:#121212;--ink2:#4f4e4a;--muted:#8a8984;--grid:#e7e6e0;--axis:#c3c2b7;--ring:rgba(0,0,0,.08);
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--e1:#d9d8d2;--e2:#bdbcb4;--e3:#9c9b93;--e4:#6f6e68}
@media (prefers-color-scheme:dark){:root{color-scheme:dark;--page:#0e0e0d;--surface:#191918;--ink:#efeeea;--ink2:#c1c0b8;--muted:#8a8984;--grid:#2b2b29;--axis:#3a3a37;
--ring:rgba(255,255,255,.09);--s1:#3f8ce6;--s2:#e06a37;--s3:#22b383;--e1:#3a3a37;--e2:#55544f;--e3:#75746d;--e4:#a3a29a}}
*{box-sizing:border-box}body{margin:0;background:var(--page);color:var(--ink);font:14px/1.6 system-ui,-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;font-variant-numeric:tabular-nums}
main{max-width:1080px;margin:0 auto;padding:28px 16px 48px}header h1{font-size:24px;margin:0;letter-spacing:.01em}.sub{margin:4px 0 0;color:var(--ink2)}
section{background:var(--surface);border-radius:12px;box-shadow:0 0 0 1px var(--ring);padding:18px 20px;margin-top:16px;min-width:0}
h2{font-size:16px;margin:0 0 6px}h4{font-size:13px;margin:14px 0 6px;color:var(--ink2);font-weight:600}.lead{margin:0 0 10px;color:var(--ink2);font-size:13px}
.part{margin:30px 0 0;display:flex;flex-wrap:wrap;align-items:baseline;gap:2px 12px}.part h3{font-size:18px;margin:0}.part p{margin:0;color:var(--ink2);font-size:13px}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:18px}.kpi{background:var(--surface);border-radius:12px;box-shadow:0 0 0 1px var(--ring);padding:14px 16px;min-width:0}
.kl{font-size:12px;color:var(--ink2);font-weight:600}.kv{font-size:30px;font-weight:700;line-height:1.25}.ks{font-size:12px;color:var(--muted)}
.notes{margin:4px 0 0;padding-left:20px}.notes li{margin:4px 0}
.tw{overflow-x:auto;-webkit-overflow-scrolling:touch;max-width:100%}
table{border-collapse:collapse;width:100%;font-size:13px;margin-top:8px}th,td{padding:5px 8px;text-align:right;border-bottom:1px solid var(--grid);white-space:nowrap}
th{color:var(--ink2);font-weight:500;font-size:12px;vertical-align:bottom}th span,.m{color:var(--muted);font-size:11px;font-weight:400}.t{text-align:left}th.c{text-align:center;border-bottom:none;padding-bottom:0}
tr.grp td{text-align:right;color:var(--muted);font-size:12px;padding-top:10px}tr.grp td.t{text-align:left;color:var(--ink2);font-weight:600}
tr.na td{color:var(--muted)}tr.sub td,tr.tot td{color:var(--ink2)}tr.tot td{font-weight:600;border-bottom:none}
.hgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:4px 28px}.hr{display:flex;justify-content:space-between;gap:8px;font-size:13px;padding:2px 0}.hr b{font-weight:600}table.narrow{width:auto;min-width:min(100%,420px)}
td.bc{width:28%;min-width:70px}td.bc .bar{width:100%}
.two{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:24px}.two>div{min-width:0}.note{font-size:12px;color:var(--muted);margin:8px 0 0}
.empty{color:var(--muted);margin:8px 0}.warn{font-size:12px;color:var(--ink2);margin:0 0 8px;padding:4px 10px;border-left:3px solid var(--s2);background:var(--page);border-radius:0 6px 6px 0}
.legend{display:flex;flex-wrap:wrap;gap:4px 16px;font-size:12px;color:var(--ink2);margin:4px 0 6px}.lg i{display:inline-block;width:12px;height:10px;border-radius:2px;margin-right:6px;vertical-align:-1px}
.lg i.band{background:var(--s3);opacity:.25}.lg i.hollow{width:8px;height:8px;border-radius:50%;border:1.5px solid var(--ink2);vertical-align:0}
.chart{position:relative}.chart svg{width:100%;height:auto;display:block}.gl{stroke:var(--grid)}.ax{font-size:11px;fill:var(--muted)}.dl{font-size:12px;fill:var(--ink2)}
.hit{fill:transparent;cursor:crosshair}.xh{stroke:var(--axis);stroke-dasharray:3 3;display:none}
.tip{position:absolute;pointer-events:none;display:none;background:var(--surface);color:var(--ink);box-shadow:0 0 0 1px var(--ring),0 4px 14px rgba(0,0,0,.14);border-radius:8px;padding:6px 10px;font-size:12px;white-space:nowrap}
.stack{margin-top:4px}.srow{display:grid;grid-template-columns:36px minmax(0,1fr) 48px 52px;align-items:center;gap:10px;margin:3px 0}.srow.sh{font-size:11px;color:var(--muted)}
.sl{text-align:right;font-size:12px;color:var(--ink2)}.sv{font-size:12px;font-weight:600;text-align:right}.sn{font-size:11px;color:var(--muted);text-align:right}.sn.pt::after{content:"*"}
.sbar{display:flex;height:16px;border-radius:4px;overflow:hidden;gap:2px;background:var(--surface)}.sbar i{display:block;height:100%}
.sbar.e{background:var(--grid);opacity:.6;font-size:11px;color:var(--muted);align-items:center;padding-left:8px}.srow.na .sv{color:var(--muted);font-weight:400}
.sw{position:sticky;top:0;z-index:5;margin-top:10px;padding:8px 12px;background:var(--surface);border-radius:12px;box-shadow:0 0 0 1px var(--ring),0 4px 12px rgba(0,0,0,.06)}
.swr{display:flex;align-items:center;gap:10px}.swl{font-size:12px;color:var(--ink2);font-weight:600;flex:none}
.seg{display:flex;flex-wrap:wrap;gap:4px}.seg button{font:inherit;font-size:13px;min-width:32px;height:30px;padding:0 6px;border-radius:6px;border:1px solid var(--grid);background:var(--page);color:var(--ink);cursor:pointer}
.seg button.nd{color:var(--muted);border-style:dashed}.seg button:focus-visible{outline:2px solid var(--s1);outline-offset:1px}
.sws{font-size:12px;color:var(--ink2);margin-top:4px}.sws em{font-style:normal;color:var(--s2);margin-left:8px}
.ap{display:none}__SELCSS__
.ancgrid{column-width:250px;column-gap:28px}.ablk{break-inside:avoid;margin-bottom:12px}.ah{border-bottom:1px solid var(--grid);padding-bottom:2px;margin-bottom:4px;font-size:13px;display:flex;justify-content:space-between;gap:8px}
.arow{display:grid;grid-template-columns:minmax(0,1fr) 56px 96px;gap:8px;align-items:center;font-size:12.5px;padding:1px 0}.an{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.bar{height:7px;background:var(--grid);border-radius:4px;overflow:hidden}.bar i{display:block;height:100%;background:var(--s1);border-radius:4px}.av{text-align:right;font-weight:600}
.live{margin-top:14px;padding:10px 14px;border-radius:10px;background:var(--surface);box-shadow:0 0 0 1px var(--ring);font-size:13px;color:var(--ink2);display:flex;flex-wrap:wrap;align-items:center;gap:6px 10px}
.live b{color:var(--ink)}.dot{width:8px;height:8px;border-radius:50%;background:var(--s3);animation:pl 1.6s infinite}@keyframes pl{50%{opacity:.3}}
.pcs{display:inline-flex;flex-wrap:wrap;gap:3px;margin-left:auto}.pc{position:relative;width:20px;height:26px;border-radius:3px;background:var(--grid);overflow:hidden;display:flex;align-items:flex-end}
.pc i{display:block;width:100%;background:var(--s1);opacity:.55}.pc b{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font-size:10px;color:var(--ink)}
.method dl{margin:0;font-size:13px;color:var(--ink2)}.method dt{font-weight:600;color:var(--ink);margin-top:8px}.method dd{margin:0}
@media (max-width:760px){.kpis{grid-template-columns:1fr 1fr}.two{grid-template-columns:minmax(0,1fr)}.kv{font-size:24px}section{padding:14px}
.sw{padding:8px 10px}.swl{display:none}.seg{gap:3px}.seg button{min-width:26px;height:28px;padding:0 2px;font-size:12px}}
</style></head><body><main>__BODY__</main>
<script>
document.querySelectorAll('.chart').forEach(function(c){var d=JSON.parse(c.dataset.series),tip=c.querySelector('.tip'),xh=c.querySelector('.xh'),svg=c.querySelector('svg');
  function f(v){return (100*v).toFixed(1)+'%'}
  c.querySelectorAll('.hit').forEach(function(h){h.addEventListener('mouseenter',show);h.addEventListener('click',show);
    function show(){var a=h.dataset.a,v=d[a],x=+h.getAttribute('x')+(+h.getAttribute('width'))/2;xh.setAttribute('x1',x);xh.setAttribute('x2',x);xh.style.display='block';
      tip.innerHTML='<b>进阶 '+a+'</b>（'+v[5].toLocaleString()+' 局）<br>过第一幕 '+f(v[0])+'<br>过第二幕 '+f(v[1])+'<br>通关 '+f(v[2])+'（'+f(v[3])+'–'+f(v[4])+'）';tip.style.display='block';
      var k=svg.getBoundingClientRect().width/svg.viewBox.baseVal.width,l=x*k+12;if(l+tip.offsetWidth>c.clientWidth)l=x*k-tip.offsetWidth-12;if(l<0)l=0;tip.style.left=l+'px';tip.style.top='10px'}});
  svg.addEventListener('mouseleave',function(){tip.style.display='none';xh.style.display='none'})});
(function(){var M=document.querySelector('main'),ok=__ASCS__,K='ablAsc',bs=document.querySelectorAll('.seg button');
  function sel(a){M.setAttribute('data-sel',a);bs.forEach(function(b){b.setAttribute('aria-pressed',b.getAttribute('data-a')===a?'true':'false')})}
  var s=null;try{s=sessionStorage.getItem('ablSel');sessionStorage.removeItem('ablSel')}catch(e){}
  if(ok.indexOf(s)<0){try{s=localStorage.getItem(K)}catch(e){s=null}}
  if(ok.indexOf(s)>=0)sel(s);
  bs.forEach(function(b){b.addEventListener('click',function(){var a=b.getAttribute('data-a');sel(a);try{localStorage.setItem(K,a)}catch(e){}})});
})();
</script></body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir"); ap.add_argument("out"); ap.add_argument("--notes")
    ap.add_argument("--target", type=int, default=0, help="每个进阶计划打几局（给了才显示进度）"); ap.add_argument("--asc", default="0-10")
    a = ap.parse_args()
    notes = [l.strip() for l in open(a.notes, encoding="utf-8") if l.strip()] if a.notes else []
    page, summ = build(Path(a.dir), notes, a.target, tuple(map(int, a.asc.split("-"))))
    Path(a.out).write_text(page, encoding="utf-8")
    Path(a.out).with_suffix(".summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    print("写到", a.out)


if __name__ == "__main__":
    main()
