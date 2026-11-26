"""轻量训练看板（2026-11-03，用户：“精简一下看板，轻量化，只需要展示核心训练参数和我们最近关注的指标就行了。
其他的维持统计存在训练机，不要自动回传了”）。取代 tools/dashboard.py（旧看板的代码留着，不再常驻）。

只读 tools/remote/pull_lite.sh 拉回的两份精简统计（tools/remote/lite_stats.py 在租用机上生成）：
  ~/.local/share/sls2/runs/remote/lite_b.json（学习端）、lite_a.json（评测机）
生成一个单页（数据内嵌、每 30 秒自动刷新）：本机 训练看板.html，公开副本 public/index.html（publish.sh 按 stamp.js 推到香港）。
    python3 tools/dashboard_lite.py
看板目标（2026-11-06 起，储君训练）：~/.local/share/sls2/runs/remote/lite_target.json 给出看哪条训练线（pull_lite.sh 按它去哪个部署目录、
哪个运行目录取统计）以及看板上的角色名、标线文件、社区选牌统计、走势图下的说明；没有这个文件时是故障机器人的主线，与原来相同。
"""
from __future__ import annotations

import html
import json
import math
import statistics as S
import time
from pathlib import Path

L = Path.home() / ".local/share/sls2/runs/remote"
OUT = Path("/mnt/c/主存储文件/脚本开发/sls2/训练看板.html")
PUBLIC = L / "public"
# 护栏（后续计划.md 第二节）：与基准相比，打过第二幕 Boss 低 4.5、进第二幕低 6、第一幕 Boss 过关低 6 个百分点，平均层数低 1 层
GUARD = {"boss2": 4.5, "act2": 6.0, "boss1": 6.0, "floor": 1.0}
EVAL_OLD_KEYS = ("win", "act2", "act3", "floor", "floor_med", "pass3a")      # 评测机 evals_old 每点的字段（版本之后），同 lite_stats.EVAL_OLD_KEYS


def jload(p):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def e(x) -> str:
    return html.escape(str(x))


def pct(x, d=1):
    return "—" if x is None else f"{100 * x:.{d}f}%"


def num(x, d=3):
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{d}g}" if abs(x) < 1e-2 and x != 0 else f"{x:,.{d}f}".rstrip("0").rstrip(".")
    return f"{x:,}" if isinstance(x, int) else e(x)


def window(evs: list[dict]) -> dict | None:
    if not evs:
        return None
    m = lambda k: S.mean(r[k] for r in evs if r.get(k) is not None)
    b1 = sum(r["boss"][0][0] for r in evs) / max(1, sum(r["boss"][0][1] for r in evs))
    b2 = S.mean(r["boss"][1][0] / r["runs"] for r in evs)
    d3 = [r["deck_act3"] for r in evs if r.get("deck_act3") and r["deck_act3"][0]]
    db = [r["deck_boss3"] for r in evs if r.get("deck_boss3") and r["deck_boss3"][0]]
    n3 = sum(r.get("n_boss3") or 0 for r in evs); n3b = sum(r.get("n_boss3b") or 0 for r in evs)
    wins = sum(r["win"] * r["runs"] for r in evs)
    return {"n": len(evs), "v0": evs[0]["version"], "v1": evs[-1]["version"], "win": m("win"), "boss1": b1, "boss2": b2,
            "boss3a": n3b / n3 if n3 else None, "boss3b": wins / n3b if n3b else None,
            "act2": m("act2"), "act3": m("act3"), "floor": m("floor"),
            "deck3": S.mean(x[1] for x in d3) if d3 else None, "hp3": S.mean(x[3] for x in d3) if d3 else None,
            "deckb": S.mean(x[1] for x in db) if db else None, "skip": m("skip"), "heal": m("heal"), "smith": m("smith")}


def mark_lines(marks: list[tuple[int, str]], x0: int, x1: int, X, W: float, H: float, pt: float, pb: float) -> list[str]:
    """标线（改动点）：虚线加说明文字。文字挨得近时错开最多 3 行，哪一行都放不下就只画虚线（悬停看文字）；靠右放不下时写在线的左边。"""
    out, row_end = [], [-1e9] * 3
    for v, lab in sorted((m for m in marks if m[0] and x0 <= m[0] <= x1), key=lambda m: m[0]):
        x = X(v)
        out.append(f'<line x1="{x:.0f}" x2="{x:.0f}" y1="{pt}" y2="{H - pb}" class="mk"><title>{e(lab)}（第 {v:,} 版）</title></line>')
        w = sum(11 if ord(ch) > 0x2E80 else 6.2 for ch in lab)
        x_s = x + 4 if x + 4 + w <= W - 2 else x - 4 - w
        r = next((k for k in range(3) if x_s >= row_end[k] + 6), None)
        if r is not None:
            row_end[r] = x_s + w
            out.append(f'<text x="{x_s:.0f}" y="{pt + 10 + 12 * r}" class="ax">{e(lab)}</text>')
    return out


def chart(evs: list[dict], hist: list[dict], marks: list[tuple[int, str]],
          eval_desc: str = "评测（贪心出手，P1；第 165,347 版前 384 局、之后 1,000 种）") -> str:
    """通关率走势（内嵌 SVG）：评测（贪心，P1 384 局）与训练整局（采样出手，20 版滑动平均）两条线；
    竖线标出势能上线与升满，线尾直接标出最新值。配色取数据可视化参考配色的第 1、2 槽（蓝、橙）。"""
    if len(evs) < 2:
        return '<p class="muted">评测记录不足</p>'
    W, H, pl, pr, pt, pb = 760, 250, 40, 58, 16, 28
    # 横轴从两条线都有数据的地方开始（训练走势只取势能上线前 3,000 版起），免得一条线从图中间冒出来
    t_first = min((r["version"] for r in hist if r.get("run_win_avg") is not None), default=None)
    if t_first is not None and sum(r["version"] >= t_first for r in evs) >= 2:
        evs = [r for r in evs if r["version"] >= t_first]
    x0, x1 = min(r["version"] for r in evs), max(r["version"] for r in evs)
    tr = [(r["version"], 100 * r["run_win_avg"]) for r in hist if r.get("run_win_avg") is not None and x0 <= r["version"] <= x1]
    ev = [(r["version"], 100 * r["win"]) for r in evs]
    ys = [y for _, y in ev + tr]
    y0 = max(0, 5 * int((min(ys) - 2) // 5)); y1 = 5 * int((max(ys) + 6) // 5)      # 通关率不会小于 0，纵轴不画负数
    X = lambda v: pl + (v - x0) / max(1, x1 - x0) * (W - pl - pr)
    Y = lambda y: pt + (y1 - y) / max(1, y1 - y0) * (H - pt - pb)
    out = []
    for g in range(y0, y1 + 1, 5):
        out.append(f'<line x1="{pl}" x2="{W - pr}" y1="{Y(g):.0f}" y2="{Y(g):.0f}" class="gl"/>'
                   f'<text x="{pl - 8}" y="{Y(g) + 4:.0f}" class="ax" text-anchor="end">{g}%</text>')
    out.extend(mark_lines(marks, x0, x1, X, W, H, pt, pb))
    for v in (x0, (x0 + x1) // 2, x1):
        out.append(f'<text x="{X(v):.0f}" y="{H - 8}" class="ax" text-anchor="middle">第 {v:,} 版</text>')
    ends = []
    def series(pts, cls, name, hover):
        if len(pts) < 2:
            return ""
        line = " ".join(f"{X(v):.0f},{Y(y):.0f}" for v, y in pts)
        hit = "".join(f'<circle cx="{X(v):.0f}" cy="{Y(y):.0f}" r="7" class="hit"><title>{name}，第 {v:,} 版：{y:.1f}%</title></circle>'
                      for v, y in pts) if hover else ""
        lv, ly = pts[-1]
        ends.append([Y(ly) + 4, X(lv) + 8, f"{ly:.1f}%"])
        return f'<polyline points="{line}" class="ln {cls}"/><circle cx="{X(lv):.0f}" cy="{Y(ly):.0f}" r="4" class="dot {cls}"/>{hit}'
    out.append(series(tr, "s2", "训练整局", False))
    out.append(series(ev, "s1", "评测", True))
    ends.sort()                     # 两条线的线尾数值靠得太近时上下错开（同 multi_chart）
    for k in range(1, len(ends)):
        if ends[k][0] - ends[k - 1][0] < 13 and abs(ends[k][1] - ends[k - 1][1]) < 60:
            ends[k][0] = ends[k - 1][0] + 13
    out.extend(f'<text x="{x:.0f}" y="{y:.0f}" class="lab">{t}</text>' for y, x, t in ends)
    legend = (f'<div class="legend"><span><i class="sw s1"></i>{e(eval_desc)}</span>'
              '<span><i class="sw s2"></i>训练整局（采样出手，20 版滑动平均）</span></div>')
    return legend + f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="评测与训练通关率走势">{"".join(out)}</svg>'


def multi_chart(series: list[tuple[list, str, str, bool]], marks: list[tuple[int, str]], aria: str, unit: str = "", digits: int = 1,
                refs: list[tuple[float, str]] = (), step: float | None = None, y0: float | None = 0, y_cap: float | None = None,
                y_top: float | None = None) -> str:
    """按版本画几条线（内嵌 SVG，样式同 chart）：series 为 [(点 [(版本, 值)], 配色槽, 图例, 悬停)]；refs 为横向参考线 [(值, 文字)]（如各幕 Boss 所在层）。
    横轴从最早有数据的点到最新；纵轴从 y0 起（None 为按数据最小值向下取整），上限按数据与参考线取整、不超过 y_cap。线尾标最新值。"""
    pts_all = [p for ps, _, _, _ in series for p in ps]
    if len(pts_all) < 2:
        return '<p class="muted">记录还不够</p>'
    W, H, pl, pr, pt, pb = 760, 250, 44, 58, 16, 28
    x0, x1 = min(v for v, _ in pts_all), max(v for v, _ in pts_all)
    hi = max([y for _, y in pts_all] + [y for y, _ in refs if y <= max(y for _, y in pts_all) * 1.3])
    st = step or next(s for s in (1, 2, 5, 10, 20, 25, 50) if hi / s <= 6)
    lo = min(y for _, y in pts_all)
    y0 = st * math.floor(lo / st) if y0 is None else y0
    y1 = st * (int(hi // st) + 1)
    if y_top is not None:           # 纵轴至少画到这里（给图顶的标线说明留空，2026-11-11 熵的全程图）
        y1 = max(y1, y_top)
    if y_cap is not None:
        y1 = min(y1, y_cap)
    X = lambda v: pl + (v - x0) / max(1, x1 - x0) * (W - pl - pr)
    Y = lambda y: pt + (y1 - y) / max(1e-9, y1 - y0) * (H - pt - pb)
    out = []
    g = y0
    while g <= y1 + 1e-9:
        out.append(f'<line x1="{pl}" x2="{W - pr}" y1="{Y(g):.0f}" y2="{Y(g):.0f}" class="gl"/>'
                   f'<text x="{pl - 8}" y="{Y(g) + 4:.0f}" class="ax" text-anchor="end">{g:g}{unit}</text>')
        g += st
    for y, lab in refs:
        if y0 < y < y1:
            out.append(f'<line x1="{pl}" x2="{W - pr}" y1="{Y(y):.0f}" y2="{Y(y):.0f}" class="mk"/>'
                       f'<text x="{pl + 4}" y="{Y(y) - 4:.0f}" class="ax">{e(lab)}</text>')
    out.extend(mark_lines(marks, x0, x1, X, W, H, pt, pb))
    for v in (x0, (x0 + x1) // 2, x1):
        out.append(f'<text x="{X(v):.0f}" y="{H - 8}" class="ax" text-anchor="middle">第 {v:,} 版</text>')
    ends = []
    for ps, cls, name, hover in series:
        if len(ps) < 2:
            continue
        out.append(f'<polyline points="{" ".join(f"{X(v):.0f},{Y(y):.0f}" for v, y in ps)}" class="ln {cls}"/>')
        if hover:
            out.append("".join(f'<circle cx="{X(v):.0f}" cy="{Y(y):.0f}" r="7" class="hit"><title>{e(name)}，第 {v:,} 版：{y:.{digits}f}{unit}</title></circle>' for v, y in ps))
        lv, ly = ps[-1]
        out.append(f'<circle cx="{X(lv):.0f}" cy="{Y(ly):.0f}" r="4" class="dot {cls}"/>')
        ends.append([Y(ly) + 4, X(lv) + 8, f"{ly:.{digits}f}{unit}"])
    ends.sort()                     # 线尾数值靠得太近时上下错开，免得叠在一起
    for k in range(1, len(ends)):
        if ends[k][0] - ends[k - 1][0] < 13 and abs(ends[k][1] - ends[k - 1][1]) < 60:
            ends[k][0] = ends[k - 1][0] + 13
    out.extend(f'<text x="{x:.0f}" y="{y:.0f}" class="lab">{t}</text>' for y, x, t in ends)
    legend = '<div class="legend">' + "".join(f'<span><i class="sw {cls}"></i>{e(name)}</span>' for ps, cls, name, _ in series if len(ps) >= 2) + "</div>"
    return legend + f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="{e(aria)}">{"".join(out)}</svg>'


def floor_section(evs: list[dict], hist: list[dict], marks: list[tuple[int, str]], eval_desc: str) -> str:
    """到达层数走势（2026-11-06，用户：“现在还处于训练早期，你可以把训练和评测的到达层数之类的东西，弄一张图给我在看板上看。”）：
    评测的平均与中位到达层数、训练整局的平均层数（20 版滑动平均），参考线为第一、二幕 Boss 所在层（17、33）；
    第二张为评测里打过各个 Boss 关口的局占全部局的比例：第一幕 Boss（进第二幕）、第二幕 Boss（到第三幕）、第三幕第一个 Boss、
    第三幕第二个 Boss（通关）（2026-11-07，用户：“打过各幕 Boss 的比例改为仅展示评测数据，然后加入第三幕第一个BOSS和第三幕第二个BOSS的曲线。”）。"""
    ev_f = [(r["version"], r["floor"]) for r in evs if r.get("floor") is not None]
    ev_m = [(r["version"], r["floor_med"]) for r in evs if r.get("floor_med") is not None]
    tr_f = [(r["version"], r["run_floor_avg"]) for r in hist if r.get("run_floor_avg") is not None]
    a = multi_chart([(ev_f, "s1", f"{eval_desc}：平均到达层数", True), (ev_m, "s3", f"{eval_desc}：中位数", True),
                     (tr_f, "s2", "训练整局（采样出手，20 版滑动平均）：平均层数", False)],
                    marks, "评测与训练的到达层数走势", unit=" 层", refs=[(17, "第一幕 Boss（第 17 层）"), (33, "第二幕 Boss（第 33 层）")], step=5)
    def p3a(r):     # 最近的完整记录没有 pass3a，按打到第二个 Boss 的局数算；更早的精简走势里已算好
        if "pass3a" in r:
            return r["pass3a"]
        return r["n_boss3b"] / r["runs"] if r.get("n_boss3b") is not None and r.get("runs") else None
    col = lambda f: [(r["version"], 100 * x) for r in evs if (x := f(r)) is not None]
    b = multi_chart([(col(lambda r: r.get("act2")), "s1", "打过第一幕 Boss（进第二幕）", True),
                     (col(lambda r: r.get("act3")), "s2", "打过第二幕 Boss（到第三幕）", True),
                     (col(p3a), "s3", "打过第三幕第一个 Boss", True),
                     (col(lambda r: r.get("win")), "s4", "打过第三幕第二个 Boss（通关）", True)], marks, "评测打过各个 Boss 的比例走势", unit="%", step=10)
    return (f'<section class="card"><h2>到达层数走势</h2>{a}<h2 style="margin-top:14px">打过各幕 Boss 的比例（评测，占全部局）</h2>{b}'
            f'<p class="note">{e(eval_desc)}；四条线依次为打过第一幕 Boss、第二幕 Boss、第三幕第一个 Boss、第三幕第二个 Boss 的局占全部局的比例，最后一条就是通关率。</p></section>')


DIAG_STAGES = (("第一幕", (("0:Monster", "普通战"), ("0:Elite", "精英"), ("0:Boss", "Boss"))),
               ("第二幕", (("1:Monster", "普通战"), ("1:Elite", "精英"), ("1:Boss", "Boss"))),
               ("第三幕", (("2:Monster", "普通战"), ("2:Elite", "精英"), ("2:Boss1", "第一个 Boss"), ("2:Boss2", "第二个 Boss"))))
DIAG_SLOTS = (("boss1", "第一幕 Boss", "0:"), ("boss2", "第二幕 Boss", "1:"), ("boss3a", "第三幕第一个 Boss", "2.1:"), ("boss3b", "第三幕第二个 Boss", "2.2:"))


def diag_section(dg: dict | None) -> str:
    """评测的死因与各 Boss 过关率（2026-11-07，用户：“能在看板上加上死因吗？第三幕BOSS分第一个和第二个显示。”
    “然后每一个BOSS的通关率也加上。第三幕BOSS同样分第一个和第二个。”）：评测机 lite_stats.diag() 给的最近 20 轮与再往前 20 轮的合计。"""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import dashboard as D
    cur, prev = (dg or {}).get("cur"), (dg or {}).get("prev")
    if not cur:
        return ""
    def span(w):
        return f'<span class="sub">{w["rounds"]} 轮 {w["runs"]:,} 局 · 第 {w["v0"]:,}–{w["v1"]:,} 版</span>' if w else '<span class="sub">—</span>'
    def share(w, v):
        return v / w["runs"] if w and w["runs"] else None
    def cell(x, n=None):        # 给了 n（局数、场数）时另占一列
        return f'<td>{pct(x)}</td>' + (f'<td class="muted">{n}</td>' if n is not None else "")
    def delta(a, b):
        if a is None or b is None:
            return "<td>—</td>"
        d = round((a - b) * 100, 1)
        return f'<td>{d:+.1f}</td>' if d else "<td>0.0</td>"
    # 死在哪里：各幕普通、精英、Boss，第三幕 Boss 分第一个、第二个；组头一行是这一幕的合计
    st_c, st_p = cur["stages"], (prev or {}).get("stages") or {}
    rows = []
    for act, items in DIAG_STAGES:
        tc = sum(st_c.get(k, 0) for k, _ in items); tp = sum(st_p.get(k, 0) for k, _ in items)
        rows.append(f'<tr class="grp"><th>{act}</th>{cell(share(cur, tc), f"{tc:,}")}{cell(share(prev, tp))}{delta(share(cur, tc), share(prev, tp) if prev else None)}</tr>')
        for k, name in items:
            c, p = st_c.get(k, 0), st_p.get(k, 0)
            rows.append(f'<tr><th>　{name}</th>{cell(share(cur, c), f"{c:,}")}{cell(share(prev, p))}{delta(share(cur, c), share(prev, p) if prev else None)}</tr>')
    for k, name in (("非战斗", "战斗以外"),):
        c, p = st_c.get(k, 0), st_p.get(k, 0)
        rows.append(f'<tr class="grp"><th>{name}</th>{cell(share(cur, c), f"{c:,}")}{cell(share(prev, p))}{delta(share(cur, c), share(prev, p) if prev else None)}</tr>')
    wc, wp = cur["wins"], (prev or {}).get("wins", 0)
    rows.append(f'<tr class="grp"><th>通关</th>{cell(share(cur, wc), f"{wc:,}")}{cell(share(prev, wp))}{delta(share(cur, wc), share(prev, wp) if prev else None)}</tr>')
    t1 = (f'<table class="grid"><tr><th>死在哪里（占全部局）</th><th>最近{span(cur)}</th><th>局数</th><th>再往前{span(prev)}</th><th>变化<span class="sub">个百分点</span></th></tr>'
          + "".join(rows) + "</table>")
    # 具体死因：前 15 项
    def where(key):
        act, room, enc = key.split(":")
        base, _ = D.enc_name(f"{act}:Boss:{enc}" if room.startswith("Boss") else key)
        place = {"Boss1": "第一个 Boss", "Boss2": "第二个 Boss", "Boss": "Boss（第一、二个未分开）" if act == "2" else "Boss",
                 "Monster": "普通战", "Elite": "精英"}.get(room, room)
        return base, f"第{'一二三'[int(act)]}幕 · {place}" if act in "012" else place
    cc, cp = cur["causes"], (prev or {}).get("causes") or {}
    rows = []
    for key, v in list(cc.items())[:15]:
        name, place = where(key) if key.count(":") == 2 else (key, "")
        p = cp.get(key)
        rows.append(f'<tr><th>{e(name)}<span class="sub">{e(place)}</span></th>{cell(share(cur, v), f"{v:,}")}'
                    f'{cell(share(prev, p)) if p is not None else "<td>—</td>"}{delta(share(cur, v), share(prev, p)) if p is not None else "<td>—</td>"}</tr>')
    t2 = (f'<table class="grid"><tr><th>遭遇</th><th>最近 {cur["rounds"]} 轮<span class="sub">占全部局</span></th><th>局数</th><th>再往前 {(prev or {}).get("rounds", 0)} 轮</th>'
          f'<th>变化<span class="sub">个百分点</span></th></tr>{"".join(rows)}</table>')
    # 各 Boss 过关率：四个关口的合计（组头）与关口里的各个 Boss
    bc, bp = cur["bosses"], (prev or {}).get("bosses") or {}
    sc, sp = cur["slots"], (prev or {}).get("slots") or {}
    rate = lambda wn: wn[0] / wn[1] if wn and wn[1] else None
    rows = []
    groups = list(DIAG_SLOTS)
    if any(k.startswith("2:") for k in bc):
        groups.append((None, "第三幕 Boss（第一、二个未分开）", "2:"))
    for slot, name, pre in groups:
        if slot:
            c, p = sc.get(slot), sp.get(slot)
            rows.append(f'<tr class="grp"><th>{name}</th>{cell(rate(c), f"{c[0]:,}/{c[1]:,}") if c else "<td>—</td><td></td>"}'
                        f'{cell(rate(p)) if p else "<td>—</td>"}{delta(rate(c), rate(p))}</tr>')
        else:
            rows.append(f'<tr class="grp"><th>{name}</th><td></td><td></td><td></td><td></td></tr>')
        for k in sorted((k for k in bc if k.startswith(pre)), key=lambda k: -bc[k][1]):
            enc = k.split(":", 1)[1]
            c, p = bc[k], bp.get(k)
            nm = D.enc_name(f"{pre[0]}:Boss:{enc}")[0]
            rows.append(f'<tr><th>　{e(nm)}</th>{cell(rate(c), f"{c[0]:,}/{c[1]:,}")}{cell(rate(p)) if p else "<td>—</td>"}{delta(rate(c), rate(p))}</tr>')
    t3 = (f'<table class="grid"><tr><th>Boss</th><th>最近 {cur["rounds"]} 轮<span class="sub">过关率</span></th><th>打赢 / 打到<span class="sub">场</span></th>'
          f'<th>再往前 {(prev or {}).get("rounds", 0)} 轮</th><th>变化<span class="sub">个百分点</span></th></tr>{"".join(rows)}</table>')
    sv0 = cur.get("split_v0")
    split_note = (f'第三幕各 Boss 分第一、二个的明细从第 {sv0:,} 版那轮起才有（评测程序 2026-11-07 起分开记），最近 {cur["rounds"]} 轮里有 {cur["split"]} 轮；'
                  '更早的轮记在“未分开”里。' if sv0 and cur["split"] < cur["rounds"] else
                  "" if sv0 else "评测程序 2026-11-07 起才把第三幕各 Boss 分第一、二个记，下一轮评测开始有明细，现在先合在“未分开”里。")
    return (f'<section class="card"><h2>死因（评测，贪心，P1 的 384 局）</h2>{t1}<h2 style="margin-top:14px">具体死在哪场（前 15 项）</h2>{t2}'
            f'<h2 style="margin-top:14px">各 Boss 过关率（评测）</h2>{t3}'
            f'<p class="note">最近 {cur["rounds"]} 轮评测合计，与再往前 {(prev or {}).get("rounds", 0)} 轮对照；比例都按局数或场数合计算。'
            '死因按输掉的局最后一场战斗记，卡住的局不算死因。各 Boss 过关率 = 打赢场数 / 打到的场数（卡住的那场不计）；'
            '第三幕两个关口的合计按局算：第一个 = 打到第二个 Boss 的局 / 打到第一个的局，第二个 = 通关局 / 打到第二个的局。'
            f'第三幕每局要连打两个 Boss，同一个 Boss 可能出现在第一个，也可能出现在第二个。{split_note}</p></section>')


def practice_section(pr: dict | None, marks: list[tuple[int, str]]) -> str:
    """对战练习（战斗课程）胜率走势（2026-11-07，用户：“你把对战练习也做一张图，搞到看板上去。”）：
    Boss、精英、普通战各一张，每条线为最近 window 版场数合计的胜率；下面一张表为最近 200 版的合计（线尾数值与表给出读数，不只靠颜色区分）。"""
    if not pr or not pr.get("pts"):
        return ""
    keys, pts = pr["keys"], pr["pts"]
    # 从零训练的头 1,000 版胜率变化太大，会把后面的变化压扁，从第 1,000 版画起；原来最长只画最近 8,000 版，
    # 2026-11-07 用户问“为什么看板上早期的评测数据之类的看不见？”后改为画到头
    v_lo = 1000
    pts = [r for r in pts if r[0] >= v_lo]
    col = lambda k: [(r[0], 100 * r[1 + keys.index(k)]) for r in pts if r[1 + keys.index(k)] is not None]
    w = pr.get("window", 50)
    boss = multi_chart([(col("boss1"), "s1", "第一幕 Boss", True), (col("boss2"), "s2", "第二幕 Boss", True), (col("boss3"), "s3", "第三幕 Boss", True)],
                       marks, "Boss 练习胜率走势", unit="%", step=10, y_cap=100)
    elite = multi_chart([(col("act1_elite"), "s1", "第一幕精英", True), (col("act2_elite"), "s2", "第二幕精英", True),
                         (col("act3_elite"), "s3", "第三幕精英", True), (col("mix_elite"), "s4", "随机构筑精英", True)],
                        marks, "精英练习胜率走势", unit="%", step=10, y0=None, y_cap=100)
    normal = multi_chart([(col("act1_monster"), "s1", "第一幕普通战", True), (col("act2_monster"), "s2", "第二幕普通战", True),
                          (col("act3_monster"), "s3", "第三幕普通战", True), (col("mix_monster"), "s4", "随机构筑普通战", True)],
                         marks, "普通战练习胜率走势", unit="%", step=5, y0=None, y_cap=100)
    names = {"act1_monster": "第一幕普通战", "act1_elite": "第一幕精英", "act2_monster": "第二幕普通战", "act2_elite": "第二幕精英",
             "act3_monster": "第三幕普通战", "act3_elite": "第三幕精英", "mix_monster": "随机构筑普通战", "mix_elite": "随机构筑精英",
             "boss1": "第一幕 Boss", "boss2": "第二幕 Boss", "boss3": "第三幕 Boss"}
    last = pr.get("last") or {}
    tot = sum(last.get("n") or []) or 1
    rows = "".join(f"<tr><th>{names.get(k, k)}</th><td>{pct(wv / nv) if nv else '—'}</td><td>{nv:,}</td><td>{pct(nv / tot)}</td></tr>"
                   for k, wv, nv in zip(keys, last.get("wins") or [], last.get("n") or []))
    table = (f'<table class="grid"><tr><th>练习</th><th>胜率</th><th>场数</th><th>占练习场数</th></tr>{rows}</table>'
             f'<p class="note">最近 200 版（第 {last.get("v0", 0):,}–{last.get("v1", 0):,} 版）的合计。走势图从第 {v_lo:,} 版画起，每个点为到这一版为止最近 {w} 版的场数合计算的胜率，不满 100 场不画。</p>')
    return (f'<section class="card"><h2>对战练习胜率 · Boss</h2>{boss}<h2 style="margin-top:14px">对战练习胜率 · 精英</h2>{elite}'
            f'<h2 style="margin-top:14px">对战练习胜率 · 普通战</h2>{normal}<h2 style="margin-top:14px">最近的练习胜率</h2>{table}</section>')


def scroll_chart(dense: dict, marks: list[tuple[int, str]], series: list[tuple[str, str, str]], aria: str, digits: int = 3, ymax: float | None = None,
                 zero: bool = True, scale: float = 1.0, unit: str = "") -> str:
    """最近 600 版逐版的走势（训练机给的紧凑数组 dense，键见 lite_stats.DENSE）：series 为 [(键, 配色槽 s1/s2, 图例)]。
    画得较宽、可左右滑动（打开时停在最新一端），纵轴固定在左边；竖线标出改动点，线尾标最新值。不放悬停点（会让页面涨到 150 KB）。
    每个改动点（都伴随一次重启）之后的头 5 版不画：重启后最先打完的多是很快输掉的短局，局外目标误差特别大（第 158,637 版曾到 8.6）。"""
    vs = dense.get("v") or []
    skip = lambda v: any(m < v <= m + 5 for m, _ in marks if m)
    # zero：纵轴从 0 起（损失、解释方差）；False 时按数据范围（通关率、熵这类起伏只在一小段里的）。scale、unit：显示时乘的倍数与单位（如百分比）
    pts = {k: [(v, y * scale) for v, y in zip(vs, dense.get(k) or []) if isinstance(y, (int, float)) and not skip(v)] for k, _, _ in series}
    allp = [x for k in pts for x in pts[k]]
    if len(allp) < 4:
        return '<p class="muted">还没有这项记录</p>'
    PX, H, pt, pb, pr, AX = 3, 210, 16, 28, 60, 44        # 每版 3 像素；AX 为左边固定纵轴的宽度
    x0, x1 = min(v for v, _ in allp), max(v for v, _ in allp)
    W = max(600, (x1 - x0) * PX + pr + 8)
    # ymax：纵轴上限，超出的点画在顶边（线尾的数值仍标真实值，加“↑”），好看清下面的变化
    lo, hi = (min(0.0, min(y for _, y in allp)) if zero else min(y for _, y in allp)), max(y for _, y in allp)
    if ymax is not None:
        hi = min(hi, ymax)
    span = max(hi - lo, 1e-6)
    step = next((st for st in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0) if span / st <= 6), 50.0)
    y0, y1 = step * int(lo // step), (ymax if ymax is not None and max(y for _, y in allp) >= ymax else step * (int(hi / step) + 1))
    X = lambda v: 8 + (v - x0) * PX
    Y = lambda y: pt + (y1 - min(y, y1)) / (y1 - y0) * (H - pt - pb)
    ticks = [round(y0 + k * step, 4) for k in range(int(round((y1 - y0) / step)) + 1)]
    dec = 2 if step < 0.1 else 1 if step < 1 else 0
    axis = "".join(f'<text x="{AX - 8}" y="{Y(g) + 4:.0f}" class="ax" text-anchor="end">{g:.{dec}f}{unit}</text>' for g in ticks)
    out = [f'<line x1="0" x2="{W}" y1="{Y(g):.0f}" y2="{Y(g):.0f}" class="gl"/>' for g in ticks]
    for v, lab in marks:
        if v and x0 <= v <= x1:
            out.append(f'<line x1="{X(v):.0f}" x2="{X(v):.0f}" y1="{pt}" y2="{H - pb}" class="mk"/><text x="{X(v) + 4:.0f}" y="{pt + 10}" class="ax">{lab}</text>')
    for v in range(x0 + (-x0) % 50, x1 + 1, 50):          # 每 50 版一个刻度
        out.append(f'<text x="{X(v):.0f}" y="{H - 8}" class="ax" text-anchor="middle">{v:,}</text>')
    for k, cls, _ in series:
        ps = pts[k]
        if len(ps) >= 2:
            out.append(f'<polyline points="{" ".join(f"{X(v)},{Y(y):.0f}" for v, y in ps)}" class="ln {cls}"/>')
            lv, ly = ps[-1]
            up = "↑" if ly > y1 else ""
            out.append(f'<circle cx="{X(lv)}" cy="{Y(ly):.0f}" r="4" class="dot {cls}"/><text x="{X(lv) + 8}" y="{Y(ly) + 4:.0f}" class="lab">{ly:.{digits}f}{unit}{up}</text>')
    legend = ('<div class="legend">' + "".join(f'<span><i class="sw {cls}"></i>{e(lab)}</span>' for k, cls, lab in series)
              + f'<span class="muted">最近 {x1 - x0 + 1} 版，左右滑动查看</span></div>')
    return (legend + f'<div class="scrollchart"><svg class="yaxis" width="{AX}" height="{H}" viewBox="0 0 {AX} {H}">{axis}</svg>'
            f'<div class="scroller"><svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-label="{e(aria)}">{"".join(out)}</svg></div></div>')


def entropy_section(hist: list[dict], dense: dict, marks: list[tuple[int, str]]) -> str:
    """策略熵的走势（2026-11-11，猎人决策记录第 11 条；用户：“这样子扩批的话，我们需要关注一下策略熵是否坍塌。”）：
    上图全程（训练机给的 hist 里 20 版简单滑动平均，抽成约 160 点：ec 战斗步、em 局外步、ep 地图选路步），下图最近 600 版逐版（dense 的 entc、entm、entp）。
    战斗步的熵由全部步的平均熵按局外步占比推算（lite_stats.ent_combat）。纵轴从 0 起，往 0 掉就是坍塌；报警线在巡检 sup_train.py。
    全程图只给战斗步一条线加悬停点（三条都加会让页面多出约 50 KB）。"""
    col = lambda k: [(r["version"], r[k]) for r in hist if isinstance(r.get(k), (int, float))]
    ec, em, ep = col("ec"), col("em"), col("ep")
    # 纵轴每格 0.2、至少画到最高点之上 0.25：标线说明占图顶三行（约纵轴的 17%），开练头几百版战斗步在 0.8–1.0，不留空说明会压在线上
    hi = max([y for _, y in ec + em + ep] or [0.0])
    full = multi_chart([(ec, "s1", "战斗步（推算）", True), (em, "s2", "局外步", False), (ep, "s3", "地图选路步", False)],
                       marks, "策略熵全程走势", digits=3, step=0.2, y0=0, y_top=0.2 * math.ceil((hi + 0.25) / 0.2 - 1e-9))
    recent = scroll_chart(dense, marks, [("entc", "s1", "战斗步（推算）"), ("entm", "s2", "局外步"), ("entp", "s3", "地图选路步")],
                          "策略熵逐版走势", digits=3)
    return (f'<div class="subh">策略熵 · 全程（20 版滑动平均）</div>{full}'
            f'<div class="subh" style="margin-top:14px">策略熵 · 逐版</div>{recent}'
            '<p class="note">战斗步的熵由日志里全部步的平均熵按局外步所占比例推算，是近似值；局外步、地图选路步是日志原值。'
            '巡检在最近 50 版的中位数低于战斗步 0.60、局外步 0.25、地图选路步 0.04 时报警。</p>')


def skip_section(sk: dict | None, marks: list[tuple[int, str]]) -> str:
    """跳过稳赢战斗（猎人决策记录第 14 条，第 24,208 版起）：训练机给的 skip（lite_stats.skip_trend）——第 14 条以来与最近 200 版的累计、
    最近 600 版每点为最近 20 版合计的比例（跳过比例、两类局进第三幕 Boss 的血量、药水头预测对实际）。没开跳过的训练线不给这一节。"""
    if not sk or not (sk.get("dense") or {}).get("v"):
        return ""
    a, l2 = sk.get("all") or {}, sk.get("last200") or {}
    r = lambda x, y: (x / y) if y else None
    hr, hs = r(a.get("bhr", 0), a.get("bnr", 0)), r(a.get("bhs", 0), a.get("bns", 0))
    rows = [("够条件的普通战、精英战里跳过的比例", f'{pct(r(a.get("dn", 0), a.get("el", 0)))}（{num(a.get("dn"))} / {num(a.get("el"))} 场）'),
            ("进第三幕 Boss 时的血量：完全真打 / 允许跳过",
             f'{pct(hr)} / {pct(hs)}（{num(a.get("bnr"))} / {num(a.get("bns"))} 次；差 {(hs - hr) * 100:+.1f} 个百分点，护栏 ±3）' if hr is not None and hs is not None else "—"),
            ("训练整局通关：完全真打 / 允许跳过", f'{pct(r(a.get("rwr", 0), a.get("rnr", 0)))} / {pct(r(a.get("rws", 0), a.get("rns", 0)))}（{num(a.get("rnr"))} / {num(a.get("rns"))} 局）'),
            ("药水头（最近 200 版，真打的够条件战斗）：预测 / 实际用掉",
             f'{pct(r(l2.get("pp", 0), l2.get("pn", 0)))} / {pct(r(l2.get("py", 0), l2.get("pn", 0)))}（{num(l2.get("pn"))} 瓶；护栏差 5 个百分点）'),
            ("卡住或引擎报错", f'{num(a.get("st"))} / {num(a.get("ep"))} 局（护栏 0.1%）'),
            ("每小时整局数（最近 100 版）", num(sk.get("runs_per_hour")))]
    kv = '<table class="kv">' + "".join(f"<tr><th>{e(k)}</th><td>{x}</td></tr>" for k, x in rows) + "</table>"
    d = sk["dense"]
    w = sk.get("win") or 20
    return (f'<section class="card"><h2>跳过稳赢战斗（第 14 条，第 {sk.get("v0", 0):,}–{sk.get("v1", 0):,} 版）</h2>{kv}'
            f'<div class="subh" style="margin-top:14px">跳过的比例 · 逐版（每点为最近 {w} 版合计）</div>'
            f'{scroll_chart(d, marks, [("sk", "s1", "够条件的战斗里跳过的比例")], "跳过比例走势", digits=1, zero=False, scale=100, unit="%")}'
            f'<div class="subh" style="margin-top:14px">进第三幕 Boss 时的血量比例 · 逐版（每点为最近 {w} 版合计）</div>'
            f'{scroll_chart(d, marks, [("b3r", "s1", "完全真打的局"), ("b3s", "s2", "允许跳过的局")], "进第三幕 Boss 血量走势", digits=1, zero=False, scale=100, unit="%")}'
            f'<div class="subh" style="margin-top:14px">药水头校准 · 逐版（每点为最近 {w} 版合计）</div>'
            f'{scroll_chart(d, marks, [("pp", "s1", "预测用掉的比例"), ("py", "s2", "实际用掉的比例")], "药水校准走势", digits=1, zero=False, scale=100, unit="%")}'
            '<p class="note">整局训练里，开战第一帧辅助头预测胜率 ≥ 0.995 的普通战、精英战（黑名单遭遇除外）以 80% 的概率不打，按结果头抽战后血量与药水直接结束；'
            '另有 25% 的整局完全真打，用来对照。Boss 战与常规评测照常真打。任一护栏成立就关掉跳过（巡检 sup_train.py、sup_eval.py；评测护栏为最近 12 轮通关低于 59.5%）。</p></section>')


def drift_dense(dr: dict) -> dict:
    """快照漂移（tools/remote/snap_drift.py）的逐份数据，拼成 scroll_chart 要的紧凑数组：de 与学生 e、dp 与上一份快照的首选不同比例（全部步）。"""
    pts = (dr or {}).get("points") or []
    g = lambda p, k: (p.get(k) or [None])[0]
    return {"v": [p["version"] for p in pts], "de": [g(p, "vs_e") for p in pts], "dp": [g(p, "vs_prev") for p in pts]}


def drift_map(dr: dict) -> str:
    """漂移轨迹地图：最近几十份快照与学生 e 两两首选不同比例的经典多维缩放二维投影，按时间连线。
    点距约等于决策差异；颜色由浅到深表示由旧到新，悬停看版本。"""
    mp = (dr or {}).get("map")
    if not mp or len(mp.get("xy") or []) < 3:
        return '<p class="muted">快照还不够（要两份以上）</p>'
    labs, xy = mp["labels"], mp["xy"]
    xs, ys = [p[0] for p in xy], [p[1] for p in xy]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1e-6)
    W, H, pad = 640, 360, 36
    cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
    k = min(W - 2 * pad, H - 2 * pad) / span
    P = lambda x, y: (W / 2 + (x - cx) * k, H / 2 - (y - cy) * k)
    snaps = [(l, P(*q)) for l, q in zip(labs, xy) if l.startswith("v")]
    refs = {l: P(*q) for l, q in zip(labs, xy) if l == "e"}
    out = []
    # 比例尺：5 个百分点的首选差异有多长
    bar = 0.05 * k
    out.append(f'<line x1="{pad}" x2="{pad + bar:.0f}" y1="{H - 12}" y2="{H - 12}" class="scalebar"/><text x="{pad + bar + 6:.0f}" y="{H - 8}" class="ax">首选差 5 个百分点</text>')
    if len(snaps) >= 2:
        out.append(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for _, (x, y) in snaps)}" class="traj"/>')
    n = len(snaps)
    for i, (l, (x, y)) in enumerate(snaps):
        op = 0.25 + 0.75 * (i + 1) / n
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{6 if i == n - 1 else 4}" class="dot s1" fill-opacity="{op:.2f}"><title>第 {int(l[1:]):,} 版</title></circle>')
    for i in (0, n - 1):
        l, (x, y) = snaps[i]
        out.append(f'<text x="{x + 8:.0f}" y="{y - 8:.0f}" class="lab">{"最早 " if i == 0 else "最新 "}{int(l[1:]):,}</text>')
    names = {"e": "学生 e（起点）"}
    for l, (x, y) in refs.items():
        out.append(f'<rect x="{x - 6:.1f}" y="{y - 6:.1f}" width="12" height="12" class="refpt {l}"><title>{e(names.get(l, l))}</title></rect>'
                   f'<text x="{x + 10:.0f}" y="{y + 4:.0f}" class="lab">{e(names.get(l, l))}</text>')
    legend = (f'<div class="legend"><span><i class="sw s1"></i>定时备份的快照（浅 → 深为旧 → 新，悬停看版本）</span>'
              f'<span><i class="sq e"></i>学生 e</span>'
              f'<span class="muted">二维投影只保留了差异的 {100 * (mp.get("explained") or 0):.0f}%，横纵轴没有固定含义</span></div>')
    return legend + f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="快照漂移轨迹">{"".join(out)}</svg>'


def dense_mean(dense: dict, key: str, v0: int, v1: int | None, marks: list[tuple[int, str]]) -> float | None:
    """逐版紧凑数组（最近 600 版）里 [v0, v1) 这段的平均，跳过改动点后头 5 版（重启后先打完的多是短局）。"""
    xs = [y for v, y in zip(dense.get("v") or [], dense.get(key) or [])
          if isinstance(y, (int, float)) and v >= v0 and (v1 is None or v < v1) and not any(m < v <= m + 5 for m, _ in marks if m)]
    return S.mean(xs) if xs else None


def picks_sections(picks: dict, public: bool = False) -> tuple[str, str, str]:
    """选牌对照、先古遗物选取率（明细与分享图）：训练机给的最近 1,000 版合计（lite_stats.py 的 picks），
    面板沿用旧看板 tools/dashboard.py 的实现（社区数据对照、中文名、图标、分享图绘制）。
    public：公开副本的分享图按网址引用网站上回放页已有的图标（replay/assets/，同源，画布可以导出），不内嵌——
    内嵌约 330 KB，公开副本每 2 分钟经中转推一次；本机文件直接打开时浏览器不许导出引用本地图片的画布，所以本机那份照旧内嵌。"""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import dashboard as D
    tg = target()
    # 游戏版本（2026-11-13 起，愚者在 v0.113.0 上训练）：牌、遗物的中文名与说明按 lite_target.json 的 game_version 读，内容目录按 catalog
    # （主线没有新版本的内容目录时指向 game-v0.113.0 分支的工作树）；没给时是 v0.111.0
    if tg.get("game_version"):
        D.set_game_version(tg["game_version"], tg.get("catalog"))
    # 社区选牌统计：给了文件名就用它；写了 card_stats 但为空（新角色还没有社区数据）时不对照，避免拿故障机器人的数据来比
    if tg.get("card_stats"):
        D.CARD_STATS = Path(__file__).resolve().parents[1] / "engine/card_stats" / tg["card_stats"]
    elif "card_stats" in tg:
        D.CARD_STATS = Path(__file__).resolve().parents[1] / "engine/card_stats" / "（这个角色没有社区数据）"
    if not picks:
        empty = '<p class="muted">训练机还没给出选牌记录</p>'
        return empty, empty, empty
    v0, v1, since = picks.get("v0"), picks["v1"], picks.get("since")
    if since:
        span, v0 = f'{since.get("text", "")} 重新计数', since["version"] + 1
    else:
        span = f"最近 {v1 - v0 + 1:,} 版"
    if not (picks.get("cards") or {}).get("_SCREEN"):
        empty = f'<p class="muted">自 {e(span)} 起还没有新的记录（每 {picks.get("every", 25)} 版记一次）</p>'
        return empty, empty, empty
    card = D.card_panel([], [{"run_cards": picks.get("cards") or {}, "time": 1.0, "version": "@V@"}])
    card = card.replace("最近 1 次更新里训练整局的选牌合计（第 @V@–@V@ 版", f"训练整局的选牌合计（{e(span)}，第 {v0:,}–{v1:,} 版")
    fake = [{"run_ancient": picks.get("ancient") or {}, "time": 1.0, "version": v0}]
    orig = D._icon_uri
    if public:
        def url(path):
            try:
                rel = Path(path).relative_to(D.ASSETS)
            except ValueError:
                return orig(path)
            return f"replay/assets/{rel.as_posix()}" if Path(path).exists() else None
        D._icon_uri = url
    face, rate = D.SHARE_FACE, D.SHARE_RATE
    if public and (D.ASSETS / "share/up_face.jpg").exists():
        D.SHARE_FACE = D.ASSETS / "share/up_face.jpg"      # 头像也放进回放图标目录（replay/assets/share/），公开副本按网址引用
    D.SHARE_RATE = target().get("share_rate") or rate     # 分享图头部的胜率按角色另设（lite_target.json 的 share_rate），没有时沿用故障机器人的
    try:
        img = D.ancient_panel(fake, 0.0, span)
    finally:
        D._icon_uri, D.SHARE_FACE, D.SHARE_RATE = orig, face, rate
    return card, D.ancient_detail(fake, 0.0, span), img


def target() -> dict:
    """看板目标（lite_target.json）：{"name", "label" 角色名, "line" 副标题里的训练线, "markers" 标线文件, "card_stats" 社区选牌统计文件, "share_rate" 遗物分享图头部的胜率,
    "note" 走势图下的说明, "deploy"/"run"/"eval" 给 pull_lite.sh}；没有时为故障机器人主线。"""
    return jload(L / "lite_target.json") or {}


def build(public: bool = False) -> str:
    tgt = target()
    label = tgt.get("label") or "故障机器人"
    b, a = jload(L / "lite_b.json") or {}, jload(L / "lite_a.json") or {}
    mk = jload(L / (tgt.get("markers") or "markers.json")) or {}        # 训练改动点（走势图标线）与三栏的版本范围：{"marks": [{"v", "label"}], "cols": [{"label", "from", "to", "last"}]}
    marks, colspec = mk.get("marks") or [], mk.get("cols") or []
    # 只显示当前主线（markers.json 的 show_from，2026-11-06 用户：“暂时回退的这些中间版本不要再显示了”）：
    # 评测、走势、逐版数组、快照漂移、标线都只取这一版及以后
    v0 = int(mk.get("show_from") or 0)
    marks = [m for m in marks if m["v"] >= v0]
    if v0:
        b = dict(b); a = dict(a)
        a["evals"] = [r for r in a.get("evals") or [] if r.get("version", 0) >= v0]
        a["evals_old"] = [r for r in a.get("evals_old") or [] if r[0] >= v0]
        b["hist"] = [r for r in b.get("hist") or [] if r.get("version", 0) >= v0]
        de = b.get("dense") or {}
        keep = [i for i, v in enumerate(de.get("v") or []) if v >= v0]
        b["dense"] = {k: [x[i] for i in keep] for k, x in de.items()}
        if b.get("drift"):
            b["drift"] = {**b["drift"], "points": [q for q in b["drift"].get("points") or [] if q["version"] >= v0]}
    st, wd, cfg, lt = b.get("status") or {}, b.get("watchdog") or {}, b.get("config") or {}, b.get("latest") or {}
    mcfg = cfg.get("model") or {}
    now = time.time()
    v = st.get("version")
    alive = wd.get("alive") == 1 and not wd.get("stop_flag")
    age = now - (st.get("time") or 0)
    running = alive and age < 600
    evs = [r for r in a.get("evals") or [] if r.get("win") is not None]
    last = evs[-1] if evs else {}
    # 走势图画全程：最近 150 轮的完整记录前面接上更早的精简走势（评测机的 evals_old，相邻几轮已合成一点）；三栏对照与护栏表只用完整记录
    evh = [dict(zip(("version",) + EVAL_OLD_KEYS, r)) for r in a.get("evals_old") or []] + evs
    hrw = [r for r in b.get("hist") or [] if r.get("run_win_avg") is not None]
    tiles = [("当前版本", num(v), ""), ("每秒入训决策数", num(st.get("sps")), ""),
             (("自重新计数起 · 训练通关", pct(b["since"].get("run_win")), f'{b["since"].get("text", "")} 起 {b["since"].get("runs", 0):,} 局')
              if b.get("since") else ("累计决策数", f'{st["total_steps"] / 1e8:.2f} 亿' if st.get("total_steps") else "—", "")),
             ("训练整局通关", pct(hrw[-1]["run_win_avg"]) if hrw else "—", "最近 20 版，采样出手"),
             ("最近一次评测", pct(last.get("win")), f'第 {last["version"]:,} 版' if last else "")]
    kpis = '<div class="kpis">' + "".join(f'<div class="tile"><span class="k">{e(k)}</span><b>{x}</b><span class="muted">{e(sub)}</span></div>'
                                          for k, x, sub in tiles) + "</div>"
    eff = (cfg.get("minibatch") or 0) * (cfg.get("grad_accum") or 1)
    anc = cfg.get("combat_anchor_coef") or 0
    params = [("学习率", num(cfg.get("lr"))), ("每批步数 / 每批学几遍", f'{num(cfg.get("batch_steps"))} / {num(cfg.get("epochs"))}'),
              ("小批量 × 梯度累积", f'{num(cfg.get("minibatch"))} × {num(cfg.get("grad_accum"))} = {eff:,}'),
              ("γ / λ / 局外 λ", f'{num(cfg.get("gamma"))} / {num(cfg.get("lam"))} / {num(cfg.get("macro_lam") or 0.95)}'),
              ("熵系数：战斗 / 局外", f'{num(cfg.get("ent_coef"))} / {num(cfg.get("ent_coef_macro"))}'),
              ("梯度裁剪阈值 / 滑动平均衰减", f'{num(cfg.get("max_grad_norm"))} / {num(cfg.get("ema_decay"))}'),
              *([("锚定（拴在学生 e 上）", f'开：{ {"all": "全部步", "macro": "局外步", "combat": "战斗步"}.get(cfg.get("combat_anchor_rows") or "combat")}，系数 {num(anc)}')] if anc > 0 else []),
              ("整局份额", pct(cfg.get("run_fraction"), 0)),
              ("采集", f'{(tgt.get("train_host") or "b").upper()} {num(cfg.get("actors"))} 进程 × {num(cfg.get("envs_per_actor"))} 引擎'
                       + ("，另有 A 的远程采集" if lt.get("remote_used") else ""))]
    if cfg.get("optimizer") == "muon":      # 隐藏层二维权重用 Muon（猎人、愚者），其余 Adam
        params[0] = ("学习率（Adam，其余参数）", num(cfg.get("lr")))
        params.insert(1, ("Muon 学习率 / 动量（隐藏层二维权重）", f'{num(cfg.get("muon_lr"))} / {num(cfg.get("muon_momentum"))}'))
    reads = [("KL / 裁剪比例", f'{num(lt.get("approx_kl"), 4)} / {pct(lt.get("clip_frac"))}'),
             ("价值损失：全部 / 局外", f'{num(lt.get("loss_v"))} / {num(lt.get("loss_v_macro"))}'),
             ("价值解释方差：战斗 / 局外", f'{num(lt.get("ev_c"))} / {num(lt.get("ev_m"))}'),
             ("熵：全部步 / 局外", f'{num(lt.get("entropy"))} / {num(lt.get("entropy_macro"))}'),
             ("熵：战斗步（推算）/ 地图选路", f'{num(lt.get("ent_c"))} / {num(lt.get("entropy_map"))}'),
             ("训练整局通关（本版）", pct(lt.get("run_win_rate"))),
             ("裁剪前梯度范数 / 超阈值比例", f'{num(lt.get("gn"))} / {pct(lt.get("gcf"))}'),
             ("非有限梯度跳过", num(lt.get("nan_skipped"))), *([("A 远程采集步数", num(lt.get("remote_used")))] if lt.get("remote_used") else [])]
    kv = lambda rows: '<table class="kv">' + "".join(f"<tr><th>{e(k)}</th><td>{x}</td></tr>" for k, x in rows) + "</table>"

    def pick(c):
        rs = [r for r in evs if r["version"] >= (c.get("from") or 0) and (c.get("to") is None or r["version"] < c["to"])]
        return rs[-c["last"]:] if c.get("last") else rs
    cols = [(c["label"], window(pick(c))) for c in colspec[:3]] or [("全部", window(evs))]
    W0, cur = cols[0][1], next((w for _, w in reversed(cols) if w), None)
    def row(name, k, f, guard=None):
        cells = "".join(f"<td>{f(w.get(k) if w else None)}</td>" for _, w in cols)
        if guard is None:
            tag = '<td class="txt muted">只看不判</td>'
        elif not (W0 and cur and cur.get(k) is not None):
            tag = '<td class="txt muted">—</td>'
        else:
            drop = (W0[k] - cur[k]) * (1 if k == "floor" else 100)
            unit = "层" if k == "floor" else "个百分点"
            tag = (f'<td class="txt"><span class="chip bad">● 触发</span></td>' if drop >= guard
                   else f'<td class="txt"><span class="chip ok">● 正常</span><span class="muted"> 低 {guard} {unit}才触发</span></td>')
        return f"<tr><th>{e(name)}</th>{cells}{tag}</tr>"
    head = "".join(f'<th>{e(n)}<span class="sub">{f"{w["n"]} 次 · 第 {w["v0"]:,}–{w["v1"]:,} 版" if w else "—"}</span></th>' for n, w in cols)
    gtab = (f'<table class="grid"><tr><th></th>{head}<th>护栏（对第一栏）</th></tr>'
            + row("第一幕 Boss 过关（打到的局里）", "boss1", pct, GUARD["boss1"]) + row("进第二幕（占全部局）", "act2", pct, GUARD["act2"])
            + row("打过第二幕 Boss（占全部局）", "boss2", pct, GUARD["boss2"])
            + row("第三幕第一个 Boss 过关（打到的局里）", "boss3a", pct) + row("第三幕第二个 Boss 过关（打到的局里）", "boss3b", pct)
            + row("通关率（占全部局）", "win", pct) + row("平均层数", "floor", lambda x: num(x, 2), GUARD["floor"]) + "</table>")
    dense, mks = b.get("dense") or {}, [(m["v"], m["label"]) for m in marks]
    card, anc_detail, anc_img = picks_sections(b.get("picks") or {}, public)
    prog = a.get("progress") or {}
    evstate = "暂停" if a.get("paused") else f'第 {num(prog.get("version"))} 版进行中 {prog.get("done", 0)}/{prog.get("total", 0)}'
    upd = time.strftime("%m-%d %H:%M", time.localtime(max(b.get("time") or 0, a.get("time") or 0) or now))
    state = (f'<span class="chip {"ok" if running else "bad"}">● {"训练中" if running else "停止"}</span>')
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import dashboard as D
    sv = b.get("since") or {}
    note = tgt.get("note") or f"只显示当前主线（第 {v0:,} 版起，从学生 e 重起、按场自举，决策记录 281）；之前回退掉的版本不再显示。评测用原始权重（决策记录 279）。"
    drift = (f'<section class="card"><h2>离学生 e 多远（逐份定时备份）</h2>{scroll_chart(drift_dense(b.get("drift")), mks, [("de", "s1", "与学生 e 首选不同"), ("dp", "s3", "与上一份快照首选不同")], "快照漂移走势", digits=1, ymax=20.0, scale=100, unit="%")}'
             f'\n<p class="note">在 {num((b.get("drift") or dict()).get("n_states"))} 个固定的真实局面上比较首选动作。</p></section>') if b.get("drift") else ""
    reset_btn = (f'<button type="button" class="rst" onclick="doReset(this)" title="训练整局通关率、选牌统计、先古遗物选取率从现在起重新累计">↺ 重新计数</button>'
                 f'<span class="muted"> {"自 " + e(sv.get("text", "")) + " 起累计" if sv else "现在是最近 1,000 版"}</span>')
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>训练看板</title><style>
:root{{color-scheme:light;--bg:#f4f4f2;--card:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--ink3:#8a8984;--muted:#8a8984;--line:#e6e5e1;--grid:#ecebe7;
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--ok:#0ca30c;--bad:#d03b3b;--act1:#7fb2ff;--act2:#b48cff;--act3:#ff8fab}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{color-scheme:dark;--bg:#111110;--card:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;
--ink3:#8f8e86;--muted:#8f8e86;--line:#2c2c2a;--grid:#262624;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500}}}}
:root[data-theme="dark"]{{color-scheme:dark;--bg:#111110;--card:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--ink3:#8f8e86;--muted:#8f8e86;--line:#2c2c2a;--grid:#262624;
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 system-ui,-apple-system,"PingFang SC","Microsoft YaHei",sans-serif}}
main{{max-width:1000px;margin:0 auto;padding:20px 16px 32px}}
header{{display:flex;justify-content:space-between;align-items:flex-end;gap:12px;flex-wrap:wrap;margin-bottom:14px}}
h1{{font-size:20px;font-weight:650;margin:0}}.subtitle{{color:var(--ink2);font-size:13px;margin-top:2px}}.meta{{color:var(--ink2);font-size:12px;text-align:right}}
h2{{font-size:14px;font-weight:600;margin:0 0 10px;color:var(--ink)}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 18px;margin-bottom:14px;overflow-x:auto}}
.kpis{{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-bottom:14px}}@media (max-width:820px){{.kpis{{grid-template-columns:repeat(2,1fr)}}}}
.tile{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 14px;display:flex;flex-direction:column;gap:2px}}
.tile .k{{color:var(--ink2);font-size:12px}}.tile b{{font-size:24px;font-weight:650;font-variant-numeric:tabular-nums;letter-spacing:-.01em}}
.two{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}.two .card{{margin-bottom:0}}.two{{margin-bottom:14px}}@media (max-width:760px){{.two{{grid-template-columns:1fr}}}}
table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}th,td{{padding:7px 8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:middle}}
tr:last-child th,tr:last-child td{{border-bottom:0}}.kv th{{color:var(--ink2);font-weight:400;width:46%}}
.grid th{{font-weight:400;color:var(--ink2)}}.grid tr:first-child th{{font-size:12px;vertical-align:bottom}}.grid td{{text-align:right;white-space:nowrap}}
.grid td.txt{{text-align:left}}.grid tr.grp th{{color:var(--ink);font-weight:600;padding-top:12px}}.sub{{display:block;color:var(--ink3);font-size:11px}}
.muted{{color:var(--ink3);font-size:12px}}.note{{color:var(--ink3);font-size:12px;margin:8px 0 0}}
.chip{{font-size:12px;white-space:nowrap}}.chip.ok{{color:var(--ok)}}.chip.bad{{color:var(--bad);font-weight:600}}
.legend{{display:flex;gap:18px;flex-wrap:wrap;color:var(--ink2);font-size:12px;margin:-2px 0 6px}}.sw{{display:inline-block;width:16px;height:2px;vertical-align:middle;margin-right:6px}}
.sw.s1{{background:var(--s1)}}.sw.s2{{background:var(--s2)}}.sw.s3{{background:var(--s3)}}.sw.s4{{background:var(--s4)}}
.sq{{display:inline-block;width:10px;height:10px;vertical-align:middle;margin-right:6px}}.sq.e,.refpt.e{{fill:var(--ok);background:var(--ok)}}
.refpt{{stroke:var(--card);stroke-width:2}}.traj{{fill:none;stroke:var(--s1);stroke-opacity:.45;stroke-width:1.5}}.scalebar{{stroke:var(--ink3);stroke-width:2}}
.chart{{width:100%;height:auto;display:block}}.gl{{stroke:var(--grid)}}.mk{{stroke:var(--ink3);stroke-dasharray:3 4}}.ax{{fill:var(--ink3);font-size:11px}}
.ln{{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}}.ln.s1{{stroke:var(--s1)}}.ln.s2{{stroke:var(--s2)}}.ln.s3{{stroke:var(--s3)}}.ln.s4{{stroke:var(--s4)}}
.dot{{stroke:var(--card);stroke-width:2}}.dot.s1{{fill:var(--s1)}}.dot.s2{{fill:var(--s2)}}.dot.s3{{fill:var(--s3)}}.dot.s4{{fill:var(--s4)}}.lab{{fill:var(--ink2);font-size:12px;font-variant-numeric:tabular-nums}}
.hit{{fill:transparent}}.hit:hover{{fill:var(--s1);fill-opacity:.25}}
footer{{color:var(--ink3);font-size:12px;margin-top:6px}}
.scrollchart{{display:flex;align-items:flex-start}}.yaxis{{flex:none}}.scroller{{overflow-x:auto;direction:rtl;flex:1}}.scroller svg{{direction:ltr;display:block}}
/* 选牌对照与先古遗物（沿用旧看板的样式） */
.fbar{{height:14px;background:var(--grid);border-radius:7px;overflow:hidden}}.fbar i{{display:block;height:100%;background:var(--s1);border-radius:7px}}
.fn{{display:block;font-size:11px;color:var(--muted);font-weight:400}}.empty{{color:var(--muted);display:flex;align-items:center;justify-content:center}}
.ancacts{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px 28px}}@media (max-width:980px){{.ancacts{{grid-template-columns:1fr}}}}.acol{{min-width:0}}
.acth{{display:flex;justify-content:space-between;align-items:baseline;gap:8px;border-top:3px solid var(--ac);padding-top:6px;margin-bottom:10px}}.acth b{{font-size:15px}}.acth .fn{{display:inline}}
.ablk{{break-inside:avoid;margin-bottom:14px}}.ah{{display:flex;justify-content:space-between;align-items:baseline;gap:8px;border-bottom:1px solid var(--grid);padding-bottom:3px;margin-bottom:4px}}.ah .fn{{display:inline}}
.arow{{position:relative;display:grid;grid-template-columns:minmax(0,1fr) 64px 92px;column-gap:8px;align-items:center;padding:2px 0;font-size:13px}}.arow .fbar{{height:8px}}
.an{{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}.av{{text-align:right;font-weight:600}}.av .fn{{display:inline;margin-left:4px}}
.arow[data-tip]:not([data-tip=""]):hover::after{{content:attr(data-tip);position:absolute;left:0;top:100%;z-index:20;width:max-content;max-width:300px;padding:6px 9px;border-radius:6px;background:var(--card);color:var(--ink);border:1px solid var(--grid);box-shadow:0 4px 14px rgba(0,0,0,.18);font-size:12px;font-weight:400;line-height:1.5;white-space:normal;pointer-events:none}}
.arow:hover .an{{color:var(--s1)}}.tscroll{{max-height:330px;overflow:auto}}.tscroll.tall{{max-height:640px}}.cfull{{margin-top:14px}}
.cfbar{{display:flex;flex-wrap:wrap;align-items:center;gap:6px 10px;margin-bottom:6px}}.cfbar .subh{{margin:0;flex:1 1 auto}}
.cfq{{flex:0 1 220px;min-width:0;font:inherit;font-size:13px;padding:4px 8px;border:1px solid var(--grid);border-radius:6px;background:transparent;color:inherit}}.cfn{{font-size:12px;color:var(--muted)}}
.cfull th{{cursor:pointer;user-select:none;white-space:nowrap}}.cfull th[data-dir="1"]::after{{content:" ▲"}}.cfull th[data-dir="-1"]::after{{content:" ▼"}}
.cg2{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}}.subh{{font-size:13px;font-weight:600;margin:4px 0 6px}}@media (max-width:760px){{.cg2{{grid-template-columns:1fr}}}}
.rst{{font:inherit;font-size:12px;padding:3px 10px;margin-top:4px;border:1px solid var(--line);border-radius:999px;background:var(--card);color:var(--ink);cursor:pointer}}.rst:hover{{border-color:var(--s1);color:var(--s1)}}
.pick table{{font-size:13px}}.pick th,.pick td{{padding:4px 6px}}.pick thead th{{position:sticky;top:0;background:var(--card);font-size:12px;color:var(--ink2);font-weight:400}}
</style></head><body><main>
<header><div><h1>训练看板</h1><div class="subtitle">杀戮尖塔 2 · {e(label)} · 进阶 10 · {e(tgt.get("line") or "强化学习主线")}</div></div>
<div class="meta">{state}<br>更新于 {upd} · 评测：{e(evstate)}<br>{reset_btn}</div></header>
{kpis}
{floor_section(evh, b.get("hist") or [], [(m["v"], m["label"]) for m in marks], tgt.get("eval_desc") or "评测（贪心，P1）")}
{practice_section(b.get("practice"), [(m["v"], m["label"]) for m in marks])}
<section class="card"><h2>通关率走势</h2>{chart([r for r in evh if r.get("win") is not None], b.get("hist") or [], [(m["v"], m["label"]) for m in marks], *([tgt["eval_desc"]] if tgt.get("eval_desc") else []))}
<h2 style="margin-top:14px">训练整局通关（逐版，20 版滑动平均）</h2>{scroll_chart(dense, mks, [("rw", "s2", "训练整局通关（采样出手）")], "训练整局通关走势", digits=1, zero=False, scale=100, unit="%")}
<p class="note">{e(note)}</p></section>
<section class="card"><h2>护栏指标</h2>{gtab}</section>
{diag_section(a.get("diag"))}
<section class="card"><h2>训练健康</h2>
{entropy_section(b.get("hist") or [], dense, mks)}
<div class="subh" style="margin-top:14px">价值与 KL · 逐版</div>
{scroll_chart(dense, mks, [("evm", "s2", "局外价值的解释方差"), ("evc", "s1", "战斗价值的解释方差")], "价值解释方差走势")}
{scroll_chart(dense, mks, [("lvm", "s2", "局外价值损失")], "局外价值损失走势")}
{scroll_chart(dense, mks, [("kl", "s1", "每版近似 KL")], "KL 走势", digits=4)}</section>
{skip_section(b.get("skip"), mks)}
{drift}
<section class="card"><h2>选牌统计 · 与社区对照</h2><div class="pick">{card}</div></section>
<section class="card"><h2>先古遗物选取率</h2>{anc_detail}<h2 style="margin-top:14px">分享图</h2>{anc_img}</section>
<div class="two"><section class="card"><h2>核心训练参数</h2>{kv(params)}</section><section class="card"><h2>最近一版读数</h2>{kv(reads)}</section></div>
<footer>本页只用两台训练机生成的精简统计，完整的训练日志、评测记录与回放留在训练机上。最近备份：{e("、".join(b.get("backups") or []) or "—")}</footer>
</main><script>
{D.ANC_SHARE_JS}
saInit(document);
// 选牌完整列表：搜索（中文、拼音、首字母）与点表头排序
var CF={{q:'',c:1,d:-1}};
function cfHit(s,q){{if(!q)return true;return s.split(' ').some(function(f){{if(f.indexOf(q)>=0)return true;var i=0;for(var j=0;j<f.length&&i<q.length;j++)if(f[j]===q[i])i++;return i===q.length}})}}
function cfDo(root){{var w=root.querySelector('.cfull');if(!w)return;var tb=w.querySelector('tbody'),rs=[].slice.call(tb.rows),q=CF.q.trim().toLowerCase().replace(/\\s+/g,''),n=0;
  w.querySelectorAll('th').forEach(function(th,i){{if(i===CF.c)th.dataset.dir=CF.d;else delete th.dataset.dir}});
  rs.sort(function(a,b){{var x=a.cells[CF.c],y=b.cells[CF.c];if(CF.c===0)return CF.d*x.textContent.localeCompare(y.textContent,'zh');
    var u=x.dataset.v,v=y.dataset.v;if(u===undefined)return v===undefined?0:1;if(v===undefined)return -1;return CF.d*(u-v)}});
  rs.forEach(function(r){{var ok=cfHit(r.dataset.s||'',q);r.hidden=!ok;n+=ok;tb.appendChild(r)}});
  var c=w.querySelector('.cfn');if(c)c.textContent=q?n+' 张':''}}
document.addEventListener('input',function(e){{if(e.target.classList&&e.target.classList.contains('cfq')){{CF.q=e.target.value;cfDo(document)}}}});
document.addEventListener('click',function(e){{var th=e.target.closest&&e.target.closest('.cfull th');if(!th)return;var i=[].indexOf.call(th.parentNode.children,th);
  CF.d=CF.c===i?-CF.d:(i===0?1:-1);CF.c=i;cfDo(document)}});
cfDo(document);
// 重新计数：只在经本机看板服务（tools/dashboard_serve.py，http://localhost:8765/）打开时可用
function doReset(btn){{
  if(location.protocol.indexOf('http')!==0){{window.alert('请用 http://localhost:8765/ 打开看板再按（直接打开文件时按钮连不到本机服务）');return}}
  if(!window.confirm('训练整局通关率、选牌统计、先古遗物选取率从现在起重新累计？'))return;
  btn.disabled=true;btn.textContent='重新计数中…';
  fetch('/reset',{{method:'POST'}}).then(function(r){{return r.json()}}).then(function(d){{
    if(d.ok){{location.reload()}}else{{btn.disabled=false;btn.textContent='↺ 重新计数';window.alert('没有成功：'+d.error)}}
  }}).catch(function(e){{btn.disabled=false;btn.textContent='↺ 重新计数';window.alert('连不到本机看板服务：'+e)}});
}}
// 每 30 秒刷新；正在搜索卡牌或页面在后台时顺延
setInterval(function(){{var a=document.activeElement;if(document.hidden||(a&&a.classList&&a.classList.contains('cfq')&&a.value))return;location.reload()}},30000);
</script></body></html>"""


def main():
    page, pub = build(), build(public=True)
    label = target().get("label")
    if label and label != "故障机器人":     # 旧看板代码里的文字（分享图、社区数据说明）写死了角色名
        page, pub = page.replace("故障机器人", label), pub.replace("故障机器人", label)
    stamp = str(int(time.time()))
    PUBLIC.mkdir(parents=True, exist_ok=True)
    for p, s in ((OUT, page), (PUBLIC / "index.html", pub), (PUBLIC / "data.js", "/* 轻量看板：数据内嵌在 index.html */\n"),
                 (PUBLIC / "stamp.js", stamp)):
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(s, encoding="utf-8"); tmp.replace(p)


if __name__ == "__main__":
    main()
