"""Rust 引擎在第三幕双 Boss 连打上与 C# 是否一致（2026-11-06，用户：“先测 Rust 引擎在这两个 Boss 上和 C# 是否一致”）。

题目取留出验收卷（学生 e 整局里第三幕第一个 Boss 战前的真实快照，连打两场，同 boss3_exam.py），两边用同一个开局规格与种子。
一、同步对拍：学生 e 在 C# 的帧上贪心出手，同一个动作同步发给 Rust；每一步比较两边的帧（头部第 12、14 项与进程有关，不比；
    浮点按 NaN 相等），记第一处不同（段、第几条、记录类型、第几项、两边的值）。候选整数不同（两边可选的动作已对不上）或
    结束状态不同时停止这一题；只有特征不同时继续走，看后面状态会不会分开。
二、各自打：学生 e 在 Rust 上独立贪心打完，与 C# 上同一题同一种子的输赢比较（结局一致率）。

    STS2_RUST_BIN=…/sts2-serve python -I tools/rust_chain_diff.py --exam exam.json --ckpt student_e.pt --out 结果.jsonl [--limit N] [--shard k/n] [--reseed 0]
    python -I tools/rust_chain_diff.py --report 结果.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

from sts2env.env import DEFAULT_DLL, EngineError, _Process  # noqa: E402
from sts2env.protocol import TokenKind, parse  # noqa: E402

VOC = (REPO / "engine" / "catalog" / "vocab.txt").read_text(encoding="utf-8").split("\n")

SEGS = ("tok_ints", "tok_feats", "tok_priv", "cand_ints", "cand_feats")
IGNORE_HEADER = (12, 14)


def first_diff(a, b) -> dict | None:
    """两帧（已解析）的第一处不同；相同返回 None。"""
    ha, hb = a.header.copy(), b.header.copy()
    if len(ha) != len(hb):
        return {"seg": "header_len", "a": len(ha), "b": len(hb)}
    for i in IGNORE_HEADER:
        if i < len(ha):
            ha[i] = hb[i] = 0
    d = np.flatnonzero(ha != hb)
    if len(d):
        return {"seg": "header", "i": int(d[0]), "a": int(ha[d[0]]), "b": int(hb[d[0]])}
    for s in SEGS:
        x, y = getattr(a, s), getattr(b, s)
        if x.shape != y.shape:
            return {"seg": s, "shape": [list(x.shape), list(y.shape)]}
        if x.dtype.kind == "f":
            ne = ~((x == y) | (np.isnan(x) & np.isnan(y)))
        else:
            ne = x != y
        idx = np.argwhere(ne)
        if len(idx):
            r, c = (int(idx[0][0]), int(idx[0][1])) if idx.ndim == 2 and idx.shape[1] == 2 else (int(idx[0][0]), 0)
            kind = int(a.tok_ints[r, 0]) if s.startswith("tok") and r < len(a.tok_ints) else (int(a.cand_ints[r, 0]) if r < len(a.cand_ints) else -1)
            return {"seg": s, "row": r, "col": c, "kind": kind, "a": float(x[r, c]), "b": float(y[r, c]), "n": int(ne.sum())}
    return None


def tok_ref(o, j: int):
    """候选引用的记录（类型、词表 ID、区域）；-1 为没有。"""
    return None if j < 0 or j >= len(o.tok_ints) else tuple(int(x) for x in o.tok_ints[j, :3])


def cand_sem(o) -> list:
    """候选的语义：动作类型、词表 ID、引用的两条记录的（类型、词表 ID、区域），不看记录编号（前面多一条少一条记录不算候选不同）。"""
    return [(int(c[0]), int(c[3]), tok_ref(o, int(c[1])), tok_ref(o, int(c[2]))) for c in o.cand_ints]


def state_sig(o) -> tuple:
    """规则层面的状态签名：决策类型、候选数、血量、最大生命、连打场次，各牌区里的牌（多重集合）、生物的生命 / 最大生命 / 格挡 / 存活、
    充能球、药水。只要这些一致，观测编码上的差异不算规则分歧。"""
    ti, tf = o.tok_ints, o.tok_feats
    h = o.header
    cards = sorted((int(i), int(z)) for k, i, z in ti[:, :3] if k == TokenKind.CARD)
    crea = sorted((int(ti[j, 1]), tuple(int(round(float(x))) for x in tf[j, :4])) for j in range(len(ti)) if ti[j, 0] == TokenKind.CREATURE)
    orbs = sorted(int(i) for k, i in ti[:, :2] if k == TokenKind.ORB)
    pots = sorted(int(i) for k, i in ti[:, :2] if k == TokenKind.POTION)
    return (int(h[1]), int(h[3]), int(h[8]), int(h[9]), int(h[23]) if len(h) > 23 else 0, tuple(cards), tuple(crea), tuple(orbs), tuple(pots))


def sig_diff(a: tuple, b: tuple) -> str:
    names = ("决策类型", "候选数", "血量", "最大生命", "连打场次", "牌", "生物", "充能球", "药水")
    return next((n for n, x, y in zip(names, a, b) if x != y), "")


def vname(i: int) -> str:
    return VOC[i] if 0 <= i < len(VOC) else str(i)


def sig_detail(a: tuple, b: tuple, oc, orr) -> dict:
    """状态签名不同处的具体内容（只在 C# 有的 / 只在 Rust 有的），便于归类。"""
    from collections import Counter as C
    out = {}
    for k, name in ((5, "牌"), (6, "生物"), (7, "充能球"), (8, "药水")):
        x, y = C(a[k]), C(b[k])
        if x != y:
            fmt = (lambda e: (vname(e[0]), e[1])) if k in (5, 6) else vname
            out[name] = {"C#": [fmt(e) for e in (x - y).elements()][:6], "Rust": [fmt(e) for e in (y - x).elements()][:6]}
    if a[1] != b[1] or cand_sem(oc) != cand_sem(orr):
        x, y = C(map(str, cand_sem(oc))), C(map(str, cand_sem(orr)))
        out["候选"] = {"C#": [e for e in (x - y).elements()][:6], "Rust": [e for e in (y - x).elements()][:6]}
    if a[0] != b[0]:
        out["决策类型"] = [a[0], b[0]]
    if a[2:5] != b[2:5]:
        out["血量等"] = [a[2:5], b[2:5]]
    return out


def kind_name(seg: str, kind: int) -> str:
    if seg.startswith("tok"):
        try:
            return TokenKind(kind).name
        except ValueError:
            return str(kind)
    return f"动作{kind}"


def main() -> None:
    import torch
    from sts2rl.actor import MODE_COMBAT
    from sts2rl.features import Ragged, collate_ragged
    from sts2rl.model import ModelConfig, PolicyValueNet, load_state
    ap = argparse.ArgumentParser()
    ap.add_argument("--exam")
    ap.add_argument("--ckpt")
    ap.add_argument("--out")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--reseed", type=int, default=0)
    ap.add_argument("--dll", default=str(DEFAULT_DLL))
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--report", default=None)
    ap.add_argument("--cs-res", default=None, help="C# 上同题同种子的贪心结果（boss3_exam.py run 的输出），报告各自打的结局对比用")
    a = ap.parse_args()
    if a.report:
        cs_res = {json.loads(l)["id"]: json.loads(l)["win"] for l in open(a.cs_res, encoding="utf-8")} if a.cs_res else {}
        return report([json.loads(l) for l in open(a.report, encoding="utf-8") if l.strip()], cs_res)
    torch.set_num_threads(2)
    blob = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    mc = dict(blob["config"]["model"]); mc.pop("oracle", None)
    net = PolicyValueNet(ModelConfig(**mc)).eval(); load_state(net, blob["state"])
    items = json.loads(Path(a.exam).read_text(encoding="utf-8"))["items"]
    k, n = map(int, a.shard.split("/"))
    items = [it for i, it in enumerate(items[: a.limit or None]) if i % n == k]
    done = {json.loads(l)["id"] for l in open(a.out, encoding="utf-8")} if Path(a.out).exists() else set()
    log = open("/dev/null", "wb")
    cs, rs = _Process(Path(a.dll), log), _Process(Path(a.dll), log, engine="rust")

    def act(o) -> int:
        with torch.no_grad():
            lg = net(collate_ragged(Ragged.from_obs([o], np.full(1, MODE_COMBAT))).to("cpu"))[0][0, :o.num_actions]
        return int(lg.argmax())

    def reset(p, spec):
        p.send({"op": "reset", "mode": "combat", "spec": spec}); return parse(p.recv())

    with open(a.out, "a", encoding="utf-8") as fh:
        for it in items:
            if it["id"] in done:
                continue
            spec = {**it["spec"], "seed": f"{it['spec']['seed']}R{a.reseed}"}
            rec = {"id": it["id"], "encounter": it["encounter"], "next": it["next_encounter"]}
            # 一、同步对拍
            try:
                oc, orr = reset(cs, spec), reset(rs, spec)
                t, first, rule, stop = 0, None, None, None
                while True:
                    d = first_diff(oc, orr)
                    if d and first is None:
                        first = {**d, "t": t, "stage": int(oc.header[23]) if len(oc.header) > 23 else 0, "kind_name": kind_name(d["seg"], d.get("kind", -1))}
                    sa, sb = state_sig(oc), state_sig(orr)
                    if sa != sb and rule is None:
                        rule = {"t": t, "what": sig_diff(sa, sb), "stage": int(oc.header[23]) if len(oc.header) > 23 else 0, "hp": [int(oc.header[8]), int(orr.header[8])],
                                "detail": sig_detail(sa, sb, oc, orr)}
                    # 可选的动作（按语义）或结束状态对不上，就不能再按同一个动作同步走
                    if (oc.done, oc.victory, oc.stuck) != (orr.done, orr.victory, orr.stuck):
                        stop = "结束状态不同"
                        break
                    if cand_sem(oc) != cand_sem(orr):
                        stop = "候选不同"
                        break
                    if oc.done or oc.stuck or oc.num_actions == 0 or t >= a.max_steps:
                        break
                    x = act(oc)
                    cs.send({"op": "step", "action": x}); rs.send({"op": "step", "action": x})
                    oc, orr = parse(cs.recv()), parse(rs.recv()); t += 1
                rec.update(steps=t, first=first, rule=rule, stop=stop, cs_win=bool(oc.done and oc.victory), lock_end=bool(oc.done),
                           rs_win_lock=bool(orr.done and orr.victory), stage=int(oc.header[23]) if len(oc.header) > 23 else 0)
            except EngineError as e:
                rec.update(error=str(e)[:300])
                cs, rs = _Process(Path(a.dll), log), _Process(Path(a.dll), log, engine="rust")
            # 二、Rust 上独立贪心打完
            try:
                o, t = reset(rs, spec), 0
                while not (o.done or o.stuck or o.num_actions == 0 or t >= a.max_steps):
                    rs.send({"op": "step", "action": act(o)}); o = parse(rs.recv()); t += 1
                rec.update(rs_win=bool(o.done and o.victory), rs_steps=t)
            except EngineError as e:
                rec.update(rs_error=str(e)[:300])
                rs = _Process(Path(a.dll), log, engine="rust")
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); fh.flush()
    cs.close(); rs.close()


def report(rows: list[dict], cs_res: dict | None = None) -> None:
    ok = [r for r in rows if "error" not in r]
    n = len(ok)
    same_all = sum(1 for r in ok if r["first"] is None)
    print(f"{len(rows)} 题（出错 {len(rows) - n}）；同步对拍全程逐帧一致 {same_all}（{same_all / max(n, 1):.1%}）")
    st = Counter(r["stop"] or ("走完" if r["lock_end"] else "未走完") for r in ok)
    print("  停止原因：", dict(st))
    c = Counter((r["first"]["seg"], r["first"]["kind_name"], r["first"].get("col", r["first"].get("i"))) for r in ok if r["first"])
    print("  第一处不同（段、类型、项）前 15：")
    for key, v in c.most_common(15):
        print(f"    {v:4d}  {key}")
    stage = Counter(r["first"]["stage"] for r in ok if r["first"])
    print("  第一处不同出现在：", {f"第{s + 1}场": v for s, v in sorted(stage.items())})
    rl = [r for r in ok if r.get("rule")]
    print(f"  规则层面（状态签名）全程一致 {n - len(rl)}（{(n - len(rl)) / max(n, 1):.1%}）；第一处不同：", dict(Counter(r["rule"]["what"] for r in rl).most_common()),
          "；出现在", dict(Counter(f"第{r['rule']['stage'] + 1}场" for r in rl)))
    lock = [r for r in ok if r["lock_end"] and r["stop"] is None]
    print(f"  同步走完的 {len(lock)} 题里两边结局相同 {sum(r['cs_win'] == r['rs_win_lock'] for r in lock)}")
    ind = [r for r in rows if "rs_win" in r and cs_res and r["id"] in cs_res]
    if ind:
        cw = sum(bool(cs_res[r["id"]]) for r in ind); rw = sum(r["rs_win"] for r in ind)
        oc = sum(bool(cs_res[r["id"]]) and not r["rs_win"] for r in ind); orr = sum(r["rs_win"] and not cs_res[r["id"]] for r in ind)
        print(f"  各自贪心打（学生 e，同题同种子）：C# 两场都赢 {cw}/{len(ind)}（{cw / len(ind):.1%}），Rust {rw}/{len(ind)}（{rw / len(ind):.1%}）；"
              f"结局相同 {len(ind) - oc - orr}（{(len(ind) - oc - orr) / len(ind):.1%}），只 C# 赢 {oc}、只 Rust 赢 {orr}")


if __name__ == "__main__":
    main()
