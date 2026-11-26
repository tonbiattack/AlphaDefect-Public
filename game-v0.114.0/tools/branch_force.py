"""选牌能力陷阱诊断（2026-11-12，猎人决策记录第 19 条；方案 reports/选牌能力陷阱诊断方案.md）。

在选卡奖励界面上比较“策略原来的选择”与“强拿某一类牌”。两枝重放到同一个界面，用同一个 salt 公平重设玩家看不到的未来
（引擎 Branch.Apply，决策记录 258），再由指定的检查点贪心打完。复用第 258 条的引擎请求，不改 tools/branch_label.py 的语义。

  nodes   底局：贪心打整局、记动作带与每个选卡奖励界面的对数概率；选分叉点：界面上提供了某类目标牌、策略贪心没选这一类，
          每局每类至多 1 个点（按种子与类名定的随机数挑），每点一个 salt。
  run     读分叉点，按臂打完：A 原选；B 强拿（这一类里策略概率最高的那张）；C 强拿之后每个选卡奖励界面都给同类牌的对数几率加 β。
          --cont-ckpt 换接着打的检查点（能力阶梯）。--verify N：另打 N 枝“不分枝、动作带多走一步原选”，应与底局逐局相同。
  report  按类 × 臂 × 检查点汇总配对差、阶梯交互项与 95% 区间，附次要指标；--base 时另给第 0b 步的统计与 β 的标定。

    PY=.venv/bin/python
    $PY -I tools/branch_force.py nodes --ckpt m24721.pt --classes classes.json --seeds-file pool.json --games 2400 --shard 0/6 --out D
    $PY -I tools/branch_force.py run --nodes "D/nodes_s*.jsonl" --base "D/base_s*.jsonl" --cont-ckpt m24721.pt --tag m24721 --arm B --shard 0/6 --out D
    python3 tools/branch_force.py report --dir D [--base]

classes.json：{"S": ["CARD.ACCURACY", …], "Y": […], "K": […]}，名字与词表一致。
每一枝记：输赢、层数、卡住、进第三幕第一个 Boss（第 3 幕第 48 层）时的血量比例、目标类牌整局被直接打出的次数（奇巧自动打出不算）、
打完时牌组里目标类牌的张数、战斗场数、战斗决策数。
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

CARD_REWARD = 3          # DecisionKind 选卡奖励
PLAY = 2                 # 候选类型：出牌（同 tools/card_cf_probe.py）
BOSS3_FLOOR = 48         # 第三幕第一个 Boss 的层数（猎人整局，第 15 条集成测试的逐局记录）
MAX_STEPS = 6000


def reset_req(character: str) -> dict:
    # 截断开关同常规评测（config.cut_flags(evaluating=True)：局面重复截断、20 回合无进展判输），与 tools/remote/silent/ens_test.sh 一致
    return {"op": "reset", "mode": "run", "character": character, "ascension": 10, "route_summary": True,
            "loop_cut": True, "stall_turns": 20}


def load_net(path: str, device: str):
    import torch
    from sts2rl.model import ModelConfig, PolicyValueNet, load_state
    blob = torch.load(path, map_location="cpu", weights_only=False)
    net = PolicyValueNet(ModelConfig(**blob["config"]["model"])).eval().to(device)
    load_state(net, blob["state"])
    return net, blob.get("version")


_VOCAB: list | None = None


def vocab() -> list:
    global _VOCAB
    if _VOCAB is None:
        from sts2rl.catalog import Catalog
        _VOCAB = Catalog().vocab
    return _VOCAB


def name_of(o, j: int) -> str:
    from sts2rl.actor import cand_name
    return cand_name(o, j, vocab())


def shard_of(items: list, spec: str | None) -> list:
    if not spec:
        return items
    k, n = (int(x) for x in spec.split("/"))
    return items[k::n]


def play(net, device, env, reqs: list[dict], character: str, record: bool = False,
         bias: tuple[set, float] | None = None, track: set | None = None) -> list[dict]:
    """在 VecEnv 上并行打一批整局，每个请求是一个 reset（可带 replay、branch、first），之后贪心打完。
    record：记动作带与选卡奖励界面（底局用）。bias=(牌名集合, β)：选卡奖励界面上这些牌的对数几率加 β。
    track：目标类牌名集合，记直接打出次数与牌组里的张数。"""
    import numpy as np
    import torch
    from sts2env.protocol import TokenKind, Zone
    from sts2rl.actor import MODE_RUN
    from sts2rl.features import Ragged, collate_ragged
    voc = vocab()
    track_ids = np.array(sorted(voc.index(n) for n in (track or ()) if n in voc), dtype=np.int64)
    q = deque(enumerate(reqs)); out: list = [None] * len(reqs); slot: dict = {}; obs: dict = {}

    def start(i):
        if not q:
            slot.pop(i, None); return
        k, req = q.popleft()
        slot[i] = {"k": k, "steps": 0, "tape": [], "screens": [], "fights": 0, "cstep": 0, "prev_c": False,
                   "hp48": None, "plays": 0, "deck_cls": None, "deck_n": None}
        env.send(i, req)

    for i in range(len(env)):
        start(i)
    while slot:
        for i, o in env.recv_ready(timeout=1.0):
            if i not in slot:
                continue
            st = slot[i]
            if isinstance(o, Exception) or o.done or o.stuck or st["steps"] > MAX_STEPS:
                win = (not isinstance(o, Exception)) and bool(o.done and o.victory)
                r = {"win": int(win), "floor": int(getattr(o, "floor", 0) or 0),
                     "stuck": int(isinstance(o, Exception) or bool(getattr(o, "stuck", False)) or st["steps"] > MAX_STEPS),
                     "hp48": st["hp48"], "plays": st["plays"], "deck_cls": st["deck_cls"], "deck_n": st["deck_n"],
                     "fights": st["fights"], "cstep": st["cstep"], "steps": st["steps"]}
                if isinstance(o, Exception):
                    r["error"] = str(o)[:300]
                if record:
                    r["tape"] = st["tape"]; r["screens"] = st["screens"]
                out[st["k"]] = r
                start(i); continue
            st["steps"] += 1; obs[i] = o
            ic = bool(o.in_combat)
            if ic and not st["prev_c"]:
                st["fights"] += 1
                if int(o.act) == 2 and int(o.floor) == BOSS3_FLOOR and st["hp48"] is None:
                    st["hp48"] = round(int(o.hp) / max(1, int(o.max_hp)), 4)
            if ic:
                st["cstep"] += 1
            elif track is not None:
                ti = o.tok_ints
                deck = (ti[:, 0] == TokenKind.CARD) & (ti[:, 2] == Zone.DECK)
                st["deck_n"] = int(deck.sum())
                st["deck_cls"] = int((deck & np.isin(ti[:, 1], track_ids)).sum()) if len(track_ids) else 0
            st["prev_c"] = ic
        if not obs:
            continue
        ids = sorted(obs)
        with torch.no_grad():
            lg = net(collate_ragged(Ragged.from_obs([obs[i] for i in ids], np.full(len(ids), MODE_RUN))).to(device))[0]
        lg = lg.float().cpu()
        for k, i in enumerate(ids):
            o = obs[i]; na = o.num_actions; st = slot[i]
            x = lg[k, :na].clone()
            screen = (not o.in_combat) and int(o.decision) == CARD_REWARD and na > 1
            names = [name_of(o, j) for j in range(na)] if screen else None
            if record and screen:
                lp = torch.log_softmax(x, -1).numpy()
                st["screens"].append({"t": len(st["tape"]), "act": int(o.act), "floor": int(o.floor),
                                      "names": names, "lp": [round(float(v), 3) for v in lp]})
            if bias is not None and screen:
                for j, nm in enumerate(names):
                    if nm in bias[0]:
                        x[j] += bias[1]
            act = int(x.argmax())
            if track is not None and o.in_combat and int(o.cand_ints[act, 0]) == PLAY:
                a = int(o.cand_ints[act, 1])
                if 0 <= a < len(o.tok_ints) and voc[int(o.tok_ints[a, 1])] in track:
                    st["plays"] += 1
            if record:
                st["tape"].append(act)
            env.send_step(i, act)
        obs.clear()
    return out


def make_env(a):
    from sts2env.env import VecEnv
    return VecEnv(a.envs, **({"dll": Path(a.dll)} if a.dll else {}))


def cmd_nodes(a) -> None:
    import numpy as np
    import torch
    torch.set_num_threads(2)
    classes = json.loads(Path(a.classes).read_text(encoding="utf-8"))
    net, ver = load_net(a.ckpt, a.device)
    seeds = json.loads(Path(a.seeds_file).read_text(encoding="utf-8"))["seeds"][a.seed_from:a.seed_from + a.games]
    seeds = shard_of(seeds, a.shard)
    tag = a.shard.split("/")[0] if a.shard else "0"
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    env = make_env(a)
    t0 = time.time()
    base = play(net, a.device, env, [{**reset_req(a.character), "seed": s} for s in seeds], a.character, record=True)
    print(f"底局 {len(base)} 局（第 {ver} 版），通关 {np.mean([b['win'] for b in base]):.1%}，卡住 {sum(b['stuck'] for b in base)}，"
          f"用时 {time.time() - t0:.0f} 秒", flush=True)
    nodes = []
    sfx = a.suffix or ""
    with open(out / f"base_s{tag}{sfx}.jsonl", "w", encoding="utf-8") as fb:
        for s, b in zip(seeds, base):
            fb.write(json.dumps({"seed": s, **b}, ensure_ascii=False) + "\n")
            for c, members in classes.items():
                mem = set(members)
                cand = []
                for sc in b["screens"]:
                    lp = sc["lp"]; pol = int(np.argmax(lp))
                    tgt = [j for j, n in enumerate(sc["names"]) if n in mem]
                    if tgt and sc["names"][pol] not in mem:
                        cand.append((sc, pol, max(tgt, key=lambda j: lp[j])))
                if not cand:
                    continue
                h = int(hashlib.sha256(f"{a.rng}:{s}:{c}".encode()).hexdigest(), 16)
                sc, pol, tg = cand[h % len(cand)]
                nodes.append({"seed": s, "cls": c, "t": sc["t"], "act": sc["act"], "floor": sc["floor"], "names": sc["names"],
                              "lp": sc["lp"], "pol": pol, "target": tg, "salt": 1 + (h >> 64) % (2 ** 31 - 1),
                              "base_win": b["win"], "base_floor": b["floor"]})
    with open(out / f"nodes_s{tag}{sfx}.jsonl", "w", encoding="utf-8") as fn:
        for n in nodes:
            fn.write(json.dumps(n, ensure_ascii=False) + "\n")
    cnt = defaultdict(int)
    for n in nodes:
        cnt[n["cls"]] += 1
    print(f"分叉点：{dict(cnt)}，写入 {out}", flush=True)


def load_nodes(pattern: str, per_class: int | None) -> list[dict]:
    rows = [json.loads(l) for f in sorted(glob.glob(pattern)) for l in open(f, encoding="utf-8") if l.strip()]
    # 次序与分片无关：按 (类, 种子) 排；每类只取前 per_class 个
    rows.sort(key=lambda r: (r["cls"], hashlib.sha256(r["seed"].encode()).hexdigest()))
    if per_class:
        keep, cnt = [], defaultdict(int)
        for r in rows:
            if cnt[r["cls"]] < per_class:
                keep.append(r); cnt[r["cls"]] += 1
        rows = keep
    return rows


def cmd_run(a) -> None:
    import torch
    torch.set_num_threads(2)
    classes = json.loads(Path(a.classes).read_text(encoding="utf-8"))
    nodes = load_nodes(a.nodes, a.per_class)
    if a.only:
        nodes = [n for n in nodes if n["cls"] in a.only.split(",")]
    tapes = {}
    for f in sorted(glob.glob(a.base)):
        for l in open(f, encoding="utf-8"):
            if l.strip():
                b = json.loads(l); tapes[b["seed"]] = (b["tape"], b["win"], b["floor"])
    if a.skip_done:
        done = set()
        for f in Path(a.out).glob(f"res_{a.arm}_{a.tag}*_s*.jsonl"):
            if a.suffix and f"{a.suffix}_s" in f.name:
                continue        # 本次补样本自己正在写的文件不算（各分片同时开打，读到一半会让分片划分不一致）
            for l in open(f, encoding="utf-8"):
                if l.strip():
                    r = json.loads(l); done.add((r["seed"], r["cls"]))
        before = len(nodes)
        nodes = [n for n in nodes if (n["seed"], n["cls"]) not in done]
        print(f"跳过已打过的 {before - len(nodes)} 个点，剩 {len(nodes)} 个", flush=True)
    nodes = shard_of(nodes, a.shard)
    tag = a.shard.split("/")[0] if a.shard else "0"
    net, ver = load_net(a.cont_ckpt, a.device)
    env = make_env(a)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    R = reset_req(a.character)
    if a.verify:
        vs = nodes[:a.verify]
        reqs = [{**R, "seed": n["seed"], "replay": tapes[n["seed"]][0][:n["t"] + 1]} for n in vs]
        res = play(net, a.device, env, reqs, a.character)
        same = sum(int(r["win"] == tapes[n["seed"]][1] and r["floor"] == tapes[n["seed"]][2]) for n, r in zip(vs, res))
        print(f"重放核对：{same} / {len(vs)} 枝与底局结局相同（应全部相同；接着打的检查点须与底局相同才有意义）", flush=True)
        with open(out / f"verify_{a.tag}_s{tag}.json", "w") as f:
            json.dump({"same": same, "n": len(vs)}, f)
    by_cls = defaultdict(list)
    for n in nodes:
        by_cls[n["cls"]].append(n)
    t0 = time.time(); total = 0
    only = (("_" + a.only.replace(",", "")) if a.only else "") + (a.suffix or "")
    with open(out / f"res_{a.arm}_{a.tag}{only}_s{tag}.jsonl", "w", encoding="utf-8") as fo:
        for c, ns in by_cls.items():
            mem = set(classes[c])
            first = (lambda n: n["pol"]) if a.arm == "A" else (lambda n: n["target"])
            reqs = [{**R, "seed": n["seed"], "replay": tapes[n["seed"]][0][:n["t"]], "branch": n["salt"], "first": first(n)} for n in ns]
            bias = (mem, a.bias) if a.arm == "C" else None
            res = play(net, a.device, env, reqs, a.character, bias=bias, track=mem)
            for n, r in zip(ns, res):
                fo.write(json.dumps({"seed": n["seed"], "cls": c, "arm": a.arm, "tag": a.tag, "ver": ver, "bias": a.bias if a.arm == "C" else None,
                                     "first": first(n), "first_name": n["names"][first(n)], **r}, ensure_ascii=False) + "\n")
            fo.flush(); total += len(ns)
            print(f"{c}：{len(ns)} 枝，通关 {sum(r['win'] for r in res) / max(1, len(res)):.1%}，卡住 {sum(r['stuck'] for r in res)}，"
                  f"累计用时 {time.time() - t0:.0f} 秒", flush=True)
    print(f"完成 {total} 枝（臂 {a.arm}，检查点 {a.tag} 第 {ver} 版），{total / max(1, time.time() - t0):.2f} 枝/秒", flush=True)


def _ms(v: list) -> tuple[float, float, int]:
    n = len(v)
    if n == 0:
        return float("nan"), float("nan"), 0
    m = sum(v) / n
    sd = (sum((x - m) ** 2 for x in v) / max(1, n - 1)) ** 0.5
    return m, 1.96 * sd / n ** 0.5, n


def cmd_report(a) -> None:
    import numpy as np
    d = Path(a.dir)
    res = defaultdict(dict)        # (tag, arm) -> {(seed, cls): row}
    for f in sorted(d.glob("res_*_s*.jsonl")):
        for l in open(f, encoding="utf-8"):
            if l.strip():
                r = json.loads(l); res[(r["tag"], r["arm"])][(r["seed"], r["cls"])] = r
    tags = sorted({t for t, _ in res}); classes = sorted({k[1] for v in res.values() for k in v})
    node_act = {(n["seed"], n["cls"]): n["act"] for f in sorted(d.glob("nodes_s*.jsonl")) for n in map(json.loads, open(f, encoding="utf-8"))}
    print(f"结果文件里的检查点：{tags}；类：{classes}")
    for f in sorted(d.glob("verify_*.json")):
        print(f"重放核对 {f.name}：{json.load(open(f))}")
    for tag in tags:
        A = res.get((tag, "A"), {})
        print(f"\n== 检查点 {tag}")
        for arm in ("B", "C"):
            X = res.get((tag, arm), {})
            for c in classes:
                keys = [k for k in X if k[1] == c and k in A]
                if not keys:
                    continue
                dw = [X[k]["win"] - A[k]["win"] for k in keys]
                df = [X[k]["floor"] - A[k]["floor"] for k in keys]
                aw = [A[k]["win"] for k in keys]; xw = [X[k]["win"] for k in keys]
                ma, mx = np.mean(aw), np.mean(xw)
                rho = float(np.corrcoef(aw, xw)[0, 1]) if np.std(aw) > 0 and np.std(xw) > 0 else float("nan")
                both = [k for k in keys if A[k]["hp48"] is not None and X[k]["hp48"] is not None]
                dh = [X[k]["hp48"] - A[k]["hp48"] for k in both]
                r48a = np.mean([A[k]["hp48"] is not None for k in keys]); r48x = np.mean([X[k]["hp48"] is not None for k in keys])
                m, h, n = _ms(dw); mf, hf, _ = _ms(df); mh, hh, nh = _ms(dh)
                stuck = sum(X[k]["stuck"] for k in keys) + sum(A[k]["stuck"] for k in keys)
                plays = np.mean([X[k]["plays"] for k in keys]); deckc = np.mean([X[k]["deck_cls"] or 0 for k in keys])
                deckca = np.mean([A[k]["deck_cls"] or 0 for k in keys])
                cs = np.mean([X[k]["cstep"] / max(1, X[k]["fights"]) for k in keys]); csa = np.mean([A[k]["cstep"] / max(1, A[k]["fights"]) for k in keys])
                print(f"  {arm}−A {c}：{n} 对，通关 A {ma:.1%} → {arm} {mx:.1%}，差 {m:+.3f}±{h:.3f}（ρ {rho:.2f}）；层数差 {mf:+.2f}±{hf:.2f}；"
                      f"到第三幕 Boss A {r48a:.1%} / {arm} {r48x:.1%}，两枝都到时进 Boss 血量差 {mh:+.3f}±{hh:.3f}（{nh} 对）；"
                      f"{arm} 枝目标牌打出 {plays:.1f} 次/局、打完牌组里 {deckc:.2f} 张（A {deckca:.2f}）；每场战斗决策 A {csa:.1f} / {arm} {cs:.1f}；卡住 {stuck}")
                byact = defaultdict(list)
                for k in keys:
                    byact[node_act.get(k, -1)].append(X[k]["win"] - A[k]["win"])
                print("      按分叉点所在的幕：" + "；".join(f"第{ac + 1}幕 {len(v)} 对 {_ms(v)[0]:+.3f}±{_ms(v)[1]:.3f}" for ac, v in sorted(byact.items())))
    # 阶梯交互：同一节点、同一 salt，(B−A)_强 − (B−A)_弱；再减对照类 K
    if len(tags) >= 2 and a.strong and a.weak:
        print(f"\n== 阶梯交互：({a.strong} 的 B−A) − ({a.weak} 的 B−A)，逐点配对")
        dd = {}
        for c in classes:
            ks = [k for k in res.get((a.strong, "B"), {}) if k[1] == c and all(k in res.get((t, x), {}) for t in (a.strong, a.weak) for x in ("A", "B"))]
            v = [(res[(a.strong, "B")][k]["win"] - res[(a.strong, "A")][k]["win"]) - (res[(a.weak, "B")][k]["win"] - res[(a.weak, "A")][k]["win"]) for k in ks]
            m, h, n = _ms(v); dd[c] = (m, h, n)
            print(f"  {c}：{n} 点，交互 {m:+.3f}±{h:.3f}")
        if a.control in dd:
            mk, hk, _ = dd[a.control]
            for c, (m, h, n) in dd.items():
                if c != a.control:
                    print(f"  {c} 减 {a.control}：{m - mk:+.3f}±{(h ** 2 + hk ** 2) ** 0.5:.3f}")
    # C 与 B 的差（同一节点）：(C−A) − (B−A) = C − B
    for tag in tags:
        B = res.get((tag, "B"), {}); C = res.get((tag, "C"), {})
        for c in classes:
            ks = [k for k in C if k[1] == c and k in B]
            if ks:
                m, h, n = _ms([C[k]["win"] - B[k]["win"] for k in ks])
                print(f"\n  {tag} {c}：C − B {m:+.3f}±{h:.3f}（{n} 点；C 臂 β = {C[ks[0]]['bias']}）")
    if a.base:
        base_report(d, a)


def base_report(d: Path, a) -> None:
    """第 0b 步：各类在选卡奖励界面上被提供的次数、策略给它们的概率（界面上同类里最高的一张）、贪心选中率；
    以及偏置 β 的标定：让“这一类被提供时约一半会选”的 β = 这类界面上 (界面最高的非本类对数概率 − 本类最高对数概率) 的中位数。"""
    import numpy as np
    classes = json.loads(Path(a.classes).read_text(encoding="utf-8"))
    games = [json.loads(l) for f in sorted(d.glob("base_s*.jsonl")) for l in open(f, encoding="utf-8") if l.strip()]
    nscreen = sum(len(g["screens"]) for g in games)
    print(f"\n== 底局 {len(games)} 局，通关 {np.mean([g['win'] for g in games]):.1%}，选卡奖励界面 {nscreen} 个（每局 {nscreen / max(1, len(games)):.2f}）")
    for c, members in classes.items():
        mem = set(members); gaps, ps, picked = [], [], 0
        for g in games:
            for sc in g["screens"]:
                lp = np.asarray(sc["lp"]); tg = [j for j, n in enumerate(sc["names"]) if n in mem]
                if not tg:
                    continue
                ot = [j for j in range(len(lp)) if j not in tg]
                bt = max(lp[j] for j in tg); gaps.append((max(lp[j] for j in ot) if ot else -1e9) - bt)
                ps.append(math.exp(bt)); picked += int(int(np.argmax(lp)) in tg)
        if not ps:
            continue
        ps_ = np.asarray(ps)
        print(f"  {c}：提供本类的界面 {len(ps)} 个（每局 {len(ps) / len(games):.2f}），贪心选中 {picked}（{picked / len(ps):.1%}）；"
              f"本类最高概率 中位 {np.median(ps_):.2e}、平均 {ps_.mean():.2e}、p90 {np.quantile(ps_, 0.9):.2e}；"
              f"按策略采样期望选中率 {ps_.mean():.2%}；β 标定（中位差距）{np.median(gaps):.2f}，p75 {np.quantile(gaps, 0.75):.2f}")


def cmd_starts(a) -> None:
    """第 2c 步：把分叉点转成 tools/sim_pimc_eval.py --starts 的开局文件（每行一局：重放到界面、公平重设、走这一臂的第一步；C 臂带偏置）。"""
    classes = json.loads(Path(a.classes).read_text(encoding="utf-8"))
    nodes = load_nodes(a.nodes, a.per_class)
    if a.only:
        nodes = [n for n in nodes if n["cls"] in a.only.split(",")]
    tapes = {}
    for f in sorted(glob.glob(a.base)):
        for l in open(f, encoding="utf-8"):
            if l.strip():
                b = json.loads(l); tapes[b["seed"]] = b["tape"]
    with open(a.out, "w", encoding="utf-8") as fo:
        for n in nodes:
            first = n["pol"] if a.arm == "A" else n["target"]
            fo.write(json.dumps({"key": f"{n['seed']}|{n['cls']}|{a.arm}", "seed": n["seed"], "cls": n["cls"], "arm": a.arm,
                                 "replay": tapes[n["seed"]][:n["t"]], "branch": n["salt"], "first": first,
                                 "bias_names": classes[n["cls"]] if a.arm == "C" else [], "bias": a.bias if a.arm == "C" else 0.0},
                                ensure_ascii=False) + "\n")
    print(f"写入 {a.out}：{len(nodes)} 局（臂 {a.arm}）")


def cmd_beta(a) -> None:
    """C 臂的 β：底局里提供这一类的选卡奖励界面上，(最高的非本类对数概率 − 本类最高对数概率) 的中位数，即加上它约一半会选。"""
    import numpy as np
    mem = set(json.loads(Path(a.classes).read_text(encoding="utf-8"))[a.cls])
    gaps = []
    for f in sorted(Path(a.dir).glob("base_s*.jsonl")):
        for l in open(f, encoding="utf-8"):
            if not l.strip():
                continue
            for sc in json.loads(l)["screens"]:
                lp = sc["lp"]; tg = [j for j, n in enumerate(sc["names"]) if n in mem]
                ot = [j for j in range(len(lp)) if j not in tg]
                if tg and ot:
                    gaps.append(max(lp[j] for j in ot) - max(lp[j] for j in tg))
    print(f"{float(np.median(gaps)):.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("nodes", "run"):
        p = sub.add_parser(name)
        p.add_argument("--classes", required=True); p.add_argument("--out", required=True)
        p.add_argument("--character", default="SILENT"); p.add_argument("--envs", type=int, default=16)
        p.add_argument("--device", default="cuda"); p.add_argument("--dll", default=None); p.add_argument("--shard", default=None)
    p = sub.choices["nodes"]
    p.add_argument("--ckpt", required=True); p.add_argument("--seeds-file", required=True)
    p.add_argument("--games", type=int, default=2400); p.add_argument("--seed-from", type=int, default=0); p.add_argument("--rng", type=int, default=19)
    p.add_argument("--suffix", default=None, help="输出文件名后缀（补底局时用，不覆盖原来的分片文件）")
    p = sub.choices["run"]
    p.add_argument("--nodes", required=True); p.add_argument("--base", required=True); p.add_argument("--cont-ckpt", required=True)
    p.add_argument("--tag", required=True); p.add_argument("--arm", choices=["A", "B", "C"], required=True); p.add_argument("--bias", type=float, default=0.0)
    p.add_argument("--per-class", type=int, default=1500); p.add_argument("--only", default=None, help="只跑这些类（逗号分隔）")
    p.add_argument("--verify", type=int, default=0)
    p.add_argument("--skip-done", action="store_true", help="跳过同一检查点同一臂已有结果的点（补样本用）")
    p.add_argument("--suffix", default=None, help="结果文件名后缀（补样本用）")
    p = sub.add_parser("starts")
    p.add_argument("--nodes", required=True); p.add_argument("--base", required=True); p.add_argument("--classes", required=True)
    p.add_argument("--arm", choices=["A", "B", "C"], required=True); p.add_argument("--bias", type=float, default=0.0)
    p.add_argument("--per-class", type=int, default=1500); p.add_argument("--only", default=None); p.add_argument("--out", required=True)
    p = sub.add_parser("beta")
    p.add_argument("--dir", required=True); p.add_argument("--classes", required=True); p.add_argument("--cls", required=True)
    p = sub.add_parser("report")
    p.add_argument("--dir", required=True); p.add_argument("--classes", default=None); p.add_argument("--base", action="store_true")
    p.add_argument("--strong", default=None); p.add_argument("--weak", default=None); p.add_argument("--control", default="K")
    a = ap.parse_args()
    {"nodes": cmd_nodes, "run": cmd_run, "report": cmd_report, "beta": cmd_beta, "starts": cmd_starts}[a.cmd](a)


if __name__ == "__main__":
    main()
