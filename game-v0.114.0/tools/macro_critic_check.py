"""局外能不能“按房间自举”的离线验证（决策记录 282）：价值头能不能分清局外选项的好坏。

用第 258 条的分枝标注（同一局面、同一个公平重设的未来下，首选与一个备选各自贪心打完的输赢；/root/diag/BLF_all.jsonl 选卡、
BLR_all.jsonl 休息处与选路线）：
  1. 用标注时的模型（--base，第 157,500 版）贪心重打每个种子，找回到标注点为止的动作序列，并核对那一步的决策类型、层数、两个选项的名字；
  2. 对每对的两个选项，按标注里的同一个 salt 重放到“选完这一项”的局面（reset 带 replay、branch、first），用 --critic（学生 e）的价值头打分；
  3. 价值差 ΔV = V(首选之后) − V(备选之后) 与真实结局差 Δwin = 首选通关 − 备选通关对照：相关、结局不同的对里方向一致的比例、
     按价值差分组的结局差；同样算策略自己的偏好（标注里的对数概率差）作对照。按决策类型分开。

    python -I tools/macro_critic_check.py --labels BLF_all.jsonl,BLR_all.jsonl --base v157500.pt --critic student_e.pt --out 结果.jsonl [--envs 48] [--dll 引擎] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "tools"))

BASE_RESET = {"op": "reset", "mode": "run", "character": "DEFECT", "ascension": 10, "route_summary": True}     # 同 branch_label.RESET
CRIT_EXTRA = {"privileged": True, "future_content": True}       # 价值头要的特权输入（不改局面）
KIND = {3: "选卡", 7: "休息处", 11: "选路线"}


def load(path, device):
    import torch
    from sts2rl.model import ModelConfig, PolicyValueNet, load_state
    b = torch.load(path, map_location="cpu", weights_only=False)
    mc = dict(b["config"]["model"]); mc.pop("oracle", None)
    net = PolicyValueNet(ModelConfig(**mc)).to(device).eval(); load_state(net, b["state"])
    return net, bool(mc.get("macro_attend"))


def main() -> None:
    import torch
    from sts2env.env import VecEnv
    from sts2rl.actor import MODE_RUN
    from sts2rl.features import Ragged, collate_ragged
    from branch_label import _name
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--critic", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--envs", type=int, default=48)
    ap.add_argument("--dll", default=None)
    ap.add_argument("--limit", type=int, default=0, help="只取前多少个种子（试跑用）")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = torch.device(a.device)
    recs = defaultdict(list)
    for f in a.labels.split(","):
        for l in open(f, encoding="utf-8"):
            r = json.loads(l)
            if len(r["options"]) == 2:
                recs[r["seed"]].append(r)
    seeds = sorted(recs)[: a.limit] if a.limit else sorted(recs)
    base, base_att = load(a.base, dev)
    crit, crit_att = load(a.critic, dev)
    env = VecEnv(a.envs, **({"dll": Path(a.dll)} if a.dll else {}))
    t0 = time.time()

    def infer(net, att, obs_list):
        with torch.no_grad():
            lg, v = net(collate_ragged(Ragged.from_obs(obs_list, np.full(len(obs_list), MODE_RUN)), attend=att).to(dev))
        return lg.float(), v.float()

    # 一、贪心重打，找回动作序列；核对标注点
    tapes, ok = {}, {}
    q = list(seeds); slot = {}

    def start(i):
        if not q:
            slot.pop(i, None); return
        s = q.pop()
        need = max(r["t"] for r in recs[s])
        slot[i] = {"seed": s, "tape": [], "need": need, "check": {r["t"]: r for r in recs[s]}, "bad": 0}
        env.send(i, {**BASE_RESET, "seed": s})
    for i in range(len(env)):
        start(i)
    obs = {}
    while slot:
        for i, o in env.recv_ready(timeout=1.0):
            st = slot.get(i)
            if st is None:
                continue
            if isinstance(o, Exception) or o.done or o.stuck or len(st["tape"]) > st["need"]:
                tapes[st["seed"]] = st["tape"]; ok[st["seed"]] = st["bad"] == 0
                start(i); continue
            t = len(st["tape"])
            if t in st["check"]:
                r = st["check"][t]
                names = [_name(o, x["a"]) if x["a"] < o.num_actions else None for x in r["options"]]
                if int(o.decision) != r["d"] or int(o.floor) != r["floor"] or names != [x["name"] for x in r["options"]]:
                    st["bad"] += 1
            obs[i] = o
        if not obs:
            continue
        ids = sorted(obs)
        lg, _ = infer(base, base_att, [obs[i] for i in ids])
        for k, i in enumerate(ids):
            act = int(lg[k, :obs[i].num_actions].argmax())
            slot[i]["tape"].append(act); env.send_step(i, act)
        obs.clear()
    good = [s for s in seeds if ok.get(s)]
    print(f"{time.strftime('%H:%M:%S')} 重打 {len(seeds)} 个种子，标注点全部对上的 {len(good)} 个（{(time.time() - t0) / 60:.1f} 分钟）", flush=True)

    # 二、重放到“选完这一项”的局面，价值头打分
    jobs = [(s, j, k) for s in good for j, r in enumerate(recs[s]) for k in range(2)]
    vals = {}
    jq = list(jobs); slot = {}

    def start2(i):
        if not jq:
            slot.pop(i, None); return
        s, j, k = jq.pop()
        r = recs[s][j]; opt = r["options"][k]
        slot[i] = (s, j, k)
        env.send(i, {**BASE_RESET, **CRIT_EXTRA, "seed": s, "replay": tapes[s][: r["t"]], "branch": int(opt["runs"][0][0]), "first": int(opt["a"])})
    for i in range(len(env)):
        start2(i)
    pend = []
    while slot or pend:
        for i, o in (env.recv_ready(timeout=1.0) if slot else []):
            if i not in slot:
                continue
            if isinstance(o, Exception) or o.stuck:
                start2(i); continue
            pend.append((slot[i], o))
            start2(i)
        if len(pend) >= 64 or (pend and not slot):
            items = pend; pend = []
            _, v = infer(crit, crit_att, [o for _, o in items])
            for (key, o), x in zip(items, v.cpu().numpy()):
                vals[key] = (float(x), bool(o.done), bool(o.done and o.victory))
    print(f"{time.strftime('%H:%M:%S')} 打分 {len(vals)} 个局面（{(time.time() - t0) / 60:.1f} 分钟）", flush=True)

    # 三、对照
    rows = []
    with open(a.out, "w", encoding="utf-8") as fh:
        for s in good:
            for j, r in enumerate(recs[s]):
                if (s, j, 0) not in vals or (s, j, 1) not in vals:
                    continue
                o0, o1 = r["options"]
                w0, w1 = o0["runs"][0][1], o1["runs"][0][1]
                row = {"d": r["d"], "act": r["act"], "dv": vals[(s, j, 0)][0] - vals[(s, j, 1)][0], "dwin": w0 - w1,
                       "dfloor": o0["runs"][0][2] - o1["runs"][0][2], "dlogp": float(np.log(max(o0["p"], 1e-6)) - np.log(max(o1["p"], 1e-6)))}
                rows.append(row); fh.write(json.dumps(row) + "\n")
    report(rows)


def report(rows) -> None:
    def stats(rs, name):
        if len(rs) < 30:
            return
        dv = np.array([r["dv"] for r in rs]); dw = np.array([r["dwin"] for r in rs], float)
        df = np.array([r["dfloor"] for r in rs], float); lp = np.array([r["dlogp"] for r in rs])
        nz = dw != 0
        agree = float(np.mean(np.sign(dv[nz]) == np.sign(dw[nz]))) if nz.any() else float("nan")
        agree_p = float(np.mean(np.sign(lp[nz]) == np.sign(dw[nz]))) if nz.any() else float("nan")
        c = lambda x, y: float(np.corrcoef(x, y)[0, 1])
        se = 1 / np.sqrt(len(rs))
        # 按价值差分三组：结局差的均值（价值头说“首选好很多 / 差不多 / 备选好”时，真实结局差是多少）
        qs = np.quantile(dv, [1 / 3, 2 / 3])
        grp = [dw[dv <= qs[0]].mean(), dw[(dv > qs[0]) & (dv <= qs[1])].mean(), dw[dv > qs[1]].mean()]
        print(f"  {name}（{len(rs):,} 对，结局不同 {int(nz.sum()):,}）：价值差与结局差相关 {c(dv, dw):+.3f}（±{1.96 * se:.3f}），与层数差相关 {c(dv, df):+.3f}；"
              f"结局不同的对里方向一致 价值头 {agree:.1%}、策略偏好 {agree_p:.1%}；按价值差三等分的结局差 {grp[0]:+.3f} / {grp[1]:+.3f} / {grp[2]:+.3f}", flush=True)
    print("价值头（学生 e）能不能分清局外选项：", flush=True)
    stats(rows, "全部")
    for d, n in KIND.items():
        stats([r for r in rows if r["d"] == d], n)
    for act in (0, 1, 2):
        stats([r for r in rows if r["act"] == act], f"第 {act + 1} 幕")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--report":
        report([json.loads(l) for l in open(sys.argv[2], encoding="utf-8")])
    else:
        main()
