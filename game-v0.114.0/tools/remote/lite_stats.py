"""租用机上运行：给轻量看板出一份精简统计（2026-11-03，用户：“精简一下看板，轻量化，只需要展示核心训练参数和我们最近关注的指标……
其他的维持统计存在训练机，不要自动回传了”）。完整的训练日志、评测记录、回放、探针档案都留在租用机，本机只拉这份几十 KB 的 JSON。

    python3 tools/remote/lite_stats.py train    # 学习端（B）：训练状态、核心参数、最近几版的读数、历史走势（抽样）、价值探针的势能前后统计
    python3 tools/remote/lite_stats.py eval     # 评测机（A）：P1 常规评测（护栏与局外指标）、手动配对复测
结果打印到标准输出。历史与评测各自在运行目录里留一份增量缓存（lite_hist.jsonl、lite_eval.jsonl），每次只读新增的部分。
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

HOME = Path.home()
# 运行目录：环境变量 LITE_RUN、LITE_EVAL（目录名，储君为 regent、regent-eval；本机 pull_lite.sh 按看板目标 lite_target.json 传），默认故障机器人的
RUN = HOME / ".local/share/sls2/runs" / os.environ.get("LITE_RUN", "defect")
EVAL = HOME / ".local/share/sls2/runs" / os.environ.get("LITE_EVAL", "defect-eval")
REPO = Path(__file__).resolve().parents[2]

CFG_KEYS = ("lr", "batch_steps", "epochs", "minibatch", "grad_accum", "gamma", "lam", "macro_lam", "ent_coef", "ent_coef_macro",
            "ent_coef_map", "ema_decay", "max_grad_norm", "actors", "envs_per_actor", "bv_shaping", "bv_start", "bv_ramp", "bv_model",
            "victory_reward", "act3_boss_reward", "act3_boss2_reward", "floor_reward", "macro_cross_segment", "macro_defer_lag", "combat_explore",
            "hl_coef", "hl_start", "hl_ramp", "hl_gate", "hl_sigma", "hl_coef_from", "hl_coef_ramp", "hl_scalar_full", "ppg_every", "ppg_keep", "ppg_epochs", "ppg_beta_clone", "aux_coef", "aux_hp_w", "probe_split",
            # 决策记录 267：只训局外、同种子成组
            "combat_greedy", "macro_only_pi", "combat_value_frac", "combat_anchor_coef", "run_group", "run_fraction", "group_ttl",
            "milestone_from", "milestone_ramp", "combat_anchor_rows",
            # 2026-11-13：看板参数表给 Muon 那一行（猎人、愚者的隐藏层二维权重用 Muon）
            "character", "optimizer", "muon_lr", "muon_momentum")
MODEL_KEYS = ("future_content", "value_hl", "hl_bins", "hl_min", "hl_max", "hl_detach", "combat_aux", "macro_attend", "content_upgrade")
LOG_KEYS = ("version", "time", "sps", "macro_deferred", "macro_pending", "loss_pi", "loss_v", "loss_v_macro", "entropy", "entropy_macro", "approx_kl", "clip_frac",
            "run_win_rate", "run_floor_mean", "run_reach_act2", "run_reach_act3", "run_bv_shaped", "run_episodes", "remote_used", "stale_steps", "nan_skipped",
            # 决策记录 261：分类价值头、战斗辅助头的读数
            "hl_beta", "hl_ratio", "hl_coef", "loss_hl", "loss_v_hl", "loss_v_scalar", "loss_hl_macro", "loss_v_hl_macro", "loss_v_scalar_macro",
            "aux_win_loss", "aux_hp_loss", "aux_win_y", "aux_win_p", "aux_n", "t_learn",
            "ppg_steps", "ppg_loss_v", "ppg_kl", "ppg_seconds",
            # 决策记录 267：战斗锚定（与老师的 KL、首选一致比例）、局外步占入训步的比例
            "anchor_kl", "anchor_agree", "macro_frac",
            # 2026-11-11（猎人决策记录第 11 条，用户：“这样子扩批的话，我们需要关注一下策略熵是否坍塌”）：地图选路步的熵
            "entropy_map")


def ent_combat(r: dict) -> dict:
    """战斗步的熵（推算）：日志的 entropy 是全部步的平均（战斗步为主），entropy_macro 是局外步的平均，macro_frac 是局外步占入训步的比例，
    战斗步 ≈ (entropy − macro_frac × entropy_macro) / (1 − macro_frac)。各小批的平均方式略有不同，是近似值。缺任一项时不给。"""
    e, m, f = r.get("entropy"), r.get("entropy_macro"), r.get("macro_frac")
    if not all(isinstance(x, (int, float)) for x in (e, m, f)) or not 0 <= f < 1:
        return {}
    return {"ent_c": round((e - f * m) / (1 - f), 4)}


def flat_group(r: dict) -> dict:
    """同种子成组采集（决策记录 267）的读数：有同伴的局外步比例、平均同伴数、控制变量系数（在用的 / 这一版估的）、
    有同伴的步上调整前后优势的均方根、第一次分开的层数（中位）、放行的组里到齐的比例、还在等的组数。"""
    g = r.get("group")
    if not g:
        return {}
    out = {"g_peer": round(g["peer_steps"] / g["steps"], 4) if g.get("steps") else None, "g_peers": g.get("peers"), "g_c": g.get("c"), "g_chat": g.get("c_hat"),
           "g_rms_raw": g.get("rms_raw") if g.get("peer_steps") else None, "g_rms_adj": g.get("rms_adj") if g.get("peer_steps") else None,
           "g_div_floor": g.get("div_floor"), "g_full": round(g["full"] / g["groups"], 4) if g.get("groups") else None, "g_pending": g.get("pending"),
           "g_groups": g.get("groups")}
    return {k: v for k, v in out.items() if v is not None}


def flat_health(r: dict) -> dict:
    """训练健康统计（log 的 health）里看板要的几项：价值对回报的解释方差（战斗、局外）、裁剪前梯度范数与超阈值比例、几组新参数这一版的相对更新量。"""
    h = r.get("health") or {}
    ev, upd = h.get("ev") or {}, h.get("upd") or {}
    out = {"ev_c": ev.get("combat"), "ev_m": ev.get("macro"), "gn": h.get("grad_norm"), "gcf": h.get("grad_clip_frac")}
    for k in ("fut", "value_hl", "value_hl_m", "value_aux", "L0"):
        if k in upd:
            out["upd_" + k] = upd[k]
    return {k: v for k, v in out.items() if v is not None}
# 最近 600 版逐版的紧凑数组（看板上可左右滑动的几张图）：键 → 记录里的字段
DENSE = {"lv": "loss_v", "lvm": "loss_v_macro", "lvs": "loss_v_scalar", "lvh": "loss_v_hl", "lvsm": "loss_v_scalar_macro", "lvhm": "loss_v_hl_macro",
         "evc": "ev_c", "evm": "ev_m", "beta": "hl_beta", "rw": "run_win_avg", "entm": "entropy_macro", "kl": "approx_kl",
         "anc": "anchor_agree", "ancl": "anchor_kl", "gpeer": "g_peer", "gc": "g_c", "gchat": "g_chat",
         # 熵的逐版走势（2026-11-11）：战斗步（推算）、地图选路步；局外步即上面的 entm
         "entc": "ent_c", "entp": "entropy_map"}
# 熵的全程走势（hist 里，2026-11-11）：20 版简单滑动平均，键 → 记录里的字段；数值留 3 位（单版波动约 0.01–0.02，再细没有意义，也省流量）
ENT_AVG = {"ec": "ent_c", "em": "entropy_macro", "ep": "entropy_map"}
DENSE_R3 = ("entc", "entm", "entp")       # 逐版数组里只留 3 位小数的键
PICK_WINDOW = 1000       # 选牌、先古遗物统计取最近多少版
RECENT_KEYS = ("version", "time", "sps", "nan_skipped", "phi_skipped", "loss_pi", "remote_idle_s", "remote_conns")


def jload(p):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def tail_lines(p: Path, nbytes: int) -> list[dict]:
    try:
        with open(p, "rb") as f:
            f.seek(0, 2); n = f.tell(); f.seek(max(0, n - nbytes))
            lines = f.read().decode(errors="ignore").splitlines()[1:]
    except OSError:
        return []
    out = []
    for l in lines:
        try:
            out.append(json.loads(l))
        except ValueError:
            pass
    return out


def incremental(src: Path, cache: Path, pick, start_bytes: int) -> list[dict]:
    """src 是只追加的 jsonl：从上次读到的位置接着读，每条用 pick 压成小记录追加进 cache；返回 cache 的全部记录。
    第一次只从文件末尾往前 start_bytes 处读起（不回溯全部历史）。"""
    state = cache.with_suffix(".pos")
    try:
        pos = int(state.read_text())
    except (OSError, ValueError):
        pos = -1
    try:
        size = src.stat().st_size
    except OSError:
        size = 0
    if pos < 0 or pos > size:
        pos = max(0, size - start_bytes)
        first = True
    else:
        first = False
    new = []
    if size > pos:
        with open(src, "rb") as f:
            f.seek(pos)
            data = f.read()
        cut = data.rfind(b"\n") + 1
        lines = data[:cut].decode(errors="ignore").splitlines()
        if first and pos > 0:
            lines = lines[1:]          # 第一行多半是半截
        for l in lines:
            try:
                r = pick(json.loads(l))
            except ValueError:
                continue
            if r:
                new.append(r)
        pos += cut
    if new:
        with open(cache, "a", encoding="utf-8") as h:
            for r in new:
                h.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
    state.write_text(str(pos))
    out = []
    try:
        for l in open(cache, encoding="utf-8"):
            if l.strip():
                out.append(json.loads(l))
    except OSError:
        pass
    return out


def drop_reverted(rows: list[dict]) -> list[dict]:
    """回退过权重（决策记录 257：第 158,636–158,843 版作废，从第 158,635 版接着训）时日志里先后有两段同号的版本：
    后面的记录版本号不大于前面的，就把前面版本号不小于它的那些（作废的时间线）去掉。"""
    out: list[dict] = []
    for r in rows:
        v = r.get("version")
        if v is None:
            continue
        while out and out[-1]["version"] >= v:
            out.pop()
        out.append(r)
    return out


# 三项改动（决策记录 261）的上线前后对比：各取一个窗口按版平均
CMP_KEYS = ("loss_v", "loss_v_macro", "ev_c", "ev_m", "loss_v_scalar", "loss_v_hl", "loss_v_scalar_macro", "loss_v_hl_macro", "loss_hl", "loss_hl_macro",
            "hl_beta", "hl_ratio", "hl_coef", "aux_win_loss", "aux_hp_loss", "aux_win_y", "aux_win_p", "aux_n", "gn", "gcf", "approx_kl", "clip_frac", "entropy", "entropy_macro",
            "upd_fut", "upd_value_hl", "upd_value_hl_m", "upd_value_aux", "upd_L0", "sps", "t_learn",
            "anchor_kl", "anchor_agree", "macro_frac", "g_peer", "g_peers", "g_c", "g_chat", "g_rms_raw", "g_rms_adj", "g_div_floor", "g_full", "g_pending")


def window_mean(rows: list[dict]) -> dict:
    out = {}
    for k in CMP_KEYS:
        xs = [r[k] for r in rows if isinstance(r.get(k), (int, float)) and math.isfinite(r[k])]
        if xs:
            out[k] = round(sum(xs) / len(xs), 6)
    return out


def thin(rows: list[dict], n: int) -> list[dict]:
    if len(rows) <= n:
        return rows
    step = len(rows) / n
    return [rows[int(i * step)] for i in range(n - 1)] + [rows[-1]]


# 对战练习（战斗课程）各类的胜率（2026-11-07，用户：“你把对战练习也做一张图，搞到看板上去。”）：日志每版的 [胜场, 场数]
PRACTICE = ("act1_monster", "act1_elite", "act2_monster", "act2_elite", "act3_monster", "act3_elite", "mix_monster", "mix_elite")
PRACTICE_WINDOW = 50        # 按最近多少版的场数合计算胜率（单版场数少、很嘈杂）
PRACTICE_MIN = 100          # 窗口里不满这么多场不出点


def practice() -> dict | None:
    """各类练习的胜率走势：每版压成 [各类胜场、场数…, 三幕 Boss 胜场、场数…] 存进增量缓存（第一次从头读整份日志），
    按最近 PRACTICE_WINDOW 版的场数合计算胜率，抽成约 160 个点；另给最近 200 版的合计（看板的表）。键为 PRACTICE 加 boss1–3。"""
    def pk(r):
        if "version" not in r:
            return None
        p = []
        for k in PRACTICE:
            w, n = (r.get(k) or [0, 0])[:2]
            p += [w, n]
        bw, bn = r.get("boss_wins") or [0, 0, 0], r.get("boss_n") or [0, 0, 0]
        for i in range(3):
            p += [bw[i] if i < len(bw) else 0, bn[i] if i < len(bn) else 0]
        return {"version": r["version"], "p": p} if any(p[1::2]) else None
    rows = drop_reverted(incremental(RUN / "log.jsonl", RUN / "lite_practice.jsonl", pk, 10 ** 12))
    if not rows:
        return None
    keys = list(PRACTICE) + ["boss1", "boss2", "boss3"]
    m = len(keys)
    pts = []
    acc = [0] * (2 * m)
    for j, r in enumerate(rows):
        for t in range(2 * m):
            acc[t] += r["p"][t]
        if j >= PRACTICE_WINDOW:
            for t in range(2 * m):
                acc[t] -= rows[j - PRACTICE_WINDOW]["p"][t]
        pts.append([r["version"]] + [round(acc[2 * i] / acc[2 * i + 1], 4) if acc[2 * i + 1] >= PRACTICE_MIN else None for i in range(m)])
    last = [0] * (2 * m)
    for r in rows[-200:]:
        for t in range(2 * m):
            last[t] += r["p"][t]
    return {"keys": keys, "window": PRACTICE_WINDOW, "pts": thin(pts, 160),
            "last": {"v0": rows[-200:][0]["version"], "v1": rows[-1]["version"], "wins": last[0::2], "n": last[1::2]}}


# 跳过稳赢战斗（猎人决策记录第 14 条，第 24,208 版起）的走势：日志每版 run_choices 里的 SKIP_* / B3_* / SKIPRUN_* 计数（只有开了 skip_easy 的版才有），
# 每版压成一行存进增量缓存 lite_skip.jsonl；看板画最近 600 版、每点为最近 SKIP_WIN 版合计的比例，另给第 14 条以来与最近 200 版的累计。
# 计数的含义见 python/sts2rl/actor.py 与 tools/remote/silent/skip_stats.py；护栏在 tools/remote/silent/supervise/sup_train.py
SKIP_WIN = 20


def skip_trend() -> dict | None:
    def pk(r):
        ch = r.get("run_choices") or {}
        if "version" not in r or not any(k.startswith("SKIP_ELIG:") for k in ch):
            return None
        g = lambda pre: sum(v for k, v in ch.items() if k.startswith(pre))
        out = {"version": r["version"], "time": r.get("time"), "el": g("SKIP_ELIG:"), "dn": g("SKIP_DONE:"),
               "st": r.get("stuck") or 0, "ep": r.get("run_episodes") or 0}
        for key, src in (("bnr", "B3_N:real"), ("bhr", "B3_HP:real"), ("bns", "B3_N:skip"), ("bhs", "B3_HP:skip"),
                         ("rnr", "SKIPRUN_N:real"), ("rwr", "SKIPRUN_WIN:real"), ("rns", "SKIPRUN_N:skip"), ("rws", "SKIPRUN_WIN:skip"),
                         ("pn", "SKIP_POT_N"), ("pp", "SKIP_POT_P"), ("py", "SKIP_POT_Y")):
            x = ch.get(src, 0)
            out[key] = round(x, 4) if isinstance(x, float) else x
        return out
    rows = drop_reverted(incremental(RUN / "log.jsonl", RUN / "lite_skip.jsonl", pk, 60_000_000))
    if not rows:
        return None
    keys = ("el", "dn", "bnr", "bhr", "bns", "bhs", "rnr", "rwr", "rns", "rws", "pn", "pp", "py", "st", "ep")
    tot = lambda w: {k: round(sum(r.get(k) or 0 for r in w), 3) for k in keys}
    ratio = lambda a, b, d=4: round(a / b, d) if b else None
    out = {"v0": rows[0]["version"], "v1": rows[-1]["version"], "win": SKIP_WIN, "all": tot(rows), "last200": tot(rows[-200:])}
    # 每小时整局数：最近 100 版，相邻两版间隔超过 120 秒的（停机、重启）不算
    w = rows[-101:]
    pr = [(b["ep"], b["time"] - a["time"]) for a, b in zip(w, w[1:]) if a.get("time") and b.get("time") and 0 < b["time"] - a["time"] <= 120]
    out["runs_per_hour"] = round(3600 * sum(e for e, _ in pr) / sum(t for _, t in pr)) if pr else None
    dense = {k: [] for k in ("v", "sk", "b3r", "b3s", "pp", "py")}
    acc = dict.fromkeys(keys, 0.0)
    start = max(0, len(rows) - 600)
    for j, r in enumerate(rows):
        for k in keys:
            acc[k] += r.get(k) or 0
        if j >= SKIP_WIN:
            for k in keys:
                acc[k] -= rows[j - SKIP_WIN].get(k) or 0
        if j >= start:
            dense["v"].append(r["version"])
            dense["sk"].append(ratio(acc["dn"], acc["el"]))
            dense["b3r"].append(ratio(acc["bhr"], acc["bnr"]))
            dense["b3s"].append(ratio(acc["bhs"], acc["bns"]))
            dense["pp"].append(ratio(acc["pp"], acc["pn"]))
            dense["py"].append(ratio(acc["py"], acc["pn"]))
    out["dense"] = dense
    return out


def pick_hist(r: dict) -> dict | None:
    """学习端日志的一条压成走势缓存（lite_hist.jsonl）里的一行。改了这里要看的字段时，旧缓存行里没有新字段，需要从 log.jsonl 重建缓存
    （2026-11-11 加熵的字段时重建过一次，见猎人决策记录第 11 条）。"""
    if "version" not in r:
        return None
    return {**{k: (round(r[k], 5) if isinstance(r.get(k), float) else r.get(k)) for k in LOG_KEYS if k in r},
            **{k: round(v, 6) for k, v in flat_health(r).items()}, **flat_group(r), **ent_combat(r)}


def train() -> dict:
    cfg = jload(RUN / "train_config.json") or {}
    hist = incremental(RUN / "log.jsonl", RUN / "lite_hist.jsonl", pick_hist, 60_000_000)
    since = int(cfg.get("bv_start") or 0) - 3000
    hist = drop_reverted([r for r in hist if r.get("version", 0) >= since])
    # 训练整局通关率单版只有几十局、很嘈杂：按局数加权取最近 20 版的滑动平均（run_win_avg），再抽样
    # 窗口里累计不满 300 局不出点（开头与重启后头几版局数很少，曲线会猛跳）；训练中断超过半小时就重新累计
    # 训练早期另看到达层数与进第二幕的比例（2026-11-06 储君从零训练，用户要看）：同样按局数加权的 20 版滑动平均
    acc = {k: ([], []) for k in ("run_win_rate", "run_floor_mean", "run_reach_act2")}
    ent_q = {k: [] for k in ENT_AVG}        # 熵的 20 版简单滑动平均（各版都是整批的平均，不按局数加权；不满 5 版不出点）
    last_t = None
    for r in hist:
        for k, src in ENT_AVG.items():
            x = r.get(src)
            if isinstance(x, (int, float)):
                q = ent_q[k]; q.append(x); del q[:-20]
            r[k] = round(sum(ent_q[k]) / len(ent_q[k]), 3) if len(ent_q[k]) >= 5 else None
        if last_t is not None and (r.get("time") or 0) - last_t > 1800:
            acc = {k: ([], []) for k in acc}
        last_t = r.get("time") or last_t
        n = r.get("run_episodes") or 0
        for k, (s_, c_) in acc.items():
            x = r.get(k)
            s_.append((x or 0) * n if x is not None else 0); c_.append(n if x is not None else 0)
        for k, out_k, d in (("run_win_rate", "run_win_avg", 4), ("run_floor_mean", "run_floor_avg", 2), ("run_reach_act2", "run_act2_avg", 4)):
            s_, c_ = acc[k]; sw, sn = sum(s_[-20:]), sum(c_[-20:])
            r[out_k] = round(sw / sn, d) if sn >= 300 else None
    recent = tail_lines(RUN / "log.jsonl", 200_000)[-15:]
    res = tail_lines(RUN / "resource.jsonl", 4_000)[-1:] or [{}]
    backups = sorted((RUN / "backups").glob("v*.pt"), key=lambda p: p.stat().st_mtime)[-3:] if (RUN / "backups").is_dir() else []
    out = {"role": "train", "time": time.time(), "status": jload(RUN / "status.json"), "watchdog": jload(RUN / "watchdog.json"),
           "config": {**{k: cfg.get(k) for k in CFG_KEYS}, "model": {k: (cfg.get("model") or {}).get(k) for k in MODEL_KEYS}}, "latest": hist[-1] if hist else None,
           "recent": [{k: r.get(k) for k in RECENT_KEYS} for r in recent], "resource": res[-1],
           "ppg_recent": [{k: r.get(k) for k in ("version", "ppg_steps", "ppg_loss_v", "ppg_kl", "ppg_seconds")} for r in hist if r.get("ppg_steps")][-5:],
           # 走势图只用版本号与训练通关的滑动平均（原来每个点带全部约 40 个字段，占了这份统计的一半）
           # 2026-11-11 起另带熵的 20 版滑动平均（ec 战斗步（推算）、em 局外步、ep 地图选路步，键见 ENT_AVG）
           "hist": [{"version": r["version"], "run_win_avg": r.get("run_win_avg"), "run_floor_avg": r.get("run_floor_avg"), "run_act2_avg": r.get("run_act2_avg"),
                     **{k: r.get(k) for k in ENT_AVG}}
                    for r in thin(hist, 160)], "backups": [p.name for p in backups],
           # 最近 600 版逐版的价值损失（看板上可左右滑动的那张图），只给版本号与两项损失，存成紧凑数组
           # 熵的三项只留 3 位小数（2026-11-11，省流量）
           "dense": {"v": [r["version"] for r in hist[-600:]],
                     **{k: [round(x, 3) if k in DENSE_R3 and isinstance(x := r.get(src), float) else r.get(src) for r in hist[-600:]] for k, src in DENSE.items()}}}
    split, base = cfg.get("probe_split"), cfg.get("probe_base")     # 上线的版本；对照窗口 [起, 止)（决策记录 261）
    if split:
        before = [r for r in hist if base and base[0] <= r["version"] < base[1]] or [r for r in hist if r["version"] <= split][-50:]
        after = [r for r in hist if r["version"] > split + 5][-50:]
        wwin = lambda rs: (round(sum((r.get("run_win_rate") or 0) * (r.get("run_episodes") or 0) for r in rs)
                                 / max(1, sum(r.get("run_episodes") or 0 for r in rs if r.get("run_win_rate") is not None)), 4) if rs else None)
        out["cmp"] = {"split": split, "before": {**window_mean(before), "run_win": wwin(before)}, "after": {**window_mean(after), "run_win": wwin(after)},
                      "before_v": [before[0]["version"], before[-1]["version"]] if before else None,
                      "after_v": [after[0]["version"], after[-1]["version"]] if after else None}
    # 快照漂移（决策记录 274 补记三，tools/remote/snap_drift.py）：有比上次结果更新的定时备份就在后台低优先级重算，读上次的结果
    sd_out, sd_lock = RUN / "snap_drift.json", RUN / "snap_drift.lock"
    try:
        if RUN.name != "defect":        # 快照漂移是对学生 e 算的（snap_drift.py 只看故障机器人的运行目录），别的训练线不算
            raise OSError
        newest = max((q.stat().st_mtime for q in (RUN / "backups").glob("v*.pt")), default=0)
        last = sd_out.stat().st_mtime if sd_out.exists() else 0
        locked = sd_lock.exists() and time.time() - sd_lock.stat().st_mtime < 1800
        if (newest > last or time.time() - last > 120) and not locked:     # 有新备份，或上次算完已过 2 分钟（“当前”点；一次约 5 秒、几十 MB 内存）
            sd_lock.touch()
            subprocess.Popen(["sh", "-c", f"nice -n 19 {sys.executable} -I {REPO / 'tools/remote/snap_drift.py'} > {RUN / 'snap_drift.log'} 2>&1; rm -f {sd_lock}"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    except OSError:
        pass
    out["drift"] = jload(sd_out)
    out["practice"] = practice()
    if cfg.get("skip_easy") or (RUN / "lite_skip.jsonl").exists():
        out["skip"] = skip_trend()
    # 选牌与先古遗物（训练整局，带探索）：日志每 card_log_every 版记一次这段时间的合计（run_cards、run_ancient，
    # 每张牌 / 每件遗物为 [第一、二、三幕被提供, 第一、二、三幕被选]）；看板取最近 PICK_WINDOW 版的合计（2026-11-06 用户要求接回这两项统计）
    pk = lambda r: ({"version": r["version"], "time": r.get("time"), "c": r.get("run_cards"), "a": r.get("run_ancient")}
                    if "version" in r and (r.get("run_cards") or r.get("run_ancient")) else None)
    rows = drop_reverted(incremental(RUN / "log.jsonl", RUN / "lite_picks.jsonl", pk, 60_000_000))
    # 看板上的“重新计数”（本机 tools/dashboard_serve.py 写 RUN/lite_since.json）：给了起点就从起点累计，否则取最近 PICK_WINDOW 版。
    # 每条记录是前 card_log_every 版的合计，只取整段都在起点之后的
    since = jload(RUN / "lite_since.json") or {}
    sv, every = since.get("version"), int(cfg.get("card_log_every") or 1)
    if rows:
        top = rows[-1]["version"]
        use = [r for r in rows if r["version"] - every >= sv] if sv is not None else [r for r in rows if r["version"] > top - PICK_WINDOW]
        def total(key):
            tot: dict = {}
            for r in use:
                for n, v in (r.get(key) or {}).items():
                    acc = tot.setdefault(n, [0] * len(v))
                    for j, x in enumerate(v):
                        acc[j] += x
            return tot
        out["picks"] = {"v0": use[0]["version"] - every + 1 if use else None, "v1": top, "n": len(use), "every": every, "since": since or None,
                        "cards": total("c"), "ancient": total("a")}
    if sv is not None:      # 自起点以来训练整局（采样出手）的通关率，按局数加权
        hs = [r for r in hist if r["version"] > sv and r.get("run_win_rate") is not None and r.get("run_episodes")]
        n = sum(r["run_episodes"] for r in hs)
        out["since"] = {**since, "runs": n, "run_win": round(sum(r["run_win_rate"] * r["run_episodes"] for r in hs) / n, 4) if n else None,
                        "v1": hs[-1]["version"] if hs else None}
    # 价值探针在势能前后的可比统计（tools/value_probe_bv.py）：较慢，最多每 30 分钟后台重算一次，读上次的结果
    vp, vp_out = RUN / "value_probe" / "episodes", RUN / "lite_vp.json"
    # 决策记录 261 起按这次上线的版本分组（上线前、上线后 300 版内、之后）；没有时按势能
    if vp.is_dir() and (split or cfg.get("bv_start")):
        try:
            age = time.time() - vp_out.stat().st_mtime
        except OSError:
            age = 1e9
        if age > 1800:
            vp_out.touch()
            args = (["--bv-start", str(split), "--bv-ramp", "300", "--names", "上线前,上线后 300 版内,300 版以后"] if split
                    else ["--bv-start", str(cfg["bv_start"]), "--bv-ramp", str(cfg.get("bv_ramp") or 0)])
            subprocess.Popen(["nice", "-n", "19", sys.executable, "-I", str(REPO / "tools/value_probe_bv.py"), "--dir", str(vp), *args, "--out", str(vp_out)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        out["value_probe"] = jload(vp_out)
    return out


def eval_pick(r: dict) -> dict | None:
    if "P1" not in str(r.get("seed_set")):
        return None
    runs = max(1, r.get("runs") or 1)
    b = r.get("bosses") or {}
    def boss(a):
        w = sum(x for k, (x, y) in b.items() if k.startswith(f"{a}:")); t = sum(y for k, (x, y) in b.items() if k.startswith(f"{a}:"))
        return [w, t]
    ch = r.get("choices") or {}
    cr = sum(v for k, v in ch.items() if k.startswith("CARD_REWARD:"))
    skip = sum(v for k, v in ch.items() if k.startswith("CARD_REWARD:") and "PICK" not in k)
    da = r.get("deck_at") or {}
    return {"version": r.get("version"), "time": r.get("time"), "runs": runs, "win": r.get("win_rate"), "act2": r.get("reach_act2"),
            "act3": r.get("reach_act3"), "floor": r.get("floor_mean"), "floor_med": r.get("floor_median"), "floor_p90": r.get("floor_p90"),
            "boss": [boss(0), boss(1), boss(2)],
            "deck_act3": da.get("act3_first"), "deck_boss3": da.get("boss3"),
            # 进到第三幕第一个、第二个 Boss 的局数（deck_at 的局数）：两者相除即第一个 Boss 的过关率
            "n_boss3": (da.get("boss3") or [0])[0], "n_boss3b": (da.get("boss3b") or [0])[0],
            "skip": round(skip / cr, 4) if cr else None,
            "heal": round(ch.get("REST_SITE:REST_OPTION:REST:HEAL", 0) / runs, 3), "smith": round(ch.get("REST_SITE:REST_OPTION:REST:SMITH", 0) / runs, 3),
            # 死因与各 Boss 过关率（diag() 汇总用，不随 evals 传回本机）：deaths、bosses 原样；bosses3 为第三幕两个 Boss 分开记（评测 2026-11-07 起才有）
            "d": r.get("deaths") or {}, "be": b, "b3": r.get("bosses3")}


DIAG_ROUNDS = 20            # 死因与各 Boss 过关率：最近这么多轮合计，另给再往前同样多轮作对照
DIAG_TOP = 60               # 具体死因（遭遇）只传前这么多项（看板显示前 15 项，多传的给前一窗口对照时查得到）


def diag(rs: list[dict]) -> dict | None:
    """几轮评测合计的死因与各 Boss 过关率（2026-11-07，用户：“能在看板上加上死因吗？第三幕BOSS分第一个和第二个显示。”
    “然后每一个BOSS的通关率也加上。第三幕BOSS同样分第一个和第二个。”）。
    stages：各幕普通、精英、Boss 战与非战斗的死亡局数和通关局数，第三幕 Boss 分 "2:Boss1"、"2:Boss2"（没有 bosses3 的轮按到第二个 Boss 的局数拆：
    死在第二个 = 到第二个的局 − 通关）；causes：具体遭遇的死亡局数，第三幕 Boss 的键为 "2:Boss1:遭遇"、"2:Boss2:遭遇"，拆不开的仍是 "2:Boss:遭遇"；
    bosses：各 Boss 的 [打赢场数, 场数]，键 "幕:遭遇"，第三幕分 "2.1:遭遇"、"2.2:遭遇"（只算有 bosses3 的轮，拆不开的仍记 "2:遭遇"）；
    slots：四个 Boss 关口合计的 [过关, 打到]，第三幕两个按到第一个、第二个 Boss 的局数（各轮都有）。"""
    if not rs:
        return None
    runs = sum(r["runs"] for r in rs)
    wins = sum(round((r.get("win") or 0) * r["runs"]) for r in rs)
    stages: dict[str, int] = {}
    causes: dict[str, int] = {}
    bosses: dict[str, list[int]] = {}
    slots = {"boss1": [0, 0], "boss2": [0, 0], "boss3a": [0, 0], "boss3b": [0, 0]}
    split = 0
    add = lambda d, k, v: d.__setitem__(k, d.get(k, 0) + v)
    def addb(k, w, n):
        t = bosses.setdefault(k, [0, 0]); t[0] += w; t[1] += n
    for r in rs:
        d, b3 = r.get("d") or {}, r.get("b3")
        for k, v in d.items():
            parts = k.split(":")
            if len(parts) != 3:
                add(stages, k, v); add(causes, k, v)
                continue
            if parts[0] == "2" and parts[1] == "Boss" and b3 is not None:
                continue            # 第三幕 Boss 按 bosses3 分第几个记（下面）
            if not (parts[0] == "2" and parts[1] == "Boss"):
                add(stages, f"{parts[0]}:{parts[1]}", v)
            add(causes, k, v)
        if b3 is not None:
            split += 1
            for k, (w, n) in b3.items():
                s, enc = k.split(":", 1)
                if n > w:
                    add(stages, f"2:Boss{s}", n - w); add(causes, f"2:Boss{s}:{enc}", n - w)
                addb(f"2.{s}:{enc}", w, n)
        else:
            d3 = sum(v for k, v in d.items() if k.startswith("2:Boss:"))
            s2 = max(0, min(d3, (r.get("n_boss3b") or 0) - round((r.get("win") or 0) * r["runs"])))
            add(stages, "2:Boss1", d3 - s2); add(stages, "2:Boss2", s2)
        for k, (w, n) in (r.get("be") or {}).items():
            if not (k.startswith("2:") and b3 is not None):
                addb(k, w, n)
        for i, key in ((0, "boss1"), (1, "boss2")):
            slots[key][0] += r["boss"][i][0]; slots[key][1] += r["boss"][i][1]
        n1, n2 = r.get("n_boss3") or 0, r.get("n_boss3b") or 0
        slots["boss3a"][0] += n2; slots["boss3a"][1] += n1
        slots["boss3b"][0] += round((r.get("win") or 0) * r["runs"]); slots["boss3b"][1] += n2
    top = dict(sorted(causes.items(), key=lambda x: -x[1])[:DIAG_TOP])
    return {"rounds": len(rs), "v0": rs[0]["version"], "v1": rs[-1]["version"], "runs": runs, "wins": wins, "split": split,
            "split_v0": next((r["version"] for r in rs if r.get("b3") is not None), None),
            "stages": stages, "causes": top, "bosses": dict(sorted(bosses.items())), "slots": slots}


def retests() -> list[dict]:
    """手动配对复测（/root/diag/RT_*，tools/remote/diag_queue.sh 打的 P1 1,000 种）：通关率、95% 区间，与第一份（基线）按种子配对。"""
    rows = []
    for d in sorted(Path("/root/diag").glob("RT_*")):
        w = {}
        for f in d.glob("runs_s*.jsonl"):
            for l in open(f, encoding="utf-8"):
                if l.strip():
                    x = json.loads(l); w[x["seed"]] = int(x["win"])
        if w:
            rows.append((d.name, w))
    out = []
    base = rows[0][1] if rows else {}
    for name, w in rows:
        n = len(w); p = sum(w.values()) / n
        rec = {"name": name, "n": n, "win": round(p, 4), "pm": round(1.96 * math.sqrt(p * (1 - p) / n), 4)}
        common = [s for s in w if s in base]
        if name != rows[0][0] and common:
            x = sum(w[s] and not base[s] for s in common); y = sum(base[s] and not w[s] for s in common)
            k, m = min(x, y), x + y
            rec.update(only_new=x, only_base=y, p=round(min(1.0, 2 * sum(math.comb(m, i) for i in range(k + 1)) / 2 ** m), 4) if m else 1.0)
        out.append(rec)
    return out


EVAL_RECENT = 150           # 完整记录只带最近这么多轮（看板的三栏对照、护栏表、最近一次评测用）
# pass3a：打过第三幕第一个 Boss（打到第二个 Boss）的局占全部局，由 n_boss3b / runs 算（2026-11-07，看板“打过各幕 Boss 的比例”加这条曲线）
EVAL_OLD_KEYS = ("win", "act2", "act3", "floor", "floor_med", "pass3a")


def old_val(r: dict, k: str):
    if k == "pass3a":
        return r["n_boss3b"] / r["runs"] if r.get("n_boss3b") is not None and r.get("runs") else None
    return r.get(k)


def evals_old(ev: list[dict], n: int = 150) -> list[list]:
    """最近 EVAL_RECENT 轮之前的评测，只留走势图要的几项，每点 [版本, *EVAL_OLD_KEYS]（2026-11-07，用户：“为什么看板上早期的评测数据之类的看不见？”
    原来只传最近 150 轮，储君约 4 分钟一轮，走势图只剩最近十来个小时）。超过 n 轮时相邻几轮合成一点（取平均，版本取这几轮的最后一轮）。"""
    rows = [[r["version"]] + [old_val(r, k) for k in EVAL_OLD_KEYS] for r in ev[:-EVAL_RECENT]]
    g = max(1, math.ceil(len(rows) / n))
    out = []
    for i in range(0, len(rows), g):
        grp = rows[i:i + g]
        out.append([grp[-1][0]] + [round(sum(x) / len(x), 4) if (x := [r[j] for r in grp if r[j] is not None]) else None
                                   for j in range(1, len(EVAL_OLD_KEYS) + 1)])
    return out


def evaluate() -> dict:
    # 缓存 lite_eval.jsonl → lite_eval2.jsonl（2026-11-07，每轮多存死因与各 Boss 的战绩）：换名后从头重读一遍评测记录，旧的轮也有死因
    ev = drop_reverted(incremental(EVAL / "eval.jsonl", EVAL / "lite_eval2.jsonl", eval_pick, 10 ** 12))
    try:
        err_m = (EVAL / "eval_error.txt").stat().st_mtime
    except OSError:
        err_m = None
    full = [r for r in ev if r.get("runs") and r.get("boss")]
    dg = {"cur": diag(full[-DIAG_ROUNDS:]), "prev": diag(full[-2 * DIAG_ROUNDS:-DIAG_ROUNDS])}
    slim = [{k: v for k, v in r.items() if k not in ("d", "be", "b3")} for r in ev[-EVAL_RECENT:]]
    return {"role": "eval", "time": time.time(), "progress": jload(EVAL / "eval_progress.json"), "error_mtime": err_m,
            "paused": (EVAL / "PAUSE").exists(), "evals": slim, "evals_old": evals_old(ev), "diag": dg,
            "retests": retests() if EVAL.name == "defect-eval" else []}


if __name__ == "__main__":
    role = sys.argv[1] if len(sys.argv) > 1 else "train"
    (RUN if role == "train" else EVAL).mkdir(parents=True, exist_ok=True)     # 新训练线刚建、目录还没有时也能出一份空统计
    print(json.dumps(train() if role == "train" else evaluate(), ensure_ascii=False, separators=(",", ":")))
