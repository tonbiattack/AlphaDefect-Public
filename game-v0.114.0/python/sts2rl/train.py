"""学习主循环：启动采集进程，收集轨迹，做 PPO 更新，发布权重，记录指标，存检查点。

  ../.venv/bin/python -I -m sts2rl.train --actors 2 --envs-per-actor 6 --total-steps 2000000
"""
from __future__ import annotations

import contextlib
import json
import multiprocessing as mp
import os
import queue
import shutil
import signal
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sts2env.protocol import FORMAT_VERSION, DecisionKind, TokenKind, Zone, format_for

import threading

from .actor import DECK_BINS, MODE_COMBAT, MODE_RUN, deck_bin, merge_counts, run_actor
from .evaluator import run_evaluator
from .resources import start_sampler
from .solver_pool import LabelStats
from .config import TrainConfig
from .distill import DistillBuffer
from .features import Ragged, collate_ragged
from .catalog import Catalog
from .model import CO_POT_SLOTS, CombatOutcomeHeads, PolicyValueNet, checkpoint_content_names, load_state, packed_meta, use_tf32, zero_new_vocab
from .content import CATALOG_VERSION, remap_columns
from .muon import Muon, select as muon_select
from .search_kl import kl_loss as search_kl_loss
from .schedulefree import AdamScheduleFree
from . import winprob_online as wpo

# 检查点开始记录词表大小（vocab_n）之前，词表都是 v0.107.1 的 1,873 项
VOCAB_N_LEGACY = 1873


def publish(net, path: Path, version: int, progress: float, shaping: float = 0.0, extra: dict | None = None) -> None:
    """extra：随权重一起发给采集进程的其他内容（胜率势能的模型与系数，决策记录 189）。"""
    tmp = path.with_suffix(".tmp")
    torch.save({"version": version, "progress": progress, "shaping": shaping,
                "state": {k: v.detach().cpu() for k, v in net.state_dict().items()}, **(extra or {})}, tmp)
    os.replace(tmp, path)


def gae(values: np.ndarray, rewards: np.ndarray, gamma: float, lam: float, cut: np.ndarray | None = None, boot: float = 0.0,
        nv: np.ndarray | None = None, luck: list | None = None) -> tuple[np.ndarray, np.ndarray]:
    """单条轨迹或整局的一段；boot 为最后一步之后那一步的价值（对局结束为 0，整局中途送出的段为下一段第一步的价值）。
    cut 为 True 的步（强制探索改过的离策略步）：前面各步的优势到这一步截断，用它的价值估计自举。
    nv（运气校正，config.luck_cv，铁甲战士决策记录第 20 条）：有限值的步，δ 里的下一步价值换成“平均运气下”的 V̄；
    luck 不为 None 时累加 [步数, Σ运气, Σ运气², Σδ², Σδ'²]（运气 = 真实下一步价值 − V̄），只统计、不影响结果。"""
    n = len(values)
    next_v = np.append(values[1:], boot)
    if nv is not None and len(nv) == n and (m := np.isfinite(nv)).any():
        if luck is not None:
            lk = next_v[m] - nv[m]
            d0 = rewards[m] + gamma * next_v[m] - values[m]
            luck[0] += int(m.sum()); luck[1] += float(lk.sum()); luck[2] += float((lk * lk).sum())
            luck[3] += float((d0 * d0).sum()); luck[4] += float(((d0 - gamma * lk) ** 2).sum())
        next_v = np.where(m, nv, next_v)
    delta = rewards + gamma * next_v - values
    adv = np.zeros(n, np.float32); acc = 0.0
    for i in range(n - 1, -1, -1):
        acc = delta[i] + gamma * lam * acc
        adv[i] = acc
        if cut is not None and cut[i]:
            acc = 0.0
    return adv, adv + values


# 梯度分解（诊断）：主干 = 编码器与各类嵌入、汇总、绑定网络等策略和价值共用的部分；这些前缀是各个头与只进价值头的特权通道
_HEAD_PREFIX = ("cand_mlp", "value_mlp", "value_hl", "value_aux", "phi_mlp", "priv_", "fut_", "act_emb", "cand_id_emb", "cand_feat", "none_ref", "cand_v19")


def grad_decomp_step(net, terms: dict, acc: dict) -> None:
    """一个小批：各项损失分别对主干参数求梯度，累加范数、与策略梯度的余弦、与总梯度的余弦（在总梯度方向上的分量占比）。"""
    trunk = [p for n, p in net.named_parameters() if p.requires_grad and not n.startswith(_HEAD_PREFIX)]
    g = {}
    for k, t in terms.items():
        gs = torch.autograd.grad(t, trunk, retain_graph=True, allow_unused=True)
        g[k] = torch.cat([(x if x is not None else torch.zeros_like(p)).reshape(-1).float() for x, p in zip(gs, trunk)])
    tot = g["total"]; pol = g["policy"]
    for k, v in g.items():
        a = acc.setdefault(k, [0.0, 0.0, 0.0, 0.0])
        a[0] += float(v.norm()); a[1] += float(F.cosine_similarity(v, pol, 0)) if k != "policy" else 1.0
        a[2] += float(F.cosine_similarity(v, tot, 0)); a[3] += float((v @ tot) / (tot @ tot).clamp(min=1e-30))
    acc["n"] += 1


def grad_decomp_report(acc: dict, version: int) -> dict:
    n = acc["n"]
    return {"version": version, "minibatches": n,
            "terms": {k: {"trunk_grad_norm": round(a[0] / n, 5), "cos_with_policy": round(a[1] / n, 4), "cos_with_total": round(a[2] / n, 4),
                          "share_of_total": round(a[3] / n, 4)} for k, a in acc.items() if k != "n"}}


def ppg_aux_phase(net, fwd, opt, buf: list[dict], cfg, device, own_clip, hl_coef_cur: float, w_scalar: float) -> dict:
    """阶段式策略梯度的辅助阶段（决策记录 266）：buf 为最近几版存下的 {"data", "ret", 辅助头标签}。先按固定分块记下当时的策略对数概率，
    再训 cfg.ppg_epochs 遍：价值损失（与 PPO 里相同：数值头均方误差 × w_scalar、分类头交叉熵 × hl_coef_cur、辅助头）× vf_coef
    + ppg_beta_clone × KL(当时的策略 ‖ 现在的策略)。梯度累积、裁剪、跳过非有限梯度都与 PPO 相同。返回统计。"""
    t0 = time.time()
    data = Ragged.concat([x["data"] for x in buf]); n = len(data)
    per = {"ppo_ret": np.concatenate([x["ret"] for x in buf])}
    if cfg.model.combat_aux:
        for k in ("aux_m", "aux_win", "aux_hp"):
            per[k] = np.concatenate([x[k] for x in buf])
    size = cfg.minibatch
    perm = np.concatenate([np.sort(c) for c in np.array_split(np.random.permutation(n), max(1, -(-n // size)))])
    old = []
    net.eval()
    with torch.no_grad():
        for _, b in minibatches(data, perm, size, device, None, per, packed=net.fast, workers=cfg.collate_workers):
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=cfg.learner_bf16 and device.type == "cuda"):
                lg = fwd(b)[0]
            old.append(F.log_softmax(lg.float(), -1).cpu())
    net.train()
    acc_n = max(1, int(cfg.grad_accum)); n_mb = len(old)
    sums = torch.zeros(3, device=device); cnt = 0; skipped = 0
    for _ in range(cfg.ppg_epochs):
        k_mb = 0
        for j, (_, b) in enumerate(minibatches(data, perm, size, device, None, per, packed=net.fast, workers=cfg.collate_workers)):
            R = b["ppo_ret"]
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=cfg.learner_bf16 and device.type == "cuda"):
                logits, v, _phi, ex = fwd(b)
            lp = F.log_softmax(logits.float(), -1); olp = old[j].to(device)
            kl = (olp.exp() * (olp - lp)).nan_to_num(0.0, 0.0, 0.0).sum(-1).mean()
            v = v.float()
            if cfg.model.value_hl:
                l_v = w_scalar * F.mse_loss(ex["v_scalar"].float(), R) + hl_coef_cur * (-(hl_target(R, net.hl_edges, cfg.hl_sigma) * F.log_softmax(ex["hl_logits"].float(), -1)).sum(-1).mean())
            else:
                l_v = F.mse_loss(v, R)
            loss = cfg.vf_coef * l_v + cfg.ppg_beta_clone * kl
            if cfg.model.combat_aux:
                am = b["aux_m"]; na = am.sum().clamp(min=1); ao = ex["aux"].float()
                loss = loss + cfg.aux_coef * ((F.binary_cross_entropy_with_logits(ao[:, 0], b["aux_win"], reduction="none") * am).sum() / na
                                              + cfg.aux_hp_w * (((ao[:, 1] - b["aux_hp"]) ** 2) * am).sum() / na)
            if k_mb % acc_n == 0:
                opt.zero_grad(set_to_none=True); g_size = min(acc_n, n_mb - k_mb)
            (loss / g_size).backward(); k_mb += 1
            if k_mb % acc_n == 0 or k_mb >= n_mb:
                skipped += int(guarded_step(net, opt, cfg.max_grad_norm, own_clip).item())
            with torch.no_grad():
                sums += torch.stack([F.mse_loss(v, R), kl, loss.detach()]); cnt += 1
    net.eval()
    s = (sums / max(cnt, 1)).tolist()
    return {"ppg_steps": n, "ppg_loss_v": round(s[0], 5), "ppg_kl": round(s[1], 6), "ppg_loss": round(s[2], 5), "ppg_skipped": skipped,
            "ppg_seconds": round(time.time() - t0, 1)}


def hl_target(r: torch.Tensor, edges: torch.Tensor, sigma: float) -> torch.Tensor:
    """HL-Gauss 软标签（决策记录 261）：以回报为中心、标准差 sigma × 桶宽的正态分布落在各桶里的概率，范围外的回报先截到两端，再按桶内总概率归一。"""
    s = sigma * float(edges[1] - edges[0]) * 2 ** 0.5
    cdf = torch.special.erf((edges[None, :] - r.clamp(float(edges[0]), float(edges[-1]))[:, None]) / s)
    p = cdf[:, 1:] - cdf[:, :-1]
    return p / p.sum(-1, keepdim=True).clamp(min=1e-12)


LAG_BINS = 11

# 局外学习诊断（决策记录 200，只统计）：ppo_g 的分组（多选局外决策为 决策类型 × 2 + 是否改选；古神三选一另成 DIAG_ANCIENT + 是否改选，
# 决策记录 202），每组每遍累加 DIAG_COLS 这几项
DIAG_COMBAT, DIAG_SINGLE, DIAG_ANCIENT = 32, 33, 34
DIAG_COMBAT_F = DIAG_ANCIENT + 2         # 战斗定向探索改选的战斗步（决策记录 238）
DIAG_N = DIAG_COMBAT_F + 1
DIAG_COLS = ("n", "r", "r2", "clip", "extreme", "lp", "lp_pos", "n_pos", "lp_neg", "n_neg", "cut_pos", "cut_neg")


def macro_diag(d: np.ndarray) -> dict:
    """d：[遍数, 组, 列]。返回 {决策类型: {"pi": [...], "forced": [...]}}，战斗步记在 "COMBAT"（战斗定向探索改选的步在 forced 下），古神三选一记在 "ANCIENT"（不再算进 EVENT）。
    每项 [步数, 比率均值, 裁剪比例, 有效样本比例, 比率 < 0.5 或 > 2 的比例, 第一遍平均对数概率, 正优势步对数概率的变化, 负优势步对数概率的变化,
    正优势步里策略梯度真被截掉的比例, 负优势步里的这个比例]。裁剪比例只看比率出不出界；真被截掉要看优势的正负（第十一份外部评审，决策记录 204）：
    正优势且比率 > 1 + ε、负优势且比率 < 1 − ε 时 min(r·A, clip(r)·A) 取到截断的一边，这一步没有梯度。
    比率、裁剪、有效样本取第一遍（参数还接近更新前，比率就是 π/μ 的重要性权重）；有效样本比例 = (Σr)² / Σr² / 步数。
    变化 = 最后一遍减第一遍的平均对数概率（两遍里参数都在动，看方向和大小：正优势的该升、负优势的该降）。"""
    f, l = d[0], d[-1]
    r4 = lambda x: None if x is None else round(float(x), 4)

    def row(g: int) -> list | None:
        n = f[g, 0]
        if n < 1:
            return None
        dp = l[g, 6] / max(l[g, 7], 1) - f[g, 6] / f[g, 7] if f[g, 7] > 0 else None
        dn = l[g, 8] / max(l[g, 9], 1) - f[g, 8] / f[g, 9] if f[g, 9] > 0 else None
        return [int(n), r4(f[g, 1] / n), r4(f[g, 3] / n), r4(f[g, 1] ** 2 / max(f[g, 2], 1e-12) / n), r4(f[g, 4] / n), r4(f[g, 5] / n), r4(dp), r4(dn),
                r4(f[g, 10] / f[g, 7]) if f[g, 7] > 0 else None, r4(f[g, 11] / f[g, 9]) if f[g, 9] > 0 else None]
    out = {}
    for k in range(DIAG_COMBAT // 2):
        e = {lab: x for lab, x in (("pi", row(2 * k)), ("forced", row(2 * k + 1))) if x is not None}
        if e:
            out[DecisionKind(k).name if k in DecisionKind._value2member_map_ else str(k)] = e
    if e := {lab: x for lab, x in (("pi", row(DIAG_ANCIENT)), ("forced", row(DIAG_ANCIENT + 1))) if x is not None}:
        out["ANCIENT"] = e
    if e := {lab: x for lab, x in (("pi", row(DIAG_COMBAT)), ("forced", row(DIAG_COMBAT_F))) if x is not None}:
        out["COMBAT"] = e
    return out


def cur_key(item: dict) -> str:
    """课程统计（决策记录 197）的类别：整局为 run（中途开局 mid）；战斗练习按 slot，Boss 练习再分幕与连打、第二个 Boss、中途牌组挑战，
    真实局面与随机构筑分普通战、精英战。"""
    if item["mode"] == MODE_RUN:
        return "mid" if item.get("start_act") else "run"
    slot = item.get("slot") or "progress"
    if slot == "boss":
        a = int(item.get("act") or 0) + 1
        return f"boss{a}" + ("_chain" if item.get("chain") else "_second" if item.get("second") else "_probe" if item.get("probe") else "")
    return f"{slot}_{str(item.get('room') or '').lower()}" if slot in ("act1", "act2", "act3", "mix") else slot


def intake_stats(items: list[dict], runs: list[dict]) -> dict:
    """（决策记录 168，只统计、不影响学习）整局各步有没有进学习，以及强制探索改选的去向。items 为这一批收到的整局轨迹
    （分段提交时含中途送出的段），runs 为这一批结束的整局；都含全部过旧、没进学习的。
    run_fresh_act：各幕 [收到步数, 入训步数]，幕取该步观测里的幕（原来还有各幕落后版本数 run_lag_act，决策记录 197 起由 lag_split 代替）；run_fresh_kind：局外决策按类型 [收到, 入训, 改选, 改选且入训]；
    explore_td：局外决策沿决策序列的一步差（两次决策间的奖励 + 下一次决策的价值 − 这一步的价值，局末下一次价值为 0）
    按类型 [步数, 合计]，键 "forced:类型" 为改选步、"natural:类型" 为其余（古神计在 EVENT 里）；
    explore_runs：[局数, 层数合计, 通关数]，"none" 为没有改选的局，其余键为局里改选过这一类的局（类别同 explore_probs）。"""
    fresh_act = [[0, 0] for _ in range(3)]
    fresh_kind: dict[str, list[int]] = {}
    td: dict[str, list[float]] = {}
    out_runs: dict[str, list[int]] = {}
    for t in items:
        hd = t["ragged"].header
        act = np.clip(hd[:, 6], 0, 2); fr = t["fresh"]; macro = t["macro"]
        forced = t["forced"] if t.get("forced") is not None and len(t["forced"]) == t["len"] else np.zeros(t["len"], bool)
        for a in range(3):
            m = act == a
            fresh_act[a][0] += int(m.sum()); fresh_act[a][1] += int((m & fr).sum())
        idx = np.flatnonzero(macro)
        csum = np.concatenate([[0.0], np.cumsum(t["rewards"], dtype=np.float64)])
        for j, i in enumerate(idx):
            name = DecisionKind(int(hd[i, 1])).name
            c = fresh_kind.setdefault(name, [0, 0, 0, 0])
            c[0] += 1; c[1] += int(fr[i]); c[2] += int(forced[i]); c[3] += int(forced[i] and fr[i])
            nxt = idx[j + 1] if j + 1 < len(idx) else len(t["rewards"])
            d = csum[nxt] - csum[i] + (float(t["values"][nxt]) if nxt < len(t["values"]) else t.get("boot", 0.0)) - float(t["values"][i])
            s = td.setdefault(("forced:" if forced[i] else "natural:") + name, [0, 0.0]); s[0] += 1; s[1] += d
    for t in runs:
        kinds = [k[8:] for k, n in t["choices"].items() if k.startswith("EXPLORE:") and n > 0] or ["none"]
        for k in kinds:
            r = out_runs.setdefault(k, [0, 0, 0]); r[0] += 1; r[1] += int(t["floor"]); r[2] += int(t["win"])
    return {"run_fresh_act": fresh_act, "run_fresh_kind": fresh_kind,
            "explore_td": {k: [n, round(s, 3)] for k, (n, s) in td.items()}, "explore_runs": out_runs}


def fight_cuts(header: np.ndarray, at_map: bool = False) -> np.ndarray:
    """每场战斗打完后的第一步为 True（决策记录 270，combat_fight_boot）：前一步在战斗中（头部 11），这一步不在战斗中，或层数（7）、
    连打的第几场（23）变了。作 gae 的 cut 用：战斗步的优势到这一步截断，最后一个战斗步用这一步（战后局面）的价值自举。
    at_map（combat_fight_boot_map，愚者决策记录第 8 条）：截断点从战后第一步往后挪到战后第一次选路（头部 1 为 MAP），
    或下一场战斗的第一步（连打第二场、事件里接着开打），以先到者为准；这一段里一直没有就不截（段末照常用 boot 自举）。
    领奖、选牌这些界面的局外估值有局部偏差（第 10,016 版 32 局：领奖界面比战斗最后一步低 0.07–0.25，领完又涨回来），
    在战后第一步自举时打完那一下吃到整个落差，拖着不打完反而优势高（小叶史莱姆的僵持）；挪到选路处，落差与随后的回升一起算，基本抵消。"""
    n = len(header)
    out = np.zeros(n, bool)
    if n > 1:
        inc = header[:, 11] != 0
        key_same = (header[1:, 7] == header[:-1, 7]) & (header[1:, 23] == header[:-1, 23])
        out[1:] = inc[:-1] & (~inc[1:] | ~key_same)
        if at_map:
            ends = np.flatnonzero(out)
            stops = np.flatnonzero((header[:, 1] == DecisionKind.MAP) | inc)    # 战后第一步若已在下一场战斗里，就停在它本身
            pos = np.searchsorted(stops, ends)
            out[:] = False
            out[stops[pos[pos < len(stops)]]] = True
    return out


def fight_spans(header: np.ndarray) -> list[tuple[int, int, int]]:
    """整局一段里每场打完、且这一段里找得到战后截断点（fight_cuts 的 at_map 规则）的战斗：(第一个战斗步, 最后一个战斗步, 幕)。
    段首就在战斗中的，第一个战斗步记为 0。"""
    n = len(header)
    if n < 2:
        return []
    inc = header[:, 11] != 0
    key_same = (header[1:, 7] == header[:-1, 7]) & (header[1:, 23] == header[:-1, 23])
    end = np.zeros(n, bool)
    end[1:] = inc[:-1] & (~inc[1:] | ~key_same)
    start = np.zeros(n, bool)
    start[0] = inc[0]
    start[1:] = inc[1:] & (~inc[:-1] | ~key_same)
    starts = np.flatnonzero(start)
    stops = np.flatnonzero((header[:, 1] == DecisionKind.MAP) | inc)
    out = []
    for e in np.flatnonzero(end):
        if np.searchsorted(stops, e) >= len(stops):
            continue
        last = int(e) - 1
        out.append((int(starts[np.searchsorted(starts, last, side="right") - 1]), last, int(header[last, 6])))
    return out


def fight_end_center(trajs, advs, rets, cuts, gl: float, min_n: int = 32) -> dict:
    """combat_fight_center（愚者决策记录第 8 条，要 combat_fight_boot 与 combat_fight_boot_map）：这一批整局里，每场战斗最后一步的优势
    （从打完那一下一直算到战后第一次选路）按幕取平均，从这场战斗各步的优势与价值目标里减掉 平均 × (γλ)^(到最后一步的步数)。
    两个价值头一致时这个平均应为 0；不为 0 的部分是战斗价值头与局外价值头在战斗收尾处的系统落差，它随“几步之后打完”变化，
    会给“早打完 / 晚打完”加偏置（第 10,016 版：截在战后第一步时第一幕 −0.13、第一二幕 Boss −0.21～−0.26；截在选路时第二三幕 +0.05～+0.16）。
    只减平均，不增加方差；某一幕不到 min_n 场时这一幕不动。cuts 为各轨迹 gae 用的截断（这场战斗里有截断时只改截断处及之后的步）。
    返回 {幕: [场数, 平均]}（平均为减之前的值，看板与日志用）。"""
    spans = []
    for k, t in enumerate(trajs):
        if t.get("deferred") or t["mode"] != MODE_RUN:
            continue
        for s, last, act in fight_spans(t["ragged"].header):
            cin = np.flatnonzero(cuts[k][s:last + 1])
            spans.append((k, s + (int(cin[-1]) if len(cin) else 0), last, act))
    by: dict[int, list[float]] = {}
    for k, s, last, act in spans:
        by.setdefault(act, []).append(float(advs[k][last]))
    mean = {a: float(np.mean(v)) for a, v in by.items() if len(v) >= min_n}
    for k, s, last, act in spans:
        m = mean.get(act)
        if m is None:
            continue
        w = (gl ** np.arange(last - s, -1, -1)).astype(np.float32) * m
        advs[k][s:last + 1] -= w
        rets[k][s:last + 1] -= w
    return {int(a): [len(v), round(float(np.mean(v)), 4)] for a, v in sorted(by.items())}


def gae_macro(values: np.ndarray, rewards: np.ndarray, macro: np.ndarray, lam: float, adv: np.ndarray, ret: np.ndarray, cut: np.ndarray | None = None,
              boot: float = 0.0) -> None:
    """整局里的非战斗决策：只沿这些决策的序列算优势（γ=1），两个决策之间所有步的奖励（爬层、过 Boss、牌组打分）
    并到前一个决策上，直接改写 adv、ret 里这些步的值。战斗步保持逐步的 GAE。
    boot 同 gae：整局分段时下一段总从选路线决策开始，它的价值也就是这一段最后一个决策之后那个决策的价值。"""
    idx = np.flatnonzero(macro)
    if not len(idx):
        return
    cs = np.concatenate([[0.0], np.cumsum(rewards, dtype=np.float64)])
    R = cs[np.append(idx[1:], len(rewards))] - cs[idx]
    v = values[idx].astype(np.float64)
    delta = R + np.append(v[1:], boot) - v
    acc = 0.0
    for j in range(len(idx) - 1, -1, -1):
        acc = delta[j] + lam * acc
        adv[idx[j]] = acc
        if cut is not None and cut[idx[j]]:
            acc = 0.0
    ret[idx] = adv[idx] + values[idx]


def macro_defer(item: dict, pending: dict, version: int, cfg) -> dict | None:
    """局外优势跨段计算（决策记录 255）：把这一段的局外决策挪进 pending（并从这一段的入训步里去掉）；这一局的最后一段到了时，
    沿整条局外决策链（γ = 1，两个决策之间的奖励并到前一个上，同 gae_macro）算 λ 优势与目标，返回一条只含局外步的“延后轨迹”，
    过旧判断用 macro_defer_lag。还没到最后一段时返回 None。"""
    idx = np.flatnonzero(item["macro"])
    gid = item["gid"]
    if len(idx):
        cs = np.concatenate([[0.0], np.cumsum(item["rewards"], dtype=np.float64)])
        mu_ok = item.get("logp_mu") is not None and len(item["logp_mu"]) == item["len"]
        forced = item["forced"] if item.get("forced") is not None and len(item["forced"]) == item["len"] else np.zeros(item["len"], bool)
        anc = item["anc"] if item.get("anc") is not None and len(item["anc"]) == item["len"] else np.zeros(item["len"], bool)
        piece = {"ragged": item["ragged"].select(idx), "actions": item["actions"][idx], "logp": item["logp"][idx],
                 "logp_mu": (item["logp_mu"] if mu_ok else item["logp"])[idx], "forced": forced[idx], "anc": anc[idx],
                 "values": item["values"][idx], "R": cs[np.append(idx[1:], item["len"])] - cs[idx], "vers": item["vers"][idx],
                 "sig": item["sig"][idx] if item.get("sig") is not None and len(item["sig"]) == item["len"] else np.zeros(len(idx), np.int64)}
        if cfg.macro_seg_adv:       # 按段自举（决策记录 269）：这一段的局外优势现在就按不跨段的办法算好，最后一段的 boot 为 0
            a_ = np.zeros(len(idx), np.float32); r_ = np.zeros(len(idx), np.float32)
            gae_macro(piece["values"], piece["R"].astype(np.float32), np.ones(len(idx), bool), cfg.macro_lam, a_, r_, piece["forced"],
                      float(item.get("boot", 0.0)))
            piece["adv"], piece["ret"] = a_, r_
        item["fresh"][idx] = False
    else:
        piece = None
    ent = pending.setdefault(gid, [time.time(), []])
    ent[0] = time.time()
    if piece is not None:
        ent[1].append(piece)
    if item.get("segment"):
        return None
    pieces = pending.pop(gid)[1]
    if not pieces:
        return None
    cat = lambda k: np.concatenate([p[k] for p in pieces])
    v, R, forced = cat("values"), cat("R"), cat("forced")
    n = len(v)
    if cfg.macro_seg_adv:
        adv, ret = cat("adv"), cat("ret")
    else:
        adv = np.zeros(n, np.float32); ret = np.zeros(n, np.float32)
        gae_macro(v, R.astype(np.float32), np.ones(n, bool), cfg.macro_lam, adv, ret, forced, 0.0)
    vers = cat("vers")
    return {"deferred": True, "segment": True, "mode": MODE_RUN, "remote": item.get("remote", False), "len": n,
            "ragged": Ragged.concat([p["ragged"] for p in pieces]), "actions": cat("actions"), "logp": cat("logp"), "logp_mu": cat("logp_mu"),
            "forced": forced, "anc": cat("anc"), "values": v, "rewards": R.astype(np.float32), "vers": vers, "macro": np.ones(n, bool),
            "adv": adv, "ret": ret, "boot": 0.0, "sig": cat("sig"),
            "fresh": (version - vers >= 0) & (version - vers <= cfg.macro_defer_lag)}


def group_release(members: list, version: int, cfg, acc: dict, c: float = 1.0) -> list[dict]:
    """同种子成组采集（决策记录 267）：一组同种子整局的延后局外轨迹（macro_defer 的结果；没有局外步的局为 None）到齐后，
    每一步的优势减去 c × 同组其他局在同一局面（帧签名相同）上的原优势的平均；没有同伴的步不变。别的局怎么走与本局这一步的动作无关，
    这个基线不引入偏差：同组几局选得一样的那一步互相抵消（比不出好坏），选得不一样的那一步就是配对比较。
    c 为控制变量的系数：同伴的优势与本局的不相关时减掉它只会添噪声，所以学习端用以前各版的 Σ A·m / Σ m²（m 为同伴平均）估计、截到 [0, 1]，
    逐版滑动平均（冒烟里两版的方差比一次降一次升，见决策记录 267）。
    过旧重新按 macro_defer_lag 判断（等同组的局又过了几版）。acc 累加统计：组数、到齐的组数、局外步数、有同伴的步数、同伴数合计、
    有同伴的步上调整前后优势的平方和、Σ A·m 与 Σ m²、各局第一次没有同伴的局外步序号与层数。"""
    ms = [d for d in members if d is not None]
    acc["groups"] = acc.get("groups", 0) + 1
    acc["full"] = acc.get("full", 0) + int(len(members) >= cfg.run_group)
    if not ms:
        return []
    by_sig: dict[int, list[tuple[int, float]]] = {}
    for i, d in enumerate(ms):
        for s_, a_ in zip(d["sig"].tolist(), d["adv"].tolist()):
            if s_:
                by_sig.setdefault(s_, []).append((i, a_))
    for i, d in enumerate(ms):
        raw = d["adv"].copy(); first = None
        for j, s_ in enumerate(d["sig"].tolist()):
            peers = [a_ for k, a_ in by_sig.get(s_, ()) if k != i] if s_ else []
            if peers:
                m_ = float(np.mean(peers))
                d["adv"][j] = raw[j] - c * m_
                acc["am"] = acc.get("am", 0.0) + float(raw[j]) * m_; acc["mm"] = acc.get("mm", 0.0) + m_ * m_
                acc["peer_steps"] = acc.get("peer_steps", 0) + 1; acc["peers"] = acc.get("peers", 0) + len(peers)
                acc["sq_raw"] = acc.get("sq_raw", 0.0) + float(raw[j]) ** 2; acc["sq_adj"] = acc.get("sq_adj", 0.0) + float(d["adv"][j]) ** 2
            elif first is None:
                first = j
        acc["steps"] = acc.get("steps", 0) + len(raw)
        if first is not None and len(ms) > 1:
            acc.setdefault("div_idx", []).append(first)
            acc.setdefault("div_floor", []).append(int(d["ragged"].header[first, 7]))
        d["fresh"] = (version - d["vers"] >= 0) & (version - d["vers"] <= cfg.macro_defer_lag)
    return ms


def save_checkpoint(path: Path, net, opt, cfg, version: int, progress: float, total_steps: int, phi_steps: int = 0, vocab_n: int = 0,
                    extra: dict | None = None) -> None:
    """先写临时文件再改名：写到一半被终止也不会留下损坏的检查点。vocab_n 为当时的词表大小，游戏更新后据此找出新增的 ID。
    参数里有 NaN、inf 时不存，留着上一份（2026-10-29 第 68,160 版起全成 NaN，崩溃时又把坏模型存了进去，决策记录 173）。"""
    if not params_finite(net):
        print(f"第 {version} 版参数里有非有限值，不保存检查点（保留上一份）", flush=True)
        return
    tmp = path.with_suffix(".tmp")
    torch.save({"version": version, "progress": progress, "total_steps": total_steps, "phi_steps": phi_steps, "vocab_n": vocab_n, "obs_format": format_for(cfg.character), "state": net.state_dict(),
                "content_names": list(getattr(net, "content_names", ())), "catalog_version": CATALOG_VERSION,
                "opt": opt.adam_state_dict() if hasattr(opt, "adam_state_dict") else opt.state_dict(),     # 免调度时存成普通 Adam 也能载入的样子
                "config": cfg.to_dict(), **(extra or {})}, tmp)
    os.replace(tmp, path)


def params_finite(net) -> bool:
    return bool(torch.stack([p.detach().isfinite().all() for p in net.parameters()]).all())


def guarded_step(net, opt, max_norm: float, own: list | None = None) -> torch.Tensor:
    """裁剪梯度后更新一步；梯度范数不是有限值（NaN、inf）时跳过，参数和 Adam 状态都不变（决策记录 173）。
    融合的 Adam 靠 found_inf 跳过（与混合精度 GradScaler 同一机制），主机不用等显卡；不融合的（CPU 上调试）同步判断。
    own：各自单独裁剪的几组参数（分类价值头、战斗辅助头，决策记录 261 补记二），不算进全局范数：它们自身的梯度大，
    算进去会让全局裁剪几乎每个小批都把策略的梯度一起缩小。返回显卡上的 1.0（跳过）或 0.0。
    opt.muon（储君决策记录第 25 条）：有时先由 Muon 更新它管的二维权重（同样按 bad 跳过），再清掉这些参数的梯度，融合 Adam 跳过它们。"""
    sep = {id(p) for g in (own or []) for p in g}
    gn = torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if id(p) not in sep] if sep else net.parameters(), max_norm)
    for g in own or []:
        gn = gn + 0.0 * torch.nn.utils.clip_grad_norm_(g, max_norm)       # 只为让非有限值也算作跳过
    guarded_step.last_gn = gn.detach()      # 裁剪前的梯度范数（训练健康统计用，决策记录 220）
    bad = (~torch.isfinite(gn)).float()
    muon = getattr(opt, "muon", None)
    if muon is not None:
        muon.step(1.0 - bad)
        for p in muon.params:
            p.grad = None
    if opt.defaults.get("fused"):
        opt.found_inf = bad
        try: opt.step()
        finally: del opt.found_inf
    elif not bad.item():
        opt.step()
    return bad


def _health_groups(net) -> dict[str, list[torch.nn.Parameter]]:
    """训练健康统计（决策记录 220）的参数分组：编码器逐层、四个头、编号嵌入，其余合为一组。"""
    g: dict[str, list] = {}
    for name, p in net.named_parameters():
        if name.startswith("encoder.layers."):
            k = "L" + name.split(".")[2]
        elif name.startswith("fut_"):       # 整局预生成内容的小网络（决策记录 259），看板看它有没有在学
            k = "fut"
        else:
            k = next((h for h in ("cand_mlp_m", "cand_mlp", "value_mlp_m", "value_mlp", "value_hl_m", "value_hl", "value_aux", "id_emb", "aux_emb", "cand_id_emb")
                      if name.startswith(h + ".")), "other")
        g.setdefault(k, []).append(p)
    return g


def explained_variance(v: np.ndarray, r: np.ndarray) -> float | None:
    """1 − Var(回报 − 价值) / Var(回报)；样本太少或回报没有方差时为 None。"""
    if len(r) < 2 or float(np.var(r)) <= 0:
        return None
    return round(1.0 - float(np.var(r - v)) / float(np.var(r)), 4)


def minibatches(data: Ragged, perm: np.ndarray, size: int, device, timing: dict | None = None, per_step: dict | None = None, packed: bool = False,
                workers: int = 1):
    """后台线程组批并拷到显卡，与当前小批的计算重叠（组批约占学习时间的 12%）。
    timing["wait"] 累计主线程等组批的时间：明显大于 0 说明组批跟不上显卡。
    per_step：逐步的数组（动作、旧对数概率、优势等），按同样的下标切好随整批一起送；packed：附上打包注意力的偏移（model.packed_meta）。
    显卡上（2026-10-28，决策记录 158）：先拷进锁页内存，在单独的拷贝流上异步拷，主线程用事件等它拷完，不占计算流、不让主机等显卡。
    workers > 1 时几个线程轮流组批（第 j 个小批归第 j % workers 个线程），主线程按原顺序取：去同步之后学习快了，单线程组批跟不上。"""
    starts = list(range(0, len(perm), size))
    workers = max(1, min(workers, len(starts)))
    qs = [queue.Queue(maxsize=2) for _ in range(workers)]
    cuda = device.type == "cuda"

    def to_dev(v: np.ndarray):
        t = torch.from_numpy(np.ascontiguousarray(v))
        return t.pin_memory().to(device, non_blocking=True) if cuda else t.to(device)

    def work(w: int):
        q = qs[w]
        stream = torch.cuda.Stream(device) if cuda else None
        try:
            for s in starts[w::workers]:
                idx = perm[s:s + size]
                bt = collate_ragged(data.select(idx))
                d = dict(bt.__dict__)
                if packed:
                    d.update(packed_meta(bt.core_idx, bt.core_mask))
                for k, v in (per_step or {}).items():
                    d[k] = v[idx]
                with torch.cuda.stream(stream) if cuda else contextlib.nullcontext():
                    out = {k: (to_dev(v) if isinstance(v, np.ndarray) else [(to_dev(i), kg) for i, kg in v] if k == "groups" else v) for k, v in d.items()}
                    ev = None
                    if cuda:
                        ev = torch.cuda.Event(); ev.record(stream)
                q.put((idx, out, ev))
        except Exception as e:      # 交给主线程抛出
            q.put(e)
    for w in range(workers):
        threading.Thread(target=work, args=(w,), daemon=True).start()
    for j in range(len(starts)):
        t0 = time.time(); item = qs[j % workers].get()
        if timing is not None:
            timing["wait"] = timing.get("wait", 0.0) + time.time() - t0
        if isinstance(item, Exception):
            raise item
        idx, b, ev = item
        if ev is not None:      # 计算流等拷贝完成；张量是在拷贝流上分配的，记一下在计算流上用过，免得被缓存分配器提前回收
            cur = torch.cuda.current_stream(device); cur.wait_event(ev)
            for v in b.values():
                if torch.is_tensor(v):
                    v.record_stream(cur)
                elif isinstance(v, list):
                    for t, _ in v:
                        t.record_stream(cur)
        yield idx, b


REMOTE_RANK0 = 100     # 远程采集机上采集进程的编号起点（tools/remote/remote_actors.py），与本机的 0…actors−1 不重叠，随机数种子也就不同


class Feeder:
    """后台取数：线程持续从采集进程的队列里取轨迹（取的时候反序列化，这一步原来在主线程里，4090 机上每次更新约 1.3 秒，
    和学习串行），放进本地缓冲，主线程组批时直接拿，与上一轮学习重叠。缓冲按条数限容（约 prefetch 批），满了线程就停着，
    采集进程照常被队列反压挡住，不会因此多囤过旧的数据。"""

    def __init__(self, src, cap_items: int):
        self.src, self.buf, self.stop = src, queue.Queue(maxsize=max(1, cap_items)), threading.Event()
        threading.Thread(target=self._run, daemon=True, name="feeder").start()

    def _run(self) -> None:
        while not self.stop.is_set():
            try:
                item = self.src.get(timeout=1.0)
            except queue.Empty:
                continue
            except Exception:        # 队列坏了（采集进程全退出等）：交给主线程报错
                import traceback
                item = ("error", -1, traceback.format_exc())
            while not self.stop.is_set():
                try:
                    self.buf.put(item, timeout=1.0); break
                except queue.Full:
                    continue

    def get(self, timeout: float):
        return self.buf.get(timeout=timeout)


PHI_BINS = 20


def phi_update(net, opt, items: list[dict], cfg, device):
    """牌组打分头训练一步，返回 (对数几率, 标签, 损失, 是否跳过)。
    cfg.phi_detach 为假时（原来的做法）：梯度经共享编码器回传，与 PPO 共用同一个 Adam，会改动策略与价值网络的表示，
    也会动用 Adam 里 PPO 留下的一阶动量（约九成），相当于每次更新末尾多走一步。
    为真时（决策记录 179）：编码器不求梯度，只更新打分头（opt 应为只含打分头参数的独立优化器），主干和主优化器都不动。"""
    vb = collate_ragged(Ragged.concat([x["view"] for x in items])).to(device)
    y = torch.tensor([x["y"] for x in items], dtype=torch.float32, device=device)
    if cfg.phi_detach:
        with torch.no_grad():
            h = net._encode(vb)[2]
        z = net.phi_mlp(h).squeeze(-1).float()
        model = net.phi_mlp
    else:
        z = net.deck_score(vb).float()
        model = net
    l_phi = F.binary_cross_entropy_with_logits(z, y)
    opt.zero_grad(set_to_none=True); (cfg.phi_coef * l_phi).backward()
    skip = int(guarded_step(model, opt, cfg.max_grad_norm).item())
    return z, y, l_phi, skip


@torch.no_grad()
def phi_holdout(net, items: list[dict], device) -> list[list]:
    """新到的打分标签在进入训练之前的成绩，按幕：[样本数, 平方误差和, 实际成绩和, 实际成绩平方和, 打赢局的预测分档计数, 打输局的预测分档计数]。
    看板合计多次更新后算区分能力（赢、输两组预测的 AUC）和误差比“只报这一幕平均成绩”降低多少；预测按 0.05 一档。"""
    vb = collate_ragged(Ragged.concat([x["view"] for x in items])).to(device)
    pred = torch.sigmoid(net.deck_score(vb).float()).cpu().numpy()
    out = []
    for a in range(3):
        idx = [j for j, x in enumerate(items) if x["act"] == a]
        y = np.array([items[j]["y"] for j in idx], np.float64); p = pred[idx].astype(np.float64)
        bins = np.minimum((p * PHI_BINS).astype(int), PHI_BINS - 1)
        win = y >= 1.0
        out.append([len(idx), round(float(((p - y) ** 2).sum()), 4), round(float(y.sum()), 4), round(float((y ** 2).sum()), 4),
                    np.bincount(bins[win], minlength=PHI_BINS).tolist(), np.bincount(bins[~win], minlength=PHI_BINS).tolist()])
    return out


def main(argv=None):
    cfg = TrainConfig.from_args(argv)
    if cfg.value_probe and (cfg.value_probe_every < 1 or cfg.value_probe_grad_every < 1 or cfg.value_probe_samples < 1
                            or not 0 <= cfg.value_probe_episode_frac <= 1 or cfg.value_probe_max_gb <= 0):
        raise SystemExit("价值诊断的频率、样本数、空间上限须为正，整局抽样比例须在 [0,1]")
    if any(any(float(x) > 0 for x in (v or {}).values()) for v in (cfg.combat_explore or {}).values()) and not cfg.explore_offpolicy:
        raise SystemExit("combat_explore 要求 explore_offpolicy 为开（改选的步要按行为分布 μ 算比率，决策记录 238）")
    if format_for(cfg.character) >= 20 and cfg.model.n_zones < 19:
        raise SystemExit(f"{cfg.character} 的帧是观测格式 {format_for(cfg.character)}，已知牌堆顶在区域 16–18：model.n_zones 要不小于 19（现在 {cfg.model.n_zones}）")
    # kill（SIGTERM）时走 finally：通知采集进程退出并保存检查点
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    run_dir = Path(cfg.run_dir); run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    torch.manual_seed(cfg.seed); torch.set_num_threads(cfg.learner_threads); use_tf32()
    device = torch.device(cfg.learner_device)
    net = PolicyValueNet(cfg.model).to(device)
    net.fast = cfg.learner_fast and device.type == "cuda"
    # PPO 小批循环里的前向（learner_compile 时编译；参数与 net 共用，存检查点、牌组打分仍走 net）
    # forward_all 多给分类价值头、辅助头的输出（决策记录 261）
    fwd = torch.compile(net.forward_all, dynamic=True) if cfg.learner_compile and device.type == "cuda" else net.forward_all
    teacher = None      # 战斗锚定（决策记录 267）：冻结的老师网络，只做前向
    if cfg.combat_anchor_coef > 0:
        if not cfg.combat_anchor_ckpt:
            raise SystemExit("combat_anchor_coef > 0 要给 combat_anchor_ckpt")
        if not Path(cfg.combat_anchor_ckpt).exists() and (run_dir / "latest.pt").exists():
            # 第一次打开时把续训的检查点存成老师（就是上线那一刻的网络）；之后重启沿用这份，不再覆盖
            shutil.copyfile(run_dir / "latest.pt", cfg.combat_anchor_ckpt)
            print(f"战斗锚定：老师文件不存在，已把 {run_dir / 'latest.pt'} 存成 {cfg.combat_anchor_ckpt}", flush=True)
        if not Path(cfg.combat_anchor_ckpt).exists():
            raise SystemExit(f"combat_anchor_ckpt 不存在：{cfg.combat_anchor_ckpt}")
        tb = torch.load(cfg.combat_anchor_ckpt, map_location="cpu", weights_only=False)
        teacher = PolicyValueNet(cfg.model).to(device).eval()
        load_state(teacher, tb["state"]); teacher.fast = net.fast
        for q in teacher.parameters():
            q.requires_grad_(False)
        print(f"战斗锚定：老师为 {cfg.combat_anchor_ckpt}（第 {tb.get('version')} 版），系数 {cfg.combat_anchor_coef}", flush=True)
        del tb
    skl = None      # 搜索标签的 KL 辅助损失（2026-11-17 训练中用搜索的试点，sts2rl/search_kl.py）；默认关
    skl_compiled = [cfg.search_kl_compiled]     # 标签批的前向走编译的 fwd（出错时改成 False，退回 net.forward_all）
    if cfg.search_kl_coef > 0:
        if not cfg.search_kl_dir:
            raise SystemExit("search_kl_coef > 0 要给 search_kl_dir")
        from .search_kl import SearchKL
        skl = SearchKL(cfg.search_kl_dir, cfg.search_kl_batch, cfg.search_kl_buffer, cfg.search_kl_max_age, cfg.search_kl_min,
                       packed=net.fast, device=device, seed=cfg.seed)
        print(f"搜索 KL：标签目录 {cfg.search_kl_dir}，系数 {cfg.search_kl_coef}，每小批 {cfg.search_kl_batch} 个，缓冲 {cfg.search_kl_buffer}，"
              f"最长 {cfg.search_kl_max_age} 版，至少 {cfg.search_kl_min} 个才加", flush=True)
    # 后来新增的头各自一个参数组，按新增顺序排在后面：旧检查点的优化器状态只有前几组，缺的组从零开始
    phi_params = list(net.phi_mlp.parameters())
    macro_params = list(net.cand_mlp_m.parameters()) + list(net.value_mlp_m.parameters())
    boss_params = list(net.boss_emb.parameters())
    move_params = list(net.move_in.parameters()) + list(net.move_out.parameters())
    v9_params = list(net.feat_x.parameters()) + list(net.cand_feat_x.parameters()) + list(net.content_proj.parameters())   # 观测格式版本 9 与内容先验
    new_ids = {id(p) for p in phi_params + macro_params + boss_params + move_params + v9_params}
    main_params = [p for p in net.parameters() if id(p) not in new_ids]
    v10_params = list(net.feat_c.parameters())     # 观测格式版本 10：生物记录的回合结束预演
    pool_n_params = [net.pool_n]                    # 汇总槽的张数
    v11_params = list(net.priv_enc.parameters()) + list(net.priv_roll.parameters()) + list(net.feat_v11.parameters())     # 观测格式版本 11
    # 观测格式版本 13 与汇总前的小网络（决策记录 181）
    v13_params = [p for m in (net.feat_v13, net.bind_pw_in, net.bind_pw_out, net.bind_pool_in, net.bind_pool_out) for p in m.parameters()]
    boss2_params = list(net.boss2_emb.parameters())     # 第二个 Boss 单独的嵌入表（决策记录 183）
    # 观测格式版本 14（决策记录 185）：遗物与战斗计数的新特征、本幕遭遇历史
    v14_params = [p for m in (net.feat_v14, net.hist_lin, net.hist_n_emb, net.hist_e_emb) for p in m.parameters()]
    # 观测格式版本 15（决策记录 194）：球被动次数、Boss 场数、占卜
    v15_params = [p for m in (net.feat_v15, net.stage_lin, net.cand_feat_v15, net.enc_next, net.enc_bag, net.enc_seen) for p in m.parameters()]
    # 观测格式版本 16（决策记录 204、205）：有效抽牌、事件选项展示过的信息、休息前确定的回血、会员卡折后价
    v16_params = [p for m in (net.draw_lin, net.feat_v16, net.cand_feat_v16) for p in m.parameters()] + [net.gain_gate] if cfg.model.obs_v16 else []
    # 观测格式版本 17（决策记录 209）：商店删牌、能力先后、回手资格
    v17_params = [p for m in (net.shop_lin, net.pw_order) for p in m.parameters()] + [net.ret_gate] if cfg.model.obs_v17 else []
    # 观测格式版本 18（决策记录 210）：奖励、抽牌张数、问号房实际概率、滑脚木桥候选池
    v18_params = [p for m in (net.v18_lin, net.map_v18) for p in m.parameters()] + [net.bridge_gate] if cfg.model.obs_v18 else []
    v19_params = [p for m in (net.cand_v19, net.v19_lin, net.feat_v19) for p in m.parameters()] + [net.shop_gate] if cfg.model.obs_v19 else []     # 决策记录 227
    # 整局预生成内容（决策记录 259）：只进价值头的小网络，排在最后一组
    fut_params = [p for m in (net.fut_emb, net.fut_slot, net.fut_in, net.fut_out) for p in m.parameters()] if cfg.model.future_content else []
    new_ids |= {id(p) for p in v10_params + pool_n_params + v11_params + v13_params + boss2_params + v14_params + v15_params + v16_params + v17_params + v18_params + v19_params + fut_params}
    main_params = [p for p in net.parameters() if id(p) not in new_ids]
    groups = [main_params, phi_params, macro_params, boss_params, move_params, v9_params, v10_params, pool_n_params, v11_params, v13_params, boss2_params, v14_params,
              v15_params] + ([v16_params] if v16_params else []) + ([v17_params] if v17_params else []) + ([v18_params] if v18_params else []) + ([v19_params] if v19_params else [])
    groups += [fut_params] if fut_params else []
    # 分类价值头、战斗辅助头（决策记录 261）：各一组，排在最后
    hl_params = [p for m in (net.value_hl, net.value_hl_m) for p in m.parameters()] if cfg.model.value_hl else []
    aux_params = list(net.value_aux.parameters()) if cfg.model.combat_aux else []
    att_params = list(net.kind_x.parameters()) if cfg.model.macro_attend else []     # 局外帧扩展注意力的类型嵌入（决策记录 262）
    # 预知记录（决策记录 280）：只进价值头的小网络，排在所有可开关组的最后
    orc_params = [p for m in (net.orc_emb, net.orc_slot, net.orc_sub, net.orc_in, net.orc_out) for p in m.parameters()] if cfg.model.oracle else []
    # 观测格式版本 26 的候选特征 12、13（cand_v26）：可开关组，排在所有可开关组的最后（续训时新开的组从零开始）
    v26_params = list(net.cand_v26.parameters()) if cfg.model.obs_v26 else []
    # 观测格式版本 27 的记录特征 20、21（feat_v27，手牌的标黄、标红）：同上，排在 v26 之后
    v27_params = list(net.feat_v27.parameters()) if cfg.model.obs_v27 else []
    # 回灌（铁甲战士决策记录第 26 条，model.recycle）的 rcy_ 各层：可开关组，排在 v27 之后
    rcy_params = [p for m in (net.rcy_a, net.rcy_b, net.rcy_c, net.rcy_p) for p in m.parameters()] if cfg.model.recycle else []
    # 格式 29 的新列映射（铁甲战士决策记录第 27 条，model.obs_v29）：可开关组，排在 rcy 之后
    v29_params = [p for m in (net.feat_v29, net.cand_v29) for p in m.parameters()] if cfg.model.obs_v29 else []
    # 候选未来读数头（愚者决策记录第 25 条，model.future_head）：单独的 Adam（fh_opt，见下），不进主优化器、免调度与 Muon，单独裁剪
    fh_params = [p for m in (net.fh_pred, net.fh_res) for p in m.parameters()] if cfg.model.future_head else []
    new_ids |= {id(p) for p in hl_params + aux_params + att_params + orc_params + v26_params + v27_params + rcy_params + v29_params + fh_params}
    own_clip = [g for g in (hl_params, aux_params) if g] if cfg.head_own_clip else None
    if fh_params:
        own_clip = (own_clip or []) + [fh_params]
    main_params = [p for p in net.parameters() if id(p) not in new_ids]
    groups[0] = main_params
    groups += [g for g in (hl_params, aux_params, att_params, orc_params, v26_params, v27_params, rcy_params, v29_params) if g]
    # 显卡上用融合实现（一次内核更新全部参数，结果与逐个张量的实现等价；2026-10-26 起，原来优化器步骤约占学习进程 6%）
    opt = torch.optim.Adam([{"params": g} for g in groups], lr=cfg.lr, eps=1e-5, fused=device.type == "cuda")
    # 打分头独立（cfg.phi_detach，决策记录 179）：只训练打分头自己的参数，用它自己的优化器（状态不存进检查点，重启后从零开始）
    opt_phi = torch.optim.Adam(phi_params, lr=cfg.lr, eps=1e-5, fused=device.type == "cuda") if cfg.phi_detach else None
    version, progress, total_steps, phi_steps = 0, cfg.progress_init, 0, 0
    # 专项日程（决策记录 287）：价值一侧的参数（只训价值时只更新它们）与做权重衰减的参数（二维及以上）
    VALUE_SIDE = ("value_mlp", "value_mlp_m", "value_hl", "value_hl_m", "value_aux", "priv_proj", "priv_enc", "priv_roll", "phi_mlp",
                  "fut_emb", "fut_slot", "fut_in", "fut_out", "orc_emb", "orc_slot", "orc_sub", "orc_in", "orc_out")
    value_params = [p for n, p in net.named_parameters() if n.split(".")[0] in VALUE_SIDE]
    nonvalue_params = [p for n, p in net.named_parameters() if n.split(".")[0] not in VALUE_SIDE]
    decay_params = [p for p in net.parameters() if p.dim() >= 2]
    decay_value = [p for p in value_params if p.dim() >= 2]
    lr_now, anc_coef, teacher_on, value_only = cfg.lr, cfg.combat_anchor_coef, teacher is not None, False
    warmup_left = 0     # 重新建立优化器后逐步升学习率还剩几次更新
    vocab_n = Catalog().vocab_size
    ckpt = run_dir / "latest.pt"
    ema_loaded = None
    cn_remap = None         # 换了内容目录时内容先验各列的对齐函数（见下）
    if ckpt.exists():
        blob = torch.load(ckpt, map_location=device, weights_only=False)
        # 换了游戏版本（内容目录）时，内容先验的列按特征名对齐（content.remap_columns）；权重、Adam 的矩、Muon 的动量、滑动平均都要对齐
        cn_old = checkpoint_content_names(net, blob["state"].get("content_proj.weight"), blob.get("content_names"))
        cn_remap = (lambda t: remap_columns(t, cn_old, net.content_names)[0]) if cn_old is not None and cn_old != net.content_names else None
        load_state(net, blob["state"], blob.get("content_names"))
        ema_loaded = blob.get("ema")
        if cn_remap is not None and ema_loaded is not None and "content_proj.weight" in ema_loaded:
            ema_loaded = {**ema_loaded, "content_proj.weight": cn_remap(ema_loaded["content_proj.weight"])}
            print("滑动平均权重：内容先验的列按特征名对齐", flush=True)
        old_n = blob.get("vocab_n") or VOCAB_N_LEGACY
        if old_n < vocab_n:
            n_unk = zero_new_vocab(net, old_n, vocab_n)
            # 滑动平均权重同样处理（2026-11-13 补：原来只清了当前参数，滑动平均里这些行还是建网时的随机值，要约 1/(1−衰减) 版才衰减掉）
            if ema_loaded is not None:
                zero_new_vocab(ema_loaded, old_n, vocab_n)
            print(f"词表 {old_n} → {vocab_n} 项：新增 ID 的嵌入行已清零（滑动平均同样处理）"
                  + (f"，其中旧版就有、当时记 <unk> 的 {n_unk} 个改取 <unk> 那一行（涅奥开局的遗物选项等，model.WAS_UNK_PREFIXES）" if n_unk else ""), flush=True)
        if cfg.route_summary and not (blob.get("config") or {}).get("route_summary"):
            # 刚打开路线摘要：这种记录类型以前没出现过，类型嵌入那一行清零，减小上线时的扰动
            with torch.no_grad():
                net.kind_emb.weight[int(TokenKind.ROUTE_SUMMARY)].zero_()
            print("打开路线摘要：记录类型 14 的嵌入行已清零", flush=True)
        if blob.get("obs_format", 8) < 9:
            # 观测格式版本 9 新增“升级预览”区域（12），以前没出现过：区域嵌入那一行清零，减小上线时的扰动
            with torch.no_grad():
                net.zone_emb.weight[int(Zone.UPGRADE_PREVIEW)].zero_()
            print("观测格式升到版本 9：区域 12 的嵌入行已清零", flush=True)
        if blob.get("obs_format", 8) < 14:
            # 观测格式升到版本 14：打出牌堆（区域 9）里正在结算的牌以前不输出，区域嵌入那一行从没练过，清零减小上线时的扰动
            with torch.no_grad():
                net.zone_emb.weight[int(Zone.PLAY)].zero_()
            print("观测格式升到版本 14：区域 9（打出牌堆）的嵌入行已清零", flush=True)
        if blob.get("obs_format", 8) < 18:
            # 观测格式升到版本 18（决策记录 210）：奖励分组的区域 13–15 以前没出现过，区域嵌入那几行清零，减小上线时的扰动
            with torch.no_grad():
                for z in (Zone.REWARD_GROUP1, Zone.REWARD_GROUP2, Zone.REWARD_GROUP3):
                    net.zone_emb.weight[int(z)].zero_()
            print("观测格式升到版本 18：区域 13–15（奖励分组）的嵌入行已清零", flush=True)
        if blob.get("obs_format", 8) < 11:
            # 观测格式升到版本 11：新记录类型 15（战斗计数）以前没出现过，类型嵌入那一行清零，上线时输出与版本 10 相同
            with torch.no_grad():
                net.kind_emb.weight[int(TokenKind.COMBAT_STAT)].zero_()
            print("观测格式升到版本 11：记录类型 15 的嵌入行已清零", flush=True)
        if "opt" in blob:
            sd = blob["opt"]
            # 排在最后的几组可开关的参数（整局预生成内容 fut、分类价值头 hl、战斗辅助头 aux；决策记录 259、261）：检查点与现在开关不同时，
            # 按名字把检查点的组对到现在的组上，关掉的组连同状态去掉，新开的组从零开始（下面补组），参数编号按新顺序重排
            om = (blob.get("config") or {}).get("model") or {}
            old_x = [n for n, k in (("fut", "future_content"), ("hl", "value_hl"), ("aux", "combat_aux"), ("att", "macro_attend"), ("orc", "oracle"), ("v26", "obs_v26"), ("v27", "obs_v27"), ("rcy", "recycle"), ("v29", "obs_v29")) if om.get(k)]
            new_x = [n for n, on in (("fut", bool(fut_params)), ("hl", bool(hl_params)), ("aux", bool(aux_params)), ("att", bool(att_params)), ("orc", bool(orc_params)), ("v26", bool(v26_params)), ("v27", bool(v27_params)), ("rcy", bool(rcy_params)), ("v29", bool(v29_params))) if on]
            nb = len(sd["param_groups"]) - len(old_x)
            if old_x != new_x and nb == len(groups) - len(new_x):
                names = [f"b{i}" for i in range(nb)] + old_x
                byname = dict(zip(names, sd["param_groups"]))
                pgs, state, nid = [], {}, 0
                for n in [f"b{i}" for i in range(nb)] + [x for x in new_x if x in byname]:
                    g = dict(byname[n]); ids = []
                    for pid in g["params"]:
                        if pid in sd["state"]:
                            state[nid] = sd["state"][pid]
                        ids.append(nid); nid += 1
                    g["params"] = ids; pgs.append(g)
                sd = {"state": state, "param_groups": pgs}
                print(f"可开关参数组 {old_x} → {new_x}：优化器状态按名字对齐（去掉 {[x for x in old_x if x not in new_x]}）", flush=True)
                # 新开的组若排在保留的组之前，要等补组后按新顺序排：只有 new_x 里新开的都排在保留的之后时才能直接补在末尾
                kept = [x for x in new_x if x in byname]
                if new_x[:len(kept)] != kept:
                    raise SystemExit(f"可开关参数组的新顺序 {new_x} 不能在保留的 {kept} 之后直接补齐，需要手工迁移优化器状态")
            while len(sd["param_groups"]) < len(groups):
                g = dict(sd["param_groups"][0]); n0 = sum(len(x["params"]) for x in sd["param_groups"])
                g["params"] = list(range(n0, n0 + len(groups[len(sd["param_groups"])])))
                sd["param_groups"].append(g)
            opt.load_state_dict(sd)
            # 参数比检查点里的宽（内容先验追加了特征列，决策记录 180）：Adam 的一、二阶矩按新形状补 0；
            # 换了游戏版本时内容先验的列按特征名对齐（cn_remap，v0.113.0 起）
            for p in net.parameters():
                st = opt.state.get(p)
                for k in ("exp_avg", "exp_avg_sq"):
                    if st and k in st and st[k].shape != p.shape and st[k].dim() == p.dim() and st[k].shape[0] == p.shape[0]:
                        if cn_remap is not None and p is net.content_proj.weight:
                            st[k] = cn_remap(st[k]).contiguous()
                            print(f"优化器状态 {tuple(st[k].shape)}：{k} 按特征名对齐", flush=True)
                        elif st[k].shape[1] < p.shape[1]:
                            st[k] = torch.cat([st[k], st[k].new_zeros(p.shape[0], p.shape[1] - st[k].shape[1])], 1)
                            print(f"优化器状态 {tuple(st[k].shape)}：{k} 按新形状补 0", flush=True)
                        else:       # 关掉后来追加的内容先验列（决策记录 264）：截掉最后几列
                            st[k] = st[k][:, :p.shape[1]].contiguous()
                            print(f"优化器状态 {tuple(st[k].shape)}：{k} 截掉多出的列", flush=True)
        else:
            # 模型扩容后的检查点（tools/grow_model.py）不带优化器状态：全新的 Adam 头几步步子偏大，先从一成学习率逐步升上去
            warmup_left = cfg.opt_warmup
            print(f"检查点不带优化器状态（模型扩容）：重新建立优化器，前 {warmup_left} 次更新学习率从一成逐步升到 {cfg.lr}", flush=True)
        # 优化器状态里带着上次的学习率，载入后会盖掉配置：按配置重设，改 lr 的分叉才真正生效（2026-10-26 降学习率前发现）
        for g in opt.param_groups:
            if g["lr"] != cfg.lr:
                print(f"学习率 {g['lr']} → {cfg.lr}（按配置）", flush=True)
            g["lr"] = cfg.lr
        version, progress, total_steps = blob["version"], blob["progress"], blob["total_steps"]
        phi_steps = blob.get("phi_steps", 0)
        print(f"从检查点继续：版本 {version}，进度 {progress:.2f}，已用 {total_steps} 步，打分头已训练 {phi_steps} 步", flush=True)
    # Muon（储君决策记录第 25 条）：隐藏层的二维权重改由 Muon 更新，挂在 opt.muon 上（guarded_step 先走它）；
    # 检查点里有 muon 就接着它的动量，否则（刚从 Adam 切过来）接着 Adam 的一阶矩
    # 免调度 Adam（cfg.sf_adam，2026-11-15）：Adam 那部分参数换成免调度更新（模型参数里存 y，平均点 x 只在导出时算），Muon 不变；
    # 检查点里有 opt_sf 就接着它，否则从当前的 Adam 切过来（z 从当前参数起，二阶矩接着 Adam 的）
    if cfg.sf_adam:
        opt, sf_msg = AdamScheduleFree.from_adam(opt, lr=cfg.sf_lr or cfg.lr, betas=(cfg.sf_beta1, cfg.sf_beta2), warmup_steps=cfg.sf_warmup,
                                                 r=cfg.sf_r, weight_lr_power=cfg.sf_power, sf_state=blob.get("opt_sf") if ckpt.exists() else None,
                                                 exclude={id(p) for _, p, _ in muon_select(net)} if cfg.optimizer == "muon" else None)
        print(f"免调度 Adam（Adam 部分，Muon 不变）：学习率 {cfg.sf_lr or cfg.lr}，β {cfg.sf_beta1} / {cfg.sf_beta2}，预热 {cfg.sf_warmup} 步，"
              f"平均权重 lr_max^{cfg.sf_power}·k^{cfg.sf_r}；{sf_msg}", flush=True)
    muon = None
    if cfg.optimizer == "muon":
        muon = Muon(muon_select(net), cfg.muon_lr or cfg.lr, cfg.muon_momentum, cfg.muon_nesterov, cfg.muon_ns_steps,
                    sf=cfg.sf_muon, sf_beta=cfg.sf_beta1, sf_warmup=cfg.sf_warmup, sf_power=cfg.sf_power, sf_r=cfg.sf_r)     # sf_muon：免调度平均（愚者决策记录第 16 条）
        msd = blob.get("muon") if ckpt.exists() else None
        if msd and cn_remap is not None and "content_proj.weight" in (msd.get("buf") or {}):
            msd = {**msd, "buf": {**msd["buf"], "content_proj.weight": cn_remap(msd["buf"]["content_proj.weight"])}}
            print("Muon 动量：内容先验的列按特征名对齐", flush=True)
        n_ok = muon.load_state_dict(msd) if msd else muon.init_from_adam(opt)
        opt.muon = muon
        n_el = sum(p.numel() for p in muon.params)
        print(f"Muon：{len(muon.params)} 个二维权重、{n_el} 个参数（共 {sum(p.numel() for p in net.parameters())}），学习率 {muon.lr}，"
              f"动量 {muon.momentum}{'（Nesterov）' if muon.nesterov else ''}，迭代 {muon.ns_steps} 次；"
              f"动量{'接着检查点里的' if msd else '接着 Adam 的一阶矩'} {n_ok} 个", flush=True)
    elif ckpt.exists() and blob.get("muon"):
        print("检查点里有 Muon 的动量，但这次用 Adam：Muon 管过的参数接着 Adam 里切换前留下的状态", flush=True)
    # 重新开始免调度平均（config.sf_restart，愚者决策记录第 19 条）：检查点里记的标记与配置不同才做（从零开训也不做），做完把标记存进检查点
    sf_restart_tag = (blob.get("sf_restart") or "") if ckpt.exists() else cfg.sf_restart
    if cfg.sf_restart and cfg.sf_restart != sf_restart_tag and (cfg.sf_adam or cfg.sf_muon):
        na = opt.restart_average() if cfg.sf_adam else 0
        nm = muon.restart_average() if muon is not None else 0
        print(f"免调度平均重新开始（标记 {cfg.sf_restart!r}，检查点里原为 {sf_restart_tag!r}）：参数与 z 都换成平均点 x、累计权重清零；"
              f"Adam 部分 {na} 个参数，Muon {nm} 个", flush=True)
        sf_restart_tag = cfg.sf_restart
    shaping = lambda: cfg.deck_shaping if phi_steps >= cfg.phi_warmup else 0.0
    # 胜率势能（决策记录 189）：胜率小模型在学习进程里与主模型一起在线训练（只训通关头，标签为训练整局最后是否通关），
    # 随权重发给采集进程；采集进程按 Φ = wp_coef × 预测通关概率 给整局做势能塑形。检查点里有就接着练，否则从离线模型（wp_init）开始
    # 样本缓冲用定长的环形列表：随机抽样要按下标取，双端队列按下标取是线性的（30 万条时一次更新多花约 0.3 秒）
    wp_net, wp_opt, wp_steps, wp_rounds, wp_buf, wp_pos, wp_loss = None, None, 0, 0, [], 0, None
    if cfg.wp_shaping > 0:
        wb = blob.get("winprob") if ckpt.exists() else None
        init = wb or (torch.load(os.path.expanduser(cfg.wp_init), map_location="cpu") if cfg.wp_init and os.path.exists(os.path.expanduser(cfg.wp_init)) else None)
        wp_net = wpo.build((init or {}).get("args")).to(device)
        if init:
            wp_net.load_state_dict(init["model"])
        wp_steps = (wb or {}).get("steps", 0)
        wp_rounds = (wb or {}).get("rounds", wp_steps)      # 训练过的更新次数（升系数按它算）；旧检查点每次更新只训一步，同步数
        wp_opt = torch.optim.AdamW(wp_net.parameters(), lr=cfg.wp_lr, weight_decay=0.01)
        print(f"胜率势能：模型{'接着检查点' if wb else ('从离线模型 ' + cfg.wp_init) if init else '从头'}，已在线训练 {wp_steps} 步", flush=True)
    wp_coef = lambda: cfg.wp_shaping * float(np.clip((wp_rounds - cfg.wp_warmup) / max(1, cfg.wp_ramp), 0, 1)) if wp_net is not None else 0.0
    wp_state = lambda: {"model": {k: v.detach().cpu() for k, v in wp_net.state_dict().items()}, "args": dict(wp_net_args), "steps": wp_steps,
                        "rounds": wp_rounds}
    wp_net_args = ({"vocab_size": wp_net.emb.num_embeddings, "d": wp_net.emb.embedding_dim, "hidden": wp_net.mlp[0].out_features,
                    "dropout": wp_net.drop.p} if wp_net is not None else {})
    wp_extra = lambda: {"winprob": wp_state(), "wp_coef": wp_coef()} if wp_net is not None else {}
    # 参数的指数滑动平均（决策记录 223）：续训时接着检查点里的 ema，形状对不上的（模型改过）从当前参数起
    ema = None
    if cfg.ema_decay > 0:
        ema = {k: (ema_loaded[k].to(device=v.device, dtype=torch.float32) if ema_loaded is not None and k in ema_loaded and ema_loaded[k].shape == v.shape
                   else v.detach().float().clone()) for k, v in net.state_dict().items() if torch.is_floating_point(v) and k != "hl_beta"}
        print(f"滑动平均权重：衰减 {cfg.ema_decay}，{'接着检查点里的' if ema_loaded is not None else '从当前参数起'}", flush=True)
    ema_state = lambda: {k: (ema[k].to(v.dtype) if k in ema else v).detach().cpu() for k, v in net.state_dict().items()}

    def sf_x_state() -> dict:
        """免调度的平均点 x（整个网络，CPU）：Adam 部分取 opt.x_state，Muon 管的二维权重取 muon.x_of；没开的部分就是当前参数。"""
        st = opt.x_state(net) if cfg.sf_adam else {k: v.detach().cpu() for k, v in net.state_dict().items()}
        if muon is not None and getattr(muon, "sf", False):
            for n, p, _ in muon.items:
                st[n] = muon.x_of(n, p).detach().cpu()
        return st
    # 跳过稳赢战斗的结果头（config.co_heads，model.CombatOutcomeHeads）：单独的模块与 Adam，读主干的 detach 副本，不影响策略与价值的学习；
    # 随检查点保存（co、co_opt）、随权重文件发给采集进程（co_state）。检查点里没有时用 co_init 的头作初值（tools/combat_model.py 训好的冻结主干模型）
    co = co_opt = None
    if cfg.co_heads:
        co = CombatOutcomeHeads(cfg.model.d).to(device)
        co_opt = torch.optim.Adam(co.parameters(), lr=cfg.co_lr)
        cblob = blob if ckpt.exists() else None
        if cblob is not None and cblob.get("co"):
            co.load_state_dict(cblob["co"])
            if cblob.get("co_opt"):
                co_opt.load_state_dict(cblob["co_opt"])
            print("结果头：接着检查点里的", flush=True)
        elif cfg.co_init:
            init = torch.load(cfg.co_init, map_location="cpu", weights_only=False)["state"]
            co.load_state_dict({k: v for k, v in init.items() if k.startswith(("hp.", "pot."))})
            print(f"结果头：初值取自 {cfg.co_init}", flush=True)
        else:
            print("结果头：从随机初值起", flush=True)
    co_extra = lambda: {"co_state": {k: v.detach().cpu() for k, v in co.state_dict().items()}} if co is not None else {}
    # 候选未来读数头（愚者决策记录第 25 条）：单独的 Adam；读数各维的滑动标准差（归一化损失用）随检查点保存
    fh_opt = torch.optim.Adam(fh_params, lr=cfg.fh_lr) if fh_params else None
    fh_sd = torch.ones(cfg.model.fh_dims, device=device) if fh_params else None
    if fh_opt is not None and ckpt.exists() and blob.get("fh_opt"):
        fh_opt.load_state_dict(blob["fh_opt"])
        fh_sd = blob.get("fh_sd", fh_sd).to(device)
        print("未来读数头：优化器接着检查点里的", flush=True)
    elif fh_opt is not None:
        print(f"未来读数头：新开（残差最后一层清零，策略输出不变；辅助损失系数 {cfg.fh_coef}、步数 {cfg.fh_h}、学习率 {cfg.fh_lr}"
              f"{'，读 z 的 detach 副本' if cfg.model.fh_detach else '，辅助梯度传回主干'}）", flush=True)
    # 免调度（sf_adam / sf_muon）时发给评测的“ema_state”换成平均点 x：常规评测带 --ema 就评 x（采集端仍用 state，即 y）
    pub_extra = lambda: {**wp_extra(), **({"ema_state": sf_x_state()} if cfg.sf_adam or cfg.sf_muon else {"ema_state": ema_state()} if ema is not None else {}),
                         **co_extra()}
    ck_extra = lambda: {**({"winprob": wp_state()} if wp_net is not None else {}), **({"ema": {k: v.detach().cpu() for k, v in ema.items()}} if ema is not None else {}),
                        **({"muon": muon.state_dict()} if muon is not None else {}),
                        **({"opt_sf": opt.state_dict()} if cfg.sf_adam else {}),     # 免调度 Adam 自己的状态（Muon 的在 muon 里）
                        **({"sf_x": sf_x_state()} if cfg.sf_adam or cfg.sf_muon else {}),     # 免调度的平均点 x（整个网络）
                        **({"sf_restart": sf_restart_tag} if sf_restart_tag else {}),     # 做过的“重新开始平均”标记（config.sf_restart）
                        **({"co": {k: v.detach().cpu() for k, v in co.state_dict().items()}, "co_opt": co_opt.state_dict()} if co is not None else {}),
                        **({"fh_opt": fh_opt.state_dict(), "fh_sd": fh_sd.detach().cpu()} if fh_opt is not None else {})} or None
    shm = Path("/dev/shm") if Path("/dev/shm").exists() else run_dir
    weights_path = shm / f"sts2rl_{os.getpid()}.pt"
    # 分类价值头的换用比例（决策记录 261）：写在网络的缓冲里（随权重发给采集进程、存进检查点），每版按门槛决定升不升，见 config.hl_gate
    hl_ratio = None
    if cfg.model.value_hl:
        print(f"分类价值头：第 {version} 版换用比例 {float(net.hl_beta):.3f}（第 {cfg.hl_start} 版起，均方误差之比不超过 {cfg.hl_gate} 时每版加 1/{cfg.hl_ramp}）", flush=True)
    publish(net, weights_path, version, progress, shaping(), pub_extra())

    ctx = mp.get_context("spawn")
    # 反压：队列按“约 1.5 批”的轨迹数设上限（战斗平均约 16 步）；学习跟不上时采集进程阻塞，而不是产生过旧的数据
    # 双机协同时本机轨迹要过两级队列（这里与取数线程的缓冲），远程的只过缓冲一级，本机的因此排得更久、更容易过旧（决策记录 196、197）：
    # local_queue_items 给出时这一级只留这么多条，本机轨迹基本只在缓冲里排队
    q_items = max(64, int(cfg.queue_steps * cfg.batch_steps / cfg.queue_item_steps))
    out_q = ctx.Queue(maxsize=cfg.local_queue_items if cfg.bridge_port and cfg.local_queue_items else q_items); stop = ctx.Event()
    actors = [ctx.Process(target=run_actor, args=(r, cfg, str(weights_path), out_q, stop), daemon=True) for r in range(cfg.actors)]
    if cfg.eval_runs > 0 and cfg.eval_envs > 0:
        # 评测进程要开自己的工作进程，所以不能是守护进程；它在本进程退出后自行退出（退出时也会被强制结束）
        actors.append(ctx.Process(target=run_evaluator, args=(cfg, str(weights_path), str(run_dir), stop), daemon=False))
    for p in actors:
        p.start()
    feeder = Feeder(out_q, int(cfg.prefetch_steps * cfg.batch_steps / cfg.queue_item_steps)) if cfg.prefetch_steps > 0 else None
    bridge = None
    if cfg.bridge_port:
        # 双机协同：远程采集机的轨迹直接放进取数线程的本地缓冲（与本机采集进程的轨迹同等对待；缓冲满了就阻塞，对端被 TCP 反压挡住）
        from .bridge import BridgeServer, load_token
        if feeder is None:
            feeder = Feeder(out_q, q_items)
        bridge = BridgeServer(cfg.bridge_bind, cfg.bridge_port, load_token(cfg.bridge_token_file), feeder.buf.put, weights_path)
        bridge.set_version(version)
        print(f"双机协同：在 {cfg.bridge_bind}:{cfg.bridge_port} 等远程采集机", flush=True)
    sampler_stop = threading.Event(); start_sampler(run_dir, sampler_stop)
    from .value_probe import LearnerProbe
    value_probe = LearnerProbe(cfg) if cfg.value_probe else None
    started_at = time.time()
    try:   # 续训时沿用第一次开训的时间，看板上的“已训练时长”不因重启归零
        started_at = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))["started_at"]
    except Exception:
        pass
    log = open(run_dir / "log.jsonl", "a", encoding="utf-8")
    recent = deque(maxlen=400)
    label_stats = LabelStats(cfg.run_dir)
    distill = DistillBuffer(cfg) if cfg.distill_on else None
    phi_buf: deque = deque(maxlen=cfg.phi_buffer)      # 牌组打分的标签：(局面视图, 幕, 成绩)
    ppg_buf: list = []              # 阶段式策略梯度（决策记录 266）存下的最近几版局面
    macro_pending: dict = {}        # 局外优势跨段计算（决策记录 255）：局编号 → [最后收到的时刻, 各段的局外决策]
    group_pending: dict = {}        # 同种子成组采集（决策记录 267）：种子 → [第一局到的时刻, 组大小, 各局的延后局外轨迹（没有局外步为 None）]
    group_c = 0.0                   # 同组基线的控制变量系数（逐版滑动平均，从 0 起：先不加噪声，等估出来再用）
    if cfg.combat_anchor_rows not in ("combat", "macro", "all"):
        raise SystemExit(f"combat_anchor_rows 只能是 combat / macro / all（现在是 {cfg.combat_anchor_rows!r}）")
    if cfg.macro_seg_adv and not cfg.macro_cross_segment:
        raise SystemExit("macro_seg_adv 要求 macro_cross_segment（局外步仍要等整局、按组放行，决策记录 269）")
    if cfg.run_group > 1 and not cfg.macro_cross_segment:
        raise SystemExit("run_group > 1 要求 macro_cross_segment（同组的局要按整局算局外优势，决策记录 267）")
    t_last = time.time(); steps_since = 0
    got_any = False
    card_acc = {"cards": [], "ancient": []}
    try:
        while total_steps < cfg.total_steps:
            # ---- 收集
            # 按步判断过旧：只把用太旧权重做出的那几步排除在学习之外，整局照常保留（优势按整局算）。
            # 以前按开局版本整局丢弃，打得越深的局越容易被丢，学不到 Boss 和后几幕。
            # 全部过旧的局不参与学习，但结局照样计入统计（ended），统计才不偏向短局。
            trajs, ended, run_items, n_steps, dropped, stale, n_deferred = [], [], [], 0, 0, 0, 0
            long_kept = 0       # 超长局里本该判过旧、照样入训的步数（long_keep_steps）
            grp_acc: dict = {}
            if cfg.macro_cross_segment:      # 太久等不到最后一段的整局（采集端重启、连接断开）丢掉
                for g in [g for g, v in macro_pending.items() if time.time() - v[0] > cfg.macro_defer_ttl]:
                    del macro_pending[g]
            for g in [g for g, v in group_pending.items() if time.time() - v[0] > cfg.group_ttl]:     # 等不齐的组按已到的局放行
                for d in group_release(group_pending.pop(g)[2], version, cfg, grp_acc, group_c):
                    trajs.append(d); n_steps += int(d["fresh"].sum()); n_deferred += int(d["fresh"].sum())
            # 落后版本数的拆分（决策记录 196，只统计）：actor 为送出时采集进程已落后几版（v_send − 该步版本），queue 为送出到收到之间又过了几版
            # （收到时版本 − v_send，含采集进程在队列满时的阻塞）；按步计数。rank 为各采集进程 [收到步数, 过旧步数, 等待秒数合计（按步）]
            lag_split = {k: {"actor": np.zeros(LAG_BINS, np.int64), "queue": np.zeros(LAG_BINS, np.int64)} for k in ("local", "remote")}
            rank_lag: dict[int, list[float]] = {}
            # 各类练习这一批的 [结束局数, 收到步数, 入训步数, 胜场, 战损合计（战斗练习：血量变化占最大生命，负为掉血）]（决策记录 197，只统计）
            cur: dict[str, list[float]] = {}
            phi_fresh = []      # 这一批新到的打分标签：训练打分头之前先打一次分，看它在没见过的局面上的区分能力
            t_collect = time.time()
            while n_steps < cfg.batch_steps:
                # 刚启动时引擎要一起初始化（v0.111.0 起 192 个引擎同时启动要 60 多秒），收到第一条轨迹之前多等一会
                wait = 60 if got_any else 600
                try:
                    kind, rank, item = (feeder or out_q).get(timeout=wait)
                except queue.Empty:
                    raise RuntimeError(f"{wait} 秒没有收到轨迹，采集进程可能已退出")
                got_any = True
                if kind == "error":
                    raise RuntimeError(f"采集进程 {rank} 出错：\n{item}")
                item["remote"] = rank >= REMOTE_RANK0       # 远程采集机（bridge.py）的采集进程编号从 REMOTE_RANK0 起
                if item["mode"] == MODE_RUN:
                    run_items.append(item)
                if not item.get("segment"):     # 整局中途送出的段只用来学习，不算一局
                    recent.append(item); ended.append(item)
                if item.get("wp") and wp_net is not None:
                    for r in item["wp"]:
                        if len(wp_buf) < cfg.wp_buffer:
                            wp_buf.append(r)
                        else:
                            wp_buf[wp_pos] = r; wp_pos = (wp_pos + 1) % cfg.wp_buffer
                if item.get("phi_label"):
                    phi_buf.append(item["phi_label"]); phi_fresh.append(item["phi_label"])
                # 版本差为负（回退之后作废时间线上的轨迹，版本号比现在大）也算过旧（决策记录 196）
                item["fresh"] = (version - item["vers"] >= 0) & (version - item["vers"] <= cfg.max_policy_lag)
                if cfg.long_keep_steps and item["len"] >= cfg.long_keep_steps:      # 超长局整份入训，不按过旧丢（用户，2026-11-07）
                    long_kept += int((~item["fresh"] & (version - item["vers"] >= 0)).sum())
                    item["fresh"] = version - item["vers"] >= 0
                if "v_send" in item:
                    src = "remote" if item["remote"] else "local"
                    lag_split[src]["actor"] += np.bincount(np.clip(item["v_send"] - item["vers"], 0, LAG_BINS - 1), minlength=LAG_BINS)
                    lag_split[src]["queue"][int(np.clip(version - item["v_send"], 0, LAG_BINS - 1))] += item["len"]
                    rl = rank_lag.setdefault(int(rank), [0, 0, 0.0])
                    rl[0] += item["len"]; rl[1] += int(item["len"] - item["fresh"].sum()); rl[2] += item["len"] * (time.time() - item["t_send"])
                if item["mode"] == MODE_RUN:
                    item["lag"] = version - item["vers"]
                k = int(item["fresh"].sum()); stale += item["len"] - k
                if cfg.macro_cross_segment and item["mode"] == MODE_RUN and item.get("gid"):
                    deferred = macro_defer(item, macro_pending, version, cfg)
                    if cfg.run_group > 1 and item.get("grp") and not item.get("segment"):
                        seed, gk = item["grp"][0], int(item["grp"][1])
                        ent = group_pending.setdefault(seed, [time.time(), gk, []]); ent[2].append(deferred)
                        if len(ent[2]) >= ent[1]:
                            for d in group_release(group_pending.pop(seed)[2], version, cfg, grp_acc, group_c):
                                trajs.append(d); n_steps += int(d["fresh"].sum()); n_deferred += int(d["fresh"].sum())
                    elif deferred is not None:
                        trajs.append(deferred); n_steps += int(deferred["fresh"].sum()); n_deferred += int(deferred["fresh"].sum())
                    k = int(item["fresh"].sum())        # 这一段的局外步已挪走，只剩战斗步按段训练
                c = cur.setdefault(cur_key(item), [0, 0, 0, 0, 0.0])
                c[1] += item["len"]; c[2] += k
                if not item.get("segment"):
                    c[0] += 1; c[3] += int(bool(item.get("win"))); c[4] += float(item.get("hp_delta") or 0.0) if item["mode"] == MODE_COMBAT else 0.0
                if k == 0:
                    dropped += 1
                    continue
                trajs.append(item); n_steps += k
            t_collect = time.time() - t_collect; t_prep = time.time()
            if grp_acc.get("mm", 0.0) > 0:     # 这一版放行的组估出的最优系数，滑动平均进下一版用
                group_c = 0.9 * group_c + 0.1 * float(np.clip(grp_acc["am"] / grp_acc["mm"], 0.0, 1.0))

            # ---- 优势与目标
            # 强制探索改过的步（actor.explore_choice）：没有这个字段的旧轨迹视为全部未改
            for t in trajs:
                if t.get("forced") is None or len(t["forced"]) != t["len"]:
                    t["forced"] = np.zeros(t["len"], bool)
                # 改选纳入行为分布（决策记录 197）：带 logp_mu 的轨迹，比率用 π_新 / μ，改选的步正常进策略梯度、价值目标用实际回报；
                # 没有 logp_mu 的（旧代码的采集端）照原来的办法：改选的步不进策略梯度、价值目标取原值。两种都在改选处截断前面的优势
                t["mu_ok"] = bool(cfg.explore_offpolicy) and t.get("logp_mu") is not None and len(t["logp_mu"]) == t["len"]
                t["legacy"] = t["forced"] if not t["mu_ok"] else np.zeros(t["len"], bool)
            # 优势只在局外改选处截断；战斗定向探索改选的战斗步不截断（战斗价值头可靠，截断会让前面的步少看到后果，决策记录 238），combat_explore_cut 时也截断
            def step_cut(t):        # 优势截断处：强制探索的步；combat_fight_boot 时另加每场战斗打完处（决策记录 270，只对整局）
                c = t["forced"] if cfg.combat_explore_cut else t["forced"] & t["macro"]
                if cfg.combat_fight_boot and t["mode"] == MODE_RUN:
                    c = c | fight_cuts(t["ragged"].header, at_map=cfg.combat_fight_boot_map)
                return c
            cuts = [None if t.get("deferred") else step_cut(t) for t in trajs]
            luck = [0, 0.0, 0.0, 0.0, 0.0]     # 运气校正（config.luck_cv）：[步数, Σ运气, Σ运气², Σδ², Σδ'²]
            advs, rets = zip(*((t["adv"].copy(), t["ret"].copy()) if t.get("deferred") else      # 跨段算好的局外优势（决策记录 255）
                               gae(t["values"], t["rewards"], cfg.gamma, cfg.lam, c, t.get("boot", 0.0),
                                   t.get("nv") if cfg.luck_cv_apply else None, luck) for t, c in zip(trajs, cuts)))
            if not cfg.luck_cv_apply:      # 只量不用：照样统计运气有多大，优势不改
                for t in trajs:
                    if not t.get("deferred") and t.get("nv") is not None:
                        v_ = t["values"]; nx = np.append(v_[1:], t.get("boot", 0.0)); nv_ = t["nv"]
                        if len(nv_) == len(v_) and (m_ := np.isfinite(nv_)).any():
                            lk = nx[m_] - nv_[m_]; d0 = t["rewards"][m_] + cfg.gamma * nx[m_] - v_[m_]
                            luck[0] += int(m_.sum()); luck[1] += float(lk.sum()); luck[2] += float((lk * lk).sum())
                            luck[3] += float((d0 * d0).sum()); luck[4] += float(((d0 - cfg.gamma * lk) ** 2).sum())
            for t, a_, r_ in zip(trajs, advs, rets):
                if not t.get("deferred"):
                    gae_macro(t["values"], t["rewards"], t["macro"], cfg.macro_lam, a_, r_, t["forced"], t.get("boot", 0.0))
                r_[t["legacy"]] = t["values"][t["legacy"]]      # 价值目标取原值：这一步的回报是被改选的动作的，不是策略的（只对旧办法的轨迹）
            # 战斗收尾落差按幕去均值（愚者决策记录第 8 条）：只动战斗步，局外步的优势已由 gae_macro 改写
            fight_end = fight_end_center(trajs, advs, rets, cuts, cfg.gamma * cfg.lam) \
                if cfg.combat_fight_boot and cfg.combat_fight_boot_map and cfg.combat_fight_center else {}
            data = Ragged.concat([t["ragged"] for t in trajs])
            actions = np.concatenate([t["actions"] for t in trajs]); old_logp = np.concatenate([t["logp"] for t in trajs])
            beh_logp = np.concatenate([t["logp_mu"] if t["mu_ok"] else t["logp"] for t in trajs])    # 算比率用的行为概率
            adv = np.concatenate(advs); ret = np.concatenate(rets)
            macro = np.concatenate([t["macro"] for t in trajs])
            fresh = np.concatenate([t["fresh"] for t in trajs])
            forced = np.concatenate([t["forced"] for t in trajs])
            legacy = np.concatenate([t["legacy"] for t in trajs])
            # 照部署方式确定性出手的步（search_collect，sts2rl/actor_search.py）：动作不是策略采样的，不进策略梯度、不参与战斗优势的标准化；回报与价值照常
            det = np.concatenate([t["det"] if t.get("det") is not None and len(t["det"]) == t["len"] else np.zeros(t["len"], bool) for t in trajs])
            use = np.flatnonzero(fresh)
            if cfg.combat_value_frac < 1:       # 只训局外（决策记录 267）：战斗步只抽一部分训价值头与辅助头
                use = use[macro[use] | (np.random.random(len(use)) < cfg.combat_value_frac)]
            # 非战斗决策与战斗步的优势尺度不同，分开标准化；策略损失里非战斗决策加权
            adv_n = np.zeros_like(adv)
            for grp in (fresh & macro & ~legacy, fresh & ~macro & ~det):
                if grp.any():
                    adv_n[grp] = (adv[grp] - adv[grp].mean()) / (adv[grp].std() + 1e-8)
            w_pi = np.where(legacy, 0.0, np.where(macro, 0.0 if cfg.combat_only_pi else cfg.macro_pi_coef, 0.0 if cfg.macro_only_pi else 1.0)).astype(np.float32)   # 旧办法：强制探索的步不进策略梯度
            w_pi[det] = 0.0
            det_used = int(det[use].sum())
            N = len(use)
            # 局外有效样本（决策记录 197，只统计）：按决策类型 [入训步数, 其中多选, 其中改选, 多选步 |标准化优势| 合计, 多选步 策略权重 × |优势| 合计]
            dec = data.header[:, 1].astype(np.int64); multi = np.diff(data.cand_off) > 1
            macro_eff = {}
            for d in np.unique(dec[fresh & macro]):
                m = fresh & macro & (dec == d); mm = m & multi
                macro_eff[DecisionKind(int(d)).name] = [int(m.sum()), int(mm.sum()), int((m & forced).sum()),
                                                        round(float(np.abs(adv_n[mm]).sum()), 1), round(float((w_pi[mm] * np.abs(adv_n[mm])).sum()), 1)]
            if (run_dir / "DUMP").exists():     # 调优用：把这一批学习数据存下来离线测速，存完删掉标记
                import pickle
                (run_dir / "DUMP").unlink()
                with open(run_dir / "batch_dump.pkl", "wb") as f:
                    pickle.dump(data.select(use), f)

            # ---- PPO 更新
            # 专项日程（决策记录 287）：学习率线性衰减、锚定系数线性降到 0、开头只训价值一侧
            lr_now = cfg.lr * (1.0 - (1.0 - cfg.lr_final_frac) * min(1.0, version / cfg.lr_decay_until)) if cfg.lr_decay_until else cfg.lr
            anc_coef = cfg.combat_anchor_coef * (max(0.0, 1.0 - version / cfg.combat_anchor_until) if cfg.combat_anchor_until else 1.0)
            teacher_on = teacher is not None and anc_coef > 0
            value_only = version < cfg.value_warmup_until
            if warmup_left > 0:
                lr_now = lr_now * (0.1 + 0.9 * (cfg.opt_warmup - warmup_left + 1) / cfg.opt_warmup)
                warmup_left -= 1
            if warmup_left > 0 or cfg.lr_decay_until or lr_now != cfg.lr:
                for g in opt.param_groups:
                    g["lr"] = lr_now * ((cfg.sf_lr or cfg.lr) / cfg.lr if cfg.sf_adam else 1.0)     # 免调度的学习率按同样比例
                if muon is not None:
                    muon.lr = (cfg.muon_lr or cfg.lr) * lr_now / cfg.lr
            t_learn = time.time(); t_prep = t_learn - t_prep
            stats = {"loss_pi": 0.0, "loss_v": 0.0, "entropy": 0.0, "approx_kl": 0.0, "clip_frac": 0.0,
                     "entropy_macro": 0.0, "loss_v_macro": 0.0, "entropy_map": 0.0}; nb = 0; nb_macro = 0; nb_map = 0
            map_coef = (cfg.ent_coef_map if cfg.ent_coef_map is not None and (cfg.ent_map_until is None or version < cfg.ent_map_until)
                        else cfg.ent_coef_macro)
            net.train()
            epochs_done = 0; timing = {}
            # ppo_olp 为行为概率（算比率），ppo_klp 为策略自己的概率（算 KL，改选的步不计：那是策略本来很少选的动作）
            # ppo_g：局外学习诊断（决策记录 200，只统计）的分组：多选的局外决策按 决策类型 × 2 + 是否改选，战斗步 DIAG_COMBAT，单选的局外决策 DIAG_SINGLE（不统计）
            anc = np.concatenate([t["anc"] if t.get("anc") is not None and len(t["anc"]) == t["len"] else np.zeros(t["len"], bool) for t in trajs])
            diag_g = np.where(macro & multi, np.where(anc, DIAG_ANCIENT + forced, dec * 2 + forced),
                              np.where(macro, DIAG_SINGLE, np.where(forced, DIAG_COMBAT_F, DIAG_COMBAT))).astype(np.int64)
            per_step = {"ppo_a": actions, "ppo_olp": beh_logp, "ppo_adv": adv_n, "ppo_ret": ret, "ppo_w": w_pi,
                        "ppo_klp": old_logp, "ppo_klw": (~forced).astype(np.float32), "ppo_g": diag_g}
            if cfg.model.combat_aux:     # 战斗辅助头的标签（决策记录 261）：旧采集端、跨段的局外轨迹没有，按没有标签算
                aux_get = lambda t, k, dt: t[k] if t.get(k) is not None and len(t[k]) == t["len"] else np.zeros(t["len"], dt)
                per_step["aux_m"] = np.concatenate([aux_get(t, "aux_m", bool) for t in trajs]).astype(np.float32) * (~macro)
                per_step["aux_win"] = np.concatenate([aux_get(t, "aux_win", np.float32) for t in trajs])
                per_step["aux_hp"] = np.concatenate([aux_get(t, "aux_hp", np.float32) for t in trajs])
            if fh_opt is not None:      # 候选未来读数头的标签：旧采集端、局外步没有，按没有标签算
                fh_get = lambda t, k, shape, dt: (t[k] if t.get(k) is not None and len(t[k]) == t["len"] else np.zeros((t["len"],) + shape, dt))
                per_step["fh_m"] = np.concatenate([fh_get(t, "fh_m", (), bool) for t in trajs]).astype(np.float32) * (~macro)
                per_step["fh_y"] = np.concatenate([fh_get(t, "fh_y", (cfg.model.fh_dims,), np.float32) for t in trajs]).astype(np.float32)
            if co is not None:      # 跳过稳赢战斗的结果头的标签：旧采集端、练习战、跨段的局外轨迹没有，按没有标签算
                co_get = lambda t, k, shape, dt, fill: (t[k] if t.get(k) is not None and len(t[k]) == t["len"]
                                                        else np.full((t["len"],) + shape, fill, dt))
                per_step["co_m"] = np.concatenate([co_get(t, "co_m", (), bool, False) for t in trajs]).astype(np.float32)
                per_step["co_hp"] = np.concatenate([co_get(t, "co_hp", (), np.float32, 0.0) for t in trajs]).astype(np.float32)
                per_step["co_vid"] = np.concatenate([co_get(t, "co_vid", (CO_POT_SLOTS,), np.int64, 0) for t in trajs]).astype(np.int64)
                per_step["co_y"] = np.concatenate([co_get(t, "co_y", (CO_POT_SLOTS,), np.float32, -1.0) for t in trajs]).astype(np.float32)
            co_acc = torch.zeros(4, device=device)    # 结果头：血量交叉熵之和、药水交叉熵之和、训了的小批数、药水标签数
            fh_acc = torch.zeros(2, device=device)    # 未来读数头：归一化均方误差之和、有标签的小批数
            rcy_acc = torch.zeros(3, device=device)   # 回灌（第 26 条）：第二遍对第一遍的 KL 之和、首选相同的比例之和、小批数
            hl_beta_cur = float(net.hl_beta) if cfg.model.value_hl else 0.0
            # 分类头交叉熵的系数（决策记录 261 补记四）：hl_coef_from 给出时从那一版起用 hl_coef_ramp 版从 0 线性升到 hl_coef；
            # 数值头的均方误差在只读主干或 hl_scalar_full 时一直按全权重，否则按 1 − β
            hl_coef_cur = (cfg.hl_coef * float(np.clip((version - cfg.hl_coef_from) / max(1, cfg.hl_coef_ramp), 0.0, 1.0))
                           if cfg.hl_coef_from is not None else cfg.hl_coef)
            w_scalar = 1.0 if (cfg.model.hl_detach or cfg.hl_scalar_full) else 1.0 - hl_beta_cur
            # 统计量留在显卡上累加，每遍末尾才取回一次（KL 提前停要用）；原来每个小批约 10 次 .item() / .any()，每次都让主机等显卡
            acc = torch.zeros(10, device=device)      # loss_pi, loss_v, entropy, kl, clip_frac, ent_macro, v_macro, n_macro, ent_map, n_map
            # 决策记录 261：交叉熵、分类头期望的均方误差、数值头的均方误差（全部与局外各一份）；辅助头输赢交叉熵、掉血均方误差、平均标签、平均预测、有标签的小批数
            acc2 = torch.zeros(11, device=device)
            acc3 = torch.zeros(3, device=device)      # 战斗锚定（决策记录 267）：KL 之和、战斗步首选与老师一致的比例之和、计入的小批数
            acc_skl = torch.zeros(4, device=device)   # 搜索 KL：KL 之和、首选与搜索目标首选一致的比例之和、计入的小批数、标签年龄（版）之和
            if skl is not None:
                skl.version = version
            # 训练健康统计（决策记录 220，只统计）：这一批开始时的价值对回报的解释方差（分战斗、局外）；裁剪前梯度范数；各组参数这一版的更新量占权重的比例
            vals_all = np.concatenate([t["values"] for t in trajs]).astype(np.float64)
            health = {"ev": {"combat": explained_variance(vals_all[fresh & ~macro], ret[fresh & ~macro].astype(np.float64)),
                             "macro": explained_variance(vals_all[fresh & macro & ~legacy], ret[fresh & macro & ~legacy].astype(np.float64))}}
            if value_probe:
                health["value_probe"] = value_probe.observe(net, data, trajs, ret, fresh & ~legacy, macro, version,
                    actions=actions, behavior_logp=beh_logp, advantages=adv_n, policy_weights=w_pi)
            hgroups = _health_groups(net)
            with torch.no_grad():
                h_before = {k: [p.detach().clone() for p in ps] for k, ps in hgroups.items()}
            gn_acc = torch.zeros(3, device=device)    # 裁剪前梯度范数之和、超过 max_grad_norm 的小批数、计入的小批数
            skipped = torch.zeros((), device=device)  # 梯度不是有限值、跳过的小批数（这些小批也不计入上面的统计）
            diag = torch.zeros(cfg.epochs, DIAG_N, len(DIAG_COLS), device=device)     # 每遍、每组的诊断累加量（列见 DIAG_COLS）
            acc_n = max(1, int(cfg.grad_accum))
            gd_acc = {"n": 0}
            n_mb_ep = -(-len(use) // cfg.minibatch)    # 每遍的小批数（最后一组可能不满 acc_n 个）
            for _ in range(cfg.epochs):
                ep = torch.zeros(2, device=device)      # 本遍 KL 之和、小批数
                k_mb = 0
                for idx, b in minibatches(data, np.random.permutation(use), cfg.minibatch, device, timing, per_step, packed=net.fast, workers=cfg.collate_workers):
                    a, olp, A, R, W = b["ppo_a"], b["ppo_olp"], b["ppo_adv"], b["ppo_ret"], b["ppo_w"]
                    klp, keep = b["ppo_klp"], b["ppo_klw"]
                    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=cfg.learner_bf16 and device.type == "cuda"):
                        logits, v, _phi, ex = fwd(b)
                    logits, v = logits.float(), v.float()
                    dist = torch.distributions.Categorical(logits=logits, validate_args=False)     # 参数校验会同步
                    lp = dist.log_prob(a); ratio = torch.exp(lp - olp)
                    l_pi = -(W * torch.min(ratio * A, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * A)).sum() / W.sum()
                    l_v = F.mse_loss(v, R)
                    l_vt = l_v      # 进损失的价值项；分类价值头打开时换成 (1 − β) × 数值头均方误差 + hl_coef × 交叉熵
                    if cfg.model.value_hl:
                        z = ex["hl_logits"].float(); vs = ex["v_scalar"].float()
                        logq = F.log_softmax(z, -1)
                        l_hl = -(hl_target(R, net.hl_edges, cfg.hl_sigma) * logq).sum(-1).mean()
                        l_vs = F.mse_loss(vs, R)
                        # 只读主干时数值头一直按全权重训练主干（交叉熵传不回主干，再按 1 − β 降数值头，主干就没有价值方面的梯度了）
                        l_vt = w_scalar * l_vs + hl_coef_cur * l_hl
                    ent_all = dist.entropy(); ent = ent_all.mean()
                    # 熵奖励按行取系数：局外决策用 ent_coef_macro（其中地图决策用 map_coef），其余用 ent_coef
                    is_map = b["decision"] == int(DecisionKind.MAP)
                    ent_w = torch.where(is_map, map_coef, torch.where(b["macro"], cfg.ent_coef_macro, 0.0 if cfg.macro_only_pi else cfg.ent_coef))
                    if cfg.combat_only_pi:      # 战斗阶段（决策记录 270）：局外步不要熵奖励
                        ent_w = torch.where(b["macro"], 0.0, ent_w)
                    loss = l_pi + cfg.vf_coef * (cfg.ppg_pi_vf if cfg.ppg_every > 0 else 1.0) * l_vt - (ent_w * ent_all).mean()
                    if cfg.model.combat_aux:
                        am = b["aux_m"]; na = am.sum().clamp(min=1); ao = ex["aux"].float()
                        l_aw = (F.binary_cross_entropy_with_logits(ao[:, 0], b["aux_win"], reduction="none") * am).sum() / na
                        l_ah = (((ao[:, 1] - b["aux_hp"]) ** 2) * am).sum() / na
                        loss = loss + cfg.aux_coef * (l_aw + cfg.aux_hp_w * l_ah)
                    if fh_opt is not None:      # 候选未来读数头：真实所选候选的读数，均方误差按各维滑动标准差归一（愚者决策记录第 25 条）
                        fm_ = b["fh_m"]
                        rowmap = torch.full(logits.shape, -1, dtype=torch.long, device=device).index_put_(
                            (b["cand_step"], b["cand_slot"]), torch.arange(b["cand_step"].shape[0], device=device))
                        rows = rowmap.gather(1, a[:, None]).squeeze(1)
                        fm_ = fm_ * (rows >= 0).float(); nf = fm_.sum().clamp(min=1)
                        pf = ex["fh"].float()[rows.clamp(min=0)]
                        yf = b["fh_y"]
                        with torch.no_grad():       # 滑动标准差：有标签的步超过 16 个的小批才更新（不同步主机）
                            mu_ = (yf * fm_[:, None]).sum(0) / nf
                            sd_b = (((yf - mu_) ** 2 * fm_[:, None]).sum(0) / nf).sqrt().clamp(min=1e-3)
                            w_sd = 0.01 * (fm_.sum() > 16).float()
                            fh_sd.mul_(1.0 - w_sd).add_(w_sd * sd_b)
                        l_fh = ((((pf - yf) / fh_sd) ** 2).mean(1) * fm_).sum() / nf
                        loss = loss + cfg.fh_coef * l_fh
                        fh_acc += torch.stack([l_fh.detach() * (fm_.sum() > 0).float(), (fm_.sum() > 0).float()])
                    if "rcy_l1" in ex:      # 回灌改了多少：KL(第二遍 ‖ 第一遍) 按行平均，首选相同的比例（只读，不进损失）
                        with torch.no_grad():
                            lp2 = torch.log_softmax(logits, -1); lp1 = torch.log_softmax(ex["rcy_l1"].float(), -1)
                            p2 = lp2.exp()
                            kl_r = torch.where(p2 > 0, p2 * (lp2 - lp1), torch.zeros_like(p2)).sum(-1).mean()
                            agree = (logits.argmax(-1) == ex["rcy_l1"].argmax(-1)).float().mean()
                            rcy_acc += torch.stack([kl_r, agree, torch.ones_like(agree)])
                    if co is not None:      # 结果头：单独的损失与 Adam 步，只读主干的 detach 副本（不进 loss，不影响策略与价值的梯度）
                        cm = b["co_m"]
                        if bool(cm.sum() > 0):
                            hl_c, pl_c = co(ex["g"], net.id_emb, b["co_vid"])
                            l_ch = ((-(hl_target(b["co_hp"], co.edges, 0.75) * F.log_softmax(hl_c, -1)).sum(-1)) * cm).sum() / cm.sum()
                            pm_ = ((b["co_y"] >= 0) & (b["co_vid"] > 0)).float() * cm[:, None]
                            l_cp = (F.binary_cross_entropy_with_logits(pl_c, b["co_y"].clamp(min=0), reduction="none") * pm_).sum() / pm_.sum().clamp(min=1)
                            co_opt.zero_grad(set_to_none=True)
                            (l_ch + 0.5 * l_cp).backward()
                            co_opt.step()
                            co_acc += torch.stack([l_ch.detach(), l_cp.detach(), torch.ones((), device=device), pm_.sum().detach()])
                    if teacher_on:     # 战斗锚定（决策记录 267）：战斗步上 KL(老师 ‖ 当前)，按战斗步平均
                        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=cfg.learner_bf16 and device.type == "cuda"):
                            t_lg = teacher(b)[0]
                        t_lp = torch.log_softmax(t_lg.float(), -1); s_lp = torch.log_softmax(logits, -1)
                        cm = (torch.ones_like(b["macro"]) if cfg.combat_anchor_rows == "all" else b["macro"] if cfg.combat_anchor_rows == "macro" else ~b["macro"]).float(); ncm = cm.sum().clamp(min=1)     # 拴哪一类步（决策记录 270；all 为全部，决策记录 274）
                        kl_row = torch.where(t_lp > -1e30, t_lp.exp() * (t_lp - s_lp), torch.zeros_like(t_lp)).nan_to_num(0.0).sum(-1)
                        l_anc = (kl_row * cm).sum() / ncm
                        loss = loss + anc_coef * l_anc
                    skl_on = False
                    if skl is not None:     # 搜索 KL（试点）：另一批搜索过的局面上 KL(搜索改进后的分布 ‖ 当前策略)；这个小批没有现成的标签批就不加
                        sb = skl.next_batch()
                        if sb is not None:
                            d_s, t_s, age_s = sb
                            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=cfg.learner_bf16 and device.type == "cuda"):
                                # 用学习端已编译的前向（形状本来就是动态的）：不编译时每版多约 5.5 秒（每次调用的固定开销）；编译出错就退回不编译的
                                try:
                                    lg_s = (fwd if skl_compiled[0] else net.forward_all)(d_s)[0]
                                except Exception as e:
                                    if not skl_compiled[0]:
                                        raise
                                    skl_compiled[0] = False
                                    print(f"搜索 KL：编译的前向出错（{str(e)[:200]}），改用不编译的", flush=True)
                                    lg_s = net.forward_all(d_s)[0]
                            l_skl, agree_s = search_kl_loss(lg_s, t_s)
                            loss = loss + cfg.search_kl_coef * l_skl
                            skl_on = True
                    if cfg.grad_decomp:      # 梯度分解（诊断，决策记录 261 补记）：各项损失（乘好系数）对主干的梯度，不更新参数，做完就退出
                        terms = {"policy": l_pi, "entropy": -(ent_w * ent_all).mean(),
                                 "value_scalar": cfg.vf_coef * ((w_scalar * l_vs) if cfg.model.value_hl else l_v)}
                        if cfg.model.value_hl:
                            terms["hl_ce"] = cfg.vf_coef * hl_coef_cur * l_hl
                        if cfg.model.combat_aux:
                            terms["aux_win"] = cfg.aux_coef * l_aw; terms["aux_hp"] = cfg.aux_coef * cfg.aux_hp_w * l_ah
                        if teacher_on:
                            terms["anchor"] = anc_coef * l_anc
                        terms["total"] = loss
                        grad_decomp_step(net, terms, gd_acc)
                        if gd_acc["n"] >= cfg.grad_decomp_mb:
                            out = grad_decomp_report(gd_acc, version)
                            (run_dir / "grad_decomp.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
                            print("梯度分解：" + json.dumps(out, ensure_ascii=False), flush=True)
                            raise SystemExit(0)
                        k_mb += 1; nb += 1
                        continue
                    # 梯度累积（决策记录 225）：一组 acc_n 个小批的损失按组内小批数平均后累积，组末走一步；acc_n 为 1 时与原来逐小批更新相同
                    if k_mb % acc_n == 0:
                        opt.zero_grad(set_to_none=True)
                        if fh_opt is not None:
                            fh_opt.zero_grad(set_to_none=True)
                        g_size = min(acc_n, n_mb_ep - k_mb)
                    (loss / g_size).backward()
                    k_mb += 1
                    if k_mb % acc_n == 0 or k_mb >= n_mb_ep:
                        if value_only:          # 只训价值一侧（决策记录 287）：其余参数没有梯度，融合 Adam 不更新它们
                            for q in nonvalue_params:
                                q.grad = None
                        bad = guarded_step(net, opt, cfg.max_grad_norm, own_clip); skipped += bad
                        if fh_opt is not None and not bool(bad):     # 读数头：梯度已在 guarded_step 里单独裁剪；非有限时跳过
                            fh_opt.step()
                        if cfg.weight_decay > 0:    # 解耦权重衰减（决策记录 287）：只训价值时只衰减价值一侧
                            with torch.no_grad():
                                torch._foreach_mul_(decay_value if value_only else decay_params, 1.0 - lr_now * cfg.weight_decay)
                        with torch.no_grad():
                            gn_ok = (1.0 - bad) * torch.isfinite(guarded_step.last_gn).float()
                            gn_acc += torch.stack([torch.nan_to_num(guarded_step.last_gn.float(), nan=0.0, posinf=0.0) * gn_ok,
                                                   (guarded_step.last_gn > cfg.max_grad_norm).float() * gn_ok, gn_ok])
                    else:
                        bad = torch.zeros((), device=device)
                    with torch.no_grad():
                        r_pi = torch.exp(lp - klp)
                        kl = (((r_pi - 1) - (lp - klp)) * keep).sum() / keep.sum().clamp(min=1)
                        mb = b["macro"].float(); mm = is_map.float()
                        n_mb, n_mm = mb.sum(), mm.sum()
                        # 局外决策单独看熵与价值误差（熵在 entropy 里与战斗步混在一起）；按小批平均后再按“有这类决策的小批”平均，与原来相同
                        row = torch.stack([l_pi.detach(), l_v.detach(), ent.detach(), kl, ((ratio - 1).abs() > cfg.clip).float().mean(),
                                           (ent_all * mb).sum() / n_mb.clamp(min=1), ((v - R) ** 2 * mb).sum() / n_mb.clamp(min=1), (n_mb > 0).float(),
                                           (ent_all * mm).sum() / n_mm.clamp(min=1), (n_mm > 0).float()])
                        acc += torch.where(bad > 0, 0.0, row)
                        if cfg.model.value_hl or cfg.model.combat_aux:
                            r2 = torch.zeros(11, device=device)
                            if cfg.model.value_hl:
                                v_hl = logq.exp() @ net.hl_centers
                                r2[0] = l_hl.detach(); r2[1] = ((v_hl - R) ** 2).mean(); r2[2] = l_vs.detach()
                                hm = -(hl_target(R, net.hl_edges, cfg.hl_sigma) * logq).sum(-1)
                                r2[3] = (hm * mb).sum() / n_mb.clamp(min=1); r2[4] = ((v_hl - R) ** 2 * mb).sum() / n_mb.clamp(min=1)
                                r2[5] = ((vs - R) ** 2 * mb).sum() / n_mb.clamp(min=1)
                            if cfg.model.combat_aux:
                                has = (am.sum() > 0).float()
                                r2[6] = l_aw.detach() * has; r2[7] = l_ah.detach() * has
                                r2[8] = (b["aux_win"] * am).sum() / na * has; r2[9] = (torch.sigmoid(ao[:, 0]) * am).sum() / na * has; r2[10] = has
                            acc2 += torch.where(bad > 0, 0.0, r2)
                        if teacher_on:
                            agree = ((t_lg.argmax(-1) == logits.argmax(-1)).float() * cm).sum() / ncm
                            acc3 += torch.where(bad > 0, 0.0, torch.stack([l_anc.detach(), agree, (cm.sum() > 0).float()]))
                        if skl_on:
                            acc_skl += torch.where(bad > 0, 0.0, torch.stack([l_skl.detach(), agree_s, torch.ones((), device=device),
                                                                              torch.full((), age_s, device=device)]))
                        ep += torch.where(bad > 0, 0.0, torch.stack([kl, torch.ones((), device=device)]))
                        pos, neg = (A > 0).float(), (A < 0).float(); lpd = lp.detach()
                        cols = torch.stack([torch.ones_like(ratio), ratio, ratio * ratio, ((ratio - 1).abs() > cfg.clip).float(),
                                            ((ratio < 0.5) | (ratio > 2.0)).float(), lpd, lpd * pos, pos, lpd * neg, neg,
                                            pos * (ratio > 1 + cfg.clip).float(), neg * (ratio < 1 - cfg.clip).float()], 1)
                        diag[epochs_done].index_add_(0, b["ppo_g"], cols * (1.0 - bad))
                    nb += 1
                epochs_done += 1
                epoch_kl, epoch_nb = ep.tolist()
                if epoch_kl / max(epoch_nb, 1) > 1.5 * cfg.target_kl:
                    break
            acc = acc.tolist(); nb_macro, nb_map = acc[7], acc[9]
            n_skip = int(skipped.item()); nb -= n_skip
            if n_skip:
                # 留档这一批学习数据，离线查非有限值从哪来；最多留 5 份（每份约 200 MB）
                nan_dir = run_dir / "nan_batches"; nan_dir.mkdir(exist_ok=True)
                if len(list(nan_dir.glob("v*.pkl"))) < 5:
                    import pickle
                    with open(nan_dir / f"v{version + 1}.pkl", "wb") as f:
                        pickle.dump({"version": version + 1, "skipped": n_skip, "data": data.select(use),
                                     **{k: v[use] for k, v in per_step.items()}, "adv_raw": adv[use], "forced": forced[use], "macro": macro[use]}, f)
                print(f"第 {version + 1} 版：{n_skip} 个小批梯度不是有限值，已跳过", flush=True)
            for i, k in enumerate(("loss_pi", "loss_v", "entropy", "approx_kl", "clip_frac")):
                stats[k] = acc[i]
            stats["entropy_macro"], stats["loss_v_macro"], stats["entropy_map"] = acc[5], acc[6], acc[8]
            for k in stats: stats[k] /= max(nb_macro if k.endswith("_macro") else nb_map if k.endswith("_map") else nb, 1)
            stats["ent_coef_map"] = map_coef
            a2 = acc2.tolist()
            if cfg.model.value_hl:      # 分类价值头（决策记录 261）：全部步与局外步各自的交叉熵、分类头期望与数值头的均方误差（loss_v 是实际用的价值的）
                stats |= {"hl_beta": round(hl_beta_cur, 4), "hl_coef": round(hl_coef_cur, 4), "loss_hl": a2[0] / max(nb, 1), "loss_v_hl": a2[1] / max(nb, 1), "loss_v_scalar": a2[2] / max(nb, 1),
                          "loss_hl_macro": a2[3] / max(nb_macro, 1), "loss_v_hl_macro": a2[4] / max(nb_macro, 1), "loss_v_scalar_macro": a2[5] / max(nb_macro, 1)}
            if cfg.model.combat_aux:    # 战斗辅助头：输赢交叉熵、剩余掉血均方误差、平均标签（打赢比例）与平均预测，按有标签的小批平均
                na_ = max(a2[10], 1)
                stats |= {"aux_win_loss": a2[6] / na_, "aux_hp_loss": a2[7] / na_, "aux_win_y": a2[8] / na_, "aux_win_p": a2[9] / na_,
                          "aux_n": int(per_step["aux_m"][use].sum())}
            if cfg.model.recycle:       # 回灌（第 26 条）：第二遍对第一遍的 KL、首选相同的比例、rcy_ 各层权重的范数（离 0 多远）
                r_ = rcy_acc.tolist()
                stats |= {"rcy_kl": r_[0] / max(r_[2], 1), "rcy_agree": r_[1] / max(r_[2], 1),
                          "rcy_w": round(float(sum(m.weight.detach().norm() ** 2 for m in (net.rcy_a, net.rcy_b, net.rcy_c, net.rcy_p)) ** 0.5), 5)}
            if fh_opt is not None:      # 未来读数头：归一化均方误差（1 约等于只猜平均数）、标签数、各维滑动标准差、残差输出层权重的范数（离 0 多远）
                f_ = fh_acc.tolist()
                stats |= {"fh_loss": f_[0] / max(f_[1], 1), "fh_n": int(per_step["fh_m"][use].sum()), **{f"fh_sd_{n_}": x for n_, x in zip(("adv", "hp", "left", "win", "lose"), fh_sd.tolist())},
                          "fh_res_w": round(float(net.fh_res[2].weight.detach().norm()), 5), "fh_res_b": round(float(net.fh_res[2].bias.detach().abs().sum()), 5)}
            if co is not None:
                c_ = co_acc.tolist()
                stats |= {"co_hp_loss": c_[0] / max(c_[2], 1), "co_pot_loss": c_[1] / max(c_[2], 1), "co_n": int(per_step["co_m"][use].sum()),
                          "co_pot_n": int(c_[3])}
            stats["nan_skipped"] = n_skip
            if det_used:
                stats["det_steps"] = det_used       # search_collect：这一版里照部署方式出手的步数（不进策略梯度）
            if teacher_on:
                a3 = acc3.tolist()
                stats |= {"anchor_kl": a3[0] / max(a3[2], 1), "anchor_agree": a3[1] / max(a3[2], 1), "anchor_coef": anc_coef}
            if skl is not None:
                s_ = acc_skl.tolist()
                stats |= {"skl_kl": s_[0] / max(s_[2], 1), "skl_agree": s_[1] / max(s_[2], 1), "skl_n": int(s_[2]), "skl_age": s_[3] / max(s_[2], 1),
                          "skl_buf": skl.size(), "skl_read": skl.stat["read"], "skl_empty": skl.stat["empty"], "skl_coef": cfg.search_kl_coef}
                if skl.stat["err"]:
                    stats["skl_err"] = skl.stat["err"]
            stats["lr_now"] = lr_now
            if muon is not None:
                stats["muon_lr"] = muon.lr
            if value_only:
                stats["value_only"] = True
            mdiag = macro_diag(diag[:max(epochs_done, 1)].cpu().numpy())
            with torch.no_grad():
                gs, gc, gnn = gn_acc.tolist()
                health["grad_norm"] = round(gs / max(gnn, 1), 4); health["grad_clip_frac"] = round(gc / max(gnn, 1), 4)
                health["upd"] = {k: round(float(torch.sqrt(sum(((p.detach() - q) ** 2).sum() for p, q in zip(ps, h_before[k])))
                                                  / torch.sqrt(sum((q ** 2).sum() for q in h_before[k])).clamp(min=1e-12)), 6)
                                 for k, ps in hgroups.items()}
            del h_before
            t_ppo = time.time() - t_learn
            # ---- 阶段式策略梯度（决策记录 266）：存下这一版抽样的局面，每 ppg_every 版做一次辅助阶段
            ppg_stats = {}
            if cfg.ppg_every > 0:
                keep = use if cfg.ppg_keep >= 1 else np.sort(np.random.choice(use, max(1, int(len(use) * cfg.ppg_keep)), replace=False))
                item = {"data": data.select(keep), "ret": ret[keep].astype(np.float32)}
                if cfg.model.combat_aux:
                    item.update({k: per_step[k][keep] for k in ("aux_m", "aux_win", "aux_hp")})
                ppg_buf.append(item); ppg_buf = ppg_buf[-cfg.ppg_every:]
                if (version + 1) % cfg.ppg_every == 0 and len(ppg_buf) >= cfg.ppg_every:
                    ppg_stats = ppg_aux_phase(net, fwd, opt, ppg_buf, cfg, device, own_clip,
                                              hl_coef_cur if cfg.model.value_hl else 0.0, w_scalar if cfg.model.value_hl else 1.0)
                    ppg_buf = []
            # ---- 牌组打分头：每次更新从最近的标签里抽一批训练一步
            phi_stats = {}
            if phi_fresh and cfg.phi_holdout:
                phi_stats["phi_fresh"] = phi_holdout(net, phi_fresh, device)
            if cfg.phi_coef > 0 and len(phi_buf) >= cfg.phi_batch:
                pick = np.random.choice(len(phi_buf), cfg.phi_batch, replace=False)
                items = [phi_buf[j] for j in pick]
                z, y, l_phi, phi_skip = phi_update(net, opt_phi if opt_phi is not None else opt, items, cfg, device)
                phi_stats["phi_skipped"] = phi_skip
                phi_steps += 1 - phi_skip
                with torch.no_grad():
                    pred = torch.sigmoid(z).cpu().numpy(); acts_ = np.array([x["act"] for x in items]); yy = y.cpu().numpy()
                    # 各幕：[样本数, 平均预测, 平均实际]（看板上看校准）
                    phi_stats |= {"phi_loss": round(l_phi.item(), 4),
                                 "phi_calib": [[int((acts_ == a).sum()), round(float(pred[acts_ == a].mean()), 4) if (acts_ == a).any() else None,
                                                round(float(yy[acts_ == a].mean()), 4) if (acts_ == a).any() else None] for a in range(3)]}
            # ---- 胜率势能的模型：每次更新从最近的整局样本里抽一批训练一步（约 5 万参数，几毫秒）
            if wp_net is not None and len(wp_buf) >= cfg.wp_batch:
                for _ in range(cfg.wp_iters):
                    pick = np.random.choice(len(wp_buf), cfg.wp_batch, replace=False)
                    rows = [wp_buf[j] for j in pick]
                    l = wpo.train_step(wp_net, wp_opt, [(r[0], r[1]) for r in rows], np.array([r[2] for r in rows], np.float32),
                                       np.array([r[3] for r in rows], np.float32), device)
                    wp_steps += 1; wp_loss = l if wp_loss is None else 0.98 * wp_loss + 0.02 * l
                wp_rounds += 1
            # ---- 搜索蒸馏：收新标签（先记一致率），再抽一批训练一步
            distill_stats = {}
            if distill is not None:
                distill.poll(net, version, device, cfg.learner_bf16)
                distill_stats = distill.step(net, opt, device, cfg.learner_bf16) | distill.stats()
            net.eval()
            t_learn = time.time() - t_learn
            if not params_finite(net):
                # 跳过非有限梯度之后仍出现：不发布、不保存，放 STOP 让看门狗不再拉起，等人决定（不自动回退）
                (run_dir / "STOP").write_text(f"第 {version + 1} 版参数出现非有限值，训练自行停下（决策记录 173）\n", encoding="utf-8")
                raise RuntimeError(f"第 {version + 1} 版参数出现非有限值，已停下")

            # ---- 课程进度（只看战斗课程）
            # 固定的 Boss 名额、第二幕真实快照练习不随进度变，不参与进度调节，单独统计
            boss = [t for t in ended if t["mode"] == MODE_COMBAT and t.get("slot") == "boss"]
            fights = {k: [t for t in ended if t["mode"] == MODE_COMBAT and t.get("slot") == k] for k in ("act1", "act2", "act3", "mix")}
            combat = [t for t in ended if t["mode"] == MODE_COMBAT and t.get("slot") is None]
            # 整局的统计（层数、通关率、死因、Boss 过关率）只算从第一层开的局；中途开局的另记 mid_runs（2026-10-26）
            runs = [t for t in ended if t["mode"] == MODE_RUN and not t.get("start_act")]
            mids = [t for t in ended if t["mode"] == MODE_RUN and t.get("start_act")]
            win_rate = float(np.mean([t["win"] for t in combat])) if combat else 0.5
            if not combat:
                pass
            elif win_rate > cfg.progress_up_winrate: progress = min(1.0, progress + cfg.progress_step)
            elif win_rate < cfg.progress_down_winrate: progress = max(0.0, progress - cfg.progress_step)
            version += 1; total_steps += N; steps_since += N
            if cfg.model.value_hl and stats.get("loss_v_scalar"):
                r = stats["loss_v_hl"] / max(stats["loss_v_scalar"], 1e-8)
                hl_ratio = r if hl_ratio is None else 0.9 * hl_ratio + 0.1 * r
                if cfg.hl_start is not None and version >= cfg.hl_start and hl_ratio <= cfg.hl_gate:
                    net.hl_beta.fill_(min(1.0, float(net.hl_beta) + 1.0 / max(1, cfg.hl_ramp)))
                stats["hl_ratio"] = hl_ratio
            if ema is not None:
                with torch.no_grad():
                    for k, v in net.state_dict().items():
                        if k in ema:
                            ema[k].lerp_(v.detach().float(), 1.0 - cfg.ema_decay)
            publish(net, weights_path, version, progress, shaping(), pub_extra())
            if bridge is not None:
                bridge.set_version(version)

            # ---- 记录
            now = time.time(); sps = steps_since / (now - t_last); t_last, steps_since = now, 0
            def mean(xs, k):     # 缺这个字段的轨迹（如协同采集机还在跑旧代码）按 0 计，不让学习进程因此退出
                return round(float(np.mean([x.get(k, 0.0) for x in xs])), 4) if xs else None
            run_floors = [t["floor"] for t in runs]
            boss_deck = [[[0, 0] for _ in range(len(DECK_BINS) + 1)] for _ in range(3)]
            for t in boss:
                if t["real"] and not t.get("probe") and t.get("deck_n") is not None:
                    c = boss_deck[t["act"]][deck_bin(t["deck_n"])]; c[0] += int(t["win"]); c[1] += 1
            # 选牌与古神遗物的计数（每行各约 3 KB，日志里最大的两项）：累计 card_log_every 次更新才写一次（决策记录 197），看板按时间窗口合计
            card_acc["cards"] += [t.get("cards") for t in runs]; card_acc["ancient"] += [t.get("ancient") for t in runs]
            flush_cards = version % max(1, cfg.card_log_every) == 0
            run_choices = {}
            for t in runs:
                for k, c in t["choices"].items():
                    run_choices[k] = run_choices.get(k, 0) + c
            # 战斗定向探索（决策记录 238）：这一批结束的对局里各牌的改选次数，整局与战斗练习分开；入训的战斗改选步数
            cx_counts = {}
            for t in ended:
                src = "run" if t["mode"] == MODE_RUN else "practice"
                for k, c in (t.get("choices") or {}).items():
                    if k.startswith("EXPLORE:combat:") and c:
                        cx_counts[f"{src}:{k[15:]}"] = cx_counts.get(f"{src}:{k[15:]}", 0) + c
            run_deaths: dict[str, int] = {}
            run_bosses: dict[str, list[int]] = {}
            for t in runs:
                if t["death"]:
                    run_deaths[t["death"]] = run_deaths.get(t["death"], 0) + 1
                for k, w in t["bosses"]:
                    b = run_bosses.setdefault(k, [0, 0]); b[0] += int(w); b[1] += 1
            rec = {"time": round(now, 1), "version": version, "total_steps": total_steps, "sps": round(sps), "episodes": len(ended), "dropped": dropped, "stale_steps": stale, "long_kept": long_kept,
                   **({"macro_deferred": n_deferred, "macro_pending": len(macro_pending)} if cfg.macro_cross_segment else {}),
                   # 同种子成组采集（决策记录 267）：放行的组数与到齐的组数、局外步数、有同伴的步数与平均同伴数、有同伴的步上调整前后优势的均方根、
                   # 各局第一次没有同伴的局外步序号与层数（中位）、还在等的组数
                   **({"group": {"groups": grp_acc.get("groups", 0), "full": grp_acc.get("full", 0), "steps": grp_acc.get("steps", 0),
                                 "peer_steps": grp_acc.get("peer_steps", 0), "peers": round(grp_acc.get("peers", 0) / max(grp_acc.get("peer_steps", 0), 1), 3),
                                 "rms_raw": round(float(np.sqrt(grp_acc.get("sq_raw", 0.0) / max(grp_acc.get("peer_steps", 0), 1))), 4),
                                 "rms_adj": round(float(np.sqrt(grp_acc.get("sq_adj", 0.0) / max(grp_acc.get("peer_steps", 0), 1))), 4),
                                 "div_idx": float(np.median(grp_acc["div_idx"])) if grp_acc.get("div_idx") else None,
                                 "div_floor": float(np.median(grp_acc["div_floor"])) if grp_acc.get("div_floor") else None,
                                 "pending": len(group_pending), "c": round(group_c, 4),
                                 "c_hat": round(grp_acc["am"] / grp_acc["mm"], 4) if grp_acc.get("mm", 0.0) > 0 else None}} if cfg.run_group > 1 else {}),
                   "combat_episodes": len(combat), "win_rate": round(win_rate, 4) if combat else None, "return": mean(combat, "reward"),
                   "hp_delta": mean(combat, "hp_delta"), "ep_len": round(float(np.mean([t.get("run_len", t["len"]) for t in ended])), 1),
                   "boss_episodes": len(boss),
                   # 各幕 Boss 的胜场与场数（一批里样本不多，看板上累计多批再算胜率）
                   "boss_wins": [int(sum(t["win"] for t in boss if t["act"] == a)) for a in range(3)],
                   "boss_n": [int(sum(1 for t in boss if t["act"] == a)) for a in range(3)],
                   # 其中用整局真实快照的
                   "boss_real_wins": [int(sum(t["win"] for t in boss if t["act"] == a and t["real"])) for a in range(3)],
                   "boss_real_n": [int(sum(1 for t in boss if t["act"] == a and t["real"])) for a in range(3)],
                   # 各幕真实快照练习（act1/act2/act3）与随机构筑（mix）：[胜场, 场数]，普通战与精英战分开，键如 act2_monster、mix_elite
                   **{f"{k}_{r.lower()}": [int(sum(t["win"] for t in ts if t["room"] == r)), int(sum(1 for t in ts if t["room"] == r))]
                      for k, ts in fights.items() for r in ("Monster", "Elite")},
                   "run_episodes": len(runs), "run_steps": int(sum(t.get("run_len", t["len"]) for t in runs)),
                   # 中途开局：按开局的幕 {"1": [局数, 步数, 层数合计, 通关数], "2": [...]}
                   **({"mid_runs": {str(a): [len(g), int(sum(t.get("run_len", t["len"]) for t in g)), int(sum(t["floor"] for t in g)), int(sum(t["win"] for t in g))]
                                    for a in (1, 2) if (g := [t for t in mids if t["start_act"] == a])}} if mids else {}),
                   "run_floor_mean": round(float(np.mean(run_floors)), 2) if runs else None,
                   "run_floor_max": int(max(run_floors)) if runs else None,
                   "run_reach_act2": round(float(np.mean([t["final_act"] >= 1 or t["win"] for t in runs])), 4) if runs else None,
                   "run_reach_act3": round(float(np.mean([t["final_act"] >= 2 or t["win"] for t in runs])), 4) if runs else None,
                   "run_win_rate": mean(runs, "win"), "run_return": mean(runs, "reward"), "run_floors": run_floors,
                   # 构筑训练：牌组打分奖励（每局合计）、非战斗决策占学习样本的比例、打分头
                   "run_shaped": mean(runs, "shaped"), "run_hp_shaped": mean(runs, "hp_shaped"), "run_pot_wasted": mean(runs, "pot_wasted"),
                   "run_deck_shaped": mean(runs, "deck_shaped"), "run_dmg_shaped": mean(runs, "dmg_shaped"), "run_pot_cost": mean(runs, "pot_cost"), "run_skip_cost": mean(runs, "skip_cost"),
                   "run_relic_shaped": mean(runs, "relic_shaped"), "macro_frac": round(float((fresh & macro).sum() / max(N, 1)), 4),
                   # 局外行为：这一批结束的整局里，各类局外选择的次数合计（choice_key；看板上累计多批、按局数折算）
                   "run_choices": run_choices,
                   **({"combat_explore": {**cx_counts, "steps": int((fresh & forced & ~macro).sum())}} if cfg.combat_explore else {}),
                   # 古神三选一遗物（训练整局，带探索）：{遗物: [第1/2/3幕提供次数, 选中次数]}，"_SCREEN" 为各幕界面数与没选遗物的次数；看板按时间累计
                   **({"run_ancient": merge_counts(card_acc["ancient"])} if flush_cards else {}),
                   # 选牌奖励（训练整局，带探索；2026-10-29 起）：键同评测的 cards；看板“选牌对照”取最近一段合计
                   **({"run_cards": merge_counts(card_acc["cards"])} if flush_cards else {}),
                   # 这一批结束的整局：死因（局数）与各 Boss 的 [打赢场数, 场数]，键同评测的 deaths、bosses；看板上从开始记录起累计
                   "run_deaths": run_deaths, "run_bosses": run_bosses,
                   # 整局各幕、各类局外决策的入训比例，强制探索改选的一步差与所在局的结局（决策记录 168；看板上累计多批）
                   **intake_stats(run_items, runs), "lag_split": {k: {j: h.tolist() for j, h in v.items()} for k, v in lag_split.items()},
                   "rank_lag": {str(r): [int(a), int(b), round(c / max(a, 1), 2)] for r, (a, b, c) in sorted(rank_lag.items())}, "run_segments": int(sum(1 for t in run_items if t.get("segment"))),
                   # 这一批结束的整局：精英战 [场数, 用的药水瓶数]；各幕 Boss 战开打时 [场数, 手上药水瓶数合计]
                   "run_elite": [int(sum(t["elite_fights"] for t in runs)), int(sum(t["elite_pots"] for t in runs))],
                   "run_boss_pots": [[int(sum(1 for t in runs for a, _ in t["boss_pots"] if a == k)),
                                      int(sum(h for t in runs for a, h in t["boss_pots"] if a == k))] for k in range(3)],
                   "shaping": shaping(), "phi_steps": phi_steps, "phi_buf": len(phi_buf),
                   # 胜率势能（决策记录 189）：系数、在线训练步数与损失（指数平均）、样本缓冲，这一批结束的整局里势能塑形的合计
                   "wp_coef": round(wp_coef(), 4), "wp_steps": wp_steps, "wp_rounds": wp_rounds, "wp_loss": None if wp_loss is None else round(wp_loss, 4), "wp_buf": len(wp_buf),
                   "run_wp_shaped": mean(runs, "wp_shaped"), "run_pg_shaped": mean(runs, "pg_shaped"), "run_bv_shaped": mean(runs, "bv_shaped"),
                   "phi_new": [int(sum(1 for t in ended if t.get("phi_label") and t["phi_label"]["act"] == a)) for a in range(3)],
                   "boss_probe_n": [int(sum(1 for t in boss if t["act"] == a and t.get("probe"))) for a in range(3)],
                   # 真实快照（不含中途快照）的 Boss 练习按牌组张数分档的 [胜场, 场数]：[幕][档]，分档见 actor.DECK_BINS
                   "boss_deck": boss_deck,
                   # 求解器判定能赢的 Boss 练习：[胜场, 场数]，第一、二幕；标注进度：[幕][能赢, 已标注]（累计）、待标注的快照数
                   **({"boss_cert": [[int(sum(t["win"] for t in boss if t["act"] == a and t.get("cert"))),
                                      int(sum(1 for t in boss if t["act"] == a and t.get("cert")))] for a in range(3)]} if cfg.solver_share > 0 else {}),
                   # 第三幕第二个 Boss 的真实快照练习（残血打第二个）：[胜场, 场数]
                   "boss_second": [int(sum(t["win"] for t in boss if t.get("second"))), int(sum(1 for t in boss if t.get("second")))],
                   **(label_stats.update() if cfg.solver_share > 0 else {}),
                   **phi_stats, **distill_stats, **ppg_stats,
                   # 双机协同：这一版用到的步数里来自远程采集机的，以及桥的流量（上一版以来）
                   "cur": {k: [int(a), int(b), int(c), int(d), round(e, 2)] for k, (a, b, c, d, e) in sorted(cur.items())}, "macro_eff": macro_eff, "macro_diag": mdiag, "health": health,
                   **({"fight_end": fight_end} if fight_end else {}),
                   # 运气校正（铁甲战士决策记录第 20 条）：校正的步数、运气的平均与均方根、这些步上 δ 校正前后的均方根（applied 为是否用进优势）
                   **({"luck": {"n": luck[0], "mean": round(luck[1] / luck[0], 5), "rms": round((luck[2] / luck[0]) ** 0.5, 5),
                               "d_rms": round((luck[3] / luck[0]) ** 0.5, 5), "d_rms_cv": round((luck[4] / luck[0]) ** 0.5, 5),
                               "applied": bool(cfg.luck_cv_apply)}} if luck[0] else {}),     # 战斗收尾落差（按幕：[场数, 去均值前的平均]），愚者决策记录第 8 条
                   "remote_used": int(sum(int(t["fresh"].sum()) for t in trajs if t.get("remote"))), **(bridge.stats() if bridge is not None else {}),
                   "stuck": int(sum(t["stuck"] for t in ended)), "progress": round(progress, 3),
                   "lag": round(float(np.mean([version - 1 - t["vers"][-1] for t in ended])), 2),
                   "t_collect": round(t_collect, 2), "t_learn": round(t_learn, 2), "t_prep": round(t_prep, 2), "t_ppo": round(t_ppo, 2), "t_mb_wait": round(timing.get("wait", 0.0), 2), "epochs": epochs_done, **{k: round(v, 4) for k, v in stats.items()}}
            if flush_cards:
                card_acc = {"cards": [], "ancient": []}
            log.write(json.dumps(rec, ensure_ascii=False) + "\n"); log.flush()
            (run_dir / "status.json").write_text(json.dumps({
                "time": now, "started_at": started_at, "version": version, "total_steps": total_steps, "sps": round(sps),
                "progress": progress, "pid": os.getpid(), "actors": cfg.actors, "envs_per_actor": cfg.envs_per_actor,
                "learner_device": cfg.learner_device, "actor_device": cfg.actor_device, "model": cfg.to_dict()["model"]}, ensure_ascii=False), encoding="utf-8")
            if version % cfg.log_every == 0:
                print(" ".join(f"{k}={v}" for k, v in rec.items() if k not in ("run_floors", "run_choices", "run_deaths", "run_bosses", "phi_fresh", "run_ancient", "run_cards", "run_fresh_kind",
                                                                            "explore_td", "explore_runs", "lag_split", "rank_lag", "cur", "macro_eff", "macro_diag", "boss_deck", "health")), flush=True)
            if version % cfg.save_every == 0:
                save_checkpoint(ckpt, net, opt, cfg, version, progress, total_steps, phi_steps, vocab_n, ck_extra())
    finally:
        stop.set()
        if value_probe:
            value_probe.writer.close()
        sampler_stop.set()
        if feeder:
            feeder.stop.set()
        save_checkpoint(ckpt, net, opt, cfg, version, progress, total_steps, phi_steps, vocab_n, ck_extra())
        for p in actors:
            p.join(timeout=10)
            if p.is_alive(): p.kill()
        try: weights_path.unlink()
        except OSError: pass


if __name__ == "__main__":
    main()
