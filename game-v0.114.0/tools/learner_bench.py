"""学习端离线测速（2026-10-27，决策记录 141、143）：用训练导出的一批真实学习数据（run_dir/DUMP → batch_dump.pkl），
在空闲的显卡上按训练里同样的小批 PPO 循环（前向、损失、反向、梯度裁剪、Adam）测每遍耗时，比较几种实现：

- base：现在的实现（注意力按核心数分 3–11 组、组内填充，每个小批 8 次 .item() 同步）；
- nosync：同上，统计量留在显卡上、每遍末尾才取回（去掉每个小批的主机同步）；
- packed：注意力改为“打包变长”：所有样本的记录首尾相接成一条 [总记录数, d]，逐记录的层（归一化、投影、前馈）一次算完，
  注意力用嵌套张量（jagged）的 scaled_dot_product_attention，每个样本只看自己的记录，不分组、不填充；
- compile：在 nosync 或 packed 上再套 torch.compile（Inductor 融合小算子，dynamic=True，不用 CUDA Graphs，形状每批都变）；
- isel：packed 之外把按下标取行一律改为 index_select；
- static / static+compile / static+graphs：整小批的各维度补齐到固定上限（CAPS），注意力改为按样本补齐的稠密注意力，
  两套头对全部行都算再按行选，前向里没有随数据变的形状，可用 CUDA Graphs（mode="reduce-overhead"）。需配 --full-only。

数值核对：同一批输入、同一份权重，packed 与 base 的策略对数几率、价值差多少（bf16 下应在 1e-2 量级）。
不改动训练代码，测完再决定是否把某种实现搬进 model.py / train.py。

    .venv/bin/python -I tools/learner_bench.py --dump batch_dump.pkl --ckpt latest.pt --variants base,nosync,packed,packed+compile
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from sts2rl.features import N_POOLS, collate_ragged  # noqa: E402
from sts2rl.model import ModelConfig, PolicyValueNet, load_state  # noqa: E402


def packed_encode(self, b: dict):
    """与 PolicyValueNet._encode 同样的输入输出，注意力按样本打包成变长序列（不分组、不填充）。"""
    ti = b["tok_ints"]
    kind, ids, zone, _owner, aux = ti.unbind(-1)
    vs = self.cfg.vocab_size
    tf = b["tok_feats"]
    content = self._content()
    e = (self.id_emb(ids.clamp(0, vs - 1)) + self.aux_emb(aux.clamp(0, vs - 1)) + self.kind_emb(kind.clamp(0, self.cfg.n_kinds - 1))
         + self.zone_emb(zone.clamp(0, self.cfg.n_zones - 1)) + self.feat(tf[:, :16]) + self._feat_tail(tf, kind)
         + (F.embedding(ids.clamp(0, vs - 1), content) + F.embedding(aux.clamp(0, vs - 1), content)).to(tf.dtype))
    e = e.index_add(0, b["glob_tok"], self.boss_emb(b["glob_boss"].clamp(0, vs - 1)).sum(1))
    d = e.shape[1]
    B, K = b["core_idx"].shape
    mv = self.move_out(F.gelu(self.move_in(e[b["mv_src"]]))).to(e.dtype)
    e = e.index_add(0, b["pw_dst"], e[b["pw_src"]]).index_add(0, b["mv_dst"], mv)
    sums = e.new_zeros(B * N_POOLS, d).index_add_(0, b["pool_dst"], e[b["pool_tok"]]).view(B, N_POOLS, d)
    pools = (sums / b["pool_cnt"][..., None].clamp(min=1) + self.pool_emb
             + (torch.log1p(b["pool_cnt"])[..., None] * self.pool_n).to(sums.dtype))

    # 打包：样本 i 的序列 = 它的核心记录（按 core_idx 顺序）+ N_POOLS 个汇总槽
    cm = b["core_mask"]
    n = cm.sum(1)
    L = n + N_POOLS
    offsets = torch.zeros(B + 1, dtype=torch.long, device=e.device)
    offsets[1:] = torch.cumsum(L, 0)
    rows, cols = cm.nonzero(as_tuple=True)
    core_tok = b["core_idx"][rows, cols]
    dest_core = offsets[rows] + cols
    dest_pool = (offsets[:-1] + n)[:, None] + torch.arange(N_POOLS, device=e.device)
    T = int(offsets[-1])
    x = e.new_zeros(T, d).index_copy(0, dest_core, e[core_tok]).index_copy(0, dest_pool.reshape(-1), pools.reshape(-1, d).to(e.dtype))

    H = self.cfg.heads
    dh = d // H
    for layer in self.encoder.layers:
        h = layer.norm1(x)
        qkv = F.linear(h, layer.self_attn.in_proj_weight, layer.self_attn.in_proj_bias).view(T, 3, H, dh)
        q, k, v = (torch.nested.nested_tensor_from_jagged(qkv[:, j], offsets).transpose(1, 2) for j in range(3))
        o = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).values().reshape(T, d)
        x = x + layer.self_attn.out_proj(o)
        x = x + layer.linear2(F.relu(layer.linear1(layer.norm2(x))))
    x = self.norm(x)

    seg = torch.repeat_interleave(torch.arange(B, device=e.device), L)
    g = x.new_zeros(B, d).index_add_(0, seg, x) / L[:, None].to(x.dtype)
    ctx = x.new_zeros(e.shape[0], d).index_put((core_tok,), x[dest_core])
    g = g + self.dec_emb(b["decision"].clamp(0, self.cfg.n_decisions - 1)) + self.mode_emb(b["mode"].clamp(0, self.cfg.n_modes - 1))
    return e, ctx, g


def packed_encode_isel(self, b: dict):
    """packed_encode 的变体：按下标取行一律用 index_select（反向是 index_add，不走按下标排序的 indexing_backward）。"""
    sel = torch.index_select
    ti = b["tok_ints"]
    kind, ids, zone, _owner, aux = ti.unbind(-1)
    vs = self.cfg.vocab_size
    tf = b["tok_feats"]
    content = self._content()
    e = (self.id_emb(ids.clamp(0, vs - 1)) + self.aux_emb(aux.clamp(0, vs - 1)) + self.kind_emb(kind.clamp(0, self.cfg.n_kinds - 1))
         + self.zone_emb(zone.clamp(0, self.cfg.n_zones - 1)) + self.feat(tf[:, :16]) + self._feat_tail(tf, kind)
         + (F.embedding(ids.clamp(0, vs - 1), content) + F.embedding(aux.clamp(0, vs - 1), content)).to(tf.dtype))
    e = e.index_add(0, b["glob_tok"], self.boss_emb(b["glob_boss"].clamp(0, vs - 1)).sum(1))
    d = e.shape[1]
    B, K = b["core_idx"].shape
    mv = self.move_out(F.gelu(self.move_in(sel(e, 0, b["mv_src"])))).to(e.dtype)
    e = e.index_add(0, b["pw_dst"], sel(e, 0, b["pw_src"])).index_add(0, b["mv_dst"], mv)
    sums = e.new_zeros(B * N_POOLS, d).index_add_(0, b["pool_dst"], sel(e, 0, b["pool_tok"])).view(B, N_POOLS, d)
    pools = (sums / b["pool_cnt"][..., None].clamp(min=1) + self.pool_emb
             + (torch.log1p(b["pool_cnt"])[..., None] * self.pool_n).to(sums.dtype))
    cm = b["core_mask"]
    n = cm.sum(1)
    L = n + N_POOLS
    offsets = torch.zeros(B + 1, dtype=torch.long, device=e.device)
    offsets[1:] = torch.cumsum(L, 0)
    rows, cols = cm.nonzero(as_tuple=True)
    core_tok = b["core_idx"][rows, cols]
    dest_core = offsets[rows] + cols
    dest_pool = (offsets[:-1] + n)[:, None] + torch.arange(N_POOLS, device=e.device)
    T = int(offsets[-1])
    x = e.new_zeros(T, d).index_copy(0, dest_core, sel(e, 0, core_tok)).index_copy(0, dest_pool.reshape(-1), pools.reshape(-1, d).to(e.dtype))
    H = self.cfg.heads
    dh = d // H
    for layer in self.encoder.layers:
        h = layer.norm1(x)
        qkv = F.linear(h, layer.self_attn.in_proj_weight, layer.self_attn.in_proj_bias).view(T, 3, H, dh)
        q, k, v = (torch.nested.nested_tensor_from_jagged(qkv[:, j], offsets).transpose(1, 2) for j in range(3))
        o = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).values().reshape(T, d)
        x = x + layer.self_attn.out_proj(o)
        x = x + layer.linear2(F.relu(layer.linear1(layer.norm2(x))))
    x = self.norm(x)
    seg = torch.repeat_interleave(torch.arange(B, device=e.device), L)
    g = x.new_zeros(B, d).index_add_(0, seg, x) / L[:, None].to(x.dtype)
    ctx = x.new_zeros(e.shape[0], d).index_copy(0, core_tok, sel(x, 0, dest_core))
    g = g + self.dec_emb(b["decision"].clamp(0, self.cfg.n_decisions - 1)) + self.mode_emb(b["mode"].clamp(0, self.cfg.n_modes - 1))
    return e, ctx, g


def forward_isel(self, b: dict):
    """PolicyValueNet.forward_phi 的前两项，按下标取行改用 index_select；编码用 self._encode。"""
    from sts2rl.model import _split
    sel = torch.index_select
    e, ctx, g = self._encode(b)
    B, K = b["core_idx"].shape
    vs = self.cfg.vocab_size
    ref_table = torch.cat([ctx.to(e.dtype), e], 0)
    def gather_ref(idx):
        return torch.where((idx >= 0)[:, None], sel(ref_table, 0, idx.clamp(min=0)), self.none_ref)
    ci = b["cand_ints"]
    ha, hb = gather_ref(b["cand_a"]), gather_ref(b["cand_b"])
    act = self.act_emb(ci[:, 0].clamp(0, self.cfg.n_actions - 1)) + self.cand_id_emb(ci[:, 3].clamp(0, vs - 1)) + self.cand_feat(b["cand_feats"][:, :4]) + self.cand_feat_x(b["cand_feats"][:, 4:8])
    score = _split(self.cand_mlp, self.cand_mlp_m, torch.cat([ha, hb, act, sel(g, 0, b["cand_step"])], -1), b["macro"][b["cand_step"]])
    logits = score.new_full((B, b["width"]), float("-inf")).index_put((b["cand_step"], b["cand_slot"]), score)
    logits = torch.where((~b["has_cand"])[:, None] & (torch.arange(logits.shape[1], device=e.device) == 0), 0.0, logits)
    top = e.new_zeros(B, e.shape[1]).index_add_(0, b["draw_step"], sel(e, 0, b["draw_tok"]) * b["draw_w"][:, None]) / b["draw_wsum"][:, None]
    value = _split(self.value_mlp, self.value_mlp_m, torch.cat([g, self.priv_proj(top)], -1), b["macro"])
    return logits, value


# ---- 固定形状（CUDA Graphs 用）：每个整小批（4,096 步）的各维度补齐到固定上限，多出一个“垫底行”（下标 B）和一个垫底记录（S_CAP − 1）
# 吸收补齐的部分；注意力改为按样本补齐到 L_CAP 的稠密注意力（核心记录 + 汇总槽，键掩码），两套头对全部行都算再按行选，
# 前向里没有 nonzero、.item() 这类形状随数据变的操作。上限按导出数据各维度最大值留约 3–5% 余量；超出时这一小批不用固定形状。
CAPS = dict(S=248_000, C=23_552, pw=22_528, mv=21_760, pool=151_040, draw=52_224, Tc=56_320, T=77_824, M=256, L=112)


def make_static(b: dict) -> dict | None:
    dev = b["tok_ints"].device
    B = b["core_idx"].shape[0]
    cm = b["core_mask"]
    n = cm.sum(1)
    S, C = b["tok_ints"].shape[0], b["cand_ints"].shape[0]
    rows, cols = cm.nonzero(as_tuple=True)
    Tc = len(rows)
    T = int((n + N_POOLS).sum())
    sizes = dict(S=S + 1, C=C, pw=len(b["pw_src"]), mv=len(b["mv_src"]), pool=len(b["pool_tok"]), draw=len(b["draw_tok"]),
                 Tc=Tc, T=T, M=b["width"], L=int(n.max()) + N_POOLS)
    if any(sizes[k] > CAPS[k] for k in sizes) or len(b["glob_tok"]) != B:
        return None
    SC = CAPS["S"]; dt = SC - 1                          # 垫底记录
    def pad(x, cap, val):
        return torch.cat([x, x.new_full((cap - x.shape[0],) + tuple(x.shape[1:]), val)])
    def pad_row(x, val):                                  # [B, ...] → [B + 1, ...]
        return torch.cat([x, x.new_full((1,) + tuple(x.shape[1:]), val)])
    o = {}
    o["tok_ints"] = pad(b["tok_ints"], SC, 0); o["tok_feats"] = pad(b["tok_feats"], SC, 0.0)
    o["glob_tok"] = b["glob_tok"]; o["glob_boss"] = b["glob_boss"]
    for k, cap in (("pw", CAPS["pw"]), ("mv", CAPS["mv"])):
        o[f"{k}_src"] = pad(b[f"{k}_src"], cap, dt); o[f"{k}_dst"] = pad(b[f"{k}_dst"], cap, dt)
    o["pool_tok"] = pad(b["pool_tok"], CAPS["pool"], dt); o["pool_dst"] = pad(b["pool_dst"], CAPS["pool"], B * N_POOLS)
    o["pool_cnt"] = pad_row(b["pool_cnt"], 0.0)
    # 打包布局：样本 i 的核心记录在 [off_i, off_i + n_i)，汇总槽在其后；x 多一行垃圾行（下标 T_CAP）
    TC = CAPS["T"]
    L = n + N_POOLS
    off = torch.zeros(B, dtype=torch.long, device=dev); off[1:] = torch.cumsum(L, 0)[:-1]
    o["core_tok"] = pad(b["core_idx"][rows, cols], CAPS["Tc"], dt)
    o["dest_core"] = pad(off[rows] + cols, CAPS["Tc"], TC)
    o["dest_pool"] = ((off + n)[:, None] + torch.arange(N_POOLS, device=dev)).reshape(-1)
    seg = torch.full((TC + 1,), B, dtype=torch.long, device=dev)
    seg[:T] = torch.repeat_interleave(torch.arange(B, device=dev), L)
    o["seg"] = seg
    o["Lcnt"] = pad_row(L, 1).to(torch.float32)
    LC = CAPS["L"]
    j = torch.arange(LC, device=dev)
    valid = j[None, :] < L[:, None]
    dense = torch.where(valid, off[:, None] + j[None, :], TC)
    o["dense_idx"] = pad_row(dense, TC).reshape(-1)
    km = pad_row(valid, False); km[B, 0] = True          # 垫底行至少一个有效键，免得 softmax 全是 −inf 出 NaN
    o["attn_mask"] = km[:, None, None, :]
    o["scatter_idx"] = torch.where(km, o["dense_idx"].view(B + 1, LC), TC).reshape(-1)
    o["scatter_idx"][B * LC:] = TC                        # 垫底行的输出只写到垃圾行
    # 候选
    o["cand_ints"] = pad(b["cand_ints"], CAPS["C"], 0); o["cand_feats"] = pad(b["cand_feats"], CAPS["C"], 0.0)
    o["cand_step"] = pad(b["cand_step"], CAPS["C"], B); o["cand_slot"] = pad(b["cand_slot"], CAPS["C"], 0)
    # 引用“记录自身嵌入”的下标是 S + 记录下标（ref_table = [上下文表示; 嵌入]），补齐后表长变成 S_CAP，偏移跟着改
    for k in ("cand_a", "cand_b"):
        r = b[k]
        o[k] = pad(torch.where(r >= S, r - S + SC, r), CAPS["C"], -1)
    o["has_cand"] = pad_row(b["has_cand"], False)
    o["draw_tok"] = pad(b["draw_tok"], CAPS["draw"], dt); o["draw_step"] = pad(b["draw_step"], CAPS["draw"], B)
    o["draw_w"] = pad(b["draw_w"], CAPS["draw"], 0.0); o["draw_wsum"] = pad_row(b["draw_wsum"], 1.0)
    for k in ("decision", "mode"):
        o[k] = pad_row(b[k], 0)
    o["macro"] = pad_row(b["macro"], False)
    o["B"] = B
    return o


def forward_static(self, o: dict):
    """固定形状的前向，与 forward_isel 等价（只看前 B 行）；o 来自 make_static。"""
    sel = torch.index_select
    ti = o["tok_ints"]
    kind, ids, zone, _owner, aux = ti.unbind(-1)
    vs = self.cfg.vocab_size
    tf = o["tok_feats"]
    content = self._content()
    e = (self.id_emb(ids.clamp(0, vs - 1)) + self.aux_emb(aux.clamp(0, vs - 1)) + self.kind_emb(kind.clamp(0, self.cfg.n_kinds - 1))
         + self.zone_emb(zone.clamp(0, self.cfg.n_zones - 1)) + self.feat(tf[:, :16]) + self._feat_tail(tf, kind)
         + (F.embedding(ids.clamp(0, vs - 1), content) + F.embedding(aux.clamp(0, vs - 1), content)).to(tf.dtype))
    e = e.index_add(0, o["glob_tok"], self.boss_emb(o["glob_boss"].clamp(0, vs - 1)).sum(1))
    d = e.shape[1]
    B1 = o["Lcnt"].shape[0]
    mv = self.move_out(F.gelu(self.move_in(sel(e, 0, o["mv_src"])))).to(e.dtype)
    e = e.index_add(0, o["pw_dst"], sel(e, 0, o["pw_src"])).index_add(0, o["mv_dst"], mv)
    sums = e.new_zeros(B1 * N_POOLS, d).index_add_(0, o["pool_dst"], sel(e, 0, o["pool_tok"])).view(B1, N_POOLS, d)
    cnt = o["pool_cnt"]
    pools = (sums / cnt[..., None].clamp(min=1) + self.pool_emb + (torch.log1p(cnt)[..., None] * self.pool_n).to(sums.dtype))
    TC, LC = CAPS["T"], CAPS["L"]
    x = e.new_zeros(TC + 1, d).index_copy(0, o["dest_core"], sel(e, 0, o["core_tok"]))
    x = x.index_copy(0, o["dest_pool"], pools[:-1].reshape(-1, d).to(e.dtype))
    H = self.cfg.heads; dh = d // H
    mask = o["attn_mask"]
    for layer in self.encoder.layers:
        h = layer.norm1(x)
        qkv = F.linear(h, layer.self_attn.in_proj_weight, layer.self_attn.in_proj_bias)
        qkv = sel(qkv, 0, o["dense_idx"]).view(B1, LC, 3, H, dh).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], attn_mask=mask)        # [B1, H, LC, dh]
        a = a.transpose(1, 2).reshape(B1 * LC, d)
        att = a.new_zeros(TC + 1, d).index_copy(0, o["scatter_idx"], a)
        x = x + layer.self_attn.out_proj(att)
        x = x + layer.linear2(F.relu(layer.linear1(layer.norm2(x))))
    x = self.norm(x)
    g = x.new_zeros(B1, d).index_add_(0, o["seg"], x) / o["Lcnt"][:, None].to(x.dtype)
    ctx = x.new_zeros(e.shape[0], d).index_copy(0, o["core_tok"], sel(x, 0, o["dest_core"]))
    g = g + self.dec_emb(o["decision"].clamp(0, self.cfg.n_decisions - 1)) + self.mode_emb(o["mode"].clamp(0, self.cfg.n_modes - 1))
    ref_table = torch.cat([ctx.to(e.dtype), e], 0)
    def gather_ref(idx):
        return torch.where((idx >= 0)[:, None], sel(ref_table, 0, idx.clamp(min=0)), self.none_ref)
    ci = o["cand_ints"]
    ha, hb = gather_ref(o["cand_a"]), gather_ref(o["cand_b"])
    act = self.act_emb(ci[:, 0].clamp(0, self.cfg.n_actions - 1)) + self.cand_id_emb(ci[:, 3].clamp(0, vs - 1)) + self.cand_feat(o["cand_feats"][:, :4]) + self.cand_feat_x(o["cand_feats"][:, 4:8])
    z = torch.cat([ha, hb, act, sel(g, 0, o["cand_step"])], -1)
    mc = sel(o["macro"], 0, o["cand_step"])
    score = torch.where(mc, self.cand_mlp_m(z).squeeze(-1), self.cand_mlp(z).squeeze(-1))
    logits = score.new_full((B1, CAPS["M"]), float("-inf")).index_put((o["cand_step"], o["cand_slot"]), score)
    logits = torch.where((~o["has_cand"])[:, None] & (torch.arange(CAPS["M"], device=e.device) == 0), 0.0, logits)
    top = e.new_zeros(B1, d).index_add_(0, o["draw_step"], sel(e, 0, o["draw_tok"]) * o["draw_w"][:, None]) / o["draw_wsum"][:, None]
    zv = torch.cat([g, self.priv_proj(top)], -1)
    value = torch.where(o["macro"], self.value_mlp_m(zv).squeeze(-1), self.value_mlp(zv).squeeze(-1))
    return logits[:-1], value[:-1]


class StaticNet(torch.nn.Module):
    """把 forward_static 包成模块，给 torch.compile 用（输入是张量字典）。"""
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, o):
        return forward_static(self.net, o)


def patch(net, name: str):
    if name.startswith("model"):            # 训练代码里的实现（PolicyValueNet.fast，决策记录 144）
        net.fast = True
        return net
    if "isel" in name:
        net._encode = packed_encode_isel.__get__(net)
        net.forward = forward_isel.__get__(net)
    elif "packed" in name:
        net._encode = packed_encode.__get__(net)
    return net


def load(a):
    blob = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    net = PolicyValueNet(ModelConfig(**blob["config"]["model"]))
    load_state(net, blob["state"])
    return net.to(a.device)


def make_batches(data, mb: int, device, seed: int = 0):
    perm = np.random.default_rng(seed).permutation(len(data))
    return [collate_ragged(data.select(perm[s:s + mb])).to(device) for s in range(0, len(perm), mb)]


def targets(net, batches):
    """伪造 PPO 的目标：动作取当前策略的最可能候选，旧对数概率用当前策略，优势、回报随机（只为测速，量级与真实接近）。"""
    out = []
    g = torch.Generator(device=batches[0]["tok_ints"].device).manual_seed(0)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for b in batches:
            logits, v = net(b)
            logits, v = logits.float(), v.float()
            a = logits.argmax(-1)
            olp = torch.distributions.Categorical(logits=logits).log_prob(a)
            A = torch.randn(len(a), generator=g, device=a.device)
            out.append((a, olp, A, v + 0.5 * torch.randn(len(a), generator=g, device=a.device), torch.ones_like(A)))
    return out


def one_pass(net, fwd, opt, batches, tg, sync_each: bool):
    """训练里一遍：每个小批前向、PPO 损失、反向、裁剪、更新。sync_each=True 时照训练每个小批取 8 个统计量（主机同步）。"""
    acc = torch.zeros(5, device=batches[0]["tok_ints"].device)
    for b, (a, olp, A, R, W) in zip(batches, tg):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, v = fwd(b)
        logits, v = logits.float(), v.float()
        dist = torch.distributions.Categorical(logits=logits)
        lp = dist.log_prob(a); ratio = torch.exp(lp - olp)
        l_pi = -(W * torch.min(ratio * A, ratio.clamp(0.8, 1.2) * A)).sum() / W.sum()
        l_v = F.mse_loss(v, R)
        ent_all = dist.entropy()
        loss = l_pi + 0.5 * l_v - 0.01 * ent_all.mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        with torch.no_grad():
            st = torch.stack([l_pi, l_v, ent_all.mean(), ((ratio - 1) - (lp - olp)).mean(), ((ratio - 1).abs() > 0.2).float().mean()])
            if sync_each:
                [x.item() for x in st]; ent_all[b["macro"]].mean().item(); F.mse_loss(v[b["macro"]], R[b["macro"]]).item(); b["macro"].any().item()
            else:
                acc += st
    return acc.tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mb", type=int, default=4096)
    ap.add_argument("--passes", type=int, default=4)
    ap.add_argument("--variants", default="base,nosync,packed,compile,packed+compile")
    ap.add_argument("--profile", action="store_true", help="对 base 与 packed 各剖析一遍，打印耗时最多的算子")
    ap.add_argument("--check-only", action="store_true", help="只做数值核对（关 TF32），不测速")
    ap.add_argument("--full-only", action="store_true", help="只用整小批（去掉最后不满 --mb 的那个）")
    ap.add_argument("--fresh", action="store_true", help="每遍换一组新打乱的小批（像训练里那样，每个小批的形状都没见过），检查编译是否反复重编")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = not a.check_only     # 核对时关掉 TF32，fp32 比较才有意义
    torch.backends.cudnn.allow_tf32 = not a.check_only
    data = pickle.load(open(a.dump, "rb"))
    t0 = time.time()
    batches = make_batches(data, a.mb, a.device)
    print(f"{len(data)} 步，{len(batches)} 个小批，组批 {time.time() - t0:.1f} 秒（训练里在后台线程做，与显卡计算重叠）；"
          f"每个小批平均核心记录 {np.mean([float(b['core_mask'].sum()) for b in batches]) / a.mb:.1f} 条", flush=True)

    # 数值核对：同一权重下 packed 与 base 的输出
    ref = load(a); ref.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        b = batches[0]
        l0, v0 = ref(b)
        ref._encode = packed_encode.__get__(ref)
        l1, v1 = ref(b)
    fin = torch.isfinite(l0)
    print(f"数值核对（bf16）：对数几率最大差 {(l0[fin] - l1[fin]).abs().max().item():.4f}，价值最大差 {(v0 - v1).abs().max().item():.4f}，"
          f"首选动作一致 {(l0.argmax(-1) == l1.argmax(-1)).float().mean().item():.4%}", flush=True)
    o = make_static(batches[0])
    if o is not None:
        r2 = load(a); r2.eval(); r3 = load(a); r3.eval()
        with torch.no_grad():
            l0, v0 = r2(batches[0]); l1, v1 = forward_static(r3, o)
        Mw = l0.shape[1]
        fin = torch.isfinite(l0)
        print(f"数值核对（fp32，static）：对数几率最大差 {(l0[fin] - l1[:, :Mw][fin]).abs().max().item():.2e}，价值最大差 {(v0 - v1).abs().max().item():.2e}，"
              f"补齐列全为 −inf {bool(torch.isinf(l1[:, Mw:]).all())}", flush=True)
        del r2, r3
    else:
        print("数值核对（static）：第一个小批超出上限", flush=True)
    for nm in ("packed", "isel", "isel+compile", "model", "model+compile"):
        r2 = load(a); r2.eval(); r3 = patch(load(a), nm); r3.eval()
        f3 = torch.compile(r3, dynamic=True) if "compile" in nm else r3
        with torch.no_grad():
            l0, v0 = r2(batches[0]); l1, v1 = f3(batches[0])
        fin = torch.isfinite(l0)
        print(f"数值核对（fp32，{nm}）：对数几率最大差 {(l0[fin] - l1[fin]).abs().max().item():.2e}，价值最大差 {(v0 - v1).abs().max().item():.2e}", flush=True)
        del r2, r3
    del ref
    if a.check_only:
        return

    base_net = load(a)
    tg = targets(base_net, batches)
    del base_net
    res = {}
    full = [i for i, b in enumerate(batches) if b["core_idx"].shape[0] == a.mb]
    if a.full_only:                                   # 只用整小批（固定形状要求行数固定），各变体同样处理，比较公平
        batches = [batches[i] for i in full]; tg = [tg[i] for i in full]
    statics = None
    for name in a.variants.split(","):
        net = load(a); net.train()
        if name.startswith("static"):
            if statics is None:
                statics = [make_static(b) for b in batches]
                print(f"固定形状：{sum(x is not None for x in statics)}/{len(statics)} 个小批在上限内", flush=True)
            sn = StaticNet(net)
            mode = "reduce-overhead" if "graphs" in name else "default"
            fs = torch.compile(sn, dynamic=False, mode=mode) if ("compile" in name or "graphs" in name) else sn
            fd = torch.compile(patch(net, "isel"), dynamic=True) if ("compile" in name or "graphs" in name) else patch(net, "isel")
            def fwd(b, fs=fs, fd=fd):
                o = b.get("_static")
                return fs(o) if o is not None else fd(b)
            for b, o in zip(batches, statics):
                b["_static"] = o
        else:
            for b in batches:
                b.pop("_static", None)
            patch(net, name)
            fwd = torch.compile(net, dynamic=True) if "compile" in name else net
        opt = torch.optim.Adam(net.parameters(), lr=1e-5, eps=1e-5, fused=True)
        sync_each = name == "base"
        torch.cuda.synchronize(); t0 = time.time()
        one_pass(net, fwd, opt, batches, tg, sync_each)          # 预热（编译在这一遍里）
        torch.cuda.synchronize(); warm = time.time() - t0
        ts = []
        from torch._dynamo.utils import counters
        for k_ in range(a.passes):
            bb, tt = batches, tg
            if a.fresh:
                bb = make_batches(data, a.mb, a.device, seed=100 + k_)
                tt = targets(net, bb)
            torch.cuda.synchronize(); t0 = time.time()
            one_pass(net, fwd, opt, bb, tt, sync_each)
            torch.cuda.synchronize(); ts.append(time.time() - t0)
        if "compile" in name:
            print(f"  编译统计：{dict(counters['stats'])}，重编原因数 {sum(counters['recompiles'].values()) if 'recompiles' in counters else '—'}", flush=True)
        res[name] = float(np.median(ts))
        mem = torch.cuda.max_memory_allocated() / 2**30
        print(f"{name:16s} 每遍 {res[name]:.3f} 秒（{a.passes} 遍中位数，预热 {warm:.1f} 秒），峰值显存 {mem:.1f} GB，"
              f"相对 base {res.get('base', res[name]) / res[name]:.2f} 倍", flush=True)
        torch.cuda.reset_peak_memory_stats()
        if a.profile and name in ("base", "packed"):
            from torch.profiler import ProfilerActivity, profile
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                one_pass(net, fwd, opt, batches[:6], tg[:6], sync_each); torch.cuda.synchronize()
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=18), flush=True)
        del net, fwd, opt
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
