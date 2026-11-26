"""模型能力测试（2026-10-27 用户要求）：同一个检查点、纯贪心，从进阶 0 打到进阶 10，每个进阶 --runs 局，全部对局留回放。

种子 TEST0000…：各进阶用同一组种子（地图与遭遇的随机数相同，进阶不同），与训练评测的 EVAL 种子不重叠。
--seed-prefix 换一组种子（如 2026-10-28 的配对复测用 RT0000…，与 TEST、EVAL 都不重叠）。
每局一行写到 <out>/runs_s<分片>.jsonl（进阶、种子、输赢、层数、各场战斗、局外选择、选牌、古神遗物、路线，同 evaluator.play 的结果），
回放写到 <out>/A<进阶>/replays/v<版本>_<种子>.jsonl.gz（回放页同样的格式；同目录有引擎日志与超步数记录）。多进程：各进程用 --shard k/n 取一部分（进阶, 种子），已做过的跳过。

    .venv/bin/python -I tools/ability_test.py --ckpt 检查点.pt --out 目录 --shard 0/16 --envs 5 --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

import torch  # noqa: E402

from sts2rl.actor import MODE_RUN  # noqa: E402
from sts2rl.evaluator import play, run_pos  # noqa: E402
from sts2rl.catalog import Catalog  # noqa: E402
from sts2rl.model import ModelConfig, PolicyValueNet, load_state, zero_new_vocab  # noqa: E402

VOCAB_N_LEGACY = 1873      # 同 train.py：检查点开始记录词表大小（vocab_n）之前，词表都是 1,873 项


def load_net(path: str):
    """载入检查点，按训练端续训、直播端（play_live.load_model）的做法处理跨版本的输入：内容先验按特征名对齐（load_state），
    检查点之后才追加的词表 ID 的嵌入行清零（旧版就有、当时记 <unk> 的取 <unk> 那一行）。原来这里不清零，
    拿旧版本练的检查点在新版本上评测时，新卡牌、遗物的嵌入是建网时的随机值（2026-11-17，猎人第 26,428 版在 v0.113.1 上评测时发现）。"""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    net = PolicyValueNet(ModelConfig(**blob["config"]["model"])).eval()
    load_state(net, blob["state"], blob.get("content_names"))
    old_n, vocab_n = blob.get("vocab_n") or VOCAB_N_LEGACY, Catalog().vocab_size
    if old_n < vocab_n:
        with torch.no_grad():
            n_unk = zero_new_vocab(net, old_n, vocab_n)
        print(f"词表 {old_n} → {vocab_n} 项：新增 ID 的嵌入行已清零（其中 {n_unk} 个旧版记 <unk> 的取 <unk> 那一行）", flush=True)
    return net, blob


class EnsembleNet(torch.nn.Module):
    """多快照决策平均（决策记录 244）：每一步各快照分别算策略概率，取平均后当作一个网络的输出交给 play。
    返回 log(平均概率) 作 logits（不合法的候选概率为 0，log 后为 -inf），所以贪心取最大即平均概率最高的那一项，
    按温度采样、合并等价动作也都按平均概率算；价值取各快照的平均（贪心对局用不到）。"""

    def __init__(self, nets: list, method: str = "prob"):
        super().__init__()
        self.nets = torch.nn.ModuleList(nets)
        self.method = method

    def forward(self, b: dict):
        return self.combine([m(b) for m in self.nets])

    def from_ragged(self, rg, device):
        """成员结构不同（有的开局外帧扩展注意力，决策记录 262）时各自按自己的配置组批（2026-11-05 跨峰集成测试）：
        组批的默认值只跟着最后建的网络走，混用会把一部分成员喂错（第 274 条那次第 165,018 版评出 56.2% 就是这样）。"""
        from sts2rl.features import collate_ragged
        batches = {}
        outs = []
        for m in self.nets:
            att = bool(getattr(m, "cfg", None) and m.cfg.macro_attend)
            if att not in batches:
                batches[att] = collate_ragged(rg, attend=att).to(device)
            outs.append(m(batches[att]))
        return self.combine(outs)

    def combine(self, outs: list):
        """method（决策记录 272，几十个快照的集成方法比较）：prob 概率平均（原来的）；logit 对数概率平均（几何平均，重归一化）；
        vote 多数投票（各快照的首选计票，同票按平均概率分高下）；median 逐候选取概率中位数；trim 逐候选去掉最高、最低各 20% 后平均。
        都返回“分数的对数”，不合法的候选为 -inf，贪心取最大即这种方法选出的动作。"""
        v = torch.stack([v.float() for _, v in outs]).mean(0)
        lp = torch.stack([torch.log_softmax(lg.float(), -1) for lg, _ in outs])     # [K, B, A]
        p = lp.exp()
        valid = torch.isfinite(lp[0])
        if self.method == "logit":
            s = torch.where(valid, lp, 0.0).mean(0)
            return torch.where(valid, torch.log_softmax(torch.where(valid, s, -1e30), -1), float("-inf")), v
        if self.method == "vote":
            top = lp.argmax(-1)                                          # [K, B]
            cnt = torch.zeros_like(p[0]).scatter_add_(-1, top.t(), torch.ones_like(top.t(), dtype=p.dtype))
            score = cnt + 1e-3 * p.mean(0)
            return torch.where(valid, torch.log(score.clamp(min=1e-30)), float("-inf")), v
        if self.method == "median":
            score = p.median(0).values + 1e-6 * p.mean(0)
            return torch.where(valid, torch.log(score.clamp(min=1e-30)), float("-inf")), v
        if self.method == "trim":
            k = len(self.nets); cut = int(k * 0.2)
            srt = p.sort(0).values
            score = srt[cut:k - cut].mean(0) if k - 2 * cut > 0 else p.mean(0)
            return torch.where(valid, torch.log(score.clamp(min=1e-30)), float("-inf")), v
        return torch.log(p.mean(0)), v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--asc", default="0-10", help="进阶范围，如 0-10")
    ap.add_argument("--runs", type=int, default=1000)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--envs", type=int, default=5)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed-prefix", default="TEST")
    ap.add_argument("--seeds-file", default=None, help="种子池（tools/seed_pool.py，决策记录 240）：给了就打池里 seeds 的前 --runs 个，不用 --seed-prefix")
    ap.add_argument("--character", default="DEFECT", help="职业（DEFECT、IRONCLAD、SILENT、REGENT、NECROBINDER；2026-11-01 用户要看模型打其他职业的水平）")
    # 采样拆分评测（决策记录 197）：默认贪心；--temp 按温度采样，--explore 整局的局外决策照训练的 explore_probs 改选（取检查点里的配置）
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--boss-snaps", action="store_true", help="另记第一、二幕 Boss 开打前的快照（结果的 boss1、boss2 项；树搜索评测取题用，2026-11-17）")
    ap.add_argument("--explore", action="store_true")
    ap.add_argument("--merge-equiv", action="store_true", help="贪心按等价动作合并（两张一模一样的牌合计概率，决策记录 205）")
    ap.add_argument("--ensemble", default=None,
                    help="多快照决策平均（决策记录 244）：另外几个检查点，逗号分隔，与 --ckpt 一起每步平均策略概率；"
                         "结果行的 version 记 --ckpt 的版本，另记 ensemble 为全部版本")
    ap.add_argument("--ensemble-file", default=None, help="集成成员的检查点列表文件（一行一个），与 --ensemble 合并（决策记录 272）")
    ap.add_argument("--ens-method", default="prob", choices=["prob", "logit", "vote", "median", "trim"], help="集成方法（决策记录 272，见 EnsembleNet.forward）")
    ap.add_argument("--time-cap", type=float, default=600.0, help="单局时长上限（秒），超过按卡住结束；几十个快照的集成推理慢，要放宽（决策记录 272）")
    ap.add_argument("--req-extra", default=None,
                    help="并进每局 reset 请求的额外字段（JSON），如储君常规评测的截断开关 {\"loop_cut\": true, \"stall_turns\": 20}（config.cut_flags(evaluating=True)），与常规评测逐局一致")
    ap.add_argument("--policy-seed", type=int, default=None,
                    help="采样出手（--temp > 0）的随机数种子（平台期诊断 D1，2026-11-03）：不给时按分片号定，重复跑同一分片会抽到同一串随机数")
    ap.add_argument("--perturb", type=float, default=0.0,
                    help="权重扰动（诊断 D1）：每个参数张量加 正态噪声 × 该张量标准差 × 这个系数，看贪心结果对微小改动有多敏感")
    ap.add_argument("--perturb-seed", type=int, default=0)
    ap.add_argument("--bias", default=None, help="成套偏离测试（决策记录 265）：tools/defect_archetypes.json 里的流派名；整局的选卡奖励与商店买牌偏向这一流派的牌")
    ap.add_argument("--bias-delta", type=float, default=2.0, help="偏向的加分（加在对数几率上，2 约为概率乘 7.4）")
    ap.add_argument("--explore-probs", default=None,
                    help="局外强制探索的概率（JSON，如 {\"card_reward\": 0.1}；类别见 config.explore_probs），给了就不看检查点配置；"
                         "构筑价值模型的数据要见到模型平时不做的构筑（2026-11-03，决策记录 252）")
    a = ap.parse_args()
    torch.set_num_threads(2)
    net, blob = load_net(a.ckpt)
    net = net.to(a.device)
    version = int(blob["version"])
    members = [version]
    ens_paths = (a.ensemble.split(",") if a.ensemble else []) + ([l.strip() for l in open(a.ensemble_file) if l.strip()] if a.ensemble_file else [])
    ens_paths = [x for x in ens_paths if os.path.abspath(x) != os.path.abspath(a.ckpt)]     # --ckpt 自己已经是成员
    if ens_paths:
        nets = [net]
        for path in ens_paths:
            m, b2 = load_net(path)
            m = m.to(a.device)
            nets.append(m); members.append(int(b2["version"]))
        net = EnsembleNet(nets, a.ens_method).eval()
        print(f"多快照集成（{a.ens_method}）：{len(members)} 个，{members[:3]}…{members[-3:]}", flush=True)
    if a.perturb > 0:
        g = torch.Generator().manual_seed(a.perturb_seed)
        with torch.no_grad():
            for prm in net.parameters():
                if prm.numel() > 1:
                    prm.add_(torch.randn(prm.shape, generator=g).to(prm.device) * prm.float().std() * a.perturb)
        print(f"权重扰动 {a.perturb}（种子 {a.perturb_seed}）", flush=True)
    req_extra = json.loads(a.req_extra) if a.req_extra else {}
    lo, hi = map(int, a.asc.split("-"))
    k, n = map(int, a.shard.split("/"))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    res_path = out / f"runs_s{k}.jsonl"
    done = set()
    for f in out.glob("runs_s*.jsonl"):
        for l in open(f, encoding="utf-8"):
            if l.strip():
                r = json.loads(l); done.add((r["asc"], r["seed"]))
    names = (json.loads(Path(a.seeds_file).read_text(encoding="utf-8"))["seeds"][:a.runs] if a.seeds_file
             else [f"{a.seed_prefix}{j:04d}" for j in range(a.runs)])
    todo = [(asc, sd) for asc in range(lo, hi + 1) for sd in names]
    todo = [x for i, x in enumerate(todo) if i % n == k and x not in done]
    print(f"版本 {version}，本分片 {len(todo)} 局待打", flush=True)
    fh = open(res_path, "a", encoding="utf-8")
    stop = threading.Event()
    caps = {"run": 6000}
    for asc in range(lo, hi + 1):
        items = [s for x, s in todo if x == asc]
        if not items:
            continue
        rd = out / f"A{asc:02d}"; rd.mkdir(parents=True, exist_ok=True)      # play 把回放写到 rd/replays/
        it = iter(items)

        def next_task():
            # play 按完成先后返回结果，请求里带 tag（= 种子），结果里原样带回
            s = next(it, None)
            return None if s is None else ({"mode": "run", "character": a.character, "ascension": asc, "seed": s, "snapshots": True,
                                            "route_summary": True, "tag": s, **req_extra}, MODE_RUN, "run")
        t0 = time.time()

        def write(r: dict, asc=asc) -> None:      # 每局打完立即写一行（报告实时更新）
            r = {k: v for k, v in r.items() if k != "kind"}
            seed = r.pop("tag")
            fh.write(json.dumps({"asc": asc, "seed": seed, "character": a.character, "version": version, **({"ensemble": members, "ens_method": a.ens_method} if len(members) > 1 else {}),
                                 "pos": run_pos(r), **r}, ensure_ascii=False) + "\n")
            fh.flush()
        policy = {"temp": a.temp, "explore": json.loads(a.explore_probs) if a.explore_probs else (blob["config"].get("explore_probs") if a.explore else None), "seed": k * 7919 + 1 + (a.policy_seed or 0) * 100003, "boss3_snap": True, "boss_snaps": a.boss_snaps,
                  "merge": a.merge_equiv}
        if a.bias:
            arch = json.loads((Path(__file__).with_name("defect_archetypes.json")).read_text(encoding="utf-8"))[a.bias]
            policy["bias"] = {"cards": {"CARD." + c for c in arch["cards"]}, "delta": a.bias_delta}
        res = play(net, a.device, next_task, a.envs, stop, None, caps, str(rd / "loops.txt"), version, replay_all=True, on_result=write, policy=policy,
                   time_cap=a.time_cap)
        print(f"{time.strftime('%H:%M:%S')} 进阶 {asc}：{len(res)} 局，通关 {sum(r['win'] for r in res)}，用时 {time.time() - t0:.0f} 秒"
              + (f"，偏离：命中 {policy['_bias_n'][0]} 次、改了选择 {policy['_bias_n'][1]} 次" if policy.get("_bias_n") else ""), flush=True)


if __name__ == "__main__":
    main()
