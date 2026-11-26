"""sts2-sim 对拍语料（engine/Sts2SimServe/方案.md）：真实引擎（Sts2Headless）整局贪心打，每到选路决策、选中的格子是战斗时，
先调 save 存下游戏自己的整局存档与目标格子，再记下这场战斗每一帧的原始观测与动作。只读真实引擎，不改训练。

每场战斗写一行 JSON 到 <out>/combats.jsonl：
  {"seed", "floor", "act", "room": 节点类型, "target": [列, 行], "save": 存档, "next_room_id", "frames": [base64 原始观测帧], "actions": [候选下标]}
最后一帧是这场战斗之后的第一个决策（战斗外），对拍时只比战斗中的帧。
--explore p：战斗里每个决策以概率 p 按策略分布采样（温度 --temp），走出贪心路线以外的局面；--acts 只记这几幕的战斗。
--privileged：reset 带 "privileged": true，帧的特权特征里有抽牌堆的真实顺序（观测格式.md 第三节），对拍时比两边的抽牌顺序；
只读真实引擎的隐藏信息做核对，模型照样按策略贪心选（策略不读特权特征），打出的局与不开时相同。
并行时每个进程给不同的 --start 与 --file，避免多个进程往同一个文件里交错写长行。

    .venv/bin/python -I tools/sim_corpus.py --ckpt <model.pt> --dll <Sts2Headless.dll> --seeds 8 --out /root/ext/simcorpus
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from sts2env.env import Sts2Env  # noqa: E402
from sts2env.protocol import ActionKind, DecisionKind, TokenKind, parse  # noqa: E402
from sts2rl.features import Ragged, collate_ragged  # noqa: E402
from sts2rl.model import ModelConfig, PolicyValueNet, load_state  # noqa: E402

MODE_RUN = 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dll", required=True)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--start", type=int, default=0, help="种子编号从几开始")
    ap.add_argument("--prefix", default="SIMC")
    ap.add_argument("--character", default="DEFECT", help="打哪个角色（检查点要是替这个角色训练的；储君的帧是观测格式 20）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--file", default="combats.jsonl")
    ap.add_argument("--privileged", action="store_true", help="reset 带 privileged（帧里有抽牌堆真实顺序）")
    ap.add_argument("--obs-format", type=int, default=None,
                    help="reset 带 obs_format（检查点训练时的观测格式与角色默认格式不同时用，如愚者第 24,181 版是 27、v0.114.0 主线愚者默认 29）")
    ap.add_argument("--explore", type=float, default=0.0,
                    help="战斗里每个决策以这个概率改为按策略分布采样（温度 --temp），其余贪心：走出贪心路线以外的局面（搜索改选后会走到的那类），给对拍找分歧")
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--acts", default=None, help="只记录这几幕的战斗（0 起，逗号分隔），其余照打不记；采样也只在这几幕的战斗里做")
    ap.add_argument("--override", default=None,
                    help='JSON，{"层数:战斗内第几个决策": 动作下标}：复现搜索臂改选后的路线（找搜索评测里镜像不一致的原因）')
    a = ap.parse_args()
    torch.set_num_threads(2)
    # 载入照 ability_test.load_net：内容先验按特征名对齐、检查点之后新增的词表 ID 的嵌入行清零（不清零是建网时的随机值）
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ability_test import load_net
    net, blob = load_net(a.ckpt)
    net.eval()
    # 局外帧扩展注意力（cfg.macro_attend）的网络组批要带 attend=True，否则输入喂错（ability_test.EnsembleNet.from_ragged 的说明；
    # 2026-11-25 用铁甲战士第 24,782 版打 v0.114.0 语料时没带，1 局只走到第 5 层）
    attend = bool(getattr(net, "cfg", None) and net.cfg.macro_attend)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    fh = open(out / a.file, "a", encoding="utf-8")
    env = Sts2Env(dll=Path(a.dll))
    proc = env._proc

    def request(req: dict) -> tuple[bytes, object]:
        proc.send(req); raw = proc.recv(); return raw, parse(raw)

    def greedy(o) -> int:
        with torch.no_grad():
            logits, _ = net(collate_ragged(Ragged.from_obs([o], np.array([MODE_RUN])), attend=attend).to("cpu"))
        return int(logits[0, :o.num_actions].argmax())

    def explore(o, rng) -> int:
        """按 --explore 的概率采样（温度 --temp），否则贪心。"""
        with torch.no_grad():
            logits, _ = net(collate_ragged(Ragged.from_obs([o], np.array([MODE_RUN])), attend=attend).to("cpu"))
        lg = logits[0, :o.num_actions].float()
        if o.num_actions > 1 and rng.random() < a.explore:
            p = torch.softmax(lg / a.temp, -1).numpy()
            return int(rng.choice(len(p), p=p / p.sum()))
        return int(lg.argmax())

    acts = {int(x) for x in a.acts.split(",")} if a.acts else None
    override = {k: int(v) for k, v in json.loads(a.override).items()} if a.override else {}

    for s in range(a.start, a.start + a.seeds):
        seed = f"{a.prefix}{s:04d}"
        req = {"op": "reset", "mode": "run", "character": a.character, "ascension": 10, "seed": seed, "route_summary": True, "snapshots": True}
        if a.privileged: req["privileged"] = True
        if a.obs_format is not None: req["obs_format"] = a.obs_format
        raw, o = request(req)
        n_comb, steps = 0, 0
        rng = np.random.default_rng(s * 7919 + 17)
        while not o.done and steps < 6000:
            act = greedy(o)
            if o.decision == DecisionKind.MAP:
                c = o.cand_ints[act]; node = int(c[1])
                f = o.tok_feats[node]
                proc.send({"op": "save"}); sv = json.loads(proc.recv())
                row, col = int(f[0]), int(f[1])
                m = sv["save"]["acts"][sv["save"]["current_act_index"]]["saved_map"]
                pts = list(m["points"]) + [x for x in (m.get("boss"), m.get("second_boss")) if isinstance(x, dict)]     # Boss 节点单独存
                pt = next((p for p in pts if p["coord"]["col"] == col and p["coord"]["row"] == row), None)
                ptype = (pt or {}).get("type", "?")
                raw, o = request({"op": "step", "action": act}); steps += 1
                # 真进了战斗才记：离开先古之民那一步也是地图决策，候选指向的不是地图节点，按特征读出的格子是乱的（2026-11-08，SIMRG0011、SIMDF0015）
                if ptype in ("monster", "elite", "boss") and o.in_combat and (acts is None or int(o.act) in acts):
                    frames, actions = [base64.b64encode(raw).decode()], []
                    while o.in_combat and not o.done and steps < 6000:
                        x = explore(o, rng) if a.explore > 0 else greedy(o)
                        x = override.get(f"{int(o.floor)}:{len(actions)}", x); actions.append(x)
                        raw, o = request({"op": "step", "action": x}); steps += 1
                        frames.append(base64.b64encode(raw).decode())
                    fh.write(json.dumps({"seed": seed, "floor": int(o.floor), "act": int(o.act), "room": ptype, "target": [col, row],
                                         "save": sv["save"], "next_room_id": sv["next_room_id"], "frames": frames, "actions": actions},
                                        ensure_ascii=False) + "\n"); fh.flush()
                    n_comb += 1
                continue
            raw, o = request({"op": "step", "action": act}); steps += 1
        print(f"{seed}：{'通关' if o.victory else '失败'}，第 {o.floor} 层，记下 {n_comb} 场战斗", flush=True)


if __name__ == "__main__":
    main()
