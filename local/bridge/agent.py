"""観測フレーム → 推論。ゲームから来た 1 フレームに 1 手を返す。

MOD が送るフレームは、無頭エンジンの serve が返す観測フレームと同じバイト列
（python/sts2env/protocol.py の parse がそのまま読める）。推論は local.infer.Policy
を共有するので、ローカル実行（#1）とゲーム内（#5）で同じネットワーク・同じ前処理になる。
"""

from __future__ import annotations

import time

from ..errors import LocalError
from ..infer import Policy, find_install
from . import protocol as P
from .autopilot import Decision


class ModelDecider:
    """チェックポイントを 1 つ持ち、フレームごとに推奨手を返す。"""

    def __init__(self, policy: Policy, top: int = 5, merge: bool = False):
        self.policy = policy
        self.top = max(1, int(top))
        self.merge = merge
        self.frames = 0
        self.seconds = 0.0
        self.character = policy.character

    @classmethod
    def load(
        cls,
        character: str,
        ckpt=None,
        device: str = "cpu",
        install=None,
        top: int = 5,
        merge: bool = False,
    ) -> ModelDecider:
        policy = Policy(character, ckpt=ckpt, device=device, install=install or find_install())
        return cls(policy, top=top, merge=merge)

    @property
    def obs_format(self) -> int:
        return self.policy.asset.obs_format

    def __call__(self, frame: P.Frame) -> Decision:
        parse = self.policy.sts2env.protocol.parse
        try:
            obs = parse(frame.frame)
        except Exception as e:
            raise LocalError(
                "観測フレームを読めませんでした。MOD 側の観測形式とネットワークの学習形式が"
                f"食い違っている可能性があります（{e}）"
            ) from e
        t0 = time.perf_counter()
        probs, _logits, value = self.policy.score(obs)
        self.seconds += time.perf_counter() - t0
        self.frames += 1

        n = obs.num_actions
        if n <= 0:
            raise LocalError("候補が 0 件のフレームが届きました（MOD 側の観測生成を確認してください）")
        best = int(max(range(n), key=lambda i: (float(probs[i]), -i)))
        if self.merge:
            best = int(self.policy.sts2rl.greedy.merged_argmax(obs, probs))
        labels = list(obs.labels) + [""] * max(0, n - len(obs.labels))
        order = sorted(range(n), key=lambda i: -float(probs[i]))[: self.top]
        return Decision(
            index=best,
            label=labels[best],
            prob=float(probs[best]),
            candidates=[[i, labels[i], round(float(probs[i]), 4)] for i in order],
            value=value,
        )


def frame_meta(obs) -> dict:
    """ログ用の短い要約（幕・層・HP・戦闘中か・候補数）。"""
    try:
        return {
            "act": int(obs.act),
            "floor": int(obs.floor),
            "hp": int(obs.hp),
            "max_hp": int(obs.max_hp),
            "in_combat": bool(obs.in_combat),
            "decision": int(obs.decision),
            "candidates": int(obs.num_actions),
            "combats_won": int(obs.combats_won),
        }
    except Exception:                               # noqa: BLE001 - 要約は無くてもよい
        return {}
