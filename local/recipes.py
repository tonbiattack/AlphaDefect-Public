"""キャラごとの「論文と同じ条件で 1 局打つ」ための設定。

観測形式は学習時のものと実行時の既定値がずれることがある（例: 铁甲战士の既定は
フォーマット 29 だが、重み v24,782 は 28 で学習している）。ずれたまま動かすと
ネットワークに違う並びの数字が入るので、必ず reset の req-extra で明示する。

数値の根拠:
  - README「Reproducing the tables」の評価コマンド（--req-extra）
  - weights/MANIFEST_*.json の obs_format
  - python/sts2env/protocol.py と game-v0.111.0/... の FORMAT_BY_CHARACTER
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .versions import GameLine


@dataclass(frozen=True)
class Recipe:
    """1 キャラ・1 系統ぶんの実行条件。"""

    character: str
    obs_format: int
    extra: dict = field(default_factory=dict)   # reset に必ず足すキー
    time_cap: float = 600.0                     # 1 局の時間上限（秒）
    note: str = ""


# 実行時の既定の観測形式（ツリーごと）。ここに無いキャラは 19（Defect）。
_DEFAULT_FORMAT = {
    "v0.111.0": {"DEFECT": 19, "REGENT": 20, "SILENT": 23},
    "v0.113.1": {"DEFECT": 19, "REGENT": 20, "SILENT": 23, "FOOL": 27, "IRONCLAD": 29, "NECROBINDER": 28},
}

# 論文のコマンドが明示している追加キー（README の Ironclad 行）。
_EXTRA = {
    "IRONCLAD": {"loop_cut": True, "stall_turns": 20},
}


def default_format(line: GameLine, character: str) -> int:
    return _DEFAULT_FORMAT.get(line.key, {}).get(character.upper(), 19)


def recipe(line: GameLine, character: str, obs_format: int | None = None, extra: dict | None = None) -> Recipe:
    """キャラの実行条件を組み立てる。

    obs_format は重みの学習時の形式（weights.ASSETS）。既定値と違うときだけ reset に渡す。
    """
    c = character.upper()
    fmt = obs_format if obs_format is not None else default_format(line, c)
    payload: dict = dict(_EXTRA.get(c, {}))
    if fmt != default_format(line, c):
        payload["obs_format"] = fmt
    if extra:
        payload.update(extra)
    return Recipe(character=c, obs_format=fmt, extra=payload, time_cap=600.0)


def req_extra_json(recipe_: Recipe) -> str | None:
    """ability_test / run_search の --req-extra に渡す JSON（空なら None）。"""
    import json

    return json.dumps(recipe_.extra, ensure_ascii=False) if recipe_.extra else None
