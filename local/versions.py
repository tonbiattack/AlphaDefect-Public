"""ゲームのバージョンと、そのバージョンを動かすコードの対応表。

このリポジトリは 2 つのゲーム版を別々のコードツリーで持っている（README「Repository layout」）。

  v0.111.0 系 … game-v0.111.0/  … 故障机器人（Defect）、储君（Regent）、静默猎手（Silent）
  v0.113.1 系 … リポジトリ直下   … 愚者（Fool）、铁甲战士（Ironclad）、亡灵契约师（Necrobinder）
                                    と、論文 3.2 節・付録 E で v0.113.1 上でも評価した Defect と Silent

同じキャラでも、どの版で動かすかで使うツリーが変わる。ここでは「インストールされて
いる版 → 使うツリー・動かせるキャラ・探索の可否」を 1 か所にまとめ、未対応の版は
黙って別のツリーで走らせず、はっきり断る。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .errors import UnsupportedVersion

ALL_CHARACTERS = ("DEFECT", "REGENT", "SILENT", "FOOL", "IRONCLAD", "NECROBINDER")

_VERSION_RE = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)(?:[-+].*)?$")


@dataclass(frozen=True)
class GameLine:
    """1 つのゲーム版に対応するコードツリーと、その版で動かせるキャラ。"""

    key: str                      # 代表バージョン（このリポジトリが明示的に持っている版）
    tree: str                     # リポジトリからの相対パス。"" は直下
    catalog: str                  # engine/catalog/<これ>/catalog.json
    characters: tuple[str, ...]   # selfcheck・評価の対象になっているキャラ
    search: bool                  # ローカルで探索（探索あり）を回せるか
    search_tool: str              # 探索を回すスクリプト（search が False なら空）
    greedy_tool: str              # 貪欲に回すスクリプト
    note: str

    @property
    def root(self):

        from .paths import REPO

        return REPO if not self.tree else REPO / self.tree


LINES: dict[str, GameLine] = {
    "v0.111.0": GameLine(
        key="v0.111.0",
        tree="game-v0.111.0",
        catalog="v0.111.0",
        characters=("DEFECT", "REGENT", "SILENT"),
        search=False,
        search_tool="",
        greedy_tool="tools/ability_test.py",
        note="論文の表 3・表 4 の 3 キャラ。探索あり（PIMC）は外部の sts2-sim を別途ビルドする必要があるため、このローカル実行系では扱わない。",
    ),
    "v0.114.0": GameLine(
        key="v0.114.0",
        tree="game-v0.114.0",
        catalog="v0.114.0",
        characters=("FOOL", "IRONCLAD", "NECROBINDER", "DEFECT", "SILENT", "REGENT"),
        search=False,
        search_tool="",
        greedy_tool="tools/ability_test.py",
        note=(
            "v0.114.0（2026-11-25 の公開ベータ）。愚者のカード改訂（OPTIMISM 追加、TWIN_SHOT が RAPID_SHOT の別名）と"
            "ダメージ計算の変更に、v0.113.1 のエンジンを適応したもの。語彙は 2,557→2,559（旧 ID は不変）。"
            "v0.114.0 で学習した公開モデルはまだ無いため、既存チェックポイントを特徴量の再整列つきで使う（未検証扱い）。"
            "sts2-sim が未対応のため探索（PIMC）は使えない。"
        ),
    ),
    "v0.113.1": GameLine(
        key="v0.113.1",
        tree="",
        catalog="v0.113.1",
        characters=("FOOL", "IRONCLAD", "NECROBINDER", "DEFECT", "SILENT", "REGENT"),
        search=True,
        search_tool="tools/future/run_search.py",
        greedy_tool="tools/ability_test.py",
        note="愚者・铁甲战士・亡灵契约师の系統。探索ありは tools/future/run_search.py の情報集約型木探索（BOSS・エリート戦）。",
    ),
}

# 明示的な対応表には無いが、同じ系統のコードで動くことを実機で確かめたパッチ版。
# v0.111.1: selfcheck が「兼容层 145 处 / 找不到 0 处」「冒烟 5 キャラ×12 局で異常 0」、
#           語彙の差分 0 件（2026-11-24、このリポジトリの local 実行系で確認）。
PATCH_COMPAT: dict[str, str] = {
    "v0.111.1": "v0.111.0",
}

# キャラごとに「その系統で検証済みか」。untested は --allow-untested を付けたときだけ動かす。
# 根拠: 論文 表 3・表 4（v0.111.0 の 3 キャラ）、3.2 節（v0.113.1 上の Defect）、付録 E（v0.113.1 上の Silent）。
VALIDATION: dict[str, dict[str, str]] = {
    "v0.111.0": {"DEFECT": "validated", "REGENT": "validated", "SILENT": "validated"},
    "v0.114.0": {
        "FOOL": "untested",
        "IRONCLAD": "untested",
        "NECROBINDER": "untested",
        "DEFECT": "untested",
        "SILENT": "untested",
        "REGENT": "untested",
    },
    "v0.113.1": {
        "FOOL": "validated",
        "IRONCLAD": "validated",
        "NECROBINDER": "validated",
        "DEFECT": "validated",
        "SILENT": "validated",
        "REGENT": "untested",
    },
}


def normalize(version: str) -> str:
    """'0.111.1' / 'v0.111.1' / 'v0.111.1-rc1' を 'v0.111.1' に揃える。"""
    m = _VERSION_RE.match(version.strip())
    if not m:
        raise UnsupportedVersion(f"ゲームのバージョン表記を解釈できません: {version!r}")
    return "v{}.{}.{}".format(*m.groups())


def parse(version: str) -> tuple[int, int, int]:
    m = _VERSION_RE.match(normalize(version))
    assert m is not None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def supported_versions() -> tuple[str, ...]:
    """このリポジトリがコードを持っている版（パッチ互換を含む）。"""
    return tuple(LINES) + tuple(PATCH_COMPAT)


def line_for(version: str) -> GameLine:
    """インストールされている版に対応するコード系統を返す。未対応なら UnsupportedVersion。"""
    v = normalize(version)
    if v in LINES:
        return LINES[v]
    if v in PATCH_COMPAT:
        base = LINES[PATCH_COMPAT[v]]
        return GameLine(
            key=base.key,
            tree=base.tree,
            catalog=base.catalog,
            characters=base.characters,
            search=base.search,
            search_tool=base.search_tool,
            greedy_tool=base.greedy_tool,
            note=base.note + f"（{v} は {base.key} のパッチ版。実機で selfcheck と 1 局完走を確認済み）",
        )
    raise UnsupportedVersion(_unsupported_message(v))


def is_patch_alias(version: str) -> bool:
    return normalize(version) in PATCH_COMPAT


def base_version(version: str) -> str:
    v = normalize(version)
    return PATCH_COMPAT.get(v, v)


def _unsupported_message(v: str) -> str:
    rows = []
    for key, line in LINES.items():
        rows.append(f"  {key}: {'/'.join(line.characters)}")
    patches = "、".join(PATCH_COMPAT) or "なし"
    return (
        f"ゲーム {v} にはまだ対応していません。\n"
        "このリポジトリがコードを持っている版:\n"
        + "\n".join(rows)
        + f"\nパッチ互換として動作確認済み: {patches}\n"
        "Steam の「プロパティ → ベータ」で v0.113.1 か v0.111.0（public-beta）に切り替えてから、もう一度実行してください。\n"
        "どの版が入っているかは `python -m local status` で確認できます。"
    )


def character_status(line: GameLine, character: str) -> str:
    """'validated' / 'untested' / 'unsupported'。"""
    c = character.upper()
    if c not in line.characters:
        return "unsupported"
    return VALIDATION.get(line.key, {}).get(c, "untested")


def require_character(line: GameLine, character: str, allow_untested: bool = False) -> None:
    from .errors import UnsupportedCharacter

    c = character.upper()
    st = character_status(line, c)
    if st == "validated":
        return
    if st == "untested":
        if allow_untested:
            return
        raise UnsupportedCharacter(
            f"{c} は {line.key} では評価実績がありません（重みは別の版向けに学習されています）。"
            "自己責任で試すなら --allow-untested を付けてください。"
        )
    supported = "/".join(line.characters)
    raise UnsupportedCharacter(
        f"{c} はこのゲーム版（{line.key}）では動かせません。この版で動かせるキャラ: {supported}。"
        "別のキャラを動かすには Steam のベータで対応する版に切り替えてください。"
    )


@dataclass(frozen=True)
class VersionReport:
    version: str
    line: GameLine
    patch_alias: bool
    characters: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "line": self.line.key,
            "tree": self.line.tree or ".",
            "patch_alias": self.patch_alias,
            "search": self.line.search,
            "characters": self.characters,
        }


def report(version: str) -> VersionReport:
    line = line_for(version)
    v = normalize(version)
    return VersionReport(
        version=v,
        line=line,
        patch_alias=is_patch_alias(v),
        characters={c: character_status(line, c) for c in line.characters},
    )
