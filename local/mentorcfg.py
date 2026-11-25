"""Mentor モードの起動設定（キャラクター・アセンション・シード）。

Mentor 連携では、ランの開始を spire-mentor が外部エージェントに問い合わせる
（`run_config` → `start_run`）。ここでは「どのキャラ・難易度・シードで始めるか」を
1 つの JSON（`mods/AlphaDefectAutopilot/mentor.json`）に書いて MOD に渡す。

対応キャラは `local/versions.py` を正とする。未対応のキャラや、その版で未検証の
キャラは**黙って代用せず**、はっきり断る（`--allow-untested` を付けたときだけ通す）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from . import versions
from .errors import LocalError

CONFIG_NAME = "mentor.json"

# spire-mentor の run_config が受け取る値に合わせる。
MODE_AUTO = "auto"        # 使えるキャラの先頭で始める
MODE_SELECT = "select"    # 開始しない（ゲーム内で選んで始める）


@dataclass(frozen=True)
class MentorChoice:
    character: str        # "DEFECT" / "auto" / "select"
    ascension: int
    seed: int | None      # None = 乱数
    note: str
    auto_order: tuple[str, ...] = ()   # auto のときに試す順（その版で動くキャラ）


def config_path(install) -> Path:
    return install.path / "mods" / "AlphaDefectAutopilot" / CONFIG_NAME


def _weights_for(character: str) -> Path | None:
    """そのキャラの重み（チェックポイント）が手元にあるかを探す。

    weights/MANIFEST_*.json に載っているファイル名を、ローカルの重み置き場から探す。
    見つからなければ None（推論時に取り寄せるか、失敗する）。
    """
    try:
        from . import paths

        names: set[str] = set()
        for manifest in (paths.REPO / "weights").glob("MANIFEST_*.json"):
            try:
                entries = json.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            for e in entries if isinstance(entries, list) else []:
                if not isinstance(e, dict):
                    continue
                if e.get("character") != character:
                    continue
                if e.get("kind") not in (None, "weights"):
                    continue
                name = e.get("file", "")
                if name.endswith(".pt"):
                    names.add(name)
        if not names and character == "DEFECT":
            # DEFECT は全キャラ共通のコア（student_e など）で動かす
            try:
                entries = json.loads((paths.REPO / "weights" / "MANIFEST_core.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                entries = []
            for e in entries if isinstance(entries, list) else []:
                file = e.get("file") if isinstance(e, dict) else None
                if isinstance(file, str) and file.endswith(".pt"):
                    names.add(file)
        if not names:
            return None
        root = paths.home() / "weights"
        for name in names:
            for hit in root.rglob(name):
                if hit.is_file():
                    return hit
    except (OSError, ValueError):   # 重みの仕組みが無い版でも、ここで落とさない
        return None
    return None


def available_characters(version: str) -> list[tuple[str, str]]:
    """(キャラ, 状態) の一覧。状態は validated / untested。"""
    line = versions.line_for(version)
    table = versions.VALIDATION.get(line.key, {})
    out = []
    for c in line.characters:
        state = table.get(c, "untested")
        out.append((c, state))
    return out


def resolve(version: str, character: str, ascension: int, seed: str | None,
            allow_untested: bool = False) -> MentorChoice:
    """要求された設定を検証して、MOD に渡す形に整える。"""
    line = versions.line_for(version)
    table = versions.VALIDATION.get(line.key, {})
    req = (character or "auto").strip()
    note = ""

    auto_order: tuple[str, ...] = ()
    if req.lower() in (MODE_AUTO, MODE_SELECT):
        req = req.lower()
        if req == MODE_SELECT:
            note = "ゲーム内で選んで開始する（自動開始しない）"
        else:
            auto_order = tuple(c for c, st in available_characters(version) if st == "validated")
            note = "使えるキャラの先頭（" + (", ".join(auto_order) or "なし") + "）で開始する"
    else:
        c = req.upper()
        if c not in versions.ALL_CHARACTERS:
            raise LocalError(
                f"キャラ名が分かりません: {character!r}（"
                + ", ".join(versions.ALL_CHARACTERS) + "）"
            )
        if c not in line.characters:
            raise LocalError(
                f"{version} では {c} を動かせません。使えるのは "
                + ", ".join(line.characters) + " です。"
            )
        state = table.get(c, "untested")
        if state != "validated" and not allow_untested:
            raise LocalError(
                f"{version} の {c} は未検証です（--allow-untested で強行できます）。"
            )
        if state != "validated":
            note = "未検証のキャラ（--allow-untested）"
        req = c
        if _weights_for(c) is None:
            note = (note + " / " if note else "") + "重みが見つかりません（推論時に失敗します）"

    if not 0 <= ascension <= 10:
        raise LocalError(f"アセンションは 0〜10 です: {ascension}")
    if ascension > 10:
        raise LocalError(f"アセンションが範囲外です: {ascension}")
    seed_int: int | None = None
    if seed not in (None, "", "random", "RANDOM"):
        try:
            seed_int = int(str(seed).strip())
        except ValueError as e:
            raise LocalError(
                f"seed は整数（または random）で指定してください: {seed!r}"
            ) from e
        if seed_int < 0:
            raise LocalError("seed は 0 以上の整数です。")
    return MentorChoice(character=req, ascension=ascension, seed=seed_int, note=note, auto_order=auto_order)


def write(install, choice: MentorChoice) -> Path:
    path = config_path(install)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "character": choice.character,
        "ascension": choice.ascension,
        "seed": choice.seed,
        "note": choice.note,
        "auto_order": list(choice.auto_order),
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def read(install) -> dict | None:
    path = config_path(install)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def clear(install) -> None:
    config_path(install).unlink(missing_ok=True)


def describe(install) -> str:
    """いまの設定の説明（`local mentor --status` 用）。"""
    cfg = read(install)
    if cfg is None:
        return "Mentor の開始設定: 未設定（MOD 側の既定 = DEFECT / A0 / 乱数）"
    seed = cfg.get("seed")
    seed_text = "乱数" if seed is None else str(seed)
    note = cfg.get("note") or ""
    return (f"Mentor の開始設定: {cfg.get('character')} / A{cfg.get('ascension')} / "
            f"seed={seed_text}" + (f"（{note}）" if note else ""))
