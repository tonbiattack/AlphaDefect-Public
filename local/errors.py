"""このパッケージの例外。どれも「何が悪いか」と「次に何をすればよいか」を 1 つの文にまとめて持つ。"""

from __future__ import annotations


class LocalError(RuntimeError):
    """利用者の環境では続行できない状態。CLI はこれをそのまま出して終了コード 1 で止まる。"""

    exit_code = 1


class UnsupportedVersion(LocalError):
    """インストールされているゲームのバージョンに対応するコードがこのリポジトリに無い。"""

    exit_code = 2


class UnsupportedCharacter(LocalError):
    """そのキャラは今のバージョンでは動かせない（重みが別のゲーム版向けなど）。"""

    exit_code = 2


class UnsupportedFeature(LocalError):
    """その版・その系統では提供していない機能（例: v0.111.0 系の探索あり）。"""

    exit_code = 2


class SetupRequired(LocalError):
    """`python -m local setup` を先に走らせる必要がある。"""

    exit_code = 3


class DownloadError(LocalError):
    """重みの取得に失敗した。"""


class ToolchainError(LocalError):
    """.NET 9 SDK や Python 3.12 / PyTorch が見つからず、自動導入もできなかった。"""
