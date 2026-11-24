"""キャッシュの置き場所。すべて 1 つの根（既定 ~/.local/share/sls2）の下に置く。

  ~/.local/share/sls2/
    game/<バージョン>/        ゲームのアセンブリとローカライズ（エンジンがここを読む）
    weights/<リリース名>/      リリースから落とした .pt（checksum 検証済み）
    runs/<日時>_<キャラ>_<seed>/ 1 回の実行の出力（結果 jsonl とリプレイ）
    state/                    スタンプ（何をいつ用意したか）

根は環境変数 SLS2_HOME で変えられる。エンジン自体は既定で
~/.local/share/sls2/game/<バージョン> を見るので、SLS2_HOME を変えたときは
実行時に STS2_GAME_DIR を明示する（runner が自動で渡す）。
"""

from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ENV_HOME = "SLS2_HOME"


def home() -> Path:
    """キャッシュの根。"""
    raw = os.environ.get(ENV_HOME)
    if raw:
        return Path(raw).expanduser().resolve()
    return Path.home() / ".local" / "share" / "sls2"


def game_dir(version: str) -> Path:
    """ゲームのアセンブリとローカライズを置く場所（バージョンごと）。"""
    return home() / "game" / version


def weights_dir(release: str) -> Path:
    """リリース名（weights-v6 など）ごとの重みの置き場。"""
    return home() / "weights" / release


def runs_dir() -> Path:
    return home() / "runs"


def state_dir() -> Path:
    return home() / "state"


def state_file(name: str) -> Path:
    return state_dir() / name


def repo(*parts: str) -> Path:
    """リポジトリ内のパス。"""
    return REPO.joinpath(*parts)


def venv_python() -> Path:
    """リポジトリ直下の仮想環境の python。"""
    return REPO / ".venv" / "bin" / "python"


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p
