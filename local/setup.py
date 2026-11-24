"""`python -m local setup` の中身: 足りないものを順に用意する。

  1. ゲームのインストールを見つけ、版を読む（未対応の版はここで断る）
  2. Python 3.12 の仮想環境（.venv）と numpy / PyTorch(CPU) を用意する
  3. .NET 9 SDK を用意する（無ければ ~/.dotnet に入れる）
  4. ゲームのアセンブリとローカライズをキャッシュに複製する
  5. headless エンジンをビルドする
  6. selfcheck（ゲームとエンジンの噛み合わせ）を走らせる
  7. 6 キャラ分の重みを取得し SHA-256 を検証する

何度実行しても、済んでいる工程は飛ばす。`--force` で作り直す。
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from . import build as B
from . import gamefiles, paths
from . import weights as W
from .errors import LocalError, SetupRequired, ToolchainError
from .steam import GameInstall, find_install
from .versions import GameLine, VersionReport, line_for, report

_UV_URL = "https://astral.sh/uv/install.sh"
_CPU_TORCH_INDEX = "https://download.pytorch.org/whl/cpu"


# ---------------------------------------------------------------- 状態


@dataclass
class State:
    install: GameInstall | None = None
    line: GameLine | None = None
    report: VersionReport | None = None
    dotnet: Path | None = None
    dotnet_sdk9: bool = False
    venv_python: Path | None = None
    venv_ok: bool = False
    torch_ok: bool = False
    torch_version: str = ""
    game_ready: bool = False
    engine_built: bool = False
    weights: dict = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)

    @property
    def usable_weights(self) -> dict:
        """この版で実際に使うキャラの重み（他の版向けの分は含めない）。"""
        chars = self.line.characters if self.line else tuple(self.weights)
        return {c: w for c, w in self.weights.items() if c in chars}

    @property
    def ready(self) -> bool:
        return bool(
            self.install
            and self.game_ready
            and self.engine_built
            and self.venv_ok
            and self.torch_ok
            and all(w.get("present") and w.get("size_ok", True) for w in self.usable_weights.values())
        )

    def to_dict(self) -> dict:
        return {
            "install": self.install.to_dict() if self.install else None,
            "version_report": self.report.to_dict() if self.report else None,
            "dotnet": str(self.dotnet) if self.dotnet else None,
            "dotnet_sdk9": self.dotnet_sdk9,
            "venv_python": str(self.venv_python) if self.venv_python else None,
            "venv_ok": self.venv_ok,
            "torch": self.torch_version,
            "game_ready": self.game_ready,
            "engine_built": self.engine_built,
            "ready": self.ready,
            "problems": self.problems,
            "weights": self.weights,
            "home": str(paths.home()),
        }


def inspect(install: GameInstall | None = None) -> State:
    """何も変更せずに現状をまとめる（`status` 用）。"""
    st = State()
    try:
        st.install = install or find_install()
    except LocalError as e:
        st.problems.append(str(e))
        return st
    try:
        st.line = line_for(st.install.version)
        st.report = report(st.install.version)
    except LocalError as e:
        st.problems.append(str(e))
        return st
    st.game_ready = gamefiles.is_prepared(st.install)
    st.engine_built = B.dll_path(st.line).is_file()
    st.dotnet = B.find_dotnet()
    st.dotnet_sdk9 = bool(st.dotnet and B.has_sdk9(st.dotnet))
    st.venv_python = paths.venv_python()
    st.venv_ok = st.venv_python.is_file()
    st.torch_ok, st.torch_version = _torch_probe(st.venv_python) if st.venv_ok else (False, "")
    st.weights = W.status()             # 6 キャラ全部の状態を出す（使えない版向けの分も「未取得」として見せる）
    return st


def _torch_probe(python: Path) -> tuple[bool, str]:
    try:
        r = subprocess.run([str(python), "-c", "import torch, numpy; print(torch.__version__)"],
                           capture_output=True, text=True, timeout=180, check=False)
    except (OSError, subprocess.SubprocessError):
        return False, ""
    return (r.returncode == 0, r.stdout.strip())


# ---------------------------------------------------------------- 実行


def setup(
    install: GameInstall | None = None,
    characters: list[str] | None = None,
    force: bool = False,
    skip_selfcheck: bool = False,
    bootstrap: bool = True,
    log=print,
) -> State:
    """足りないものを用意する（冪等）。"""
    install = install or find_install()
    line = line_for(install.version)          # 未対応版はここで UnsupportedVersion
    log(f"ゲーム: {install.path}")
    log(f"  バージョン {install.version}（ブランチ {install.branch or '?'}, コミット {install.commit or '?'}）")
    log(f"  コード系統: {line.key}（{line.tree or 'リポジトリ直下'}）")
    # 既定は 6 キャラ全部。今の版で動かせない分も、ベータを切り替えたときにそのまま使える
    chars = [c.upper() for c in (characters or W.ASSETS)]
    for c in chars:
        if c not in W.ASSETS:
            raise LocalError(f"未知のキャラ: {c}（{', '.join(W.ASSETS)}）")

    paths.ensure_dir(paths.home())
    paths.ensure_dir(paths.state_dir())

    # 1. Python 環境
    venv_python = ensure_venv(bootstrap=bootstrap, force=force, log=log)
    ensure_python_deps(venv_python, bootstrap=bootstrap, log=log)

    # 2. .NET
    dotnet = B.ensure_dotnet(install=bootstrap, log=log)
    log(f"  .NET: {dotnet}")

    # 3. ゲーム資産
    gamefiles.prepare(install, line.root, force=force, log=log)

    # 4. エンジン
    B.build(line, install.version, paths.game_dir(install.version), dotnet=dotnet, log=log)

    # 5. selfcheck
    if skip_selfcheck:
        log("  selfcheck: 省略（--skip-selfcheck）")
    else:
        ok, out = B.selfcheck(line, paths.game_dir(install.version), dotnet=dotnet, log=log)
        if not ok:
            tail = "\n".join(out.strip().splitlines()[-20:])
            raise LocalError(
                "selfcheck が通りませんでした。エンジンとゲーム資産の噛み合わせが合っていません。\n"
                f"{tail}\n"
                "  ゲームの版が対応表（python -m local versions）にあるか確認してください。"
            )

    # 6. 重み
    W.ensure_all(chars, force=force, log=log)

    st = inspect(install)
    log("")
    if st.ready:
        log("準備完了。`python -m local play --character "
            + (chars[0] if chars else "DEFECT") + " --seed ABC123 --asc 10` で 1 局打てます。")
    else:
        log("準備は終わりましたが、未完了の項目があります: " + "; ".join(st.problems or ["status を確認してください"]))
    return st


# ---------------------------------------------------------------- Python 環境


def ensure_venv(bootstrap: bool = True, force: bool = False, log=print) -> Path:
    python = paths.venv_python()
    if python.is_file() and not force:
        return python
    if force:
        shutil.rmtree(paths.REPO / ".venv", ignore_errors=True)
    uv = find_uv()
    if uv is None:
        if not bootstrap:
            raise ToolchainError(_venv_help())
        uv = install_uv(log)
    interp = _python3_12()
    log(f"  Python 環境: {paths.REPO / '.venv'} を作成（{interp}）")
    r = subprocess.run([str(uv), "venv", "--python", interp, str(paths.REPO / ".venv")],
                       capture_output=True, text=True, check=False)
    if r.returncode != 0:
        raise ToolchainError("仮想環境の作成に失敗しました。\n" + (r.stdout + r.stderr).strip()[-800:])
    if not python.is_file():
        raise ToolchainError(f"仮想環境に python がありません: {python}\n" + _venv_help())
    return python


def _python3_12() -> str:
    for cand in (os.environ.get("SLS2_PYTHON"), "python3.12", "python3"):
        if not cand:
            continue
        p = shutil.which(cand) or (cand if Path(cand).is_file() else None)
        if not p:
            continue
        r = subprocess.run([str(p), "-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
                           capture_output=True, text=True, check=False)
        if r.returncode == 0 and r.stdout.strip() == "3.12":
            return str(p)
    raise ToolchainError(
        "Python 3.12 が見つかりません。\n"
        "  Ubuntu/WSL: sudo apt install python3.12\n"
        "  または SLS2_PYTHON で python3.12 の場所を指定してください。"
    )


def find_uv() -> Path | None:
    for cand in (os.environ.get("SLS2_UV"), shutil.which("uv"), str(Path.home() / ".local" / "bin" / "uv"),
                 str(Path.home() / ".cargo" / "bin" / "uv")):
        if cand and Path(cand).is_file():
            return Path(cand)
    return None


def install_uv(log=print) -> Path:
    """uv を入れる（venv モジュールの無いディストリビューションでも仮想環境を作れる）。"""
    import urllib.request

    log("  uv を入れます（~/.local/bin、管理者権限不要）")
    tmp = paths.ensure_dir(paths.state_dir() / "downloads") / "uv-install.sh"
    try:
        with urllib.request.urlopen(_UV_URL, timeout=60) as r:
            tmp.write_bytes(r.read())
    except OSError as e:
        raise ToolchainError(f"uv の導入スクリプトを取得できませんでした（{e}）。\n  手動: curl -LsSf https://astral.sh/uv/install.sh | sh") from e
    r = subprocess.run(["bash", str(tmp)], capture_output=True, text=True, check=False)
    uv = find_uv()
    if r.returncode != 0 or uv is None:
        raise ToolchainError("uv の導入に失敗しました。\n" + (r.stdout[-500:] + r.stderr[-500:]).strip())
    log(f"  uv: {uv}")
    return uv


def ensure_python_deps(python: Path, bootstrap: bool = True, log=print) -> None:
    ok, ver = _torch_probe(python)
    if ok:
        log(f"  PyTorch: {ver}（導入済み）")
        return
    if not bootstrap:
        raise ToolchainError(
            "PyTorch が入っていません。\n"
            f"  {python} -m pip install torch --index-url {_CPU_TORCH_INDEX}\n"
            f"  {python} -m pip install numpy"
        )
    uv = find_uv()
    log("  PyTorch(CPU) と numpy を入れます（数百 MB のダウンロード）")
    if uv is not None:
        cmd_np = [str(uv), "pip", "install", "--python", str(python), "numpy"]
        cmd_torch = [str(uv), "pip", "install", "--python", str(python), "torch", "--index-url", _CPU_TORCH_INDEX]
    else:
        cmd_np = [str(python), "-m", "pip", "install", "numpy"]
        cmd_torch = [str(python), "-m", "pip", "install", "torch", "--index-url", _CPU_TORCH_INDEX]
    for cmd in (cmd_np, cmd_torch):
        r = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if r.returncode != 0:
            raise ToolchainError(
                "Python の依存を入れられませんでした。\n"
                + (r.stdout[-800:] + r.stderr[-800:]).strip()
            )
    ok, ver = _torch_probe(python)
    if not ok:
        raise ToolchainError("PyTorch を入れたのに import できません。`python -m local setup` をやり直してください。")
    log(f"  PyTorch: {ver}")


def _venv_help() -> str:
    return (
        "Python 3.12 の仮想環境を作れませんでした。\n"
        "  Debian/Ubuntu で venv が無い場合: sudo apt install python3.12-venv\n"
        "  または uv を入れる: curl -LsSf https://astral.sh/uv/install.sh | sh\n"
        "  そのあと `python -m local setup` をやり直してください。"
    )


def require_ready(install: GameInstall | None = None):
    """play / api の前に、準備が済んでいるか確かめる。"""
    st = inspect(install)
    if st.install is None:
        raise LocalError("\n".join(st.problems) or "ゲームが見つかりません")
    from .versions import line_for

    line_for(st.install.version)          # 未対応の版はここで UnsupportedVersion（準備不足と混同しない）
    if not st.ready:
        missing = []
        if not st.venv_ok:
            missing.append("Python 仮想環境")
        if not st.torch_ok:
            missing.append("PyTorch")
        if not st.game_ready:
            missing.append("ゲーム資産の複製")
        if not st.engine_built:
            missing.append("エンジンのビルド")
        if not all(w.get("present") for w in st.usable_weights.values()):
            missing.append("重み")
        raise SetupRequired("準備が済んでいません: " + "、".join(missing) + "\n  `python -m local setup` を実行してください。")
    return st
