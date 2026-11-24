""".NET 9 SDK の検出と導入、headless エンジンのビルド、selfcheck。

エンジンはゲームの sts2.dll を参照してビルドする（リポジトリには含まれない）。
ビルド時は -p:Sts2GameVersion と -p:Sts2GameDir でキャッシュのゲーム資産を指す。

.NET が見つからないときは ~/.dotnet に SDK 9 を入れる（dotnet-install.sh、管理者権限不要）。
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import urllib.request
from pathlib import Path

from . import paths
from .errors import ToolchainError
from .versions import GameLine

DOTNET_CHANNEL = "9.0"
_INSTALL_URL = "https://dot.net/v1/dotnet-install.sh"


def candidate_dotnets() -> list[Path]:
    out: list[Path] = []
    env = os.environ.get("SLS2_DOTNET")
    if env:
        out.append(Path(env).expanduser())
    which = shutil.which("dotnet")
    if which:
        out.append(Path(which))
    home = Path.home()
    out += [home / ".dotnet" / "dotnet", Path("/usr/share/dotnet/dotnet"), Path("/usr/lib/dotnet/dotnet"), Path("/opt/dotnet/dotnet")]
    return out


def find_dotnet() -> Path | None:
    for p in candidate_dotnets():
        if p.is_file() and os.access(p, os.X_OK):
            return p
    return None


def dotnet_version(dotnet: Path) -> str:
    """SDK の一覧（'9.0.321' などを 1 行ずつ）。"""
    try:
        r = subprocess.run([str(dotnet), "--list-sdks"], capture_output=True, text=True, timeout=120, check=False)
    except (OSError, subprocess.SubprocessError):
        return ""
    return r.stdout.strip()


def has_sdk9(dotnet: Path) -> bool:
    return any(line.startswith("9.") for line in dotnet_version(dotnet).splitlines())


def install_dotnet(log=print, install_dir: Path | None = None) -> Path:
    """dotnet-install.sh で SDK 9 を入れる（~/.dotnet、管理者権限不要）。"""
    target = install_dir or (Path.home() / ".dotnet")
    tmp = paths.ensure_dir(paths.state_dir() / "downloads") / "dotnet-install.sh"
    log(f"  .NET 9 SDK を {target} に入れます（約 220 MB のダウンロード）")
    try:
        with urllib.request.urlopen(_INSTALL_URL, timeout=60) as r:
            script = r.read()
    except OSError as e:
        raise ToolchainError(
            f".NET の導入スクリプトを取得できませんでした（{e}）。\n"
            "  手動で入れる場合: https://dotnet.microsoft.com/download/dotnet/9.0\n"
            "  または `sudo apt install dotnet-sdk-9.0`"
        ) from e
    tmp.write_bytes(script)
    cmd = ["bash", str(tmp), "--channel", DOTNET_CHANNEL, "--install-dir", str(target), "--no-path"]
    r = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if r.returncode != 0:
        raise ToolchainError(
            ".NET SDK の導入に失敗しました。\n" + (r.stdout[-600:] + r.stderr[-600:]).strip()
        )
    dotnet = target / "dotnet"
    if not dotnet.is_file():
        raise ToolchainError(f"導入後の {dotnet} が見つかりません")
    log(f"  .NET SDK: {dotnet_version(dotnet).splitlines()[-1] if dotnet_version(dotnet) else '?'}")
    return dotnet


def ensure_dotnet(install: bool = True, log=print) -> Path:
    dotnet = find_dotnet()
    if dotnet is not None and has_sdk9(dotnet):
        return dotnet
    if not install:
        raise ToolchainError(
            ".NET 9 SDK が見つかりません。\n"
            "  `python -m local setup` は自動で ~/.dotnet に入れます（管理者権限不要）。\n"
            "  手動: curl -sSL https://dot.net/v1/dotnet-install.sh | bash -s -- --channel 9.0"
        )
    dotnet = install_dotnet(log)
    if not has_sdk9(dotnet):
        raise ToolchainError(f"{dotnet} に .NET 9 SDK がありません")
    return dotnet


def dll_path(line: GameLine) -> Path:
    return line.root / "engine" / "Sts2Headless" / "bin" / "Release" / "net9.0" / "Sts2Headless.dll"


def build_env(dotnet: Path, game_dir: Path) -> dict:
    env = dict(os.environ)
    env.update(
        {
            "DOTNET_CLI_TELEMETRY_OPTOUT": "1",
            "DOTNET_NOLOGO": "1",
            "DOTNET_ROOT": str(dotnet.parent),
            "STS2_GAME_DIR": str(game_dir),
            "PATH": f"{dotnet.parent}{os.pathsep}" + env.get("PATH", ""),
        }
    )
    return env


def build(line: GameLine, version: str, game_dir: Path, dotnet: Path | None = None, log=print) -> Path:
    """headless エンジンをビルドして DLL のパスを返す（増分ビルドなので 2 回目以降は数秒）。"""
    dotnet = dotnet or ensure_dotnet(log=log)
    proj = line.root / "engine" / "Sts2Headless"
    if not (proj / "Sts2Headless.csproj").is_file():
        raise ToolchainError(f"エンジンのプロジェクトがありません: {proj}")
    cmd = [
        str(dotnet),
        "build",
        "-c",
        "Release",
        f"-p:Sts2GameVersion={version}",
        f"-p:Sts2GameDir={game_dir}",
        "--nologo",
    ]
    r = subprocess.run(cmd, cwd=str(proj), env=build_env(dotnet, game_dir), capture_output=True, text=True, check=False)
    if r.returncode != 0:
        raise ToolchainError(
            "エンジンのビルドに失敗しました。\n"
            + "\n".join((r.stdout + r.stderr).strip().splitlines()[-25:])
        )
    dll = dll_path(line)
    if not dll.is_file():
        raise ToolchainError(f"ビルドは通ったのに DLL がありません: {dll}")
    log(f"  エンジン: ビルド完了（{dll}）")
    return dll


_SELFCHECK_OK = re.compile(r"自检通过|selfcheck.*(ok|passed)", re.IGNORECASE)


def selfcheck(line: GameLine, game_dir: Path, dotnet: Path | None = None, log=print, timeout: float = 1800.0) -> tuple[bool, str]:
    """selfcheck を走らせる（ゲーム更新後に何が壊れたかを一覧する）。戻り値は (通ったか, 出力)。"""
    dotnet = dotnet or ensure_dotnet(log=log)
    dll = dll_path(line)
    if not dll.is_file():
        raise ToolchainError("エンジンがまだビルドされていません。`python -m local setup` を先に実行してください。")
    cmd = [str(dotnet), str(dll), "selfcheck"]
    try:
        r = subprocess.run(
            cmd,
            cwd=str(line.root),
            env=build_env(dotnet, game_dir),
            capture_output=True,
            text=True,
            timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return False, f"selfcheck が {timeout:.0f} 秒で終わりませんでした"
    out = (r.stdout or "") + (r.stderr or "")
    ok = r.returncode == 0 and bool(_SELFCHECK_OK.search(out))
    log("  selfcheck: 通過" if ok else f"  selfcheck: 失敗（終了コード {r.returncode}）")
    return ok, out
