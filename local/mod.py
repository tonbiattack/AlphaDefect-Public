"""ゲーム側の MOD（game-mod/）のビルドと、ゲームへの導入・撤去。

ゲームは実行ファイルの隣の `mods/` を再帰的に読み、`<id>.json`（マニフェスト）と
`<id>.dll` を見つけると起動時に読み込む（MegaCrit.Sts2.Core.Modding.ModManager）。
ここではその 2 つをビルドして `mods/AlphaDefectAutopilot/` に置く。

ゲームの版が変わったら DLL も作り直す（参照しているアセンブリが変わるため）。
スタンプに版を残し、`install` は必要ならビルドし直す。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from . import build as B
from . import paths
from .errors import LocalError
from .steam import GameInstall

MOD_ID = "AlphaDefectAutopilot"
STAMP_NAME = ".installed.json"


def source_dir() -> Path:
    return paths.repo("game-mod", MOD_ID)


def mods_dir(install: GameInstall) -> Path:
    return install.path / "mods"


def target_dir(install: GameInstall) -> Path:
    return mods_dir(install) / MOD_ID


@dataclass
class ModState:
    installed: bool
    path: Path
    built_for: str | None
    game_version: str
    stale: bool

    def to_dict(self) -> dict:
        return {
            "installed": self.installed,
            "path": str(self.path),
            "built_for": self.built_for,
            "game_version": self.game_version,
            "stale": self.stale,
        }


def status(install: GameInstall) -> ModState:
    target = target_dir(install)
    stamp = target / STAMP_NAME
    built_for = None
    if stamp.is_file():
        try:
            built_for = json.loads(stamp.read_text(encoding="utf-8")).get("game_version")
        except (OSError, ValueError):
            built_for = None
    installed = (target / f"{MOD_ID}.dll").is_file() and (target / f"{MOD_ID}.json").is_file() \
        and (target / "vocab.txt").is_file() and (target / "keys.json").is_file()
    return ModState(
        installed=installed,
        path=target,
        built_for=built_for,
        game_version=install.version,
        stale=installed and built_for != install.version,
    )


def build(install: GameInstall, log=print, force: bool = False) -> Path:
    """MOD の DLL をビルドして、出力ディレクトリを返す（増分ビルド）。"""
    src = source_dir()
    proj = src / f"{MOD_ID}.csproj"
    if not proj.is_file():
        raise LocalError(f"MOD のプロジェクトがありません: {proj}")
    dotnet = B.ensure_dotnet(log=log)
    game_dir = paths.game_dir(install.version)
    env = B.build_env(dotnet, game_dir)
    out = src / "bin" / "Release"
    if force:
        shutil.rmtree(out, ignore_errors=True)
    cmd = [
        str(dotnet), "build", str(proj), "-c", "Release", "--nologo",
        f"-p:Sts2GameDir={game_dir}",
        "-o", str(out),
    ]
    log(f"  MOD: ビルド中（ゲーム {install.version} のアセンブリを参照）")
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    if r.returncode != 0:
        raise LocalError(
            "MOD のビルドに失敗しました。\n" + "\n".join((r.stdout + r.stderr).strip().splitlines()[-25:])
        )
    dll = out / f"{MOD_ID}.dll"
    if not dll.is_file():
        raise LocalError(f"ビルドは通ったのに DLL がありません: {dll}")
    log(f"  MOD: ビルド完了（{dll}）")
    return out


def install(install: GameInstall, log=print, force: bool = False, allow_running: bool = False) -> Path:
    """ビルドしてゲームの mods/ に置く（既にあれば置き換え）。

    ゲームが起動中でも、まだ入っていない MOD を置くだけなら安全（次の起動から効く）。
    すでに入っている DLL を起動中に置き換えるのは Windows が許さないので、そのときだけ断る。
    """
    st = status(install)
    if st.installed and game_running() and not allow_running:
        raise LocalError(
            "ゲームが動いています。いま入っている MOD の DLL を置き換えるには、ゲームを終了してください。\n"
            "  （まだ入っていない MOD を置くだけなら `python -m local mod install --force` で構いません。次の起動から効きます）"
        )
    # ソースを直したら入れ直す必要があるので、スタンプだけでは判断しない。
    # 毎回ビルド（増分なので速い）し、できた DLL が入っているものと同じなら書き換えない。
    out = build(install, log=log, force=force)
    target = target_dir(install)
    target.mkdir(parents=True, exist_ok=True)
    same = False
    if (target / f"{MOD_ID}.dll").is_file():
        same = _digest(target / f"{MOD_ID}.dll") == _digest(out / f"{MOD_ID}.dll") and \
               _digest(target / f"{MOD_ID}.json") == _digest(out / f"{MOD_ID}.json")
    for name in (f"{MOD_ID}.dll", f"{MOD_ID}.json", "vocab.txt", "keys.json"):
        if not same or not (target / name).is_file():
            shutil.copy2(out / name, target / name)
    (target / STAMP_NAME).write_text(
        json.dumps({
            "mod_id": MOD_ID,
            "game_version": install.version,
            "game_commit": install.commit,
            "built_by": "local.mod",
        }, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    log(f"  MOD: {target} に{'置きました' if not same else '最新です（変更なし）'}")
    return target


def _digest(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def remove(install: GameInstall, log=print) -> bool:
    target = target_dir(install)
    if not target.is_dir():
        return False
    shutil.rmtree(target)
    log(f"  MOD: {target} を削除しました")
    # 空になった mods/ は残しても害はない（ゲームは空でも動く）
    return True


def ensure_enabled_hint(install: GameInstall) -> str:
    """ゲーム側で必要な操作の案内（初回だけ出る同意画面など）。"""
    return (
        "ゲーム側で必要な操作:\n"
        "  1. ゲームを起動する（Steam のライブラリから）\n"
        "  2. 初回は MOD の同意画面が出るので「同意する」を選ぶ\n"
        "  3. 設定 → Modding で AlphaDefect Autopilot Bridge が有効になっているか確認\n"
        "  4. 1 ランを開始する（自動プレイはモデルの観測層を載せてから）"
    )


def gameplay_note() -> str:
    """規約・互換性についての注意（イシュー #5 の確認項目）。"""
    return (
        "注意:\n"
        "  * オフラインのシングルプレイだけで使う（マルチプレイ・スコアボードに持ち込まない）\n"
        "  * ゲームの利用規約と MOD の互換性は自分で確認する（ゲーム本体は無保証）\n"
        "  * MOD を入れるとセーブの扱いが変わる（初回に未 MOD のセーブを複製する）\n"
        "  * ゲームを更新したら `python -m local mod install` で入れ直す"
    )


def game_running() -> bool:
    """ゲームのプロセスが動いているか（WSL から tasklist で見る）。"""
    if os.name == "nt":                       # Windows ネイティブ
        try:
            r = subprocess.run(["tasklist"], capture_output=True, text=True, check=False)
            return "SlayTheSpire2" in r.stdout
        except OSError:
            return False
    try:
        # Windows のコンソールは UTF-8 とは限らない（日本語環境では CP932）。読めない字は置き換える
        r = subprocess.run(["cmd.exe", "/c", "tasklist"], capture_output=True, check=False, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return False
    text = (r.stdout or b"").decode("utf-8", errors="replace") + (r.stderr or b"").decode("utf-8", errors="replace")
    return "SlayTheSpire2" in text
