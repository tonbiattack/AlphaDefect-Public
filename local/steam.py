"""Steam のインストール先から Slay the Spire 2 のゲーム資産を見つけ、版を読む。

Windows の Steam を WSL から見る場合、Steam のライブラリはドライブごとに散らばる
（C:\\Program Files (x86)\\Steam と D:\\SteamLibrary など）。libraryfolders.vdf を読んで
全部のライブラリを回り、steamapps/common/Slay the Spire 2 を探す。

版は同梱の release_info.json（ゲーム自身が書く）を正とする。Steam の appmanifest は
別のアプリの中身になっている壊れた例があるので、版の判定には使わない。
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from .errors import LocalError

GAME_DIR_NAME = "Slay the Spire 2"
APP_ID = "1012880"

# Steam のインストール先として現実的な場所（WSL の /mnt/<drive> と Linux ネイティブの両方）
_WINDOWS_ROOTS = (
    "Program Files (x86)/Steam",
    "Program Files/Steam",
    "Steam",
    "SteamLibrary",
)
_LINUX_ROOTS = (
    ".steam/steam",
    ".local/share/Steam",
    ".var/app/com.valvesoftware.Steam/.local/share/Steam",
)


@dataclass(frozen=True)
class GameInstall:
    """ゲームのインストール 1 つ分。"""

    path: Path
    version: str
    branch: str
    commit: str
    date: str
    assembly_hash: int
    data_dir: Path
    pck: Path

    @property
    def assembly(self) -> Path:
        return self.data_dir / "sts2.dll"

    def to_dict(self) -> dict:
        return {
            "path": str(self.path),
            "version": self.version,
            "branch": self.branch,
            "commit": self.commit,
            "date": self.date,
            "assembly_hash": self.assembly_hash,
            "data_dir": str(self.data_dir),
            "pck": str(self.pck),
        }


# ---------------------------------------------------------------- VDF


def parse_vdf(text: str) -> dict:
    """Valve の KeyValues を読む（libraryfolders.vdf に必要な分だけ）。

    `"key" "value"` と `"key" { ... }` の入れ子だけを扱う。コメントと、
    値をクォートしない書き方（appmanifest の一部）も読めるようにしてある。
    """
    tokens = _tokenize(text)
    if not tokens:
        return {}
    if tokens[0] == "{":                     # 波括弧から始まる（名前なし）形
        value, rest = _parse_object(tokens, 0)
    else:                                    # "libraryfolders" { ... } の形
        value, rest = _parse_object(tokens, 1)
        value = {tokens[0]: value}
    if rest != len(tokens):
        raise ValueError("VDF の構文が終端で合いません")
    return value


_TOKEN_RE = re.compile(r'"((?:[^"\\]|\\.)*)"|([{}])|([^\s{}"]+)')


def _tokenize(text: str) -> list[str]:
    out: list[str] = []
    for m in _TOKEN_RE.finditer(_strip_comments(text)):
        quoted, brace, bare = m.group(1), m.group(2), m.group(3)
        if quoted is not None:
            out.append(quoted.replace("\\\\", "\\").replace('\\"', '"'))
        elif brace is not None:
            out.append(brace)
        elif bare is not None:
            out.append(bare)
    return out


def _strip_comments(text: str) -> str:
    return re.sub(r"//[^\n]*", "", text)


def _parse_object(tokens: list[str], i: int) -> tuple[dict, int]:
    out: dict = {}
    if i >= len(tokens) or tokens[i] != "{":
        raise ValueError("VDF: '{' がありません")
    i += 1
    while i < len(tokens) and tokens[i] != "}":
        key = tokens[i]
        i += 1
        if i >= len(tokens):
            raise ValueError(f"VDF: {key} の値がありません")
        if tokens[i] == "{":
            val, i = _parse_object(tokens, i)
        else:
            val, i = tokens[i], i + 1
        # 同じキーが並ぶことがある（libraryfolders の "0" "1" …）
        if key in out:
            prev = out[key]
            out[key] = prev + [val] if isinstance(prev, list) else [prev, val]
        else:
            out[key] = val
    if i >= len(tokens):
        raise ValueError("VDF: '}' がありません")
    return out, i + 1


# ---------------------------------------------------------------- パス変換


def _is_dir(p: Path) -> bool:
    """/mnt/e のように「マウントポイントはあるが中身が無い」パスで例外を出さない is_dir。"""
    try:
        return p.is_dir()
    except OSError:
        return False


def _is_file(p: Path) -> bool:
    try:
        return p.is_file()
    except OSError:
        return False


def is_wsl() -> bool:
    if os.environ.get("WSL_DISTRO_NAME"):
        return True
    try:
        return "microsoft" in Path("/proc/version").read_text(encoding="utf-8", errors="ignore").lower()
    except OSError:
        return False


def windows_to_local(path: str) -> Path:
    """'D:\\\\SteamLibrary\\\\steamapps' を WSL から見えるパスにする。

    WSL でなければそのまま（Windows ネイティブ実行を想定）。WSL では /mnt/<drive>
    に読み替える。ドライブが実際にマウントされているかまでは見ない（呼び出し側で確認）。
    """
    p = path.strip()
    if not is_wsl():
        return Path(p)
    m = re.match(r"^([A-Za-z]):[\\/](.*)$", p)
    if not m:
        return Path(p)
    drive, rest = m.group(1).lower(), m.group(2).replace("\\", "/")
    return Path(f"/mnt/{drive}") / rest


def drive_roots() -> list[Path]:
    """/mnt/c … /mnt/z のうち実在するもの（WSL）。"""
    out = []
    mnt = Path("/mnt")
    if not _is_dir(mnt):
        return out
    for d in sorted(mnt.iterdir()):
        if re.fullmatch(r"[a-z]", d.name) and _is_dir(d):
            out.append(d)
    return out


# ---------------------------------------------------------------- 探索


def steam_roots() -> list[Path]:
    """Steam のインストール先の候補（実在するものだけ）。"""
    env = os.environ.get("SLS2_STEAM_DIR")
    roots: list[Path] = []
    if env:
        roots.append(Path(env).expanduser())
    for base in drive_roots():
        for rel in _WINDOWS_ROOTS:
            roots.append(base / rel)
    home = Path.home()
    for rel in _LINUX_ROOTS:
        roots.append(home / rel)
    return [r for r in _dedupe(roots) if _is_dir(r / "steamapps")]


def _dedupe(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for p in paths:
        try:
            key = p.resolve()
        except OSError:
            key = p
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def library_paths(steam_root: Path) -> list[Path]:
    """Steam の全ライブラリ（steam_root 自身を含む）。"""
    out = [steam_root]
    vdf = steam_root / "steamapps" / "libraryfolders.vdf"
    if not _is_file(vdf):
        return out
    try:
        data = parse_vdf(vdf.read_text(encoding="utf-8", errors="replace"))
    except (OSError, ValueError):
        return out
    folders = data.get("libraryfolders", data)
    for key, val in folders.items():
        if not key.isdigit():
            continue
        entries = val if isinstance(val, list) else [val]
        for e in entries:
            raw = e.get("path") if isinstance(e, dict) else None
            if raw:
                out.append(windows_to_local(raw))
    return _dedupe(out)


def find_installs() -> list[GameInstall]:
    """見つかった Slay the Spire 2 のインストール全部（新しい順）。"""
    env = os.environ.get("SLS2_GAME_PATH")
    candidates: list[Path] = []
    if env:
        candidates.append(Path(env).expanduser())
    for root in steam_roots():
        for lib in library_paths(root):
            candidates.append(lib / "steamapps" / "common" / GAME_DIR_NAME)
    installs: list[GameInstall] = []
    for c in _dedupe(candidates):
        try:
            installs.append(read_install(c))
        except (LocalError, OSError, ValueError, json.JSONDecodeError):
            continue
    installs.sort(key=lambda i: parse_version(i.version), reverse=True)
    return installs


def explicit_install() -> GameInstall | None:
    """SLS2_GAME_PATH（--game-path）で名指しされたゲーム。名指しは自動検出より優先する。"""
    raw = os.environ.get("SLS2_GAME_PATH")
    if not raw:
        return None
    return read_install(Path(raw).expanduser())


def find_install(preferred_version: str | None = None) -> GameInstall:
    """1 つに決める。名指しがあればそれを、無ければ対応版を優先して選ぶ。"""
    explicit = explicit_install()
    if explicit is not None:
        return explicit
    installs = find_installs()
    if not installs:
        raise LocalError(_not_found_message())
    if preferred_version:
        want = preferred_version if preferred_version.startswith("v") else "v" + preferred_version
        for i in installs:
            if i.version == want:
                return i
    # 対応版があればそれを、無ければ一番新しいもの（呼び出し側が未対応として断る）
    from .versions import supported_versions

    supported = set(supported_versions())
    for i in installs:
        if i.version in supported:
            return i
    return installs[0]


def read_install(path: Path) -> GameInstall:
    p = path
    info_path = p / "release_info.json"
    data_dir = p / "data_sts2_windows_x86_64"
    pck = p / "SlayTheSpire2.pck"
    if not _is_file(info_path):
        raise LocalError(f"ゲームの目印（release_info.json）がありません: {p}")
    if not _is_file(data_dir / "sts2.dll"):
        raise LocalError(f"ゲームのアセンブリがありません: {data_dir / 'sts2.dll'}")
    if not _is_file(pck):
        raise LocalError(f"ゲームのリソース（SlayTheSpire2.pck）がありません: {pck}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    version = str(info.get("version") or "").strip()
    if not version:
        raise LocalError(f"{info_path} に version がありません")
    return GameInstall(
        path=p,
        version=version,
        branch=str(info.get("branch") or ""),
        commit=str(info.get("commit") or ""),
        date=str(info.get("date") or ""),
        assembly_hash=int(info.get("main_assembly_hash") or 0),
        data_dir=data_dir,
        pck=pck,
    )


def parse_version(v: str) -> tuple[int, int, int]:
    m = re.match(r"^v?(\d+)\.(\d+)\.(\d+)", v)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else (0, 0, 0)


def _not_found_message() -> str:
    roots = steam_roots()
    lines = [
        "Slay the Spire 2 が見つかりませんでした。",
        f"探した場所（Steam のインストール先）: {len(roots)} か所",
    ]
    for r in roots[:5]:
        lines.append(f"  - {r}")
    lines += [
        "確認すること:",
        "  1. Steam に Slay the Spire 2 をインストールしてあるか（ライブラリのフォルダごと探します）",
        "  2. WSL から Windows のドライブが見えているか（ls /mnt/c で確認）",
        "  3. 自動検出できない場所にあるなら、ゲームのフォルダを SLS2_GAME_PATH で指定する",
        f"     例: SLS2_GAME_PATH='/mnt/d/SteamLibrary/steamapps/common/{GAME_DIR_NAME}' python -m local status",
    ]
    return "\n".join(lines)


def ensure_supported(install: GameInstall):
    """未対応の版なら UnsupportedVersion を投げる（メッセージは versions 側）。"""
    from . import versions

    return versions.line_for(install.version)
