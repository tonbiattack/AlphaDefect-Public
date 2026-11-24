"""AI の 1 局をブラウザのリプレイで観戦する（イシュー #12）。

やること:

  1. キャラ・アセンブリ・シードを指定して 1 局打ち、リプレイ（1 手 1 行の記録）を残す
     （#1 のローカル実行と同じ経路。ability_test が `out/A<難易度>/replays/*.jsonl.gz` に書く）
  2. 既存のリプレイビューア（tools/replay_viewer.py）で、その記録からオフラインの
     再生ページを組み立てる（ネットワーク不要・リモート取得なし）
  3. Windows のブラウザで開く

作者の環境向けの決め打ち（Steam のパス、ゲーム版、中国語ローカライズ、出力先）は使わず、
すべて#1のキャッシュ（`~/.local/share/sls2`）と検出済みのゲームから引く。日本語名を優先し、
無ければ英語・ID に落とす。

    python -m local replay --character DEFECT --seed ABC123 --asc 10      # 打って観る
    python -m local replay --from ~/.local/share/sls2/runs/<...>          # 既存の記録を観る
    python -m local replay --from ... --no-open                            # 生成だけ
"""

from __future__ import annotations

import gzip
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from . import paths, runner
from . import weights as W
from .errors import LocalError
from .recipes import recipe
from .steam import GameInstall
from .versions import GameLine

# 名前の言語の優先順（要件: 日本語 → 英語 → ID）
LANGS = ("jpn", "eng")


@dataclass
class Recording:
    """1 局ぶんのリプレイ（ability_test の出力から集める）。"""

    replays: list[Path] = field(default_factory=list)
    results: list[dict] = field(default_factory=list)
    out: Path | None = None

    @property
    def ok(self) -> bool:
        return bool(self.replays)


def _load_viewer():
    """tools/replay_viewer.py をモジュールとして読む（パッケージではないので importlib）。"""
    path = paths.repo("tools", "replay_viewer.py")
    spec = importlib.util.spec_from_file_location("_replay_viewer", path)
    if spec is None or spec.loader is None:
        raise LocalError(f"リプレイビューアを読めません: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def record(
    line: GameLine,
    install: GameInstall,
    character: str,
    seed: str,
    ascension: int = 10,
    device: str = "cpu",
    out: Path | None = None,
    log=print,
) -> Recording:
    """1 局打ってリプレイを残す（ability_test を 1 局だけ走らせる）。"""
    asset = W.ASSETS[character]
    stamp = __import__("time").strftime("%Y%m%d-%H%M%S")
    out = out or (paths.runs_dir() / f"{stamp}_{character}_{seed}_replay")
    rec = recipe(line, character, obs_format=asset.obs_format)
    spec = runner.RunSpec(character=character, seeds=[seed], ascension=ascension, out=out,
                          envs=1, device=device, time_cap=rec.time_cap)
    plan = runner.plan(line, spec, rec, paths.venv_python(), W.weight_path(asset),
                       paths.game_dir(install.version))
    log(f"  1 局打ちます（{character} / A{ascension} / seed {seed}）")
    outcome = runner.execute(plan, stream=True)
    replays = sorted(plan.out.glob("**/replays/*.jsonl.gz"))
    return Recording(replays=replays, results=outcome.results, out=plan.out)


def collect(src: Path) -> list[Path]:
    """既存の出力ディレクトリ（または 1 つの .jsonl.gz）からリプレイを集める。"""
    if src.is_file():
        return [src] if src.suffix == ".gz" else []
    return sorted(src.glob("**/replays/*.jsonl.gz")) or sorted(src.glob("*.jsonl.gz"))


def build(
    records: list[Path],
    install: GameInstall,
    line: GameLine,
    out: Path,
    langs: tuple[str, ...] = LANGS,
    assets: bool = True,
    log=print,
) -> Path:
    """リプレイから再生ページを作る。戻り値は index.html。"""
    if not records:
        raise LocalError("リプレイがありません（先に 1 局打つか --from で記録を指定してください）")
    viewer = _load_viewer()
    staging = out / "records"
    staging.mkdir(parents=True, exist_ok=True)
    for p in records:                         # ビューアは 1 ディレクトリの *.jsonl.gz を読む
        shutil.copy2(p, staging / p.name)
    game_dir = paths.game_dir(install.version)
    loc_root = game_dir / "res" / "localization"
    catalog = line.root / "engine" / "catalog" / line.catalog / "catalog.json"
    vocab = line.root / "engine" / "catalog" / "vocab.txt"
    log(f"  再生ページを組み立てます（名前: {'/'.join(langs)}、{len(records)} 局）")
    page = None
    if "jpn" in langs:
        from .replay_ja import localize

        page = out / "index_ja.html"
        page.write_text(localize(paths.repo("tools", "replay_player.html").read_text(encoding="utf-8")), encoding="utf-8")
    viewer.build(
        staging,
        out,
        "",
        300,
        langs=langs,
        loc_root=loc_root,
        catalog_path=catalog,
        vocab_path=vocab,
        assets_python=paths.venv_python(),
        pck=install.pck,
        player_html=page,
        want_assets=assets,
    )
    html = out / "index.html"
    if not html.is_file():
        raise LocalError(f"再生ページができませんでした: {html}")
    return html


def windows_path(p: Path) -> str | None:
    """WSL のパスを Windows のパスに直す（/mnt/c → C:/、それ以外 → //wsl.localhost/<distro>）。"""
    s = str(p.resolve())
    if s.startswith("/mnt/") and len(s) > 6 and s[5].isalpha() and s[6] == "/":
        return s[5].upper() + ":" + s[6:]
    if os.environ.get("WSL_DISTRO_NAME") or Path("/proc/version").read_text(errors="ignore").lower().find("microsoft") >= 0:
        distro = os.environ.get("WSL_DISTRO_NAME") or "Ubuntu"
        return f"//wsl.localhost/{distro}{s}"
    return None


def windows_home() -> Path | None:
    """Windows のユーザーフォルダ（/mnt/c/Users/<name>）。ブラウザから開ける場所に出すのに使う。"""
    users = Path("/mnt/c/Users")
    if not users.is_dir():
        return None
    for d in users.iterdir():
        try:
            if (d / "AppData" / "Roaming" / "SlayTheSpire2").is_dir():
                return d
        except OSError:            # 他ユーザーのフォルダは読めないことがある
            continue
    return None


def open_in_browser(html: Path, log=print) -> bool:
    """Windows の既定のブラウザで開く（WSL から）。"""
    win = windows_path(html)
    if win is None:
        log(f"  Windows のパスにできないので、ブラウザで開いてください: {html}")
        return False
    url = ("file:" + win) if not win.startswith("//") else ("file://" + win)
    for cmd in (
        ["cmd.exe", "/c", "start", "", url],
        ["powershell.exe", "-NoProfile", "-Command", f"Start-Process '{url}'"],
    ):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
        except (OSError, subprocess.SubprocessError):
            continue
        if r.returncode == 0:
            log(f"  ブラウザで開きました: {url}")
            return True
    log(f"  ブラウザで開けませんでした。手動で開いてください: {url}")
    return False


def default_out(character: str, seed: str) -> Path:
    """Windows のユーザーフォルダ配下に出す（Windows のブラウザからそのまま開ける）。

    WSL の外から見えない場所だと `--open` が使えないので、まず Windows 側に置く。
    """
    stamp = __import__("time").strftime("%Y%m%d-%H%M%S")
    home = windows_home()
    base = (home / "alphadefect-replay") if home is not None else (paths.runs_dir() / "replay")
    return base / f"{stamp}_{character}_{seed}"


def summarize(records: list[Path], log=print) -> None:
    """何が記録されているかを短く出す（戦闘数・報酬選択の有無など）。"""
    for p in records:
        try:
            with gzip.open(p, "rt", encoding="utf-8") as f:
                head = json.loads(f.readline())
                steps = [json.loads(x) for x in f if x.strip()]
        except (OSError, ValueError):
            log(f"  {p.name}: 読めませんでした")
            continue
        kinds = {}
        for s in steps:
            kinds[s.get("d")] = kinds.get(s.get("d"), 0) + 1
        combats = len(head.get("combats") or [])
        log(f"  {p.name}: {'勝利' if head.get('win') else '敗北'} / {head.get('floor')} 層 / "
            f"{len(steps)} 手 / 戦闘 {combats} 回 / 修了 {head.get('steps')} 手")
