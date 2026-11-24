"""ゲーム資産をキャッシュに用意する。

やること 2 つ:

  1. `data_sts2_windows_x86_64/` の .NET アセンブリ一式を ~/.local/share/sls2/game/<版>/ に複製する
     （エンジンはここを STS2_GAME_DIR として読む。ゲーム本体のフォルダは触らない）
  2. ローカライズ（カード名・遺物名など）を SlayTheSpire2.pck から取り出す

どちらも一度やれば済む。何をいつ用意したかをスタンプに残し、版・コミット・pck の
サイズと更新時刻が変わっていなければやり直さない（`--force` で強制）。
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from . import paths
from .errors import LocalError
from .steam import GameInstall

# ローカライズとして取り出すもの（README と同じ）
LOC_PREFIXES = ("localization/eng/", "localization/zhs/", "localization/completion.json")
STAMP_NAME = ".local-stamp.json"


@dataclass(frozen=True)
class Prepared:
    path: Path
    changed: bool
    copied_files: int
    extracted_files: int
    seconds: float

    def to_dict(self) -> dict:
        return {
            "path": str(self.path),
            "changed": self.changed,
            "copied_files": self.copied_files,
            "extracted_files": self.extracted_files,
            "seconds": round(self.seconds, 1),
        }


def stamp_for(install: GameInstall) -> dict:
    stat = install.pck.stat()
    return {
        "version": install.version,
        "commit": install.commit,
        "assembly_hash": install.assembly_hash,
        "pck_size": stat.st_size,
        "pck_mtime": int(stat.st_mtime),
        "prepared_by": "local.gamefiles",
    }


def is_prepared(install: GameInstall) -> bool:
    target = paths.game_dir(install.version)
    stamp_path = target / STAMP_NAME
    if not stamp_path.is_file():
        return False
    if not (target / "sts2.dll").is_file():
        return False
    eng = target / "res" / "localization" / "eng"
    if not eng.is_dir() or not any(eng.iterdir()):
        return False
    try:
        return json.loads(stamp_path.read_text(encoding="utf-8")) == stamp_for(install)
    except (OSError, ValueError):
        return False


def prepare(install: GameInstall, tree_root: Path, force: bool = False, log=print) -> Prepared:
    """ゲーム資産をキャッシュに用意する（済んでいれば何もしない）。"""
    target = paths.game_dir(install.version)
    t0 = time.time()
    if not force and is_prepared(install):
        log(f"  ゲーム資産: 準備済み（{target}）")
        return Prepared(target, False, 0, 0, time.time() - t0)

    paths.ensure_dir(target)
    log(f"  ゲーム資産: {install.path} → {target}")
    copied = _copy_assemblies(install.data_dir, target, log)
    extracted = _extract_localization(tree_root, install.pck, target / "res", log)
    (target / STAMP_NAME).write_text(json.dumps(stamp_for(install), ensure_ascii=False, indent=1), encoding="utf-8")
    return Prepared(target, True, copied, extracted, time.time() - t0)


def _copy_assemblies(src: Path, dst: Path, log) -> int:
    files = [p for p in src.rglob("*") if p.is_file()]
    rsync = shutil.which("rsync")
    if rsync:
        # rsync があると 100 MB の複製が数秒で済む
        cmd = [rsync, "-a", str(src) + "/", str(dst) + "/"]
        r = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if r.returncode == 0:
            log(f"  ゲーム資産: アセンブリ {len(files)} ファイルを複製（rsync）")
            return len(files)
        log(f"  ゲーム資産: rsync に失敗したので通常のコピーに切り替えます（{r.stderr.strip()[:200]}）")
    n = 0
    for p in files:
        rel = p.relative_to(src)
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, out)
        n += 1
    log(f"  ゲーム資産: アセンブリ {n} ファイルを複製")
    return n


def _extract_localization(tree_root: Path, pck: Path, out_dir: Path, log) -> int:
    script = tree_root / "tools" / "extract_pck.py"
    if not script.is_file():
        raise LocalError(f"ローカライズ抽出スクリプトがありません: {script}")
    if (out_dir / "localization" / "eng").is_dir():
        # スタンプだけ書き直す（版が同じで中身が揃っている）
        return _count_files(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-I", str(script), str(pck), str(out_dir), *LOC_PREFIXES]
    r = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if r.returncode != 0:
        raise LocalError(
            "ローカライズの抽出に失敗しました。\n"
            f"  コマンド: {' '.join(cmd)}\n"
            f"  {r.stderr.strip()[-800:]}"
        )
    n = _count_files(out_dir)
    log(f"  ゲーム資産: ローカライズ {n} ファイルを抽出")
    return n


def _count_files(root: Path) -> int:
    return sum(1 for p in root.rglob("*") if p.is_file())
