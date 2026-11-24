"""6 キャラ分の重みを GitHub Releases から取得し、SHA-256 を検証してキャッシュする。

重みはリポジトリに入っていない（合計 2.5 GB）。リリース資産としてだけ配られていて、
どのファイルがどの版・どのキャラ・どの観測形式かは weights/MANIFEST_*.json に書いてある。
ここではその目録と SHA256SUMS_*.txt を読み、足りないファイルだけ落として検証する。

ダウンロードは認証不要（公開リポジトリのリリース資産）。途中で切れても .part から再開する。
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from . import paths
from .errors import DownloadError, LocalError

DEFAULT_REPO = "hmty-CHEN/AlphaDefect-Public"


@dataclass(frozen=True)
class Asset:
    """1 キャラ分の採用ネットワーク。"""

    character: str
    release: str        # リリース名（weights-v6 など）
    file: str           # リリース資産のファイル名
    version: int        # 学習の版（ログ用）
    obs_format: int     # 学習した観測形式（実行時に reset へ渡す）
    note: str

    @property
    def key(self) -> str:
        return f"{self.release}/{self.file}"


# 論文の表 3 が使っているネットワーク（README「Reproducing the tables」）。
# v0.111.0 系: Defect = student e、Regent = student e8、Silent = EMA v26,428
# v0.113.1 系: Fool = EMA v24,181、Ironclad = EMA v24,782、Necrobinder = raw v25,875
ASSETS: dict[str, Asset] = {
    "DEFECT": Asset("DEFECT", "weights-v1", "student_e_v165018.pt", 165018, 19,
                    "16 スナップショットの確率平均を蒸留した生徒 e（v0.111.0 系）"),
    "REGENT": Asset("REGENT", "weights-v2", "regent_student_e8.pt", 54603, 20,
                    "储君の蒸留モデル student e8（v0.111.0 系）"),
    "SILENT": Asset("SILENT", "weights-v4", "silent_ema_v26428.pt", 26428, 23,
                    "静默猎手 EMA 重み v26,428（v0.111.0 系）"),
    "FOOL": Asset("FOOL", "weights-v5", "fool_ema_v24181.pt", 24181, 27,
                  "愚者のライブ配信用 EMA 重み v24,181（v0.113.1 系）"),
    "IRONCLAD": Asset("IRONCLAD", "weights-v6", "ironclad_ema_v24782.pt", 24782, 28,
                      "铁甲战士 EMA 重み v24,782（v0.113.1 系）"),
    "NECROBINDER": Asset("NECROBINDER", "weights-v6", "necrobinder_raw_v25875.pt", 25875, 28,
                         "亡灵契约师 raw 重み v25,875（v0.113.1 系）"),
}


def repo_slug() -> str:
    """リリースを配っているリポジトリ（フォークやミラーを使うときは SLS2_RELEASE_REPO）。"""
    return os.environ.get("SLS2_RELEASE_REPO", DEFAULT_REPO)


def release_url(release: str, file: str) -> str:
    return f"https://github.com/{repo_slug()}/releases/download/{release}/{file}"


# ---------------------------------------------------------------- 目録


def load_expected() -> dict[str, dict]:
    """リポジトリ内の目録から {ファイル名: {sha256, bytes, obs_format, character, release}} を作る。

    weights/MANIFEST_*.json を主に、SHA-256 が無い項目は SHA256SUMS_*.txt で補う。
    目録とチェックサムが食い違っていたら、その場で失敗する（検証の意味が無くなるため）。
    """
    expected: dict[str, dict] = {}
    for p in sorted(paths.repo("weights").glob("MANIFEST_*.json")):
        tag = _release_of_manifest(p.name)
        for entry in _iter_manifest(p):
            name = entry.get("file")
            if not name:
                continue
            rec = expected.setdefault(name, {})
            rec.update({k: entry[k] for k in ("sha256", "bytes", "obs_format", "character") if entry.get(k) is not None})
            rec.setdefault("release", tag)
    for p in sorted(paths.repo("weights").glob("SHA256SUMS_*.txt")):
        tag = _release_of_sums(p.name)
        for name, digest in _iter_sums(p):
            rec = expected.setdefault(name, {})
            if rec.get("sha256") and rec["sha256"] != digest:
                raise LocalError(f"目録の食い違い: {name} の SHA-256 が MANIFEST と SHA256SUMS で違います")
            rec.setdefault("sha256", digest)
            rec.setdefault("release", tag)
    return expected


def _release_of_manifest(name: str) -> str:
    body = name.removeprefix("MANIFEST_").removesuffix(".json")
    return {"core": "weights-v1", "teacher_pool": "teacher-pool-v1"}.get(body, body)


def _release_of_sums(name: str) -> str:
    body = name.removeprefix("SHA256SUMS_").removesuffix(".txt")
    return {"core": "weights-v1", "teacher_pool": "teacher-pool-v1"}.get(body, body)


def _iter_manifest(p: Path):
    data = json.loads(p.read_text(encoding="utf-8"))
    yield from (data if isinstance(data, list) else [data])


def _iter_sums(p: Path):
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2 and len(parts[0]) == 64:
            yield parts[-1].lstrip("*"), parts[0].lower()


# ---------------------------------------------------------------- 検証


def sha256(path: Path, chunk: int = 4 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


@dataclass(frozen=True)
class VerifyResult:
    path: Path
    ok: bool
    expected: str
    actual: str
    size_ok: bool

    def to_dict(self) -> dict:
        return {"path": str(self.path), "ok": self.ok, "expected": self.expected, "actual": self.actual, "size_ok": self.size_ok}


def expected_for(asset: Asset, expected: dict | None = None) -> dict:
    table = expected if expected is not None else load_expected()
    rec = table.get(asset.file)
    if not rec or not rec.get("sha256"):
        raise LocalError(
            f"{asset.file} の SHA-256 がリポジトリの目録にありません（weights/MANIFEST_*.json か SHA256SUMS_*.txt を確認）。"
        )
    return rec


def verify(asset: Asset, path: Path | None = None, expected: dict | None = None) -> VerifyResult:
    rec = expected_for(asset, expected)
    p = path or weight_path(asset)
    if not p.is_file():
        raise LocalError(f"重みがありません: {p}（`python -m local setup` で取得できます）")
    size_ok = rec.get("bytes") in (None, p.stat().st_size)
    actual = sha256(p)
    return VerifyResult(p, actual == rec["sha256"], rec["sha256"], actual, bool(size_ok))


# ---------------------------------------------------------------- 取得


def weight_path(asset: Asset) -> Path:
    return paths.weights_dir(asset.release) / asset.file


def ensure(asset: Asset, force: bool = False, expected: dict | None = None, log=print) -> Path:
    """重みを 1 つ用意する（検証済みなら何もしない）。"""
    rec = expected_for(asset, expected)
    dest = weight_path(asset)
    if dest.is_file() and not force:
        if dest.stat().st_size == rec.get("bytes", dest.stat().st_size) and sha256(dest) == rec["sha256"]:
            log(f"  {asset.character}: 取得済み（{dest.name}）")
            return dest
        log(f"  {asset.character}: 検証に失敗したので取り直します（{dest.name}）")
        dest.unlink()
    dest.parent.mkdir(parents=True, exist_ok=True)
    _download(asset, dest, rec, log)
    got = sha256(dest)
    if got != rec["sha256"]:
        dest.unlink(missing_ok=True)
        raise DownloadError(
            f"{asset.file} の SHA-256 が一致しません（期待 {rec['sha256'][:16]}… / 実際 {got[:16]}…）。\n"
            "  配信元が変わったか、ダウンロードが壊れています。`python -m local setup` をやり直してください。"
        )
    log(f"  {asset.character}: {dest.name} を取得し SHA-256 を確認しました")
    return dest


def _download(asset: Asset, dest: Path, rec: dict, log) -> None:
    url = release_url(asset.release, asset.file)
    part = dest.with_suffix(dest.suffix + ".part")
    total = rec.get("bytes")
    have = part.stat().st_size if part.is_file() else 0
    if total and have > total:
        part.unlink()
        have = 0
    headers = {"User-Agent": "AlphaDefect-local/1.0", "Accept": "application/octet-stream"}
    if have:
        headers["Range"] = f"bytes={have}-"
    log(f"  {asset.character}: ダウンロード中 {url}"
        + (f"（{have / 1e6:.0f} MB から再開）" if have else f"（{total / 1e6:.0f} MB）" if total else ""))
    req = urllib.request.Request(url, headers=headers)
    t0 = time.time()
    tty = sys.stdout.isatty()
    next_pct = 10
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            mode = "ab" if have and resp.status == 206 else "wb"
            if mode == "wb":
                have = 0
            with open(part, mode) as f:
                done = have
                last = 0.0
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    now = time.time()
                    pct_num = done * 100 / total if total else 0.0
                    if tty and now - last > 2.0:
                        last = now
                        pct = f"{pct_num:.0f}%" if total else f"{done / 1e6:.0f} MB"
                        rate = done / max(now - t0, 1e-6) / 1e6
                        print(f"\r    {pct}（{rate:.1f} MB/s）", end="", file=sys.stdout, flush=True)
                    elif not tty and total and pct_num >= next_pct:
                        rate = done / max(now - t0, 1e-6) / 1e6
                        print(f"    {int(pct_num)}%（{rate:.1f} MB/s）", flush=True)
                        while pct_num >= next_pct:
                            next_pct += 10
    except urllib.error.HTTPError as e:
        raise DownloadError(
            f"重みを取得できませんでした（HTTP {e.code}）: {url}\n"
            "  リリース名かファイル名が変わっていないか確認してください（SLS2_RELEASE_REPO で配信元を変えられます）。"
        ) from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise DownloadError(
            f"重みのダウンロードが途中で止まりました: {e}\n"
            "  もう一度 `python -m local setup` を実行すると .part から再開します。"
        ) from e
    print(flush=True)
    os.replace(part, dest)


def ensure_all(characters, force: bool = False, log=print) -> dict[str, Path]:
    """指定キャラの重みをまとめて用意する。"""
    expected = load_expected()
    out: dict[str, Path] = {}
    for c in characters:
        asset = ASSETS[c.upper()]
        out[c.upper()] = ensure(asset, force=force, expected=expected, log=log)
    return out


def status(characters=None) -> dict[str, dict]:
    """各キャラの取得・検証の状態（何もダウンロードしない）。"""
    expected = load_expected()
    out: dict[str, dict] = {}
    for c in characters or ASSETS:
        asset = ASSETS[c.upper()]
        p = weight_path(asset)
        rec = expected.get(asset.file, {})
        row = {
            "release": asset.release,
            "file": asset.file,
            "path": str(p),
            "present": p.is_file(),
            "obs_format": asset.obs_format,
            "version": asset.version,
        }
        if p.is_file():
            row["bytes"] = p.stat().st_size
            row["size_ok"] = rec.get("bytes") in (None, p.stat().st_size)
        out[c.upper()] = row
    return out
