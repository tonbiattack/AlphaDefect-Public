"""1 局（または数局）を実際に打つ。既存の評価ツールをそのまま呼び、結果を読みやすくまとめる。

貪欲（探索なし）は、どの版でも tools/ability_test.py（論文の評価と同じ道具）。
探索ありは v0.113.1 系だけ tools/future/run_search.py（BOSS・エリート戦の木探索）。
どちらも「リポジトリのコードを書き換えずに」呼べるので、論文の結果と同じ経路で走る。

サブプロセスで呼ぶ理由: 2 つのコードツリー（game-v0.111.0/ と直下）は同じ名前の
python パッケージを持つ。1 つのプロセスに両方を読み込むと取り違えるので、実行ごとに
どちらか一方だけを選んで別プロセスで走らせる。
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import paths
from .errors import LocalError, SetupRequired
from .recipes import Recipe, req_extra_json
from .versions import GameLine


@dataclass
class RunSpec:
    character: str
    seeds: list[str]
    ascension: int = 10
    search: bool = False
    sims: int = 32
    rooms: str = "Boss,Elite"
    device: str = "cpu"
    envs: int = 1
    procs: int = 2
    out: Path | None = None
    time_cap: float = 600.0
    extra: dict | None = None
    replay: bool = True

    @property
    def label(self) -> str:
        return "search" if self.search else "greedy"


@dataclass
class CommandPlan:
    """走らせるコマンド 1 つ分（dry-run でもこの中身を表示する）。"""

    argv: list[str]
    cwd: Path
    env: dict
    out: Path
    seeds_file: Path
    seeds: list[str]
    spec: RunSpec
    recipe: Recipe
    line: GameLine

    def display(self) -> str:
        rel = []
        for a in self.argv:
            s = str(a)
            rel.append(os.path.relpath(s, str(self.cwd)) if os.path.isabs(s) and s.startswith(str(self.cwd)) else s)
        env_pairs = [(k, self.env[k]) for k in ("STS2_GAME_DIR", "DOTNET_ROOT") if k in self.env]
        head = " ".join(f"{k}={v}" for k, v in env_pairs)
        return (f"cwd: {self.cwd}\n" + (f"env: {head}\n" if head else "") + "cmd: " + " ".join(_shq(x) for x in rel))


def _shq(s: str) -> str:
    return s if all(c.isalnum() or c in "-_./:=," for c in s) else "'" + s.replace("'", "'\\''") + "'"


# ---------------------------------------------------------------- 組み立て


def seed_file(out: Path, seeds: list[str]) -> Path:
    p = paths.ensure_dir(out) / "seeds.json"
    p.write_text(json.dumps({"name": "local", "seeds": seeds}, ensure_ascii=False), encoding="utf-8")
    return p


def make_out_dir(spec: RunSpec, create: bool = True) -> Path:
    """出力先を決める。create=False（dry-run）ではディレクトリを作らない。"""
    if spec.out is not None:
        return paths.ensure_dir(spec.out) if create else Path(spec.out)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    tag = spec.seeds[0] if len(spec.seeds) == 1 else f"{len(spec.seeds)}seeds"
    out = paths.runs_dir() / f"{stamp}_{spec.character}_{tag}_{spec.label}"
    return paths.ensure_dir(out) if create else out


def plan(
    line: GameLine,
    spec: RunSpec,
    recipe: Recipe,
    python: Path,
    weights: Path,
    game_dir: Path,
    dotnet: Path | None = None,
    dry_run: bool = False,
) -> CommandPlan:
    """実行コマンドを組み立てる（実行はしない）。dry_run では書き込みも一切しない。"""
    if not python.is_file():
        raise SetupRequired(f"Python の仮想環境がありません: {python}\n  `python -m local setup` を実行してください。")
    if not weights.is_file():
        raise SetupRequired(f"重みがありません: {weights}\n  `python -m local setup` を実行してください。")
    out = make_out_dir(spec, create=not dry_run)
    sfile = seed_file(out, spec.seeds) if not dry_run else out / "seeds.json"
    tool = line.search_tool if spec.search else line.greedy_tool
    script = line.root / tool
    if not script.is_file():
        raise LocalError(f"実行スクリプトがありません: {script}")

    env = dict(os.environ)
    env["STS2_GAME_DIR"] = str(game_dir)
    env["PYTHONUNBUFFERED"] = "1"
    if dotnet is not None:
        env["DOTNET_ROOT"] = str(dotnet.parent)
        env["PATH"] = f"{dotnet.parent}{os.pathsep}" + env.get("PATH", "")

    argv = [str(python), "-I", str(script)]
    extra = req_extra_json(recipe)
    if spec.search:
        if int(spec.ascension) != 10:
            raise LocalError("探索ありは A10 固定です（run_search.py が ascension 10 を直接書いています）。--asc 10 を指定してください。")
        argv += [
            "--seeds", str(sfile),
            "--runs", str(len(spec.seeds)),
            "--ckpt", str(weights),
            "--out", str(out),
            "--character", spec.character,
            "--sims", str(spec.sims),
            "--rooms", spec.rooms,
            "--device", spec.device,
            "--procs", str(max(1, spec.procs)),
            "--cap", "6000",
        ]
        if extra:
            argv += ["--req-extra", extra]
    else:
        argv += [
            "--ckpt", str(weights),
            "--out", str(out),
            "--asc", f"{spec.ascension}-{spec.ascension}",
            "--runs", str(len(spec.seeds)),
            "--seeds-file", str(sfile),
            "--character", spec.character,
            "--envs", str(max(1, min(spec.envs, len(spec.seeds)))),
            "--device", spec.device,
            "--time-cap", f"{spec.time_cap:g}",
        ]
        if extra:
            argv += ["--req-extra", extra]
    return CommandPlan(argv, line.root, env, out, sfile, list(spec.seeds), spec, recipe, line)


# ---------------------------------------------------------------- 実行


@dataclass
class RunOutcome:
    plan: CommandPlan
    seconds: float
    returncode: int
    results: list[dict] = field(default_factory=list)
    tail: str = ""

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and len(self.results) >= len(self.plan.seeds)


def execute(plan_: CommandPlan, stream: bool = True, log=print) -> RunOutcome:
    """サブプロセスで打つ。出力はそのまま流す（長いので進捗が見えるように）。"""
    t0 = time.time()
    tail: list[str] = []
    proc = subprocess.Popen(
        plan_.argv,
        cwd=str(plan_.cwd),
        env=plan_.env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\n")
        tail.append(line)
        if len(tail) > 40:
            tail.pop(0)
        if stream:
            log("  | " + line)
    proc.wait()
    seconds = time.time() - t0
    results = collect(plan_.out)
    return RunOutcome(plan_, seconds, proc.returncode, results, "\n".join(tail))


def collect(out: Path) -> list[dict]:
    """評価ツールが書いた runs_s*.jsonl を読む（1 局 1 行）。"""
    rows: list[dict] = []
    for p in sorted(out.glob("runs_s*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
    seen: set[str] = set()
    uniq: list[dict] = []
    for r in rows:                      # 同じ seed が複数ファイルにあれば最後の 1 行
        s = str(r.get("seed"))
        if s in seen:
            uniq = [x for x in uniq if str(x.get("seed")) != s]
        seen.add(s)
        uniq.append(r)
    return uniq


def pick(outcome: RunOutcome, seed: str) -> dict | None:
    for r in outcome.results:
        if str(r.get("seed")) == seed:
            return r
    return None


# ---------------------------------------------------------------- 表示


def describe(r: dict) -> list[str]:
    """1 局の結果を日本語の数行にする。

    エンジンの act は 0 始まり（0 が第 1 幕）。利用者向けには 1 始まりに直す。
    """
    win = bool(r.get("win"))
    stuck = bool(r.get("stuck"))
    floor = r.get("floor")
    act = int(r.get("act") or 0)
    act_no = act + 1
    combats = r.get("combats") or []
    lines = []
    if win:
        lines.append("結果: 勝利（全 3 幕クリア）")
    elif stuck:
        lines.append(f"結果: 中断（{_stuck_reason(r)}）")
    else:
        lines.append(f"結果: 敗北（第 {act_no} 幕・{floor} 層）")
    lines.append(f"到達: 第 {act_no} 幕 {floor} 層 / 戦闘 {len(combats)} 回")
    if combats:
        last = combats[-1]
        room = last.get("room") or "?"
        enc = last.get("encounter")
        lines.append(f"最後の戦闘: {room}" + (f" {enc}" if enc else "") + f"（第 {int(last.get('act', 0)) + 1} 幕 {last.get('floor')} 層）")
    hp = r.get("hp_delta")
    if isinstance(hp, (int, float)):
        lines.append(f"HP 増減: {hp * 100:+.0f}%")
    if r.get("searches") is not None:
        lines.append(f"探索: {r.get('searches')} 回 / 貪欲から改選: {r.get('changed')} 回")
    if r.get("s") is not None:
        lines.append(f"プレイ時間: {float(r['s']):.1f} 秒")
    return lines


def _stuck_reason(r: dict) -> str:
    """中断の理由（評価ツールが記録したエラーか、エンジンが止まったか）。"""
    err = r.get("err") or r.get("death")
    if err:
        return str(err)
    return "エンジンが停止（上限歩数に到達）"


def replay_paths(out: Path) -> list[Path]:
    return sorted(out.glob("**/replays/*.jsonl.gz"))
