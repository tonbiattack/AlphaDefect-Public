"""実ゲームで複数ランを回して結果を記録する（勝率検証用）。

1 ランごとに:

  1. 推論サーバを立てる（`--autostart "DEFECT|0|<seed>"`、`--exit-on-done`）
  2. Steam 経由でゲームを起動する
  3. サーバが 1 ランを終えて終了するまで待つ（タイムアウト付き）
  4. ゲームを閉じる
  5. サーバの JSONL から、勝敗・到達階・意思決定数・所要時間・停止理由・モデル・版を拾う

結果は 1 行 1 ランの JSONL（既定 `~/.local/share/sls2/runs/measure.jsonl`）に追記する。
クラッシュや時間切れも隠さず記録する（`result` に `crash` / `timeout` を残す）。
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from . import launcher as L
from . import paths
from .errors import LocalError

DEFAULT_OUT = paths.home() / "runs" / "measure.jsonl"


@dataclass
class RunRow:
    seed: str
    character: str
    ascension: int
    game: str
    mod_version: str
    model: str
    win: bool | None
    floor: int | None
    actions: int
    seconds: float
    result: str            # finished / crash / timeout / aborted
    reason: str = ""
    started: str = ""

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "character": self.character,
            "ascension": self.ascension,
            "game": self.game,
            "mod_version": self.mod_version,
            "model": self.model,
            "win": self.win,
            "floor": self.floor,
            "actions": self.actions,
            "seconds": round(self.seconds, 1),
            "result": self.result,
            "reason": self.reason,
            "started": self.started,
        }


def _game_running() -> bool:
    return L.game_running()


def _kill_game(log=print) -> None:
    if not _game_running():
        return
    try:
        subprocess.run(["cmd.exe", "/c", "taskkill /IM SlayTheSpire2.exe /F"],
                       capture_output=True, check=False, timeout=30)
        log("  ゲームを閉じました")
    except (OSError, subprocess.SubprocessError):
        log("  ゲームを閉じられませんでした")
    for _ in range(20):
        if not _game_running():
            return
        time.sleep(0.5)


def _server_log_for(started: float) -> Path | None:
    logs = paths.home() / "logs"
    if not logs.is_dir():
        return None
    files = [p for p in logs.glob("autopilot-*.jsonl") if p.stat().st_mtime >= started - 5]
    return max(files, key=lambda p: p.stat().st_mtime) if files else None


def _summarize_jsonl(path: Path) -> dict:
    """サーバの JSONL から必要な情報だけ拾う。"""
    out: dict = {"done": None, "hello": None, "panel": None, "actions": 0, "errors": []}
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            ev = row.get("event")
            if ev == "done":
                out["done"] = row
            elif ev == "hello":
                out["hello"] = row
            elif ev == "panel":
                out["panel"] = row
            elif ev == "panel-command":
                pass
            elif ev == "action":
                out["actions"] += 1
            elif ev == "mod-log" and "auto stop" in str(row.get("msg") or ""):
                out["stalls"] = out.get("stalls", []) + [row.get("msg")]
            elif ev in ("error", "stopped"):
                out["errors"].append(row)
    except OSError:
        pass
    return out


def run_once(seed: str, character: str = "DEFECT", ascension: int = 0, *, timeout: float = 900.0,
             settle: float = 3.0, log=print) -> RunRow:
    d = L.detect()
    started = time.time()
    started_iso = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    row = RunRow(seed=seed, character=character, ascension=ascension,
                 game=d.install.version, mod_version=d.mod_state.built_for or "",
                 model=character, win=None, floor=None, actions=0, seconds=0.0,
                 result="crash", started=started_iso)
    if _game_running():
        raise LocalError("ゲームが動いています。測定の前に閉じてください。")

    spec = f"{character}|{ascension}|{seed}"
    argv = [str(paths.venv_python()), "-m", "local", "autopilot", "--character", "auto",
            "--no-stdin", "--keep-listening", "--control-port", str(L.DEFAULT_CONTROL_PORT),
            "--autostart", spec, "--exit-on-done"]
    log(f"  サーバ: {' '.join(argv[-6:])}")
    server = subprocess.Popen(argv, cwd=str(paths.REPO), stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, start_new_session=True)
    try:
        deadline = time.time() + 120
        while time.time() < deadline:
            if L.control_status() is not None:
                break
            time.sleep(1.0)
        else:
            row.reason = "server did not become ready"
            return row

        L.launch_game(d.install, [])
        log("  ゲーム起動")
        # ゲームのプロセスが出るまで待つ
        t0 = time.time()
        while time.time() - t0 < 120:
            if _game_running():
                break
            time.sleep(2.0)
        else:
            row.reason = "game did not start"
            return row

        # サーバが終わる（= 1 ラン終わり）まで待つ。停止したまま進まないランは打ち切る。
        t0 = time.time()
        last_actions = -1
        stalled_since = 0.0
        while time.time() - t0 < timeout:
            if server.poll() is not None:
                row.result = "finished"
                break
            # ゲームが落ちた → ランは中断
            if not _game_running():
                time.sleep(2.0)
                if server.poll() is not None:
                    row.result = "finished"
                else:
                    row.result = "game exited"
                break
            st = L.control_status() or {}
            actions = int(st.get("actions") or 0)
            if actions != last_actions:
                last_actions = actions
                stalled_since = 0.0
            else:
                paused = st.get("state") == "paused"
                limit = 15 if paused else 90      # 停止中は 15 秒、進行中の無進捗は 90 秒で打ち切り
                if stalled_since == 0.0:
                    stalled_since = time.time()
                elif time.time() - stalled_since > limit:
                    row.result = "stalled"
                    row.reason = ("session paused" if paused else "no progress") + f" for {limit} s"
                    break
            time.sleep(3.0)
        else:
            row.result = "timeout"

        row.seconds = time.time() - started
        logfile = _server_log_for(started)
        if logfile is not None:
            info = _summarize_jsonl(logfile)
            done = info["done"] or {}
            row.actions = info["actions"]
            if row.result == "finished":
                row.win = bool(done.get("win"))
                row.floor = done.get("floor")
                row.reason = "game over"
            elif row.result == "stalled":
                panel = info.get("panel") or {}
                stall = (info.get("stalls") or [""])[-1]
                how = "paused" if panel.get("paused") else "no progress"
                row.reason = stall or f"{how} (mode={panel.get('mode')})"
            else:
                row.reason = row.result
            hello = info["hello"] or {}
            row.mod_version = hello.get("mod_version") or row.mod_version
            row.model = hello.get("character") or row.model
        return row
    finally:
        try:
            if server.poll() is None:
                os.killpg(os.getpgid(server.pid), signal.SIGTERM)
                try:
                    server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(os.getpgid(server.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass
        _kill_game(log=log)
        time.sleep(settle)


def measure(seeds: list[str], character: str = "DEFECT", ascension: int = 0, *,
            timeout: float = 900.0, out: Path = DEFAULT_OUT, log=print) -> list[dict]:
    out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for i, seed in enumerate(seeds, 1):
        log(f"[{i}/{len(seeds)}] seed={seed}")
        try:
            row = run_once(seed, character, ascension, timeout=timeout, log=log)
        except (LocalError, OSError) as e:
            log(f" 失敗: {e}")
            row = RunRow(seed=seed, character=character, ascension=ascension,
                         game="", mod_version="", model=character, win=None, floor=None,
                         actions=0, seconds=0.0, result="crash", reason=str(e),
                         started=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        d = row.to_dict()
        rows.append(d)
        with out.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
        log("  " + json.dumps(d, ensure_ascii=False))
    return rows


def _spire_conf_path(install) -> Path:
    return install.path / "mods" / "SpireMentor" / "SpireMentor.conf"


def _set_spire_autostart(install, enabled: bool, num_runs: int) -> None:
    path = _spire_conf_path(install)
    if not path.is_file():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    data["auto_start"] = bool(enabled)
    data["num_runs"] = int(num_runs)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _mentor_flag(install) -> Path:
    return install.path / "mods" / "AlphaDefectAutopilot" / "mentor.txt"


def run_once_mentor(install, runs_in_session: int = 1, *, timeout: float = 1800.0,
                    character: str = "DEFECT", log=print) -> list[dict]:
    """Spire Mentor モードでランを回し、終わったランを記録する（MentorClient 経由）。

    推論サーバは起動するだけで、ランの開始・画面遷移は spire-mentor が行う。
    終了は MentorClient が game_over/run_complete を done として送ってくるので、
    サーバの JSONL から拾って 1 ラン 1 行にする。
    """
    rows: list[dict] = []
    started = time.time()
    started_iso = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    flag = _mentor_flag(install)
    flag.parent.mkdir(parents=True, exist_ok=True)
    flag.write_text("1", encoding="utf-8")
    _set_spire_autostart(install, True, runs_in_session)
    log_path = paths.home() / "logs" / f"mentor-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    argv = [str(paths.venv_python()), "-m", "local", "autopilot", "--character", "auto",
            "--no-stdin", "--keep-listening", "--control-port", str(L.DEFAULT_CONTROL_PORT),
            "--log", str(log_path)]
    server = subprocess.Popen(argv, cwd=str(paths.REPO), stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, start_new_session=True)
    log("  Mentor モードで起動（推論サーバ + ゲーム）")
    try:
        deadline = time.time() + 120
        while time.time() < deadline:
            if L.control_status() is not None:
                break
            time.sleep(1.0)
        L.launch_game(install, [])
        # ゲームが立ち上がるまで待つ（起動直後に「動いていない」と判断して落とさない）
        spawn_deadline = time.time() + 120
        while time.time() < spawn_deadline and not _game_running():
            time.sleep(2.0)
        if not _game_running():
            log("  ゲームが起動しませんでした")
            return []
        while time.time() - started < timeout:
            if not _game_running():
                time.sleep(3.0)
                break
            time.sleep(5.0)
    finally:
        try:
            if server.poll() is None:
                os.killpg(os.getpgid(server.pid), signal.SIGTERM)
                try:
                    server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(os.getpgid(server.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass
        _kill_game(log=log)
        flag.unlink(missing_ok=True)
        _set_spire_autostart(install, False, runs_in_session)
        # 終わったランをサーバのログから拾う
        try:
            if log_path.is_file():
                for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue
                    if row.get("event") != "done":
                        continue
                    rows.append(RunRow(seed="mentor", character=character, ascension=0,
                                       game=install.version, mod_version="", model=character,
                                       win=bool(row.get("win")), floor=row.get("floor"),
                                       actions=0, seconds=0.0,
                                       result="finished" if row.get("floor") else "stalled",
                                       reason="game over" if row.get("floor") else "restart",
                                       started=started_iso).to_dict())
        except OSError:
            pass
    return rows


def summarize(rows: list[dict]) -> str:
    played = [r for r in rows if r["result"] == "finished"]
    wins = [r for r in played if r["win"]]
    floors = [r["floor"] for r in played if r["floor"] is not None]
    lines = [
        f"ラン: {len(rows)} 件（完走 {len(played)} / 中断・失敗 {len(rows) - len(played)}）",
        f"勝利: {len(wins)} / 完走 {len(played)}" + (f"（{len(wins) / len(played):.0%}）" if played else ""),
    ]
    if floors:
        lines.append(f"到達階: 平均 {sum(floors) / len(floors):.1f}、最良 {max(floors)}、最悪 {min(floors)}")
    if played:
        secs = [r["seconds"] for r in played]
        lines.append(f"所要時間: 平均 {sum(secs) / len(secs):.0f} 秒")
    lines.append("※ 少数試行の勝率は参考値。論文の結果と同等とは扱わない。")
    return "\n".join(lines)
