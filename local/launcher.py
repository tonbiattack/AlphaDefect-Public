"""Windows から 1 クリックで「通常プレイ」と「AI 観戦」を選ぶランチャー。

これまでの部品をそのまま使う:

  * ゲームの検出・版の判定 … local/steam.py, local/versions.py
  * MOD の導入・更新      … local/mod.py
  * 推論サーバ            … python -m local autopilot（local/bridge）
  * HTTP の操作口         … 127.0.0.1:18766（pause / resume / stop / speed）

モードは 2 つ:

  normal … 推論を起動せず、ゲームを `--nomods` で起動する（MOD は読み込まれない）。
           セーブは通常プロファイルのまま。ゲームの起動オプション（Steam）は書き換えない。
  ai     … MOD を（必要ならビルドして）導入し、推論サーバを起動して準備完了を待ってから
           ゲームを起動する。ゲームが終わったら、このランチャーが起動したサーバだけを止める。

安全側の決まり:

  * すでにゲームが動いているときは起動しない（勝手に終了させない）。
  * ポートが使われているときは、自分が起動したものか確かめ、他人のものは触らない。
  * Steam の起動オプションに `-nomods` があると AI モードで MOD が読まれないので、事前に知らせる。
  * 2 重起動はロックファイルで防ぐ。
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from . import mod as M
from . import paths, steam
from .errors import LocalError
from .versions import GameLine, line_for

DEFAULT_PORT = 18765
DEFAULT_CONTROL_PORT = 18766
APP_ID = "2868840"
LOCK_NAME = "launcher.lock"


# ---------------------------------------------------------------- 検出


@dataclass
class Detection:
    install: steam.GameInstall
    line: GameLine
    exe: Path                      # Windows のゲーム実行ファイル
    exe_windows: str               # Windows 形式のパス（Start-Process に渡す）
    mod_state: M.ModState
    game_running: bool
    server_up: bool                # 推論サーバが応答しているか（/status）
    control_up: bool               # 操作口が応答しているか
    server_pid: int | None
    wsl_distro: str
    repo: Path
    steam_launch_options: str      # Steam 側の起動オプション（localconfig.vdf）
    problems: list[str] = field(default_factory=list)

    @property
    def nomods_conflict(self) -> bool:
        return "nomods" in self.steam_launch_options.replace(" ", "").lower()

    def to_dict(self) -> dict:
        return {
            "game": self.install.to_dict(),
            "line": self.line.key,
            "exe": self.exe_windows,
            "mod": self.mod_state.to_dict(),
            "game_running": self.game_running,
            "server_up": self.server_up,
            "control_up": self.control_up,
            "server_pid": self.server_pid,
            "wsl_distro": self.wsl_distro,
            "repo": str(self.repo),
            "steam_launch_options": self.steam_launch_options,
            "nomods_conflict": self.nomods_conflict,
            "problems": self.problems,
        }


def game_exe(install: steam.GameInstall) -> Path:
    return install.path / "SlayTheSpire2.exe"


def windows_path(p: Path) -> str:
    """WSL の /mnt/c/... を Windows の C:\\... に戻す（動かしているのが WSL のときだけ）。"""
    s = str(p)
    if not steam.is_wsl():
        return s
    parts = Path(s).parts
    if len(parts) >= 3 and parts[0] == "/" and parts[1] == "mnt" and len(parts[2]) == 1:
        drive = parts[2].upper()
        rest = "\\".join(parts[3:])
        return f"{drive}:\\{rest}" if rest else f"{drive}:\\"
    return s


def _ps(value: str) -> str:
    """PowerShell の単一引用符文字列にする（' は '' にする）。"""
    return "'" + str(value).replace("'", "''") + "'"


def wsl_unc_path(p: Path) -> str:
    r"""WSL のパスを、Windows から見た UNC パス（\wsl.localhost\<distro>\...）にする。"""
    distro = os.environ.get("WSL_DISTRO_NAME", "")
    if not steam.is_wsl() or not distro:
        return windows_path(p)
    parts = [x for x in str(p).split("/") if x]
    return "\\\\wsl.localhost\\" + distro + "\\" + "\\".join(parts)


def _http_json(url: str, timeout: float = 1.5, payload: dict | None = None) -> dict | None:
    try:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return None


def control_status(port: int = DEFAULT_CONTROL_PORT) -> dict | None:
    return _http_json(f"http://127.0.0.1:{port}/status")


def control(command: str, port: int = DEFAULT_CONTROL_PORT) -> dict | None:
    return _http_json(f"http://127.0.0.1:{port}/control", payload={"command": command})


def _listen_pids(port: int) -> list[int]:
    """そのポートを待ち受けているプロセスの PID（分からなければ空）。"""
    pids: list[int] = []
    ss = shutil.which("ss")
    if ss:
        r = subprocess.run([ss, "-ltnp"], capture_output=True, text=True, check=False)
        for line in r.stdout.splitlines():
            if f":{port}" in line and "pid=" in line:
                for chunk in line.split("pid=")[1:]:
                    num = chunk.split(",")[0]
                    if num.isdigit():
                        pids.append(int(num))
    return pids


def _windows_listener_pid(port: int) -> int | None:
    """Windows 側でそのポートを待ち受けている PID（WSL から見た relay も含む）。"""
    try:
        r = subprocess.run(["cmd.exe", "/c", f"netstat -ano | findstr :{port}"], capture_output=True, check=False, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    text = (r.stdout or b"").decode("utf-8", errors="replace")
    for line in text.splitlines():
        cols = line.split()
        if len(cols) >= 5 and cols[1].endswith(f":{port}") and cols[3] == "LISTENING" and cols[4].isdigit():
            return int(cols[4])
    return None


def steam_launch_options(install: steam.GameInstall) -> str:
    """Steam の起動オプション（localconfig.vdf の LaunchOptions）を読む。無ければ空。"""
    roots = steam.steam_roots()
    for root in roots:
        userdata = root / "userdata"
        if not userdata.is_dir():
            continue
        for user in sorted(userdata.iterdir()):
            vdf = user / "config" / "localconfig.vdf"
            if not vdf.is_file():
                continue
            try:
                data = steam.parse_vdf(vdf.read_text(encoding="utf-8", errors="replace"))
            except (OSError, ValueError):
                continue
            found = _find_launch_options(data, APP_ID)
            if found is not None:
                return found
    return ""


def _find_launch_options(node, app_id: str, depth: int = 0):
    """入れ子のどこかに app_id → LaunchOptions があれば返す。"""
    if depth > 8 or not isinstance(node, dict):
        return None
    for key, val in node.items():
        if key == app_id and isinstance(val, dict):
            opts = val.get("LaunchOptions")
            if isinstance(opts, str):
                return opts
        if isinstance(val, dict):
            got = _find_launch_options(val, app_id, depth + 1)
            if got is not None:
                return got
    return None


def detect(install: steam.GameInstall | None = None) -> Detection:
    install = install or steam.find_install()
    line = line_for(install.version)          # 未対応版はここで断る
    exe = game_exe(install)
    problems: list[str] = []
    if not exe.is_file():
        problems.append(f"ゲームの実行ファイルがありません: {exe}")
    st = control_status()
    return Detection(
        install=install,
        line=line,
        exe=exe,
        exe_windows=windows_path(exe),
        mod_state=M.status(install),
        game_running=M.game_running(),
        server_up=st is not None,
        control_up=st is not None,
        server_pid=_listen_pids(DEFAULT_PORT)[0] if _listen_pids(DEFAULT_PORT) else None,
        wsl_distro=os.environ.get("WSL_DISTRO_NAME", ""),
        repo=paths.REPO,
        steam_launch_options=steam_launch_options(install),
        problems=problems,
    )


# ---------------------------------------------------------------- 計画


@dataclass
class LaunchPlan:
    mode: str                 # "normal" | "ai"
    character: str            # "auto" or DEFECT など
    speed: float
    seed: str
    exe_windows: str
    game_args: list[str]
    start_server: bool
    autostart: str | None
    notes: list[str] = field(default_factory=list)


def plan(mode: str, d: Detection, character: str = "auto", speed: float = 1.0,
         seed: str = "", autostart: bool = True) -> LaunchPlan:
    """モードごとの起動内容を組み立てる（実行はしない）。"""
    mode = mode.lower()
    if mode not in ("normal", "ai"):
        raise LocalError(f"モードは normal か ai です: {mode}")
    notes: list[str] = []
    if mode == "normal":
        # MOD を読み込ませない。Steam 側の設定は触らず、この起動だけ --nomods を付ける
        plans = LaunchPlan("normal", "auto", 1.0, seed, d.exe_windows, ["--nomods"], False, None, notes)
        notes.append("MOD は読み込まれません（--nomods）。セーブは通常のプロファイルです。")
        if d.nomods_conflict:
            notes.append("Steam の起動オプションにも nomods が入っています（通常プレイでは問題ありません）。")
        return plans

    if not d.mod_state.installed:
        notes.append("MOD を導入します（初回、または版が変わったとき）。")
    elif d.mod_state.stale:
        notes.append("MOD が今のゲーム版向けにビルドされていないので、入れ直します。")
    if d.nomods_conflict:
        notes.append(
            "注意: Steam の起動オプションに -nomods があります。このままだと MOD が読み込まれず AI が動きません。"
            "Steam の「プロパティ → 起動オプション」から消してください（このランチャーは設定を書き換えません）。"
        )
    auto = None
    if autostart:
        # ランを始めるキャラはゲーム側で決める必要がある。auto のときは Defect で始め、
        # 推論サーバ側（--character auto）がゲームの選択に合わせて重みを選ぶ。
        run_character = "DEFECT" if character.lower() == "auto" else character.upper()
        auto = f"{run_character}|0|{seed or 'AUTO'}"
        notes.append("ゲーム側は通常のラン開始（Neow → マップ）。デバッグの戦闘直行は使いません。")
    return LaunchPlan("ai", character, float(speed), seed, d.exe_windows, [], True, auto, notes)


def server_argv(plan_: LaunchPlan) -> list[str]:
    py = paths.venv_python()
    if not py.is_file():
        raise LocalError("Python の仮想環境がありません。`./setup.sh` を実行してください。")
    argv = [str(py), "-m", "local", "autopilot", "--character", plan_.character,
            "--no-stdin", "--keep-listening", "--control-port", str(DEFAULT_CONTROL_PORT)]
    if plan_.autostart:
        argv += ["--autostart", plan_.autostart]
    return argv


# ---------------------------------------------------------------- 起動


def steam_exe(install: steam.GameInstall) -> Path:
    """Steam 本体（steam.exe）。ゲームのパスから辿る。"""
    p = install.path
    for parent in p.parents:
        cand = parent / "steam.exe"
        if cand.is_file():
            return cand
    for root in steam.steam_roots():
        cand = root / "steam.exe"
        if cand.is_file():
            return cand
    raise LocalError("Steam 本体（steam.exe）が見つかりません。Steam を起動してからやり直してください。")


def launch_game(install: steam.GameInstall, args: list[str]) -> None:
    """Steam 経由でゲームを起動する。

    実行ファイルを直接起動すると Steamworks が初期化されず（ログに
    "Cannot initialize Steam Input ... Steamworks is not initialized" が出る）、
    プロファイルも既定のものになってしまう。`steam.exe -applaunch <appid> <引数>` なら
    Steam 経由のまま、この起動だけに引数（--nomods など）を渡せる。Steam 側の
    起動オプション（localconfig.vdf）は書き換えない。
    """
    steam_bin = windows_path(steam_exe(install))
    quoted_args = ",".join("'" + a.replace("'", "''") + "'" for a in args)
    script = f"$p = Start-Process -FilePath '{steam_bin}' -ArgumentList @('-applaunch','{APP_ID}'"
    if args:
        script += f",{quoted_args}"
    script += ") -PassThru; $p.Id"
    r = subprocess.run(["powershell.exe", "-NoProfile", "-Command", script], capture_output=True, text=True, check=False)
    if r.returncode != 0:
        raise LocalError("ゲームを起動できませんでした:\n" + (r.stdout + r.stderr).strip()[-800:])


def game_running() -> bool:
    return M.game_running()


def wait_server(plan_: LaunchPlan, timeout: float = 120.0, log=print) -> None:
    """推論サーバが操作口に応答するまで待つ。"""
    t0 = time.time()
    seen = False
    while time.time() - t0 < timeout:
        if control_status() is not None:
            if not seen:
                log("  推論サーバ: 準備完了")
                seen = True
            return
        time.sleep(1.0)
    raise LocalError(
        f"推論サーバが {timeout:.0f} 秒以内に応答しませんでした。\n"
        "  ログを確認してください: " + str(paths.home() / "logs")
    )


def start(mode: str, character: str = "auto", speed: float = 1.0, seed: str = "",
          autostart: bool = True, wait: bool = True, dry_run: bool = False,
          log=print) -> int:
    """ランチャーの本体。戻り値は終了コード。"""
    d = detect()
    for p in d.problems:
        raise LocalError(p)

    lock = _acquire_lock()
    if lock is None:
        log("別のランチャーが動いています（ロックファイル）。終了します。")
        return 1
    try:
        return _start_locked(mode, d, character, speed, seed, autostart, wait, dry_run, log)
    finally:
        lock.unlink(missing_ok=True)


def _start_locked(mode, d, character, speed, seed, autostart, wait, dry_run, log) -> int:
    plan_ = plan(mode, d, character, speed, seed, autostart)

    log(f"ゲーム: {d.install.path}（{d.install.version}）")
    log(f"モード: {plan_.mode}" + ("（AI 観戦）" if plan_.mode == "ai" else "（通常プレイ）"))
    log(f"起動: {plan_.exe_windows} {' '.join(plan_.game_args)}".rstrip())
    if plan_.start_server:
        log("  推論サーバ: " + " ".join(server_argv(plan_))
            + (f"  （autostart {plan_.autostart}）" if plan_.autostart else ""))
    for n in plan_.notes:
        log("  " + n)

    if d.game_running:
        log("ゲームが既に動いています。二重起動はしません（そのゲームはそのまま使えます）。")
        return 2
    if dry_run:
        log("（--dry-run のため起動しません）")
        return 0

    if plan_.mode == "normal":
        if d.server_up:
            log("注意: 推論サーバが動いています（このモードでは起動しません）。止めるには --stop。")
        launch_game(d.install, plan_.game_args)
        log("通常プレイで起動しました。MOD は読み込まれません。このウィンドウは閉じて構いません。")
        return 0

    # AI モード: ポートの持ち主を確かめる
    if d.server_up:
        log("推論サーバは既に動いています。このまま使います（このランチャーは停止しません）。")
        server_proc = None
    else:
        for port in (DEFAULT_PORT, DEFAULT_CONTROL_PORT):
            pids = _listen_pids(port)
            if pids:
                raise LocalError(
                    f"ポート {port} が別のプロセスに使われています（PID {pids[0]}）。"
                    "そのプロセスは止めません。使っているアプリを終了してからやり直してください。"
                )
        M.install(d.install, log=lambda s: log(s), allow_running=False)
        log_path = _log_path()
        argv = server_argv(plan_)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(log_path, "ab")  # noqa: SIM115 - サーバの寿命だけ開いておく
        server_proc = subprocess.Popen(argv, cwd=str(paths.REPO), stdout=fh, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        log(f"  推論サーバ: 起動しました（PID {server_proc.pid}、ログ {log_path}）")
        try:
            wait_server(plan_, log=log)
        except LocalError:
            _stop_own(server_proc, log)
            raise
        if plan_.speed != 1.0:
            control(f"speed {plan_.speed}")

    launch_game(d.install, plan_.game_args)
    # Steam 経由の起動は、プロセスが現れるまで少し間がある。すぐ「終わった」と
    # 判断してサーバを止めてしまわないよう、現れるまで待つ。
    if not _wait_game_start(timeout=120.0):
        _stop_own(server_proc, log) if server_proc is not None else None
        raise LocalError(
            "ゲームのプロセスが現れませんでした（Steam の状態・ログインを確認してください）。\n"
            "  Steam を起動してから、もう一度お試しください。"
        )
    log("ゲームを起動しました。AI はメインメニューが出た時点でランを始めます。")
    if not wait:
        log("サーバはこのまま動きます。止めるには `--stop`。")
        return 0
    log("ゲームが終わるまで待ちます（Ctrl-C で止められます）。停止: pause / resume / stop は --pause など。")
    try:
        while game_running():
            time.sleep(3.0)
        log("ゲームが終了しました。")
    except KeyboardInterrupt:
        log("中断しました。")
    finally:
        if server_proc is not None:
            _stop_own(server_proc, log)
        else:
            log("推論サーバは元から動いていたので停止しません。")
    return 0


def _wait_game_start(timeout: float = 120.0) -> bool:
    """ゲームのプロセスが現れるまで待つ（Steam 経由の起動は少し遅れる）。"""
    t0 = time.time()
    while time.time() - t0 < timeout:
        if game_running():
            return True
        time.sleep(2.0)
    return False


def _stop_own(proc: subprocess.Popen, log=print) -> None:
    """このランチャーが起動したサーバだけを止める。"""
    if proc.poll() is not None:
        return
    log("  推論サーバを停止します。")
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        try:
            proc.terminate()
        except OSError:
            pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()


def _log_path() -> Path:
    return paths.home() / "logs" / f"launcher-{time.strftime('%Y%m%d-%H%M%S')}.log"


def _acquire_lock() -> Path | None:
    lock = paths.state_file(LOCK_NAME)
    lock.parent.mkdir(parents=True, exist_ok=True)
    if lock.is_file():
        try:
            pid = int(lock.read_text(encoding="utf-8").strip() or 0)
        except (OSError, ValueError):
            pid = 0
        if pid and _pid_alive(pid):
            return None
        lock.unlink(missing_ok=True)
    lock.write_text(str(os.getpid()), encoding="utf-8")
    return lock


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


# ---------------------------------------------------------------- 停止・操作


def windows_local_appdata() -> Path | None:
    """Windows の %LOCALAPPDATA% を WSL のパスで返す。"""
    try:
        r = subprocess.run(["cmd.exe", "/c", "echo", "%LOCALAPPDATA%"], capture_output=True, check=False, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    raw = (r.stdout or b"").decode("utf-8", errors="replace").strip().splitlines()
    if not raw or not raw[0]:
        return None
    win = raw[0].strip()
    if len(win) > 2 and win[1] == ":":
        return Path("/mnt") / win[0].lower() / win[3:].replace("\\", "/")
    return None


def install_shortcuts(log=print) -> list[Path]:
    """Windows のデスクトップとスタートメニューに、通常プレイ / AI 観戦のショートカットを作る。

    ショートカットの TargetPath には UNC パス（\\wsl.localhost\\...）を置けないので、
    %LOCALAPPDATA%\\AlphaDefect に小さな .cmd を置き、ショートカットはそれを指す。
    .cmd の中身は launch.cmd と同じ形（wsl.exe --cd <UNC> -- bash -lc ...）。
    """
    if not (paths.REPO / "launch.cmd").is_file():
        raise LocalError(f"launch.cmd がありません: {paths.REPO / 'launch.cmd'}")
    local = windows_local_appdata()
    if local is None:
        raise LocalError("%LOCALAPPDATA% を取得できませんでした（WSL から cmd.exe を実行できるか確認してください）。")
    bin_dir = local / "AlphaDefect"
    bin_dir.mkdir(parents=True, exist_ok=True)
    unc = wsl_unc_path(paths.REPO)
    made: list[Path] = []
    for name, arg in (("AlphaDefect AI 観戦", "ai"), ("AlphaDefect 通常プレイ", "normal")):
        cmd_path = bin_dir / f"launch-{arg}.cmd"
        body = (
            "@echo off\r\n"
            "rem AlphaDefect ランチャー（自動生成）。作り直すには: ./play.sh launch --install-shortcuts\r\n"
            f'wsl.exe --cd "{unc}" -- bash -lc "./play.sh launch {arg}"\r\n'
            "if errorlevel 1 pause\r\n"
        )
        cmd_path.write_text(body, encoding="utf-8", newline="")
        made.append(cmd_path)
        lnk_name = name + ".lnk"
        script = (
            "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; "
            "$ws = New-Object -ComObject WScript.Shell; "
            "foreach ($dir in @([Environment]::GetFolderPath('Desktop'), [Environment]::GetFolderPath('Programs'))) { "
            "  $lnk = Join-Path $dir " + _ps(lnk_name) + "; "
            "  $sc = $ws.CreateShortcut($lnk); "
            "  $sc.TargetPath = " + _ps(windows_path(cmd_path)) + "; "
            "  $sc.WorkingDirectory = " + _ps(windows_path(bin_dir)) + "; "
            "  $sc.Description = " + _ps("AlphaDefect: " + name) + "; "
            "  $sc.Save(); Write-Output $lnk }"
        )
        r = subprocess.run(["powershell.exe", "-NoProfile", "-Command", script],
                           capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
        out = (r.stdout or "").strip()
        if r.returncode != 0 or not out:
            raise LocalError("ショートカットを作れませんでした:\n" + (out + (r.stderr or "")).strip()[-600:])
        for line in out.splitlines():
            if line.strip():
                made.append(Path(line.strip()))
    for p in made:
        log(f"  作成: {p}")
    return made


def stop_server(log=print) -> int:
    """動いている推論サーバを止める（操作口 → 無ければポートの持ち主）。"""
    if control("stop") is not None:
        log("推論サーバに停止を送りました。")
        time.sleep(1.0)
        if control_status() is None:
            log("停止しました。")
            return 0
    pids = _listen_pids(DEFAULT_PORT)
    if not pids:
        log("推論サーバは動いていません。")
        return 0
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
            log(f"PID {pid} に SIGTERM を送りました。")
        except (ProcessLookupError, PermissionError) as e:
            log(f"PID {pid} を止められません: {e}")
    return 0


def close_game(log=print, timeout: float = 20.0) -> int:
    """ゲームを安全に閉じる（AI を止めてから、まず正常終了 → だめなら強制）。"""
    import subprocess as _sp

    if not game_running():
        log("ゲームは動いていません。")
        return 0
    # まず MOD 経由で正常終了（セーブしてから終わる）。MOD が無い/未接続なら次の手段へ。
    if control("quit-game") is not None:
        log("MOD に終了を要求しました（保存して閉じます）")
        t0 = time.time()
        while time.time() - t0 < 20:
            if not game_running():
                log("ゲームを閉じました。")
                return 0
            time.sleep(1.0)
    # MOD が居なければ（または応答しなければ）AI を止めてからウィンドウを閉じる
    try:
        stop_server(log=lambda *_a, **_k: None)
    except (OSError, RuntimeError) as e:      # 止められなくても閉じる処理は続ける
        log(f"推論サーバの停止に失敗（続行します）: {e}")
    log("ウィンドウを閉じる要求を送ります")
    _sp.run(["cmd.exe", "/c", "taskkill /IM SlayTheSpire2.exe"], capture_output=True, check=False)
    t0 = time.time()
    while time.time() - t0 < timeout:
        if not game_running():
            log("ゲームを閉じました。")
            return 0
        time.sleep(1.0)
    log(f"{timeout:.0f} 秒たっても閉じないので、強制終了します。")
    _sp.run(["cmd.exe", "/c", "taskkill /IM SlayTheSpire2.exe /F"], capture_output=True, check=False)
    time.sleep(2.0)
    if game_running():
        log("ゲームを閉じられませんでした。手動で終了してください。")
        return 1
    log("ゲームを強制終了しました。")
    return 0


def status_text(d: Detection | None = None) -> str:
    d = d or detect()
    st = control_status()
    lines = [
        f"ゲーム: {d.install.path}",
        f"  バージョン: {d.install.version}（コード系統 {d.line.key}）",
        f"  起動ファイル: {d.exe_windows}",
        f"  MOD: {'導入済み' if d.mod_state.installed else '未導入'}"
        + (f"（ビルド時の版 {d.mod_state.built_for or '?'}）" if d.mod_state.installed else ""),
        f"  ゲームの起動: {'動いています' if d.game_running else '止まっています'}",
        f"  推論サーバ: {'動いています' if d.server_up else '止まっています'}",
        f"  WSL: {d.wsl_distro or '(不明)'}",
        f"  リポジトリ: {d.repo}",
        f"  Steam の起動オプション: {d.steam_launch_options or '(なし)'}"
        + ("  ← -nomods があると AI モードで MOD が読まれません" if d.nomods_conflict else ""),
    ]
    if st:
        lines.append(f"  観戦: {st.get('state')} / 行動 {st.get('actions')} / 重複 {st.get('duplicates')}")
    for p in d.problems:
        lines.append(f"  問題: {p}")
    return "\n".join(lines)
