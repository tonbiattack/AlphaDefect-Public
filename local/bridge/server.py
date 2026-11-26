"""TCP の受け口と、人が操作するための標準入力。

MOD（ゲーム側）が 127.0.0.1:<port> に接続してくるのを待ち、1 行 1 JSON で話す。
同じプロセスで標準入力を読んで、pause / resume / stop / speed / status を受け付ける。

    ./play.sh autopilot --character DEFECT
    > 一時停止: pause        > 再開: resume      > 即時停止: stop
    > 速度: speed 2  （MOD に渡す待ち時間を 1/2 にする）
"""

from __future__ import annotations

import json
import socket
import sys
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import protocol as P
from .autopilot import Autopilot

HELP = """操作:
  pause        一時停止（MOD は次の行動を待つ）
  resume       再開
  stop         即時停止（止まったまま。再開は MOD を起動し直す）
  speed N      ゲームの演出速度 0.1〜20（Godot の TimeScale）
  step         1 手だけ実行して再び止める
  manual       手動に切り替える（AI は操作しない。ゲームは止めない）
  auto         AI 自動に戻す
  confirm on|off  各判断の前に止めて、候補を見せてから実行する
  pacing N     行動の間隔（ミリ秒）。速度とは独立
  entry N      部屋に入ってから操作するまでの待ち（ミリ秒、既定 1500）
  stopcombat on|off  戦闘終了で自動停止
  stopreward on|off  カード報酬画面で自動停止
  status       いまの状態と手数の表示
  help         この一覧
"""


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 18765
    control_port: int = 18766      # 操作（pause/resume/stop/speed/status）を HTTP で受ける。0 で無効
    idle_timeout: float = 300.0      # これだけ無通信なら止める（既定 5 分）
    exit_on_done: bool = True        # 1 ラン終わったら受け口も閉じる
    stdin: bool = True


class _Conn:
    """接続 1 本。書き込みはロックで直列化する（読み手のスレッドと操作スレッドが同時に書くため）。"""

    def __init__(self, sock: socket.socket, peer: str):
        self.sock = sock
        self.peer = peer
        self.lock = threading.Lock()

    def send(self, msg: dict) -> bool:
        data = P.encode(msg)
        try:
            with self.lock:
                self.sock.sendall(data)
            return True
        except OSError:
            return False

    def send_many(self, msgs) -> bool:
        for m in msgs:
            if not self.send(m):
                return False
        return True

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass


class _Controls:
    """標準入力からの操作。接続が無くても状態は変わる（次の接続に引き継ぐ）。"""

    def __init__(self, ap: Autopilot, stop_event: threading.Event, log):
        self.ap = ap
        self.stop_event = stop_event
        self.log = log
        self.conn: _Conn | None = None

    def attach(self, conn: _Conn | None) -> None:
        self.conn = conn

    def apply(self, line: str) -> None:
        parts = line.strip().split()
        if not parts:
            return
        cmd = parts[0].lower()
        try:
            if cmd in ("pause", "p"):
                self._send(self.ap.pause("operator"))
            elif cmd in ("resume", "r"):
                self._send(self.ap.resume())
            elif cmd in ("stop", "s"):
                self._send(self.ap.stop("operator"))
                self.stop_event.set()
            elif cmd == "speed":
                self._send(self.ap.set_speed(float(parts[1])))
            elif cmd == "step":
                self._send(self.ap.step())
            elif cmd in ("handover", "hand-over", "manual-takeover"):
                self._send(self.ap.command_handover())
            elif cmd in ("manual", "auto"):
                self._send(self.ap.set_mode(cmd))
            elif cmd in ("confirm", "confirm-on", "confirm-off"):
                on = cmd != "confirm-off"
                if cmd == "confirm" and len(parts) > 1:
                    on = parts[1].lower() not in ("off", "false", "0", "no")
                self._send(self.ap.set_confirm(on))
            elif cmd == "pacing":
                self._send(self.ap.set_pacing(int(parts[1])))
            elif cmd == "entry":
                self._send(self.ap.set_entry_delay(int(parts[1])))
            elif cmd in ("stopcombat", "stopreward"):
                on = len(parts) < 2 or parts[1].lower() not in ("off", "false", "0", "no")
                self._send(self.ap.set_stop_points(
                    combat_end=on if cmd == "stopcombat" else None,
                    card_reward=on if cmd == "stopreward" else None))
            elif cmd == "status":
                print(_status_text(self.ap), flush=True)
            elif cmd in ("help", "h", "?"):
                print(HELP, flush=True)
            elif cmd in ("quit", "q", "exit"):
                self._send(self.ap.stop("quit"))
                self.stop_event.set()
            else:
                print(f"知らない操作: {cmd}（help で一覧）", file=sys.stderr, flush=True)
        except (IndexError, ValueError) as e:
            print(f"操作の引数が変です: {line!r}（{e}）", file=sys.stderr, flush=True)

    def _send(self, msgs) -> None:
        if not msgs:
            return
        if self.conn is not None and self.conn.send_many(msgs):
            return
        # 未接続なら状態だけ変えて、次の接続時に伝わる（welcome の後に送るのは serve 側の責務）

    def read_loop(self) -> None:
        for line in sys.stdin:
            self.apply(line)
        self.stop_event.set()


def _status_text(ap: Autopilot) -> str:
    s = ap.stats
    result = f" / 結果 {s.result}" if s.result else ""
    panel = ""
    if ap.panel:
        p = ap.panel
        mode = "手動" if p.get("mode") == "manual" else "自動"
        paused = "停止" if p.get("paused") else "実行"
        panel = f" / パネル {mode}・{paused}"
        if p.get("confirm"):
            panel += "・確認"
        if p.get("pacing_ms"):
            panel += f"・間隔{p['pacing_ms']}ms"
        if p.get("entry_ms") is not None:
            panel += f"・入場待ち{p['entry_ms']}ms"
    return (f"[autopilot] {ap.state} / キャラ {ap.character} / 速度 {ap.speed}倍{panel}"
            f" / 状態 {s.states} / 行動 {s.actions} / 重複 {s.duplicates} / {s.seconds:.0f} 秒{result}")


def _control_server(ap: Autopilot, controls: _Controls, port: int, log) -> ThreadingHTTPServer:
    """HTTP の操作口。/status は状態、/control は {"command": "pause"|"resume"|"stop"|"speed N"}。

    端末の無い環境（バックグラウンド起動）でも一時停止・再開・停止ができるようにするため。
    """

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):        # 既定のログはうるさいので黙らせる
            return

        def _send(self, code: int, body: dict) -> None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.rstrip("/") in ("/status", ""):
                self._send(200, ap.summary())
            else:
                self._send(404, {"error": "use /status or POST /control"})

        def do_POST(self):
            if self.path.rstrip("/") != "/control":
                self._send(404, {"error": "use /control"})
                return
            try:
                n = int(self.headers.get("Content-Length") or 0)
                payload = json.loads(self.rfile.read(n) or b"{}")
                command = str(payload.get("command") or "").strip()
            except (ValueError, OSError) as e:
                self._send(400, {"error": f"bad body: {e}"})
                return
            try:
                controls.apply(command)
            except Exception as e:                # noqa: BLE001 - 操作の失敗はそのまま返す
                self._send(400, {"error": str(e)})
                return
            self._send(200, {"ok": True, "command": command, "state": ap.state, "steps": ap.stats.steps})

    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, name="autopilot-control", daemon=True).start()
    log("control", {"url": f"http://127.0.0.1:{port}/status"})
    return srv


def serve(ap: Autopilot, cfg: ServerConfig, log=None) -> int:
    """接続を待って 1 ランを付き合う。戻り値は終了コード。"""
    event = log or ap.log
    stop_event = threading.Event()
    controls = _Controls(ap, stop_event, event)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((cfg.host, cfg.port))
    srv.listen(1)
    srv.settimeout(1.0)
    print(f"[autopilot] {cfg.host}:{cfg.port} でゲーム側の接続を待っています（終了は Ctrl-C / quit）", flush=True)
    print(HELP, flush=True)
    event("listen", {"host": cfg.host, "port": cfg.port})

    if cfg.stdin:
        threading.Thread(target=controls.read_loop, daemon=True, name="autopilot-stdin").start()
    control_srv = None
    if cfg.control_port:
        try:
            control_srv = _control_server(ap, controls, cfg.control_port, event)
            print(f"[autopilot] 操作は http://127.0.0.1:{cfg.control_port}/status と POST /control", flush=True)
        except OSError as e:
            print(f"[autopilot] 操作口を開けませんでした（{e}）", file=sys.stderr, flush=True)

    try:
        while not stop_event.is_set():
            try:
                sock, addr = srv.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            conn = _Conn(sock, f"{addr[0]}:{addr[1]}")
            event("connected", {"peer": conn.peer})
            controls.attach(conn)
            try:
                _session(ap, conn, cfg, event, stop_event)
            finally:
                controls.attach(None)
                conn.close()
                event("disconnected", {"peer": conn.peer, "state": ap.state})
            if ap.state in ("stopped", "error") and cfg.exit_on_done:
                stop_event.set()
    finally:
        srv.close()
        if control_srv is not None:
            control_srv.shutdown()
        controls.attach(None)

    code = 0 if ap.state in ("stopped", "error") else 1
    if ap.state == "error":
        code = 1
    print(_status_text(ap), flush=True)
    event("exit", ap.summary())
    return code


def _session(ap: Autopilot, conn: _Conn, cfg: ServerConfig, event, stop_event: threading.Event) -> None:
    """接続 1 本を最後まで面倒みる。"""
    conn.send_many(ap.start())
    conn.sock.settimeout(1.0)
    buf = b""
    last_rx = time.time()
    while not stop_event.is_set() and ap.state not in ("stopped", "error"):
        try:
            chunk = conn.sock.recv(65536)
        except TimeoutError:
            if cfg.idle_timeout and time.time() - last_rx > cfg.idle_timeout:
                # 止めない。ゲームがメニューに居る間は何も来ないのが普通。
                ap.log("idle", {"seconds": round(time.time() - last_rx)})
                last_rx = time.time()
            continue
        except OSError:
            break
        if not chunk:
            break
        last_rx = time.time()
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            if not line.strip():
                continue      # 空行（別のアプリのポート確認など）は黙って無視する
            try:
                msg = P.decode(line)
            except P.ProtocolError as e:
                ap.log("bad-line", {"error": str(e)})
                conn.send(P.error(str(e)))
                return
            replies = ap.handle(msg)
            if replies and not conn.send_many(replies):
                return
            if ap.state in ("stopped", "error"):
                return
