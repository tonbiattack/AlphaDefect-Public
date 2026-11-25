"""ゲーム（Windows 側の MOD）と推論（WSL 側）をつなぐプロトコル。

1 本の TCP 接続の上を、1 行 1 つの JSON が流れる（JSON Lines）。接続は MOD 側から
127.0.0.1 に張る。MOD は「いまの公開状態」を送り、こちらが「どの候補を選ぶか」を返す。

  MOD → 推論:  hello / probe / state / done / log / bye
  推論 → MOD:  welcome / start / action / control / error

決まりごと:

  * 版（v）が合わなければ hello の時点で切る。知らない種別は受け取っても無視する。
  * state には seq（MOD 側で単調増加）を付ける。こちらは同じ seq に 2 回 action を返さない。
    同じ seq が再送されたら、前回返した action をそのまま返す（重複操作の防止）。
  * state の frame は、無頭エンジンの serve が返す観測フレームと同じバイト列（base64）。
    形式は python/sts2env/protocol.py の parse がそのまま読める。隠し情報（tok_priv）は
    推論では使わない（論文と同じ情報境界）。
  * control は推論側からの指示（paused / running / stopped）。MOD はそれに従う。
  * action の hold_ms は「次の状態を送るまで待つ時間」で、速度制御に使う。
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

PROTOCOL_VERSION = 1
MAX_LINE_BYTES = 32 * 1024 * 1024      # 1 行の上限（観測フレームは大きめ）

# メッセージ種別
C_HELLO = "hello"
C_STATE = "state"
C_DONE = "done"          # 1 ラン終了（勝敗つき）
C_PROBE = "probe"        # MOD が読める公開状態の要約（観測フレームを送る前段）
C_PANEL = "panel"        # MOD のパネル状態（自動/手動・停止・速度・間隔・接続）
C_LOG = "log"
C_BYE = "bye"
S_SPEED = "speed"         # ゲーム側の演出速度（Godot の TimeScale）
S_COMMAND = "command"     # ゲーム内パネルと同じ操作（step / manual / auto / pacing …）
S_PANEL = "panel"         # MOD → サーバ: パネルの状態（/status と表示を一致させる）
S_START = "start"         # ゲーム側で 1 局を始める（観戦の起動を簡単にする）
S_WELCOME = "welcome"
S_ACTION = "action"
S_CONTROL = "control"
S_ERROR = "error"

CONTROL_RUNNING = "running"
CONTROL_PAUSED = "paused"
CONTROL_STOPPED = "stopped"


class ProtocolError(ValueError):
    """相手が壊れた／想定外の行を送ってきた。接続を切る。"""


@dataclass
class Frame:
    """MOD から届いた 1 つの観測フレーム。"""

    seq: int
    frame: bytes
    meta: dict = field(default_factory=dict)

    @property
    def digest(self) -> str:
        """同じ局面かどうかの判定に使う（フレームの SHA-256 の先頭 16 桁）。"""
        return hashlib.sha256(self.frame).hexdigest()[:16]


@dataclass
class Hello:
    protocol: int
    game_version: str
    character: str
    mod_id: str = ""
    mod_version: str = ""
    client: str = "game"

    @classmethod
    def parse(cls, msg: dict) -> Hello:
        game = msg.get("game") or {}
        mod = msg.get("mod") or {}
        try:
            return cls(
                protocol=int(msg["v"]),
                game_version=str(game.get("version") or ""),
                character=str(msg.get("character") or "").upper(),
                mod_id=str(mod.get("id") or ""),
                mod_version=str(mod.get("version") or ""),
                client=str(msg.get("client") or "game"),
            )
        except (KeyError, TypeError, ValueError) as e:
            raise ProtocolError(f"hello の項目が足りません: {e}") from e


def encode(msg: dict) -> bytes:
    """1 行の JSON にする（改行付き）。"""
    return (json.dumps(msg, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def decode(line: bytes) -> dict:
    """1 行の JSON を読む。壊れていれば ProtocolError。"""
    if len(line) > MAX_LINE_BYTES:
        raise ProtocolError(f"1 行が大きすぎます（{len(line)} バイト）")
    try:
        msg = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as e:
        raise ProtocolError(f"JSON として読めません: {e}") from e
    if not isinstance(msg, dict):
        raise ProtocolError("JSON の一番外側がオブジェクトではありません")
    return msg


def pack_frame(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def unpack_frame(text: str) -> bytes:
    try:
        return base64.b64decode(text.encode("ascii"), validate=True)
    except (binascii.Error, ValueError) as e:
        raise ProtocolError(f"frame が base64 として読めません: {e}") from e


def hello(client: str, protocol: int, game_version: str, character: str, mod_id: str = "", mod_version: str = "") -> dict:
    return {
        "t": C_HELLO,
        "v": protocol,
        "client": client,
        "game": {"version": game_version},
        "character": character,
        "mod": {"id": mod_id, "version": mod_version},
    }


def state(frame: bytes, seq: int, **meta: Any) -> dict:
    return {"t": C_STATE, "seq": int(seq), "frame": pack_frame(frame), "meta": meta}


def probe(data: dict, **extra: Any) -> dict:
    """MOD から送る状態の要約。行動は返さない（ログと同期の確認用）。"""
    return {"t": C_PROBE, "data": dict(data), **extra}


def done(win: bool, floor: int, act: int, **extra: Any) -> dict:
    return {"t": C_DONE, "win": bool(win), "floor": int(floor), "act": int(act), **extra}


def action(seq: int, index: int, label: str = "", prob: float = 0.0, candidates: list | None = None, hold_ms: int = 0) -> dict:
    """推奨手。candidates は [[index, label, prob], ...]（表示用）。"""
    msg = {"t": S_ACTION, "seq": int(seq), "index": int(index), "label": label, "prob": round(float(prob), 6), "hold_ms": int(hold_ms)}
    if candidates:
        msg["candidates"] = candidates
    return msg


def start(character: str, ascension: int = 0, seed: str = "ALPHADEFECT", encounter: str = "") -> dict:
    """ゲーム側に 1 局を始めさせる。encounter は weak / regular / elite / boss か ENCOUNTER.XXX。"""
    return {
        "t": S_START,
        "character": character.upper(),
        "ascension": int(ascension),
        "seed": seed,
        "encounter": encounter,
    }


def command(name: str, value: object | None = None) -> dict:
    """ゲーム内パネルと同じ操作を MOD へ送る。"""
    msg = {"t": S_COMMAND, "name": str(name)}
    if value is not None:
        msg["value"] = str(value)
    return msg


def panel(state: dict) -> dict:
    """MOD から送られてくるパネルの状態（そのまま返す）。"""
    return {"t": S_PANEL, "state": state}


def speed(scale: float) -> dict:
    """ゲーム側の演出速度。MOD は Godot の Engine.TimeScale にそのまま入れる。"""
    return {"t": S_SPEED, "scale": round(float(scale), 3)}


def control(state_name: str, reason: str = "") -> dict:
    return {"t": S_CONTROL, "state": state_name, "reason": reason}


def error(message: str, fatal: bool = True) -> dict:
    return {"t": S_ERROR, "msg": message, "fatal": bool(fatal)}


def welcome(run_id: str, protocol: int, character: str, speed: float, obs_format: int | None = None) -> dict:
    msg = {"t": S_WELCOME, "v": protocol, "run": run_id, "character": character, "speed": speed}
    if obs_format is not None:
        msg["obs_format"] = int(obs_format)     # MOD はこの形式で観測を符号化する
    return msg
