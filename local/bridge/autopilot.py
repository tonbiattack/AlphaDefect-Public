"""ブリッジ（ゲーム MOD ↔ 推論）の中核。

ソケットを持たない部分をここにまとめ、接続の作法（autopilot.serve）と
テスト（sim）から同じロジックを使う。受け取ったメッセージ 1 つに対して、
返すメッセージの列を返す純粋な形にしてあるので、テストが書きやすい。

守ること（イシュー #5 の要件）:

  * 手動起動・一時停止・再開・即時停止: start / pause / resume / stop
  * 速度制御: speed 倍率。MOD に渡す hold_ms（次の状態を送るまでの待ち）で表す
  * 状態同期: state の seq と、フレームの SHA-256 で同じ局面を見分ける
  * 重複操作の防止: 同じ seq / 同じ局面には 2 回 action を返さない（返すのは前回と同じもの）
  * 例外時停止: 推論や入力の例外は握りつぶさず、error を送って stopped にする
  * ログ: すべての出来事を JSONL と標準出力の両方に残す
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from . import protocol as P


class AutopilotError(RuntimeError):
    """推論側で続行できないことが起きた（例外時停止の対象）。"""


@dataclass
class Decision:
    """推論の結果（MOD に返す 1 手）。"""

    index: int
    label: str = ""
    prob: float = 0.0
    candidates: list = field(default_factory=list)   # [[index, label, prob], ...]
    value: float | None = None


# 推論の入口: フレーム（エンジンと同じバイト列）と meta から 1 手を決める
Decider = Callable[[P.Frame], Decision]


@dataclass
class SessionStats:
    states: int = 0
    actions: int = 0
    duplicates: int = 0
    pauses: int = 0
    probes: int = 0
    steps: int = 0
    started: float = field(default_factory=time.time)
    finished: float | None = None
    result: dict | None = None

    @property
    def seconds(self) -> float:
        return (self.finished or time.time()) - self.started

    def to_dict(self) -> dict:
        return {
            "states": self.states,
            "actions": self.actions,
            "duplicates": self.duplicates,
            "pauses": self.pauses,
            "probes": self.probes,
            "steps": self.steps,
            "seconds": round(self.seconds, 1),
            "result": self.result,
        }


class Autopilot:
    """1 つの接続ぶんの状態機械。"""

    def __init__(
        self,
        decider: Decider,
        character: str,
        run_id: str,
        log: Callable[[str, dict], None] | None = None,
        hold_ms: int = 0,
        max_states: int | None = None,
        merge: bool = False,
        obs_format: int | None = None,
        autostart: str | None = None,
    ):
        self.decider = decider
        self.character = character.upper()
        self.run_id = run_id
        self.log = log or (lambda event, data: None)
        self.hold_ms_base = max(0, int(hold_ms))
        self.max_states = max_states
        self.merge = merge
        self.obs_format = obs_format
        self.autostart = autostart            # "DEFECT|0|SEED|regular" 形式（接続時に start を送る）
        self.panel: dict | None = None        # MOD から届くパネルの状態（表示の同期用）
        self.run_config: dict | None = None    # Mentor の run_config で決めた開始設定
        # 接続し直したときに同じ操作を送り直すための控え（操作名 → メッセージ）
        self.pending_controls: dict[str, dict] = {}

        self.state = "idle"          # idle / running / paused / stopped / error
        self.speed = 1.0
        self.stats = SessionStats()
        self.hello: P.Hello | None = None
        self.stopped_reason = ""
        self._answers: dict[int, dict] = {}          # seq -> 返した action
        self._by_digest: dict[str, dict] = {}        # フレームの digest -> 返した action
        self._last_seq = -1
        self._last_digest = ""

    # ------------------------------------------------------------ 制御

    @property
    def running(self) -> bool:
        return self.state == "running"

    def pause(self, reason: str = "operator") -> list[dict]:
        self.pending_controls["pause"] = P.control("paused", reason)
        if self.state in ("stopped", "error"):
            return []
        if self.state != "paused":
            self.stats.pauses += 1
            self.log("pause", {"reason": reason, "at": self.stats.steps})
        self.state = "paused"
        return [P.control(P.CONTROL_PAUSED, reason)]

    def resume(self) -> list[dict]:
        self.pending_controls.pop("pause", None)
        if self.state != "paused":
            return []
        self.state = "running"
        self.log("resume", {"at": self.stats.steps})
        return [P.control(P.CONTROL_RUNNING, "operator")]

    def set_speed(self, speed: float) -> list[dict]:
        if not 0.1 <= float(speed) <= 20:
            raise AutopilotError(f"速度は 0.1〜20 の範囲で指定してください: {speed}")
        self.speed = float(speed)
        self.log("speed", {"speed": self.speed, "hold_ms": self.hold_ms()})
        return [P.speed(self.speed)]

    def stop(self, reason: str = "operator") -> list[dict]:
        self.pending_controls.pop("pause", None)
        if self.state == "stopped":
            return []
        self.state = "stopped"
        self.stopped_reason = reason
        self.stats.finished = time.time()
        self.log("stop", {"reason": reason, **self.stats.to_dict()})
        return [P.control(P.CONTROL_STOPPED, reason)]

    def step(self) -> list[dict]:
        """1 手だけ進める（ゲーム内パネルの「1手」と同じ）。"""
        self.log("step", {})
        return [P.command("step")]

    def set_mode(self, mode: str) -> list[dict]:
        """auto / manual を切り替える（手動では AI が操作しない）。"""
        mode = mode.lower()
        if mode not in ("auto", "manual"):
            raise AutopilotError(f"モードは auto か manual です: {mode}")
        self.log("mode", {"mode": mode})
        msg = P.command(mode)
        self.pending_controls["mode"] = msg
        return [msg]

    def set_confirm(self, on: bool) -> list[dict]:
        """各判断の前に止めて、候補と確率を見せてから実行する。"""
        self.log("confirm", {"on": on})
        msg = P.command("confirm-on" if on else "confirm-off")
        self.pending_controls["confirm"] = msg
        return [msg]

    def set_entry_delay(self, ms: int) -> list[dict]:
        """部屋に入ってから操作を始めるまでの待ち（実時間。TimeScale に依存しない）。"""
        ms = max(0, min(int(ms), 8000))
        self.log("entry_delay", {"ms": ms})
        msg = P.command("entry", ms)
        self.pending_controls["entry"] = msg
        return [msg]

    def set_pacing(self, ms: int) -> list[dict]:
        """行動の間隔（ミリ秒）。速度（TimeScale）とは独立。"""
        ms = max(0, min(int(ms), 10000))
        self.log("pacing", {"ms": ms})
        msg = P.command("pacing", ms)
        self.pending_controls["pacing"] = msg
        return [msg]

    def set_stop_points(self, combat_end: bool | None = None, card_reward: bool | None = None) -> list[dict]:
        out = []
        if combat_end is not None:
            self.log("stop_point", {"where": "combat_end", "on": combat_end})
            msg = P.command("stopcombat", "on" if combat_end else "off")
            self.pending_controls["stopcombat"] = msg
            out.append(msg)
        if card_reward is not None:
            self.log("stop_point", {"where": "card_reward", "on": card_reward})
            msg = P.command("stopreward", "on" if card_reward else "off")
            self.pending_controls["stopreward"] = msg
            out.append(msg)
        return out

    def hold_ms(self) -> int:
        """MOD に指示する待ち時間（速度倍率で割る）。"""
        if self.hold_ms_base <= 0:
            return 0
        return round(self.hold_ms_base / self.speed)

    # ------------------------------------------------------------ 受信

    def start(self) -> list[dict]:
        """接続直後に呼ぶ（welcome を返す）。

        接続前に操作で一時停止していたら、その状態を保つ（接続で勝手に再開しない）。
        エラーで止まったセッションは、そのまま止まったままにする。
        """
        if self.state in ("idle", "stopped"):
            self.state = "running"
        self.log("start", {"character": self.character, "run": self.run_id})
        out = [P.welcome(self.run_id, P.PROTOCOL_VERSION, self.character, self.speed, self.obs_format), P.speed(self.speed)]
        # 接続前に操作していた内容（停止・手動・確認・間隔・停止点）を送り直す。
        # これが無いと、つなぎ直したあとに勝手に動き出してしまう。
        out.extend(self.pending_controls.values())
        if self.autostart:
            out.append(self._start_message(self.autostart))
        return out

    def _start_message(self, spec: str) -> dict:
        parts = spec.split("|")
        ch = (parts[0] or self.character).upper() if parts else self.character
        asc = int(parts[1]) if len(parts) > 1 and parts[1] else 0
        seed = parts[2] if len(parts) > 2 and parts[2] else "ALPHADEFECT"
        enc = parts[3] if len(parts) > 3 else ""      # 空ならゲーム本来の流れ（Neow → マップ）で始める
        self.log("start-request", {"character": ch, "ascension": asc, "seed": seed, "encounter": enc})
        return P.start(ch, asc, seed, enc)

    def handle(self, msg: dict) -> list[dict]:
        """MOD からの 1 メッセージを処理し、返すメッセージの列を返す。"""
        kind = msg.get("t")
        if kind == P.C_HELLO:
            return self._on_hello(msg)
        if kind == P.C_STATE:
            return self._on_state(msg)
        if kind == P.C_DONE:
            return self._on_done(msg)
        if kind == P.C_PROBE:
            data = dict(msg.get("data") or {})
            self.stats.probes += 1
            self.log("probe", {"data": data, "n": self.stats.probes})
            return []
        if kind == P.C_PANEL:
            state = dict(msg.get("state") or {})
            self.panel = state
            # MOD が正なので、サーバ側の state も合わせる（/status とランチャーの表示を一致させる）
            if self.state not in ("stopped", "error", "error-stopped"):
                self.state = "paused" if state.get("paused") else "running"
            self.log("panel", state)
            return []
        if kind == P.C_RUNCONFIG:
            cfg = {"character": msg.get("character"), "ascension": msg.get("ascension"),
                   "seed": msg.get("seed")}
            self.run_config = dict(cfg)
            self.log("runconfig", cfg)
            return []
        if kind == P.C_LOG:
            self.log("mod-log", {"level": msg.get("level"), "msg": msg.get("msg")})
            return []
        if kind == P.C_BYE:
            self.stats.finished = time.time()
            self.log("bye", {"reason": msg.get("reason"), **self.stats.to_dict()})
            self.state = "stopped"
            self.stopped_reason = str(msg.get("reason") or "bye")
            return []
        self.log("unknown", {"type": kind})
        return []

    def _on_hello(self, msg: dict) -> list[dict]:
        hello = P.Hello.parse(msg)
        self.hello = hello
        self.character = hello.character or self.character
        if hello.protocol != P.PROTOCOL_VERSION:
            self.state = "error"
            self.log("protocol-mismatch", {"theirs": hello.protocol, "ours": P.PROTOCOL_VERSION})
            return [P.error(f"プロトコルの版が違います（MOD {hello.protocol} / こちら {P.PROTOCOL_VERSION}）")]
        self.log("hello", {
            "game": hello.game_version,
            "character": hello.character,
            "mod": hello.mod_id,
            "mod_version": hello.mod_version,
            "client": hello.client,
        })
        return []

    def _on_state(self, msg: dict) -> list[dict]:
        if self.state in ("stopped", "error"):
            return []
        try:
            seq = int(msg["seq"])
            frame = P.Frame(seq=seq, frame=P.unpack_frame(msg.get("frame", "")), meta=dict(msg.get("meta") or {}))
        except (KeyError, TypeError, ValueError, P.ProtocolError) as e:
            return self._fail(f"state を読めません: {e}")

        # 同じ seq の再送: 前回と同じ答えを返す（重複操作の防止）
        if seq in self._answers:
            self.stats.duplicates += 1
            self.log("duplicate-seq", {"seq": seq})
            return [self._answers[seq]]

        self.stats.states += 1
        if self.max_states is not None and self.stats.states > self.max_states:
            return self.stop("max-states")

        digest = frame.digest
        cached = self._by_digest.get(digest)
        if cached is not None:
            # 違う seq だが同じ局面（ポーズ解除後の再送など）。推論し直さず同じ手を返す
            self.stats.duplicates += 1
            self.log("duplicate-frame", {"seq": seq, "digest": digest, "same_as": cached["seq"]})
            answer = dict(cached, seq=seq)
            self._answers[seq] = answer
            return [answer]

        if self.state == "paused":
            # 一時停止中は推論しない（MOD 側も待つ）。再開後に同じ状態が来たら答える
            self.log("paused-state", {"seq": seq})
            return []

        t0 = time.perf_counter()
        try:
            decision = self.decider(frame)
            if not isinstance(decision, Decision):
                raise AutopilotError(f"decider の戻り値が Decision ではありません: {type(decision)!r}")
        except Exception as e:                     # noqa: BLE001 - 例外時は必ず止める（要件）
            return self._fail(f"推論に失敗しました: {e}")
        ms = (time.perf_counter() - t0) * 1000
        # 観測・候補の一致確認（#28）: MOD の候補とヘッドレス系デコードの突き合わせ
        checked = getattr(self.decider, "label_checked", 0)
        if checked and checked % 50 == 0:
            self.log("label-check", {
                "checked": checked,
                "mismatch": getattr(self.decider, "label_mismatch", 0),
                "first": getattr(self.decider, "label_first", None),
            })
            hdr = getattr(self.decider, "header_checked", 0)
            if hdr:
                self.log("header-check", {
                    "checked": hdr,
                    "mismatch": getattr(self.decider, "header_mismatch", 0),
                    "first": getattr(self.decider, "header_first", None),
                })

        answer = P.action(
            seq,
            decision.index,
            decision.label,
            decision.prob,
            decision.candidates,
            self.hold_ms(),
        )
        self._answers[seq] = answer
        self._by_digest[digest] = dict(answer)
        self._last_seq, self._last_digest = seq, digest
        self.stats.actions += 1
        self.stats.steps += 1
        self.log("action", {
            "seq": seq,
            "index": decision.index,
            "label": decision.label,
            "prob": round(decision.prob, 4),
            "value": decision.value,
            "decide_ms": round(ms, 1),
            "meta": frame.meta,
        })
        if decision.candidates:
            self.log("candidates", {"seq": seq, "top": decision.candidates[:5]})
        return [answer]

    def _on_done(self, msg: dict) -> list[dict]:
        self.stats.result = {
            "win": bool(msg.get("win")),
            "floor": int(msg.get("floor") or 0),
            "act": int(msg.get("act") or 0),
            "steps": self.stats.steps,
        }
        self.state = "stopped"
        self.stats.finished = time.time()
        self.stopped_reason = "run-finished"
        self.log("done", {**self.stats.result, **self.stats.to_dict()})
        return [P.control(P.CONTROL_STOPPED, "run-finished")]

    def _fail(self, message: str) -> list[dict]:
        self.state = "error"
        self.stopped_reason = message
        self.stats.finished = time.time()
        self.log("error", {"msg": message, **self.stats.to_dict()})
        return [P.error(message), P.control(P.CONTROL_STOPPED, "error")]

    # ------------------------------------------------------------ 後始末

    def summary(self) -> dict:
        return {
            "run": self.run_id,
            "panel": self.panel,
            "character": self.character,
            "state": self.state,
            "reason": self.stopped_reason,
            "speed": self.speed,
            "hello": None if self.hello is None else {
                "game": self.hello.game_version,
                "mod": self.hello.mod_id,
                "mod_version": self.hello.mod_version,
                "client": self.hello.client,
            },
            **self.stats.to_dict(),
        }


class SessionLog:
    """JSONL のログ（1 行 1 イベント）と標準出力を同時に書く。"""

    def __init__(self, path: Path | None = None, echo: bool = True):
        self.path = path
        self.echo = echo
        self._fh = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(path, "a", encoding="utf-8")  # noqa: SIM115 - close() で閉じる（セッション中ずっと書く）

    def __call__(self, event: str, data: dict) -> None:
        row = {"t": round(time.time(), 3), "event": event, **data}
        if self._fh is not None:
            self._fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            self._fh.flush()
        if self.echo:
            extra = _echo_fields(event, data)
            print(f"[autopilot] {event}{extra}", flush=True)

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def _echo_fields(event: str, data: dict) -> str:
    """標準出力に出す短い要約（JSON をそのまま流さない）。"""
    if event == "action":
        meta = data.get("meta") or {}
        where = f" 第{int(meta.get('act') or 0) + 1}幕 {meta.get('floor')}層" if meta else ""
        cand = f" / 候補 {data.get('label')}" if data.get("label") else ""
        return f"{where} -> [{data.get('index')}]{cand} p={data.get('prob')} ({data.get('decide_ms')}ms)"
    if event == "hello":
        return f" game={data.get('game')} character={data.get('character')} mod={data.get('mod')}"
    if event in ("pause", "resume", "speed", "stop", "done", "error", "protocol-mismatch"):
        return " " + json.dumps({k: v for k, v in data.items() if k not in ("seconds", "started")}, ensure_ascii=False)
    if event in ("duplicate-seq", "duplicate-frame"):
        return f" seq={data.get('seq')}"
    return ""
