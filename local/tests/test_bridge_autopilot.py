"""ブリッジの状態機械（停止・一時停止・速度・重複防止・例外時停止・ログ）のテスト。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from local.bridge import protocol as P
from local.bridge.autopilot import Autopilot, AutopilotError, Decision, SessionLog


def frame_bytes(tag: int) -> bytes:
    """中身は問わない（decider が差し替え可能なので）。"""
    return b"FRAME" + bytes([tag])


def make_ap(decider=None, **kw) -> Autopilot:
    def default(frame: P.Frame) -> Decision:
        return Decision(index=0, label="cand-0", prob=0.9, candidates=[[0, "cand-0", 0.9]], value=1.5)

    ap = Autopilot(decider or default, character="DEFECT", run_id="test-run",
                   log=lambda *_: None, **kw)
    ap.start()
    return ap


class HandshakeTest(unittest.TestCase):
    def test_welcome_carries_obs_format(self):
        ap = make_ap(obs_format=19)
        msgs = [P.welcome(ap.run_id, 1, "DEFECT", 1.0, 19)]
        self.assertEqual(msgs[0]["obs_format"], 19)
        self.assertEqual(ap.state, "running")

    def test_protocol_mismatch_stops(self):
        ap = make_ap()
        replies = ap.handle(P.hello("game", 99, "v0.111.1", "DEFECT"))
        self.assertEqual(replies[0]["t"], P.S_ERROR)
        self.assertEqual(ap.state, "error")

    def test_hello_logs_client(self):
        ap = make_ap()
        self.assertEqual(ap.handle(P.hello("game", 1, "v0.111.1", "DEFECT", "mod", "0.1")), [])
        self.assertEqual(ap.hello.game_version, "v0.111.1")
        self.assertEqual(ap.summary()["hello"]["mod"], "mod")


class ActionTest(unittest.TestCase):
    def test_state_gets_action(self):
        ap = make_ap()
        replies = ap.handle(P.state(frame_bytes(1), 1, floor=3))
        self.assertEqual(len(replies), 1)
        self.assertEqual(replies[0]["t"], P.S_ACTION)
        self.assertEqual(replies[0]["seq"], 1)
        self.assertEqual(replies[0]["index"], 0)
        self.assertEqual(ap.stats.actions, 1)

    def test_same_seq_is_not_recomputed(self):
        calls = []

        def decider(frame):
            calls.append(frame.seq)
            return Decision(index=len(calls) - 1)

        ap = make_ap(decider)
        a1 = ap.handle(P.state(frame_bytes(1), 1))[0]
        a2 = ap.handle(P.state(frame_bytes(1), 1))[0]
        self.assertEqual(a1, a2)
        self.assertEqual(calls, [1])
        self.assertEqual(ap.stats.duplicates, 1)

    def test_same_frame_new_seq_reuses_the_answer(self):
        calls = []

        def decider(frame):
            calls.append(frame.seq)
            return Decision(index=2)

        ap = make_ap(decider)
        ap.handle(P.state(frame_bytes(1), 1))
        again = ap.handle(P.state(frame_bytes(1), 2))[0]
        self.assertEqual(again["seq"], 2)
        self.assertEqual(again["index"], 2)
        self.assertEqual(calls, [1])                 # 推論は 1 回だけ
        self.assertEqual(ap.stats.duplicates, 1)

    def test_different_frames_are_decided(self):
        ap = make_ap(lambda f: Decision(index=0 if f.seq % 2 else 1))
        self.assertEqual(ap.handle(P.state(frame_bytes(1), 1))[0]["index"], 0)
        self.assertEqual(ap.handle(P.state(frame_bytes(2), 2))[0]["index"], 1)

    def test_bad_frame_is_an_error(self):
        ap = make_ap()
        replies = ap.handle({"t": "state", "seq": 1, "frame": "!!!!"})
        self.assertEqual(replies[0]["t"], P.S_ERROR)
        self.assertEqual(ap.state, "error")

    def test_decider_exception_stops(self):
        def boom(frame):
            raise RuntimeError("model exploded")

        ap = make_ap(boom)
        replies = ap.handle(P.state(frame_bytes(1), 1))
        self.assertEqual(replies[0]["t"], P.S_ERROR)
        self.assertEqual(replies[1]["state"], P.CONTROL_STOPPED)
        self.assertEqual(ap.state, "error")
        self.assertIn("model exploded", ap.stopped_reason)

    def test_ignores_states_after_stop(self):
        ap = make_ap()
        ap.stop("operator")
        self.assertEqual(ap.handle(P.state(frame_bytes(1), 1)), [])


class PauseTest(unittest.TestCase):
    def test_paused_states_get_no_action(self):
        ap = make_ap()
        self.assertEqual(ap.pause(), [P.control(P.CONTROL_PAUSED, "operator")])
        self.assertEqual(ap.handle(P.state(frame_bytes(1), 1)), [])
        self.assertEqual(ap.stats.actions, 0)

    def test_resume_then_action(self):
        ap = make_ap()
        ap.pause()
        ap.handle(P.state(frame_bytes(1), 1))
        ap.resume()
        replies = ap.handle(P.state(frame_bytes(1), 1))    # MOD が同じ状態を送り直す
        self.assertEqual(replies[0]["t"], P.S_ACTION)

    def test_pause_counted(self):
        ap = make_ap()
        ap.pause()
        ap.pause()
        self.assertEqual(ap.stats.pauses, 1)


class SpeedTest(unittest.TestCase):
    def test_speed_scales_hold(self):
        ap = make_ap(hold_ms=200)
        self.assertEqual(ap.hold_ms(), 200)
        ap.set_speed(2.0)
        self.assertEqual(ap.hold_ms(), 100)
        replies = ap.handle(P.state(frame_bytes(1), 1))
        self.assertEqual(replies[0]["hold_ms"], 100)
        ap.set_speed(0.5)
        self.assertEqual(ap.hold_ms(), 400)

    def test_speed_bounds(self):
        ap = make_ap()
        for bad in (0, -1, 100):
            with self.assertRaises(AutopilotError):
                ap.set_speed(bad)


class StopTest(unittest.TestCase):
    def test_done_records_result(self):
        ap = make_ap()
        ap.handle(P.done(True, 52, 3))
        self.assertEqual(ap.state, "stopped")
        self.assertEqual(ap.stats.result["win"], True)
        self.assertEqual(ap.summary()["result"]["floor"], 52)

    def test_stop_sends_control_once(self):
        ap = make_ap()
        self.assertEqual(ap.stop("operator"), [P.control(P.CONTROL_STOPPED, "operator")])
        self.assertEqual(ap.stop("again"), [])

    def test_bye(self):
        ap = make_ap()
        self.assertEqual(ap.handle({"t": "bye", "reason": "game closed"}), [])
        self.assertEqual(ap.state, "stopped")

    def test_max_states(self):
        ap = make_ap(max_states=2)
        ap.handle(P.state(frame_bytes(1), 1))
        ap.handle(P.state(frame_bytes(2), 2))
        replies = ap.handle(P.state(frame_bytes(3), 3))
        self.assertEqual(replies[0]["state"], P.CONTROL_STOPPED)
        self.assertEqual(ap.stopped_reason, "max-states")

    def test_unknown_message_is_ignored(self):
        ap = make_ap()
        self.assertEqual(ap.handle({"t": "nonsense"}), [])


class LogTest(unittest.TestCase):
    def test_jsonl_written(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "logs" / "ap.jsonl"
            log = SessionLog(path, echo=False)
            ap = make_ap()
            ap.log = log
            ap.handle(P.state(frame_bytes(1), 1))
            ap.stop("operator")
            log.close()
            lines = [__import__("json").loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
            events = [x["event"] for x in lines]
            self.assertIn("action", events)
            self.assertIn("stop", events)
            action = next(x for x in lines if x["event"] == "action")
            self.assertEqual(action["index"], 0)
            self.assertIn("decide_ms", action)

    def test_summary_shape(self):
        ap = make_ap()
        ap.handle(P.state(frame_bytes(1), 1))
        s = ap.summary()
        for key in ("run", "character", "state", "speed", "states", "actions", "duplicates", "seconds"):
            self.assertIn(key, s)
        self.assertEqual(s["states"], 1)


if __name__ == "__main__":
    unittest.main()
