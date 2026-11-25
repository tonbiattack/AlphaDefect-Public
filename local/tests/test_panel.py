"""ゲーム内パネル（速度・停止・1手・手動・確認）とサーバ側の状態同期のテスト。"""

from __future__ import annotations

import unittest

from local.bridge import protocol as P
from local.bridge.autopilot import Autopilot, AutopilotError, Decision
from local.bridge.server import HELP, _Controls, _status_text


def quiet_log(*_args, **_kw):
    pass


def make_ap(**kw) -> Autopilot:
    def decider(frame) -> Decision:                      # 中身は問わない
        return Decision(index=0, label="cand-0", prob=0.9, candidates=[[0, "cand-0", 0.9]])

    return Autopilot(decider, character="DEFECT", run_id="test-run", log=quiet_log, **kw)


class FakeConn:
    def __init__(self):
        self.sent = []

    def send_many(self, messages) -> bool:
        self.sent.extend(messages)
        return True


class ProtocolTest(unittest.TestCase):
    def test_command_message(self):
        self.assertEqual(P.command("step"), {"t": "command", "name": "step"})
        self.assertEqual(P.command("pacing", 1500), {"t": "command", "name": "pacing", "value": "1500"})

    def test_panel_message_roundtrip(self):
        state = {"mode": "manual", "paused": True, "speed": 2.0}
        msg = P.panel(state)
        self.assertEqual(msg["t"], "panel")
        self.assertEqual(msg["state"], state)

    def test_panel_kind_is_known(self):
        self.assertEqual(P.C_PANEL, "panel")


class AutopilotPanelTest(unittest.TestCase):
    def test_step_sends_a_command(self):
        ap = make_ap()
        self.assertEqual(ap.step(), [P.command("step")])

    def test_mode_switch_and_remembering(self):
        ap = make_ap()
        self.assertEqual(ap.set_mode("manual"), [P.command("manual")])
        self.assertEqual(ap.set_mode("manual"), [P.command("manual")])   # 何度でも同じ
        with self.assertRaises(AutopilotError):
            ap.set_mode("wat")

    def test_entry_delay(self):
        ap = make_ap()
        self.assertEqual(ap.set_entry_delay(2000), [P.command("entry", 2000)])
        self.assertEqual(ap.set_entry_delay(-1), [P.command("entry", 0)])
        self.assertEqual(ap.set_entry_delay(99999), [P.command("entry", 8000)])

    def test_entry_is_replayed_on_reconnect(self):
        ap = make_ap()
        ap.set_entry_delay(2500)
        names = [m.get("name") for m in ap.start()]
        self.assertIn("entry", names)

    def test_confirm_and_pacing_and_stop_points(self):
        ap = make_ap()
        self.assertEqual(ap.set_confirm(True), [P.command("confirm-on")])
        self.assertEqual(ap.set_confirm(False), [P.command("confirm-off")])
        self.assertEqual(ap.set_pacing(1500), [P.command("pacing", 1500)])
        self.assertEqual(ap.set_pacing(-5), [P.command("pacing", 0)])
        self.assertEqual(ap.set_stop_points(combat_end=True),
                         [P.command("stopcombat", "on")])
        self.assertEqual(ap.set_stop_points(card_reward=False),
                         [P.command("stopreward", "off")])

    def test_panel_state_from_mod_is_stored_and_reported(self):
        ap = make_ap()
        state = {"mode": "auto", "paused": False, "speed": 1.0, "pacing_ms": 1000, "connected": True}
        msgs = ap.handle({"t": "panel", "state": state})
        self.assertEqual(msgs, [])
        self.assertEqual(ap.panel, state)
        self.assertEqual(ap.summary()["panel"], state)

    def test_reconnect_replays_the_operator_state(self):
        ap = make_ap()
        ap.set_mode("manual")
        ap.pause("operator")
        ap.set_confirm(True)
        ap.set_pacing(2000)
        ap.set_stop_points(combat_end=True)
        out = ap.start()
        kinds = [(m.get("t"), m.get("name"), m.get("state")) for m in out]
        self.assertIn(("command", "manual", None), kinds)
        self.assertIn(("command", "confirm-on", None), kinds)
        self.assertIn(("command", "pacing", None), kinds)
        self.assertIn(("command", "stopcombat", None), kinds)
        self.assertIn(("control", None, "paused"), kinds)     # 停止は welcome の後に必ず伝わる

    def test_resume_clears_the_remembered_pause(self):
        ap = make_ap()
        ap.pause("operator")
        ap.resume()
        out = ap.start()
        self.assertNotIn(("control", None, "paused"), [(m.get("t"), m.get("name"), m.get("state")) for m in out])

    def test_status_text_shows_the_panel(self):
        ap = make_ap()
        ap.handle({"t": "panel", "state": {"mode": "manual", "paused": True, "pacing_ms": 500,
                                           "entry_ms": 2000, "confirm": True}})
        text = _status_text(ap)
        self.assertIn("手動", text)
        self.assertIn("停止", text)
        self.assertIn("確認", text)
        self.assertIn("間隔500ms", text)
        self.assertIn("入場待ち2000ms", text)


class ControlsDispatchTest(unittest.TestCase):
    def setUp(self):
        self.ap = make_ap()
        self.conn = FakeConn()
        self.controls = _Controls(self.ap, stop_event=_Stop(), log=quiet_log)
        self.controls.attach(self.conn)

    def test_step_manual_auto_confirm_pacing(self):
        for line in ("step", "manual", "auto", "confirm on", "confirm off", "pacing 750",
                     "entry 2000", "stopcombat on", "stopreward off"):
            self.controls.apply(line)
        names = [m.get("name") for m in self.conn.sent]
        self.assertEqual(names, ["step", "manual", "auto", "confirm-on", "confirm-off", "pacing",
                                 "entry", "stopcombat", "stopreward"])

    def test_pause_and_resume_reach_the_mod(self):
        self.controls.apply("pause")
        self.controls.apply("resume")
        states = [m.get("state") for m in self.conn.sent if m.get("t") == "control"]
        self.assertEqual(states, ["paused", "running"])

    def test_unknown_command_only_warns(self):
        self.controls.apply("nonsense")
        self.assertEqual(self.conn.sent, [])

    def test_help_mentions_the_panel_commands(self):
        for word in ("step", "manual", "confirm", "pacing", "entry"):
            self.assertIn(word, HELP)


class _Stop:
    def __init__(self):
        self.flag = False

    def set(self):
        self.flag = True

    def is_set(self):
        return self.flag


if __name__ == "__main__":
    unittest.main()
