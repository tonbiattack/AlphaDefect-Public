"""ブリッジのプロトコル（1 行 1 JSON）のテスト。"""

from __future__ import annotations

import json
import unittest

from local.bridge import protocol as P


class CodecTest(unittest.TestCase):
    def test_roundtrip(self):
        msg = {"t": P.C_STATE, "seq": 7, "label": "雷撃＋", "frame": "AAAA"}
        self.assertEqual(P.decode(P.encode(msg)), msg)

    def test_encode_is_one_line(self):
        data = P.encode({"t": "log", "msg": "改行\n入り"})
        self.assertEqual(data.count(b"\n"), 1)
        self.assertTrue(data.endswith(b"\n"))

    def test_bad_json(self):
        with self.assertRaises(P.ProtocolError):
            P.decode(b"{not json}")

    def test_non_object(self):
        with self.assertRaises(P.ProtocolError):
            P.decode(b"[1,2,3]")

    def test_oversized_line(self):
        with self.assertRaises(P.ProtocolError):
            P.decode(b"x" * (P.MAX_LINE_BYTES + 1))

    def test_invalid_utf8(self):
        with self.assertRaises(P.ProtocolError):
            P.decode(b"\xff\xfe")


class FrameTest(unittest.TestCase):
    def test_base64_roundtrip(self):
        raw = bytes(range(256)) * 3
        self.assertEqual(P.unpack_frame(P.pack_frame(raw)), raw)

    def test_invalid_base64(self):
        with self.assertRaises(P.ProtocolError):
            P.unpack_frame("!!!not base64!!!")

    def test_digest_is_stable_and_short(self):
        a = P.Frame(seq=1, frame=b"\x00\x01")
        b = P.Frame(seq=2, frame=b"\x00\x01")
        c = P.Frame(seq=3, frame=b"\x00\x02")
        self.assertEqual(a.digest, b.digest)
        self.assertNotEqual(a.digest, c.digest)
        self.assertEqual(len(a.digest), 16)


class HelloTest(unittest.TestCase):
    def test_parse(self):
        msg = P.hello("game", 1, "v0.111.1", "defect", "alphadefect.autopilot", "0.1.0")
        hello = P.Hello.parse(msg)
        self.assertEqual(hello.protocol, 1)
        self.assertEqual(hello.character, "DEFECT")
        self.assertEqual(hello.game_version, "v0.111.1")
        self.assertEqual(hello.mod_id, "alphadefect.autopilot")

    def test_missing_fields(self):
        with self.assertRaises(P.ProtocolError):
            P.Hello.parse({"t": "hello"})


class BuildersTest(unittest.TestCase):
    def test_state_and_action(self):
        s = P.state(b"\x01\x02", 3, floor=5)
        self.assertEqual(s["t"], "state")
        self.assertEqual(s["seq"], 3)
        self.assertEqual(s["meta"], {"floor": 5})
        a = P.action(3, 1, "打 ZAP", 0.62, [[1, "打 ZAP", 0.62]], hold_ms=120)
        self.assertEqual(json.loads(json.dumps(a))["index"], 1)
        self.assertEqual(a["hold_ms"], 120)

    def test_welcome_has_obs_format(self):
        self.assertEqual(P.welcome("run", 1, "DEFECT", 1.0, 19)["obs_format"], 19)
        self.assertNotIn("obs_format", P.welcome("run", 1, "DEFECT", 1.0))

    def test_control_and_error(self):
        self.assertEqual(P.control(P.CONTROL_PAUSED, "operator")["state"], "paused")
        self.assertTrue(P.error("boom")["fatal"])


if __name__ == "__main__":
    unittest.main()
