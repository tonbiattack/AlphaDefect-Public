"""Mentor モードの開始設定（#41）のテスト。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from local import mentorcfg as M
from local.errors import LocalError

VERSION = "v0.111.1"


class AvailableTest(unittest.TestCase):
    def test_v0111_lists_three_chars(self):
        chars = [c for c, _ in M.available_characters(VERSION)]
        self.assertEqual(chars, ["DEFECT", "REGENT", "SILENT"])

    def test_states_come_from_versions_table(self):
        states = dict(M.available_characters(VERSION))
        self.assertEqual(states["DEFECT"], "validated")


class ResolveTest(unittest.TestCase):
    def test_auto_keeps_auto(self):
        choice = M.resolve(VERSION, "auto", 0, None)
        self.assertEqual(choice.character, "auto")
        self.assertIsNone(choice.seed)

    def test_named_character(self):
        choice = M.resolve(VERSION, "silent", 3, "42")
        self.assertEqual(choice.character, "SILENT")
        self.assertEqual(choice.ascension, 3)
        self.assertEqual(choice.seed, 42)

    def test_random_seed(self):
        for seed in ("", None, "random", "RANDOM"):
            self.assertIsNone(M.resolve(VERSION, "DEFECT", 0, seed).seed)

    def test_unsupported_character_is_refused(self):
        with self.assertRaises(LocalError) as ctx:
            M.resolve(VERSION, "FOOL", 0, None)
        self.assertIn("FOOL", str(ctx.exception))
        self.assertIn("DEFECT", str(ctx.exception))

    def test_unknown_name_is_refused(self):
        with self.assertRaises(LocalError):
            M.resolve(VERSION, "BANANA", 0, None)

    def test_untested_character_needs_the_flag(self):
        # v0.113.1 の REGENT は untested
        with self.assertRaises(LocalError):
            M.resolve("v0.113.1", "REGENT", 0, None)
        choice = M.resolve("v0.113.1", "REGENT", 0, None, allow_untested=True)
        self.assertEqual(choice.character, "REGENT")
        self.assertIn("未検証", choice.note)

    def test_ascension_bounds(self):
        M.resolve(VERSION, "DEFECT", 10, None)
        for bad in (-1, 11):
            with self.assertRaises(LocalError):
                M.resolve(VERSION, "DEFECT", bad, None)

    def test_seed_must_be_integer(self):
        with self.assertRaises(LocalError):
            M.resolve(VERSION, "DEFECT", 0, "abc")
        with self.assertRaises(LocalError):
            M.resolve(VERSION, "DEFECT", 0, "-5")


class ChoicesTest(unittest.TestCase):
    def test_choices_payload_shape(self):
        # GUI が読む形（版・キャラ・状態・モード）を固定する
        chars = [{"id": c, "status": st} for c, st in M.available_characters(VERSION)]
        self.assertEqual(chars[0]["id"], "DEFECT")
        self.assertEqual(chars[0]["status"], "validated")
        self.assertTrue(all(set(c) == {"id", "status"} for c in chars))

    def test_choices_include_untested_for_v0114(self):
        chars = dict(M.available_characters("v0.114.0"))
        self.assertEqual(chars["FOOL"], "untested")


class FileTest(unittest.TestCase):
    def _install(self, root: Path):
        class Install:
            path = root

        return Install()

    def test_auto_order_lists_validated_characters(self):
        choice = M.resolve(VERSION, "auto", 0, None)
        self.assertEqual(choice.auto_order, ("DEFECT", "REGENT", "SILENT"))
        with tempfile.TemporaryDirectory() as tmp:
            install = self._install(Path(tmp))
            path = M.write(install, choice)
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data["auto_order"], ["DEFECT", "REGENT", "SILENT"])

    def test_write_read_clear(self):
        with tempfile.TemporaryDirectory() as tmp:
            install = self._install(Path(tmp))
            choice = M.resolve(VERSION, "SILENT", 5, "777")
            path = M.write(install, choice)
            self.assertTrue(path.is_file())
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data["character"], "SILENT")
            self.assertEqual(data["ascension"], 5)
            self.assertEqual(data["seed"], 777)
            self.assertIn("SILENT", M.describe(install))
            self.assertEqual(M.read(install)["seed"], 777)
            M.clear(install)
            self.assertIsNone(M.read(install))
            self.assertIn("未設定", M.describe(install))

    def test_describe_random(self):
        with tempfile.TemporaryDirectory() as tmp:
            install = self._install(Path(tmp))
            M.write(install, M.resolve(VERSION, "DEFECT", 0, None))
            self.assertIn("乱数", M.describe(install))


if __name__ == "__main__":
    unittest.main()
