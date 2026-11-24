"""バージョン判定のテスト（ゲームもネットも使わない）。"""

from __future__ import annotations

import unittest

from local.errors import UnsupportedCharacter, UnsupportedVersion
from local.versions import (
    ALL_CHARACTERS,
    LINES,
    PATCH_COMPAT,
    character_status,
    is_patch_alias,
    line_for,
    normalize,
    parse,
    report,
    require_character,
    supported_versions,
)


class NormalizeTest(unittest.TestCase):
    def test_accepts_with_and_without_v(self):
        self.assertEqual(normalize("0.111.1"), "v0.111.1")
        self.assertEqual(normalize("v0.113.1"), "v0.113.1")
        self.assertEqual(normalize(" v0.113.1 "), "v0.113.1")

    def test_strips_suffix(self):
        self.assertEqual(normalize("v0.113.1-beta1"), "v0.113.1")
        self.assertEqual(normalize("0.111.0+build7"), "v0.111.0")

    def test_rejects_garbage(self):
        for bad in ("", "latest", "v0.1", "2026-11-13"):
            with self.assertRaises(UnsupportedVersion):
                normalize(bad)

    def test_parse(self):
        self.assertEqual(parse("v0.111.1"), (0, 111, 1))


class LineTest(unittest.TestCase):
    def test_supported_lines(self):
        self.assertEqual(line_for("v0.111.0").tree, "game-v0.111.0")
        self.assertEqual(line_for("v0.113.1").tree, "")

    def test_patch_alias_uses_base_tree(self):
        line = line_for("v0.111.1")
        self.assertEqual(line.tree, "game-v0.111.0")
        self.assertTrue(is_patch_alias("v0.111.1"))
        self.assertFalse(is_patch_alias("v0.113.1"))

    def test_unknown_version_is_rejected_with_guidance(self):
        with self.assertRaises(UnsupportedVersion) as ctx:
            line_for("v0.112.3")
        msg = str(ctx.exception)
        self.assertIn("v0.113.1", msg)
        self.assertIn("v0.111.0", msg)
        self.assertIn("status", msg)

    def test_supported_versions(self):
        self.assertIn("v0.113.1", supported_versions())
        self.assertTrue(set(PATCH_COMPAT) <= set(supported_versions()))

    def test_every_line_has_a_greedy_tool(self):
        for key, line in LINES.items():
            self.assertTrue((line.root / line.greedy_tool).is_file(), f"{key}: {line.greedy_tool}")
            if line.search:
                self.assertTrue((line.root / line.search_tool).is_file(), f"{key}: {line.search_tool}")


class CharacterTest(unittest.TestCase):
    def test_validated(self):
        self.assertEqual(character_status(line_for("v0.111.0"), "DEFECT"), "validated")
        self.assertEqual(character_status(line_for("v0.113.1"), "IRONCLAD"), "validated")

    def test_regent_untested_on_v0113(self):
        self.assertEqual(character_status(line_for("v0.113.1"), "REGENT"), "untested")
        with self.assertRaises(UnsupportedCharacter):
            require_character(line_for("v0.113.1"), "REGENT")
        require_character(line_for("v0.113.1"), "REGENT", allow_untested=True)

    def test_fool_not_in_v0111(self):
        self.assertEqual(character_status(line_for("v0.111.0"), "FOOL"), "unsupported")
        with self.assertRaises(UnsupportedCharacter) as ctx:
            require_character(line_for("v0.111.0"), "FOOL")
        self.assertIn("DEFECT", str(ctx.exception))

    def test_every_line_character_is_known(self):
        for line in LINES.values():
            for c in line.characters:
                self.assertIn(c, ALL_CHARACTERS)

    def test_report(self):
        r = report("v0.111.1")
        self.assertTrue(r.patch_alias)
        self.assertEqual(r.line.key, "v0.111.0")
        self.assertEqual(r.characters["DEFECT"], "validated")
        self.assertIn("line", r.to_dict())


if __name__ == "__main__":
    unittest.main()
