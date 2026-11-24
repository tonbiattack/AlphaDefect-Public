"""ゲーム資産の用意（複製・ローカライズ抽出・スタンプ）のテスト。"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from local import gamefiles, paths, steam

FAKE_EXTRACT = '''
import sys
from pathlib import Path
pck, out = Path(sys.argv[1]), Path(sys.argv[2])
for rel in ("localization/eng/cards.json", "localization/zhs/cards.json", "localization/completion.json"):
    p = out / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{}", encoding="utf-8")
print("ok")
'''


class PrepareTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.old_home = os.environ.get("SLS2_HOME")
        os.environ["SLS2_HOME"] = str(self.root / "cache")
        self.addCleanup(self._restore)

        game = self.root / "game"
        (game / "data_sts2_windows_x86_64").mkdir(parents=True)
        (game / "data_sts2_windows_x86_64" / "sts2.dll").write_bytes(b"x" * 10)
        (game / "data_sts2_windows_x86_64" / "GodotSharp.dll").write_bytes(b"y")
        (game / "SlayTheSpire2.pck").write_bytes(b"pck")
        (game / "release_info.json").write_text(
            json.dumps({"version": "v0.111.1", "branch": "v0.111.1", "commit": "abc", "main_assembly_hash": 1}), encoding="utf-8")
        self.install = steam.read_install(game)

        self.tree = self.root / "tree"
        (self.tree / "tools").mkdir(parents=True)
        (self.tree / "tools" / "extract_pck.py").write_text(FAKE_EXTRACT, encoding="utf-8")

    def _restore(self):
        if self.old_home is None:
            os.environ.pop("SLS2_HOME", None)
        else:
            os.environ["SLS2_HOME"] = self.old_home

    def test_prepare_copies_and_extracts(self):
        res = gamefiles.prepare(self.install, self.tree, log=lambda *_: None)
        self.assertTrue(res.changed)
        self.assertEqual(res.copied_files, 2)
        target = paths.game_dir("v0.111.1")
        self.assertTrue((target / "sts2.dll").is_file())
        self.assertTrue((target / "res" / "localization" / "eng").is_dir())
        self.assertTrue((target / gamefiles.STAMP_NAME).is_file())
        self.assertTrue(gamefiles.is_prepared(self.install))

    def test_second_run_is_cached(self):
        gamefiles.prepare(self.install, self.tree, log=lambda *_: None)
        res = gamefiles.prepare(self.install, self.tree, log=lambda *_: None)
        self.assertFalse(res.changed)

    def test_force_redoes_it(self):
        gamefiles.prepare(self.install, self.tree, log=lambda *_: None)
        res = gamefiles.prepare(self.install, self.tree, force=True, log=lambda *_: None)
        self.assertTrue(res.changed)

    def test_changed_pck_invalidates_the_stamp(self):
        gamefiles.prepare(self.install, self.tree, log=lambda *_: None)
        self.install.pck.write_bytes(b"pck2")
        self.assertFalse(gamefiles.is_prepared(self.install))

    def test_missing_localization_marks_unprepared(self):
        gamefiles.prepare(self.install, self.tree, log=lambda *_: None)
        target = paths.game_dir("v0.111.1")
        for p in (target / "res").rglob("*"):
            if p.is_file():
                p.unlink()
        self.assertFalse(gamefiles.is_prepared(self.install))

    def test_broken_extract_script_is_reported(self):
        (self.tree / "tools" / "extract_pck.py").write_text("raise SystemExit(3)\n", encoding="utf-8")
        with self.assertRaises(Exception) as ctx:
            gamefiles.prepare(self.install, self.tree, log=lambda *_: None)
        self.assertIn("ローカライズ", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
