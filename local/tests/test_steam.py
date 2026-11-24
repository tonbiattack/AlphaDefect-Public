"""Steam の検出とゲーム資産の読み取りのテスト（実機のゲームは使わない）。"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from local import steam

VDF = """
"libraryfolders"
{
	"0"
	{
		"path"		"C:\\\\Program Files (x86)\\\\Steam"
		"label"		""
		"contentstatsid"		"-12345"
		"apps"
		{
			"228980"		"228980"
			"1012880"		"2994995211"
		}
	}
	"1"
	{
		"path"		"D:\\\\SteamLibrary"
		"label"		""
	}
}
"""


def make_install(root: Path, version="v0.111.1") -> Path:
    """ゲームのインストールを模したディレクトリ。"""
    (root / "data_sts2_windows_x86_64").mkdir(parents=True)
    (root / "data_sts2_windows_x86_64" / "sts2.dll").write_bytes(b"x")
    (root / "SlayTheSpire2.pck").write_bytes(b"pck")
    (root / "release_info.json").write_text(
        json.dumps({"commit": "cc1f9462", "version": version, "date": "2026-11-13", "branch": version, "main_assembly_hash": 28247836}),
        encoding="utf-8",
    )
    return root


class VdfTest(unittest.TestCase):
    def test_parse_library_folders(self):
        data = steam.parse_vdf(VDF)
        folders = data["libraryfolders"]
        self.assertEqual(folders["0"]["path"], "C:\\Program Files (x86)\\Steam")
        self.assertEqual(folders["1"]["path"], "D:\\SteamLibrary")
        self.assertEqual(folders["0"]["apps"]["1012880"], "2994995211")

    def test_parse_rejects_truncated(self):
        with self.assertRaises(ValueError):
            steam.parse_vdf('"a" { "b" "c"')

    def test_library_paths_include_root(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "steamapps").mkdir()
            (root / "steamapps" / "libraryfolders.vdf").write_text(VDF, encoding="utf-8")
            with mock.patch("local.steam.is_wsl", return_value=True):
                paths = steam.library_paths(root)
            self.assertEqual(paths[0], root)
            self.assertIn(Path("/mnt/d/SteamLibrary"), paths)
            self.assertIn(Path("/mnt/c/Program Files (x86)/Steam"), paths)


class PathConvertTest(unittest.TestCase):
    def test_windows_to_wsl(self):
        with mock.patch("local.steam.is_wsl", return_value=True):
            self.assertEqual(steam.windows_to_local("D:\\SteamLibrary"), Path("/mnt/d/SteamLibrary"))
            self.assertEqual(steam.windows_to_local("c:/Steam"), Path("/mnt/c/Steam"))

    def test_plain_path_is_untouched(self):
        with mock.patch("local.steam.is_wsl", return_value=True):
            self.assertEqual(steam.windows_to_local("/opt/steam"), Path("/opt/steam"))


class ReadInstallTest(unittest.TestCase):
    def test_reads_version_from_release_info(self):
        with tempfile.TemporaryDirectory() as td:
            inst = steam.read_install(make_install(Path(td) / "Slay the Spire 2"))
            self.assertEqual(inst.version, "v0.111.1")
            self.assertEqual(inst.branch, "v0.111.1")
            self.assertEqual(inst.commit, "cc1f9462")
            self.assertEqual(inst.assembly.name, "sts2.dll")
            self.assertEqual(inst.to_dict()["version"], "v0.111.1")

    def test_missing_pieces_fail_with_path(self):
        with tempfile.TemporaryDirectory() as td:
            root = make_install(Path(td) / "game")
            (root / "SlayTheSpire2.pck").unlink()
            with self.assertRaises(Exception) as ctx:
                steam.read_install(root)
            self.assertIn("SlayTheSpire2.pck", str(ctx.exception))

    def test_env_override_finds_install(self):
        with tempfile.TemporaryDirectory() as td:
            root = make_install(Path(td) / "game")
            with (
                mock.patch.dict(os.environ, {"SLS2_GAME_PATH": str(root)}),
                mock.patch("local.steam.steam_roots", return_value=[]),
            ):
                installs = steam.find_installs()
            self.assertEqual([i.path for i in installs], [root])

    def test_no_install_message_offers_override(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SLS2_GAME_PATH", None)
            with (
                mock.patch("local.steam.steam_roots", return_value=[]),
                mock.patch("local.steam.drive_roots", return_value=[]),
                self.assertRaises(Exception) as ctx,
            ):
                steam.find_install()
            self.assertIn("SLS2_GAME_PATH", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
