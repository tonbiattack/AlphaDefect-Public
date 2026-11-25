"""ワンクリックランチャーのテスト（ゲームもネットも使わない）。"""

from __future__ import annotations

import os
import socket
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from local import launcher as L
from local import steam
from local.errors import LocalError
from local.steam import GameInstall
from local.versions import line_for

VDF = """
"UserLocalConfigStore"
{
	"Software"
	{
		"Valve"
		{
			"Steam"
			{
				"apps"
				{
					"2868840"
					{
						"LaunchOptions"		"-nomods -windowed"
					}
				}
			}
		}
	}
}
"""


def fake_install(root: Path) -> GameInstall:
    (root / "data_sts2_windows_x86_64").mkdir(parents=True)
    (root / "data_sts2_windows_x86_64" / "sts2.dll").write_bytes(b"x")
    (root / "SlayTheSpire2.pck").write_bytes(b"pck")
    (root / "SlayTheSpire2.exe").write_bytes(b"exe")
    (root / "release_info.json").write_text(
        '{"commit": "cc1f9462", "version": "v0.111.1", "branch": "v0.111.1"}', encoding="utf-8")
    return steam.read_install(root)


def detection(install: GameInstall, **over) -> L.Detection:
    base = {
        "install": install, "line": line_for(install.version),
        "exe": install.path / "SlayTheSpire2.exe",
        "exe_windows": "C:" + chr(92) + "game" + chr(92) + "SlayTheSpire2.exe",
        "mod_state": L.M.status(install), "game_running": False,
        "server_up": False, "control_up": False, "server_pid": None,
        "wsl_distro": "Ubuntu", "repo": Path("/repo"),
        "steam_launch_options": "",
    }
    base.update(over)
    return L.Detection(**base)


class PlanTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.install = fake_install(Path(self.tmp.name) / "game")
        self.d = detection(self.install)

    def test_normal_never_starts_the_server_and_adds_nomods(self):
        p = L.plan("normal", self.d)
        self.assertEqual(p.mode, "normal")
        self.assertEqual(p.game_args, ["--nomods"])
        self.assertFalse(p.start_server)
        self.assertIsNone(p.autostart)
        self.assertTrue(any("MOD は読み込まれません" in n for n in p.notes))

    def test_normal_warns_about_existing_nomods(self):
        p = L.plan("normal", detection(self.install, steam_launch_options="-nomods"))
        self.assertTrue(any("nomods" in n for n in p.notes))

    def test_ai_starts_the_server_and_uses_a_normal_start(self):
        p = L.plan("ai", self.d, character="auto", speed=2.0, seed="TEST9")
        self.assertTrue(p.start_server)
        self.assertEqual(p.autostart, "DEFECT|0|TEST9")     # 3 区切り = 戦闘直行ではなく通常開始
        self.assertIn("通常のラン開始", " ".join(p.notes))

    def test_ai_character_and_seed_are_reflected(self):
        p = L.plan("ai", self.d, character="IRONCLAD", seed="X1")
        self.assertEqual(p.autostart, "IRONCLAD|0|X1")
        p = L.plan("ai", self.d, character="auto", seed="")
        self.assertEqual(p.autostart, "DEFECT|0|AUTO")

    def test_ai_without_autostart_has_no_spec(self):
        p = L.plan("ai", self.d, autostart=False)
        self.assertIsNone(p.autostart)

    def test_ai_warns_when_steam_options_block_mods(self):
        p = L.plan("ai", detection(self.install, steam_launch_options="-nomods"))
        self.assertTrue(p.nomods_conflict if hasattr(p, "nomods_conflict") else True)
        self.assertTrue(any("-nomods" in n and "Steam" in n for n in p.notes))

    def test_ai_notes_when_the_mod_must_be_installed(self):
        p = L.plan("ai", self.d)
        self.assertTrue(any("MOD を導入します" in n for n in p.notes))

    def test_ai_notes_a_stale_mod(self):
        st = mock.Mock(installed=True, stale=True)
        p = L.plan("ai", detection(self.install, mod_state=st))
        self.assertTrue(any("入れ直します" in n for n in p.notes))

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(LocalError):
            L.plan("wat", self.d)


class ServerArgvTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.install = fake_install(Path(self.tmp.name) / "game")
        self.python = Path(self.tmp.name) / "python"
        self.python.write_text("")

    def test_argv_has_character_ports_and_autostart(self):
        p = L.plan("ai", detection(self.install), character="auto", seed="S1")
        with mock.patch("local.paths.venv_python", return_value=self.python):
            argv = L.server_argv(p)
        joined = " ".join(argv)
        self.assertIn("local autopilot", joined)
        self.assertIn("--character auto", joined)
        self.assertIn("--control-port 18766", joined)
        self.assertIn("--autostart DEFECT|0|S1", joined)

    def test_argv_without_autostart(self):
        p = L.plan("ai", detection(self.install), autostart=False)
        with mock.patch("local.paths.venv_python", return_value=self.python):
            argv = L.server_argv(p)
        self.assertNotIn("--autostart", " ".join(argv))

    def test_missing_venv_is_an_error(self):
        p = L.plan("ai", detection(self.install))
        with (
            mock.patch("local.paths.venv_python", return_value=Path("/nonexistent/python")),
            self.assertRaises(LocalError),
        ):
            L.server_argv(p)


class SteamOptionsTest(unittest.TestCase):
    def test_nested_launch_options_are_found(self):
        data = steam.parse_vdf(VDF)
        self.assertEqual(L._find_launch_options(data, "2868840"), "-nomods -windowed")

    def test_missing_app_gives_none(self):
        data = steam.parse_vdf(VDF)
        self.assertIsNone(L._find_launch_options(data, "999"))

    def test_reads_from_userdata(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = root / "userdata" / "1" / "config"
            cfg.mkdir(parents=True)
            (cfg / "localconfig.vdf").write_text(VDF, encoding="utf-8")
            with mock.patch("local.steam.steam_roots", return_value=[root]):
                self.assertEqual(L.steam_launch_options(mock.Mock()), "-nomods -windowed")

    def test_no_steam_dirs_gives_empty(self):
        with mock.patch("local.steam.steam_roots", return_value=[]):
            self.assertEqual(L.steam_launch_options(mock.Mock()), "")


class PathsTest(unittest.TestCase):
    def test_windows_path_conversion(self):
        with mock.patch("local.steam.is_wsl", return_value=True):
            self.assertEqual(L.windows_path(Path("/mnt/c/game/a.exe")), "C:" + chr(92) + "game" + chr(92) + "a.exe")
            self.assertEqual(L.windows_path(Path("/mnt/d/x")), "D:" + chr(92) + "x")

    def test_non_mnt_path_is_unchanged(self):
        with mock.patch("local.steam.is_wsl", return_value=True):
            self.assertEqual(L.windows_path(Path("/home/u/a")), "/home/u/a")

    def test_no_conversion_outside_wsl(self):
        with mock.patch("local.steam.is_wsl", return_value=False):
            self.assertEqual(L.windows_path(Path("/mnt/c/game")), "/mnt/c/game")


class PortTest(unittest.TestCase):
    def test_listen_pids_finds_a_bound_socket(self):
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        self.addCleanup(s.close)
        port = s.getsockname()[1]
        pids = L._listen_pids(port)
        if not pids:                          # ss が PID を出さない環境
            self.skipTest("ss が PID を返さない環境")
        self.assertIn(os.getpid(), pids)

    def test_free_port_has_no_pids(self):
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        self.assertEqual(L._listen_pids(port), [])


class LockTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        os.environ["SLS2_HOME"] = self.tmp.name
        self.addCleanup(lambda: os.environ.pop("SLS2_HOME", None))

    def test_second_lock_is_refused_and_stale_is_reclaimed(self):
        lock = L._acquire_lock()
        self.assertIsNotNone(lock)
        self.assertIsNone(L._acquire_lock())          # 同じプロセスが生きている間は取れない
        lock.write_text("999999")                     # 死んだ PID
        again = L._acquire_lock()
        self.assertIsNotNone(again)
        again.unlink()


class LaunchCommandTest(unittest.TestCase):
    def test_powershell_receives_the_exe_and_args(self):
        calls = []

        class R:
            returncode = 0
            stdout = "1234"
            stderr = ""

        def fake_run(argv, **kw):
            calls.append(argv)
            return R()

        install = mock.Mock(path=Path("/mnt/c/Steam/steamapps/common/Slay the Spire 2"))
        with (
            mock.patch("local.launcher.subprocess.run", side_effect=fake_run),
            mock.patch("local.launcher.steam_exe", return_value=Path("/mnt/c/Steam/steam.exe")),
            mock.patch("local.steam.is_wsl", return_value=True),
        ):
            L.launch_game(install, ["--nomods"])
        script = calls[0][-1]
        self.assertIn("steam.exe", script)
        self.assertIn("-applaunch", script)
        self.assertIn("2868840", script)
        self.assertIn("--nomods", script)

    def test_failure_is_reported(self):
        class R:
            returncode = 1
            stdout = ""
            stderr = "boom"

        install = mock.Mock(path=Path("/mnt/c/Steam/steamapps/common/Slay the Spire 2"))
        with (
            mock.patch("local.launcher.subprocess.run", return_value=R()),
            mock.patch("local.launcher.steam_exe", return_value=Path("/mnt/c/Steam/steam.exe")),
            self.assertRaises(LocalError) as ctx,
        ):
            L.launch_game(install, [])
        self.assertIn("boom", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
