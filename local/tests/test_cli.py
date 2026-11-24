"""CLI の入口（引数の解釈と終了コード）のテスト。外部のものは使わない。"""

from __future__ import annotations

import contextlib
import io
import unittest

from local import cli


def run(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(list(argv))
    return code, out.getvalue(), err.getvalue()


class CliTest(unittest.TestCase):
    def test_versions(self):
        code, out, _ = run("versions")
        self.assertEqual(code, 0)
        self.assertIn("v0.113.1", out)
        self.assertIn("v0.111.0", out)
        self.assertIn("v0.111.1", out)

    def test_help_exits_zero(self):
        with self.assertRaises(SystemExit) as ctx:
            cli.main(["--help"])
        self.assertEqual(ctx.exception.code, 0)

    def test_unknown_command_exits_two(self):
        with self.assertRaises(SystemExit) as ctx:
            cli.main(["nope"])
        self.assertEqual(ctx.exception.code, 2)

    def test_bad_ascension_is_rejected_before_work(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as ctx:
            cli.main(["play", "--asc", "not-a-number", "--dry-run"])
        self.assertEqual(ctx.exception.code, 2)
        self.assertIn("--asc", err.getvalue())


if __name__ == "__main__":
    unittest.main()
