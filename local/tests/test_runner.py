"""実行コマンドの組み立てと結果の読み取りのテスト（ゲームもモデルも使わない）。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from local import runner
from local import weights as W
from local.errors import LocalError
from local.recipes import recipe
from local.versions import line_for


class PlanTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.python = self.root / "venv" / "bin" / "python"
        self.python.parent.mkdir(parents=True)
        self.python.write_text("#!/bin/sh\n")
        self.ckpt = self.root / "w.pt"
        self.ckpt.write_bytes(b"x")
        self.game_dir = self.root / "game"
        self.game_dir.mkdir()

    def _plan(self, line_key="v0.111.0", character="DEFECT", **kw):
        line = line_for(line_key)
        asset = W.ASSETS[character]
        rec = recipe(line, character, obs_format=asset.obs_format)
        spec = runner.RunSpec(character=character, seeds=kw.pop("seeds", ["ABC123"]),
                              out=self.root / "out", **kw)
        return runner.plan(line, spec, rec, self.python, self.ckpt, self.game_dir)

    def test_greedy_uses_ability_test_and_seed_file(self):
        plan = self._plan()
        argv = [str(x) for x in plan.argv]
        self.assertIn("tools/ability_test.py", " ".join(argv))
        self.assertIn("--seeds-file", argv)
        self.assertIn("--asc", argv)
        self.assertEqual(argv[argv.index("--asc") + 1], "10-10")
        self.assertEqual(argv[argv.index("--character") + 1], "DEFECT")
        self.assertEqual(argv[argv.index("--runs") + 1], "1")
        self.assertEqual(argv[argv.index("--device") + 1], "cpu")
        self.assertNotIn("--req-extra", argv)                 # Defect は既定の形式と同じ
        self.assertEqual(plan.env["STS2_GAME_DIR"], str(self.game_dir))
        seeds = json.loads(plan.seeds_file.read_text(encoding="utf-8"))
        self.assertEqual(seeds["seeds"], ["ABC123"])

    def test_v0113_characters_get_their_obs_format(self):
        plan = self._plan("v0.113.1", "IRONCLAD")
        argv = [str(x) for x in plan.argv]
        payload = json.loads(argv[argv.index("--req-extra") + 1])
        self.assertEqual(payload["obs_format"], 28)
        self.assertTrue(payload["loop_cut"])
        self.assertEqual(plan.cwd, line_for("v0.113.1").root)

    def test_search_uses_tree_search_tool(self):
        plan = self._plan("v0.113.1", "FOOL", search=True, sims=64, procs=3)
        joined = " ".join(str(x) for x in plan.argv)
        self.assertIn("tools/future/run_search.py", joined)
        argv = [str(x) for x in plan.argv]
        self.assertEqual(argv[argv.index("--sims") + 1], "64")
        self.assertEqual(argv[argv.index("--procs") + 1], "3")
        self.assertEqual(argv[argv.index("--device") + 1], "cpu")
        self.assertIn("Boss,Elite", argv[argv.index("--rooms") + 1])

    def test_search_outside_a10_is_refused(self):
        with self.assertRaises(LocalError):
            self._plan("v0.113.1", "FOOL", search=True, ascension=5)

    def test_multiple_seeds(self):
        plan = self._plan(seeds=["A", "B", "C"], envs=3)
        argv = [str(x) for x in plan.argv]
        self.assertEqual(argv[argv.index("--runs") + 1], "3")
        self.assertEqual(argv[argv.index("--envs") + 1], "3")

    def test_display_shows_cwd_and_command(self):
        text = self._plan().display()
        self.assertIn("cwd:", text)
        self.assertIn("cmd:", text)
        self.assertIn("STS2_GAME_DIR=", text)

    def test_missing_python_is_setup_required(self):
        spec = runner.RunSpec(character="DEFECT", seeds=["A"], out=self.root / "o")
        with self.assertRaises(Exception) as ctx:
            runner.plan(line_for("v0.111.0"), spec, recipe(line_for("v0.111.0"), "DEFECT"),
                        self.root / "nope" / "python", self.ckpt, self.game_dir)
        self.assertIn("setup", str(ctx.exception))


class CollectAndDescribeTest(unittest.TestCase):
    def test_collect_and_describe_win(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            row = {"seed": "ABC", "win": True, "floor": 49, "act": 2, "combats": [{"room": "Boss", "act": 2, "floor": 49, "encounter": "AEONGLASS_BOSS"}], "hp_delta": 0.2}
            (out / "runs_s0.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            rows = runner.collect(out)
            self.assertEqual(len(rows), 1)
            text = "\n".join(runner.describe(rows[0]))
            self.assertIn("勝利", text)
            self.assertIn("49", text)
            self.assertIn("第 3 幕", text)          # エンジンの act は 0 始まり
            self.assertIn("AEONGLASS_BOSS", text)

    def test_describe_loss_and_stuck(self):
        loss = {"seed": "A", "win": False, "floor": 17, "act": 1, "combats": []}
        text = "\n".join(runner.describe(loss))
        self.assertIn("敗北", text)
        self.assertIn("第 2 幕", text)
        stuck = {"seed": "B", "win": False, "floor": 9, "act": 1, "combats": [], "stuck": True}
        self.assertIn("中断", "\n".join(runner.describe(stuck)))

    def test_describe_search_run(self):
        row = {"seed": "C", "win": False, "floor": 40, "act": 3, "combats": [], "searches": 12, "changed": 3, "s": 41.5}
        text = "\n".join(runner.describe(row))
        self.assertIn("探索: 12 回", text)
        self.assertIn("41.5", text)

    def test_collect_deduplicates_by_seed_keeping_last(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            (out / "runs_s0.jsonl").write_text(json.dumps({"seed": "A", "win": False, "floor": 1, "act": 0}) + "\n", encoding="utf-8")
            (out / "runs_s1.jsonl").write_text(json.dumps({"seed": "A", "win": True, "floor": 52, "act": 3}) + "\n", encoding="utf-8")
            rows = runner.collect(out)
            self.assertEqual(len(rows), 1)
            self.assertTrue(rows[0]["win"])

    def test_pick(self):
        outcome = runner.RunOutcome(plan=None, seconds=1.0, returncode=0, results=[{"seed": "A"}, {"seed": "B"}])
        self.assertEqual(runner.pick(outcome, "B")["seed"], "B")
        self.assertIsNone(runner.pick(outcome, "C"))


if __name__ == "__main__":
    unittest.main()
