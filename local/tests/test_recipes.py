"""論文と同じ条件で走らせるための設定（観測形式・追加キー）のテスト。"""

from __future__ import annotations

import json
import unittest

from local import recipes
from local.versions import line_for

V0111 = line_for("v0.111.0")
V0113 = line_for("v0.113.1")


class DefaultFormatTest(unittest.TestCase):
    def test_table(self):
        self.assertEqual(recipes.default_format(V0111, "DEFECT"), 19)
        self.assertEqual(recipes.default_format(V0111, "SILENT"), 23)
        self.assertEqual(recipes.default_format(V0113, "IRONCLAD"), 29)
        self.assertEqual(recipes.default_format(V0113, "NECROBINDER"), 28)
        self.assertEqual(recipes.default_format(V0113, "FOOL"), 27)

    def test_matches_tree_protocol(self):
        import unittest as _u

        try:
            import numpy  # noqa: F401
        except ImportError:
            raise _u.SkipTest("numpy が無い（.venv/bin/python で実行すると照合できる）")
        """テーブルが実際の protocol.format_for と一致していること（値を写し間違えないため）。"""
        import importlib.util
        import sys

        for line in (V0111, V0113):
            path = line.root / "python" / "sts2env" / "protocol.py"
            spec = importlib.util.spec_from_file_location(f"protocol_{line.key}", path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)
            for c in line.characters:
                self.assertEqual(mod.format_for(c), recipes.default_format(line, c), f"{line.key}/{c}")


class RecipeTest(unittest.TestCase):
    def test_obs_format_is_added_only_when_it_differs(self):
        r = recipes.recipe(V0113, "IRONCLAD", obs_format=28)
        self.assertEqual(r.extra["obs_format"], 28)
        r = recipes.recipe(V0113, "IRONCLAD", obs_format=29)
        self.assertNotIn("obs_format", r.extra)
        r = recipes.recipe(V0111, "DEFECT", obs_format=19)
        self.assertEqual(r.extra, {})

    def test_paper_flags_for_ironclad(self):
        r = recipes.recipe(V0113, "IRONCLAD", obs_format=28)
        self.assertTrue(r.extra["loop_cut"])
        self.assertEqual(r.extra["stall_turns"], 20)

    def test_req_extra_json(self):
        r = recipes.recipe(V0113, "NECROBINDER", obs_format=28)
        self.assertIsNone(recipes.req_extra_json(r))          # 既定 28 と同じなので何も足さない
        r = recipes.recipe(V0113, "IRONCLAD", obs_format=28)
        payload = json.loads(recipes.req_extra_json(r))
        self.assertEqual(payload["obs_format"], 28)
        self.assertTrue(payload["loop_cut"])

    def test_caller_can_add_keys(self):
        r = recipes.recipe(V0111, "DEFECT", obs_format=19, extra={"stall_turns": 30})
        self.assertEqual(r.extra, {"stall_turns": 30})


if __name__ == "__main__":
    unittest.main()
