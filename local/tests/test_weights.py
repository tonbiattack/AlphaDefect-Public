"""重みの目録と SHA-256 検証のテスト（ネットワークは使わない）。"""

from __future__ import annotations

import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from local import paths
from local import weights as W
from local.errors import LocalError


class ManifestTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.expected = W.load_expected()

    def test_all_six_characters_resolve(self):
        self.assertEqual(set(W.ASSETS), {"DEFECT", "REGENT", "SILENT", "FOOL", "IRONCLAD", "NECROBINDER"})
        for c, asset in W.ASSETS.items():
            rec = W.expected_for(asset, self.expected)
            self.assertRegex(rec["sha256"], r"^[0-9a-f]{64}$", c)
            self.assertGreater(rec["bytes"], 1_000_000, c)

    def test_known_hashes(self):
        # weights/SHA256SUMS_core.txt と MANIFEST_v6.json に書かれている値
        self.assertTrue(self.expected["student_e_v165018.pt"]["sha256"].startswith("f51545673c57"))
        self.assertTrue(self.expected["ironclad_ema_v24782.pt"]["sha256"].startswith("78ba58312335"))

    def test_manifests_agree_on_obs_format(self):
        for c, asset in W.ASSETS.items():
            rec = self.expected.get(asset.file) or {}
            if rec.get("obs_format") is not None:
                self.assertEqual(rec["obs_format"], asset.obs_format, c)

    def test_manifest_entries_are_consistent(self):
        for name, rec in self.expected.items():
            if rec.get("sha256"):
                self.assertRegex(rec["sha256"], r"^[0-9a-f]{64}$", name)


class VerifyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.old = os.environ.get("SLS2_HOME")
        os.environ["SLS2_HOME"] = self.tmp.name
        self.addCleanup(self._restore_home)

    def _restore_home(self):
        if self.old is None:
            os.environ.pop("SLS2_HOME", None)
        else:
            os.environ["SLS2_HOME"] = self.old

    def _write(self, asset, data: bytes) -> Path:
        p = W.weight_path(asset)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return p

    def test_verify_ok(self):
        asset = W.ASSETS["DEFECT"]
        data = b"not really a checkpoint"
        p = self._write(asset, data)
        expected = {"student_e_v165018.pt": {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}}
        res = W.verify(asset, p, expected=expected)
        self.assertTrue(res.ok)
        self.assertTrue(res.size_ok)

    def test_verify_detects_corruption(self):
        asset = W.ASSETS["DEFECT"]
        p = self._write(asset, b"short")
        expected = {"student_e_v165018.pt": {"sha256": "0" * 64, "bytes": 5}}
        res = W.verify(asset, p, expected=expected)
        self.assertFalse(res.ok)
        self.assertTrue(res.size_ok)

    def test_verify_reports_size_mismatch(self):
        asset = W.ASSETS["DEFECT"]
        p = self._write(asset, b"1234")
        expected = {"student_e_v165018.pt": {"sha256": hashlib.sha256(b"1234").hexdigest(), "bytes": 99}}
        self.assertFalse(W.verify(asset, p, expected=expected).size_ok)

    def test_missing_file_is_an_error(self):
        with self.assertRaises(LocalError):
            W.verify(W.ASSETS["SILENT"], Path(self.tmp.name) / "nope.pt", expected={"silent_ema_v26428.pt": {"sha256": "0" * 64}})

    def test_status_does_not_download(self):
        rows = W.status(["DEFECT"])
        self.assertFalse(rows["DEFECT"]["present"])
        self.assertEqual(rows["DEFECT"]["release"], "weights-v1")

    def test_release_url(self):
        self.assertEqual(
            W.release_url("weights-v6", "ironclad_ema_v24782.pt"),
            "https://github.com/hmty-CHEN/AlphaDefect-Public/releases/download/weights-v6/ironclad_ema_v24782.pt",
        )
        with mock.patch.dict(os.environ, {"SLS2_RELEASE_REPO": "someone/fork"}):
            self.assertIn("someone/fork", W.release_url("weights-v6", "x.pt"))

    def test_weight_path_uses_cache_home(self):
        self.assertEqual(W.weight_path(W.ASSETS["FOOL"]), Path(self.tmp.name) / "weights" / "weights-v5" / "fool_ema_v24181.pt")
        self.assertEqual(paths.home(), Path(self.tmp.name))


if __name__ == "__main__":
    unittest.main()
