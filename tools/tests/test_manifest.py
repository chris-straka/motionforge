# MIT (see LICENSE-MIT). Manifest gate tests (stdlib only).
"""Run: python3 -m unittest discover -s tools/tests -p "test_*.py"."""

import hashlib
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import manifest  # noqa: E402  (tools/manifest.py)


class TestManifest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.clip_path = os.path.join(self.tmp.name, "clip.json")
        with open(self.clip_path, "w", encoding="utf-8") as f:
            f.write('{"hello": "mocap"}')
        with open(self.clip_path, "rb") as f:
            self.sha = hashlib.sha256(f.read()).hexdigest()

    def tearDown(self):
        self.tmp.cleanup()

    def _manifest(self, clips):
        path = os.path.join(self.tmp.name, "manifest.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"format": "motionforge-dataset", "version": 1, "clips": clips}, f)
        return path

    def _clip(self, **kw):
        row = {"id": "c1", "path": "clip.json", "source": "cmu",
               "license": "cmu-mocap",
               "terms_snapshot": "free for research+commercial (read 2026-10-01)",
               "split": "train", "sha256": self.sha}
        row.update(kw)
        return row

    def test_ok(self):
        ok, lines = manifest.check_manifest(self._manifest([self._clip()]))
        self.assertTrue(ok, lines)
        self.assertIn("OK", lines[0])

    def test_blocked_license(self):
        ok, lines = manifest.check_manifest(
            self._manifest([self._clip(license="amass")]))
        self.assertFalse(ok)
        self.assertIn("BLOCKED", lines[0])

    def test_unknown_license_rejected(self):
        ok, lines = manifest.check_manifest(
            self._manifest([self._clip(license="cc-by-nc")]))
        self.assertFalse(ok)

    def test_cmu_needs_snapshot(self):
        row = self._clip()
        del row["terms_snapshot"]
        ok, _ = manifest.check_manifest(self._manifest([row]))
        self.assertFalse(ok)

    def test_owner_recorded_ok(self):
        row = self._clip(license="owner-recorded")
        del row["terms_snapshot"]
        ok, _ = manifest.check_manifest(self._manifest([row]))
        self.assertTrue(ok)

    def test_mixamo_default_excluded(self):
        ok, _ = manifest.check_manifest(
            self._manifest([self._clip(license="mixamo-training")]))
        self.assertFalse(ok)
        ok, _ = manifest.check_manifest(
            self._manifest([self._clip(license="mixamo-training")]),
            allow_mixamo_training=True)
        self.assertTrue(ok)

    def test_sha_mismatch(self):
        ok, lines = manifest.check_manifest(
            self._manifest([self._clip(sha256="0" * 64)]))
        self.assertFalse(ok)
        self.assertIn("sha256", lines[0])

    def test_missing_file(self):
        ok, _ = manifest.check_manifest(self._manifest([self._clip(path="nope.json")]))
        self.assertFalse(ok)

    def test_duplicate_id(self):
        ok, _ = manifest.check_manifest(
            self._manifest([self._clip(), self._clip()]))
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
