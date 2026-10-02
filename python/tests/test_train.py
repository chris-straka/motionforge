# MIT (see LICENSE-MIT). Training dry-run tests (no torch needed).
"""End-to-end data path: manifest gate -> clip load -> shapes.

Run: python3 -m unittest discover -s python/tests -p "test_*.py".
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
FIX = os.path.join(REPO, "tests", "fixtures")
TRAIN = os.path.join(REPO, "python", "autopose", "train.py")


class TestDryRun(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        shutil.copy(os.path.join(FIX, "walk_src.json"),
                    os.path.join(self.tmp.name, "walk.json"))
        with open(os.path.join(self.tmp.name, "walk.json"), "rb") as f:
            self.sha = hashlib.sha256(f.read()).hexdigest()

    def tearDown(self):
        self.tmp.cleanup()

    def _manifest(self, **kw):
        row = {"id": "walk", "path": "walk.json", "source": "owner",
               "license": "owner-recorded", "split": "train", "sha256": self.sha}
        row.update(kw)
        path = os.path.join(self.tmp.name, "manifest.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"format": "motionforge-dataset", "version": 1, "clips": [row]}, f)
        return path

    def _run(self, *args):
        return subprocess.run(
            [sys.executable, TRAIN, *args],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120,
        )

    def test_dry_run_ok(self):
        proc = self._run("--manifest", self._manifest(), "--out-dir",
                         os.path.join(self.tmp.name, "out"), "--dry-run",
                         "--feet", "LeftFoot,RightFoot")
        self.assertEqual(proc.returncode, 0, proc.stdout)
        self.assertIn("train frames: 48", proc.stdout)
        self.assertIn("12 bones", proc.stdout)
        self.assertIn("DRY RUN OK", proc.stdout)

    def test_blocked_manifest_refuses(self):
        proc = self._run("--manifest", self._manifest(license="amass"),
                         "--out-dir", os.path.join(self.tmp.name, "out"), "--dry-run")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("MANIFEST REJECTED", proc.stdout)

    def test_missing_torch_errors_cleanly(self):
        # Full (non-dry) run without torch: clean message, no traceback.
        proc = self._run("--manifest", self._manifest(), "--out-dir",
                         os.path.join(self.tmp.name, "out"),
                         "--feet", "LeftFoot,RightFoot",
                         "--epochs", "1", "--batch-size", "48")
        if "torch is not installed" in proc.stdout:
            self.assertNotEqual(proc.returncode, 0)
            self.assertNotIn("Traceback", proc.stdout)
        else:
            # torch IS installed here: a tiny real training run must work.
            self.assertIn("GATE", proc.stdout)

    def test_unknown_feet_errors_cleanly(self):
        proc = self._run("--manifest", self._manifest(), "--out-dir",
                         os.path.join(self.tmp.name, "out"), "--dry-run",
                         "--feet", "Nope")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("unknown foot bone", proc.stdout)

    def test_limits_loads_on_dry_run(self):
        proc = self._run("--manifest", self._manifest(), "--out-dir",
                         os.path.join(self.tmp.name, "out"), "--dry-run",
                         "--feet", "LeftFoot,RightFoot",
                         "--limits", os.path.join(FIX, "limits_hero.json"))
        self.assertEqual(proc.returncode, 0, proc.stdout)
        self.assertIn("limits:", proc.stdout)
        self.assertIn("hll_hero", proc.stdout)

    def test_bad_limits_errors_cleanly(self):
        bad = os.path.join(self.tmp.name, "bad.json")
        with open(bad, "w", encoding="utf-8") as f:
            f.write('{"format": "motionforge-limits", "version": 1, '
                    '"bones": {"A": {"max_angle_deg": 999}}}')
        proc = self._run("--manifest", self._manifest(), "--out-dir",
                         os.path.join(self.tmp.name, "out"), "--dry-run",
                         "--limits", bad)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("bad --limits file", proc.stdout)
        self.assertNotIn("Traceback", proc.stdout)


if __name__ == "__main__":
    unittest.main()
