# MIT (see LICENSE-MIT). Torch-free autopose tests.
"""Run: python3 -m unittest discover -s python/tests -p "test_*.py"."""

import json
import math
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from autopose import dataset, inference, limits  # noqa: E402

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
FIX = os.path.join(REPO, "tests", "fixtures")


class TestFk(unittest.TestCase):
    def test_fk_matches_blender_probe(self):
        # Same scene + numbers as motion_core's fk_matches_blender test:
        # root basis 90 deg about local Z -> child head (99.5, 0, 1.5).
        skel = dataset.Skeleton([
            ("Root", None, (100.0, 0.0, 1.0), (100.0, 0.0, 2.0)),
            ("Child", "Root", (100.5, 0.0, 1.5), (100.5, 1.0, 1.5)),
        ])
        q = (0.7071067811865476, 0.0, 0.0, 0.7071067811865476)
        heads = dataset.fk(skel, [((0, 0, 0), q), ((0, 0, 0), (1, 0, 0, 0))])
        self.assertLess(math.dist(heads[0], (100.0, 0.0, 1.0)), 1e-9)
        self.assertLess(math.dist(heads[1], (99.5, 0.0, 1.5)), 1e-9)

    def test_identity_is_rest(self):
        skel = dataset.Skeleton([
            ("Root", None, (0, 0, 1), (0, 0, 2)),
            ("Mid", "Root", (0, 0, 2), (0, 1, 2)),
        ])
        heads = dataset.fk(skel, [((0, 0, 0), (1, 0, 0, 0))] * 2)
        self.assertLess(math.dist(heads[0], (0, 0, 1)), 1e-12)
        self.assertLess(math.dist(heads[1], (0, 0, 2)), 1e-12)

    def test_load_walk_fixture(self):
        skel, frames = dataset.load_clip(os.path.join(FIX, "walk_src.json"))
        self.assertEqual(len(skel.names), 12)
        self.assertEqual(len(frames), 48)
        # Root travels forward (+Y) over the clip.
        z0 = dataset.fk(skel, frames[0])[0]
        z1 = dataset.fk(skel, frames[-1])[0]
        self.assertGreater(z1[1] - z0[1], 1.0)
        dataset.check_shared_skeleton([skel, skel])


class TestEffectors(unittest.TestCase):
    def test_sampling_reproducible(self):
        a = dataset.sample_effectors(random.Random(3), 12)
        b = dataset.sample_effectors(random.Random(3), 12)
        self.assertEqual(a, b)
        self.assertTrue(1 <= len(a) <= 6)
        self.assertEqual(sorted(a), a)

    def test_input_layout(self):
        skel = dataset.Skeleton([
            ("A", None, (0, 0, 0), (0, 1, 0)),
            ("B", "A", (0, 1, 0), (0, 2, 0)),
        ])
        heads = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)]
        x = dataset.build_input(skel, heads, [1])
        self.assertEqual(len(x), 5 * 2)
        self.assertEqual(x[0:4], [0.0, 0.0, 0.0, 0.0])
        self.assertEqual(x[4:8], [4.0, 5.0, 6.0, 1.0])
        self.assertEqual(x[8:], [1.0, 1.0])


class TestInference(unittest.TestCase):
    def test_python_matches_rust_golden(self):
        # Same weights + effectors as the Rust autopose golden: the
        # pure-Python forward pass must agree with the Rust CLI output.
        weights = inference.load_weights(os.path.join(FIX, "autopose_weights.json"))
        with open(os.path.join(FIX, "autopose_effectors.json"), encoding="utf-8") as f:
            req = json.load(f)
        bones = [(b["name"], b["parent"], tuple(b["head"]), tuple(b["tail"]))
                 for b in req["skeleton"]["bones"]]
        skel = dataset.Skeleton(bones)
        index = {n: i for i, n in enumerate(skel.names)}
        pos = {index[e["bone"]]: tuple(e["position"]) for e in req["effectors"]}
        heads = [pos.get(i, (0.0, 0.0, 0.0)) for i in range(len(skel.names))]
        x = dataset.build_input(skel, heads, list(pos))
        got = inference.forward(weights, x)
        with open(os.path.join(FIX, "golden", "autopose.0.json"), encoding="utf-8") as f:
            rust = json.load(f)
        frame = rust["frames"][0]
        for i, name in enumerate(skel.names):
            want = tuple(frame[name]["quat"])
            # Component-wise: the angle metric has sqrt-amplified noise
            # near zero (acos(1-eps)), hiding last-ULP agreement.
            for g, w in zip(got[i], want):
                self.assertLess(abs(g - w), 1e-12, name)

    def test_evaluate_runs(self):
        skel, frames = dataset.load_clip(os.path.join(FIX, "walk_src.json"))
        # Random weights: metrics must simply be finite and well-formed.
        rng = random.Random(11)
        n = len(skel.names)
        layers = [
            ([[rng.uniform(-0.1, 0.1) for _ in range(5 * n)] for _ in range(16)],
             [0.0] * 16),
            ([[rng.uniform(-0.1, 0.1) for _ in range(16)] for _ in range(4 * n)],
             [0.0] * (4 * n)),
        ]
        weights = inference.export_weights(skel.names, layers)
        rng2 = random.Random(5)
        sets = [dataset.sample_effectors(rng2, n) for _ in frames[:4]]
        rep = inference.evaluate(weights, skel, frames[:4], sets,
                                 foot_names=["LeftFoot", "RightFoot"])
        self.assertEqual(rep["frames"], 4)
        self.assertTrue(math.isfinite(rep["mean_joint_m"]))
        self.assertGreaterEqual(rep["max_joint_m"], rep["mean_joint_m"])

    def test_export_validates_shapes(self):
        with self.assertRaises(ValueError):
            inference.export_weights(["A"], [([[1.0]], [0.0])])

    def test_evaluate_with_limits(self):
        skel, frames = dataset.load_clip(os.path.join(FIX, "walk_src.json"))
        n = len(skel.names)
        # Dead weights + an 8-degree bias quat per bone: every frame
        # predicts exactly 8 deg on every bone, deterministically.
        half = math.radians(4.0)
        bias = [math.cos(half), math.sin(half), 0.0, 0.0] * n
        layers = [
            ([[0.0] * (5 * n) for _ in range(8)], [0.0] * 8),
            ([[0.0] * 8 for _ in range(4 * n)], bias),
        ]
        weights = inference.export_weights(skel.names, layers)
        lim = {"rig": "test", "bones": {"Hips": 10.0, "Nope": 45.0}}
        rep = inference.evaluate(weights, skel, frames[:2], [[0]], limits=lim)
        self.assertEqual(rep["limits_unmatched"], ["Nope"])
        self.assertEqual(rep["limit_violations"], [])
        lim2 = {"rig": "test", "bones": {"Hips": 5.0}}
        rep2 = inference.evaluate(weights, skel, frames[:2], [[0]], limits=lim2)
        self.assertEqual(len(rep2["limit_violations"]), 2)
        fi, bone, ang, mx = rep2["limit_violations"][0]
        self.assertEqual((fi, bone), (0, "Hips"))
        self.assertAlmostEqual(ang, 8.0, places=9)
        self.assertEqual(mx, 5.0)


class TestLimits(unittest.TestCase):
    def test_load_fixture(self):
        lim = limits.load_limits(os.path.join(FIX, "limits_hero.json"))
        self.assertEqual(lim["rig"], "hll_hero")
        self.assertEqual(len(lim["bones"]), 14)
        self.assertEqual(lim["bones"]["DEF-spine"], 45.0)

    def test_load_rejects_bad_files(self):
        import tempfile
        bad_docs = [
            {"format": "nope", "version": 1, "bones": {}},
            {"format": "motionforge-limits", "version": 2, "bones": {}},
            {"format": "motionforge-limits", "version": 1},
            {"format": "motionforge-limits", "version": 1, "bones": []},
            {"format": "motionforge-limits", "version": 1,
             "bones": {"a": {"max_angle_deg": 0}}},
            {"format": "motionforge-limits", "version": 1,
             "bones": {"a": {"max_angle_deg": 181}}},
        ]
        for doc in bad_docs:
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
                json.dump(doc, f)
                path = f.name
            try:
                with self.assertRaises(ValueError, msg=str(doc)):
                    limits.load_limits(path)
            finally:
                os.unlink(path)

    def test_python_matches_rust_limit_report(self):
        # Same golden prediction the CLI checks: both consumers must
        # report the same 8 violations in the same bone order.
        lim = limits.load_limits(os.path.join(FIX, "limits_hero.json"))
        with open(os.path.join(FIX, "golden", "autopose.0.json"), encoding="utf-8") as f:
            rust = json.load(f)
        bones = [b["name"] for b in rust["skeleton"]["bones"]]
        frame = rust["frames"][0]
        pred = [tuple(frame[name]["quat"]) for name in bones]
        got = limits.frame_violations(pred, bones, lim)
        with open(os.path.join(FIX, "golden", "autopose_limits.stdout"),
                  encoding="utf-8") as f:
            lines = [ln for ln in f.read().splitlines() if ln.startswith("  frame")]
        self.assertEqual(len(got), len(lines))
        self.assertEqual(len(got), 8)
        for (bone, ang, mx), line in zip(got, lines):
            self.assertIn(bone, line)
            self.assertIn(f"{ang:.2f}", line)
            self.assertIn(f"{mx:.2f}", line)
        self.assertEqual(limits.unmatched_bones(bones, lim), [])


if __name__ == "__main__":
    unittest.main()
