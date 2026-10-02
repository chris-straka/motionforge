# SPDX-License-Identifier: GPL-3.0-or-later
"""Pure tests for the bpy-free extension modules (quats, cli arg builders).

Run with system Python (no Blender, stdlib only):
    python3 -m unittest discover -s blender/tests -p "test_pure.py"
"""

import importlib.util
import os
import stat
import tempfile
import unittest


def _load(name):
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "motionforge", name + ".py"
    )
    spec = importlib.util.spec_from_file_location("motionforge_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Loaded by file path: importing the package would execute __init__.py,
# which needs bpy.
cli = _load("cli")
quats = _load("quats")


class TestQuats(unittest.TestCase):
    def assertRowsAlmostEqual(self, a, b, eps=1e-9):
        for ra, rb in zip(a, b):
            for x, y in zip(ra, rb):
                self.assertLess(abs(x - y), eps)

    def test_rest_basis_matches_rust(self):
        # Same empirical Blender 5.2 roll-0 bases the Rust core pins.
        cases = [
            ((1, 0, 0), ((0, -1, 0), (1, 0, 0), (0, 0, 1))),
            ((0, 1, 0), ((1, 0, 0), (0, 1, 0), (0, 0, 1))),
            ((0, -1, 0), ((-1, 0, 0), (0, -1, 0), (0, 0, 1))),
            ((0, 0, 1), ((1, 0, 0), (0, 0, 1), (0, -1, 0))),
            ((0, 0, -1), ((1, 0, 0), (0, 0, -1), (0, 1, 0))),
        ]
        for direction, cols in cases:
            rows = quats.rest_basis((0, 0, 0), direction)
            got_cols = tuple(tuple(rows[r][c] for r in range(3)) for c in range(3))
            for got, want in zip(got_cols, cols):
                for g, w in zip(got, want):
                    self.assertLess(abs(g - w), 1e-9)

    def test_max_abs_diff(self):
        ident = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
        self.assertEqual(quats.max_abs_diff(ident, ident), 0.0)
        other = ((1, 0, 0), (0, 1, 0), (0, 0.5, 1))
        self.assertAlmostEqual(quats.max_abs_diff(ident, other), 0.5)


class TestCliBuilders(unittest.TestCase):
    def test_retarget_args(self):
        args = cli.build_retarget_args("s", "t", "m", "o", yaw="flip", pin=False)
        self.assertEqual(
            args,
            ["retarget", "--source", "s", "--target", "t", "--map", "m",
             "--output", "o", "--yaw", "flip", "--no-pin"],
        )

    def test_stylize_args(self):
        params = {
            "exaggeration": 1.5,
            "chain_factors": [("arm", 1.8)],
            "angle_threshold": 0.2,
            "min_spacing": 3,
            "hold": 1,
            "anticipation": 0.0,
            "anticipation_frames": 2,
            "overshoot": 0.0,
        }
        args = cli.build_stylize_args("i", "o", params)
        self.assertIn("--exaggeration", args)
        self.assertIn("arm:1.8", args)
        self.assertEqual(args[0], "stylize")

    def test_physics_args(self):
        params = {
            "root": "R", "feet": ["FL", "FR"], "contact_margin": 0.03,
            "foot_radius": 0.12, "balance_margin": 0.05, "min_air_frames": 4,
            "accel_limit": 25.0, "turn_deg": 40.0, "min_turn_speed": 0.5,
            "blend_frames": 2, "smooth_sigma": 1.0, "smooth_pad": 2,
            "fix_ballistic": True, "fix_momentum": False,
        }
        check = cli.build_physics_args("physics-check", "i", None, params)
        self.assertNotIn("--output", check)
        fix = cli.build_physics_args("physics-fix", "i", "o", params)
        self.assertIn("--no-momentum", fix)
        self.assertNotIn("--no-ballistic", fix)
        self.assertIn("FL,FR", fix)

    def test_parsers(self):
        self.assertEqual(cli.parse_chain_factors(""), [])
        self.assertEqual(cli.parse_chain_factors("arm:1.8, spine:1.2"),
                         [("arm", 1.8), ("spine", 1.2)])
        with self.assertRaises(ValueError):
            cli.parse_chain_factors("arm")
        with self.assertRaises(ValueError):
            cli.parse_chain_factors("arm:xyz")
        self.assertEqual(cli.parse_feet("FL, FR"), ["FL", "FR"])
        with self.assertRaises(ValueError):
            cli.parse_feet("  ")

    def test_find_binary_explicit(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            path = f.name
        os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)
        try:
            self.assertEqual(cli.find_motionforge_binary(path), path)
        finally:
            os.unlink(path)
        # A bogus explicit path falls through to PATH/build-tree search,
        # never returned as-is.
        self.assertNotEqual(cli.find_motionforge_binary("/nonexistent"), "/nonexistent")


if __name__ == "__main__":
    unittest.main()
