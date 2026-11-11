import math
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import facing  # noqa: E402


def box(rng, n, cx, cy, cz, sx, sy, sz):
    return [(cx + (rng.random() - 0.5) * sx, cy + (rng.random() - 0.5) * sy, cz + (rng.random() - 0.5) * sz)
            for _ in range(n)]


def biped(seed=1):
    """A figure facing +Y: legs, feet with toes ahead of the ankles,
    torso, arms out to the sides, head."""
    rng = random.Random(seed)
    pts = []
    for side in (-1, 1):
        pts += box(rng, 300, side * 0.1, 0.0, 0.45, 0.1, 0.1, 0.8)      # leg
        pts += box(rng, 160, side * 0.1, 0.07, 0.03, 0.09, 0.24, 0.06)  # foot, toes forward
        pts += box(rng, 220, side * 0.42, 0.0, 1.25, 0.5, 0.08, 0.08)   # arm
    pts += box(rng, 600, 0.0, 0.0, 1.15, 0.36, 0.2, 0.6)                # torso
    pts += box(rng, 200, 0.0, 0.01, 1.6, 0.18, 0.2, 0.22)               # head
    return pts


def quadruped(seed=2):
    """A beast facing +Y: body along Y, four legs, neck and head up at
    the front, a thin tail low at the back."""
    rng = random.Random(seed)
    pts = box(rng, 900, 0.0, 0.0, 0.7, 0.3, 1.0, 0.35)
    for sx in (-1, 1):
        for sy in (-1, 1):
            pts += box(rng, 180, sx * 0.11, sy * 0.4, 0.27, 0.08, 0.1, 0.55)
    pts += box(rng, 260, 0.0, 0.6, 0.95, 0.14, 0.3, 0.3)   # neck and head
    pts += box(rng, 60, 0.0, -0.62, 0.6, 0.04, 0.25, 0.05)  # tail
    return pts


def turned(pts, degrees):
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    return [(x * c - y * s, x * s + y * c, z) for x, y, z in pts]


def angle_between(a, b):
    return math.degrees(math.acos(max(-1.0, min(1.0, a[0] * b[0] + a[1] * b[1]))))


class Facing(unittest.TestCase):
    def test_a_biped_is_found_facing_any_way(self):
        for degrees in (0, 90, 180, 270, 37):
            m = facing.measure(turned(biped(), degrees))
            want = (-math.sin(math.radians(degrees)), math.cos(math.radians(degrees)))
            self.assertEqual(m["kind"], "biped")
            self.assertLess(angle_between(m["forward"], want), 6.0, degrees)
            self.assertGreater(m["mirror_contrast"], facing.MIN_MIRROR_CONTRAST)
            self.assertGreater(m["confidence"], facing.MIN_CONFIDENCE)

    def test_a_quadruped_is_found_by_its_head_end(self):
        for degrees in (0, 90, 180, 270):
            m = facing.measure(turned(quadruped(), degrees))
            want = (-math.sin(math.radians(degrees)), math.cos(math.radians(degrees)))
            self.assertEqual(m["kind"], "quadruped")
            self.assertLess(angle_between(m["forward"], want), 6.0, degrees)

    def test_a_strong_measurement_beats_the_convention(self):
        m = facing.measure(turned(biped(), 180))  # facing -Y
        fwd, by, off = facing.decide(m, "+x")
        self.assertEqual(by, "measurement")
        self.assertLess(angle_between(fwd, (0.0, -1.0)), 6.0)
        self.assertGreater(off, 80.0)

    def test_agreement_snaps_to_the_convention(self):
        m = facing.measure(turned(biped(), -90))  # facing +X
        fwd, by, _ = facing.decide(m, "+x")
        self.assertEqual(by, "both")
        self.assertEqual(fwd, (1.0, 0.0))

    def test_a_cloak_to_the_ground_falls_back_to_the_convention(self):
        # A cone: no feet, no one mirror plane better than another.
        rng = random.Random(5)
        pts = []
        for _ in range(2500):
            z = rng.random() * 1.8
            r = 0.45 * (1.0 - z / 2.2)
            a = rng.random() * math.tau
            pts.append((r * math.cos(a), r * math.sin(a), z))
        m = facing.measure(pts, kind="biped")
        fwd, by, _ = facing.decide(m, "+x")
        self.assertEqual(by, "convention")
        self.assertEqual(fwd, (1.0, 0.0))

    def test_no_convention_keeps_the_measurement(self):
        m = facing.measure(biped())
        fwd, by, off = facing.decide(m, "none")
        self.assertEqual(by, "measurement")
        self.assertIsNone(off)

    def test_the_turn_puts_forward_on_minus_y(self):
        for fwd in ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.6, -0.8)):
            a = facing.turn_to_minus_y(fwd)
            x = fwd[0] * math.cos(a) - fwd[1] * math.sin(a)
            y = fwd[0] * math.sin(a) + fwd[1] * math.cos(a)
            self.assertAlmostEqual(x, 0.0, places=6)
            self.assertAlmostEqual(y, -1.0, places=6)


if __name__ == "__main__":
    unittest.main()
