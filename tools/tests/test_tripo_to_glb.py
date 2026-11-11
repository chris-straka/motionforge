import json
import os
import shutil
import subprocess
import tempfile
import unittest

TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def blender():
    for p in (os.environ.get("BLENDER"), "/Applications/Blender.app/Contents/MacOS/Blender", shutil.which("blender")):
        if p and os.path.exists(p):
            return p
    return None


def figure_obj(path):
    """A blocky figure facing +X (the Tripo way) as an OBJ: legs, feet with
    toes ahead, torso, arms, head. Each box's six faces are loose sheets
    (so each has an open rim), and the torso's top face is left out."""
    boxes = [  # centre, size (x forward, y side, z up)
        ((0.0, -0.1, 0.45), (0.1, 0.1, 0.8)), ((0.0, 0.1, 0.45), (0.1, 0.1, 0.8)),
        ((0.07, -0.1, 0.03), (0.24, 0.09, 0.06)), ((0.07, 0.1, 0.03), (0.24, 0.09, 0.06)),
        ((0.0, 0.0, 1.15), (0.2, 0.36, 0.6)),
        ((0.0, -0.42, 1.25), (0.08, 0.5, 0.08)), ((0.0, 0.42, 1.25), (0.08, 0.5, 0.08)),
        ((0.01, 0.0, 1.6), (0.2, 0.18, 0.22)),
    ]
    lines, n = [], 0
    for bi, (c, s) in enumerate(boxes):
        # Subdivided faces so the facing measure has points to work with.
        k = 6
        faces = [((0, 1), 2, -1), ((0, 1), 2, 1), ((0, 2), 1, -1), ((0, 2), 1, 1), ((1, 2), 0, -1), ((1, 2), 0, 1)]
        for fi, ((a, b), fixed, sign) in enumerate(faces):
            if bi == 4 and fi == 5:
                continue  # the torso's front face is missing: a hole
            for i in range(k + 1):
                for j in range(k + 1):
                    p = [0.0, 0.0, 0.0]
                    p[a] = c[a] + (i / k - 0.5) * s[a]
                    p[b] = c[b] + (j / k - 0.5) * s[b]
                    p[fixed] = c[fixed] + sign * s[fixed] / 2
                    # OBJ is Y up, -Z forward: Blender reads (x, y, z) as (x, -z, y).
                    lines.append("v %.5f %.5f %.5f" % (p[0], p[2], -p[1]))
            for i in range(k):
                for j in range(k):
                    v = n + i * (k + 1) + j + 1
                    quad = (v, v + k + 1, v + k + 2, v + 1)
                    lines.append("f %d %d %d %d" % (quad if sign > 0 else quad[::-1]))
            n += (k + 1) * (k + 1)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


@unittest.skipUnless(blender(), "needs Blender")
class TripoToGlb(unittest.TestCase):
    def run_tool(self, *extra):
        d = tempfile.mkdtemp(prefix="tripo_to_glb_test_")
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        src, out, rep = os.path.join(d, "figure.obj"), os.path.join(d, "figure.glb"), os.path.join(d, "r.json")
        figure_obj(src)
        r = subprocess.run([blender(), "-b", "--factory-startup", "-P", os.path.join(TOOLS, "tripo_to_glb.py"), "--",
                            src, out, "--report", rep, *extra], capture_output=True, text=True, timeout=300)
        self.assertEqual(r.returncode, 0, r.stderr[-800:])
        self.assertTrue(os.path.getsize(out) > 1000)
        with open(rep) as f:
            return json.load(f)

    def test_a_figure_facing_x_is_turned_forward_and_its_holes_are_counted(self):
        info = self.run_tool("--holes", "0")
        self.assertEqual(info["kind"], "biped")
        self.assertEqual(info["decided_by"], "both")
        self.assertEqual(info["forward"], [1.0, 0.0])
        self.assertGreater(info["holes"]["found"], 0)
        self.assertEqual(info["holes"]["filled"], 0)
        self.assertEqual(info["holes"]["left"], info["holes"]["found"])

    def test_small_holes_are_filled_and_big_ones_left(self):
        # Every loose face of these boxes is an open sheet: with a generous
        # limit all of them close, with a tight one none do.
        closed = self.run_tool("--holes", "5")
        self.assertGreater(closed["holes"]["filled"], 0)
        self.assertEqual(closed["holes"]["left"], 0)
        tight = self.run_tool("--holes", "0.0001")
        self.assertEqual(tight["holes"]["filled"], 0)
        self.assertEqual(tight["holes"]["left"], tight["holes"]["found"])

    def test_a_budget_cuts_the_triangles(self):
        info = self.run_tool("--tris", "300")
        self.assertLessEqual(info["tris"], 330)
        self.assertGreater(info["source_tris"], info["tris"])


if __name__ == "__main__":
    unittest.main()
