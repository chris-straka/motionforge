# SPDX-License-Identifier: GPL-3.0-or-later
"""P3 pipeline test: synthetic BVH -> batch retarget -> manifest gate.

Needs Blender ($BLENDER, PATH, or the macOS app) + a built CLI; skips
otherwise (MF_REQUIRE_BLENDER=1 turns the skip into a failure, for CI).
Slow (~1-2 min, two headless Blender runs). Run: python3 -m unittest
blender.tests.test_batch from the repo root.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "..", ".."))
# $BLENDER, else `blender` on PATH, else the owner's macOS install.
BLENDER = (os.environ.get("BLENDER") or shutil.which("blender")
           or "/Applications/Blender.app/Contents/MacOS/Blender")
CLI = os.path.join(REPO, "rust", "target", "release", "motionforge")
CLI_DEBUG = os.path.join(REPO, "rust", "target", "debug", "motionforge")

MINI_BVH = """HIERARCHY
ROOT Hips
{
  OFFSET 0.0 0.98 0.0
  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
  JOINT Chest
  {
    OFFSET 0.0 0.12 0.0
    CHANNELS 3 Zrotation Xrotation Yrotation
    End Site
    {
      OFFSET 0.0 0.2 0.0
    }
  }
}
MOTION
Frames: 3
Frame Time: 0.008333
0 0 0 0 0 0 0 0 0
0 0.01 0 0 5 0 0 10 0
0 0.02 0 0 10 0 0 20 0
"""

BUILD_TARGET = """
import bpy
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)
bpy.ops.object.armature_add(enter_editmode=False, location=(0, 0, 0))
arm = bpy.context.view_layer.objects.active
arm.name = "Target"
bpy.ops.object.mode_set(mode="EDIT")
ed = arm.data.edit_bones
for b in list(ed):
    ed.remove(b)
root = ed.new("Root")
root.head, root.tail = (0, 0, 1), (0, 0, 2)
tip = ed.new("Tip")
tip.head, tip.tail = (0, 0, 2), (0, 1, 2)
tip.parent = root
bpy.ops.object.mode_set(mode="OBJECT")
bpy.ops.wm.save_as_mainfile(filepath=r"%s")
"""

MAP = {
    "format": "motionforge-bonemap", "version": 1,
    "pairs": [
        {"source": "Hips", "target": "Root"},
        {"source": "Chest", "target": "Tip"},
    ],
    "root_source": "Hips", "root_target": "Root",
}


def have_prereqs():
    binary = CLI if os.path.isfile(CLI) else CLI_DEBUG
    return os.path.isfile(BLENDER) and os.path.isfile(binary)


@unittest.skipUnless(have_prereqs() or os.environ.get("MF_REQUIRE_BLENDER"),
                     "needs Blender + built CLI")
class TestBatch(unittest.TestCase):
    def test_bvh_to_manifest(self):
        tmp = tempfile.mkdtemp(prefix="mf_batch_")
        try:
            bvh_dir = os.path.join(tmp, "bvh")
            os.makedirs(bvh_dir)
            with open(os.path.join(bvh_dir, "mini.bvh"), "w", encoding="utf-8") as f:
                f.write(MINI_BVH)
            map_path = os.path.join(tmp, "map.json")
            with open(map_path, "w", encoding="utf-8") as f:
                json.dump(MAP, f)
            target_blend = os.path.join(tmp, "target.blend")
            build_py = os.path.join(tmp, "build_target.py")
            with open(build_py, "w", encoding="utf-8") as f:
                f.write(BUILD_TARGET % target_blend)
            out_dir = os.path.join(tmp, "clips")
            manifest = os.path.join(tmp, "manifest.json")

            build = subprocess.run(
                [BLENDER, "--background", "--factory-startup", "--python", build_py],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300)
            self.assertEqual(build.returncode, 0, "target blend build failed")

            batch = subprocess.run(
                [BLENDER, "--background", "--factory-startup",
                 "--python", os.path.join(REPO, "blender", "tools", "batch_retarget.py"),
                 "--",
                 "--target-blend", target_blend, "--bvh-dir", bvh_dir,
                 "--bonemap", map_path, "--out-dir", out_dir,
                 "--manifest", manifest, "--source", "test",
                 "--license", "owner-recorded", "--fps", "120",
                 "--target-name", "Target"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=300)
            self.assertEqual(batch.returncode, 0, batch.stdout[-2000:])
            self.assertIn("wrote 1 clips", batch.stdout)

            clip_path = os.path.join(out_dir, "mini.json")
            self.assertTrue(os.path.isfile(clip_path))
            binary = CLI if os.path.isfile(CLI) else CLI_DEBUG
            info = subprocess.run(
                [binary, "clip-info", "--input", clip_path],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=60)
            self.assertEqual(info.returncode, 0, info.stdout)
            self.assertIn("bones: 2, frames: 3", info.stdout)

            gate = subprocess.run(
                [sys.executable, os.path.join(REPO, "tools", "manifest.py"),
                 "--check", manifest],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=60)
            self.assertEqual(gate.returncode, 0, gate.stdout)
            self.assertIn("MANIFEST OK", gate.stdout)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
