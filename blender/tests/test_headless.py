# Headless end-to-end test for the MotionForge Blender extension.
#
#   /Applications/Blender.app/Contents/MacOS/Blender --background \
#       --factory-startup --python blender/tests/test_headless.py
#
# Needs a built `motionforge` CLI (rust/target/release|debug in the repo,
# or on PATH). Exits 0 on pass or SKIP (binary missing), 1 on failure.

import json
import math
import os
import sys

import bpy
from mathutils import Matrix

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "..", ".."))
sys.path.insert(0, os.path.join(REPO, "blender"))

import motionforge  # noqa: E402  (the extension package under test)
from motionforge import cli, clip_io  # noqa: E402


def check(condition, message):
    print(("PASS" if condition else "FAIL") + ": " + message)
    if not condition:
        raise SystemExit(1)


def check_cancel(label, func):
    """Negative-path operator call: passes on a cancel, whether Blender
    hands back {'CANCELLED'} or raises RuntimeError (an ERROR-reporting
    cancel raises through bpy.ops on Blender 5.x)."""
    try:
        result = func()
    except RuntimeError as exc:
        print(f"PASS: {label} cancels (raised: {exc})")
        return
    check("CANCELLED" in result, f"{label} cancels (got {result})")


def clean_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for coll in (bpy.data.meshes, bpy.data.armatures, bpy.data.actions):
        for x in list(coll):
            coll.remove(x)


def make_armature(name, specs):
    """specs: [(bone, head, tail, parent)]. Roll-0 bones at identity."""
    bpy.ops.object.armature_add(enter_editmode=False, location=(0, 0, 0))
    arm = bpy.context.view_layer.objects.active
    arm.name = name
    bpy.ops.object.mode_set(mode="EDIT")
    ed = arm.data.edit_bones
    for b in list(ed):
        ed.remove(b)
    made = {}
    for bone, head, tail, parent in specs:
        eb = ed.new(bone)
        eb.head, eb.tail = head, tail
        eb.roll = 0.0
        made[bone] = eb
        if parent is not None:
            eb.parent = made[parent]
    bpy.ops.object.mode_set(mode="POSE")
    for pb in arm.pose.bones:
        pb.rotation_mode = "QUATERNION"
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.context.view_layer.objects.active = arm
    return arm


def key_action(arm, action_name, n_frames, pose_fn):
    """New action with n_frames; pose_fn(arm, i) sets matrix_basis each frame."""
    action = bpy.data.actions.new(action_name)
    arm.animation_data_create()
    arm.animation_data.action = action
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 1, n_frames
    for i in range(1, n_frames + 1):
        scene.frame_set(i)
        pose_fn(arm, i)
        bpy.context.view_layer.update()
        for pb in arm.pose.bones:
            pb.keyframe_insert("location", frame=i)
            pb.keyframe_insert("rotation_quaternion", frame=i)
    scene.frame_set(1)
    return action


def swing_pose(arm, i):
    ph = 2.0 * math.pi * (i - 1) / 8
    arm.pose.bones["Root"].location = (0.0, 0.1 * (i - 1), 0.0)
    arm.pose.bones["Mid"].rotation_quaternion = Matrix.Rotation(
        0.3 * math.sin(ph), 4, "X").to_quaternion()
    arm.pose.bones["Tip"].rotation_quaternion = Matrix.Rotation(
        0.2 * math.cos(ph), 4, "Z").to_quaternion()


CHAIN = [
    ("Root", (0, 0, 1), (0, 0, 2), None),
    ("Mid", (0, 0, 2), (0, 1, 2), "Root"),
    ("Tip", (0, 1, 2), (0, 2, 2), "Mid"),
]


def clips_close(a, b, eps=2e-6):
    """Clip dicts equal within Blender float32 round-trip noise."""
    if (a["skeleton"] != b["skeleton"] or len(a["frames"]) != len(b["frames"])):
        return False
    for fa, fb in zip(a["frames"], b["frames"]):
        if set(fa) != set(fb):
            return False
        for bone in fa:
            for key in ("loc", "quat"):
                for x, y in zip(fa[bone][key], fb[bone][key]):
                    if abs(x - y) > eps:
                        return False
    return True


def all_fcurves(action):
    """Every fcurve in an action, layered (Blender 5.x) or legacy."""
    if hasattr(action, "fcurves"):
        return list(action.fcurves)
    out = []
    for layer in action.layers:
        for strip in layer.strips:
            for bag in getattr(strip, "channelbags", ()):
                out.extend(bag.fcurves)
    return out


def fcurve_key_count(action, data_path, index):
    for fc in all_fcurves(action):
        if fc.data_path == data_path and fc.array_index == index:
            return len(fc.keyframe_points)
    return 0


def main():
    bpy.ops.preferences.addon_enable(module="motionforge")
    try:
        check(hasattr(bpy.ops.motionforge, "retarget"), "retarget operator registered")
        check(hasattr(bpy.ops.motionforge, "stylize"), "stylize operator registered")
        check(hasattr(bpy.ops.motionforge, "physics_check"), "physics_check registered")
        check(hasattr(bpy.ops.motionforge, "physics_fix"), "physics_fix registered")
        check(hasattr(bpy.ops.motionforge, "physics_overlay"), "physics_overlay registered")
        check(hasattr(bpy.ops.motionforge, "autopose"), "autopose registered")

        binary = cli.find_motionforge_binary("")
        check(bool(binary), f"CLI binary found ({binary})")

        # --- export/import round-trip (float32-close) ---------------------
        clean_scene()
        arm = make_armature("Chain", CHAIN)
        key_action(arm, "Swing", 8, swing_pose)
        scene = bpy.context.scene
        exported = clip_io.export_action(arm, arm.animation_data.action, 1, 8)
        check(len(exported["frames"]) == 8, "exported 8 frames")
        check([b["name"] for b in exported["skeleton"]["bones"]] == ["Root", "Mid", "Tip"],
              "skeleton order is topological")
        imported = clip_io.import_clip(arm, exported, "RoundTrip", 1)
        reexported = clip_io.export_action(arm, imported, 1, 8)
        check(clips_close(exported, reexported), "export/import round-trips float32-close")

        # --- prefix strip ------------------------------------------------
        prefixed = make_armature("Prefixed", [
            ("mixamorig:Hips", (0, 0, 1), (0, 0, 2), None),
            ("mixamorig:Spine", (0, 0, 2), (0, 0, 3), "mixamorig:Hips"),
        ])
        skel = clip_io.export_skeleton(prefixed, strip_prefix=True)
        check([b["name"] for b in skel["bones"]] == ["Hips", "Spine"], "prefix stripped")
        check(skel["bones"][1]["parent"] == "Hips", "stripped parent relinked")

        # --- refusals ----------------------------------------------------
        rolled = make_armature("Rolled", CHAIN)
        bpy.ops.object.mode_set(mode="EDIT")
        rolled.data.edit_bones["Mid"].roll = 0.5
        bpy.ops.object.mode_set(mode="OBJECT")
        try:
            clip_io.export_skeleton(rolled)
            check(False, "rolled bone refused")
        except clip_io.MotionforgeError as exc:
            check("Mid" in str(exc), f"rolled bone named ({exc})")
        scaled = make_armature("Scaled", CHAIN)
        key_action(scaled, "ScaledAct", 2, lambda a, i: None)
        scaled.pose.bones["Tip"].scale = (1.0, 2.0, 1.0)
        scaled.pose.bones["Tip"].keyframe_insert("scale", frame=1)
        try:
            clip_io.export_action(scaled, scaled.animation_data.action, 1, 2)
            check(False, "scaled basis refused")
        except clip_io.MotionforgeError as exc:
            check("Tip" in str(exc), f"scaled bone named ({exc})")

        # --- stylize operator -------------------------------------------
        clean_scene()
        arm = make_armature("Chain", CHAIN)
        key_action(arm, "Swing", 8, swing_pose)
        bpy.context.view_layer.objects.active = arm
        params = scene.motionforge_params
        params.stylize_exaggeration = 2.0
        params.stylize_anticipation = 0.0
        result = bpy.ops.motionforge.stylize()
        check("FINISHED" in result, "stylize finishes")
        check("Swing_stylized" in bpy.data.actions, "stylized action created")
        stylized = bpy.data.actions["Swing_stylized"]
        thin_count = fcurve_key_count(stylized, 'pose.bones["Mid"].rotation_quaternion', 0)
        check(2 <= thin_count < 8, f"thinned keys imported ({thin_count} < 8)")
        frames = sorted(
            kp.co.x
            for fc in all_fcurves(stylized)
            if fc.data_path == 'pose.bones["Mid"].rotation_quaternion' and fc.array_index == 0
            for kp in fc.keyframe_points
        )
        check(frames[0] == 1 and frames[-1] == 8, "thinned keys span endpoints")
        # Root travel passes through the stylizer unchanged and keeps a
        # location key on every frame; static locations stay thinned.
        check(fcurve_key_count(stylized, 'pose.bones["Root"].location', 1) == 8,
              "moving root location keyed every frame")
        check(fcurve_key_count(stylized, 'pose.bones["Mid"].location', 1) < 8,
              "static location stays thinned")
        travel = [kp.co.y for fc in all_fcurves(stylized)
                  if fc.data_path == 'pose.bones["Root"].location' and fc.array_index == 1
                  for kp in fc.keyframe_points]
        check(all(abs(v - 0.1 * i) < 1e-6 for i, v in enumerate(travel)),
              f"root travel unchanged ({[round(v, 3) for v in travel]})")
        check(
            all(kp.interpolation == "LINEAR" for fc in all_fcurves(stylized)
                for kp in fc.keyframe_points),
            "imported keys are LINEAR",
        )
        check("keys kept" in params.last_report, "report stored")
        # Dense path: thin off restores every-frame keys.
        params.stylize_thin_keys = False
        arm.animation_data.action = bpy.data.actions["Swing"]
        result = bpy.ops.motionforge.stylize()
        check("FINISHED" in result, "dense stylize finishes")
        dense = bpy.data.actions["Swing_stylized.001"]
        check(fcurve_key_count(dense, 'pose.bones["Mid"].rotation_quaternion', 0) == 8,
              "dense keys imported (8 frames)")
        params.stylize_thin_keys = True

        # --- physics operators ------------------------------------------
        arm.animation_data.action = bpy.data.actions["Swing_stylized"]
        params.physics_root = "Root"
        params.physics_feet = "Tip"
        result = bpy.ops.motionforge.physics_check()
        check("FINISHED" in result, "physics_check finishes")
        check("balance:" in params.last_report, "physics report stored")
        result = bpy.ops.motionforge.physics_fix()
        check("FINISHED" in result, "physics_fix finishes")
        # Active action at fix time is the stylized one.
        check("Swing_stylized_physics" in bpy.data.actions, "physics action created")
        scene.frame_set(4)
        result = bpy.ops.motionforge.physics_overlay()
        check("FINISHED" in result, "physics_overlay finishes")
        com_empty = scene.objects.get("MF_COM")
        support_empty = scene.objects.get("MF_SUPPORT")
        check(com_empty is not None and support_empty is not None, "overlay empties created")
        check(all(math.isfinite(c) for c in com_empty.location), "COM marker placed")
        check("excursion" in params.last_report or "airborne" in params.last_report,
              "overlay report stored")
        # Re-run reuses the empties (no duplicates).
        result = bpy.ops.motionforge.physics_overlay()
        check("FINISHED" in result, "overlay re-run finishes")
        check(len([o for o in scene.objects if o.name.startswith("MF_")]) == 2,
              "overlay empties reused")

        # --- retarget operator (procedural pair + generated map) --------
        clean_scene()
        src = make_armature("Src", [
            ("Hips", (0, 0, 1), (0, 0, 1.5), None),
            ("Foot", (0.1, 0, 0.1), (0.1, 0.4, 0.1), "Hips"),
        ])
        def travel_pose(arm, i):
            # Hips is Z-directed (Rx90 rest): world +Y needs local -Z.
            arm.pose.bones["Hips"].location = (0.0, 0.0, -0.05 * (i - 1))

        key_action(src, "SrcAct", 4, travel_pose)
        tgt = make_armature("Tgt", [
            ("DEF-spine", (0, 0, 1), (0, 0, 1.5), None),
            ("DEF-foot", (0.1, 0, 0.1), (0.1, -0.4, 0.1), "DEF-spine"),
        ])
        bonemap = {
            "format": "motionforge-bonemap", "version": 1,
            "pairs": [
                {"source": "Hips", "target": "DEF-spine"},
                {"source": "Foot", "target": "DEF-foot"},
            ],
            "root_source": "Hips", "root_target": "DEF-spine",
            "feet_source": ["Foot"], "feet_target": ["DEF-foot"],
        }
        map_path = os.path.join(REPO, "blender", "tests", "_tmp_map.json")
        with open(map_path, "w", encoding="utf-8") as f:
            json.dump(bonemap, f)
        try:
            params = scene.motionforge_params
            params.retarget_source = src
            params.bonemap_preset = "CUSTOM"
            params.bonemap_path = map_path
            bpy.context.view_layer.objects.active = tgt
            result = bpy.ops.motionforge.retarget()
            check("FINISHED" in result, "retarget finishes")
            check("SrcAct_retargeted" in bpy.data.actions, "retargeted action created")
            check("yaw: flip" in params.last_report, "opposite facing auto-flipped")
            # Forward (+Y) source travel arrives as -Y target travel.
            out = tgt.animation_data.action
            scene.frame_set(1)
            y0 = tgt.pose.bones["DEF-spine"].matrix.translation.y
            tgt.animation_data.action = out
            scene.frame_set(4)
            bpy.context.view_layer.update()
            y1 = tgt.pose.bones["DEF-spine"].matrix.translation.y
            check(y1 < y0 - 0.1, f"yaw flip reverses travel ({y0:.3f} -> {y1:.3f})")
        finally:
            os.unlink(map_path)

        # --- autopose operator ------------------------------------------
        clean_scene()
        arm = make_armature("Chain", CHAIN)
        bpy.context.view_layer.objects.active = arm
        bpy.ops.object.mode_set(mode="POSE")
        # Move one joint, select it as the effector.
        arm.pose.bones["Tip"].location = (0.0, 0.5, 0.0)
        bpy.context.view_layer.update()
        bpy.ops.pose.select_all(action="DESELECT")
        # Blender 5: selection lives on PoseBone, not Bone.
        arm.pose.bones["Tip"].select = True
        # Identity-ish weights: hidden passthrough is overkill here; a
        # single layer mapping the Tip mask bit to a fixed quat wins.
        # out = bias, so every bone gets the bias quat, normalized.
        weights = {
            "format": "motionforge-weights", "version": 1,
            "bones": ["Root", "Mid", "Tip"],
            "layers": [{
                "weights": [[0.0] * 15 for _ in range(12)],
                "bias": [1, 0, 0, 0, 0.7071, 0.7071, 0, 0, 1, 0, 0, 0],
            }],
        }
        weights_path = os.path.join(REPO, "blender", "tests", "_tmp_weights.json")
        with open(weights_path, "w", encoding="utf-8") as f:
            json.dump(weights, f)
        try:
            scene.motionforge_params.autopose_weights = weights_path
            result = bpy.ops.motionforge.autopose()
            check("FINISHED" in result, "autopose finishes")
            q = arm.pose.bones["Mid"].rotation_quaternion
            check(abs(q.w - 0.7071) < 1e-3 and abs(q.x - 0.7071) < 1e-3,
                  f"predicted quat applied ({tuple(round(c, 4) for c in q)})")
            tip_loc = arm.pose.bones["Tip"].location
            check(abs(tip_loc.y - 0.5) < 1e-6,
                  f"effector location preserved ({tuple(round(c, 4) for c in tip_loc)})")
        finally:
            os.unlink(weights_path)

        # --- negative paths ---------------------------------------------
        bpy.ops.object.mode_set(mode="OBJECT")
        clean_scene()
        bpy.ops.mesh.primitive_cube_add()
        cube = bpy.context.view_layer.objects.active
        check_cancel("stylize with no armature", lambda: bpy.ops.motionforge.stylize())
        arm = make_armature("NoAction", CHAIN)
        bpy.context.view_layer.objects.active = arm
        check_cancel("stylize with no action", lambda: bpy.ops.motionforge.stylize())

        print("ALL HEADLESS CHECKS PASSED")
    finally:
        bpy.ops.preferences.addon_disable(module="motionforge")


if __name__ == "__main__":
    # Airtight gate: Blender 5.2 exits 0 on uncaught non-SystemExit
    # errors (verified), so convert any escapee into a traceback plus
    # exit 1. check() failures already raise SystemExit(1).
    import traceback

    try:
        main()
    except SystemExit:
        raise
    except BaseException:
        traceback.print_exc()
        raise SystemExit(1)
