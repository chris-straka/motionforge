# SPDX-License-Identifier: GPL-3.0-or-later
"""Blender <-> clip JSON I/O (needs bpy).

Export reads matrix_basis (location + rotation quaternion) per bone per
frame plus head/tail rest data; import writes dense keys back. Bones
with non-zero roll are refused (the v1 clip format has no roll
channel); non-uniform basis scale is refused (v1 has no scale
channel). Both limits are reported with the offending bone names.
"""

import bpy  # noqa: F401  (this module needs Blender)

from . import quats


class MotionforgeError(Exception):
    """User-facing error (shown, not traced)."""


def bone_depth(bone):
    n = 0
    b = bone
    while b.parent is not None:
        n += 1
        b = b.parent
    return n


def topological_bones(arm_obj):
    """Armature bones, parents before children, deterministic."""
    bones = list(arm_obj.data.bones)
    bones.sort(key=lambda b: (bone_depth(b), b.name))
    return bones


def strip_mixamo_prefix(name):
    """Strip Mixamo FBX name decoration to the bonemap short form."""
    short = name.split(":")[-1]
    if short.lower().startswith("mixamorig_"):
        short = short[len("mixamorig_"):]
    return short


def check_zero_roll(arm_obj, bones, tol=1e-4):
    """Refuse bones whose rest orientation is not zero-roll."""
    bad = []
    for bone in bones:
        expected = quats.rest_basis(tuple(bone.head_local), tuple(bone.tail_local))
        actual = bone.matrix_local.to_3x3()
        actual_rows = (tuple(actual[0][:]), tuple(actual[1][:]), tuple(actual[2][:]))
        if quats.max_abs_diff(expected, actual_rows) > tol:
            bad.append(bone.name)
    if bad:
        raise MotionforgeError(
            "bones with non-zero roll are unsupported in v1 "
            f"(no roll channel): {', '.join(sorted(bad))}"
        )


def export_skeleton(arm_obj, strip_prefix=False):
    """Skeleton dict for a clip/skeleton file. With strip_prefix, Mixamo
    FBX prefixes are removed (retarget sources); collisions are an error."""
    bones = topological_bones(arm_obj)
    check_zero_roll(arm_obj, bones)
    names = {}
    for bone in bones:
        name = strip_mixamo_prefix(bone.name) if strip_prefix else bone.name
        if name in names:
            raise MotionforgeError(f"prefix strip collides on bone name '{name}'")
        names[bone.name] = name
    out = []
    for bone in bones:
        out.append(
            {
                "name": names[bone.name],
                "parent": names[bone.parent.name] if bone.parent else None,
                "head": [c for c in bone.head_local],
                "tail": [c for c in bone.tail_local],
            }
        )
    return {"bones": out}


def export_action(arm_obj, action, frame_start, frame_end, strip_prefix=False):
    """Full clip dict for an action over a frame range (inclusive)."""
    if action is None:
        raise MotionforgeError("no action to export")
    if frame_end < frame_start:
        raise MotionforgeError("frame range is empty")
    bones = topological_bones(arm_obj)
    check_zero_roll(arm_obj, bones)
    names = {}
    for bone in bones:
        name = strip_mixamo_prefix(bone.name) if strip_prefix else bone.name
        if name in names.values():
            raise MotionforgeError(f"prefix strip collides on bone name '{name}'")
        names[bone.name] = name
    scene = bpy.context.scene
    fps = scene.render.fps / scene.render.fps_base
    prev_frame = scene.frame_current
    frames = []
    try:
        for f in range(frame_start, frame_end + 1):
            scene.frame_set(f)
            bpy.context.view_layer.update()
            frame = {}
            for bone in bones:
                pose_bone = arm_obj.pose.bones.get(bone.name)
                if pose_bone is None:
                    raise MotionforgeError(f"pose bone missing: {bone.name}")
                basis = pose_bone.matrix_basis
                scale = basis.to_scale()
                if any(abs(s - 1.0) > 1e-4 for s in scale):
                    raise MotionforgeError(
                        f"bone '{bone.name}' has non-unit basis scale at frame {f} "
                        "(no scale channel in v1)"
                    )
                quat = basis.to_quaternion()
                frame[names[bone.name]] = {
                    "loc": [c for c in basis.translation],
                    "quat": [quat.w, quat.x, quat.y, quat.z],
                }
            frames.append(frame)
    finally:
        scene.frame_set(prev_frame)
        bpy.context.view_layer.update()
    return {
        "format": "motionforge-clip",
        "version": 1,
        "fps": fps,
        "skeleton": export_skeleton(arm_obj, strip_prefix),
        "frames": frames,
    }


def import_clip(arm_obj, clip, action_name, frame_start):
    """Write a clip dict as dense keys on an armature; returns the action.

    Clip bones missing from the armature are an error; armature bones
    missing from the clip are left untouched (control rigs survive).
    """
    if clip.get("format") != "motionforge-clip":
        raise MotionforgeError("not a motionforge-clip document")
    skeleton = clip.get("skeleton", {}).get("bones", [])
    frames = clip.get("frames", [])
    if not frames:
        raise MotionforgeError("clip has no frames")
    for bone in skeleton:
        if bone["name"] not in arm_obj.pose.bones:
            raise MotionforgeError(
                f"clip bone '{bone['name']}' is not on armature '{arm_obj.name}'"
            )
    for pose_bone in arm_obj.pose.bones:
        pose_bone.rotation_mode = "QUATERNION"
    action = bpy.data.actions.new(action_name)
    if arm_obj.animation_data is None:
        arm_obj.animation_data_create()
    arm_obj.animation_data.action = action
    for i, frame in enumerate(frames):
        f = frame_start + i
        for bone in skeleton:
            name = bone["name"]
            pose = frame.get(name)
            if pose is None:
                raise MotionforgeError(f"clip frame {i} misses bone '{name}'")
            pose_bone = arm_obj.pose.bones[name]
            pose_bone.location = pose["loc"]
            w, x, y, z = pose["quat"]
            pose_bone.rotation_quaternion = (w, x, y, z)
            pose_bone.keyframe_insert("location", frame=f)
            pose_bone.keyframe_insert("rotation_quaternion", frame=f)
    bpy.context.view_layer.update()
    return action


def apply_pose_frame(arm_obj, clip, key=True):
    """Apply clip frame 0's ROTATIONS at the current frame (AutoPose).

    Locations are deliberately untouched: the model predicts rotations
    only, and zeroing loc channels would snap the animator's effector
    moves (and the root placement) back to rest.
    """
    if clip.get("format") != "motionforge-clip":
        raise MotionforgeError("not a motionforge-clip document")
    frames = clip.get("frames", [])
    if not frames:
        raise MotionforgeError("clip has no frames")
    frame = frames[0]
    f = bpy.context.scene.frame_current
    for bone in clip.get("skeleton", {}).get("bones", []):
        name = bone["name"]
        if name not in arm_obj.pose.bones:
            raise MotionforgeError(
                f"clip bone '{name}' is not on armature '{arm_obj.name}'"
            )
    for bone in clip.get("skeleton", {}).get("bones", []):
        name = bone["name"]
        pose = frame.get(name)
        if pose is None:
            raise MotionforgeError(f"pose misses bone '{name}'")
        pose_bone = arm_obj.pose.bones[name]
        pose_bone.rotation_mode = "QUATERNION"
        w, x, y, z = pose["quat"]
        pose_bone.rotation_quaternion = (w, x, y, z)
        if key:
            pose_bone.keyframe_insert("rotation_quaternion", frame=f)
    bpy.context.view_layer.update()


def export_effectors(arm_obj, bone_names):
    """Effectors dict from bones' CURRENT head positions (armature space).

    The animator moves the joints, then runs AutoPose: this captures where
    the effectors are now. Armature-space, so a moved armature still works.
    """
    if not 1 <= len(bone_names) <= 6:
        raise MotionforgeError("AutoPose needs 1-6 effector bones")
    effectors = []
    for name in bone_names:
        pose_bone = arm_obj.pose.bones.get(name)
        if pose_bone is None:
            raise MotionforgeError(f"effector bone '{name}' is not on '{arm_obj.name}'")
        # PoseBone.matrix is already armature-space (world would need
        # matrix_world applied); effectors want armature space.
        head = pose_bone.matrix.translation
        effectors.append({"bone": name, "position": [c for c in head]})
    return {
        "format": "motionforge-effectors",
        "version": 1,
        "skeleton": export_skeleton(arm_obj),
        "effectors": effectors,
    }
