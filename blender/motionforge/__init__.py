# SPDX-License-Identifier: GPL-3.0-or-later
# MotionForge Blender extension: retarget, stylize, physics, AutoPose.
#
# Flow per feature: export the action/armature to a temp clip JSON file,
# run the `motionforge` CLI as a subprocess, import the result as a new
# action. The user never sees a file; temp files live in the system temp
# dir and are removed afterwards.
#
# The engine stays MIT-licensed: this GPL extension talks to it only as a
# subprocess over clip JSON files, never linked, never imported.

import json
import os
import shutil
import tempfile

import bpy
from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)

from . import cli, clip_io

# Must equal the module name Blender loaded us under (see retopoforge).
ADDON_ID = __package__

PRESET_MAPS = {
    "MIXAMO_HERO": ("mixamo_hllhero.json", "Mixamo -> HLL Hero"),
}


def _addon_dir():
    return os.path.dirname(os.path.abspath(__file__))


def poll_armature(_self, obj):
    return obj.type == "ARMATURE"


def _prefs(context):
    return context.preferences.addons[ADDON_ID].preferences


def _params(context):
    return context.scene.motionforge_params


def _require_binary(context):
    binary = cli.find_motionforge_binary(_prefs(context).motionforge_binary)
    if not binary:
        raise clip_io.MotionforgeError(
            "no motionforge CLI found (set it in the add-on preferences)"
        )
    return binary


def _active_armature(context):
    obj = context.view_layer.objects.active
    if obj is None or obj.type != "ARMATURE":
        raise clip_io.MotionforgeError("make an armature the active object first")
    return obj


def _active_action(obj):
    action = obj.animation_data.action if obj.animation_data else None
    if action is None:
        raise clip_io.MotionforgeError(f"'{obj.name}' has no active action")
    return action


def _write_json(path, doc):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f)
        f.write("\n")


def _read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _store_report(context, text):
    # Panel label budget: keep the head; the full text is one screen.
    lines = text.splitlines()[:40]
    context.scene.motionforge_params.last_report = "\n".join(lines)


def _run_feature(context, exports, args, import_result):
    """Shared export -> CLI -> import driver.

    exports: {filename: doc}; args: CLI argv with {tmp} placeholders
    resolved to the temp dir; import_result(tmpdir, report) imports.
    Returns the CLI report text.
    """
    binary = _require_binary(context)
    tmpdir = tempfile.mkdtemp(prefix="motionforge_")
    try:
        for name, doc in exports.items():
            _write_json(os.path.join(tmpdir, name), doc)
        argv = [a.replace("{tmp}", tmpdir) for a in args]
        report = cli.run_cli(binary, argv)
        import_result(tmpdir, report)
        _store_report(context, report)
        return report
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


class MOTIONFORGE_PG_params(bpy.types.PropertyGroup):
    """Feature parameters, stored on the scene for panel editing."""

    retarget_source: PointerProperty(
        name="Source Armature",
        description="Armature carrying the clip to retarget (target is the active object)",
        type=bpy.types.Object,
        poll=poll_armature,
    )
    bonemap_preset: EnumProperty(
        name="Bone Map",
        description="Correspondence preset, or a custom bonemap file",
        items=[
            ("MIXAMO_HERO", "Mixamo -> HLL Hero", "Mixamo clip onto the rigforge hll_hero DEF bones"),
            ("CUSTOM", "Custom", "Use the bonemap path below"),
        ],
        default="MIXAMO_HERO",
    )
    bonemap_path: StringProperty(
        name="Bonemap",
        description="Custom bonemap JSON (when Bone Map is Custom)",
        default="",
        subtype="FILE_PATH",
    )
    retarget_yaw: EnumProperty(
        name="Yaw",
        description="Facing correction between source and target",
        items=[
            ("auto", "Auto", "Flip when foot bones face opposite ways"),
            ("flip", "Flip", "Always rotate 180 deg about Z"),
            ("none", "None", "Never flip"),
        ],
        default="auto",
    )
    retarget_pin: BoolProperty(
        name="Pin Feet",
        description="Ramp stance drift out through the chain roots",
        default=True,
    )

    stylize_exaggeration: FloatProperty(
        name="Exaggeration",
        description="Push extremes away from neutral (1 = none)",
        default=1.35, min=0.0, max=4.0,
    )
    stylize_chains: StringProperty(
        name="Chain Factors",
        description="Per-chain overrides, e.g. 'arm:1.8, spine:1.2' (empty = none)",
        default="",
    )
    stylize_angle_threshold: FloatProperty(
        name="Angle Threshold",
        description="Min joint angle (rad) from rest to count as an extreme",
        default=0.15, min=0.0,
    )
    stylize_min_spacing: IntProperty(
        name="Min Spacing",
        description="Min frames between kept extremes",
        default=4, min=1,
    )
    stylize_hold: IntProperty(
        name="Hold",
        description="Frames frozen at each segment's start key",
        default=2, min=0,
    )
    stylize_anticipation: FloatProperty(
        name="Anticipation",
        description="Counter-pose strength (0 = off)",
        default=0.25, min=0.0, max=1.0,
    )
    stylize_anticipation_frames: IntProperty(
        name="Anticipation Frames",
        description="Frames before the extreme to place the counter-key",
        default=3, min=1,
    )
    stylize_overshoot: FloatProperty(
        name="Overshoot",
        description="Ease-out-back strength (0 = pure ease-out)",
        default=0.6, min=0.0, max=2.0,
    )
    stylize_thin_keys: BoolProperty(
        name="Thin Keys",
        description="Import sparse keys at the kept extremes (off = dense baked keys)",
        default=True,
    )

    physics_root: StringProperty(
        name="Root Bone",
        description="Hips/root bone carrying body motion",
        default="DEF-spine",
    )
    physics_feet: StringProperty(
        name="Feet",
        description="Comma-separated foot bone names",
        default="DEF-foot.L, DEF-foot.R",
    )
    physics_contact_margin: FloatProperty(
        name="Contact Margin", default=0.03, min=0.0,
        description="Foot height above the clip minimum that still counts as contact (m)",
    )
    physics_foot_radius: FloatProperty(
        name="Foot Radius", default=0.12, min=0.001,
        description="Support radius around each contacting foot (m)",
    )
    physics_balance_margin: FloatProperty(
        name="Balance Margin", default=0.05, min=0.0,
        description="Allowed COM excursion past the support edge (m)",
    )
    physics_min_air_frames: IntProperty(
        name="Min Air Frames", default=4, min=3,
        description="Shortest airborne run analyzed for ballistic motion",
    )
    physics_accel_limit: FloatProperty(
        name="Accel Limit", default=25.0, min=0.001,
        description="Flag root horizontal acceleration above this (m/s2)",
    )
    physics_turn_deg: FloatProperty(
        name="Turn Limit", default=40.0, min=0.001,
        description="Flag heading changes above this (deg/frame)",
    )
    physics_min_turn_speed: FloatProperty(
        name="Min Turn Speed", default=0.5, min=0.0,
        description="Turn check applies above this root speed (m/s)",
    )
    physics_blend_frames: IntProperty(
        name="Blend Frames", default=2, min=1,
        description="Crossfade at ballistic-fix phase boundaries",
    )
    physics_smooth_sigma: FloatProperty(
        name="Smooth Sigma", default=1.0, min=0.001, max=1000.0,
        description="Gaussian sigma (frames) for the momentum fix",
    )
    physics_smooth_pad: IntProperty(
        name="Smooth Pad", default=2, min=0, max=1000,
        description="Frames around each flag the momentum fix touches",
    )
    physics_fix_ballistic: BoolProperty(name="Fix Ballistic", default=True)
    physics_fix_momentum: BoolProperty(name="Fix Momentum", default=True)
    physics_fix_balance: BoolProperty(
        name="Fix Balance", default=True,
        description="Lean the upper body to bring the COM over the support feet",
    )
    physics_max_lean_deg: FloatProperty(
        name="Max Lean", default=8.0, min=0.1, max=45.0,
        description="Total lean cap per frame for the balance fix (deg)",
    )

    autopose_weights: StringProperty(
        name="Model",
        description="AutoPose weights JSON for this rig",
        default="",
        subtype="FILE_PATH",
    )

    last_report: StringProperty(
        name="Last Report",
        description="Most recent CLI report",
        default="",
    )


class MotionForgePreferences(bpy.types.AddonPreferences):
    bl_idname = ADDON_ID

    motionforge_binary: StringProperty(
        name="MotionForge CLI",
        description="Path to the motionforge binary (empty: search PATH, then the build tree)",
        default="",
        subtype="FILE_PATH",
    )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "motionforge_binary")
        found = cli.find_motionforge_binary(self.motionforge_binary)
        box = layout.box()
        if found:
            box.label(text="Using: " + found, icon="CHECKMARK")
        else:
            box.label(text="No motionforge binary found", icon="ERROR")


def _bonemap_path(context):
    params = _params(context)
    if params.bonemap_preset == "CUSTOM":
        path = bpy.path.abspath(params.bonemap_path)
        if not path or not os.path.isfile(path):
            raise clip_io.MotionforgeError("custom bonemap path is not a file")
        return path
    filename = PRESET_MAPS[params.bonemap_preset][0]
    return os.path.join(_addon_dir(), "presets", filename)


class MOTIONFORGE_OT_retarget(bpy.types.Operator):
    bl_idname = "motionforge.retarget"
    bl_label = "Retarget Clip"
    bl_description = "Transfer the source action onto the active armature"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        try:
            return self._run(context)
        except (clip_io.MotionforgeError, cli.CliError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

    def _run(self, context):
        params = _params(context)
        target = _active_armature(context)
        source = params.retarget_source
        if source is None or source.type != "ARMATURE":
            raise clip_io.MotionforgeError("pick a source armature in the panel first")
        action = _active_action(source)
        scene = context.scene
        start, end = scene.frame_start, scene.frame_end
        exports = {
            "source.json": clip_io.export_action(source, action, start, end, strip_prefix=True),
        }
        skel = clip_io.export_skeleton(target)
        exports["target.json"] = {"format": "motionforge-skeleton", "version": 1, **skel}
        bonemap = _bonemap_path(context)
        args = cli.build_retarget_args(
            "{tmp}/source.json", "{tmp}/target.json", bonemap, "{tmp}/out.json",
            yaw=params.retarget_yaw, pin=params.retarget_pin,
        )

        def do_import(tmpdir, report):
            clip_io.import_clip(target, _read_json(os.path.join(tmpdir, "out.json")),
                                f"{action.name}_retargeted", start)

        report = _run_feature(context, exports, args, do_import)
        first = report.splitlines()[0] if report else ""
        self.report({"INFO"}, f"Retargeted {action.name} ({first})")
        return {"FINISHED"}


class MOTIONFORGE_OT_stylize(bpy.types.Operator):
    bl_idname = "motionforge.stylize"
    bl_label = "Stylize Action"
    bl_description = "Exaggerate and retime the active action"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        try:
            return self._run(context)
        except (clip_io.MotionforgeError, cli.CliError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

    def _run(self, context):
        params = _params(context)
        arm = _active_armature(context)
        action = _active_action(arm)
        scene = context.scene
        start, end = scene.frame_start, scene.frame_end
        try:
            chain_factors = cli.parse_chain_factors(params.stylize_chains)
        except ValueError as exc:
            raise clip_io.MotionforgeError(str(exc))
        stylize_params = {
            "exaggeration": params.stylize_exaggeration,
            "chain_factors": chain_factors,
            "angle_threshold": params.stylize_angle_threshold,
            "min_spacing": params.stylize_min_spacing,
            "hold": params.stylize_hold,
            "anticipation": params.stylize_anticipation,
            "anticipation_frames": params.stylize_anticipation_frames,
            "overshoot": params.stylize_overshoot,
        }
        exports = {"in.json": clip_io.export_action(arm, action, start, end)}
        thin = params.stylize_thin_keys
        args = cli.build_stylize_args("{tmp}/in.json", "{tmp}/out.json", stylize_params,
                                      keys_out="{tmp}/keys.json" if thin else None)

        def do_import(tmpdir, report):
            keys = _read_json(os.path.join(tmpdir, "keys.json")) if thin else None
            clip_io.import_clip(arm, _read_json(os.path.join(tmpdir, "out.json")),
                                f"{action.name}_stylized", start, keys=keys)

        report = _run_feature(context, exports, args, do_import)
        keys_line = next((line for line in report.splitlines() if "keys kept" in line), "")
        self.report({"INFO"}, f"Stylized {action.name} ({keys_line.strip()})")
        return {"FINISHED"}


def _physics_params(params):
    try:
        feet = cli.parse_feet(params.physics_feet)
    except ValueError as exc:
        raise clip_io.MotionforgeError(str(exc))
    return {
        "root": params.physics_root,
        "feet": feet,
        "contact_margin": params.physics_contact_margin,
        "foot_radius": params.physics_foot_radius,
        "balance_margin": params.physics_balance_margin,
        "min_air_frames": params.physics_min_air_frames,
        "accel_limit": params.physics_accel_limit,
        "turn_deg": params.physics_turn_deg,
        "min_turn_speed": params.physics_min_turn_speed,
        "blend_frames": params.physics_blend_frames,
        "smooth_sigma": params.physics_smooth_sigma,
        "smooth_pad": params.physics_smooth_pad,
        "fix_ballistic": params.physics_fix_ballistic,
        "fix_momentum": params.physics_fix_momentum,
        "fix_balance": params.physics_fix_balance,
        "max_lean_deg": params.physics_max_lean_deg,
    }


class MOTIONFORGE_OT_physics_check(bpy.types.Operator):
    bl_idname = "motionforge.physics_check"
    bl_label = "Physics Check"
    bl_description = "Report balance/ballistic/momentum errors (no changes)"
    bl_options = {"REGISTER"}

    def execute(self, context):
        try:
            return self._run(context)
        except (clip_io.MotionforgeError, cli.CliError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

    def _run(self, context):
        arm = _active_armature(context)
        action = _active_action(arm)
        scene = context.scene
        exports = {"in.json": clip_io.export_action(arm, action, scene.frame_start, scene.frame_end)}
        args = cli.build_physics_args("physics-check", "{tmp}/in.json", None,
                                      _physics_params(_params(context)))
        report = _run_feature(context, exports, args, lambda tmpdir, rep: None)
        balance = next((line for line in report.splitlines() if line.startswith("balance:")), "")
        self.report({"INFO"}, f"Physics check ({balance.strip()})")
        return {"FINISHED"}


class MOTIONFORGE_OT_physics_fix(bpy.types.Operator):
    bl_idname = "motionforge.physics_fix"
    bl_label = "Physics Fix"
    bl_description = "Fix ballistic, momentum and balance errors via the root curves"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        try:
            return self._run(context)
        except (clip_io.MotionforgeError, cli.CliError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

    def _run(self, context):
        params = _params(context)
        arm = _active_armature(context)
        action = _active_action(arm)
        scene = context.scene
        start, end = scene.frame_start, scene.frame_end
        exports = {"in.json": clip_io.export_action(arm, action, start, end)}
        args = cli.build_physics_args("physics-fix", "{tmp}/in.json", "{tmp}/out.json",
                                      _physics_params(params))

        def do_import(tmpdir, report):
            clip_io.import_clip(arm, _read_json(os.path.join(tmpdir, "out.json")),
                                f"{action.name}_physics", start)

        report = _run_feature(context, exports, args, do_import)
        fixed = next((line for line in report.splitlines() if line.startswith("fixed:")), "")
        self.report({"INFO"}, f"Physics fix ({fixed.strip()})")
        return {"FINISHED"}


class MOTIONFORGE_OT_physics_overlay(bpy.types.Operator):
    bl_idname = "motionforge.physics_overlay"
    bl_label = "Physics Overlay"
    bl_description = "Show COM + support markers for the current frame"
    bl_options = {"REGISTER"}

    def execute(self, context):
        try:
            return self._run(context)
        except (clip_io.MotionforgeError, cli.CliError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

    def _run(self, context):
        arm = _active_armature(context)
        action = _active_action(arm)
        scene = context.scene
        start, end = scene.frame_start, scene.frame_end
        current = scene.frame_current
        if not start <= current <= end:
            raise clip_io.MotionforgeError(
                f"current frame {current} is outside the scene range {start}..{end}"
            )
        exports = {"in.json": clip_io.export_action(arm, action, start, end)}
        args = cli.build_physics_args("physics-frame", "{tmp}/in.json", None,
                                      _physics_params(_params(context)),
                                      frame=current - start)

        def do_import(tmpdir, report):
            from mathutils import Vector

            snap = json.loads(report)
            if snap.get("format") != "motionforge-frame-physics":
                raise clip_io.MotionforgeError("bad physics-frame reply")
            to_world = arm.matrix_world
            com_empty = self._marker(context, "MF_COM", "SPHERE")
            com_empty.location = to_world @ Vector(snap["com"])
            support_empty = self._marker(context, "MF_SUPPORT", "CIRCLE")
            if snap["support_center"] is None:
                support_empty.hide_viewport = True
            else:
                support_empty.hide_viewport = False
                support_empty.location = to_world @ Vector(snap["support_center"])
                radius = snap["support_radius"]
                support_empty.scale = (radius, radius, radius)

        report = _run_feature(context, exports, args, do_import)
        snap = json.loads(report)
        if snap["airborne"]:
            self.report({"INFO"}, f"Frame {current}: airborne, no support")
        else:
            self.report({"INFO"},
                        f"Frame {current}: excursion {snap['excursion_m']:+.4f} m "
                        f"({'balanced' if snap['balanced'] else 'VIOLATION'})")
        return {"FINISHED"}

    @staticmethod
    def _marker(context, name, display_type):
        empty = context.scene.objects.get(name)
        if empty is None:
            empty = bpy.data.objects.new(name, None)
            empty.empty_display_type = display_type
            context.scene.collection.objects.link(empty)
        return empty


class MOTIONFORGE_OT_autopose(bpy.types.Operator):
    bl_idname = "motionforge.autopose"
    bl_label = "AutoPose From Selection"
    bl_description = "Predict the full pose from 1-6 selected pose bones (already moved)"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        try:
            return self._run(context)
        except (clip_io.MotionforgeError, cli.CliError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

    def _run(self, context):
        params = _params(context)
        arm = _active_armature(context)
        selected = context.selected_pose_bones or []
        # Deterministic order: armature order, not selection order.
        order = {b.name: i for i, b in enumerate(arm.data.bones)}
        names = sorted((b.name for b in selected), key=lambda n: order.get(n, 0))
        if not 1 <= len(names) <= 6:
            raise clip_io.MotionforgeError(
                "select 1-6 pose bones in Pose Mode first (the moved joints)"
            )
        weights = bpy.path.abspath(params.autopose_weights)
        if not weights or not os.path.isfile(weights):
            raise clip_io.MotionforgeError("pick an AutoPose weights file in the panel first")
        exports = {"effectors.json": clip_io.export_effectors(arm, names)}
        args = cli.build_autopose_args(weights, "{tmp}/effectors.json", "{tmp}/out.json")

        def do_import(tmpdir, report):
            if arm.animation_data is None:
                arm.animation_data_create()
            if arm.animation_data.action is None:
                arm.animation_data.action = bpy.data.actions.new("AutoPose")
            clip_io.apply_pose_frame(arm, _read_json(os.path.join(tmpdir, "out.json")), key=True)

        _run_feature(context, exports, args, do_import)
        self.report({"INFO"}, f"AutoPose from {len(names)} joints ({', '.join(names)})")
        return {"FINISHED"}


class MOTIONFORGE_OT_export_godot(bpy.types.Operator):
    bl_idname = "motionforge.export_godot"
    bl_label = "Export Game GLB"
    bl_description = "Export via rigforge's game GLB path (needs rigforge enabled)"
    bl_options = {"REGISTER"}

    def execute(self, context):
        if not hasattr(bpy.ops.wm, "rigforge_game_export"):
            self.report({"WARNING"}, "rigforge is not enabled; export from rigforge directly")
            return {"CANCELLED"}
        return bpy.ops.wm.rigforge_game_export("INVOKE_DEFAULT")


class MOTIONFORGE_PT_panel(bpy.types.Panel):
    bl_label = "MotionForge"
    bl_idname = "MOTIONFORGE_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "MotionForge"

    def draw(self, context):
        layout = self.layout
        params = _params(context)

        box = layout.box()
        box.label(text="Retarget", icon="ARMATURE_DATA")
        box.prop(params, "retarget_source")
        box.prop(params, "bonemap_preset")
        if params.bonemap_preset == "CUSTOM":
            box.prop(params, "bonemap_path")
        row = box.row()
        row.prop(params, "retarget_yaw")
        row.prop(params, "retarget_pin")
        box.operator("motionforge.retarget", icon="PLAY")
        box.operator("motionforge.export_godot", icon="EXPORT")

        box = layout.box()
        box.label(text="Stylize", icon="GRAPH")
        box.prop(params, "stylize_exaggeration")
        box.prop(params, "stylize_chains")
        box.prop(params, "stylize_angle_threshold")
        row = box.row()
        row.prop(params, "stylize_min_spacing")
        row.prop(params, "stylize_hold")
        row = box.row()
        row.prop(params, "stylize_anticipation")
        row.prop(params, "stylize_anticipation_frames")
        box.prop(params, "stylize_overshoot")
        box.prop(params, "stylize_thin_keys")
        box.operator("motionforge.stylize", icon="PLAY")

        box = layout.box()
        box.label(text="Physics", icon="PHYSICS")
        box.prop(params, "physics_root")
        box.prop(params, "physics_feet")
        row = box.row()
        row.prop(params, "physics_contact_margin")
        row.prop(params, "physics_foot_radius")
        box.prop(params, "physics_balance_margin")
        row = box.row()
        row.prop(params, "physics_accel_limit")
        row.prop(params, "physics_turn_deg")
        row = box.row()
        row.prop(params, "physics_fix_ballistic")
        row.prop(params, "physics_fix_momentum")
        row = box.row()
        row.prop(params, "physics_fix_balance")
        sub = row.row()
        sub.active = params.physics_fix_balance
        sub.prop(params, "physics_max_lean_deg")
        row = box.row()
        row.operator("motionforge.physics_check", icon="VIEWZOOM")
        row.operator("motionforge.physics_fix", icon="PLAY")
        box.operator("motionforge.physics_overlay", icon="EMPTY_DATA")

        box = layout.box()
        box.label(text="AutoPose", icon="BONE_DATA")
        box.prop(params, "autopose_weights")
        box.label(text="Move 1-6 joints, select them, run:")
        box.operator("motionforge.autopose", icon="PLAY")

        report = params.last_report.strip()
        if report:
            box = layout.box()
            box.label(text="Last Report", icon="TEXT")
            for line in report.splitlines()[:30]:
                box.label(text=line)


_CLASSES = (
    MOTIONFORGE_PG_params,
    MotionForgePreferences,
    MOTIONFORGE_OT_retarget,
    MOTIONFORGE_OT_stylize,
    MOTIONFORGE_OT_physics_check,
    MOTIONFORGE_OT_physics_fix,
    MOTIONFORGE_OT_physics_overlay,
    MOTIONFORGE_OT_autopose,
    MOTIONFORGE_OT_export_godot,
    MOTIONFORGE_PT_panel,
)


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.motionforge_params = PointerProperty(type=MOTIONFORGE_PG_params)


def unregister():
    del bpy.types.Scene.motionforge_params
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
