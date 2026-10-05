# SPDX-License-Identifier: MIT
"""Render a pose-test GLB into one labeled contact sheet (runs INSIDE Blender).

    blender --background --factory-startup --python pose_sheet.py -- \
        POSES.glb SHEET.png [--forward x,y,z] [--cols 6] [--tile 320x400]
        [--title TEXT]

POSES.glb is what `motionforge pose-test` writes: the character plus one
single-key animation per pose (all joints keyed). Each animation is
rendered as one tile from a three-quarter front view (camera placed from
the character's glTF forward vector), labeled with the animation name,
and the tiles are packed into SHEET.png. Prints `MOTIONFORGE_SHEET_OK
<path>` on success; any failure exits 1 with a reason on stderr.

Embedded in the motionforge binary (`include_str!`), so the CLI works
from any install location. MIT like the CLI it ships in: it only calls
Blender's Python API.
"""

import math
import os
import sys
import tempfile

import bpy
import numpy as np
from mathutils import Vector


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    if len(argv) < 2:
        raise SystemExit("usage: pose_sheet.py -- POSES.glb SHEET.png [flags]")
    opts = {"forward": "0,0,1", "cols": "6", "tile": "320x400", "title": ""}
    i = 2
    while i < len(argv):
        key = argv[i].lstrip("-")
        if key not in opts or i + 1 >= len(argv):
            raise SystemExit(f"unknown or incomplete flag {argv[i]}")
        opts[key] = argv[i + 1]
        i += 2
    fx, fy, fz = (float(v) for v in opts["forward"].split(","))
    tw, th = (int(v) for v in opts["tile"].split("x"))
    # glTF (+Y up) -> Blender (+Z up): (x, y, z) -> (x, -z, y).
    forward = Vector((fx, -fz, fy))
    forward.z = 0.0
    if forward.length < 1e-6:
        forward = Vector((0.0, -1.0, 0.0))
    forward.normalize()
    return argv[0], argv[1], forward, int(opts["cols"]), tw, th, opts["title"]


def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_glb(path):
    bpy.ops.import_scene.gltf(filepath=path)
    arms = [o for o in bpy.context.scene.objects if o.type == "ARMATURE"]
    # The importer also adds a hidden bone-shape mesh (Icosphere): keep
    # visible meshes, preferring those the armature deforms.
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH" and o.visible_get() and not o.hide_render]
    bound = [o for o in meshes if o.find_armature() is not None]
    meshes = bound or meshes
    if not arms or not meshes:
        raise RuntimeError("GLB has no armature or no mesh")
    return arms[0], meshes


def actions_in_file_order(arm):
    acts = [a for a in bpy.data.actions]
    # glTF importer names actions after the animations; our names sort
    # in sheet order ("01 rest", "02 ...", then clip samples).
    return sorted(acts, key=lambda a: a.name)


def set_action(arm, act):
    ad = arm.animation_data or arm.animation_data_create()
    ad.action = act
    slots = getattr(act, "slots", None)
    if slots is not None and len(slots) and hasattr(ad, "action_slot"):
        ad.action_slot = slots[0]
    start = int(act.frame_range[0])
    bpy.context.scene.frame_set(start)
    bpy.context.view_layer.update()


def bounds(meshes):
    deps = bpy.context.evaluated_depsgraph_get()
    lo = Vector((1e9, 1e9, 1e9))
    hi = Vector((-1e9, -1e9, -1e9))
    for o in meshes:
        eo = o.evaluated_get(deps)
        me = eo.to_mesh()
        mw = eo.matrix_world
        for v in me.vertices:
            p = mw @ v.co
            lo = Vector(map(min, lo, p))
            hi = Vector(map(max, hi, p))
        eo.to_mesh_clear()
    return lo, hi


def setup(forward, tw, th):
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = tw
    scene.render.resolution_y = th
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.view_settings.view_transform = "Standard"
    try:
        scene.eevee.taa_render_samples = 16
    except AttributeError:
        pass
    world = bpy.data.worlds.new("sheet")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.17, 0.18, 0.2, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.9
    scene.world = world
    for name, energy, rot in (("key", 3.5, (0.9, 0.2, 0.6)), ("fill", 1.2, (1.1, 0.0, -2.2))):
        light = bpy.data.lights.new(name, "SUN")
        light.energy = energy
        obj = bpy.data.objects.new(name, light)
        obj.rotation_euler = rot
        scene.collection.objects.link(obj)
    cam_data = bpy.data.cameras.new("cam")
    cam_data.type = "ORTHO"
    cam = bpy.data.objects.new("cam", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    label_data = bpy.data.curves.new("label", "FONT")
    label_data.align_x = "CENTER"
    label = bpy.data.objects.new("label", label_data)
    mat = bpy.data.materials.new("label")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = (1, 1, 1, 1)
        if "Emission Color" in bsdf.inputs:
            bsdf.inputs["Emission Color"].default_value = (1, 1, 1, 1)
            bsdf.inputs["Emission Strength"].default_value = 1.0
    label_data.materials.append(mat)
    scene.collection.objects.link(label)
    return cam, label


def frame_camera(cam, label, forward, lo, hi, tw, th, text):
    """Orthographic three-quarter view; the label rides on the camera."""
    center = (lo + hi) / 2
    height = max(hi.z - lo.z, 1e-3)
    radius = max((hi - lo).xy.length / 2, 1e-3)
    ang = math.radians(40)
    view = Vector(
        (
            forward.x * math.cos(ang) - forward.y * math.sin(ang),
            forward.x * math.sin(ang) + forward.y * math.cos(ang),
            0.0,
        )
    )
    dist = 10.0 * max(height, radius)
    vertical = th / tw  # tile height / width
    # ortho_scale spans the larger image side (height for tall tiles).
    span_v = height * 1.18  # figure + room for the label underneath
    span_h = 2 * radius * 1.08
    if vertical >= 1:
        scale = max(span_v, span_h * vertical)
    else:
        scale = max(span_v / vertical, span_h)
    cam.data.ortho_scale = scale
    cam.data.clip_end = dist * 4
    # Shift the view up so the figure sits above the label band.
    look = center - Vector((0, 0, 0.06 * scale))
    cam.location = look + view * dist
    cam.rotation_euler = (look - cam.location).to_track_quat("-Z", "Y").to_euler()
    bpy.context.view_layer.update()
    half_v = scale / 2 if vertical >= 1 else scale * vertical / 2
    label.parent = cam
    label.matrix_parent_inverse.identity()
    label.location = (0.0, -half_v * 0.9, -1.0)
    label.rotation_euler = (0.0, 0.0, 0.0)
    label.data.body = text
    label.data.size = scale * 0.052


def render_tile(path):
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    img = bpy.data.images.load(path)
    w, h = img.size
    px = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(px)
    bpy.data.images.remove(img)
    return px.reshape(h, w, 4)


def main():
    src, out, forward, cols, tw, th, title = parse_args()
    reset_scene()
    arm, meshes = import_glb(src)
    acts = actions_in_file_order(arm)
    if not acts:
        raise RuntimeError("GLB has no pose animations")
    cam, label = setup(forward, tw, th)
    # Frame every tile the same way: union of all pose bounds.
    lo = Vector((1e9, 1e9, 1e9))
    hi = Vector((-1e9, -1e9, -1e9))
    for act in acts:
        set_action(arm, act)
        a, b = bounds(meshes)
        lo = Vector(map(min, lo, a))
        hi = Vector(map(max, hi, b))
    pad = (hi - lo) * 0.04
    lo, hi = lo - pad, hi + pad
    tiles = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, act in enumerate(acts):
            set_action(arm, act)
            frame_camera(cam, label, forward, lo, hi, tw, th, act.name)
            tiles.append(render_tile(os.path.join(tmp, f"tile{i:03d}.png")))
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.zeros((rows * th, cols * tw, 4), dtype=np.float32)
    sheet[..., :3] = 0.08
    sheet[..., 3] = 1.0
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        # Blender pixel rows start at the bottom; sheet row 0 is the top.
        y0 = (rows - 1 - r) * th
        sheet[y0 : y0 + th, c * tw : (c + 1) * tw] = tile
        # 2 px gutter lines.
        sheet[y0 : y0 + th, c * tw : c * tw + 2, :3] = 0.08
        sheet[y0 : y0 + 2, c * tw : (c + 1) * tw, :3] = 0.08
    img = bpy.data.images.new("sheet", cols * tw, rows * th, alpha=False)
    img.pixels.foreach_set(sheet.ravel())
    img.filepath_raw = out
    img.file_format = "PNG"
    img.save()
    print(f"MOTIONFORGE_SHEET_OK {out} tiles={len(tiles)} title={title}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 - CLI verdict line
        print(f"pose_sheet: {exc}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
