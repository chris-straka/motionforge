# SPDX-License-Identifier: MIT
"""Render the clips of an animated GLB as a contact sheet and/or video
(runs INSIDE Blender).

    blender --background --factory-startup --python clip_sheet.py -- \\
        ANIMATED.glb SHEET.png [--clips a,b,c] [--frames 6] [--tile 260x330]
        [--sword SWORD.glb] [--sword-scale 1.0] [--video OUT.mp4]
        [--video-tile 480x540] [--ffmpeg PATH] [--title TEXT]
        [--at clip@0.40,clip@1.2] [--compare OTHER.glb]

One row per clip (file order, or --clips order), --frames evenly spaced
samples per row, each an orthographic three-quarter view framed on that
clip's whole motion (so travel shows), labeled "clip t=0.42s". With
--sword, the GLB's first mesh is parented to the `Socket_Hand_R` node
(motionforge standardize adds it; blade along the socket's +Y), scaled by
--sword-scale times character height / 2.0 (HLL's sword is sized for a
2.0-unit Andras). With --video, every frame of every clip is rendered at
30 fps and joined into one MP4 with ffmpeg. With --at, each listed
clip@seconds is one row instead, framed on that pose; with --compare the
same rows are rendered from OTHER.glb too and placed after the first
file's tiles (before | after), from two angles each. Prints
`MOTIONFORGE_CLIPSHEET_OK <path>`.

MIT like the motionforge CLI: only Blender's Python API.
"""

import math
import os
import shutil
import subprocess
import sys
import tempfile

import bpy
import numpy as np
from mathutils import Matrix, Vector


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    if len(argv) < 2:
        raise SystemExit("usage: clip_sheet.py -- ANIMATED.glb SHEET.png [flags]")
    opts = {
        "clips": "",
        "frames": "6",
        "tile": "260x330",
        "sword": "",
        "sword-scale": "1.0",
        "video": "",
        "video-tile": "480x540",
        "ffmpeg": "",
        "title": "",
        "at": "",
        "compare": "",
    }
    i = 2
    while i < len(argv):
        key = argv[i].lstrip("-")
        if key not in opts or i + 1 >= len(argv):
            raise SystemExit(f"unknown or incomplete flag {argv[i]}")
        opts[key] = argv[i + 1]
        i += 2
    opts["src"] = os.path.abspath(argv[0])
    opts["out"] = os.path.abspath(argv[1])
    return opts


def import_glb(path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=path)
    arms = [o for o in bpy.context.scene.objects if o.type == "ARMATURE"]
    meshes = [
        o
        for o in bpy.context.scene.objects
        if o.type == "MESH" and o.find_armature() is not None
    ]
    if not arms or not meshes:
        raise RuntimeError("GLB has no armature or no skinned mesh")
    return arms[0], meshes


def find_socket(arm):
    for o in bpy.context.scene.objects:
        if o.name.startswith("Socket_Hand_R"):
            return o
    # The importer may turn a node under a joint into a bone.
    if "Socket_Hand_R" in arm.data.bones:
        return ("bone", "Socket_Hand_R")
    return None


def attach_sword(arm, path, scale):
    socket = find_socket(arm)
    if socket is None:
        print("clip_sheet: no Socket_Hand_R, sword skipped", file=sys.stderr)
        return
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=path)
    new = [o for o in bpy.data.objects if o not in before]
    roots = [o for o in new if o.parent is None]
    holder = bpy.data.objects.new("SwordHolder", None)
    bpy.context.scene.collection.objects.link(holder)
    for o in roots:
        o.parent = holder
    # glTF socket +Y (blade) is Blender's +Z after the importer's Y-up
    # conversion of the sword itself, so identity under the socket matches
    # the game's held pose.
    if isinstance(socket, tuple):
        holder.parent = arm
        holder.parent_type = "BONE"
        holder.parent_bone = socket[1]
        holder.matrix_parent_inverse = Matrix.Translation(
            (0, -arm.data.bones[socket[1]].length, 0)
        )
    else:
        holder.parent = socket
    holder.scale = (scale, scale, scale)


def setup(tw, th):
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = tw
    scene.render.resolution_y = th
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.view_settings.view_transform = "Standard"
    scene.render.fps = 30
    try:
        scene.eevee.taa_render_samples = 8
    except AttributeError:
        pass
    world = bpy.data.worlds.new("sheet")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (
        0.17,
        0.18,
        0.2,
        1,
    )
    scene.world = world
    for name, energy, rot in (
        ("key", 3.5, (0.9, 0.2, 0.6)),
        ("fill", 1.2, (1.1, 0.0, -2.2)),
    ):
        light = bpy.data.lights.new(name, "SUN")
        light.energy = energy
        obj = bpy.data.objects.new(name, light)
        obj.rotation_euler = rot
        scene.collection.objects.link(obj)
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    cam.data.type = "ORTHO"
    scene.collection.objects.link(cam)
    scene.camera = cam
    label_data = bpy.data.curves.new("label", "FONT")
    label_data.align_x = "CENTER"
    label = bpy.data.objects.new("label", label_data)
    mat = bpy.data.materials.new("label")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None and "Emission Color" in bsdf.inputs:
        bsdf.inputs["Emission Color"].default_value = (1, 1, 1, 1)
        bsdf.inputs["Emission Strength"].default_value = 1.0
    label_data.materials.append(mat)
    scene.collection.objects.link(label)
    return cam, label


def set_action(arm, act):
    ad = arm.animation_data or arm.animation_data_create()
    ad.action = act
    slots = getattr(act, "slots", None)
    if slots is not None and len(slots) and hasattr(ad, "action_slot"):
        ad.action_slot = slots[0]


def goto(act, t):
    f = act.frame_range[0] + t * 30.0
    bpy.context.scene.frame_set(math.floor(f), subframe=f - math.floor(f))
    bpy.context.view_layer.update()


def bounds(objs):
    deps = bpy.context.evaluated_depsgraph_get()
    lo = Vector((1e9, 1e9, 1e9))
    hi = Vector((-1e9, -1e9, -1e9))
    for o in objs:
        eo = o.evaluated_get(deps)
        me = eo.to_mesh()
        mw = eo.matrix_world
        for v in me.vertices:
            p = mw @ v.co
            lo = Vector(map(min, lo, p))
            hi = Vector(map(max, hi, p))
        eo.to_mesh_clear()
    return lo, hi


def forward_of(arm):
    """Character front: heel-to-toe of the feet (Blender space), else -Y."""
    sums = Vector((0, 0, 0))
    for side in ("L", "R"):
        f, t = (
            arm.data.bones.get(f"DEF-foot.{side}"),
            arm.data.bones.get(f"DEF-toe.{side}"),
        )
        if f and t:
            sums += (arm.matrix_world @ t.head_local) - (
                arm.matrix_world @ f.head_local
            )
    sums.z = 0
    return sums.normalized() if sums.length > 1e-6 else Vector((0, -1, 0))


def frame_camera(cam, label, forward, lo, hi, tw, th, text):
    center = (lo + hi) / 2
    height = max(hi.z - lo.z, 1e-3)
    radius = max((hi - lo).xy.length / 2, 1e-3)
    ang = math.radians(35)
    view = Vector(
        (
            forward.x * math.cos(ang) - forward.y * math.sin(ang),
            forward.x * math.sin(ang) + forward.y * math.cos(ang),
            0.18,
        )
    ).normalized()
    dist = 10.0 * max(height, radius)
    vertical = th / tw
    span_v = height * 1.2
    span_h = 2 * radius * 1.05
    scale = (
        max(span_v, span_h * vertical)
        if vertical >= 1
        else max(span_v / vertical, span_h)
    )
    cam.data.ortho_scale = scale
    cam.data.clip_end = dist * 4
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
    label.data.size = scale * 0.05


def render_to(path):
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(write_still=True)


def load_px(path):
    img = bpy.data.images.load(path)
    w, h = img.size
    px = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(px)
    bpy.data.images.remove(img)
    return px.reshape(h, w, 4)


def clip_bounds(arm, meshes, act, samples):
    lo = Vector((1e9, 1e9, 1e9))
    hi = Vector((-1e9, -1e9, -1e9))
    for t in samples:
        goto(act, t)
        a, b = bounds(meshes)
        lo = Vector(map(min, lo, a))
        hi = Vector(map(max, hi, b))
    pad = (hi - lo) * 0.05
    return lo - pad, hi + pad


def render_at(path, picks, tw, th, o, tmp, tag):
    """Tiles for clip@time picks from one file: two angles per pick."""
    arm, meshes = import_glb(path)
    lo0, hi0 = bounds(meshes)
    if o["sword"]:
        attach_sword(
            arm,
            os.path.abspath(o["sword"]),
            float(o["sword-scale"]) * (hi0.z - lo0.z) / 2.0,
        )
    acts = {a.name: a for a in bpy.data.actions}
    forward = forward_of(arm)
    cam, label = setup(tw, th)
    rows = []
    for i, (name, t) in enumerate(picks):
        act = acts.get(name)
        if act is None:
            raise RuntimeError(f"no clip {name} in {path}")
        set_action(arm, act)
        goto(act, t)
        lo, hi = bounds(meshes)
        pad = (hi - lo) * 0.08
        lo, hi = lo - pad, hi + pad
        row = []
        for k, turn in enumerate((0.0, 70.0)):
            c, s_ = math.cos(math.radians(turn)), math.sin(math.radians(turn))
            f2 = Vector(
                (forward.x * c - forward.y * s_, forward.x * s_ + forward.y * c, 0.0)
            )
            frame_camera(cam, label, f2, lo, hi, tw, th, f"{tag}  {name} {t:.2f}s")
            p = os.path.join(tmp, f"{tag}{i:03d}{k}.png")
            render_to(p)
            row.append(load_px(p))
        rows.append(row)
    return rows


def main_at(o):
    picks = []
    for item in o["at"].split(","):
        name, t = item.rsplit("@", 1)
        picks.append((name.strip(), float(t)))
    tw, th = (int(v) for v in o["tile"].split("x"))
    with tempfile.TemporaryDirectory() as tmp:
        rows = render_at(
            o["src"], picks, tw, th, o, tmp, "before" if o["compare"] else ""
        )
        if o["compare"]:
            after = render_at(
                os.path.abspath(o["compare"]), picks, tw, th, o, tmp, "after"
            )
            rows = [a + b for a, b in zip(rows, after)]
        cols = len(rows[0])
        n = len(rows)
        sheet = np.zeros((n * th, cols * tw, 4), dtype=np.float32)
        sheet[..., 3] = 1.0
        for r, row in enumerate(rows):
            for c, tile in enumerate(row):
                y0 = (n - 1 - r) * th
                sheet[y0 : y0 + th, c * tw : (c + 1) * tw] = tile
                sheet[y0 : y0 + th, c * tw : c * tw + 2, :3] = 0.08
                sheet[y0 : y0 + 2, c * tw : (c + 1) * tw, :3] = 0.08
                if o["compare"] and c == cols // 2:
                    sheet[y0 : y0 + th, c * tw : c * tw + 6, :3] = (0.9, 0.6, 0.1)
        img = bpy.data.images.new("sheet", cols * tw, n * th, alpha=False)
        img.pixels.foreach_set(sheet.ravel())
        img.filepath_raw = o["out"]
        img.file_format = "PNG"
        img.save()
    print(f"MOTIONFORGE_CLIPSHEET_OK {o['out']} rows={len(picks)}")


def main():
    o = parse_args()
    if o["at"]:
        return main_at(o)
    arm, meshes = import_glb(o["src"])
    lo0, hi0 = bounds(meshes)
    char_h = hi0.z - lo0.z
    if o["sword"]:
        attach_sword(
            arm, os.path.abspath(o["sword"]), float(o["sword-scale"]) * char_h / 2.0
        )
    acts = {a.name: a for a in bpy.data.actions}
    names = [c for c in o["clips"].split(",") if c] or sorted(acts)
    names = [n for n in names if n in acts]
    if not names:
        raise RuntimeError("no matching clips")
    forward = forward_of(arm)
    tw, th = (int(v) for v in o["tile"].split("x"))
    cam, label = setup(tw, th)
    cols = int(o["frames"])
    sheet_rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for ci, name in enumerate(names):
            act = acts[name]
            set_action(arm, act)
            dur = (act.frame_range[1] - act.frame_range[0]) / 30.0
            samples = [dur * k / max(cols - 1, 1) for k in range(cols)]
            lo, hi = clip_bounds(arm, meshes, act, [dur * k / 8 for k in range(9)])
            row = []
            for k, t in enumerate(samples):
                goto(act, t)
                frame_camera(cam, label, forward, lo, hi, tw, th, f"{name}  {t:.2f}s")
                p = os.path.join(tmp, f"r{ci:03d}c{k:02d}.png")
                render_to(p)
                row.append(load_px(p))
            sheet_rows.append(row)
        rows = len(sheet_rows)
        sheet = np.zeros((rows * th, cols * tw, 4), dtype=np.float32)
        sheet[..., 3] = 1.0
        for r, row in enumerate(sheet_rows):
            for c, tile in enumerate(row):
                y0 = (rows - 1 - r) * th
                sheet[y0 : y0 + th, c * tw : (c + 1) * tw] = tile
                sheet[y0 : y0 + th, c * tw : c * tw + 2, :3] = 0.08
                sheet[y0 : y0 + 2, c * tw : (c + 1) * tw, :3] = 0.08
        img = bpy.data.images.new("sheet", cols * tw, rows * th, alpha=False)
        img.pixels.foreach_set(sheet.ravel())
        img.filepath_raw = o["out"]
        img.file_format = "PNG"
        img.save()
        if o["video"]:
            vw, vh = (int(v) for v in o["video-tile"].split("x"))
            scene = bpy.context.scene
            scene.render.resolution_x, scene.render.resolution_y = vw, vh
            frames_dir = os.path.join(tmp, "video")
            os.makedirs(frames_dir)
            idx = 0
            for name in names:
                act = acts[name]
                set_action(arm, act)
                dur = (act.frame_range[1] - act.frame_range[0]) / 30.0
                lo, hi = clip_bounds(arm, meshes, act, [dur * k / 8 for k in range(9)])
                n = round(dur * 30) + 1
                # Loop short clips so each shows for at least 1.5 s.
                reps = max(1, math.ceil(45 / n))
                for _ in range(reps):
                    for f in range(n):
                        goto(act, f / 30.0)
                        frame_camera(cam, label, forward, lo, hi, vw, vh, name)
                        render_to(os.path.join(frames_dir, f"f{idx:06d}.png"))
                        idx += 1
            ff = o["ffmpeg"] or shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
            subprocess.run(
                [
                    ff,
                    "-y",
                    "-loglevel",
                    "error",
                    "-framerate",
                    "30",
                    "-i",
                    os.path.join(frames_dir, "f%06d.png"),
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-crf",
                    "20",
                    os.path.abspath(o["video"]),
                ],
                check=True,
            )
    print(f"MOTIONFORGE_CLIPSHEET_OK {o['out']} clips={len(names)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 - CLI verdict line
        print(f"clip_sheet: {exc}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
