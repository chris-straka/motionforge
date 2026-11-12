"""Render clips of rigged GLBs frame by frame for strips and reels
(headless Blender, Workbench):

    blender -b --factory-startup --python tools/bench/motion_render.py -- SPEC.json

SPEC = {
  "out": "dir/prefix",              # tiles: <out>.r<row>.<i:04d>.png
  "size": [480, 540],
  "view": "3q" | "front" | "side",  # 3q: front, 30 deg toward the right hand
  "shared_camera": true,            # one framing for every row (fair compare)
  "trail": "DEF-hand.R",            # optional: that joint's path over the clip
  "rows": [{"label": "orig", "glb": "a.glb", "clip": "attack_1",
            "frames": "all" | [0, 5, 9], "mark": [9]}]
}

The camera is fixed in the world (it does not follow the character) over a
10 cm ground grid, so a sliding foot shows as a foot moving against the
grid. The trail (red) is the joint's whole path; a red ball sits on the
joint. Each tile carries its row label and frame; marked frames say so.
"""

import json
import math
import sys

import bpy
from mathutils import Vector

SPEC = json.load(open(sys.argv[sys.argv.index("--") + 1:][0]))
W, H = SPEC.get("size", [480, 540])


def clear():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)
    for a in list(bpy.data.actions):
        bpy.data.actions.remove(a)
    for c in list(bpy.data.curves):
        bpy.data.curves.remove(c)


def mat(name, rgba):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.diffuse_color = rgba
    return m


def load(row):
    clear()
    # Clips are keyed at 30 fps: import one key per scene frame.
    bpy.context.scene.render.fps = 30
    bpy.context.scene.render.fps_base = 1.0
    bpy.ops.import_scene.gltf(filepath=row["glb"])
    arm = next(o for o in bpy.data.objects if o.type == 'ARMATURE')
    meshes = [o for o in bpy.data.objects if o.type == 'MESH']
    act = bpy.data.actions.get(row["clip"])
    if act is None:
        cands = [a for a in bpy.data.actions if a.name.split("_Armature")[0] == row["clip"]
                 or a.name.startswith(row["clip"] + "_")]
        cands.sort(key=lambda a: len(a.name))
        act = cands[0] if cands else None
    if act is None:
        raise SystemExit(f"no clip {row['clip']} in {row['glb']}: {[a.name for a in bpy.data.actions]}")
    arm.animation_data_create()
    arm.animation_data.action = act
    if hasattr(arm.animation_data, "action_slot") and act.slots:
        arm.animation_data.action_slot = act.slots[0]
    for o in meshes:
        o.data.materials.clear()
        o.data.materials.append(mat("body", (0.32, 0.38, 0.52, 1)))
    f0, f1 = (int(round(x)) for x in act.frame_range)
    return arm, meshes, f0, f1


def joint(arm, name):
    pb = arm.pose.bones.get(name)
    return (arm.matrix_world @ pb.head) if pb else None


def points(arm, meshes, f0, f1):
    scn = bpy.context.scene
    pts = []
    for f in range(f0, f1 + 1):
        scn.frame_set(f)
        pts += [arm.matrix_world @ b.head for b in arm.pose.bones]
    scn.frame_set(f0)
    dg = bpy.context.evaluated_depsgraph_get()
    zs = []
    for o in meshes:
        ev = o.evaluated_get(dg)
        zs += [(ev.matrix_world @ v.co).z for v in ev.data.vertices]
    height = (max(zs) - min(zs)) if zs else 1.8
    return pts, height


def facing(arm):
    a, b = joint(arm, "DEF-foot.L"), joint(arm, "DEF-toe.L")
    if a is None or b is None:
        return Vector((0, -1, 0))
    d = Vector((b.x - a.x, b.y - a.y, 0))
    return d.normalized() if d.length > 1e-6 else Vector((0, -1, 0))


def camera_basis(fwd):
    up = Vector((0, 0, 1))
    view = SPEC.get("view", "3q")
    right = fwd.cross(up).normalized()
    if view == "side":
        d = (-right).normalized()  # from the character's left
    elif view == "front":
        d = fwd
    else:
        a = math.radians(30)
        d = (fwd * math.cos(a) + right * math.sin(a)).normalized()
    return d, up


def setup(frame_pts, height, fwd):
    scn = bpy.context.scene
    scn.render.engine = 'BLENDER_WORKBENCH'
    scn.display.shading.light = 'STUDIO'
    scn.display.shading.color_type = 'MATERIAL'
    scn.display.shading.show_shadows = True
    scn.render.resolution_x, scn.render.resolution_y = W, H
    scn.render.resolution_percentage = 100
    scn.world = scn.world or bpy.data.worlds.new('w')
    scn.world.color = (1, 1, 1)
    scn.display.shading.background_type = 'WORLD'
    d, up = camera_basis(fwd)
    side = d.cross(up).normalized()
    c = sum(frame_pts, Vector()) / len(frame_pts)
    xs = [(p - c).dot(side) for p in frame_pts]
    zs = [p.z for p in frame_pts] + [0.0, height]
    ext_x = max(xs) - min(xs) + 0.5 * height
    ext_z = max(zs) - min(zs) + 0.15 * height
    scale = max(ext_x, ext_z * W / H) * 1.05
    mid_x = (max(xs) + min(xs)) / 2
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    scn.collection.objects.link(cam)
    cam.data.type = 'ORTHO'
    cam.data.ortho_scale = scale
    cam.data.clip_end = 1000
    target = c + side * mid_x
    target.z = (max(zs) + min(zs)) / 2
    cam.location = target + d * 20 * height + Vector((0, 0, 0.6 * height))
    cam.rotation_euler = (target - cam.location).to_track_quat('-Z', 'Y').to_euler()
    scn.camera = cam
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=80, y_subdivisions=80, size=8, location=(c.x, c.y, 0))
    grid = bpy.context.active_object
    wf = grid.modifiers.new("wf", 'WIREFRAME')
    wf.thickness = 0.003
    grid.data.materials.append(mat("grid", (0.72, 0.72, 0.78, 1)))
    # Label, fixed to the camera's view.
    txt = bpy.data.objects.new("label", bpy.data.curves.new("label", 'FONT'))
    scn.collection.objects.link(txt)
    txt.parent = cam
    txt.data.size = scale * 0.045
    txt.location = (-scale / 2 * 0.95, scale * H / W / 2 * 0.86, -1.0)
    txt.data.materials.append(mat("ink", (0, 0, 0, 1)))
    return cam, txt, scale


def trail(arm, name, f0, f1, scale):
    scn = bpy.context.scene
    pts = []
    for f in range(f0, f1 + 1):
        scn.frame_set(f)
        p = joint(arm, name)
        if p is not None:
            pts.append(p)
    if len(pts) < 2:
        return
    cu = bpy.data.curves.new("trail", 'CURVE')
    cu.dimensions = '3D'
    cu.bevel_depth = scale * 0.0025
    sp = cu.splines.new('POLY')
    sp.points.add(len(pts) - 1)
    for i, p in enumerate(pts):
        sp.points[i].co = (p.x, p.y, p.z, 1)
    ob = bpy.data.objects.new("trail", cu)
    ob.data.materials.append(mat("red", (0.85, 0.12, 0.1, 1)))
    scn.collection.objects.link(ob)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=scale * 0.012)
    ball = bpy.context.active_object
    ball.data.materials.append(mat("red", (0.85, 0.12, 0.1, 1)))
    ball.parent = arm
    ball.parent_type = 'BONE'
    ball.parent_bone = name
    ball.location = (0, -arm.data.bones[name].length, 0)
    scn.frame_set(f0)


def main():
    rows = SPEC["rows"]
    loaded = []
    # Shared camera: frame the union of every row's whole clip.
    union = []
    hmax = 0
    fwd = None
    if SPEC.get("shared_camera", True):
        for row in rows:
            arm, meshes, f0, f1 = load(row)
            pts, h = points(arm, meshes, f0, f1)
            union += pts
            hmax = max(hmax, h)
            fwd = fwd or facing(arm)
    for r, row in enumerate(rows):
        arm, meshes, f0, f1 = load(row)
        if SPEC.get("shared_camera", True):
            pts, h = union, hmax
            fw = fwd
        else:
            pts, h = points(arm, meshes, f0, f1)
            fw = facing(arm)
        cam, txt, scale = setup(pts, h, fw)
        if SPEC.get("trail") and SPEC["trail"] in arm.data.bones:
            trail(arm, SPEC["trail"], f0, f1, scale)
        frames = row.get("frames", "all")
        frames = list(range(0, f1 - f0 + 1)) if frames == "all" else frames
        marks = set(row.get("mark", []))
        scn = bpy.context.scene
        for i, f in enumerate(frames):
            scn.frame_set(f0 + f)
            tag = "  << worst" if f in marks else ""
            txt.data.body = f"{row['label']}  {f}{tag}"
            txt.data.materials[0] = mat("mark", (0.8, 0.05, 0.05, 1)) if f in marks else mat("ink", (0, 0, 0, 1))
            scn.render.filepath = f"{SPEC['out']}.r{r}.{i:04d}.png"
            bpy.ops.render.render(write_still=True)
        loaded.append((row["label"], len(frames)))
    print("MOTION_RENDER", json.dumps(loaded))


main()
