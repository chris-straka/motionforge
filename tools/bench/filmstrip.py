"""Filmstrip of a retargeted clip, headless Blender (Workbench):

    blender -b --factory-startup --python tools/bench/filmstrip.py -- \
        OUT.png CLIP_NAME LABEL=anim.glb [LABEL=anim.glb ...]

One row per GLB: the clip at 8 evenly spaced frames, side view, camera
fixed in the world (it does not follow the character) over a 10 cm ground
grid, so a planted foot that slides shows as a foot moving against the
grid lines between frames. Feet are tinted orange. Rows are written as
tiles next to OUT (OUT.<row>.<frame>.png); bench.py composes the sheet.
"""

import math
import sys

import bpy

argv = sys.argv[sys.argv.index("--") + 1:]
OUT, CLIP = argv[0], argv[1]
ROWS = [a.split("=", 1) for a in argv[2:]]
FRAMES = 8


def clear():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)
    for a in list(bpy.data.actions):
        bpy.data.actions.remove(a)


def setup(height, cx, cy):
    scn = bpy.context.scene
    scn.render.engine = 'BLENDER_WORKBENCH'
    scn.display.shading.light = 'STUDIO'
    scn.display.shading.color_type = 'MATERIAL'
    scn.render.resolution_x, scn.render.resolution_y = 900, 300
    scn.world = scn.world or bpy.data.worlds.new('w')
    scn.world.color = (1, 1, 1)
    scn.display.shading.background_type = 'WORLD'
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    scn.collection.objects.link(cam)
    cam.data.type = 'ORTHO'
    cam.data.ortho_scale = height * 0.8
    cam.data.clip_end = 100
    # side view of the legs, from the character's left
    cam.location = (cx + 6 * height, cy, height * 0.25)
    cam.rotation_euler = (math.radians(90), 0, math.radians(90))
    scn.camera = cam
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=60, y_subdivisions=60, size=6, location=(cx, cy, 0))
    grid = bpy.context.active_object
    wf = grid.modifiers.new("wf", 'WIREFRAME')
    wf.thickness = 0.004
    mat = bpy.data.materials.new("grid")
    mat.diffuse_color = (0.75, 0.75, 0.8, 1)
    grid.data.materials.append(mat)


def render_row(r, path):
    clear()
    bpy.ops.import_scene.gltf(filepath=path)
    arm = next(o for o in bpy.data.objects if o.type == 'ARMATURE')
    meshes = [o for o in bpy.data.objects if o.type == 'MESH' and o.visible_get()]
    act = bpy.data.actions.get(CLIP) or next((a for a in bpy.data.actions if a.name.startswith(CLIP)), None)
    if act is None:
        raise SystemExit(f"no clip {CLIP} in {path}")
    arm.animation_data_create()
    arm.animation_data.action = act
    f0, f1 = act.frame_range
    body = bpy.data.materials.new("body")
    body.diffuse_color = (0.25, 0.3, 0.45, 1)
    for o in meshes:
        o.data.materials.clear()
        o.data.materials.append(body)
    zs = [(o.matrix_world @ v.co).z for o in meshes for v in o.data.vertices]
    height = max(zs) - min(zs)
    scn = bpy.context.scene
    scn.frame_set(int(f0))
    pb = arm.pose.bones
    hips = next((b for b in pb if b.name in ("DEF-spine", "pelvis", "Hips")), pb[0])
    start = arm.matrix_world @ hips.head
    scn.frame_set(int(f1))
    end = arm.matrix_world @ hips.head
    setup(height, (start.x + end.x) / 2, (start.y + end.y) / 2)
    cam = scn.camera
    span = max(abs(end.x - start.x), abs(end.y - start.y))
    cam.data.ortho_scale = max(height * 0.8, span + height * 0.5)
    cam.location.z = cam.data.ortho_scale * 300 / 900 / 2 - height * 0.02
    for k in range(FRAMES):
        f = f0 + (f1 - f0) * k / (FRAMES - 1)
        scn.frame_set(int(round(f)))
        scn.render.filepath = f"{OUT}.{r}.{k}.png"
        bpy.ops.render.render(write_still=True)


for r, (label, path) in enumerate(ROWS):
    render_row(r, path)
print("FILMSTRIP", OUT)
