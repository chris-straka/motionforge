"""Render clips of rigged GLBs frame by frame as the game shows them
(headless Blender; driven by `motion_video.py`):

    blender -b --factory-startup --python tools/bench/motion_video_render.py -- SPEC.json

SPEC = {
  "out": "dir/prefix",               # frames: <out>.r<row>.<i:04d>.png
  "size": [960, 540],
  "cam": "game" | "lock" | "close" | "hand" | "gamefit",  # see CAMERA
                                     # below ("hand": close on the sword hand;
                                     # "gamefit": the game camera's angle,
                                     # moved in so the move fills the frame)
  "yaw_deg": 0,                      # game/lock: orbit round him (0 = behind)
  "engine": "eevee" | "workbench",
  "prop": "sword.glb@Socket_Hand_R", # optional: a model held at a socket
  "prop_pose": "socket" | "game",    # blade along the socket's +Y, or the
                                     # game's `tracks::held_pose`
  "rows": [{"label": "attack_1", "glb": "a.glb", "clip": "attack_1",
            "marks": {"12": "load", "15": "strike"}}]
}

CAMERA (HLH's follow camera, `bevy/src/fx.rs` `camera_follow` in source
levels; the game's own record of it is `bevy/src/proportions.rs`):
focus 0.535 m above the body's centre, which is 1.0 m above his feet (the
2 m body capsule, `assets/data/actor_physics.json`), so 1.535 m up; the
eye 3.0 m from the focus (`Fence::distance` default) at pitch -0.32 rad
(-18.3 deg, `rig.pitch` at spawn), straight behind him (`rig.yaw` = his
facing at spawn); perspective, 75 deg vertical field of view, 16:9
(1280x720 shots). "lock" is the same camera eased over his right
shoulder as when locked on (0.85 m right, 0.25 m up, looking 0.6 of the
way right). "close" is a 3/4 front view (30 deg toward his right hand,
10 deg down, 34 deg lens: `run_clip.rs`'s lens for moving clips) framing
the whole clip. The camera does not move during a clip: the game's
follows his root, and these clips keep the root still.

Look: the character's own painted materials, a plain floor, one soft sun
from the camera's side (55 deg up, 45 deg left of the view, 3 deg wide)
and sky light, so the shadow falls short and to his right in view.
"""

import json
import math
import sys

import bpy
from mathutils import Matrix, Quaternion, Vector

SPEC = json.load(open(sys.argv[sys.argv.index("--") + 1:][0]))
W, H = SPEC.get("size", [960, 540])

# The game's camera, metres (see the module doc).
GAME_FOCUS = 1.0 + 0.535
GAME_DIST = 3.0
GAME_PITCH = -0.32
GAME_FOV_DEG = 75.0
CLOSE_FOV_DEG = 34.0
# `tracks::held_pose` (Bevy, in the socket's space): the blade cocked off
# the forearm, the grip's middle at this point.
HELD_ROT = (0.30965, -0.67445, 0.00552, 0.67023)  # x, y, z, w
HELD_GRIP = (0.0, 0.0428, -0.0850)
# glTF (Y up) to Blender (Z up).
CONV = Matrix(((1, 0, 0, 0), (0, 0, -1, 0), (0, 1, 0, 0), (0, 0, 0, 1)))


def clear():
    for coll in (bpy.data.objects, bpy.data.actions, bpy.data.curves, bpy.data.meshes,
                 bpy.data.armatures, bpy.data.lights, bpy.data.cameras):
        for x in list(coll):
            coll.remove(x)


def plain_mat(name, rgb, rough=0.9):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True
    m.diffuse_color = (*rgb, 1)
    b = next(n for n in m.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')
    b.inputs["Base Color"].default_value = (*rgb, 1)
    b.inputs["Roughness"].default_value = rough
    return m


def shown_meshes(objs):
    """Meshes that render: not the importer's bone-shape icosphere."""
    return [o for o in objs if o.type == 'MESH'
            and not any(c.name.startswith("glTF_not_exported") for c in o.users_collection)]


def stature_unit(arm):
    """Metres of the game per metre of this GLB: Andras is 2 m with his
    head joint at 1.6836 m (`character.rs` `RigRest::stature`), and the
    game scales a rig to that. The camera is moved instead of the
    character (same picture; scaling the imported armature broke its
    skinning). `stature` 0 keeps the GLB's own size."""
    st = SPEC.get("stature", 2.0)
    head = arm.data.bones.get("DEF-spine.006")
    if not st or head is None:
        return 1.0
    bpy.context.view_layer.update()
    y = (arm.matrix_world @ head.head_local).z
    return st * (1.6836 / 2.0) / y if y > 1e-3 else 1.0


def find_action(clip):
    act = bpy.data.actions.get(clip)
    if act is None:
        cands = [a for a in bpy.data.actions if a.name.split("_Armature")[0] == clip
                 or a.name.startswith(clip + "_")]
        cands.sort(key=lambda a: len(a.name))
        act = cands[0] if cands else None
    return act


def load(row):
    clear()
    scn = bpy.context.scene
    scn.render.fps = 30
    scn.render.fps_base = 1.0
    bpy.ops.import_scene.gltf(filepath=row["glb"])
    arm = next(o for o in bpy.data.objects if o.type == 'ARMATURE')
    meshes = shown_meshes(bpy.data.objects)
    act = find_action(row["clip"])
    if act is None:
        raise SystemExit(f"no clip {row['clip']} in {row['glb']}: {[a.name for a in bpy.data.actions]}")
    arm.animation_data_create()
    arm.animation_data.action = act
    if hasattr(arm.animation_data, "action_slot") and act.slots:
        arm.animation_data.action_slot = act.slots[0]
    if SPEC.get("prop"):
        meshes += attach_prop(arm, row["glb"], SPEC["prop"], SPEC.get("prop_pose", "socket"))
    f0, f1 = (int(round(x)) for x in act.frame_range)
    return arm, meshes, f0, f1


def gltf_node_world(path, name):
    """A node's rest world matrix straight from the GLB (glTF axes)."""
    import struct
    b = open(path, "rb").read()
    n = struct.unpack("<I", b[12:16])[0]
    j = json.loads(b[20:20 + n])
    nodes = j["nodes"]
    parent = {}
    for i, nd in enumerate(nodes):
        for c in nd.get("children", []):
            parent[c] = i

    def local(nd):
        if "matrix" in nd:
            m = nd["matrix"]
            return Matrix([[m[c * 4 + r] for c in range(4)] for r in range(4)])
        t = nd.get("translation", [0, 0, 0])
        q = nd.get("rotation", [0, 0, 0, 1])
        s = nd.get("scale", [1, 1, 1])
        return (Matrix.Translation(t) @ Quaternion((q[3], q[0], q[1], q[2])).to_matrix().to_4x4()
                @ Matrix.Diagonal((s[0], s[1], s[2], 1)))
    idx = next(i for i, nd in enumerate(nodes) if nd.get("name") == name)
    m = local(nodes[idx])
    while idx in parent:
        idx = parent[idx]
        m = local(nodes[idx]) @ m
    return m


def gltf_parent_name(path, name):
    """The name of a node's parent in the GLB, or None."""
    import struct
    b = open(path, "rb").read()
    n = struct.unpack("<I", b[12:16])[0]
    nodes = json.loads(b[20:20 + n])["nodes"]
    idx = next(i for i, nd in enumerate(nodes) if nd.get("name") == name)
    for nd in nodes:
        if idx in nd.get("children", []):
            return nd.get("name")
    return None


def holder_world(arm, socket):
    if socket in arm.pose.bones:
        return arm.matrix_world @ arm.pose.bones[socket].matrix
    return bpy.data.objects[socket].matrix_world.copy()


def attach_prop(arm, glb, spec, pose):
    """Import `path.glb@Socket` and hold it at that socket, as the game
    does: the prop's geometry relative to its root's origin (its grip:
    the props kit's swords) with the blade turned to run +Y, placed in the
    socket node's own glTF frame, either straight out of the fist along
    its +Y ("socket") or as `tracks::held_pose` cocks it ("game")."""
    path, socket = spec.rsplit("@", 1)
    bpy.context.view_layer.update()
    if socket not in arm.pose.bones and socket not in bpy.data.objects:
        raise SystemExit(f"no socket {socket} in the rig")
    # A socket that came in as an empty under a bone is held through that
    # bone (Blender's importer misplaces an empty under a scaled armature:
    # a Mixamo rig's 0.01 put the sword 4 m away).
    hand = gltf_parent_name(glb, socket)
    via_bone = socket not in arm.pose.bones and hand in arm.pose.bones
    # The socket's glTF frame relative to whatever Blender made of it.
    arm.data.pose_position = 'REST'
    bpy.context.view_layer.update()
    if via_bone:
        # The bone's rest frame -> the socket's glTF frame, full matrix.
        rest = arm.matrix_world @ arm.pose.bones[hand].matrix
    else:
        rest = holder_world(arm, socket)
    arm.data.pose_position = 'POSE'
    bpy.context.view_layer.update()
    corr = rest.inverted() @ (CONV @ gltf_node_world(glb, socket))
    keep_offset = via_bone
    # The bone's head is the node's origin: keep only the axes (a root
    # node's offset, such as the game rig's 1 m lift, is not always where
    # Blender puts it). Held through the parent bone, the socket's offset
    # from it is real and kept.
    if not keep_offset:
        corr.translation = (0, 0, 0)
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=path)
    new = [o for o in bpy.data.objects if o not in before]
    bpy.context.view_layer.update()
    roots = [o for o in new if o.parent is None]
    # One root: its origin is the grip. Several: the scene's origin.
    root_inv = roots[0].matrix_world.inverted() if len(roots) == 1 else Matrix.Identity(4)
    meshes = shown_meshes(new)
    for o in meshes:
        o.data = o.data.copy()
        o.data.transform(CONV.inverted() @ root_inv @ o.matrix_world)
    ys = [v.co.y for o in meshes for v in o.data.vertices]
    flip = Matrix.Rotation(math.pi, 4, 'Z') if -min(ys) > max(ys) else Matrix.Identity(4)
    if pose == "game":
        q = Quaternion((HELD_ROT[3], HELD_ROT[0], HELD_ROT[1], HELD_ROT[2])).normalized()
        local = Matrix.Translation(HELD_GRIP) @ q.to_matrix().to_4x4()
    else:
        local = Matrix.Identity(4)
    # The prop is in metres; the socket is in the GLB's units.
    u = 1.0 / stature_unit(arm)
    for o in meshes:
        o.data.transform(corr @ Matrix.Scale(u, 4) @ local @ flip)
        o.parent = None
        o.matrix_parent_inverse = Matrix.Identity(4)
        o.matrix_basis = Matrix.Identity(4)
        bone = hand if via_bone else (socket if socket in arm.data.bones else None)
        if bone is not None:
            o.parent = arm
            o.parent_type = 'BONE'
            o.parent_bone = bone
            # Bone parenting puts the child at the bone's tail.
            o.location = (0, -arm.data.bones[bone].length, 0)
        else:
            o.parent = bpy.data.objects[socket]
    for o in meshes:
        o["mf_prop"] = True
    if SPEC.get("debug"):
        bpy.context.view_layer.update()
        for o in meshes:
            ps = [o.matrix_world @ v.co for v in o.data.vertices]
            print("PROPDBG", o.name, "parent", o.parent and o.parent.name, o.parent_type,
                  [round(min(q[i] for q in ps), 3) for i in range(3)], [round(max(q[i] for q in ps), 3) for i in range(3)])
    for o in new:
        if o.type != 'MESH':
            bpy.data.objects.remove(o)
    return meshes


def facing(arm):
    """His forward, horizontal, at rest: the sum of both feet's heel-to-toe
    (as `motion check` and `animate::forward` take it; one foot alone turns
    out 10-20 deg)."""
    b = arm.data.bones
    d = Vector((0, 0, 0))
    for side in "LR":
        if f"DEF-foot.{side}" in b and f"DEF-toe.{side}" in b:
            h = arm.matrix_world @ b[f"DEF-foot.{side}"].head_local
            t = arm.matrix_world @ b[f"DEF-toe.{side}"].head_local
            d += Vector((t.x - h.x, t.y - h.y, 0))
    return d.normalized() if d.length > 1e-6 else Vector((0, -1, 0))


def clip_points(arm, meshes, f0, f1):
    """Corners of every mesh's posed bounding box, over the clip (every
    other frame). A held prop does not count when the body is there: the
    frame is fitted to the figure, and the blade may leave it at full
    stretch."""
    scn = bpy.context.scene
    body = [o for o in meshes if not o.get("mf_prop")]
    meshes = body or meshes
    pts = []
    for f in range(f0, f1 + 1, 2):
        scn.frame_set(f)
        dg = bpy.context.evaluated_depsgraph_get()
        for o in meshes:
            ev = o.evaluated_get(dg)
            vs = ev.data.vertices
            step = max(1, len(vs) // 400)
            pts += [ev.matrix_world @ vs[i].co for i in range(0, len(vs), step)]
    scn.frame_set(f0)
    if SPEC.get("debug"):
        dg = bpy.context.evaluated_depsgraph_get()
        for o in meshes:
            ev = o.evaluated_get(dg)
            ps = [ev.matrix_world @ v.co for v in ev.data.vertices]
            print("BOUNDS", o.name, [round(min(p[i] for p in ps), 3) for i in range(3)],
                  [round(max(p[i] for p in ps), 3) for i in range(3)])
    return pts


def fit_view(arm, meshes, f0, f1, view, up, fov):
    """Eye and look point along `view` that fit the figure over the clip
    (the move's projected extents, not a sphere round it) to about 80%
    of the frame's height; the blade may reach past the sides at full
    stretch."""
    pts = clip_points(arm, meshes, f0, f1) + [Vector((q.x, q.y, 0.0)) for q in clip_points(arm, meshes, f0, f0)]
    cr = view.cross(up).normalized()
    cu = cr.cross(view).normalized()
    c0 = sum(pts, Vector()) / len(pts)
    xs = [(q - c0).dot(cr) for q in pts]
    ys = [(q - c0).dot(cu) for q in pts]
    zs = [(q - c0).dot(view) for q in pts]
    tan_v = math.tan(fov / 2)
    tan_h = tan_v * W / H
    mid = c0 + cr * ((min(xs) + max(xs)) / 2) + cu * ((min(ys) + max(ys)) / 2)
    half_x = (max(xs) - min(xs)) / 2 * 1.15
    half_y = (max(ys) - min(ys)) / 2 * 1.25
    dist = max(half_x / tan_h, half_y / tan_v) + max(zs)
    return mid, mid + view * dist


def place_camera(arm, meshes, f0, f1):
    scn = bpy.context.scene
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    scn.collection.objects.link(cam)
    cam.data.sensor_fit = 'VERTICAL'
    cam.data.clip_start = 0.02
    cam.data.clip_end = 500
    fwd = facing(arm)
    up = Vector((0, 0, 1))
    right = fwd.cross(up).normalized()
    origin = arm.matrix_world.translation.copy()
    origin.z = 0.0
    kind = SPEC.get("cam", "game")
    if kind in ("game", "lock"):
        yaw = math.radians(SPEC.get("yaw_deg", 0.0))
        look_dir = fwd * math.cos(yaw) + right * math.sin(yaw)
        back = -look_dir
        side_dir = look_dir.cross(up).normalized()  # the camera's right
        u = 1.0 / stature_unit(arm)
        focus = origin + up * (GAME_FOCUS * u)
        shoulder = 1.0 if kind == "lock" else 0.0
        side = side_dir * (0.85 * shoulder * u)
        lift = up * (0.25 * shoulder * u)
        p = -GAME_PITCH
        d = GAME_DIST * u
        eye = focus + side + lift + back * (d * math.cos(p)) + up * (d * math.sin(p))
        look = focus + side * 0.6
        cam.data.angle_y = math.radians(GAME_FOV_DEG)
    elif kind == "gamefit":
        # The game camera's direction and pitch, moved in until the move
        # fills the frame (tall frames for a phone).
        yaw = math.radians(SPEC.get("yaw_deg", 0.0))
        look_dir = fwd * math.cos(yaw) + right * math.sin(yaw)
        p = -GAME_PITCH
        view = (-look_dir * math.cos(p) + up * math.sin(p)).normalized()
        fov = math.radians(30)
        look, eye = fit_view(arm, meshes, f0, f1, view, up, fov)
        cam.data.angle_y = fov
    elif kind == "hand":
        # Close on the sword hand, from his front right and a little above,
        # riding with the hand (not turning with it): for judging the grip.
        bone = SPEC.get("hand_bone", "DEF-hand.R")
        u = 1.0 / stature_unit(arm)
        rider = bpy.data.objects.new("rider", None)
        scn.collection.objects.link(rider)
        c = rider.constraints.new('COPY_LOCATION')
        c.target = arm
        c.subtarget = bone
        cam.parent = rider
        view = (fwd * 0.75 + right * 0.45 + up * 0.5).normalized()
        cam.location = view * (0.62 * u)
        cam.rotation_euler = (-view).to_track_quat('-Z', 'Y').to_euler()
        cam.data.angle_y = math.radians(40)
        cam.data.clip_start = 0.005 * u
        scn.camera = cam
        return cam, fwd, -view
    elif SPEC.get("fit", "sphere") == "tight":
        a = math.radians(30)
        d = (fwd * math.cos(a) + right * math.sin(a)).normalized()
        el = math.radians(10)
        view = (d * math.cos(el) + up * math.sin(el)).normalized()
        fov = math.radians(CLOSE_FOV_DEG)
        look, eye = fit_view(arm, meshes, f0, f1, view, up, fov)
        cam.data.angle_y = fov
    else:
        a = math.radians(30)
        d = (fwd * math.cos(a) + right * math.sin(a)).normalized()
        el = math.radians(10)
        view = (d * math.cos(el) + up * math.sin(el)).normalized()
        pts = clip_points(arm, meshes, f0, f1)
        lo = Vector((min(p.x for p in pts), min(p.y for p in pts), 0.0))
        hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
        c = (lo + hi) / 2
        rad = max((p - c).length for p in pts) * 1.02
        half = math.radians(CLOSE_FOV_DEG) / 2
        # Fit the clip's bounding sphere in the narrower of the two angles.
        half_w = math.atan(math.tan(half) * W / H)
        dist = rad / math.sin(min(half, half_w))
        look = c
        eye = c + view * dist
        cam.data.angle_y = math.radians(CLOSE_FOV_DEG)
    if SPEC.get("debug"):
        scn.frame_set(f0)
        hips = arm.matrix_world @ arm.pose.bones["DEF-spine"].head if "DEF-spine" in arm.pose.bones else None
        print("CAMDBG origin", tuple(round(x, 3) for x in origin), "fwd", tuple(round(x, 3) for x in fwd),
              "hips", hips and tuple(round(x, 3) for x in hips), "eye", tuple(round(x, 3) for x in eye))
    cam.location = eye
    cam.rotation_euler = (look - eye).to_track_quat('-Z', 'Y').to_euler()
    scn.camera = cam
    return cam, fwd, (look - eye).normalized()


def light_and_floor(view_dir):
    scn = bpy.context.scene
    # Sun from the camera's side, 45 deg left of the view, 55 deg up.
    vd = Vector((view_dir.x, view_dir.y, 0)).normalized()
    left = Vector((0, 0, 1)).cross(vd).normalized()
    toward_sun = (-vd * math.cos(math.radians(45)) + left * math.sin(math.radians(45))).normalized()
    el = math.radians(55)
    to_sun = (toward_sun * math.cos(el) + Vector((0, 0, math.sin(el)))).normalized()
    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", 'SUN'))
    scn.collection.objects.link(sun)
    sun.rotation_euler = (-to_sun).to_track_quat('-Z', 'Y').to_euler()
    sun.data.energy = 3.2
    sun.data.angle = math.radians(3.0)
    sun.data.color = (1.0, 0.97, 0.92)
    bpy.ops.mesh.primitive_plane_add(size=80, location=(0, 0, 0))
    floor = bpy.context.active_object
    floor.name = "floor"
    floor.data.materials.append(plain_mat("floor", (0.36, 0.36, 0.34)))
    scn.world = scn.world or bpy.data.worlds.new("w")
    scn.world.use_nodes = True
    bg = next(n for n in scn.world.node_tree.nodes if n.type == 'BACKGROUND')
    bg.inputs["Color"].default_value = (0.55, 0.62, 0.72, 1)
    bg.inputs["Strength"].default_value = 0.9
    scn.world.color = (0.55, 0.62, 0.72)
    return sun


def setup_render():
    scn = bpy.context.scene
    eng = SPEC.get("engine", "eevee")
    if eng == "eevee":
        names = [e.identifier for e in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items]
        scn.render.engine = 'BLENDER_EEVEE_NEXT' if 'BLENDER_EEVEE_NEXT' in names else 'BLENDER_EEVEE'
        ee = scn.eevee
        for attr, val in (("taa_render_samples", 16), ("use_shadows", True),
                          ("shadow_ray_count", 2), ("shadow_step_count", 6),
                          ("use_raytracing", False)):
            if hasattr(ee, attr):
                setattr(ee, attr, val)
    else:
        scn.render.engine = 'BLENDER_WORKBENCH'
        sh = scn.display.shading
        sh.light = 'STUDIO'
        sh.color_type = 'TEXTURE'
        sh.show_shadows = True
        sh.shadow_intensity = 0.45
    scn.view_settings.view_transform = 'AgX' if eng == "eevee" else 'Standard'
    scn.render.resolution_x, scn.render.resolution_y = W, H
    scn.render.resolution_percentage = 100
    scn.render.film_transparent = False
    scn.render.image_settings.file_format = 'PNG'
    # Label and frame number in the corner (the render's stamp).
    r = scn.render
    r.use_stamp = True
    for a in dir(r):
        if a.startswith("use_stamp_") and a not in ("use_stamp_note",):
            try:
                setattr(r, a, False)
            except (AttributeError, TypeError):
                pass
    r.use_stamp = True
    r.use_stamp_note = True
    r.stamp_font_size = max(12, H // 26)
    r.stamp_foreground = (1, 1, 1, 1)
    r.stamp_background = (0, 0, 0, 0.55)


def sun_for_workbench(to_sun):
    # Workbench's studio shadows come from `light_direction` (view space
    # is not involved: it is in world space, toward the light).
    bpy.context.scene.display.light_direction = to_sun


def main():
    setup_render()
    rows = SPEC["rows"]
    done = []
    for r, row in enumerate(rows):
        arm, meshes, f0, f1 = load(row)
        cam, fwd, view_dir = place_camera(arm, meshes, f0, f1)
        sun = light_and_floor(view_dir)
        if SPEC.get("engine", "eevee") != "eevee":
            sun_for_workbench(-(sun.matrix_world.to_3x3() @ Vector((0, 0, -1))))
        marks = {int(k): v for k, v in row.get("marks", {}).items()}
        scn = bpy.context.scene
        n = f1 - f0 + 1
        for i in (SPEC.get("only") or range(n)):
            if i >= n:
                continue
            scn.frame_set(f0 + i)
            tag = f"  {marks[i]}" if i in marks else ""
            scn.render.stamp_note_text = f"{row['label']}   frame {i}{tag}"
            scn.render.filepath = f"{SPEC['out']}.r{r}.{i:04d}.png"
            bpy.ops.render.render(write_still=True)
        done.append((row["label"], n))
    print("MOTION_VIDEO", json.dumps(done))


main()
