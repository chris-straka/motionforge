"""Render a clip on a rigged creature through the profile's two cameras.

Runs inside Blender (headless):

    blender --background --factory-startup --python python/creature/render.py -- \
        --glb clean.glb --creature creature.json --clip poses.json --out DIR \
        [--mode poses|motion] [--limit N] [--seed 1] [--samples 16]

Mode `poses` takes a `creature-poses` clip (frames 3k, 3k+1, 3k+2 =
before, pose, after) and renders frame 3k+1 of each sample with a
one-frame shutter, so fast samples blur half way to their neighbours;
light, colour and exposure are randomized per sample. Mode `motion`
renders every frame of a clip as is (steady light, no blur): the exam's
"film", and the side-by-side proof.

Per rendered sample `i` and view `v` it writes `DIR/f{i:05d}_{v}.png`
and, once, `DIR/truth.json`: the 2D joint positions (pixels, in each
view's own image, Blender's own projection) and whether each joint is in
front of the camera. The Rust FK must agree with Blender's posed bones:
the script checks that on the first frame and aborts on a mismatch.

Why Blender: the start frame the video model continues, the training
pictures and the exam film must all come out of one renderer.
"""

import argparse
import json
import math
import os
import random
import sys
import time

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Quaternion, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # python/
from autopose.dataset import Skeleton, _mmul, _mvec, _qmat  # noqa: E402  (stdlib FK matching Rust)


def fk_full(skeleton, poses):
    """Armature-space (rotation rows, head) per bone, matching motion_core::clip::fk."""
    rots = [None] * len(skeleton.names)
    heads = [None] * len(skeleton.names)
    for i in range(len(skeleton.names)):
        brot = _qmat(poses[i][1])
        local_rot = _mmul(skeleton.rel_rot[i], brot)
        local_off = tuple(a + b for a, b in zip(_mvec(skeleton.rel_rot[i], poses[i][0]), skeleton.rel_off[i]))
        p = skeleton.parents[i]
        if p is None:
            rots[i], heads[i] = local_rot, local_off
        else:
            rots[i] = _mmul(rots[p], local_rot)
            moved = _mvec(rots[p], local_off)
            heads[i] = (moved[0] + heads[p][0], moved[1] + heads[p][1], moved[2] + heads[p][2])
    return list(zip(rots, heads))


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--glb", required=True)
    ap.add_argument("--creature", required=True)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["poses", "motion"], default="poses")
    ap.add_argument("--limit", type=int, default=0, help="render at most N samples/frames")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--samples", type=int, default=16, help="Eevee render samples")
    ap.add_argument("--start", type=int, default=0, help="first sample index (resume)")
    return ap.parse_args(argv)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def clean_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_glb(path):
    bpy.ops.import_scene.gltf(filepath=path)
    arms = [o for o in bpy.data.objects if o.type == "ARMATURE"]
    if len(arms) != 1:
        raise SystemExit(f"expected one armature in {path}, found {len(arms)}")
    arm = arms[0]
    if any(abs(arm.matrix_world[i][j] - (1.0 if i == j else 0.0)) > 1e-6 for i in range(4) for j in range(4)):
        raise SystemExit("armature object transform is not identity; the profile assumes it is")
    return arm


def set_camera(scene, view):
    cam = bpy.data.cameras.new(view["name"])
    cam.type = "PERSP"
    cam.sensor_fit = "VERTICAL"
    cam.angle_y = math.radians(view["fov_y_deg"])
    cam.clip_start = 0.05
    cam.clip_end = 100.0
    co = bpy.data.objects.new(view["name"], cam)
    scene.collection.objects.link(co)
    pos = Vector(view["position"])
    look = Vector(view["look_at"])
    fwd = (look - pos).normalized()
    # Camera looks down -Z with +Y up: build the rotation from the basis.
    right = fwd.cross(Vector((0, 0, 1)))
    if right.length < 1e-9:
        right = Vector((1, 0, 0))
    right.normalize()
    up = right.cross(fwd).normalized()
    m = Matrix((
        (right.x, up.x, -fwd.x, pos.x),
        (right.y, up.y, -fwd.y, pos.y),
        (right.z, up.z, -fwd.z, pos.z),
        (0, 0, 0, 1),
    ))
    co.matrix_world = m
    return co


def make_lights(scene, rng, randomize):
    for o in [o for o in bpy.data.objects if o.type == "LIGHT"]:
        bpy.data.objects.remove(o, do_unlink=True)
    def add(name, kind, loc, energy, color=(1, 1, 1), size=3.0):
        ld = bpy.data.lights.new(name, kind)
        ld.energy = energy
        ld.color = color
        if kind == "AREA":
            ld.size = size
        lo = bpy.data.objects.new(name, ld)
        scene.collection.objects.link(lo)
        lo.location = loc
        lo.rotation_euler = (Vector(loc) * -1).to_track_quat("-Z", "Y").to_euler()
        return lo
    if randomize:
        key_az = rng.uniform(-math.pi, math.pi)
        key_el = rng.uniform(0.3, 1.2)
        key_e = rng.uniform(400, 1400)
        warm = rng.uniform(0.85, 1.0)
        cool = rng.uniform(0.85, 1.0)
        key_col = (1.0, warm, warm * rng.uniform(0.85, 1.0))
        fill_col = (cool * rng.uniform(0.85, 1.0), cool, 1.0)
        fill_e = rng.uniform(100, 500)
        world_s = rng.uniform(0.4, 1.0)
    else:
        key_az, key_el, key_e, key_col, fill_col, fill_e, world_s = 0.8, 0.8, 900, (1, 0.97, 0.92), (0.92, 0.95, 1.0), 300, 0.7
    r = 6.0
    add("key", "AREA", (r * math.cos(key_az) * math.cos(key_el), r * math.sin(key_az) * math.cos(key_el), r * math.sin(key_el) + 1.0), key_e, key_col)
    add("fill", "AREA", (-r * math.cos(key_az), -r * math.sin(key_az), 2.5), fill_e, fill_col, size=5.0)
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes.get("Background")
    if bg is not None:
        bg.inputs[0].default_value = (0.5, 0.5, 0.5, 1.0)
        bg.inputs[1].default_value = world_s


def set_material(arm, rng, randomize):
    """One plain material on every mesh: a neutral body colour (varied a
    little per sample) so the detector does not latch on to a texture."""
    for o in bpy.data.objects:
        if o.type != "MESH":
            continue
        mat = bpy.data.materials.get("creature_body")
        if mat is None:
            mat = bpy.data.materials.new("creature_body")
            mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf is not None:
            if randomize:
                base = rng.uniform(0.35, 0.75)
                tint = [base * rng.uniform(0.85, 1.0) for _ in range(3)]
            else:
                tint = [0.62, 0.58, 0.52]
            bsdf.inputs["Base Color"].default_value = (*tint, 1.0)
            bsdf.inputs["Roughness"].default_value = rng.uniform(0.4, 0.9) if randomize else 0.6
        o.data.materials.clear()
        o.data.materials.append(mat)


def pose_from_clip(arm, skel, frame_poses, matrix_local, name_to_pbone):
    """Set every pose bone's matrix_basis so Blender's posed heads equal
    the Rust FK of the clip frame (the armature's bones carry the glTF
    importer's rolls, so the clip quats cannot be used as-is)."""
    posed = fk_full(skel, frame_poses)  # list of (rot 3x3 rows, head)
    world = {}
    for i, name in enumerate(skel.names):
        rot, head = posed[i]
        R = Matrix(rot)  # armature-space posed rotation of the zero-roll bone
        B = Matrix(skel.world[i])
        D = R @ B.transposed()  # world delta
        ml = matrix_local[name]
        pose_mat = Matrix.Translation(Vector(head)) @ (D @ ml.to_3x3()).to_4x4()
        world[name] = pose_mat
        pb = name_to_pbone[name]
        if pb.parent is None:
            basis = ml.inverted() @ pose_mat
        else:
            parent_pose = world[pb.parent.name]
            basis = (parent_pose @ matrix_local[pb.parent.name].inverted() @ ml).inverted() @ pose_mat
        pb.matrix_basis = basis


def key_pose(arm, frame):
    for pb in arm.pose.bones:
        pb.keyframe_insert("rotation_quaternion", frame=frame)
        pb.keyframe_insert("location", frame=frame)


def project(scene, cam, p):
    v = world_to_camera_view(scene, cam, Vector(p))
    return [v.x * scene.render.resolution_x, (1.0 - v.y) * scene.render.resolution_y, v.z]


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    creature = load_json(args.creature)
    clip = load_json(args.clip)
    skel = Skeleton([(b["name"], b["parent"], tuple(b["head"]), tuple(b["tail"])) for b in creature["bones"]])
    names = list(skel.names)
    clean_scene()
    arm = import_glb(args.glb)
    for pb in arm.pose.bones:
        pb.rotation_mode = "QUATERNION"
    missing = [n for n in names if n not in arm.pose.bones]
    if missing:
        raise SystemExit(f"bones missing from the armature: {missing[:5]}")
    name_to_pbone = {pb.name: pb for pb in arm.pose.bones}
    matrix_local = {b.name: b.matrix_local.copy() for b in arm.data.bones}

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.eevee.taa_render_samples = args.samples
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "Standard"
    size = creature["views"][0]["width"]
    scene.render.resolution_x = creature["views"][0]["width"]
    scene.render.resolution_y = creature["views"][0]["height"]
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    if scene.world is None:
        scene.world = bpy.data.worlds.new("world")
    cams = {v["name"]: set_camera(scene, v) for v in creature["views"]}
    rng = random.Random(args.seed)
    randomize = args.mode == "poses"
    set_material(arm, rng, randomize)
    make_lights(scene, rng, randomize)

    frames = clip["frames"]
    if args.mode == "poses":
        if len(frames) % 3 != 0:
            raise SystemExit("a poses clip has 3 frames per sample")
        samples = list(range(len(frames) // 3))
        scene.render.use_motion_blur = True
        scene.render.motion_blur_shutter = 1.0
        if hasattr(scene.render, "motion_blur_position"):
            scene.render.motion_blur_position = "CENTER"
    else:
        samples = list(range(len(frames)))
        scene.render.use_motion_blur = False
    if args.limit:
        samples = samples[: args.start + args.limit]
    samples = samples[args.start:]

    def frame_poses(k):
        return [(tuple(frames[k][n]["loc"]), tuple(frames[k][n]["quat"])) for n in names]

    truth = {"format": "motionforge-render-truth", "version": 1, "mode": args.mode,
             "joints": names, "views": [v["name"] for v in creature["views"]],
             "fps": clip.get("fps", 24), "samples": []}
    truth_path = os.path.join(args.out, "truth.json")
    if args.start and os.path.exists(truth_path):
        truth = load_json(truth_path)
        truth["samples"] = [s for s in truth["samples"] if s["index"] < args.start]

    bpy.context.preferences.edit.keyframe_new_interpolation_type = "LINEAR"

    def clear_action():
        ad = arm.animation_data
        if ad and ad.action:
            act = ad.action
            ad.action = None
            bpy.data.actions.remove(act)

    def key_frames(indices):
        """Key clip frames `indices` at scene frames 1..len (a fresh action:
        inserting keys into a long action gets slower with every key)."""
        clear_action()
        for n, k in enumerate(indices):
            pose_from_clip(arm, skel, frame_poses(k), matrix_local, name_to_pbone)
            key_pose(arm, n + 1)

    if args.mode == "motion":
        t0 = time.time()
        key_frames(range(len(frames)))
        print(f"keyed {len(frames)} frames in {time.time() - t0:.1f}s", flush=True)

    # Self-check: Blender's posed heads vs the Rust-matching FK on one frame.
    first = samples[0] * 3 + 1 if args.mode == "poses" else samples[0]
    if args.mode == "poses":
        key_frames([first - 1, first, first + 1])
        scene.frame_set(2)
    else:
        scene.frame_set(first + 1)
    posed = fk_full(skel, frame_poses(first))
    worst = 0.0
    for i, name in enumerate(names):
        h = arm.matrix_world @ name_to_pbone[name].head
        worst = max(worst, (h - Vector(posed[i][1])).length)
    if worst > 1e-3:
        raise SystemExit(f"FK mismatch between Blender and the clip FK: {worst:.4f} m")
    print(f"FK check ok (max head mismatch {worst:.2e} m)", flush=True)

    t0 = time.time()
    for n, s in enumerate(samples):
        if randomize:
            make_lights(scene, rng, True)
            set_material(arm, rng, True)
            scene.view_settings.exposure = rng.uniform(-0.35, 0.35)
        if args.mode == "poses":
            centre = s * 3 + 1
            key_frames([centre - 1, centre, centre + 1])
            scene.frame_set(2)
        else:
            centre = s
            scene.frame_set(centre + 1)
        posed = fk_full(skel, frame_poses(centre))
        entry = {"index": s, "clip_frame": centre, "views": {}}
        for vname, cam in cams.items():
            scene.camera = cam
            scene.render.filepath = os.path.join(args.out, f"f{s:05d}_{vname}.png")
            bpy.ops.render.render(write_still=True)
            entry["views"][vname] = [project(scene, cam, posed[i][1]) for i in range(len(names))]
        truth["samples"].append(entry)
        if (n + 1) % 25 == 0 or n + 1 == len(samples):
            with open(truth_path, "w") as f:
                json.dump(truth, f)
            el = time.time() - t0
            print(f"rendered {n + 1}/{len(samples)} ({el:.0f}s, {el / (n + 1):.2f}s per sample)", flush=True)
    with open(truth_path, "w") as f:
        json.dump(truth, f)
    print(f"done: {len(samples)} samples -> {args.out} (size {size})")


if __name__ == "__main__":
    main()
