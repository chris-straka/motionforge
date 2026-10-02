# SPDX-License-Identifier: GPL-3.0-or-later
"""Batch-retarget a BVH directory onto the hero rig (headless Blender).

Data-pipeline workhorse for P3: for each BVH (e.g. CMU mocap), import it
into a fresh copy of the target rig .blend, retarget through the
motionforge CLI, and save a clip JSON. Also writes the dataset manifest
rows (id + source + license + sha) for tools/manifest.py --check.

Inputs (all local-only, never committed): the rig .blend, the BVH
directory, and a bonemap written against the BVH joint names (import one
BVH, list its bones, write the map; the CLI's skip report guides you).

    /Applications/Blender.app/Contents/MacOS/Blender --background \\
        --factory-startup --python blender/tools/batch_retarget.py -- \\
        --target-blend /path/to/hero.blend --bvh-dir data/bvh \\
        --bonemap /path/to/cmu_hero.json --out-dir data/clips \\
        --manifest data/manifest.json --source cmu --license cmu-mocap \\
        --terms-snapshot "free for research+commercial (read DATE)" \\
        --fps 120

Every BVH becomes one row (split=train, or heldout every --heldout-every
clips). Failures are collected and reported at the end (exit 1) instead
of aborting the batch on the first bad file.
"""

import argparse
import hashlib
import json
import math
import os
import sys

import bpy

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "..", ".."))
sys.path.insert(0, os.path.join(REPO, "blender"))

from motionforge import cli, clip_io  # noqa: E402


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def score_armature(obj, names):
    bones = {b.name for b in obj.data.bones}
    return len(bones & set(names))


def find_source(armatures, bonemap_sources):
    best, hits = None, 0
    for obj in armatures:
        score = score_armature(obj, bonemap_sources)
        if score > hits:
            best, hits = obj, score
    return best if hits >= 2 else None


def find_target(armatures, pinned_name):
    if pinned_name:
        obj = bpy.data.objects.get(pinned_name)
        if obj is not None and obj.type == "ARMATURE":
            return obj
        return None
    best, count = None, 0
    for obj in armatures:
        n = sum(1 for b in obj.data.bones if b.name.startswith("DEF-"))
        if n > count:
            best, count = obj, n
    return best


def main(argv):
    parser = argparse.ArgumentParser(description="batch-retarget BVH to clips")
    parser.add_argument("--target-blend", required=True)
    parser.add_argument("--bvh-dir", required=True)
    parser.add_argument("--bonemap", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--license", required=True)
    parser.add_argument("--terms-snapshot", default="")
    parser.add_argument("--fps", type=int, default=120)
    parser.add_argument("--yaw", default="auto", choices=["auto", "flip", "none"])
    parser.add_argument("--no-pin", action="store_true")
    parser.add_argument("--heldout-every", type=int, default=0)
    parser.add_argument("--target-name", default="")
    parser.add_argument("--cli", default="")
    args = parser.parse_args(argv)

    binary = cli.find_motionforge_binary(args.cli)
    if not binary:
        print("ERROR: no motionforge CLI found")
        return 1
    with open(args.bonemap, encoding="utf-8") as f:
        bonemap = json.load(f)
    sources = [p["source"] for p in bonemap.get("pairs", [])]
    os.makedirs(args.out_dir, exist_ok=True)

    bvhs = sorted(f for f in os.listdir(args.bvh_dir) if f.lower().endswith(".bvh"))
    if not bvhs:
        print(f"ERROR: no BVH files in {args.bvh_dir}")
        return 1

    try:
        bpy.ops.preferences.addon_enable(module="io_anim_bvh")
    except Exception as exc:
        print(f"BVH importer note: {exc}")
    if not hasattr(bpy.ops.import_anim, "bvh"):
        print("ERROR: BVH importer unavailable")
        return 1

    rows, failures = [], []
    for n, filename in enumerate(bvhs, start=1):
        clip_id = os.path.splitext(filename)[0]
        try:
            bpy.ops.wm.open_mainfile(filepath=args.target_blend)
            bpy.ops.import_anim.bvh(filepath=os.path.join(args.bvh_dir, filename))
            armatures = [o for o in bpy.data.objects if o.type == "ARMATURE"]
            src = find_source(armatures, sources)
            if src is None:
                raise clip_io.MotionforgeError("no armature matches the bonemap sources")
            tgt = find_target([o for o in armatures if o is not src], args.target_name)
            if tgt is None:
                raise clip_io.MotionforgeError("no target rig found (need DEF- bones or --target-name)")
            action = (
                src.animation_data.action if src.animation_data else None
            )
            if action is None:
                raise clip_io.MotionforgeError("imported BVH carries no action")
            lo, hi = action.frame_range
            start, end = math.floor(lo), math.ceil(hi)
            bpy.context.scene.render.fps = args.fps
            source_doc = clip_io.export_action(src, action, start, end)
            target_doc = {"format": "motionforge-skeleton", "version": 1,
                          **clip_io.export_skeleton(tgt)}
            tmp_src = os.path.join(args.out_dir, f".{clip_id}.src.json")
            tmp_tgt = os.path.join(args.out_dir, f".{clip_id}.tgt.json")
            out_path = os.path.join(args.out_dir, clip_id + ".json")
            try:
                with open(tmp_src, "w", encoding="utf-8") as f:
                    json.dump(source_doc, f)
                with open(tmp_tgt, "w", encoding="utf-8") as f:
                    json.dump(target_doc, f)
                report = cli.run_cli(binary, cli.build_retarget_args(
                    tmp_src, tmp_tgt, args.bonemap, out_path,
                    yaw=args.yaw, pin=not args.no_pin))
            finally:
                for tmp in (tmp_src, tmp_tgt):
                    if os.path.exists(tmp):
                        os.unlink(tmp)
            slide = next((line for line in report.splitlines() if "mean" in line), "")
            print(f"[{n}/{len(bvhs)}] {clip_id}: frames {start}..{end} {slide.strip()}")
            split = "train"
            if args.heldout_every > 0 and n % args.heldout_every == 0:
                split = "heldout"
            row = {
                "id": clip_id,
                "path": os.path.relpath(out_path, os.path.dirname(os.path.abspath(args.manifest))),
                "source": args.source,
                "license": args.license,
                "split": split,
                "sha256": sha256_file(out_path),
            }
            if args.terms_snapshot:
                row["terms_snapshot"] = args.terms_snapshot
            rows.append(row)
        except (clip_io.MotionforgeError, cli.CliError, OSError) as exc:
            print(f"[{n}/{len(bvhs)}] {clip_id}: FAILED ({exc})")
            failures.append(clip_id)

    with open(args.manifest, "w", encoding="utf-8") as f:
        json.dump({"format": "motionforge-dataset", "version": 1, "clips": rows}, f, indent=1)
    print(f"wrote {len(rows)} clips + manifest ({len(failures)} failures)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []))
