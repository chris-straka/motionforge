#!/usr/bin/env python3
"""Rig a creature mesh with skintokens several times and keep the try the
checks like best (stdlib only).

    python3 tools/creature_rig.py MESH.glb OUT_DIR [--seeds 1,2,3] [--class quadruped|custom|humanoid]
        [--forward auto|+x|-x|+y|-y] [--name NAME] [--skintokens PATH] [--motionforge PATH]

For every seed: `skintokens rig` then `motionforge creature-profile`
(every leg on the ground has a bone, left and right match, fingers
folded into the hand, bones that stick out cut off). The try with the
highest score wins (ties: the lowest seed). OUT_DIR gets creature.json,
clean.glb, report.json and a tries.json with every try's checks.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys


def which(name, explicit, fallbacks):
    if explicit:
        return explicit
    for p in [shutil.which(name)] + fallbacks:
        if p and os.path.exists(p):
            return p
    raise SystemExit(f"{name} not found; pass --{name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh")
    ap.add_argument("out")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--class", dest="klass", default="custom")
    ap.add_argument("--forward", default="auto")
    ap.add_argument("--name", default="")
    ap.add_argument("--skintokens", default="")
    ap.add_argument("--motionforge", default="")
    args = ap.parse_args()
    home = os.path.expanduser("~")
    sk = which("skintokens", args.skintokens, [os.path.join(home, "Games/_blender/skintokens/bin/skintokens")])
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mf = which("motionforge", args.motionforge, [os.path.join(repo, "rust/target/release/motionforge"),
                                                 os.path.join(os.environ.get("CARGO_TARGET_DIR", ""), "release/motionforge")])
    name = args.name or os.path.splitext(os.path.basename(args.mesh))[0]
    os.makedirs(args.out, exist_ok=True)
    tries = []
    for seed in [int(s) for s in args.seeds.split(",") if s]:
        rig = os.path.join(args.out, f"try_s{seed}.glb")
        rep = os.path.join(args.out, f"try_s{seed}.skintokens.json")
        r = subprocess.run([sk, "rig", args.mesh, rig, "--report", rep, "--class", args.klass, "--seed", str(seed)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(f"seed {seed}: skintokens failed: {r.stderr.strip()[-300:]}")
            tries.append({"seed": seed, "ok": False, "error": r.stderr.strip()[-300:]})
            continue
        prof = os.path.join(args.out, f"try_s{seed}.creature.json")
        clean = os.path.join(args.out, f"try_s{seed}.clean.glb")
        chk = os.path.join(args.out, f"try_s{seed}.report.json")
        r = subprocess.run([mf, "creature-profile", "--rig", rig, "--output", prof, "--clean-glb", clean,
                            "--report", chk, "--forward", args.forward, "--name", name], capture_output=True, text=True)
        if r.returncode != 0:
            print(f"seed {seed}: creature-profile failed: {r.stderr.strip()[-300:]}")
            tries.append({"seed": seed, "ok": False, "error": r.stderr.strip()[-300:]})
            continue
        with open(chk) as f:
            checks = json.load(f)
        checks.update({"seed": seed, "ok": True})
        tries.append(checks)
        print(f"seed {seed}: score {checks['score']:.4f} ({checks['joints_out']} bones, legs {checks['legs_with_bone']}/{checks['legs_found']}, "
              f"mirror {checks['side_matched']}/{checks['side_joints']}, fingers folded {checks['fingers_folded']}, "
              f"cut {checks['out_of_body_cut']}, outside {checks['out_of_body_left']})")
    good = [t for t in tries if t.get("ok")]
    if not good:
        raise SystemExit("no rig try succeeded")
    best = sorted(good, key=lambda t: (-t["score"], t["seed"]))[0]
    for src, dst in (("creature.json", "creature.json"), ("clean.glb", "clean.glb"), ("report.json", "report.json")):
        shutil.copyfile(os.path.join(args.out, f"try_s{best['seed']}.{src}"), os.path.join(args.out, dst))
    with open(os.path.join(args.out, "tries.json"), "w") as f:
        json.dump({"mesh": os.path.abspath(args.mesh), "class": args.klass, "best_seed": best["seed"], "tries": tries}, f, indent=2)
    print(f"best: seed {best['seed']} (score {best['score']:.4f}) -> {args.out}/creature.json, clean.glb")


if __name__ == "__main__":
    main()
