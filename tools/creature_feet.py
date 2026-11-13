#!/usr/bin/env python3
"""Does a solved creature clip stand and step? (stdlib only)

    python3 tools/creature_feet.py --creature creature.json clip.json [clip.json ...]

Per foot of the creature (its profile's `feet`), from the clip's own FK:
the foot joint's height range, how many times it lifts more than 3 cm
above its lowest point (down again under 1.5 cm), its sideways and
fore-aft spread (sd, along the creature's right and forward), and the
height's frame-to-frame acceleration (rms; jitter shows here).
"""

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
from autopose.dataset import fk, load_clip  # noqa: E402


def sd(v):
    m = sum(v) / len(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / len(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--creature", required=True)
    ap.add_argument("clips", nargs="+")
    args = ap.parse_args()
    cre = json.load(open(args.creature))
    fwd = cre["forward"]
    right = (fwd[1], -fwd[0], 0.0)  # forward x up
    for path in args.clips:
        sk, frames = load_clip(path)
        heads = [fk(sk, f) for f in frames]
        print(f"{os.path.basename(path)} ({len(frames)} frames)")
        for foot in cre["feet"]:
            i = sk.index[foot]
            pts = [h[i] for h in heads]
            zs = [p[2] for p in pts]
            side = [p[0] * right[0] + p[1] * right[1] for p in pts]
            ahead = [p[0] * fwd[0] + p[1] * fwd[1] for p in pts]
            lo = min(zs)
            lifts, up = 0, False
            for z in zs:
                if not up and z > lo + 0.03:
                    lifts, up = lifts + 1, True
                elif up and z < lo + 0.015:
                    up = False
            acc = math.sqrt(sum((zs[k + 1] - 2 * zs[k] + zs[k - 1]) ** 2 for k in range(1, len(zs) - 1))
                            / max(1, len(zs) - 2))
            print(f"  {foot}: height {1000 * lo:.0f}..{1000 * max(zs):.0f} mm, lifts {lifts}, "
                  f"sideways sd {1000 * sd(side):.0f} mm, fore-aft sd {1000 * sd(ahead):.0f} mm, "
                  f"height accel rms {1000 * acc:.1f} mm/frame^2")


if __name__ == "__main__":
    main()
