#!/usr/bin/env python3
"""Generate committed test fixtures (procedural, synthetic — not mocap).

Writes tests/fixtures/*.json deterministically (no timestamps, fixed
seeds). Run from the repo root: `python3 tools/make_fixtures.py`.

Local-frame note: clip loc/quat channels live in each bone's
parent-relative rest frame (Blender matrix_basis). This generator works
in world space and converts through the zero-roll rest basis
(shortest-arc +Y -> bone direction, matching motion_core::math).
"""

import json
import math
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "tests", "fixtures")

# --- minimal vec/quat math -------------------------------------------------


def vsub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vadd(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def vscale(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def vdot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def vcross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def vnorm(a):
    l = math.sqrt(vdot(a, a))
    return vscale(a, 1.0 / l)


def qmul(p, q):
    pw, px, py, pz = p
    qw, qx, qy, qz = q
    return (
        pw * qw - px * qx - py * qy - pz * qz,
        pw * qx + px * qw + py * qz - pz * qy,
        pw * qy - px * qz + py * qw + pz * qx,
        pw * qz + px * qy - py * qx + pz * qw,
    )


def qconj(q):
    return (q[0], -q[1], -q[2], -q[3])


def qnorm(q):
    l = math.sqrt(sum(c * c for c in q))
    return tuple(c / l for c in q)


def qaxis(axis, angle):
    ax, ay, az = vnorm(axis)
    s, c = math.sin(angle / 2.0), math.cos(angle / 2.0)
    return (c, ax * s, ay * s, az * s)


def shortest_arc(fr, to):
    d = vdot(fr, to)
    if d >= 1.0 - 1e-12:
        return (1.0, 0.0, 0.0, 0.0)
    if d <= -1.0 + 1e-12:
        return qaxis((0.0, 0.0, 1.0), math.pi)
    return qnorm(qaxis(vnorm(vcross(fr, to)), math.acos(max(-1.0, min(1.0, d)))))


def qrotvec(q, v):
    w, x, y, z = q
    # q * (0,v) * q' expanded.
    uv = vcross((x, y, z), v)
    uuv = vcross((x, y, z), uv)
    return (
        v[0] + 2.0 * (w * uv[0] + uuv[0]),
        v[1] + 2.0 * (w * uv[1] + uuv[1]),
        v[2] + 2.0 * (w * uv[2] + uuv[2]),
    )


# --- skeleton helpers ------------------------------------------------------

Y = (0.0, 1.0, 0.0)


class Skel:
    def __init__(self, bones):
        """bones: list of (name, parent or None, head, tail)."""
        self.bones = bones
        self.index = {b[0]: i for i, b in enumerate(bones)}
        self.rest = {}
        for name, _parent, head, tail in bones:
            self.rest[name] = shortest_arc(Y, vnorm(vsub(tail, head)))

    def local_quat(self, name, world_quat):
        r = self.rest[name]
        return qnorm(qmul(qconj(r), qmul(world_quat, r)))

    def local_loc(self, name, world_offset):
        return qrotvec(qconj(self.rest[name]), world_offset)

    def to_json(self):
        return [
            {
                "name": n,
                "parent": p,
                "head": list(h),
                "tail": list(t),
            }
            for (n, p, h, t) in self.bones
        ]


def emit_clip(path, fps, skel, frames):
    """frames: list of dict bone -> (loc_xyz, quat_wxyz) in LOCAL frames."""
    doc = {
        "format": "motionforge-clip",
        "version": 1,
        "fps": fps,
        "skeleton": {"bones": skel.to_json()},
        "frames": [
            {b: {"loc": list(loc), "quat": list(q)} for b, (loc, q) in fr.items()}
            for fr in frames
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1, sort_keys=False)
        f.write("\n")


IDENT = (1.0, 0.0, 0.0, 0.0)
ZERO = (0.0, 0.0, 0.0)


def main():
    os.makedirs(OUT, exist_ok=True)

    # --- source walk clip (Mixamo-like, faces +Y) ---------------------------
    src = Skel(
        [
            ("Hips", None, (0, 0, 0.98), (0, 0, 1.10)),
            ("Spine", "Hips", (0, 0, 1.10), (0, 0, 1.50)),
            ("LeftUpLeg", "Hips", (0.11, 0, 0.98), (0.11, 0, 0.52)),
            ("LeftLeg", "LeftUpLeg", (0.11, 0, 0.52), (0.11, 0, 0.10)),
            ("LeftFoot", "LeftLeg", (0.11, 0, 0.10), (0.11, 0.16, 0.02)),
            ("RightUpLeg", "Hips", (-0.11, 0, 0.98), (-0.11, 0, 0.52)),
            ("RightLeg", "RightUpLeg", (-0.11, 0, 0.52), (-0.11, 0, 0.10)),
            ("RightFoot", "RightLeg", (-0.11, 0, 0.10), (-0.11, 0.16, 0.02)),
            ("LeftArm", "Spine", (0.20, 0, 1.45), (0.48, 0, 1.45)),
            ("LeftForeArm", "LeftArm", (0.48, 0, 1.45), (0.72, 0, 1.45)),
            ("RightArm", "Spine", (-0.20, 0, 1.45), (-0.48, 0, 1.45)),
            ("RightForeArm", "RightArm", (-0.48, 0, 1.45), (-0.72, 0, 1.45)),
        ]
    )
    # One full gait cycle in 48 frames: leg swing L*A*w ~= root speed so
    # stance feet genuinely plant (same match as rigforge's proof source).
    n = 48
    frames = []
    for i in range(n):
        ph = 2.0 * math.pi * i / n
        fr = {}
        # Root: travel + sway + bob.
        off = (0.02 * math.sin(ph), 1.2 * i / (n - 1), 0.025 * math.cos(2 * ph))
        fr["Hips"] = (
            src.local_loc("Hips", off),
            src.local_quat("Hips", qaxis((0, 0, 1), 0.06 * math.sin(ph))),
        )
        fr["Spine"] = (ZERO, src.local_quat("Spine", qaxis((0, 0, 1), 0.05 * math.sin(ph))))
        for side, s in (("Left", 0.0), ("Right", math.pi)):
            p = ph + s
            up = 0.22 * math.sin(p)
            knee = -0.14 * (1.0 + math.cos(p - 0.4))
            fr[f"{side}UpLeg"] = (ZERO, src.local_quat(f"{side}UpLeg", qaxis((1, 0, 0), up)))
            fr[f"{side}Leg"] = (ZERO, src.local_quat(f"{side}Leg", qaxis((1, 0, 0), knee)))
            fr[f"{side}Foot"] = (
                ZERO,
                src.local_quat(f"{side}Foot", qaxis((1, 0, 0), -(up + knee) * 0.75)),
            )
            sw = -0.15 * math.sin(p)
            bend = 0.20 if side == "Left" else -0.20
            fr[f"{side}Arm"] = (ZERO, src.local_quat(f"{side}Arm", qaxis((0, 0, 1), sw)))
            fr[f"{side}ForeArm"] = (
                ZERO,
                src.local_quat(f"{side}ForeArm", qaxis((0, 0, 1), sw + bend)),
            )
        frames.append(fr)
    emit_clip(os.path.join(OUT, "walk_src.json"), 30, src, frames)

    # --- hero target skeleton (HLL-like, faces -Y, shorter legs) ------------
    hero_bones = [
        ("DEF-spine", None, (0, 0, 0.70), (0, 0, 0.80)),
        ("DEF-spine.001", "DEF-spine", (0, 0, 0.80), (0, 0, 1.05)),
        ("DEF-thigh.L", "DEF-spine", (0.09, 0, 0.70), (0.09, 0, 0.38)),
        ("DEF-shin.L", "DEF-thigh.L", (0.09, 0, 0.38), (0.09, 0, 0.08)),
        ("DEF-foot.L", "DEF-shin.L", (0.09, 0, 0.08), (0.09, -0.12, 0.02)),
        ("DEF-toe.L", "DEF-foot.L", (0.09, -0.12, 0.02), (0.09, -0.20, 0.01)),
        ("DEF-thigh.R", "DEF-spine", (-0.09, 0, 0.70), (-0.09, 0, 0.38)),
        ("DEF-shin.R", "DEF-thigh.R", (-0.09, 0, 0.38), (-0.09, 0, 0.08)),
        ("DEF-foot.R", "DEF-shin.R", (-0.09, 0, 0.08), (-0.09, -0.12, 0.02)),
        ("DEF-toe.R", "DEF-foot.R", (-0.09, -0.12, 0.02), (-0.09, -0.20, 0.01)),
        ("DEF-upper_arm.L", "DEF-spine.001", (0.16, 0, 1.02), (0.38, 0, 1.02)),
        ("DEF-forearm.L", "DEF-upper_arm.L", (0.38, 0, 1.02), (0.56, 0, 1.02)),
        ("DEF-upper_arm.R", "DEF-spine.001", (-0.16, 0, 1.02), (-0.38, 0, 1.02)),
        ("DEF-forearm.R", "DEF-upper_arm.R", (-0.38, 0, 1.02), (-0.56, 0, 1.02)),
    ]
    hero = Skel(hero_bones)
    with open(os.path.join(OUT, "hero_skel.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"format": "motionforge-skeleton", "version": 1, "bones": hero.to_json()},
            f,
            indent=1,
        )
        f.write("\n")

    # --- bonemap -------------------------------------------------------------
    pairs = [
        ("Hips", "DEF-spine"),
        ("Spine", "DEF-spine.001"),
        ("LeftUpLeg", "DEF-thigh.L"),
        ("LeftLeg", "DEF-shin.L"),
        ("LeftFoot", "DEF-foot.L"),
        ("RightUpLeg", "DEF-thigh.R"),
        ("RightLeg", "DEF-shin.R"),
        ("RightFoot", "DEF-foot.R"),
        ("LeftArm", "DEF-upper_arm.L"),
        ("LeftForeArm", "DEF-forearm.L"),
        ("RightArm", "DEF-upper_arm.R"),
        ("RightForeArm", "DEF-forearm.R"),
        # Toe rows have no source bones: they pin the skip path in goldens.
        ("LeftToeBase", "DEF-toe.L"),
        ("RightToeBase", "DEF-toe.R"),
    ]
    with open(os.path.join(OUT, "hero_map.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "format": "motionforge-bonemap",
                "version": 1,
                "pairs": [{"source": s, "target": t} for s, t in pairs],
                "root_source": "Hips",
                "root_target": "DEF-spine",
                "stride_source": ["LeftUpLeg", "LeftFoot"],
                "stride_target": ["DEF-thigh.L", "DEF-foot.L"],
                "feet_source": ["LeftFoot", "RightFoot"],
                "feet_target": ["DEF-foot.L", "DEF-foot.R"],
            },
            f,
            indent=1,
        )
        f.write("\n")

    # --- autopose weights (seeded random MLP for the hero bones) ------------
    rng = random.Random(7)
    names = [b[0] for b in hero_bones]
    nb = len(names)
    layers = []
    dims = [(32, 5 * nb), (4 * nb, 32)]
    for rows, cols in dims:
        layers.append(
            {
                "weights": [[rng.uniform(-0.1, 0.1) for _ in range(cols)] for _ in range(rows)],
                "bias": [rng.uniform(-0.05, 0.05) for _ in range(rows)],
            }
        )
    with open(os.path.join(OUT, "autopose_weights.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "format": "motionforge-weights",
                "version": 1,
                "bones": names,
                "input": "effector-pos-mask+lengths",
                "layers": layers,
            },
            f,
        )
        f.write("\n")

    # --- autopose effectors --------------------------------------------------
    with open(os.path.join(OUT, "autopose_effectors.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "format": "motionforge-effectors",
                "version": 1,
                "skeleton": {"bones": hero.to_json()},
                "effectors": [
                    {"bone": "DEF-forearm.L", "position": [0.56, 0.1, 1.3]},
                    {"bone": "DEF-forearm.R", "position": [-0.56, 0.1, 1.3]},
                    {"bone": "DEF-spine.001", "position": [0.0, 0.0, 1.1]},
                ],
            },
            f,
            indent=1,
        )
        f.write("\n")

    # --- jump clip (physics golden: airborne phase + momentum spike) ---------
    jframes = []
    for i in range(12):
        air = 2 <= i <= 9
        t = (i - 2) / 30.0 if air else 0.0
        root_off = [0.0, 0.0, 0.0]
        if air:
            root_off[2] = 1.6 * t - 4.9 * t * t
        if i == 5:
            root_off[0] += 0.5  # teleport spike
        fr = {}
        for n_, _p, _h, _t in hero_bones:
            fr[n_] = (ZERO, IDENT)
        fr["DEF-spine"] = (hero.local_loc("DEF-spine", root_off), IDENT)
        tuck = hero.local_loc("DEF-foot.L", (0, 0, 0.25 if air else 0.0))
        fr["DEF-foot.L"] = (tuck, IDENT)
        fr["DEF-foot.R"] = (tuck, IDENT)
        jframes.append(fr)
    emit_clip(os.path.join(OUT, "jump.json"), 30, hero, jframes)

    print(f"fixtures written to {OUT}")


if __name__ == "__main__":
    sys.exit(main())
