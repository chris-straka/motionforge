"""Which way a model faces, measured from its points (stdlib only).

A generated model arrives in whatever frame its maker exports: Tripo's FBX
files face +X (Blender, Z up) where everything downstream wants a
character facing Blender -Y (glTF +Z): skintokens names left and right by
the sign of X, weightforge's poses swing arms "forward" along +Z, and the
game stands a body at the origin facing +Z. Rigged sideways, a humanoid's
arms split front and back and the rig comes out missing its core bones.

`measure(points)` works in a Z-up frame:

1. Side to side: the upright mirror plane the points fit best. A body is
   close to symmetric about one plane only; `mirror_contrast` says how
   much better the best plane fits than the worst (1: no better, a
   cloaked or round thing).
2. Kind: a standing figure is a few tenths as deep as it is tall; a beast
   on four legs is about as long as it is tall, or longer.
3. Front from back. A biped's toes lead its ankles. A quadruped carries
   its head end higher and heavier than its tail end.

`decide()` then weighs the measurement against the source's convention:
a weak measurement (a cloak to the ground hides the feet; a scythe breaks
the symmetry) gives way to it.
"""

import math
import random

CONVENTIONS = {"+x": (1.0, 0.0), "-x": (-1.0, 0.0), "+y": (0.0, 1.0), "-y": (0.0, -1.0)}

# Below these the measurement is called weak and the convention decides.
MIN_MIRROR_CONTRAST = 2.5
MIN_CONFIDENCE = 0.02


def _mirror_error(pts, cells, cell, centre, angle):
    """Mean distance, in cells (0, 1 or 2+), from each point's mirror image
    across the upright plane through `centre` with its normal at `angle`
    from +X to the nearest occupied cell."""
    nx, ny = math.cos(angle), math.sin(angle)
    total = 0
    for x, y, z in pts:
        d = (x - centre[0]) * nx + (y - centre[1]) * ny
        qx, qy = x - 2.0 * d * nx, y - 2.0 * d * ny
        i, j, k = int(math.floor(qx / cell)), int(math.floor(qy / cell)), int(math.floor(z / cell))
        if (i, j, k) in cells:
            continue
        near = False
        for a in (-1, 0, 1):
            for b in (-1, 0, 1):
                for c in (-1, 0, 1):
                    if (i + a, j + b, k + c) in cells:
                        near = True
                        break
                if near:
                    break
            if near:
                break
        total += 1 if near else 2
    return total / len(pts)


def measure(points, kind="auto", samples=2500, seed=7):
    """Measure a Z-up point cloud. Returns a dict: `kind` (biped or
    quadruped), `forward` (unit x, y), `confidence` (how far the
    front-from-back cue leans, as a share of the height), `how`,
    `mirror_contrast`, `height`, `length` (front to back), `width`."""
    pts = list(points)
    if len(pts) < 16:
        raise ValueError("too few points to measure")
    if len(pts) > samples:
        pts = random.Random(seed).sample(pts, samples)
    lo = [min(p[i] for p in pts) for i in range(3)]
    hi = [max(p[i] for p in pts) for i in range(3)]
    height = hi[2] - lo[2]
    if height <= 0:
        raise ValueError("flat model")
    centre = [(lo[i] + hi[i]) / 2 for i in range(3)]
    cell = max(hi[0] - lo[0], hi[1] - lo[1], height) / 40.0
    cells = {(int(math.floor(x / cell)), int(math.floor(y / cell)), int(math.floor(z / cell))) for x, y, z in pts}
    errs = [(_mirror_error(pts, cells, cell, centre, math.radians(a)), a) for a in range(0, 180, 5)]
    best = min(errs)[1]
    err, deg = min((_mirror_error(pts, cells, cell, centre, math.radians(a)), a) for a in range(best - 4, best + 5))
    worst = max(errs)[0]
    side = (math.cos(math.radians(deg)), math.sin(math.radians(deg)))
    fwd = (-side[1], side[0])
    along = [(p[0] - centre[0]) * fwd[0] + (p[1] - centre[1]) * fwd[1] for p in pts]
    across = [(p[0] - centre[0]) * side[0] + (p[1] - centre[1]) * side[1] for p in pts]
    length = max(along) - min(along)
    width = max(across) - min(across)
    if kind == "auto":
        kind = "quadruped" if length > height * 0.7 else "biped"
    if kind == "biped":
        feet = [a for p, a in zip(pts, along) if p[2] < lo[2] + 0.04 * height]
        shins = [a for p, a in zip(pts, along) if lo[2] + 0.07 * height < p[2] < lo[2] + 0.2 * height]
        lead = (sum(feet) / max(len(feet), 1)) - (sum(shins) / max(len(shins), 1))
        confidence = abs(lead) / height
        sign = 1.0 if lead >= 0 else -1.0
        how = "toes"
    else:
        q = length * 0.25
        front = [p[2] for p, a in zip(pts, along) if a > max(along) - q]
        back = [p[2] for p, a in zip(pts, along) if a < min(along) + q]
        tall = (max(front) - max(back)) / height
        # Heavier: more of the body sits in the head-end half.
        heavy = (sum(1 for a in along if a > 0) - sum(1 for a in along if a < 0)) / len(along)
        lean = tall + 0.5 * heavy
        confidence = abs(lean)
        sign = 1.0 if lean >= 0 else -1.0
        how = "head end"
    return {
        "kind": kind,
        "forward": (fwd[0] * sign, fwd[1] * sign),
        "how": how,
        "confidence": confidence,
        "mirror_error": err,
        "mirror_contrast": worst / max(err, 1e-9),
        "height": height,
        "length": length,
        "width": width,
    }


def decide(m, convention="+x"):
    """The forward direction to use: the measurement when it is strong,
    the source's convention when it is weak or the two agree. Returns
    (forward x, y), decided_by, degrees between the two (None without a
    convention)."""
    conv = CONVENTIONS.get(convention)
    fwd = m["forward"]
    if conv is None:
        return fwd, "measurement", None
    dot = max(-1.0, min(1.0, fwd[0] * conv[0] + fwd[1] * conv[1]))
    off = math.degrees(math.acos(dot))
    strong = m["mirror_contrast"] >= MIN_MIRROR_CONTRAST and m["confidence"] >= MIN_CONFIDENCE
    if not strong:
        return conv, "convention", off
    if off < 3.0:
        return conv, "both", off
    return fwd, "measurement", off


def turn_to_minus_y(fwd):
    """Radians to turn about +Z so `fwd` points along -Y (glTF +Z)."""
    return math.atan2(-1.0, 0.0) - math.atan2(fwd[1], fwd[0])
