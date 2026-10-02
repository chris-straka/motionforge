# MIT (see LICENSE-MIT). Weights export + pure-Python inference (stdlib only).
"""Export trained MLPs to motionforge-weights JSON and evaluate them.

export_weights() takes plain nested lists (train.py extracts them from
the torch state dict), so this module stays torch-free and testable on
any interpreter. forward() is the reference implementation of inference
(same op order as the Rust core); the cross-check test runs it against
the Rust golden output.
"""

import json
import math


def export_weights(bones, layers):
    """bones: [names]; layers: [(weights_rows, bias)] with plain floats.
    Returns the weights document dict (deterministic key order)."""
    n = len(bones)
    if not n:
        raise ValueError("no bones")
    if not layers:
        raise ValueError("no layers")
    if len(layers[0][0][0]) != 5 * n:
        raise ValueError(f"input dim {len(layers[0][0][0])} != 5 * {n}")
    for (w0, _b0), (w1, _b1) in zip(layers, layers[1:]):
        if len(w0) != len(w1[0]):
            raise ValueError(f"layer dims {len(w0)} -> {len(w1[0])} disagree")
    if len(layers[-1][0]) != 4 * n:
        raise ValueError(f"output dim {len(layers[-1][0])} != 4 * {n}")
    return {
        "format": "motionforge-weights",
        "version": 1,
        "bones": list(bones),
        "input": "effector-pos-mask+lengths",
        "layers": [
            {"weights": [list(map(float, row)) for row in w],
             "bias": [float(b) for b in bias]}
            for (w, bias) in layers
        ],
    }


def write_weights(path, doc):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f)
        f.write("\n")


def load_weights(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def forward(weights_doc, x):
    """MLP forward pass over plain lists. Returns [quats]."""
    layers = weights_doc["layers"]
    for li, layer in enumerate(layers):
        y = []
        for r, row in enumerate(layer["weights"]):
            acc = layer["bias"][r]
            for w, v in zip(row, x):
                acc += w * v
            y.append(acc if li == len(layers) - 1 else max(0.0, acc))
        x = y
    quats = []
    for i in range(0, len(x), 4):
        w, a, b, c = x[i : i + 4]
        l = math.sqrt(w * w + a * a + b * b + c * c)
        quats.append((1.0, 0.0, 0.0, 0.0) if l == 0.0 else (w / l, a / l, b / l, c / l))
    return quats


def quat_angle(a, b):
    """Angular distance between quats in radians."""
    d = abs(a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3])
    return 2.0 * math.acos(min(1.0, d))


def evaluate(weights_doc, skeleton, frames, effector_sets, foot_names=(), limits=None):
    """Held-out style metrics over sampled effector sets.

    frames: list of poses (dataset format); effector_sets: list of id
    lists (one per frame, reused cyclically). Returns dict with
    mean/max joint position error (m), mean angular error (rad), foot
    penetration count, per-bone max angle (rad) for limit review, and
    (when limits is given) limit_violations [(frame, bone, angle_deg,
    max_deg)] plus limits_unmatched [sorted bone names].
    """
    from .dataset import build_input, fk
    from .limits import frame_violations, unmatched_bones

    if skeleton.names != weights_doc["bones"]:
        raise ValueError("weights bones do not match the skeleton")
    feet = [skeleton.names.index(n) for n in foot_names]
    ground = min(
        fk(skeleton, poses)[f][2] for poses in frames for f in feet
    ) if feet else 0.0
    pos_errs = []
    ang_errs = []
    penetrations = 0
    bone_max_angle = [0.0] * len(skeleton.names)
    violations = []
    for fi, poses in enumerate(frames):
        heads = fk(skeleton, poses)
        x = build_input(skeleton, heads, effector_sets[fi % len(effector_sets)])
        pred = forward(weights_doc, x)
        pred_heads = fk(skeleton, [(p[0], q) for p, q in zip(poses, pred)])
        for i, (got, want) in enumerate(zip(pred_heads, heads)):
            pos_errs.append(math.dist(got, want))
        for i, (q, (_loc, want)) in enumerate(zip(pred, poses)):
            ang = quat_angle(q, want)
            ang_errs.append(ang)
            bone_max_angle[i] = max(bone_max_angle[i], ang)
        if limits is not None:
            for bone, ang_deg, max_deg in frame_violations(pred, skeleton.names, limits):
                violations.append((fi, bone, ang_deg, max_deg))
        for f in feet:
            if pred_heads[f][2] < ground - 1e-9:
                penetrations += 1
    return {
        "frames": len(frames),
        "mean_joint_m": sum(pos_errs) / len(pos_errs),
        "max_joint_m": max(pos_errs),
        "mean_angle_rad": sum(ang_errs) / len(ang_errs),
        "foot_penetrations": penetrations,
        "bone_max_angle_rad": dict(zip(skeleton.names, bone_max_angle)),
        "limit_violations": violations,
        "limits_unmatched": unmatched_bones(skeleton.names, limits) if limits is not None else [],
    }
