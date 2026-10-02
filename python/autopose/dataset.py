# MIT (see LICENSE-MIT). Clip loading + FK + effector sampling (stdlib only).
"""Training pairs for AutoPose without importing torch: parsing, forward
kinematics, and random effector-subset sampling. train.py converts the
plain-list batches to tensors; evaluate.py runs the same path for
metrics. The FK here must match motion_core (same zero-roll rest
convention); test_fk_matches_blender pins the shared probe numbers."""

import json
import math
import random


def _vsub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _vdot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _vcross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _vnorm(a):
    l = math.sqrt(_vdot(a, a))
    return (a[0] / l, a[1] / l, a[2] / l)


def _qmul(p, q):
    pw, px, py, pz = p
    qw, qx, qy, qz = q
    return (
        pw * qw - px * qx - py * qy - pz * qz,
        pw * qx + px * qw + py * qz - pz * qy,
        pw * qy - px * qz + py * qw + pz * qx,
        pw * qz + px * qy - py * qx + pz * qw,
    )


def _qconj(q):
    return (q[0], -q[1], -q[2], -q[3])


def _qmat(q):
    w, x, y, z = q
    return (
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ),
        (
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        ),
        (
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ),
    )


def _mtranspose(m):
    return tuple(tuple(m[r][c] for r in range(3)) for c in range(3))


def _mmul(a, b):
    return tuple(
        tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)) for i in range(3)
    )


def _mvec(m, v):
    return tuple(sum(m[i][k] * v[k] for k in range(3)) for i in range(3))


def _shortest_arc(fr, to):
    d = _vdot(fr, to)
    if d >= 1.0 - 1e-12:
        return (1.0, 0.0, 0.0, 0.0)
    if d <= -1.0 + 1e-12:
        return (0.0, 0.0, 0.0, 1.0)  # 180 deg about +Z
    ax = _vnorm(_vcross(fr, to))
    half = math.acos(max(-1.0, min(1.0, d))) / 2.0
    s = math.sin(half)
    return (math.cos(half), ax[0] * s, ax[1] * s, ax[2] * s)


class Skeleton:
    """Clip skeleton with cached rest kinematics."""

    def __init__(self, bones):
        # bones: [(name, parent_name_or_None, head, tail)].
        self.names = [b[0] for b in bones]
        self.index = {b[0]: i for i, b in enumerate(bones)}
        self.parents = [self.index[b[1]] if b[1] else None for b in bones]
        self.heads = [tuple(b[2]) for b in bones]
        self.lengths = [math.dist(b[2], b[3]) for b in bones]
        self.world = []
        for (_name, _p, head, tail) in bones:
            direction = _vnorm(_vsub(tail, head))
            self.world.append(_qmat(_shortest_arc((0.0, 1.0, 0.0), direction)))
        self.rel_rot = []
        self.rel_off = []
        for i in range(len(bones)):
            p = self.parents[i]
            if p is None:
                self.rel_rot.append(self.world[i])
                self.rel_off.append(self.heads[i])
            else:
                pw = _mtranspose(self.world[p])
                self.rel_rot.append(_mmul(pw, self.world[i]))
                self.rel_off.append(_mvec(pw, _vsub(self.heads[i], self.heads[p])))


def load_clip(path):
    """(Skeleton, frames). frames[f][b] = (loc, quat), skeleton order."""
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    if doc.get("format") != "motionforge-clip" or doc.get("version") != 1:
        raise ValueError(f"{path}: not a motionforge-clip v1 document")
    bones = [
        (b["name"], b["parent"], tuple(b["head"]), tuple(b["tail"]))
        for b in doc["skeleton"]["bones"]
    ]
    skeleton = Skeleton(bones)
    frames = []
    for frame in doc["frames"]:
        poses = []
        for name in skeleton.names:
            pose = frame[name]
            poses.append((tuple(pose["loc"]), tuple(pose["quat"])))
        frames.append(poses)
    return skeleton, frames


def fk(skeleton, poses):
    """Armature-space head positions for one frame's poses."""
    rots = [None] * len(skeleton.names)
    heads = [None] * len(skeleton.names)
    for i in range(len(skeleton.names)):
        brot = _qmat(poses[i][1])
        local_rot = _mmul(skeleton.rel_rot[i], brot)
        local_off = tuple(
            a + b
            for a, b in zip(
                _mvec(skeleton.rel_rot[i], poses[i][0]), skeleton.rel_off[i]
            )
        )
        p = skeleton.parents[i]
        if p is None:
            rots[i], heads[i] = local_rot, local_off
        else:
            rots[i] = _mmul(rots[p], local_rot)
            moved = _mvec(rots[p], local_off)
            heads[i] = (moved[0] + heads[p][0], moved[1] + heads[p][1], moved[2] + heads[p][2])
    return heads


def check_shared_skeleton(skeletons):
    """All training clips must share one rig (names, order, rest)."""
    first = skeletons[0]
    for other in skeletons[1:]:
        if other.names != first.names:
            raise ValueError("training clips must share bone names and order")
        for a, b in zip(other.heads, first.heads):
            if math.dist(a, b) > 1e-9:
                raise ValueError("training clips must share rest heads")
        if any(abs(a - b) > 1e-12 for a, b in zip(other.lengths, first.lengths)):
            raise ValueError("training clips must share bone lengths")
    return first


def sample_effectors(rng, n_bones, min_k=1, max_k=6):
    """Random effector subset: uniform count in [min_k, max_k], uniform set."""
    if not isinstance(rng, random.Random):
        raise TypeError("pass a random.Random (seeded, reproducible)")
    k = rng.randint(min_k, min(max_k, n_bones))
    return sorted(rng.sample(range(n_bones), k))


def build_input(skeleton, heads, effector_ids):
    """Flat model input: per bone [x, y, z, mask], then bone lengths."""
    eff = set(effector_ids)
    x = []
    for i in range(len(skeleton.names)):
        if i in eff:
            x.extend([heads[i][0], heads[i][1], heads[i][2], 1.0])
        else:
            x.extend([0.0, 0.0, 0.0, 0.0])
    x.extend(skeleton.lengths)
    return x


def build_target(poses):
    """Flat target: per bone (w, x, y, z)."""
    t = []
    for (_loc, quat) in poses:
        t.extend([quat[0], quat[1], quat[2], quat[3]])
    return t
