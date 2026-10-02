# SPDX-License-Identifier: GPL-3.0-or-later
"""Pure tuple-based quat/vec math (no bpy): the zero-roll rest basis.

Must match motion_core::math exactly: shortest-arc rotation taking +Y
onto the bone direction, degenerate -Y via 180 deg about +Z. The
exporter compares this basis against each bone's rest orientation and
refuses bones with non-zero roll (v1 clip format has no roll channel).
"""

import math

Y = (0.0, 1.0, 0.0)


def vsub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


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
    return (a[0] / l, a[1] / l, a[2] / l)


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


def qaxis(axis, angle):
    ax, ay, az = vnorm(axis)
    s, c = math.sin(angle / 2.0), math.cos(angle / 2.0)
    return (c, ax * s, ay * s, az * s)


def shortest_arc(fr, to):
    """Unit quat taking unit `fr` onto unit `to` (matches Rust)."""
    d = vdot(fr, to)
    if d >= 1.0 - 1e-12:
        return (1.0, 0.0, 0.0, 0.0)
    if d <= -1.0 + 1e-12:
        return qaxis((0.0, 0.0, 1.0), math.pi)
    axis = vnorm(vcross(fr, to))
    return qaxis(axis, math.acos(max(-1.0, min(1.0, d))))


def quat_to_mat3(q):
    """Unit quat -> 3x3 rows (same formula as the Rust core)."""
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


def rest_basis(head, tail):
    """Zero-roll rest basis rows for a bone running head->tail."""
    direction = vnorm(vsub(tail, head))
    return quat_to_mat3(shortest_arc(Y, direction))


def max_abs_diff(a_rows, b_rows):
    """Max |a-b| over two 3x3 row tuples."""
    return max(abs(a - b) for ra, rb in zip(a_rows, b_rows) for a, b in zip(ra, rb))
