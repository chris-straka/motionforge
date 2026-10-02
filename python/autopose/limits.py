# MIT (see LICENSE-MIT). Joint-limit tables (motionforge-limits JSON).
"""Load and check rigforge-exported joint limits.

v1 semantics: each entry caps one bone's LOCAL pose rotation from rest
(identity), measured as 2*acos(|w|) in degrees. Bones missing from the
table are unconstrained; table entries for unknown bones are reported
as unmatched (sorted) so a stale table can never silently pass.
"""

import json
import math

LIMITS_FORMAT = "motionforge-limits"


def load_limits(path):
    """Read + validate a limits file. Returns {"rig": str, "bones": {name: max_deg}}."""
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    if doc.get("format") != LIMITS_FORMAT:
        raise ValueError("invalid motionforge-limits: bad \"format\"")
    if doc.get("version") != 1:
        raise ValueError("invalid motionforge-limits: unsupported version")
    bones = doc.get("bones")
    if not isinstance(bones, dict):
        raise ValueError("limits: \"bones\" must be an object")
    out = {}
    for name, entry in bones.items():
        max_deg = (entry or {}).get("max_angle_deg") if isinstance(entry, dict) else None
        if not isinstance(max_deg, (int, float)) or not math.isfinite(max_deg):
            raise ValueError(f"limits: bone {name} missing numeric max_angle_deg")
        if not 0.0 < max_deg <= 180.0:
            raise ValueError(f"limits: bone {name} max_angle_deg {max_deg} out of (0, 180]")
        out[name] = float(max_deg)
    return {"rig": doc.get("rig", ""), "bones": out}


def quat_angle_deg(q):
    """Local rotation from rest (identity) in degrees."""
    return 2.0 * math.degrees(math.acos(min(1.0, abs(q[0]))))


def frame_violations(pred_quats, bones, limits):
    """Per-frame [(bone, angle_deg, max_deg)] in bone order. Unconstrained bones skip."""
    table = limits["bones"]
    out = []
    for name, q in zip(bones, pred_quats):
        if name in table:
            ang = quat_angle_deg(q)
            if ang > table[name] + 1e-9:
                out.append((name, ang, table[name]))
    return out


def unmatched_bones(bones, limits):
    """Sorted table bones absent from the skeleton."""
    have = set(bones)
    return sorted(n for n in limits["bones"] if n not in have)
