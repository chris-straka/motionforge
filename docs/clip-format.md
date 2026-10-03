# motionforge file formats (v1)

All CLI inputs/outputs are JSON files. The Blender extension and the
CLI exchange exactly these formats; nothing else crosses the
subprocess boundary. Floats are finite decimal numbers; `-0.0` is
normalized to `0.0` on emit. Emitted files are byte-deterministic:
struct key order is fixed, bone order follows the skeleton, and no
timestamps or paths are embedded.

## Clip (`motionforge-clip`)

A skeleton plus dense per-frame poses. Every bone appears in every
frame (no sparse tracks in v1).

```json
{
  "format": "motionforge-clip",
  "version": 1,
  "fps": 30,
  "skeleton": { "bones": [
    {"name": "Hips", "parent": null,
     "head": [0.0, 0.0, 0.98], "tail": [0.0, 0.0, 1.10]}
  ]},
  "frames": [
    {"Hips": {"loc": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0]}}
  ]
}
```

- `loc`/`quat` are the bone's pose transform relative to rest,
  expressed in the bone's parent-relative rest frame — Blender's
  `matrix_basis` decomposed into location + rotation quaternion
  (`[w, x, y, z]`, unit length). Frame count >= 1, 0 < fps <= 10000.
  Coordinates (`head`, `tail`, `loc`) must satisfy |x| <= 1e6 m;
  quaternions are normalized on load and must be neither zero nor so
  large that their length overflows.
- Forward kinematics: `M = P @ R @ B`, where `R` is the
  parent-relative rest matrix (from head/tail/parent, same convention
  as rigforge's `rest_parent_rel`), `B = T(loc) @ Q(quat)`, and `P`
  is the parent's posed matrix. A bone head's world position is the
  translation of `M`.
- Bone rest orientation: bone local +Y runs head->tail with zero roll
  (Blender `matrix` convention for a roll-0 bone). The Blender
  exporter bakes each bone's actual rest orientation into head/tail
  only when roll is 0; bones with non-zero roll are an error until
  the exporter gains a roll channel (documented limit, see
  `docs/retarget.md`).

## Skeleton (`motionforge-skeleton`)

The `skeleton` object of a clip as a standalone file (retarget
targets, bone-length inputs):

```json
{"format": "motionforge-skeleton", "version": 1,
 "bones": [{"name": "DEF-spine", "parent": null,
            "head": [0,0,1], "tail": [0,0,2]}]}
```

`motionforge clip-info --emit-skeleton in.json --output skel.json`
extracts one from a clip.

## Bone map (`motionforge-bonemap`)

Source->target correspondence for `retarget`:

```json
{"format": "motionforge-bonemap", "version": 1,
 "pairs": [{"source": "mixamorig:Hips", "target": "DEF-spine"}],
 "root_source": "mixamorig:Hips",
 "root_target": "DEF-spine",
 "stride_source": ["mixamorig:LeftUpLeg", "mixamorig:LeftFoot"],
 "stride_target": ["DEF-thigh.L", "DEF-foot.L"],
 "feet_source": ["mixamorig:LeftFoot", "mixamorig:RightFoot"],
 "feet_target": ["DEF-foot.L", "DEF-foot.R"]}
```

Only `pairs`, `root_source`, `root_target` are required. Without the
stride pair, root translation transfers unscaled (reported). Without
feet, foot-slide measurement is skipped (reported).

## Keys (`motionforge-keys`)

Per-bone key frames for thinned import (from `stylize --keys-out`).
Indices are clip frame numbers (0-based), ascending:

```json
{"format": "motionforge-keys", "version": 1,
 "bones": {"Hips": [0, 47], "LeftUpLeg": [0, 9, 12, 33, 36, 47]}}
```

## Effectors (`motionforge-effectors`)

AutoPose request: a skeleton plus 1-6 constrained joints in armature
space, and the root bone's (skeleton bone 0's) current head. That
`root_position` is required by root-relative weights and ignored by
absolute ones (see `docs/autopose.md`):

```json
{"format": "motionforge-effectors", "version": 1,
 "skeleton": {"bones": [...]},
 "root_position": [0.0, 0.0, 0.98],
 "effectors": [{"bone": "DEF-hand.L", "position": [0.5, 0.0, 1.2]}]}
```

## Frame physics (`motionforge-frame-physics`)

Single-frame balance snapshot (stdout of `physics-frame`, parsed by the
Blender overlay operator):

```json
{"format": "motionforge-frame-physics", "version": 1, "frame": 1,
 "com": [0, 0, 0.64], "root": [0, 0, 0.7],
 "feet": {"DEF-foot.L": [0.09, 0, 0.08]},
 "supporters": ["DEF-foot.L", "DEF-foot.R"],
 "support_center": [0, 0, 0.08], "support_radius": 0.21,
 "excursion_m": -0.2, "balanced": true, "airborne": false}
```

Airborne frames carry `"support_center": null`, empty supporters, and
`"airborne": true`.

## Weights (`motionforge-weights`)

AutoPose MLP in a small custom format (CPU inference, no runtime):

```json
{"format": "motionforge-weights", "version": 1,
 "bones": ["DEF-spine", "..."],
 "input": "effector-pos-mask+lengths/root-relative",
 "layers": [{"weights": [[0.01]], "bias": [0.0]}]}
```

Input vector: per bone in `bones` order, `[px, py, pz, mask]`
(constrained position, or zeros + mask 0), followed by the `n` bone
lengths (tail-head distance). `"input"` names the position encoding:
`effector-pos-mask+lengths/root-relative` (position minus the
effectors file's `root_position`; current training output) or
`effector-pos-mask+lengths` (armature-space; the original encoding,
also assumed when `"input"` is absent). Any other value is an error. Output: `4n` numbers, one
`(w, x, y, z)` quat per bone, normalized by inference. Hidden layers
use ReLU; the last layer is linear. `layers[i].weights` is
row-major, `rows = outputs`, `cols = inputs`.

## Report (stdout)

Every command prints a human-readable `=== motionforge <cmd> ===`
report to stdout (numbers the docs quote) and writes files only to
`--output`. Reports go to stdout; warnings/errors to stderr; exit
codes: 0 ok, 1 usage error, 2 input error, 3 processing error.
