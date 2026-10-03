# AutoPose (P3/P4)

The Cascadeur-like part: the animator moves 1-6 joints; a small trained
MLP predicts the rest of the body in a natural pose. Used while
keyframing in Blender (move joints, select them, run AutoPose).

## Model

Input per bone (rig order): constrained `[x, y, z, mask]`, plus the
`n` bone lengths (input dim 5n). Positions are **root-relative**: the
root bone's (bone 0's) current head is subtracted, so a pose predicts
the same wherever the character stands. The weights file records the
encoding in `"input"`: `effector-pos-mask+lengths/root-relative`
(written by `train.py` since 2026-10-03; inference needs the
effectors file's `root_position`, which the Blender extension writes)
or `effector-pos-mask+lengths` (the original armature-space encoding;
also assumed when the tag is missing). Older weights therefore keep
working unchanged; retrain to get the root-relative behaviour. With
absolute inputs, training clips that travel (a walk covers 1.2 m)
taught the model positions that mix pose with location, and a
character posed away from the origin was off-distribution unless its
hips were an effector. Output: one quat per
bone (4n), normalized. MLP with ReLU hidden layers (default 256x256);
a small transformer only if the MLP fails its gate. Training samples
random effector subsets (uniform count 1-6, uniform set, seeded) so any
combination works. Loss: cosine quat distance `1 - |dot|` (smooth; the
acos angle is reported, not backpropped) plus a foot-penetration
penalty (predicted feet below the clip ground, `--foot-weight`).

Inference is plain CPU matvecs in the Rust core over the
`motionforge-weights` custom format (no ONNX runtime): whole-command
1.7 ms on the fixture rig (14 bones), far under the 10 ms gate. The
pure-Python forward pass agrees with the Rust CLI to last-ULP
(component diffs < 1e-12 on the golden).

## Data (licensing is the hard constraint)

Commercial-safe only (see `docs/licensing.md`). CMU mocap is OK with a
recorded terms snapshot; owner clips OK. BLOCKED: AMASS/SMPL lineage,
research-only/NC terms, Hunyuan output anywhere; Mixamo is excluded
from training by default (`--allow-mixamo-training` needs owner
approval). Mocap enters this repo as manifests only (id + source +
license + sha); `tools/manifest.py --check` is the gate, and training
refuses rejected manifests. Retarget everything onto the rigforge
skeleton first: `blender/tools/batch_retarget.py` imports each BVH
into the hero .blend, retargets through the CLI, and writes clips +
manifest rows (all inputs local-only, never committed).

## Training

```bash
pip install -r python/requirements.txt  # torch, CPU or MPS
python3 python/autopose/train.py --manifest data/manifest.json \\
    --out-dir models/run1 --feet DEF-foot.L,DEF-foot.R --epochs 50 --seed 1
python3 python/autopose/train.py --manifest ... --out-dir ... --dry-run  # no torch
```

CPU training is deterministic (`use_deterministic_algorithms`); other
devices warn. Every run writes `weights.json` + `run.json` (config,
manifest sha, metrics, gate verdict).

## Gates (all must pass; owner loosens nothing silently)

- Held-out mean joint position error under `--gate-mm` (default 50
  mm) AND zero foot penetrations AND zero joint-limit violations
  (when `--limits` is given), else exit 1 (`GATE FAIL`).
- Joint limits come from a `motionforge-limits` table (see
  `docs/limits.md`; `tests/fixtures/limits_hero.json` is the example).
  The format and both consumers are done; the per-preset tables are a
  rigforge export (handoff in `docs/limits.md`). Without `--limits`,
  the per-bone max-angle table (`evaluate()`) is still reported for
  owner review.
- Inference under 10 ms per pose on CPU (measured 1.7 ms whole-command
  on 14 bones, release build).
- Owner keyframes one attack faster than without it (needs the owner).

## Smoke proof (toy run, this repo)

48-frame walk clip as train+heldout, MLP 64x64, 30 epochs, CPU: loss
0.589 -> 0.022, heldout angle 110 -> 21 deg. The loop learns; real
accuracy needs real data and epochs (GATE FAIL on the toy, as it should
be). Trained weights load straight into `motionforge autopose`.

## Blender use

Move 1-6 joints in Pose Mode, select them, set the weights path, run
AutoPose From Selection. Only rotation channels are applied (locations
untouched: the model predicts rotations, and zeroing locs would snap
effector moves and root placement). The pose is keyed on the current
frame.
