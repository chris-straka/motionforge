# motionforge — build plan

Written 2026-10-01 for the coding agents who will build it. Goal: make
game animation for HLL (a stylized action-adventure, Godot) fast for one
person. Context: `~/Games/hll/tools/roadmap.md`, `asset-pipeline.md`
("Animation" section). The rig is rigforge's (`~/SWE/rigforge`,
Rigify-based, presets `hll_hero` and `hll_stalker`).

## What it does (three features, in build order)

1. **Retarget assist**: one click from a Mixamo FBX, Cascadeur export,
   or video-to-motion clip onto the rigforge rig, saved as a Blender
   action and exported for Godot. Checks bone mapping, foot contacts
   (no sliding), and root motion.
2. **Motion stylizer**: realistic clip in, snappy game motion out. Find
   extreme poses, drop in-betweens (key reduction), push extremes away
   from neutral per bone chain (exaggeration factor), retime with holds
   and snappy easing, add anticipation and overshoot. Output editable
   keyframes. Pure curve math; no training.
3. **AutoPose (ML, the Cascadeur-like part)**: the animator moves a few
   joints (hands, feet, hips, head); a small trained model predicts the
   rest of the body in a natural pose. Used while keyframing in
   Blender.

4. **Physics pass (no ML, Cascadeur's other half)**: make keyed motion
   physically believable: center-of-mass check against the support
   feet (balance), ballistic arcs for jumps and airborne frames, momentum
   carried through turns and hits. Shows the error as an overlay and
   offers a one-click fix that adjusts the root/hips curves. Optimization
   math, runs fine on a Mac mini; build after the stylizer.

## Architecture

- Blender extension (GPL-3.0-or-later) for the UI and Blender I/O.
- Core math + model inference in a Rust CLI or library (MIT), same split
  as `~/SWE/retopoforge` (subprocess over files, deterministic).
- Training in Python (PyTorch, local on an Apple M4, 16 GB). Exported
  model (ONNX or a small custom format) runs on CPU in the Rust core.

## AutoPose details

- Input: positions (and optionally rotations) of 3-6 "effector" joints
  plus the character's bone lengths. Output: rotations for every
  deform bone of the rigforge rig.
- Model: start small (MLP); a small transformer only if the MLP fails
  the gate. Train with random effector subsets so any combination works.
- Data, licensing is the hard constraint: commercial-safe only.
  - OK: the CMU Graphics Lab motion capture database (free for any use,
    per its site — verify at build time), the owner's own animations
    and recordings.
  - BLOCKED: AMASS / SMPL-based data (non-commercial), anything under a
    research-only license. Mixamo: allowed in games, but check its terms
    before training on it; default exclude.
  - Retarget all data onto the rigforge skeleton first (feature 1).
- Gates: on held-out clips, predicted joint positions within a set
  error, no joint-limit violations, no foot penetration; runs under
  10 ms per pose on CPU.

## Phases

| Phase | Builds | Gate |
|---|---|---|
| P0 | Repo scaffold (Rust workspace, Blender extension, AGENTS.md: no game assets committed, determinism, licensing rules) | builds, CLI `--help` |
| P1 | Retarget assist | a Mixamo clip plays on the `hll_hero` rig in Godot with no foot sliding |
| P2 | Motion stylizer | owner prefers the stylized version of 3 test clips |
| P3 | Data pipeline for AutoPose (licensed sources only, retargeted) | dataset manifest with source + license per clip |
| P4 | AutoPose training + Rust inference + Blender tool | gates above; owner keyframes one attack faster than without it |

## Rules

- No game assets or licensed mocap committed; manifests only.
- Every claim in docs has a measured number and the command behind it.
- Ask the owner before loosening any gate.
