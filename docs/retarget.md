# Retarget assist (P1)

One click from a Mixamo FBX, Cascadeur export, or video-to-motion clip
onto the rigforge rig, saved as a Blender action and exported for
Godot. The Blender operator handles FBX/action/GLB I/O; the Rust core
does the frame math; the report validates bone mapping, foot contacts,
and root motion.

## Flow

1. Import the clip into Blender (Mixamo FBX via `io_scene_fbx`, or any
   armature + action). Its rest pose should be a neutral stance.
2. Select the rigforge rig (active object), pick the source armature in
   the MotionForge panel, choose the `Mixamo -> HLL Hero` bonemap
   preset (or a custom file), run Retarget.
3. The operator exports the source action (stripping `mixamorig:` /
   `mixamorig_` prefixes to the bonemap short form) and the target
   skeleton, runs `motionforge retarget`, and imports the result as
   `<action>_retargeted`.
4. Export the game GLB via rigforge (`motionforge.export_godot`
   forwards to `wm.rigforge_game_export` when rigforge is enabled).

## Math

Ported from rigforge's `tools/retarget_mixamo.py` onto clip JSON, with
one deliberate improvement. Per mapped pair per frame, the source
bone's local motion `d` (deviation from source rest) is re-expressed in
the target bone's local frame by conjugation through armature space:
`d_t = C d C^-1` with `C = R_t^-1 P_t^-1 Q P_s R_s` (`R` =
parent-relative rest, `P` = posed parent from FK, `Q` = yaw fix).
Rest source frames map to rest target frames, and transfer between
identical skeletons is exact (pinned by
`retarget::tests::identity_transfer_is_exact`). The reference instead
applies the armature-space motion on top of the target's posed parent
chain, which is exact only at the root and first-order elsewhere; on
real clips the two agree closely (both are same-angle-about-mapped-axis
transfers), with the C form exact in the degenerate cases.

Root translation transfers scaled by the leg-length ratio (stride
scale) onto every driven chain root, since the DEF hierarchy does not
hang the limbs under the pelvis. Undriven target bones keep identity
bases (rigid follow of the posed parent, matching Blender behavior for
unkeyed bones). Facing: yaw auto-flips when the two sides' foot bones
point opposite ways on Y (sign of the rest foot direction); without
feet on either side, yaw is left unchanged and reported.

Foot slide is measured exactly as in the reference (stance = bottom
35% of foot height AND below-median horizontal speed), and the pin
pass ramps each stance interval's drift out through the chain-root
location curves. The pin is root-shift compensation, not IK: boundary
pops at stance edges are possible (visible in the fixture golden as a
higher post-pin overall max), and residual slide is always re-measured
and reported honestly.

## Blender-exactness proofs

- Rest bases: the zero-roll basis (shortest arc +Y -> bone direction,
  -Y via 180 deg about +Z) matches Blender 5.2's `vec_roll_to_mat3`
  roll-0 output on all six axis directions plus diagonals (headless
  probe, 2026-10-01; pinned in `math::tests` and `test_pure.py`).
- FK: `M = P @ R @ B` reproduces Blender's evaluated head positions
  (probe scene: root basis 90 deg about local Z puts the child head at
  (99.5, 0, 1.5); pinned in `clip::tests::fk_matches_blender` and the
  Python dataset tests).
- Walk fidelity: the procedural fixture walk reproduces rigforge's
  reference procedural walk foot-slide numbers to 4 decimals
  (stance 8, mean 0.5449, max 0.7468, overall 1.4869 m/s) — same walk
  math, same stance port, exact agreement.

## Limits (v1)

- Zero-roll bones only; non-zero roll and non-unit basis scale are
  refused with the offending bone names (no roll/scale channels yet).
- Chain-root translation approximation is inherited from the reference
  (mm-scale for locomotion, wrong for acrobatics; a control-rig
  retarget is the v2 fix).
- Fingers map proximals only in the preset; distal joints rigid-follow.
- Live comparison against the reference script on a REAL Mixamo clip
  (same clip through both, compare slide/travel) is owner follow-up;
  the preset ships from the reference's constants unmodified.

## Commands

```bash
./rust/target/release/motionforge retarget --source SRC.json --target TGT.json \\
    --map blender/motionforge/presets/mixamo_hllhero.json --output OUT.json
./rust/target/release/motionforge retarget --help  # --yaw auto|flip|none, --pin/--no-pin
```
