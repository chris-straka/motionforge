# GLB rig adapters (genforge character chain)

genforge's `gen character` chain (genforge `PLAN.md` P7) runs three
stages through motionforge: **standardize**, **pose-test** and
**animate**. Each is a plain CLI command on GLB files plus an `adapter`
wrapper that speaks genforge's contract. Everything is in the MIT Rust
core (`rust/core/src/{glb,rig,humanoid,standardize,animate,posetest}.rs`);
the only Blender step is rendering the pose sheet
(`blender/tools/pose_sheet.py`, embedded in the binary).

## Contract

```sh
motionforge adapter standardize IN.glb OUT.glb RESULT.json [--class humanoid|quadruped|custom]
motionforge adapter pose-test   IN.glb OUT.glb RESULT.json [--clips LIB] [--blender BIN]
motionforge adapter animate     IN.glb OUT.glb RESULT.json --clips LIB [--fps 30]
```

`RESULT.json` matches the weightforge adapter's schema:
`{"ok": bool, "outputs": [paths relative to RESULT.json's folder],
"tool": "motionforge", "version", "stage", "<stage>": {details},
"reason"?}`. Exit 0 = ok; 1 = the stage ran and failed (`ok: false`,
result written, e.g. a rig without the humanoid core); 2 = error
(unreadable input, missing `--clips`, no Blender): no result file, so
genforge stops instead of guessing. Inputs are never modified.
`--clips` takes a GLB or a folder of GLBs (sorted), repeatable.

Outputs: standardize and animate write `OUT.glb`. pose-test writes
`pose-sheet.png` and `pose-test.glb` (the poses as animations, for a
spinnable view) next to `OUT.glb`, and passes the model on unchanged
as `OUT.glb`; `outputs` lists the sheet first (genforge's preview) and
the model last (the next stage's input).

## The HLL humanoid skeleton

Canonical names are rigforge's `hll_hero` deform bones, and the table
(`humanoid::canonical`) is the 65-bone set of rigforge's mobile game
export: spine chain `DEF-spine` (hips) to `DEF-spine.006` (head),
shoulders, arms, hands, palms, three-segment fingers, pelvis, breast,
thighs, shins, feet, toes. Parents are the anatomical chain (thigh
under the hips, upper arm under the shoulder, shoulder under
`DEF-spine.003`), not Rigify's flat deform parenting, so a local
rotation means the same thing on every character. A rig keeps the
subset it has; animation needs the 14-bone core (`humanoid::REQUIRED`:
hips, head, both arms to the hand, both legs to the foot). Extra
rigforge bones (face, twists) stay when already `DEF-` named.

**Twist/helper bones** (2026-10-05, `helpers.rs`): `DEF-upper_arm_twist.L/R`
and `DEF-thigh_twist.L/R`. Each sits on its driver's joint (same head,
same rest orientation), hangs from the driver's parent (shoulder, hips)
and turns by half (`SHARE` 0.5) of the driver's rotation relative to that
parent, swing and twist alike. Why half the whole rotation and not a
classic counter-roll twist bone: the poses that fail weightforge's gate
(arm up, arm forward, hip forward) are pure swings with zero roll, so a
roll-only bone does nothing there; a half-swing joint lets the armpit and
groin blend over two 45 deg steps instead of one 90 deg step. Nodes carry
`extras.hll_helper = {"driver", "share"}`. The 69-bone canonical table
lists them after their driver's siblings.

Why motionforge and not rigforge: standardizing is what makes clips
shareable, and it needs the same bone knowledge as retargeting; it is a
pure GLB rewrite (no Blender), so it belongs in the deterministic Rust
core next to the retarget math. rigforge stays the rig builder (its
rerig is the `repair-rig` adapter).

## standardize

1. Map joints to canonical names. Exact `DEF-*` names first, then
   aliases with side detection: Mixamo (`mixamorig:LeftArm`, also what
   Tripo returns for `spec: mixamo`, which genforge requests for
   humanoids), Unreal-style (`upperarm_l`, `calf_r`, `index_01_l`),
   plain (`UpperArm_L`, `Clavicle_R`, `Thigh.L`). The spine between hips
   and neck maps by position (1 bone -> `.001`; 2 -> `.001`, `.003`; 3+ ->
   first two and the last). Hips fall back to the thighs' parent.
2. Missing core bones: `ok: false` with the list. No skin: `ok: false`.
3. Rename, then reparent into the canonical chain keeping each joint's
   world rest transform, so inverse bind matrices stay valid and the
   mesh does not move (tests check skinned rest positions to 1e-5 m).
4. Joints with no canonical name leave the skin; their weights go to
   the nearest kept ancestor (else the nearest kept joint). Each vertex
   keeps its 4 largest influences, renormalized.
5. Old animations are dropped (authored for the old hierarchy).
6. Missing twist/helpers are added (humanoids): a node per helper at its
   driver's joint, appended last to the skin with the driver's inverse
   bind matrix, **with no weight**. weightforge's fix weights them when
   the deformation check needs it (its `helper-band` candidates); until
   then they change nothing. `RESULT.json` lists them (`helpers_added`).
   Standardizing twice gives the same file (helpers are found by name).

7. Hand sockets `Socket_Hand_L/R` are added under the hands (plain
   nodes, not joints; `docs/contact.md`), and the dropped animations'
   buffer data leaves the file.

Non-humanoids (`--class quadruped|custom`) keep their skeleton and only
gain the `DEF-` prefix the rig contract (rfcheck) requires.

## animate (retarget)

Both rigs are mapped onto canonical names, so any rig the mapper reads
can be a clip source. Per frame and shared bone, the source bone's
world rotation change from rest is applied to the target bone's world
rest rotation, after a yaw fix when the rigs face different ways
(facing = heel-to-toe direction, glTF +Y up), then expressed under the
target's posed parent. Hips travel scales by the hip-height ratio.
Unshared target bones keep their rest pose relative to their parent,
except the twist/helpers: every clip keys them with half their driver's
rotation (`helpers::drive`), so the game plays them as ordinary joints
with no constraint code (test: `clips_and_poses_bake_half_the_driver_into_helpers`).
This is `retarget.rs`'s transfer (`d_t = C d C^-1`) on glTF joints with
arbitrary rest orientations (no zero-roll limit), and it stays exact
when the source has extra in-between bones. Rest alignment comes first:
each target bone is turned (shortest arc, toward the first child both
rigs have) so its rest direction matches the source's, so a clip made on
an A-pose rig plays with the same bone directions on a T-pose rig
instead of lifting its arms by the 45 deg rest difference (test:
`retarget_aligns_rest_poses`, directions within 0.05 deg). Each clip reports a
self-check (`max_error_deg`, world rotation mismatch; the adapter fails
above 0.5 deg) and root travel. Clip names are kept (`-2` on clashes).

Then the contact pass runs on every clip (`docs/contact.md`; the
adapter's `animate.clips[].contact` reports it per arm), and the
target's hand sockets turn to the clips' grip. `--no-contact` skips it.

v1 limits: humanoids only; no foot pinning (the source's contacts are
reproduced exactly only when proportions match); CUBICSPLINE source
keys are sampled linearly.

## pose-test

Twelve range-of-motion poses (arms up 70, arms forward 80, elbows 140,
shrug/wrists/fists, arms back 40, knee up 90/110, lunge, legs out 45,
bend forward 45, lean left 30, twist 40 + head 35, plus rest), in the
spirit of weightforge's ROM set: bend a bone toward a character
direction or twist it, applied in the bone's rest frame on top of its
parent. Helpers are driven the same way as in clips
(clip samples without helper keys get them too), the report lists them
(`helpers`), and `pose-test.glb` keys them. With `--clips`, two frames each (25%, 60%) of up to three
retargeted clips join the sheet. Every pose is a one-key animation with
all joints keyed; `pose_sheet.py` renders each as an orthographic
three-quarter tile (EEVEE, 320x400) with its label and packs a
six-column PNG.

## Measurements (2026-10-05, M4, Blender 5.2.1)

- Tests: `cd rust && cargo test --locked --release`: 71 core unit +
  11 rig-adapter (`core/tests/rig_adapters.rs`) + 3 adapter contract
  (`cli/tests/adapter_contract.rs`, the pose-sheet case renders with
  real Blender when present, ~20 s) + 10 CLI goldens.
- Fixture coverage (`motionforge fixture-glb`): Mixamo, plain and `DEF`
  naming, each with straight and arbitrarily rotated rest joints;
  standardize keeps skinned rest positions within 1e-5 m; retargeting
  a walk onto a standardized copy of the same body reproduces every
  shared joint position within 2e-4 m (f32 key storage); a source facing
  backwards still walks the target toward its own front.
- A real 28-bone, 9-clip game rig with a 10,395-vertex textured mesh
  (owner's asset, not committed): standardize 28 -> 22 joints, six
  finger joints merged into the hands, 0 reparents needed; animate
  retargets all 9 clips with max self-check error 0.0001 deg; output
  passes rfcheck's contract (budget warnings only: verts, texture
  size, no normal map). Its pose-test sheet (12 ROM poses + 6 attack
  clip samples, textured, ~25 s with Blender) shows wrist creasing,
  elbow pinch and knee crumple at the extremes: the deformation the
  owner approves or sends back.
- In genforge's rehearsal mode (genforge `docs/phase-7.md`) all three
  adapters ran inside the chain on real files: standardize, pose-test
  (gate) and animate (9 game clips, then idle/walk/run), through the
  post-animation recheck to delivery.
- Twist/helpers (2026-10-05): `cargo test --locked --release` is 72 core
  unit + 13 rig-adapter + 3 adapter contract + 10 CLI goldens. On the
  SkinTokens rig of the Andras game mesh (5,766 verts) standardize gives
  22 + 4 helper joints; weightforge with the helpers scores 47.6 raw ->
  68.8 after `weights fix` (2 failing regions left) vs 47.3 -> 62.8 (4)
  without them. Numbers and sheets: weightforge README, "Twist/helper
  bones".

## Shoulder helpers (tried 2026-10-06, not shipped)

Owner decision: shoulder helper bones first, corrective shapes only if
weightforge still shows shoulder collapse. Two designs were built and
measured on Andras (genforge rehearsal rig, after the weights gate) with
weightforge's post-animation fix; neither beat the existing half-turn
twist helper, so neither ships (patches kept outside the repo):

| helpers on the upper arm | placeholder clips, fix | CC0 library, fix |
|---|---|---|
| twist (0.5) only, as shipped | 44.7 -> 51.0 | 29.9 |
| + glenohumeral helper (0.25 share) | 44.7 -> 44.9 | 30.2 |
| + glenohumeral helper (0.333), before the contact pass | 45.7 -> 48.3 | |
| + clavicle-pivot helper (0.2-0.5) | 52.5 -> 52.5 (fix finds nothing) | |

Upper bounds tried on the same clips (weightforge check with other
skinning in place of LBS): dual-quaternion skinning 25.9-41.0,
smoothed-weight DQS 27-30, a quick ARAP relaxation 18.6. The CC0 source
mannequin fails the same gate on its own clips (35.5). The shoulder and
thigh findings are what a deep lunge, a tuck roll and an overhead swing
do to a skinned mesh, so the next lever is the gate's policy for clip
poses or corrective shapes (README of weightforge, genforge D63).
