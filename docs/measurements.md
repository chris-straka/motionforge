# Measurements

Every numeric claim in these docs, with the command behind it. All run
2026-10-01/02 on Apple Silicon (M4, 16 GB), Blender 5.2.1 LTS, release
build unless noted. Fixture paths are `tests/fixtures/`; goldens pin
the CLI outputs byte-for-byte.

## GLB rig adapters (2026-10-05)

See `docs/adapters.md` (measurements section): 10 rig-adapter + 3
adapter-contract tests on top of the suite below, and a real game rig
through standardize/animate/pose-test.

## Rust suite

- 64 unit + 10 contract tests green, zero warnings, fmt clean:
  `cd rust && cargo fmt --all --check && cargo test --locked --release`
  (2026-10-03, also on Linux x86_64 / glibc, rustc 1.97.0).
- Cross-platform byte-determinism: before `detmath`, the stylize
  golden (generated on macOS aarch64) drifted on Linux x86_64 in 5 of
  its lines (1-ulp `slerp` differences from the platform `acos`/`sin`).
  `detmath` is bit-identical to rust-lang/libm 0.2.15 on 84,000,000
  comparisons (sin, cos, sincos, asin, acos, exp over random bit
  patterns and dense [-10,10], [-1,1], [-750,710] sweeps; throwaway
  harness linking the libm crate source, 2026-10-03). Regenerating
  the goldens changed 3 of 4106 numbers in `stylize.0.json`, each by
  1 ulp; the other 14 golden files were already identical.
  In-tree: `cargo test --locked --release -p motion_core detmath`
  (within 2 ulp of the host libm on 80k inputs, pinned bit patterns,
  guard against platform libm calls in core code).
- Offline build (zero crates): `cargo build --offline` (no network).
- CI (`.github/workflows/ci.yml`) runs the suite above on Linux x86_64
  and macOS arm64, plus the stdlib Python suites and, on Linux with
  Blender 5.2.1, `blender/tests/test_headless.py` and
  `blender.tests.test_batch` (`MF_REQUIRE_BLENDER=1`: fail, not skip).
- Robustness: a throwaway structured fuzzer (723 CLI runs over mutated
  clips/skeletons/bone maps and edge-case flags, 2026-10-03) found a
  stack-overflow abort on deeply nested JSON, a quadratic parse of
  long non-ASCII strings (200k chars > 20 s), and hangs on
  `--smooth-sigma 1e11` / `--smooth-pad 1e11`. Now: nesting capped at
  512 (clean error), per-char UTF-8 decode (same 400k-char string in
  the unit test parses instantly), smoothing flags bounded to 1000
  frames. Absurd magnitudes (1e300 positions or fps, 1e308 foot
  radius) used to print `inf` in text reports, and a quaternion like
  `[1e300, 1e300, 0, 0]` normalized silently to all zeros; clips now
  bound coordinates (1e6 m), fps (1e4) and quaternion length, and
  physics margins are bounded to 100 m. Re-run: 723 cases, 0 problems.

## Blender-exactness

- Roll-0 bases match Blender 5.2 on 6/6 axis directions (headless
  probe `/tmp/mf_matrix_probe2.py`, since superseded): pinned in
  `math::tests::zero_roll_basis_matches_blender` and
  `blender/tests/test_pure.py`.
- FK matches evaluated Blender heads: probe scene puts the child head
  at (99.5, 0, 1.5); pinned in `clip::tests::fk_matches_blender` and
  `python/tests/test_autopose.py::TestFk`.
- Fixture walk reproduces rigforge's reference procedural walk slide
  exactly (stance 8, mean 0.5449, max 0.7468, overall 1.4869 m/s):
  reference via headless `retarget_mixamo.build_procedural_source` +
  `measure_foot_slide` (`/tmp/mf_ref_slide.py`); ours via
  `motionforge retarget` with an identity map (`--yaw none --no-pin`).

## Retarget (fixture golden)

- `motionforge retarget --source walk_src.json --target hero_skel.json
  --map hero_map.json --output ...`: mapped 12 pairs (2 toe rows
  skipped, no source bones), chain root DEF-spine, stride 0.7045, yaw
  flip, root travel 0.8686 m (1.2 x 0.7045 + sway/bob), max swing
  0.2689 rad.
- Foot slide: before mean 0.3963 / max 0.5410 (L); after mean 0.1380 /
  max 0.4626 (L), 0.2088 / 0.4798 (R); pin 4 intervals, 0.1249 m drift.
  Post-pin overall max rises (1.01 -> 1.80/2.16) from stance-boundary
  steps — the faithful-port tradeoff, documented in `docs/retarget.md`.
- Identity transfer is bit-near-exact (unit test); rest maps to rest
  across different proportions; yaw flip reverses travel (headless
  Blender check: 0.000 -> -0.150).

## Stylize (fixture golden)

- `motionforge stylize --input walk_src.json ... --chain Arm:1.8`: 48
  frames -> 14 keys, 13 counters added / 1 skipped.
- Unit pins: 2x exaggeration 0.5 -> 1.0 rad; ease-out-back midpoint
  0.1088 vs 0.0875 without; holds exact; 2-frame clips exact.
- Root travel (Hips loc, golden vs `walk_src.json`, 2026-10-03):
  before, 15 of 47 steps backward, 2 frozen, steps up to 0.104 m
  (4x the source) and a 0.046 m overshoot past the end; now identical
  to the source on all 48 frames (steady -0.0255 m/frame). Rotations
  in the golden byte-identical before/after; keys sidecar unchanged.

## Physics (fixture golden)

- `motionforge physics-check --input jump.json --root DEF-spine --feet
  ...`: 4 contact / 8 airborne, 0 balance violations (mean -0.2038 m),
  1 ballistic phase residual 0.0000 m, momentum max 900.00 m/s2 flags
  [4,5,6] + 180.0 deg turn at 5.
- `physics-fix`: max accel 900.00 -> 141.31 m/s2, 7 frames smoothed.

## AutoPose

- Inference: whole `motionforge autopose` command 1.7 ms (release,
  14 bones, `--time` on stderr) — gate is 10 ms per pose.
- Python forward pass agrees with the Rust golden to last ULP
  (component diffs < 1e-12; `test_python_matches_rust_golden`), for
  both input encodings (`autopose.0.json` absolute,
  `autopose_rootrel.0.json` root-relative).
- Root-relative inputs: identical prediction with the character moved
  7 m (`autopose::tests::root_relative_inputs_ignore_where_the_character_stands`),
  and identical model input for a walk frame shifted by (3, -2, 0.5) m
  to 1e-12 (`test_root_relative_input_ignores_where_the_character_stands`).
- Training smoke (walk clip as train+heldout, MLP 64x64, 30 epochs,
  CPU, seed 1): loss 0.589 -> 0.022, heldout angle 110 -> 21 deg;
  GATE FAIL on the toy as expected; trained weights load into the CLI.

## Blender suites

- Pure: 7 tests (`python3 -m unittest discover -s blender/tests -p
  "test_pure.py"`).
- Headless: 30/30 checks, exit 0 (`Blender --background
  --factory-startup --python blender/tests/test_headless.py`), needs a
  built CLI. Gate note: Blender 5.2 exits 0 on uncaught non-SystemExit
  errors (verified), so the script converts escapees to exit 1.
- Batch P3: synthetic BVH -> clip -> `clip-info` + manifest gate, all
  green (`python3 -m unittest blender.tests.test_batch`).

## Python suites

- Manifest gate 9 tests, autopose/training-path 20 tests (torch-free;
  `test_missing_torch_errors_cleanly` runs a real 1-epoch training
  when torch is installed).
- Torch modules compile: `python3 -m py_compile
  python/autopose/model.py python/autopose/train.py`.

## Creature motion (2026-10-29, f: RTX 2060 SUPER 8 GB, Blender 5.2.1, torch 2.7.1+cu128)

Spec: `docs/creature.md`. Stand-ins: `gen 3d text` (SDXL reference ->
TripoSG, MIT, $0): a low-poly wolf (40k faces, 3 min) and a rock golem
biped (2 min). Meshes, rigs, renders and weights stay local.

- Rig tries, `skintokens rig` on CPU (no CUDA build): 365-640 s per
  try. `motionforge creature-profile` checks (`tools/creature_rig.py`):
  wolf seeds 1/2/3 -> 32/41/29 bones, 4 of 4 legs grounded, mirror
  16/20, 24/31, 17/18, scores 1.400 / 1.370 / 1.472 (seed 3 kept);
  golem seeds 1/2/3 -> 46/40/44 joints with 24/18/23 finger joints
  folded into the hands -> 22/22/21 bones, 2 of 2 legs grounded once
  the floating debris under the feet (<2% of the vertices) is ignored,
  mirror 16/16, all 1.500 (seed 1 kept). Facing: wolf -X from the ears
  sitting 0.57 m forward of the centre; golem head over the centre ->
  glTF +Z.
- Rust vs Blender: `render.py` FK self-check, max head mismatch 4.6e-7 m;
  `creature-project` vs Blender's `world_to_camera_view` on 6 samples x
  2 views x 29 joints: max 0.0001 px.
- Training renders: `creature-poses --count 3000 --seed 1`, `render.py
  --mode poses --samples 16`, Eevee: 0.34 s per sample (two 512x512
  views), 17 min per creature; 3 frames keyed per sample (keying the
  whole 9,000-frame clip was quadratic in Blender).
- Joint finder, `train.py --epochs 12 --batch 16` on 6,000 view images
  (10% held out), dinov2-small + conv decoder, fp16: wolf 1,517 s, peak
  3,996 MB VRAM, held-out 6.35 px mean / 4.14 px median / 84.9% within
  10 px (degraded 7.41 px); golem 1,393 s, 3,964 MB, 4.59 px mean /
  2.92 px median / 92.4% within 10 px (degraded 5.19 px). Pixels of the
  512 px view (wolf 1.63 m tall over ~290 px: 1 px = 5.6 mm).
- The exam, `python/creature/exam.py --seeds 1,2,3 --frames 48 --video`
  (both creatures, 2 min 47 s): wolf mean joint error 24.9 / 20.6 /
  21.5 mm (mean 22.3 mm = 1.4% of height), inner-bone rotation 8.2 /
  8.2 / 9.0 deg, max 95 / 146 / 75 mm (front paws and lower front legs,
  the near/far leg pair), foot slide 18-21 mm per still frame (0-1
  plants found: the exam motion keeps the legs swaying); golem 16.3 /
  15.0 / 15.5 mm (mean 15.6 mm = 1.1% of height), 7.9 / 8.2 / 9.4 deg,
  max 138 / 252 / 194 mm (one forearm/hand, bones 8-9), foot slide
  10-16 mm, 1-2 plants. Views agree on every clip (median ray gap
  0.83-0.93% wolf, 0.52-0.57% golem; motion ratio under 0.5); fit
  residual 15-18 mm rms; mean detection confidence 0.93. Baseline:
  `tests/fixtures/creature/exam_baseline.json` (`--check` fails a
  creature whose mean_mm or mean_deg_inner grows >5%, floor 2 mm /
  0.5 deg). Side-by-side videos (film over motionforge, front | side)
  were made for every seed.
- Deterministic half in `cargo test` (fixture biped, 27 bones):
  perfect detections -> mean head error < 1 mm, residual < 1 mm; 2 px
  noise + 5% dropped detections -> mean < 40 mm and byte-identical on
  a second run; front view of one motion + side view of another ->
  rejected (`motion_ratio` ~4 vs limit 0.5), `--force` solves anyway.
  CLI goldens: `creature_profile`, `creature_poses`, `creature_motion`,
  `creature_project`, `creature_solve` (+ GLB), `creature_compare`.
- Blender: `test_headless.py` "creature_solve finishes", the animated
  rig GLB imports with its action (3 objects), missing inputs cancel.
- Video step (free, Wan 2.2 TI2V-5B, Apache-2.0): see the line below.

## Pending owner gates (not runnable here)

- P1: a Mixamo clip plays on the biped_hero rig in the Bevy game with no
  foot sliding (needs owner Mixamo account + a Bevy scene check).
- P2: owner prefers the stylized version of 3 test clips.
- P4: owner keyframes one attack faster with AutoPose; 10 ms holds on
  the full hero rig (measured 1.7 ms on 14 bones; full rig is wider).
- CMU download + full training run (needs the dataset + GPU time).
