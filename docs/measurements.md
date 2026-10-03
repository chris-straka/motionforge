# Measurements

Every numeric claim in these docs, with the command behind it. All run
2026-10-01/02 on Apple Silicon (M4, 16 GB), Blender 5.2.1 LTS, release
build unless noted. Fixture paths are `tests/fixtures/`; goldens pin
the CLI outputs byte-for-byte.

## Rust suite

- 62 unit + 9 contract tests green, zero warnings, fmt clean:
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
  (component diffs < 1e-12; `test_python_matches_rust_golden`).
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

- Manifest gate 9 tests, autopose/training-path 12 tests (torch-free;
  `test_missing_torch_errors_cleanly` runs a real 1-epoch training
  when torch is installed).
- Torch modules compile: `python3 -m py_compile
  python/autopose/model.py python/autopose/train.py`.

## Pending owner gates (not runnable here)

- P1: a Mixamo clip plays on the hll_hero rig in Godot with no foot
  sliding (needs owner Mixamo account + Godot import).
- P2: owner prefers the stylized version of 3 test clips.
- P4: owner keyframes one attack faster with AutoPose; 10 ms holds on
  the full hero rig (measured 1.7 ms on 14 bones; full rig is wider).
- CMU download + full training run (needs the dataset + GPU time).
