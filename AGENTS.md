# motionforge — agent notes

Game-animation sidekick for HLL: retarget assist, motion stylizer,
physics pass, and ML AutoPose. Same split as `~/SWE/retopoforge`:
an MIT-licensed Rust core + CLI, driven as a subprocess over files by
a GPL Blender extension. Training lives in `python/` (PyTorch, local).

Scope and gates: read `PLAN.md` first; open follow-ups are in
`TODO.md`. Format + feature docs live in
`docs/`. Every claim in docs carries a measured number and the command
behind it (`docs/measurements.md`).

## Standing rules

- The owner's game assets, rigs, and any mocap (CMU, Mixamo,
  recordings) must never be committed — not even file names or paths
  in tracked files. Datasets enter this repo as manifests only
  (`dataset manifest` = clip id + source + license + checksum).
  `data/`, `models/`, and `bench/` are gitignored.
- Licensing is the hard constraint (see `docs/licensing.md`):
  commercial-safe sources only. CMU mocap is OK (terms verified
  2026-10-01, recorded in `docs/licensing.md`); owner's own clips OK.
  BLOCKED: AMASS / SMPL-based data (non-commercial), anything
  research-only, Hunyuan output (territory exclusion). Mixamo clips
  may be retargeted for in-game use but are excluded from training
  data by default — ask the owner before loosening any gate.
- Determinism: the Rust core is byte-deterministic (same input bytes
  -> same output bytes). No hash-map iteration in output paths, no
  timestamps or temp paths in outputs, no unseeded RNG, and no
  platform libm: call `motion_core::detmath::{sin, cos, sin_cos, asin,
  acos, exp}` instead of the `f64` methods (Apple's and glibc's libm
  differ in the last ulp; a unit test rejects `.sin()` & co. in core
  code). `sqrt`, `floor`, `powi`, `to_degrees` are exact and fine. CLI
  contract goldens pin this; regenerate deliberately and review the
  diff.
- Never launch GUI binaries without explicit user approval. Headless
  Blender (`--background --factory-startup`) is fine.
- Blender binary: `/Applications/Blender.app/Contents/MacOS/Blender`.

## Build

- `cd rust && cargo build --locked --release` produces
  `rust/target/release/motionforge`. Zero external crates (offline
  build); keep it that way unless the owner approves a dependency.
- Python training needs torch (not vendored):
  `pip install -r python/requirements.txt`. Manifest tooling is
  stdlib-only.

## Checks

- `cd rust && cargo fmt --all --check && cargo test --locked --release`
  must be green with zero warnings. CI (`.github/workflows/ci.yml`)
  runs it on Linux x86_64 and macOS arm64, so goldens must be
  byte-identical on both.
- CLI contract goldens (`rust/cli/tests/` + `tests/fixtures/`): after
  an intended CLI output change, regenerate with
  `UPDATE_GOLDENS=1 cargo test --locked --release -p motionforge`
  and review the diff before committing.
- Blender smoke (headless, procedural armature, no rigforge needed):
  `blender --background --factory-startup --python blender/tests/test_headless.py`.
  CI runs it (and `blender.tests.test_batch`, which takes `$BLENDER`)
  on Linux with Blender 5.2.1.
  With rigforge + a real clip, the live checks are in
  `docs/measurements.md`.
- Manifest gate: `python3 tools/manifest.py --check data/manifest.json`
  rejects blocked licenses; every training run records its manifest
  checksum in the run log.

## Layout

- `rust/core/` engine (`motion_core` lib), `rust/cli/` the
  `motionforge` binary + CLI contract test.
- `blender/motionforge/` = Blender extension (GPL, talks to the CLI
  as a subprocess over clip JSON). `blender/tests/` = headless smoke.
- `python/autopose/` = training (torch) + weights export to the
  Rust-readable format. `tools/` = stdlib-only helpers (fixtures,
  manifest checks).
- `tests/fixtures/` = procedural synthetic clips/skeletons (generated,
  committed; these are test vectors, not mocap).
- `docs/` = format + feature docs. `data/`, `models/` = gitignored
  local-only inputs/outputs (never committed).

## Siblings (separate repos, separate sessions)

- `~/SWE/rigforge` — the rig (Rigify fork, `hll_hero`/`hll_stalker`
  presets). Its `tools/retarget_mixamo.py` is the reference P1
  implementation; this repo's Rust retarget ports its rotation-transfer
  math onto clip JSON. Never commit game rigs here.
- `~/SWE/retopoforge` — mesh stage; the structural template
  (workspace layout, subprocess-over-files split, golden tests).
- `~/SWE/games/hll` — the game (Bevy, Rust; the Godot project was
  removed 2026-10-05). Retargeted clips land there
  via rigforge's deform-only GLB export; `tools/validate_assets.py`
  is the downstream gate.
