# motionforge

Game-animation sidekick for HLL (Godot 4.7 action-adventure): retarget
assist, motion stylizer, physics pass, and ML AutoPose. One person,
one panel, no mocap cleanup marathons. Plan and gates: `PLAN.md`;
measured numbers: `docs/measurements.md`.

## Layout

- `rust/` — MIT core (`motion_core`) + `motionforge` CLI. Zero
  dependencies, offline build, byte-deterministic. All math lives here.
- `blender/motionforge/` — GPL Blender extension (panel + operators).
  Talks to the CLI only as a subprocess over clip JSON files, same
  split as retopoforge. `blender/tools/batch_retarget.py` is the P3
  BVH batch driver.
- `python/autopose/` — PyTorch training for AutoPose (local, M4).
  Dataset loading, weights export, and evaluation are stdlib-only;
  only the training loop needs torch.
- `tools/` — stdlib helpers: fixture generator, dataset manifest gate.
- `tests/fixtures/` — procedural synthetic clips/skeletons (test
  vectors, not mocap) + CLI goldens.

## Build

```bash
cd rust && cargo build --locked --release   # -> rust/target/release/motionforge
pip install -r python/requirements.txt      # training only (torch)
```

Install the Blender extension from `blender/motionforge`
(`blender_manifest.toml`), then point it at the CLI in its preferences
(it also searches PATH and the build tree).

## CLI usage

```bash
motionforge --help
motionforge retarget --source SRC.json --target hero_skel.json \\
    --map blender/motionforge/presets/mixamo_hllhero.json --output OUT.json
motionforge stylize --input IN.json --output OUT.json --exaggeration 1.35
motionforge physics-check --input IN.json --root DEF-spine --feet DEF-foot.L,DEF-foot.R
motionforge physics-fix --input IN.json --output OUT.json --root DEF-spine --feet DEF-foot.L,DEF-foot.R
motionforge autopose --model weights.json --effectors eff.json --output POSE.json
motionforge clip-info --input IN.json
```

Reports go to stdout (deterministic, golden-pinned); files only to
`--output`; `--time` prints wall ms to stderr. Formats:
`docs/clip-format.md`.

## Checks

```bash
cd rust && cargo fmt --all --check && cargo test --locked --release
python3 -m unittest discover -s blender/tests -p "test_pure.py"
python3 -m unittest discover -s python/tests -p "test_*.py"
python3 -m unittest discover -s tools/tests -p "test_*.py"
/Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup \\
    --python blender/tests/test_headless.py
python3 -m unittest blender.tests.test_batch
python3 tools/manifest.py --check data/manifest.json   # training gate
```

Regenerate CLI goldens deliberately after intended output changes:
`UPDATE_GOLDENS=1 cargo test --locked --release -p motionforge --test
cli_contract`, then review the diff.

## Rules

- No game assets, rigs, or mocap committed — datasets enter as
  manifests only (`data/`, `models/` are gitignored).
- Licensing is the hard constraint: commercial-safe training sources
  only (`docs/licensing.md`); training refuses rejected manifests.
- Ask the owner before loosening any gate (`PLAN.md`, `docs/autopose.md`).
