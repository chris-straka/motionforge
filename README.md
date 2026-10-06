# motionforge

Game-animation sidekick for HLL (Bevy/Rust action-adventure): retarget
assist, motion stylizer, physics pass, ML AutoPose, and the GLB rig
adapters genforge's character chain runs (standardize, animate,
pose-test; `docs/adapters.md`). One person,
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
motionforge standardize --input rig.glb --output std.glb [--class humanoid]
motionforge animate --input std.glb --clips clips/ --output animated.glb \\
    [--pick "lib.glb:Sword_Regular_A=attack_1,..."] [--no-contact] [--weapon R|L|none]
motionforge grip --input lib-with-weapon-clips.glb --output lib.glb
motionforge pose-test --input std.glb --output poses.glb --sheet sheet.png [--clips clips/]
motionforge fixture-glb --output test.glb --naming mixamo|plain|def [--twisted] [--walk]
motionforge adapter standardize|animate|pose-test IN.glb OUT.glb RESULT.json [flags]
```

The GLB commands and the genforge `adapter` contract are in
`docs/adapters.md` (HLL humanoid skeleton, mapping rules, limits).
`animate` ends with the **contact pass** (`docs/contact.md`): collision
proxies fitted to the character's mesh, two-bone arm IK that eases the
hands, forearms and a held weapon out of the body per frame, verified on
the mesh. `standardize` adds `Socket_Hand_L/R` weapon sockets; `grip`
finds the grip a weapon clip set was animated for. Contact sheets and
videos of clips: `blender/tools/clip_sheet.py`. The
skeleton has twist/helper bones at the upper arms and thighs
(`DEF-upper_arm_twist.L` ...): standardize adds them, animate and
pose-test key them with half their driver's rotation, so the game needs
no constraint code.

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
