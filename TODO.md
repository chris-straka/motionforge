# TODO — open follow-ups

Catch-up list as of 2026-10-03. Scope and gates stay in `PLAN.md`;
numbers and commands in `docs/measurements.md`.

## Done 2026-10-03 (PRs #1–#5, all merged, CI green)

- #1 `detmath`: own sin/cos/asin/acos/exp, so outputs are
  byte-identical on macOS and Linux (platform libm differed by 1 ulp).
- #2 Fuzz fixes (deep-JSON stack overflow, quadratic non-ASCII parse,
  smoothing-flag hangs); Blender "Fix Balance" / "Max Lean"; CI on
  Linux + macOS.
- #3 Clip loads reject absurd magnitudes (|x| > 1e6 m, fps > 1e4,
  overflowing quats); CI runs the headless Blender tests (5.2.1).
- #4 Stylizer passes root travel through unchanged (a stylized walk's
  root used to freeze, sprint, overshoot and walk backwards).
- #5 AutoPose inputs are root-relative (weights tag
  `effector-pos-mask+lengths/root-relative`); old weights still work.

## Owner (needs you)

- [ ] Run `python/autopose/train.py` once locally: torch is not in CI,
      so the #5 training path is only compile-checked.
- [ ] Retrain AutoPose to get root-relative weights (ideally after the
      heading item below, to retrain only once).
- [ ] Look at a stylized walk in Blender after #4 (P2 gate: owner
      prefers stylized on 3 test clips).
- [ ] P1 gate: a Mixamo clip on `hll_hero` in the Bevy game with no foot slide.

## Next code tasks (agent-ready)

- [ ] GLB adapters v2 (`docs/adapters.md` limits): foot pinning in
      `animate` when proportions differ; quadruped ROM poses and clips;
      CUBICSPLINE keys sampled as splines.

- [ ] **AutoPose heading invariance** (recommended next). Inputs are
      root-relative but not facing-relative: the same pose facing
      another way gives different inputs. Rotate effector offsets by
      the inverse of the root's yaw (about +Z) in
      `python/autopose/dataset.py::build_input` and
      `rust/core/src/autopose.rs::infer`; Blender `export_effectors`
      must also send the root's yaw/rotation. New weights `input` tag
      (keep both older tags working); open question: how the predicted
      root rotation is re-expressed. Mirror #5's tests (invariance,
      Python<->Rust cross-check golden, headless Blender check).
- [ ] **Retarget with unmapped intermediate bones** (design call).
      Transfer copies each bone's local motion (same as rigforge's
      reference), so an unmapped source bone's motion is dropped, not
      folded into the next mapped bone. Property check: full map
      matches world rotation deltas to 7e-16; skipping the middle bone
      of a 3-chain differs by up to 0.93. Only matters for targets
      with fewer bones than the source.
- [ ] Stylizer: bones whose angle never exceeds `--angle-threshold`
      keep only endpoint keys, so subtle motion (hip sway) is replaced
      by one eased transition. By design; revisit if it reads flat.
- [ ] Stylizer: exaggerating an extreme near 180 deg can wrap past pi
      (slerp extrapolation); rare for joints, unguarded.
- [ ] `tools/make_fixtures.py` uses Python's platform libm: regenerating
      on Linux shifts 14 numbers in `walk_src.json` by <=1.2e-16.
      Commit only intended files (note in its header).
- [ ] Windows (only if it becomes a target): extension's build-tree
      binary lookup lacks `.exe`; goldens need `.gitattributes`
      (`-text`) against CRLF conversion.
