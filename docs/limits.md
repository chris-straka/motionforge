# Joint limits (`motionforge-limits`)

Per-bone rotation caps authored against the rig, consumed by the
AutoPose gate. rigforge owns the export (handoff below); motionforge
owns the format and both consumers.

## Format (v1)

```json
{
  "format": "motionforge-limits",
  "version": 1,
  "rig": "hll_hero",
  "bones": {
    "DEF-thigh.L": {"max_angle_deg": 120.0},
    "DEF-shin.L": {"max_angle_deg": 150.0}
  }
}
```

- `max_angle_deg`: max rotation of the bone's LOCAL pose quaternion
  from rest (identity), measured as `2*acos(|w|)`, in degrees. Range
  is (0, 180]. v1 is scalar-only on purpose: it matches the
  per-bone angle telemetry `evaluate()` already reports, and one
  number per bone is authorable by hand. Per-axis or swing/twist
  limits are a v2 format extension, not silent extra keys.
- Bones missing from the table are unconstrained.
- Table entries for bones the skeleton does not have are reported as
  `unmatched` (sorted) — a stale table can never silently pass.
- Consumers check `format` + `version` and reject anything else.

Example: `tests/fixtures/limits_hero.json` (14 hero bones).

## Consumers

- `motionforge autopose --limits L.json` reports violations in stdout
  (`limits: N violations, M unmatched` + per-bone lines) and still
  exits 0 — inference succeeded; the report is the product, same as
  `physics-check`. Bad limit files are input errors (exit 2).
- `train.py --limits L.json` enforces the gate: any heldout-prediction
  violation fails the run (`GATE FAIL`, exit 1). Counts print on
  stdout; `run.json` records `limit_violations`, `limits_file`, and
  `limits_unmatched`.
- `inference.evaluate(..., limits=...)` returns `limit_violations`
  `[(frame, bone, angle_deg, max_deg)]` and `limits_unmatched`.

The Rust core (`core/src/limits.rs`) and the Python consumer
(`python/autopose/limits.py`) are cross-checked: the same golden
prediction reports the same 8 violations in the same order on both
sides (`test_python_matches_rust_limit_report`).

## rigforge handoff

rigforge exports one table per preset (`hll_hero`, `hll_stalker`) from
the Blender rig:

1. Source of truth: a hand-authored per-bone max-angle table checked
   in next to the preset (anatomy + style judgment, not derivable from
   geometry). A starting point can be scraped from Limit Rotation
   constraints (max over axes of max(|min|, |max|)), but the checked-in
   table is what ships.
2. Bone names are the exported DEF- names (what lands in the GLB and
   the clip JSON), not the rig-layer names.
3. Validate before export: every entry in (0, 180], every DEF-bone
   either capped or deliberately absent.

motionforge needs nothing else: point `--limits` at the exported file.
