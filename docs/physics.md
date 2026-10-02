# Physics pass (P2.5)

Keyed motion in, physically believable motion out. No ML, no IK: three
checks over a clip, with one-click root-curve fixes where a root-only
fix is well-defined. `physics-check` reports only; `physics-fix`
writes a corrected clip. The report numbers are the overlay data (the
Blender side shows them in the panel; 3D overlay markers are follow-up).

## Checks

- **Balance**: horizontal center of mass against the support feet on
  contact frames (foot within `--contact-margin`, default 0.03 m, of
  the clip minimum). Support center = mean of contacting feet,
  radius = spread + `--foot-radius` (default 0.12 m); frames past
  `--balance-margin` (default 0.05 m) are violations. COM is a
  segment-weighted joint mean (Dempster weights by bone-name keyword,
  normalized; unmatched bones share a small remainder) — approximate
  by design. Check-only in v1: a rigid root shift moves the feet with
  the body, so it cannot change COM-support geometry without foot
  pinning (IK, follow-up). Each violation carries a suggested nudge
  vector for the animator.
- **Ballistic**: airborne runs of `--min-air-frames` (default 4)+ with
  no contacts get a least-squares parabola fit of root height;
  residual is reported per phase. Fix replaces root height with the
  fit, crossfaded over `--blend-frames` (default 2) at the boundaries
  (boundary frames stay exact).
- **Momentum**: root horizontal acceleration over `--accel-limit`
  (default 25 m/s2) and heading changes over `--turn-deg` (default
  40 deg/frame above `--min-turn-speed`, default 0.5 m/s) are flagged.
  Fix selectively smooths root x/y (Gaussian `--smooth-sigma`,
  default 1 frame) around flagged frames +- `--smooth-pad` (default
  2), crossfaded at region edges.

World-to-loc conversion uses the root's rest armature orientation
(exact when the root's ancestors are unposed, true of v1 clips).

## Numbers (fixture golden)

Jump clip: 4 contact / 8 airborne frames, 0 balance violations (mean
excursion -0.2038 m), 1 ballistic phase (frames 2..9, residual 0.0000
m), momentum max 900.00 m/s2 with flags [4, 5, 6] and a 180.0 deg turn
at 5 (the planted teleport). After fix: max accel 141.31 m/s2, 7
frames smoothed, 1 phase processed. Unit tests pin each check and fix
(balance flag + nudge direction, parabola residual collapse with exact
boundaries, teleport smoothing, 90 deg turn flag).

## Limits (v1)

- Balance has no auto-fix (needs IK); the nudge vector is the output.
- COM weights are name-heuristic approximations, not measured masses.
- No 3D overlay yet; numbers live in the report/panel.

## Commands

```bash
./rust/target/release/motionforge physics-check --input IN.json \\
    --root DEF-spine --feet DEF-foot.L,DEF-foot.R
./rust/target/release/motionforge physics-fix --input IN.json --output OUT.json \\
    --root DEF-spine --feet DEF-foot.L,DEF-foot.R
```
