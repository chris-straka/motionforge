# Physics pass (P2.5)

Keyed motion in, physically believable motion out. No ML, one small
IK: three checks over a clip plus a mini-IK balance fix. `physics-check`
reports only; `physics-fix` writes a corrected clip. The report numbers
are the overlay data (the Blender side shows them in the panel and as
3D overlay markers).

## Checks

- **Balance**: horizontal center of mass against the support feet on
  contact frames (foot within `--contact-margin`, default 0.03 m, of
  the clip minimum). Support center = mean of contacting feet,
  radius = spread + `--foot-radius` (default 0.12 m); frames past
  `--balance-margin` (default 0.05 m) are violations. COM is a
  segment-weighted joint mean (Dempster weights by bone-name keyword,
  normalized; unmatched bones share a small remainder) — approximate
  by design. Each violation carries a suggested nudge vector for the
  animator. Fix (mini-IK, runs last): lean the whole body about the
  ground-level support pivot toward the PINNED pre-fix support center
  (pinning keeps the iterations from chasing a support set the lean
  itself moves), targeting the margin edge so the correction ramps to
  zero at violation boundaries, then counter-rotate the ankles so the
  feet stay flat. Total lean per frame is capped twice: `--max-lean-deg`
  (default 8) and an analytic support cap that keeps both feet within
  the contact margin (a bigger lean would float one foot while the sunk
  one drags the ground down — collapsing support). No contact lift: the
  lean leaves the feet symmetric about the ground (+/-dz), which is
  already optimal. Small violations (a few cm) clear fully with feet
  planted; big ones get an honest partial fix and the residual stays
  reported. Untouched frames stay bit-exact.
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
frames smoothed, 1 phase processed, 0 balance frames (none violated).
Unit tests pin each check and fix (balance flag + nudge direction,
small-violation clear with planted feet, honest partial + capped lean
on big violations, bit-exact clear frames, ankle preservation,
parabola residual collapse with exact boundaries, teleport smoothing,
90 deg turn flag).

## 3D overlay

Physics Overlay exports the action, snapshots the current scene frame
via `physics-frame`, and places two empties (reused across runs):
`MF_COM` (sphere at the center of mass) and `MF_SUPPORT` (circle at
the support center, scaled to the support radius; hidden when
airborne). Positions convert through the armature's world matrix, so a
moved armature still overlays correctly.

## Limits (v1)

- The balance fix leans the rigid body: residual foot penetration up
  to spread * sin(cap) is the documented cost (mm-scale for realistic
  fixes). True planted-feet correction needs leg IK (knee/ankle solve),
  which is v2.
- COM weights are name-heuristic approximations, not measured masses.

## Commands

```bash
./rust/target/release/motionforge physics-check --input IN.json \\
    --root DEF-spine --feet DEF-foot.L,DEF-foot.R
./rust/target/release/motionforge physics-fix --input IN.json --output OUT.json \\
    --root DEF-spine --feet DEF-foot.L,DEF-foot.R
```
