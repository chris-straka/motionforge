# Log: what was tried, kept and dropped

Each entry: the change, the measure it was judged on, the result on the
game's clip set, and whether it stayed. The set is the hero's rig with
the game's 31 clips (HLH `player/andras/models/glb/andras_rig.glb`,
copied outside this repo; never committed). Scores: `motion check
--json`, totals by `tools/bench/motion_compare.py`. Pictures:
`tools/bench/motion_sheet.py` (painted).

## 2026-11-13, round 1

### Kept

- **Hinge arm/leg IK** (`contact::solve_arm`, used by foot pinning, the
  stylizer's re-plant, the contact pass and `clean`). The old solver bent
  the limb in the plane through shoulder, old elbow and target, so a
  target off the limb's own plane bent the elbow or knee sideways. Now
  the bend changes about the joint's own axis and the whole limb swings
  from the shoulder. On the set with the contact pass: joint-limit
  frames 734 -> 544, deepest through-body sum 99 -> 70 cm. Test:
  `arm_ik_keeps_the_elbow_a_hinge` (hand pulled 5 cm out of plane,
  reaches the target exactly, no elbow past a limit).
- **`motionforge clean`** (ground, lifts, plant lock; `docs/clean.md`),
  also `animate --clean`. Final, on the 31 clips (`motion_compare.py`):

  | measure (sum over clips) | base | clean |
  |---|---|---|
  | skate cm | 344.5 | 188.4 |
  | sink cm (deepest per clip) | 710.6 | 0.0 |
  | floating frames | 826 | 93 |
  | through-body frames | 894 | 895 |
  | back-and-forth mm | 37.9 | 39.7 |
  | joint-limit frames | 243 | 245 |
  | key-pose contrast deg (mean) | 33.8 | 33.9 |

  What is left of skate is the two long deaths' slides (falls past the
  35 cm lock cap: enemy-death 71 cm, quirin-death 78 cm; they floated
  before, so no plants were scored) and sf_walk's 30 cm (an in-place
  walk, left to the game's speed). The floating frames left are jumps
  (leap 18, twirl 7, run 6) and the shadowflame draw's 17. Pictures: the
  death's boots no longer sink to the shins at frames 9-12; sf_idle stops
  standing in the floor to its chest; the enemy attack's feet meet their
  shadows. Cost: back-and-forth +1.8 mm over the set (leg chains; the
  lifts and locks), 2 more joint-limit frames.
  Iterations that got it there, each judged on the set:
  - v1, ground = 10th percentile of the lowest foot point over all
    frames, applied whenever 90% of frames float or sink: lifted the
    death clip 43 cm, so once he lies down he floats (picture). Fixed
    in v4/v5: only upright frames count, and only a steady offset (the
    grounded half of frames within 4 cm).
  - v1 lift: a foot under the floor lifted on exactly those frames:
    kinks (back-and-forth up on the legs). Spread over 3 frames each
    side, and when both feet are under the hips come up instead.
  - v1 lock: anchored each contact point where it first went down; on a
    flat foot the heel and ball tie, the contact flips every frame and
    each flip let one frame of slide through (an 8 cm slide kept
    2.5 cm). Now a newly-down point is anchored where the lock had it
    the frame before: the slide is locked exactly.
  - v1 locked in-place cycles to their median belt: the walk's skate got
    worse (5.3 -> 9.7 cm). Cycles are now left alone (their stance speed
    is the game's to match).
  - A 12 cm cap on how far a lock may move a foot (to spare falls) threw
    away most locks (sf_cough's feet drift 33 cm while standing): set
    skate 17 -> 185 cm. Cap raised to 35 cm: only the two deaths' long
    slides are left.
- **Treadmill detection** in `motion check` needs every foot to agree
  (a single foot skimming the floor next to a standing one was read as
  a belt) and the cycle to loop (each foot back under the hips; the
  Quirin lunge `quirin-attack_2` was read as a 1.9 m/s belt).

### Dropped (for now)

- **Contact pass in `clean` by default.** Through-body frames 889 -> 186
  on the set, but back-and-forth doubled (death 0.2 -> 9.1 mm, sf_cough
  2.5 -> 13.9) and wrists turned past their limits (frames 274 -> 544):
  it keeps each hand's world rotation while moving the wrist up to
  24 cm, and its per-frame offsets jitter. Kept behind `clean --contact`.
  Next: smooth its offsets over time and let a free hand turn with the
  forearm.
- **Roughness as `(|a| + |a'| - |a + a'|) / 2`**: nonzero on a fast even
  arc (0.12 mm at 7.5 rad/s, curvature). Replaced by
  `max(0, -a . a') / max(|a|, |a'|)`, exactly 0 on any arc or corner.
