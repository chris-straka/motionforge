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

## 2026-11-13, round 2: the contact pass

Goal: smooth enough to be on by default in `clean`. Judged on the 31
clips, `clean --contact` against `clean` without it (base re-scored with
the round-2 treadmill: skate 355.8).

| measure (sum) | clean | contact, round 1 | + anatomical hinge IK | + rate limit | + free hand | + 0.3 s ramp | + shallow free-arm floor |
|---|---|---|---|---|---|---|---|
| through-body frames | 895 | 184 | 184 | 231 | 140 | 114 | 229 |
| through-body cm (deepest) | 238.0 | 81.1 | 81.1 | 99.3 | 91.6 | 65.9 | 80.1 |
| back-and-forth mm | 39.9 | 94.2 | 94.2 | 62.1 | 62.3 | 55.4 | 55.5 |
| joint-limit frames | 245 | 521 | 521 | 509 | 301 | 291 | 291 |

### Kept

- **Anatomical hinge in the IK** (`contact::hinge_axis`: upper bone x
  facing, carried in the upper bone's frame). The round-1 hinge came from
  the limb's own bend, which is noise on a nearly straight limb: the
  elbow could flip between frames (the death's left arm spiked 265 mm).
  A first try took the axis from the rest pose's bend: Andras's knees
  rest bent 3 deg partly sideways, and joint-limit frames rose 245 -> 303
  without contact; the anatomical axis brings them back to 245.
- **Rate limit on the contact corrections** (0.9 character heights/s,
  blade turn 3 rad/s): where the surface to clear switches, the chosen
  offset jumped 40 cm in two frames (the death lying down: pushed out one
  side, then the other). Death's back-and-forth 9.1 -> 1.0 mm.
- **A free hand turns with its forearm** (the pass kept every hand's
  world rotation while moving the wrist up to 24 cm): limit frames
  509 -> 301, and fewer through-body frames.
- **0.3 s ease for `clean --contact`** (0.15 s stays `animate`'s): a 20 cm
  correction eased over 0.15 s flips its acceleration within a frame or
  two. 0.5 s traded frames for depth without a clear win.
- **Free arms resting shallower than 2% of height are left alone**
  (`clean --contact`): pushing a hand off the coat splayed the arm like a
  puppet (picture, attack_2 and hit). The sword arm is always cleared.
- `motion check`'s treadmill follows the lowest point across each frame
  (a heel-to-ball switch read as a foot-length jump).

### Dropped

- Smoothing the corrections' direction (binomial, size kept): no change
  on any measure.
- Smoothing the corrections themselves (binomial): shaved the peak off a
  one-frame contact; the hand stayed in the chest (`contact` test).

### Verdict

Not on by default yet: through-body frames 895 -> 229 (deepest sum
238 -> 80 cm), but back-and-forth +15.6 mm and joint-limit frames +46
over the set, and in `hit` the sword hand is held off the body where the
move swings it across. Next: let a correction move the elbow (not only
the wrist) for forearm-through-chest, and judge `hit` with the owner.
