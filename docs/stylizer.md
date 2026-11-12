# Motion stylizer (P2)

Realistic clip in, snappy game motion out. Pure curve math, no
training. Two ways in:

- **On a rigged GLB** (the path the game's clips take, after
  `animate`): `motionforge stylize --input hero.glb --output out.glb
  [--clip attack_1] [--suffix _new]`. Default `--mode body` (below);
  `--mode per-bone` runs the original algorithm unchanged.
- **On clip JSON** (the Blender operator): exports the active action,
  runs `motionforge stylize`, imports `<action>_stylized`. Per-bone only
  (`--mode body` needs the rigged GLB: it reads the skeleton's world
  positions and re-plants the feet).

## Body mode (default on GLBs, 2026-11-12)

Per-bone keys put every bone on its own clock: the arm reaches its
extreme on frame 9, the spine on 11, the hips on 7, so the body never
holds a pose as a whole and the easings overlap: attacks read rubbery.
Body mode (`rust/core/src/stylize_body.rs`):

1. **Body speed**: per frame, the summed speed of the hands, forearms,
   feet, shins, chest and head relative to the hips, plus the hips' own
   speed; smoothed [1,2,1]/4.
2. **Shared keys**: the ends, plus every speed minimum (the body at an
   extreme or turning around) whose dip is at least `--key-prominence`
   (0.25) of the lower of the speeds on its two sides, in motion faster
   than 5% of the clip's top speed, `--min-spacing` apart, each moved to
   the slowest unsmoothed frame next to it.
3. **One clock**: each segment between keys keeps its frame count. A
   big segment (top speed at least half the clip's) eases the whole pose
   `--anticipation` of the way back against the coming move over
   `--anticipation-frames`, holds `--hold` frames, then plays the source
   motion sped up evenly into the rest, unwinding the counter pose
   (only when anticipation and hold take at most half the segment;
   otherwise it holds a quarter of it at most and plays the rest). The
   next segment starts with a moving hold: the pose carries on past the
   key along the strike's rotation (up to `0.1 * --overshoot` of it) and
   settles over `--hold + 2` frames. Other segments hold, then ease out.
   The source is resampled at the warped time (arcs and secondary motion
   survive); every bone uses the same warp, so holds and overshoots land
   on shared frames. (A first version eased the strike with
   ease-out-back like the per-bone mode; on the real attacks that warp
   jumped over the arm's widest point in the first strike frame and the
   swing's arc came out narrower than the source's.)
4. **Exaggeration from the clip's own neutral**: each bone's rotation is
   pushed away from its mean over the clip (not the bind pose),
   `neutral * (neutral^-1 q)^factor`, ramped to none over the first and
   last 4 frames so the clip still starts and ends on its own poses.
5. **Feet re-planted**: plants found on the original clip (ankle near its
   lowest and nearly still, as `animate`'s foot pinning finds them)
   follow the warp; inside one, the ankle goes back to the original's
   position at that source time and the foot to its rotation, by leg IK,
   eased over 3 frames (`--no-replant` to skip).

Translations follow the warp (no exaggeration or overshoot), so root
travel ends where it ended. Body mode is for attacks and other one-off
moves: on an in-place walk whose stance foot slides (the procedural
fixture walk), there are no plants to keep, and exaggerating the legs
lifts both feet off the floor on 21 of 31 frames, up to 47 cm
(`core/tests/motion_check.rs`). Unit tests (`stylize_body::tests`): keys land
on a synthetic attack's windup and strike extremes (within the one frame
30 fps sampling allows); identity settings reproduce the source on keys;
two bones with different extremes freeze on the same frames; the strike
gets a deeper windup and an overshoot that settles exactly on its key;
exaggeration widens a wiggle about its own middle instead of doubling its
offset from rest. `core/tests/motion_check.rs` runs both modes on a GLB.

## Real test: the hero's three attacks (2026-11-12)

The game's own sword attacks on the hero's rig (hand-keyed at 24 fps,
0.67-0.83 s), each stylized both ways with default settings, then scored
with `motion check` (`docs/motion-check.md`) and rendered as one
side-by-side reel and an 8-frame strip per attack (kept with the game's
files, not in this repo). Commands: `stylize --mode per-bone --suffix
_old`, then `stylize --mode body --suffix _new`, on f.

| clip | skate frames / cm | through body frames / deepest | arm jerk mean | windup / strike / hold |
|---|---|---|---|---|
| attack_1 original | 86% / 13.6 | 21 / 4.4 cm | 3,650 | 0.13 s / 0.10 s / 3 fr |
| attack_1 old | 48% / 8.9 | 21 / 9.1 cm | 10,980 | 0.03 / 0.07 / 0 |
| attack_1 new | 43% / 8.6 | 21 / 8.6 cm | 10,890 | 0.03 / 0.03 / 1 |
| attack_2 original | 86% / 13.6 | 21 / 3.3 cm | 3,150 | 0.13 / 0.13 / 2 |
| attack_2 old | 48% / 8.9 | 21 / 8.8 cm | 14,350 | 0.00 / 0.03 / 2 |
| attack_2 new | 43% / 9.0 | 21 / 13.1 cm | 10,050 | 0.07 / 0.07 / 4 |
| attack_3 original | 88% / 27.4 | 4 / 2.2 cm | 2,070 | 0.13 / 0.10 / 14 |
| attack_3 old | 27% / 6.0 | 4 / 4.1 cm | 7,560 | 0.00 / 0.03 / 0 |
| attack_3 new | 42% / 16.4 | 9 / 2.4 cm | 4,970 | 0.23 / 0.10 / 4 |

By eye (the strips and reels):

- **Old (per-bone)** breaks every attack: exaggerating from the bind pose
  lifts the sword arm to the head in attack_1 and wraps it round the head
  in attack_2; attack_3 never raises its arms overhead and kicks a leg
  out. Its lower skate is the legs being pushed elsewhere, not better
  feet.
- **New (body)** keeps each attack's shape and arc. Its one clear gain
  is attack_3: it holds the overhead windup a few frames longer before
  the chop (a held load, the "weight" the original lacks). In attack_1
  and attack_2 it mostly shifts timing and sharpens the corners of the
  hand's path; exaggeration pushes the sword forearm deeper through the
  chest (attack_2: 13.1 cm against 3.3).
- **The original still reads best** on attack_1 and attack_2: they are
  already fast, keyed snaps (hand 12-13 m/s), and both stylizers mostly
  add roughness. New beats old on all three.

Where the numbers and the eye disagree: jerk calls new as rough as old,
but by eye new is far cleaner (jerk counts the deliberate snap of a
faster strike as roughness); skate says both stylized versions beat the
original, which at strip scale I cannot see; the attack timing numbers
are noisy on 21-frame clips (one frame is 0.03 s), so "windup 0.03 s"
means "no windup the score could find", not a measured improvement.

Next, if the stylizer is to win on these clips: exaggerate arms against
the contact proxies (stop at the body instead of through it), scale
exaggeration by how fast a clip already is, and pick keys from the weapon
hand's extremes as well as the whole body's (the body's speed minima
missed the arm's widest point in attack_1).

## Per-bone mode (the original; `--mode per-bone`, clip JSON)

Steps 3-5 act on rotations only. Locations carry root travel and pass
through unchanged, frame for frame (see "Root travel" below).

1. **Find extremes**: local maxima of the joint-angle-from-rest signal
   above `--angle-threshold` (default 0.15 rad), strongest-first with
   `--min-spacing` (default 4) frames apart, plus endpoints above
   threshold. Raw signal, no smoothing: ripples lose to real extremes
   in the greedy pass.
2. **Key reduction**: dense frames -> the kept key set (union reported).
3. **Exaggeration**: push each extreme rotation away from neutral,
   `slerp(identity, q, factor)` with `--exaggeration` (default 1.35)
   or the first matching `--chain substr:factor` rule.
4. **Anticipation**: before each extreme with room, insert a
   counter-key `--anticipation-frames` (default 3) earlier at
   `slerp(prev, extreme, ---anticipation)` (default 0.25) from the
   exaggerated poses. Skips without room are counted in the report.
5. **Retime**: resample every frame between keys with `--hold` frozen
   frames (default 2) at each segment start, then ease-out cubic, or
   ease-out-back scaled by `--overshoot` (default 0.6) for settle
   wobble. Slerp/lerp extrapolation carries overshoot past the keys.

## Numbers (fixture golden)

48-frame walk -> 14 keys `[0, 3, 9, 12, 14, 17, 24, 27, 33, 36, 38,
41, 44, 47]`, 13 anticipation counters added, 1 skipped. Unit tests
pin the math: 2x exaggeration doubles a 0.5 rad extreme to 1.0;
ease-out-back midpoint hits 0.1088 on a 0.1 target (overshoot) vs
0.0875 without; holds freeze exactly; 2-frame clips round-trip exactly.

## Root travel

Until 2026-10-03 locations went through anticipation and the retime
too. A walk's root has no rotation extremes, so its only keys were the
endpoints and its whole travel became one eased segment: on the
fixture walk the root froze 2 frames, sprinted ahead (frame 16 at
-0.99 m vs the source's -0.41 m), overshot the 1.2 m end (-1.246 m at
frame 32), then walked backwards for 15 of 47 frames, with the legs
(keyed separately) sliding under it. Now the stylized root travel
equals the source on every frame (steady -0.0255 m/frame); rotations
in the golden are byte-identical to before. Pinned by
`stylize::tests::locations_pass_through_unchanged`.

## Import: thinned (default) or dense

`stylize --keys-out` writes the per-bone key frames; the importer keys
only those frames (LINEAR interpolation, since baked samples would
overshoot under BEZIER). Thinning applies to rotations: a bone whose
location moves over the clip (the root) keeps a location key on every
frame so travel and bob survive exactly. Thinned keys capture the extreme poses for
hand-tweaking; in-betweens approximate the baked easing. Unticking
Thin Keys imports dense every-frame keys (exact). The headless test
pins both paths.

## Limits (v1)

- The owner-preference gate ("stylized beats original on 3 test
  clips") needs the owner; defaults are starting points, not verdicts.

## Commands

```bash
./rust/target/release/motionforge stylize --input IN.json --output OUT.json \\
    --exaggeration 1.35 --chain arm:1.8 --hold 2 --overshoot 0.6
```
