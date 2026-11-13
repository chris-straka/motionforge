# Clean: fix a rigged GLB's own clips

`motionforge clean --input hero.glb --output out.glb [--clip NAME]
[--suffix _clean] [--no-ground] [--no-lock] [--contact] [--report r.json]`
(`rust/core/src/clean.rs`), or `animate ... --clean` after a retarget.
It fixes, on every clip, the defects `motion check` (`docs/motion-check.md`)
measures where the fix is not a judgment call. Thresholds are motion
check's, scaled by the character's height.

1. **Ground.** A clip whose upright frames (hips at 60%+ of their
   standing height above the feet) nearly all (90%) float, or nearly all
   sink, by a steady amount (the median grounded height within 4 cm of
   the 10th percentile) carries a height offset: the whole clip moves by
   that 10th percentile. Needs a quarter of the clip upright. A jump
   keeps its air (most frames are on the floor); a death is not lifted by
   its lying frames (the feet are no floor reference there).
2. **No foot under the floor.** When both feet are under the floor the
   hips come up by the smaller sink; a foot still under is lifted by leg
   IK with its world rotation kept. Lifts ease in and out over 3 frames.
3. **Plants locked.** Plants that slide more than 2.5 cm: the point that
   is down (heel side or ball) stays where it went down; when the other
   point takes over it starts from where the lock had it. Leg IK, eased
   over 3 frames. Not on in-place cycles (their stance speed is the
   game's to match), not on a plant that would move the foot more than
   35 cm (a fall or a drag).
4. **Arms out of the body** (`--contact`, off by default): the contact
   pass `animate` runs (`docs/contact.md`). Off because on the game's
   clips it jitters and turns wrists past their limits (`docs/log.md`).

Twist helpers are re-driven from their drivers. Clips are resampled at
30 fps (as `motion check` and the stylizer sample them).

## Tests with known answers (`core/tests/clean.rs`)

- hips 15 cm up on every frame: the clip comes down exactly 15 cm, no
  frame floats or sinks after;
- hips 4 cm down on every frame: up exactly 4 cm, nothing in the floor;
- a 6 cm dip into the floor on 4 frames of 20: no whole-clip shift, the
  hips come up 6 cm there, frames 4 or more away are untouched;
- a jump (hips 20 cm up on 6 of 20 frames): no shift, still 6 frames
  in the air;
- a foot slid 8 cm through a plant: locked, the plant's slide after is 0.

## The game's clips (2026-11-13)

| measure (sum over the 31 clips) | base | clean |
|---|---|---|
| skate cm | 344.5 | 188.4 |
| sink cm (deepest per clip) | 710.6 | 0.0 |
| floating frames | 826 | 93 |
| back-and-forth mm | 37.9 | 39.7 |
| joint-limit frames | 243 | 245 |

The seven `sf_*` clips come up 98-103 cm (they stood in the floor to the
waist), the enemy and Quirin takes down 7-25 cm (they hovered). 26 s for
the 31 clips on the Mac mini (z), most of it sampling. Iterations and
what was dropped: `docs/log.md`.
