# Motion stylizer (P2)

Realistic clip in, snappy game motion out. Pure curve math, no
training. The operator exports the active action, runs `motionforge
stylize`, and imports the result as `<action>_stylized` (dense editable
keys; the report lists the sparse key frames each bone kept).

## Pipeline (per bone, independently)

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
