# Motion stylizer (P2)

Realistic clip in, snappy game motion out. Pure curve math, no
training. The operator exports the active action, runs `motionforge
stylize`, and imports the result as `<action>_stylized` (dense editable
keys; the report lists the sparse key frames each bone kept).

## Pipeline (per bone, independently)

1. **Find extremes**: local maxima of the joint-angle-from-rest signal
   above `--angle-threshold` (default 0.15 rad), strongest-first with
   `--min-spacing` (default 4) frames apart, plus endpoints above
   threshold. Raw signal, no smoothing: ripples lose to real extremes
   in the greedy pass.
2. **Key reduction**: dense frames -> the kept key set (union reported).
3. **Exaggeration**: push each extreme rotation away from neutral,
   `slerp(identity, q, factor)` with `--exaggeration` (default 1.35)
   or the first matching `--chain substr:factor` rule. Locations are
   never exaggerated (they carry root travel).
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

## Limits (v1)

- Sparse-key import is follow-up: the importer writes dense keys
  (exact), and the animator thins to the reported key frames by hand.
- The owner-preference gate ("stylized beats original on 3 test
  clips") needs the owner; defaults are starting points, not verdicts.

## Commands

```bash
./rust/target/release/motionforge stylize --input IN.json --output OUT.json \\
    --exaggeration 1.35 --chain arm:1.8 --hold 2 --overshoot 0.6
```
