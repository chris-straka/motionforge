# keypose: sword attacks from drawn key poses

`motionforge keypose` turns a few drawn key poses of a swing (front and
side views, a few figures per sheet: load, strike, follow-through) into a
timed clip on a standard `DEF-` rig. The point is readability: the poses
are the drawings', the timing is an animator's (hold the load, snap the
strike, carry through, settle).

## Commands

MediaPipe is not vendored (Apache-2.0; local venv, model file next to it):

```sh
uv venv --python 3.12 ~/.cache/motionforge-keypose-venv
VIRTUAL_ENV=~/.cache/motionforge-keypose-venv uv pip install mediapipe   # 1.2.3 tested
curl -sSLo ~/.cache/motionforge-keypose-models/pose_landmarker_heavy.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task
# sha256 64437af838a65d18e5ba7a0d39b465540069bc8aae8308de3e318aad31fcbc7b (2026-11-13)
```

```sh
PY=~/.cache/motionforge-keypose-venv/bin/python
$PY python/keypose/landmarks.py --model <pose_landmarker_heavy.task> \
    --front <sheet_front.jpg> --side <sheet_side.jpg> --poses 3 \
    --name cut --out bench/keypose/cut.landmarks.json --debug bench/keypose/debug
rust/target/release/motionforge keypose --rig <rig.glb> \
    --landmarks bench/keypose/cut.landmarks.json --output bench/keypose/out.glb \
    --name kp_cut_right --key-clips --report bench/keypose/kp_cut_right.report.json
$PY python/keypose/overlay.py --report bench/keypose/kp_cut_right.report.json \
    --landmarks bench/keypose/cut.landmarks.json --front <front> --side <side> \
    --out bench/keypose/kp_cut_right_overlay.png
rust/target/release/motionforge motion check bench/keypose/out.glb --clip kp_cut_right
python3 tools/bench/motion_sheet.py reel bench/keypose/reel.mp4 \
    "keypose=bench/keypose/out.glb:kp_cut_right" "game=bench/keypose/out.glb:attack_1" \
    --prop <sword.glb>@Socket_Hand_R --trail DEF-hand.R
```

Name the clip with an attack word (`cut`, `attack`, `slash` ...) so
`motion check` scores its windup and strike.

## How

1. **Read** (`python/keypose/landmarks.py`). Pose Landmarker (heavy) on the
   sheet; when it does not find every figure at once, on equal panels.
   Per figure: 33 image landmarks, MediaPipe's 3D guess, and the blade:
   dark grey pixels, opened with a disc a little wider than a line-art
   outline (outlines go, the filled blade stays), the longest thin
   component whose near end is by the wrists. The floor is the drawn
   ground line. (`mp.Image.numpy_view()` rows did not line up with the
   file's; pictures are read with OpenCV.)
2. **Fuse** (`keypose/mod.rs`). A front view is exact across and up, a
   side view (from the character's left) exact front-back and up; each
   view's depth is MediaPipe's guess at weight 0.2. Per landmark a
   weighted least-squares point. The side's scale is fitted to the front's
   heights. A profile often names the far leg or arm as the near one: per
   chain the side's naming is kept or swapped, whichever agrees with the
   front. The blade's direction is where the two views' blade planes
   meet. `--side-turn auto` fits a turn between the views; it hit its 40
   degree bound and fitted worse than the drawings' own claim (front and
   strict profile: on the right cut's load, joints off by 5.1% / 3.7% of
   the figure's height turned, 4.7% / 2.6% not), so the default is 0.
3. **Fit** (`keypose/fit.rs`). Hips from the hip line (tilted by 40% of
   the torso's lean), the chest's lean first and its twist second, the
   spine bent until the hips-to-shoulders chord meets the drawn torso
   (a chain's chord lags its top bone), neck and head from ears and nose.
   Elbows and knees are hinges in their anatomical plane
   (`contact::hinge_axis`), bend clamped to -5..150 degrees; the drawn
   elbow and knee choose the swivel. Two hands on the grip: both fists
   where the drawing has them, relative to the shoulders and scaled by
   arm length, the sword socket's +Y along the drawn blade and its edge
   (+-X) toward the swing, the off fist in the sword socket's frame
   turned by whichever quarter turn the wrist reaches best. The forearm
   lends up to 80 degrees of roll before the wrist is capped at 82.
   Feet within 10 cm of the floor (heel or toe) are planted flat, pointing
   where the shin points; an ankle past 60 degrees rises onto its toes.
4. **Time** (`keypose/timing.rs`, 30 fps defaults). Idle -> load 9 frames
   ease in-out; load held 4 frames while it coils 6% back against the
   strike; strike 3 frames, accelerating then full speed through the
   contact (the smear); follow-through at the same speed (its frames =
   strike frames x the blade's travel ratio), overshoot 6% of its last
   tenth and settle over 3 frames, held 7; back to the idle in 10.
   Rotations per bone by squad, so the sword hand sweeps an arc through the
   strike instead of turning a corner. Anticipation: when the drawings
   show no windup (the blade tip does not move against the strike by 30%
   of it before the load), the coil is 15% instead (`--anticipation`).
   Feet (`--feet`): `hold` (default) stands every drawn key on the
   strike's feet, one step into the stance and one out; `order` keeps each
   key's own drawn feet but gives them the strike's leg order (a step
   wherever they differ); `drawn` takes them as read. A step: the later
   key's rear foot anchors, the other lifts and arcs 7 cm, down on the
   key (a step into a strike starts in the load's hold). The hips come
   down where a leg cannot reach its mark, spread over neighbouring frames.
5. **Pick the follow-through** (`--carry auto`): the clip is built with
   the drawn follow-through and with it carried on along the strike, and
   the one that reads best at the game camera is written (below).

## Numbers

Known answer (`cargo test --release -p motion_core known_answer -- --nocapture`):
a fixture rig (twisted rest frames, A-pose) posed by hand (turned hips,
leaning chest, turned head, arms bent 70 and 30 degrees, a knee up),
drawn front and side as exact landmarks, fused and fitted back:

| views | MediaPipe depth off by | limbs worst | torso/hip/shoulder lines worst | head |
|---|---|---|---|---|
| front + side | 0 | 0.00 deg | 0.54 deg | 0.00 deg |
| front + side | 5 cm | 0.85 | 1.33 | 1.79 |
| front + side | 15 cm | 2.64 | 3.88 | 5.50 |
| front only | 0 | 0.00 | 0.54 | 0.00 |
| front only | 5 cm | 12.98 | 29.02 | 20.65 |

Two views carry a bad depth guess; one view leans on it.

The HLH drawn sheets for the two diagonal two-handed cuts (three figures
each, front and side, 2026-11-13). The fit's joints drawn back into the
pictures, mean distance from MediaPipe's landmarks as a share of the
figure's height (report `overlay.mean_err_share`; the pictures:
`bench/keypose/kp_cut_*_overlay.png`):

| key | cut right front / side | cut left front / side |
|---|---|---|
| load | 6.8% / 18.3% | 4.5% / 4.4% |
| strike | 9.1% / 12.1% | 7.0% / 6.6% |
| follow | 9.0% / 21.7% | 6.8% / 23.6% |

The large side numbers are the keys whose legs were given the strike's
leg order (the drawings swap which foot is in front between figures:
the front view's depth guess and heights put the right cut's load with
the right foot forward, its strike with the left forward). Fitted blades
are within 1.3 degrees of the drawn blades; the off fist on the grip
within 7.8 cm (cut left strike), else 0.

`motion check --camera game` on Andras's rig (2 m), the keypose clips
beside the game's attacks and the Kimodo clips made from the same
drawings (`bench/keypose/motion_check.txt`). *read* is the game-camera
read score (`docs/motion-check.md`: body change, blade sweep, tip arc,
held load against readable references, 0..1):

| clip | frames | skate | sink | through body | limits | windup / strike / hold | tip peak | read (body / sweep / arc) |
|---|---|---|---|---|---|---|---|---|
| kp_cut_right | 36 | 0% | 0 cm | 20 fr, 10.2 cm | 4 fr, worst 5 deg | 0.20 s / 0.10 s / 1 fr | 48.3 m/s | **0.53** (0.79 / 0.19 / 1.79) |
| kp_cut_left | 36 | 0% | 0 cm | 23 fr, 8.3 cm | 1 fr, worst 1 deg | 0.23 s / 0.10 s / 10 fr | 41.6 m/s | **0.43** (0.84 / 0.13 / 1.12) |
| attack_1 | 21 | 86% | 4.8 cm | 21 fr, 4.4 cm | 0 | 0.13 s / 0.10 s / 3 fr | 40.6 m/s | 0.37 (0.26 / 0.37 / 1.28) |
| attack_3 | 26 | 88% | 10.8 cm | 4 fr, 2.2 cm | 0 | 0.13 s / 0.10 s / 14 fr | 45.5 m/s | 0.49 (0.39 / 0.33 / 1.96) |
| sf_cut_right (Kimodo) | 31 | 26% | 89 cm | 25 fr, 13.3 cm | 3 fr, 7 deg | 0.20 s / 0.30 s / 1 fr | 24.1 m/s | 0.75 (0.73 / 0.84 / 2.18) |
| sf_cut_left (Kimodo) | 31 | 58% | 93 cm | 27 fr, 13.4 cm | 3 fr, 1 deg | 0.27 s / 0.20 s / 0 fr | 21.3 m/s | 0.35 (0.73 / 0.12 / 0.49) |

The follow-through carry (`--carry auto`): the drawn follow-throughs
pull the hands back in after the contact, so the blade tip's path hooks
(arc consistency 0.32 on the right cut, 0.53 on the left). Carrying the
upper body on along the strike (each bone's load-to-strike turn
continued by 35%, blended with the drawn follow-through; legs, hips and
the off arm kept; elbows, torso, head and wrists brought back into their
limits) straightens it. Read at the game camera, carry 0 / 0.5 / 1: right
0.38 / 0.41 / 0.53, left 0.39 / 0.36 / 0.43; auto keeps 1 on both (a
carried follow-through must read 0.02 better than the drawn one). By eye
(`attacks_reel.mp4`) the right cut's carried follow-through ends with
the blade hanging straight down in front, hands at the chest: it reads
as a finished cut but is less like the drawing than carry 0.

What still keeps the read down: the sweep. Seen from behind, a
two-handed cut in front of the body covers little screen (0.13-0.19
against the 0.8 a readable reference reaches); the Kimodo right cut's
blade swings wide to the side (0.84).

Feet modes on the right cut (skate / root jerk mean / steps):
`hold` 0% / 1150 / 2 steps of 121 cm; `order` 0% / 1582 / 4 steps
(43-78 cm); `drawn` 0% / 2233 / 5 steps up to 220 cm and 4 floating
frames. `hold` is the default; `order` stays for sheets whose figures
step on purpose.

Back-and-forth (`rough`) peaks at 146-268 mm on the arms at the strike
(frames 14-16): the coil's reversal into the strike and the contact.
The reel shows no zigzag there (one wind back, one cut); the measure
counts any single acceleration reversal, so a deliberate anticipation
scores as roughness. Not changed here (other clips' numbers lean on it).

## Limits and next

- The drawings are not consistent between figures and views (legs swap
  front and back; the right cut's follow-through folds 39 degrees past
  the torso cap). The fit reports this rather than hiding it.
- Drawn arms are about 20% longer than the rig's; fists sit nearer the
  shoulders than drawn (same extension).
- No horizontal-slash sheet exists yet; both cuts are diagonal.
- One-handed swords: `--hands one`, untested on a sheet.
