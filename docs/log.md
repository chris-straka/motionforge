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

## 2026-11-13, round 2: the stylizer on the game's attacks

Eleven attack clips (attack_1-3, quirin-attack_1-3, enemy-attack_1-3,
sf_cut_left/right), from the cleaned set, stylized with round 1's body
mode (`--snap-speed 0 --accent none`) and with this round's (the sword
hand's stops are keys too; exaggeration scaled by `6 m/s / hand peak`,
x0.41-0.94 on these clips).

| measure (sum over 11 clips) | original | round 1 | round 2 |
|---|---|---|---|
| skate cm | 4.6 | 50.0 | 39.7 |
| floating frames | 8 | 55 | 35 |
| through-body frames | 361 | 426 | 419 |
| back-and-forth mm | 9.3 | 236.4 | 200.0 |
| joint-limit frames | 32 | 197 | 180 |
| key-pose contrast deg (mean) | 45.2 | 46.3 | 44.9 |

attack_1's windup by the numbers: 0.13 s original, 0.03 round 1, 0.23
round 2. By eye (painted reel with the hand's path, `attack_1_stylized`):
the original's windup swings the arm wide to the right (frame 6); both
stylized versions keep the hand by the body there, so the backswing reads
smaller, and the hand's path gains corners. **Kept** over round 1's
settings (better on every defect), **not for the game**: the original
clips still read best and score far smoother. The stylizer stays opt-in
(`stylize`, `animate --stylize`).

Next: find the back-and-forth in the stylized clips (hold-to-strike
corners in the time warp, and the re-plant moving ankles up to 25 cm);
keep the source's widest windup pose as the anticipation instead of a
counter-rotation from the key.

## 2026-11-13, round 3: the stylizer's double windup

Owner rule (2026-11-13, via the orchestrator): a method that wins on some
clips stays and is chosen per case; a promising method that loses is
worked on; when score and pictures disagree, fix the benchmark first.

The score said round 2's stylized attack_1 was 150x rougher than the
original (63 vs 0.4 mm, one spike of 743 mm at frame 8). The frames agree
with it: the sword hand goes out (6), up (7), out wide (8): a zigzag.
Cause: the stylizer added its counter pose (anticipation) on top of the
source's own windup, so the arm wound up twice. Two fixes, each on the
eleven attack clips:

| measure (sum over 11 clips) | original | round 2 | no counter on a source windup | + the strike's load is a key |
|---|---|---|---|---|
| skate cm | 4.6 | 39.7 | 30.1 | 34.4 |
| floating frames | 8 | 35 | 30 | 38 |
| through-body frames | 361 | 419 | 408 | 408 |
| back-and-forth mm | 9.3 | 200.0 | 180.3 | 133.5 |
| joint-limit frames | 32 | 180 | 152 | 159 |
| joint-limit worst deg | 43.1 | 274.4 | 182.3 | 174.2 |

- No counter pose when the segment before a strike already moved the
  weapon hand against it by 30%+ of the strike.
- The strike's load (last frame before the hand's fastest where it still
  moved against the strike, as `motion check` finds it) is a key: the
  hand turns round at speed there, so speed minima missed it.

attack_1 now: back-and-forth 29 mm (round 2: 63), the windup is held up
beside the head for a beat where the original swings wide to the right
(`bench/round1/attack_1_round3.*`): a different windup, the owner's call.
Still rougher than the originals everywhere. Next: find the remaining
spikes (quirin-attack_2 652 mm at frame 14, attack_1 284 at 6: the
hold-to-strike corner of the time warp), then `--stylize auto`: stylize a
clip only where its defects stay within the original's and its windup or
hold grows, and report per clip which won.

## 2026-11-13: videos at the game camera, and a readability measure

The owner on the old strips: "In pic 2 I have no idea what he's doing,
is that an attack animation? If so, it fails completely." "The sword
model in pic 2 is bad." "Shadows ... are wild." "Wouldn't video be
better for me to evaluate?"

**Videos** (`tools/bench/motion_video.py`, kept): MP4 panels at HLH's
follow camera (3.0 m behind, focus 1.535 m up, pitch -18.3 deg, 75 deg
vertical; `docs/motion-check.md` has the sources), EEVEE, painted, plain
floor, one soft sun. On the grafted Andras (the starred head on the
travel body, 1.0 m tall in its GLB, shown at the game's 2 m by moving
the camera) with the game's clips put on by `motionforge animate
--no-contact`, and the props kit's sword held as the game holds its
blade (`tracks::held_pose`). Checked by eye: the grip is in the right
hand and the blade comes out of the fist in idle and through the
attacks; laid along the socket's +Y instead, the blade crosses his body.
The grafted rig has no finger animation, so the hand stays open round
the grip. Scaling the imported armature broke its skinning in Blender
(the arms stayed in the rest pose); moving the camera instead does not.

**Read at the game camera** (`motion check --camera game`, kept). First
cut: pose change against the idle, blade sweep, tip arc, held load.
Two disagreements with the pictures, fixed in the benchmark:

- Against the idle's silhouette, the CC0 Sword_Idle scored 0.54 (its
  stance differs from an idle); the neutral is now the clip's own first
  frame (0.15).
- The pose change counted the blade, so a thin blade sweeping across a
  still body scored like a whole-body move. By eye at the game camera
  attack_1 is only the arm and blade crossing at shoulder height over an
  upright, unchanged body; attack_3 raises both arms and steps into a
  chop; the CC0 Sword_Attack crouches, twists and swings high overhead.
  The score now uses the body alone ("body change").

Also: the held load counts only as far as an arc follows it (idle and
other still clips "held" all clip long), and turns on the tip path are
weighted by step length (slow jitter at the ends is not a wiggle).

| clip | read | body | sweep | arc | held load |
|---|---|---|---|---|---|
| CC0 Sword_Attack (reference) | 0.58 | 0.69 | 0.55 | 1.30 | 0 |
| CC0 Sword_Regular_B | 0.54 | 0.56 | 0.34 | 1.82 | 0 |
| quirin-attack_2 | 0.66 | 0.87 | 0.51 | 2.15 | 0 |
| quirin-attack_3 | 0.62 | 0.55 | 0.80 | 1.43 | 0 |
| quirin-attack_1 | 0.60 | 0.71 | 0.65 | 0.96 | 1 |
| enemy-attack_2 | 0.59 | 0.69 | 0.47 | 1.62 | 0 |
| attack_3 | 0.49 | 0.40 | 0.35 | 1.93 | 0 |
| attack_2 | 0.48 | 0.31 | 0.52 | 1.66 | 0 |
| attack_1 | 0.42 | 0.29 | 0.38 | 1.59 | 0 |
| enemy-attack_1 | 0.41 | 0.54 | 0.36 | 0.86 | 0 |
| enemy-attack_3 | 0.31 | 0.37 | 0.38 | 0.36 | 1 |
| idle | 0.04 | 0.07 | 0.01 | 0.03 | (28, not counted) |

Command: `motionforge motion check <grafted>.glb --camera game --json`
(and the same on the CC0 Quaternius UAL clips put on the same body).
The ranking the owner asked for holds: a clearly readable attack (the
CC0 reference 0.58, attack_3 0.49) above attack_1 (0.42) above idle
(0.04). What it says about the game: Andras's three attacks barely move
the body (0.29-0.40), where Quirin's, the husk's first two and the CC0
ones reach 0.54-0.88 (the husk's third is 0.37); none of the attacks,
the CC0 ones included, holds its load a beat. Not yet checked against the
owner's own picks.

## 2026-11-13, creature video: views aligned

The first real creature clip (Seedance 2.0 Fast, a generated wolf walking
on the spot, 121 frames at 24 fps) was rejected by `creature-solve` on
2026-11-11: ray gap 4.7% of the height, motion ratio 1.14 against 0.50.
Built for it, in `creature::align` (`docs/creature.md`, "Lining up the
views"), each judged on known answers first
(`rust/core/tests/creature_align.rs`) and then on the clip:

| piece | known answer | the wolf clip | verdict |
|---|---|---|---|
| camera fit on frame 0 (`--align-views`) | zoom 1.25 / 0.9 and shifts recovered within 0.3% and 0.13 px; gap 14.7% -> 0.36% | zoom 1.195 front, 1.249 side (its halves were read at 640 px for 512 px views); gap 4.69% -> 2.67%; forced-solve residual 112 -> 13 mm, limited bone-frames 369 -> 120 | **kept**; use it on every generated clip |
| constant lag (`--align-time lag`) | 5 frames late: found 5.06, ratio 0.602 -> 0.019 | best lag -0.38 frames, ratio 1.164 -> 1.143 | **kept** (wins where a view is late; this clip isn't) |
| warp (`--align-time warp`, chosen by `auto` only when 10% better than the lag) | 8% slow view: warp 0.033 vs lag 0.080, chosen | 1.049 (lag -3.0..3.4), not 10% better | **kept** (wins on a drifting view) |
| frame timing (`times`, detect.py via ffprobe) | every fifth frame dropped: resampled to 50, within 6 px | timestamps uniform, nothing to do | **kept** |

The forced solve with the fit, rendered on the wolf beside the video,
stands on the floor (lowest foot joint 11-30 mm, rest 22-26) and steps
with the video (three lifts per front foot); without the fit the feet
are 3-16 cm under the floor. The hind feet wobble sideways 20-23 mm
(front 4 mm).

Still rejected (motion ratio 1.14), and rightly: the ratio over constant
lags has its minimum at 0 (-4: 2.07, 0: 1.14, +4: 1.69), every leg at its
own best lag only reaches 0.80, following the front view's zoom drift
(1.187 -> 1.217) leaves 1.15, and weighting by the finder's confidence
1.00. The disagreement follows occlusion: the front-right leg, seen in
both views, 0.33; front-left (hidden in the side view) 0.64; hind-right
(hidden in the front view) 1.20; hind-left (hidden in both) 1.61. The
finder knows it in the front view (confidence 0.44 on hind legs) but not
in the side view (0.90 on its far legs). `tools/creature_views.py`
prints this for any clip.

Also: `detect.py` wrote positions in the video half's pixels instead of
the view's; fixed (positions in the profile's view pixels).

Next for creature video: joints seen in one view only solved from that
view's ray and their bone length; a finder confidence that knows the
far legs are hidden; or a start frame from two three-quarter views that
each see all four feet.

## 2026-11-13, keypose: sword attacks from drawn key poses

Owner's idea (a): key-pose pictures of a swing, read onto the rig, keyed
between with an animator's timing. Built `motionforge keypose` (Rust:
two-view fusion, bone-by-bone fit, timing; Python only to run MediaPipe
and draw the fit back over the pictures). How and every number:
`docs/keypose.md`.

- Known answer: a hand-posed fixture drawn front and side fits back
  within 0.00 deg per limb (0.85 deg with MediaPipe's depth 5 cm off;
  one view alone with that depth error: 13 deg).
- Made the two diagonal two-handed cuts from HLH's drawn sheets
  (`kp_cut_right`, `kp_cut_left`, 36 frames): no skate, no sink, limits
  4 and 1 frames (worst 5 deg), windup 0.20 s, strike 0.10 s, tip 41-49
  m/s. The game's attack_1 and attack_3 skate 86-88% and sink 5-11 cm;
  the Kimodo clips made from the same drawings sink 89-93 cm.
- Read at the game camera (the new `motion check --camera game`): right
  cut 0.53, left 0.43; attack_1 0.37, attack_3 0.49, Kimodo 0.75 / 0.35.
  The drawn follow-throughs pull the hands back (the tip's path hooks:
  0.38 / 0.39); `--carry auto` carries the upper body on along the strike
  when that reads better (it did on both). Still low: the sweep (a cut in
  front of the body covers little screen seen from behind).
- Tried and kept as options: fitting a turn between the views (worse
  than none on these sheets), feet as drawn (5 steps up to 2.2 m: the
  drawings swap which foot is in front) and per-key stances (4 steps).
  Default: every key on the strike's feet.
- Open: `rough` scores the anticipation's single reversal as roughness
  (240 mm at the strike; the reel shows one clean wind back and cut);
  no horizontal-slash sheet yet.

## 2026-11-13, round 4: back-and-forth vs snaps (the benchmark fixed first)

The key-pose attacks read cleanly on video but scored 146-268 mm of
back-and-forth peaks: a strike that starts and stops within two frames
flips its acceleration once, and that counted as jitter. A flip now counts
only inside a run of flips (the frame before or after flips too). Test: a
0.6 rad two-frame snap keeps under 3% of its travel (it was about half);
the +-0.02 rad jitter and the even swing keep their known answers.

| back-and-forth mm, worst chain mean (11 attack clips) | before | after |
|---|---|---|
| originals, sum | 9.3 | 1.9 (attack_3 2.78 -> 0.13: all snaps) |
| round-2 stylized, sum | 200.0 | 149.8 |
| round-3 stylized, sum | 133.5 | 109.1 |

The round-2 stylized attack_1's zigzag (743 mm peak at frame 8, seen in
the frames) still counts in full.

## 2026-11-13, round 5: attacks authored from the sword's path

The unreadable attack was settled as the clip's fault (same keys on four
bodies, two of them professionally rigged, read the same). Method changed:
`motionforge attack` (`docs/attack.md`) authors an attack from sword keys
(grip and tip) with the body solved under them every frame.

- **Measured first**: six CC0 pro sword attacks turn the hips 60-120 deg,
  lean 30-57 deg, drop the hips 0.15-0.2 of the height and stand 0.5-0.95
  of it wide; the game's attacks did none of that (body change at the
  game camera 0.29-0.40 against 0.56-0.88).
- **Result** at the game camera: sweep 0.78, cleave 0.68, backcut 0.65
  (pros 0.41-0.67, game 0.42-0.49, round 2's key-pose cut 0.52), each with
  zero skate, sink and float.
- **Fixed on the way** (each seen in the score first): both feet stepping
  at once read as floating (steps staggered over two keys); the hips'
  drop searched in steps made them jitter (now exact); a foot turning
  about its ankle slid its ball (now pivots on the ball); the wrist limit
  moved the grip 2 mm off its point unreported (the arm is solved again
  for the final hand turn, misses are measured, test); the hinge solve
  landed 6 mm off on the hero (aimed again until on the point); an
  overhead chop straight down is edge-on to a camera behind him (arc
  0.47 one-way: replaced by the backhand diagonal).
- **Parked**: thrust as a camera-readable attack (reason: foreshortened
  from behind; it would take a side-on lunge or a camera that swings).
- **Next**: backcut's arm through the chest; back ankle in cleave; body
  values derived from the sword path when a key leaves them out; the
  owner's picks to calibrate the score; Kimodo seeds as the rival when f
  has room.

## 2026-11-13, round 6: grip, weight, phone clips (attack tool)

The owner on round 5's authored attacks: "clearly sword attacks. They
need refinement. And he's not holding the sword properly". Kept, on the
same tool (`docs/attack.md`, "Round 2"):

- **Grip from the hand**: handle across the palm, fingers at 0.9 of the
  hero's own fist (hand-tools grips), socket rewritten into the palm.
  Tried first: closing each finger until it touches the handle; it
  stopped at once (straight fingers already lie within touch of a handle
  under the knuckles): dropped for the fixed share.
- **Overlap**: hips 35% early, chest 20% early, sword 20% late. Hips now
  peak a frame before the tip at 730-1,060 deg/s (pros 0-2 frames,
  700-1,700); a 7-frame hold after the strike (pros 1-15).
- **Phone clips**: tall, figure-filling, close then game camera, normal
  then slow. Fitting the camera to a sphere round the sword's sweep left
  the figure at a third of the height; fitted to the body's projected
  extents instead.
- Cost: the wrist limit is passed on 17-22 frames per attack (palm
  grip); backcut's arm still crosses the chest (8 frames).

## 2026-11-13, pro bench: correctness against professional clips

`docs/pro-bench.md`. The 84 Quaternius UAL1+2 clips (CC0) on their own
mannequin, then through `animate`, `clean`, `clean --contact` and
`stylize` onto the hero rig and Quaternius' female mannequin, each stage
compared with the pro clip frame by frame.

### Kept

- **Back-and-forth needs a zigzag** (the acceleration turning back twice
  running). One turn is the speed peak of any fast move: pro clips
  flagged 53 -> 23.
- **Floating needs the whole body off the floor** (lowest skinned
  vertex), not both feet: lying and rolling frames no longer float.
- **No elbow sideways limit; knees, wrists and ankles from joint
  positions.** Pro clips past a limit 58 -> 40 with the first two; the
  position-based ones make a clip and its retarget read the same on rigs
  that rest differently.
- **Hand rest alignment without shared finger names** (`animate`): the
  hero's wrists were bent 62-70 deg off the pro's on every clip made on a
  T-pose rig; 5-8 after.
- `motion check --tracks` and `--no-through`; `pro_bench.py` (calibrate,
  run, rescore, why).

### Dropped

- Contact points on the sole instead of the joints: the pros pivot the
  foot on the joints (Walk_Loop 5.4 -> 13.9 cm).
- Sink measured on the mesh: the pros' heel slap sweeps the heel 2.5 cm
  into the floor (sink clips 18 -> 24).
- A knee hinge following the foot's direction across the thigh: flips
  when the toes point (knee "straighter than straight" on 45 pro clips).
- Key-pose distance as the timing check: speed minima on loops and idles
  moved up to 25 frames for nothing; the speed curve's shift replaced it.

### First numbers (before the hand fix; hero rig, clips passing of 84)

raw 6, retarget 21, clean 35, clean --contact 35, stylize 0 of 5. `clean`
removes every sink the retarget leaves (44 clips) and all skate; it is
worse on Crouch_Fwd_Loop, Slide_Exit, Sword_Regular_B and Jump_Start.
