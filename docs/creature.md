# Creature motion: AI video of our creature in, an animation on its own skeleton out

For bodies with no mocap: four legs, wings, tails, rock people, a
centaur. Built 2026-10-29 after CanItCode's method (his video
`AS92HYsEBqo`; his tool is unreleased, this is our own). The whole
problem is finding every joint in every frame and knowing which is
which; two known cameras then make 3D a simple crossing of two lines.

Dead ends he measured, so nothing here repeats them: point tracking
(points slide off legs), SAM 2 outlines (legs blur, far legs vanish or
swap), single-image depth (near/far leg right 4 times in 10), painted
socks per leg (the video model repaints them), animal pose detectors
(far legs drawn on the belly), MoCapAnything (animals only, wrong
gait), and one camera (a person lifting a stone matched the outline and
was wrong in 3D).

Numbers below are measured; the commands are in `docs/measurements.md`
("Creature motion").

## The pipeline

1. **Skeleton** (`tools/creature_rig.py`): `skintokens rig` over several
   seeds, then `motionforge creature-profile` scores each try: every
   leg on the ground has a bone (ground-contact blobs of the mesh each
   need a bone end within 12% of the height), left and right match
   (off-centre joints need a mirror partner within 6%), fingers are
   folded into the hand (a joint whose three or more child chains all
   stay within a quarter of the height loses them; tiny leaves fold
   into their parent), bones that stick out of the body are cut off
   (leaf joints outside the mesh and clear of its surface). Pruned
   joints hand their skin weights to the nearest kept ancestor; the
   cleaned GLB is what the game loads. Floating mesh debris (under 2%
   of the vertices) is ignored for the ground and the bounds. Highest
   score wins, ties to the lowest seed.
2. **Profile and start frame**: `creature-profile` writes
   `motionforge-creature` v1: the skeleton (Blender Z-up, bones named
   after the GLB joints, tails at the mean child head), a scalar joint
   limit per bone (root 20°, hubs with three or more children 25°,
   leaves 30°, everything else 40°; edit the JSON to tighten), the
   feet, the facing direction (from the mesh: the top of the creature
   sits forward of the centre on a quadruped; a biped's head is over
   the centre, then glTF +Z), the height, the ground and two fixed
   cameras: `front` on the forward side and `side` on the right, both
   at the centre's height, 35° vertical field of view, far enough that
   1.2 bounding radii fill the frame, 512×512 each. The start frame for
   a video model is the rest pose rendered through both (front on the
   left, side on the right) on plain grey (`render.py --mode motion` on
   a one-frame clip).
3. **Joint finder** (`python/creature/`): `creature-poses` makes 3,000
   random poses, every joint bent about a random axis by a random
   fraction of its limit, the root shifted a little; half are "fast"
   (frames 3k, 3k+1, 3k+2 = before, pose, after; the render of 3k+1
   with a one-frame shutter blurs half way to each). `render.py` films
   them in headless Blender (Eevee) with random key/fill light,
   colour, exposure and body tint; `train.py` degrades them further on
   the fly (exposure, contrast, gamma, cast, saturation, blur, codec
   softness, noise, blocks, drift) and fine-tunes
   `facebook/dinov2-small` (ViT-S/14, Apache-2.0, 22 M parameters) with
   a small conv decoder to one heatmap per joint at 448×448 input,
   soft-argmax positions, peak-based confidence. Occluded joints are
   still targets, so a far leg gets a guess instead of a hole. Sized
   for the 8 GB RTX 2060 SUPER: batch 16, fp16 autocast.
4. **Reading a video** (`detect.py` + `creature-solve`): the composite
   is split into its halves, each half goes through the finder, and
   `creature-solve` first checks that both halves show the same move:
   per joint, the signed height disagreement between the two rays over
   time is compared with the joint's own height motion
   (`motion_ratio`: near 0 for one move seen twice, around 4 for two
   different moves; reject above 0.5) and the median ray gap must stay
   under 8% of the height. Then triangulation (closest point of the two
   rays, confidence shrinking with the gap), Gaussian smoothing of the
   points, a top-down fit of every bone's world rotation to its
   children's observed offsets (shortest arc for one child, Horn's
   quaternion fit for more) clamped to the joint limits, quaternion
   smoothing, and planted feet: a foot low and slow for at least three
   frames is held at its mean position by CCD on the leg chain within
   limits, blended in and out over 0.12 s. Output: a motionforge clip
   JSON and, with `--glb-in/--glb-out`, the rig GLB with the animation
   (what the games load; Blender's "Solve Creature Motion" imports it).
5. **The exam** (`python/creature/exam.py`): per creature and seed a
   random smooth motion within the limits (`creature-motion`: two
   sinusoids per bone, legs at 45% amplitude so feet stay low, root
   bobbing and turning on the spot), filmed through the same cameras,
   run through finder, triangulation and solve, and compared bone by
   bone with the truth (`creature-compare`: world rotation error per
   bone, head position error per joint, foot slide over still frames).
   `--check tests/fixtures/creature/exam_baseline.json` fails when any
   creature's mean joint error or mean inner-bone error grows by more
   than 5% (floor 2 mm / 0.5°). Run it for every change; a change that
   makes any creature's exam worse is out. `cargo test` keeps the
   deterministic half of the exam (perfect and noisy synthetic
   detections, the reject case) on the fixture biped.

## A four-legged walk and trot without video

`creature-gait --creature c.json --glb-in rig.glb --glb-out out.glb --rom
rom.ron [--clips walk,trot,attack] [--clips-json DIR]` reads a quadruped's
parts off its profile (the four chains that end in feet, front or hind by
where they leave the body; the spine between them; the chain reaching
forward from the chest; the one reaching back from the hips) and writes
clips on that skeleton into the GLB, and its range of motion as a
weightforge pose set (`weights check --poses rom.ron`). `tools/tripo_rig.py`
(`docs/tripo-rig.md`) runs it for every creature.

**`walk` and `trot` keep their feet where they are put** (`gait::stride`,
2026-11-11). Each foot is given a spot on the ground for as long as it
bears weight and an arc through the air when it steps; the leg is solved
to reach it:

- Timing from the creature's own legs: the cadence from the gait's Froude
  number (walk 0.20, trot 0.90) and its stride in hip heights (1.6, 2.3);
  a foot is down 65% of a walk's cycle (four beats: hind left, front
  left, hind right, front right) and 42% of a trot's (diagonal pairs).
- The stride is cut to what the legs reach with the body lowered 7% of
  the hip height, and the speed goes down with it, not the cadence up: a
  model that stands on nearly straight legs takes short, calm steps.
- The leg: joint positions by reaching back and forth from the rest shape
  as the body carries it (24 passes; every joint keeps bending the way it
  does at rest), each bone then aimed along its segment with no added
  twist, the paw kept flat as it stands at rest. Bones never stretch: a
  spot out of reach is met as far as a nearly straight leg allows.
- The clip stays on the spot. The command prints the speed to move the
  body at ("move the body 0.935 m/s"); moved at that speed, a planted
  foot holds its place (the test allows 0.2% of the height). While a foot
  steps it leaves and lands at the body's speed and rises 11% (walk) or
  16% (trot) of the hip height.
- The body bobs twice a cycle, the spine sways with the hind legs, the
  head nods against the bob, the tail swings behind.
- Measured (2026-11-11, two generated models): a horse walks 0.94 m/s
  with a 0.47 m stance in a 0.77 s cycle and trots 1.93 m/s; a wolf whose
  legs stand almost straight walks 0.64 m/s with a 0.30 m stance and
  trots 1.60 m/s.
- Not done: flat ground only (placing feet on slopes is the game's, at
  run time); no turn or start and stop clips; the paw does not roll off
  its toes.

`attack` (a lunge) and `walk_spot` (the older walk, legs swung by angle,
feet sliding; kept to set beside `walk`) are plain forward kinematics.
`--clips-json DIR` also writes each clip as `DIR/<name>.clip.json` for
`python/creature/render.py --mode motion`.

## Lining up the views of a generated video

A video model is handed the start picture and returns a clip that need
not keep to it. `creature-solve` can line the two halves up before it
triangulates (`creature::align`; all off by default, so the exam and the
goldens are unchanged):

- **`--align-views`: the cameras re-fitted on the first frame.** Per
  view, a zoom about the image centre and a shift (three numbers) that
  take the rest pose, projected through the profile's camera, onto the
  finder's joints on frame `--align-frame` (0: the start picture, where
  the creature still stands at rest); confidence-weighted least squares,
  five reweighting rounds so a misplaced joint doesn't pull it. Every
  frame of that view is then mapped back through it. A zoom outside
  0.5..2 refuses (the frame is probably not the rest pose).
- **`--align-time lag|warp|auto`: one view slid in time against the
  other** (the second of the pair; up to `--max-lag 12` frames, capped
  at a quarter of the clip). `lag` tries every whole-frame lag, keeps the
  one with the lowest height-disagreement motion ratio (the solve's own
  check), then refines it to a fraction of a frame (parabola through the
  neighbours; kept only when it measures lower). `warp` is dynamic time
  warping on height, the one coordinate two orthogonal views share: per
  frame and lag, each joint's signed height disagreement between the two
  rays less its mean along the best constant lag, over the joints'
  height variance; one lag per frame, moving at most a frame per frame
  at a cost of 0.1 a step, solved exactly (Viterbi), smoothed (sigma 2
  frames). `auto` takes the constant lag when it lowers the ratio and
  the warp only when its ratio is under 90% of the constant lag's (it
  has more freedom). Frames whose partner falls outside the clip are
  dropped; the report says how many and from where.
- **The clip's own frame timing.** `detect.py` writes the video's
  timestamps (`times`, from ffprobe) into the detections; a clip whose
  timestamps are more than a quarter frame off a uniform grid is
  resampled to its fps before anything else.
- `detect.py` now writes positions in each view's own pixels (the
  profile's `width`/`height`). Before, it wrote them in the pixels of the
  video's half, and the first real clip's halves were 640 px for 512 px
  views; the camera fit absorbs that too (zoom 1.25 and a 64 px shift).
- `--aligned-out det.json` writes the detections after lining up, and
  `tools/creature_views.py` (stdlib) says where two views disagree: the
  motion ratio at a range of lags, per leg (each foot's chain) and for
  the rest of the body the ratio at lag 0 and at its own best lag, the
  finder's confidence per view, and the pooled ratio with every group at
  its own best lag.

Known answers (`rust/core/tests/creature_align.rs`, the fixture biped,
1 px noise; `cargo test --locked --release -p motion_core --test
creature_align -- --nocapture`, f, 2026-11-13):

| case | before | after |
|---|---|---|
| side view 5 frames late (`lag` and `auto`) | motion ratio 0.602, rejected | lag 5.06 frames, ratio 0.019, accepted, joints 6.0 mm from the truth |
| side view playing 8% slow (lag 0 -> 7.7 frames) | ratio 0.442 | constant lag 4.43 frames: 0.080; warp (lag 2..6): 0.033, chosen by `auto` |
| front zoomed 1.25 and shifted (30, -12) px, side 0.9 and (-8, 15) | median ray gap 14.7% | fit 1.2467, (30.13, -11.94) and 0.8984, (-7.92, 15.07); frame-0 rms 43.6 -> 1.56 and 20.3 -> 0.85 px; gap 0.36% (untouched clip 0.34%) |
| both halves read at 640 px for 512 px views, side 6 frames late (1.5 px noise, 2% lost) | gap 5.33%, ratio 0.604, rejected | gap 0.54%, ratio 0.079, accepted, lag within a frame of 6 |
| two different moves, one per view | rejected | still rejected under `lag`, `warp` and `auto` |
| every fifth frame dropped, timestamps kept | 40 frames | resampled to 50, every joint within 6 px of the full clip |

**The Seedance wolf walk with the views lined up (2026-11-13): still
rejected, and rightly.** Same detections as 2026-11-11 (fleet job
`vvyr2dq3` on f), `creature-solve --align-views --align-time auto`:

| | as read | lined up |
|---|---|---|
| camera fit (frame 0) | - | front zoom 1.195, shift (64.6, 61.5) px; side 1.249, (63.8, 64.8); rest-pose fit 89.3 -> 8.6 and 91.5 -> 8.6 px rms |
| median ray gap | 4.69% of the height | 2.67% |
| motion ratio (limit 0.50) | 1.142 | 1.143 (lag -0.38 frames; warp 1.049, lag -3.0..3.4, not 10% better) |
| fit residual (forced solve) | 112.4 mm rms | 13.0 mm |
| bone-frames held at a limit (forced) | 369 | 120 |

The camera fit is worth keeping on for every generated clip: the rays
now cross (residual 112 -> 13 mm). Rendered on the wolf through both
cameras beside the video (`python/creature/render.py --mode motion`,
y8, every 6th frame of the first 48; local `bench/creature-video/`),
the forced solve with the fit stands on the floor and steps as the
video does (`tools/creature_feet.py --creature creature.json
clip.json`): lowest foot joint 11-30 mm over the clip (rest 22-26 mm),
three lifts over 3 cm per front foot in 5 s; without the fit the whole
wolf sinks, feet 3-16 cm under the floor (the 640/512 px scale). Its
weak part is where the views disagree: the hind feet wobble sideways
20-23 mm (sd) against 4 mm for the front feet, and the hind-left foot
counts six lifts to the others' three. (A render from the clip's own
two cameras matches each view's detections by construction, so the
sideways wobble is the honest number.) The time slide finds nothing because
the clip is not out of step: the motion ratio over constant lags has its
minimum at 0 (`tools/creature_views.py`: -12: 8.65, -4: 2.07, -2: 1.35,
0: 1.14, +2: 1.31, +4: 1.69, +12: 4.05), and even letting every leg and
the body take its own best lag only reaches 0.80. Its timestamps are
uniform (ffprobe: every step 1/24 s), so the frame timing changes
nothing either. Two more things measured and ruled out: following the
front view's slow zoom drift (1.187 at frame 0 to 1.217 at 120,
per-frame fit smoothed over 6, 12 or 24 frames: ratio 1.21, 1.15, 1.15),
and weighting the ratio by the finder's confidence (1.00, or 0.85 with
confidence squared).

What the disagreement follows is what each view can see. Per leg, ratio
at lag 0 and the finder's mean confidence front / side: front right
(seen in both) 0.33, 0.96 / 0.94; front left (behind the near legs in
the side view) 0.64, 0.93 / 0.89; hind right (behind the front legs in
the front view) 1.20, 0.44 / 0.99; hind left (hidden in both) 1.61,
0.45 / 0.90; the rest of the body (spine and tail behind the chest in
the front view) 1.41, 0.66 / 0.97. The pictures agree: in the front view
the hind paws stand behind the front ones, and the side view's far legs
are behind the near ones (`bench/creature-video/`, local). So the score
and the pictures agree and the benchmark stands; the joints one view
guesses are what reject the clip. What it would take: solve a joint
seen well in one view only from that view's ray and its bone length
(the confidence is already there for the front view, 0.3-0.6 on hidden
joints, but the side view is sure of its far legs, 0.9), or film the
start frame from two views that each see all four feet (front-left and
rear-right three-quarters), or prompt and choose clips by this check.

## Formats

- `motionforge-creature` v1: `name`, `height`, `ground`, `forward`
  (horizontal unit vector), `feet` (bone names), `bones`
  (`name, parent, head, tail, max_angle_deg`; same skeleton rules as
  `motionforge-skeleton`), `views` (`name, position, look_at,
fov_y_deg, width, height`; pixel origin top-left of each view).
- `motionforge-detections` v1: `fps`, `views`, `joints`, optional
  `times` (one presentation time in seconds per frame, increasing),
  `frames` =
  `[[ [u, v, confidence] per joint ] per view] per frame`, pixels of
  the view's own image; confidence 0 = missing.
- `motionforge-render-truth` v1 (render.py): per sample the clip frame
  and, per view, `[u, v, depth]` per joint from Blender's projection.
- Poses clip: three frames per sample (before, pose, after).

## Limits

- Limits are v1 scalar caps (angle from rest), not per-axis; a knee can
  bend sideways as far as forward. Tighten per bone in the profile.
- The twist of a leaf bone about its own axis is invisible to any
  two-view joint finder; `creature-compare` reports `mean_deg_inner`
  (bones with children) beside the all-bone mean for that reason.
- Two orthogonal views only share one coordinate (height), so the
  consistency check sees two different moves through height alone; a
  clip whose halves differ only sideways passes it.
- The video step is pluggable. The free Wan 2.2 TI2V-5B try
  (`python/creature/video_wan.py`) kept the layout and passed the
  consistency check but barely moved the creature
  (`docs/measurements.md`).
- **The first real clip was rejected (2026-11-11).** A five-second walk
  of a generated wolf from Seedance 2.0 Fast (720p, the two-view start
  frame padded to 16:9, 121 frames at 24 fps): the creature walks on the
  spot in both views with still cameras. Its joint finder (3,000 poses,
  12 epochs, 1,252 s on the 8 GB card) held out at 8.46 px mean, 5.19 px
  median; on the clip the mean confidence was 0.82. `creature-solve`
  refused it: median ray gap 4.7% of the height (limit 8%), but the
  height-disagreement motion ratio was 1.14 (limit 0.50): the two
  views disagree about the legs (first read as out of step). The view
  alignment built for it (camera fit, time slide, frame timing;
  "Lining up the views")
  brings the ray gap to 2.67% and the residual from 112 to 13 mm but
  leaves the ratio at 1.14: the disagreement is the legs each view
  hides, not timing.
- The exam films through Blender, the same renderer the finder trained
  on; real video shifts the domain (the degradation is the hedge).
