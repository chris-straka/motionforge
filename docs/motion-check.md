# Motion check: a score for how a clip looks

`motionforge motion check <rigged.glb> [--clip NAME] [--json]` measures
every clip on a rigged character (`rust/core/src/motioncheck.rs`). It
reads the GLB only: the posed skeleton per frame, and the skinned mesh for
limbs inside the body. It changes nothing.

Why these measures: the automatic measures that track what people prefer
in generated and retargeted motion are foot skate, limbs passing through
the body, and jerk; floating matters little. Attacks add the parts of
"weight and snap" a number can see: the windup before the fastest frame,
how fast the weapon moves, how long the pose holds after the hit, and how
far the body travels.

## What it measures

All lengths are scaled to a 1.75 m body (`k = height / 1.75`, height
from the skinned mesh, else the skeleton), so one threshold fits a child
and a giant and numbers compare across characters.

- **Foot skate.** Each foot has two contact points: the ankle (heel side)
  and the ball of the foot (the toe joint). A point's height is how far
  it is above where it stands in the rest pose, so a flat foot is 0
  whatever the ankle's own height. Per frame the lower point is the
  foot's contact. A foot is *planted* when that point is under **5 cm**
  high **and** moving slower than **0.75 m/s** horizontally (central
  difference), so a foot skimming the floor mid-stride is a step, not a
  plant. A plant is a run of 2+ planted frames. Its slide is the
  horizontal path of the contact point, frame to frame, using the point
  that was lowest on the first frame of each pair (a heel-to-toe roll
  moves the contact from heel to ball and is not a slide). A plant
  **skates** when it slides more than **2.5 cm**. Reported: share of
  frames inside skating plants, total skate distance, the worst plant
  (foot, start frame, slide).
- **Ground.** Frames where any foot point is more than **1 cm** below
  its standing height (into the floor), and frames where both feet are
  more than 5 cm up (floating; a jump floats on purpose).
- **Limbs through the body.** The contact pass's own checks
  (`docs/contact.md`), read-only (`contact::check_frames`): the exact
  mesh check (forearm and hand vertices, and the blade on weapon clips,
  inside the torso/head/thigh mesh beyond what the rest pose already
  touches) and the capsule proxies. Reported: frames inside, deepest,
  worst frame, which part hit which.
- **Jerk.** Third difference of joint positions relative to the hips,
  `(p[f+2] - 3p[f+1] + 3p[f] - p[f-1]) / dt^3`, per chain: spine+head
  (spine.001-.006), each arm (elbow, wrist), each leg (knee, ankle,
  ball); mean and peak, in m/s^3 on a 1.75 m body. The root (the hips'
  world position) is reported apart, so travel does not hide in the
  limbs. The table shows the worst chain's mean and peak.
- **Back-and-forth** (2026-11-13; the roughness number to read before
  jerk). Per joint relative to the hips, how far the next frame's
  acceleration points back against this frame's:
  `max(0, -a_f . a_f+1) / max(|a_f|, |a_f+1|)` with `a` the second
  difference of positions. A fast even arc and a sharp single corner
  score exactly 0 (their accelerations agree, or come alone); a joint
  jittering +-e every frame scores 4e. Millimetres at 30 fps on a 1.75 m
  body; per chain mean and peak, the table shows the worst chain. Jerk
  stays in the JSON, but it ranks fast moves as rough.
- **Joint limits** (2026-11-13). Knees and elbows as hinges: the hinge
  axis is fixed in the upper bone (rest `upper x facing`), the bend is
  measured in the hinge plane, positive the natural way (shin back,
  forearm forward). Past a limit: straighter than straight by more than
  10 deg, bent more than 160 deg, or bent out of the hinge plane by more
  than 25 deg. Ball joints capped from their rest pose relative to a
  parent: chest on hips 80 deg, head on chest 90, hand on forearm 90,
  foot on shin 65. Reported: frames past any limit, the worst excess,
  which joint and how, every joint that went past.
- **Readability of the key poses** (2026-11-13; for comparing a clip
  with its own variants, higher is better). Key poses are the body's
  speed minima (the stylizer's own keys, `stylize_body::body_keys`),
  interior ones only when there are any. *Clear*: per key pose, four
  level orthographic views round the character's current facing (0, 45,
  90, 135 deg); in each, the share of limb points (elbows, hands, knees,
  ankles) further from the projected hips-chest-head line than the
  torso's half width in that view (half the shoulder width side to
  side, 0.3 of it front to back); the best view counts; mean over key
  poses. *Contrast*: between successive key poses, the mean angle each
  limb segment and the spine turned in the hips' frame. No published
  readability score was found to copy (research notes, 2026-11-13); this
  one is a first cut, to be checked against the owner's picks.
- **In-place cycles** (2026-11-13). When the hips travel under 10 cm
  over the clip and both feet's low frames move the same way at 0.3 m/s
  or more (their median velocities agree within 37 deg), the clip is a
  treadmill cycle: skate is measured with the feet's positions minus the
  belt (the mean of the two medians) times time. Reported as `belt`.
- **Attacks** (clip names containing attack, slash, cut, combo, stab,
  swing, strike, punch or kick; `--attack-clips` to change): the weapon
  hand is `--weapon` (default R). The *tip* is the blade's end when the
  rig has a hand socket (the contact pass's blade capsule), else the
  hand. **Peak**: the tip's fastest frame. **Load**: the last frame
  before the peak where the hand still moved against (or across) the
  strike's direction; if it never did, the slowest hand frame in the
  second before the peak. **Windup**: how long the hand moved against
  the strike leading into the load. **Strike**: load to peak. **Hold**:
  frames after the peak with the hand under 15% of its peak speed,
  counted from the first such frame. **Root**: the hips' horizontal
  start-to-end travel and furthest excursion.

- **Read at the game camera** (2026-11-13, `--camera game`;
  `rust/core/src/readcam.rs`). The owner's question: can a viewer tell
  an attack from the silhouette, seen from where the player sees him?
  Per frame the skinned body and the blade are projected through HLH's
  follow camera and filled into a 480x270 mask (the game's 1280x720 at
  three eighths). The camera: focus 1.535 m up (0.535 m over the body's
  centre, 1.0 m over the feet), 3.0 m back at pitch -0.32 rad (-18.3
  deg), straight behind him, 75 deg vertical field of view (HLH
  `bevy/src/fx.rs` `camera_follow` in source levels, `Fence::distance`
  default, `rig.pitch` at spawn; the game records the same numbers in
  `proportions.rs`). He is scaled to the game's 2 m (head joint at
  1.6836 m, `character.rs` `RigRest::stature`; `--stature`). The blade is
  a 5 cm ribbon from the guard (0.11 m) to the tip (0.91 m, the props
  kit's sword; `--blade-length`) held as the game holds it
  (`tracks::held_pose`: grip 4 cm up and 8.5 cm back in the socket, the
  blade cocked off the socket's +Y; `--blade-pose socket` for straight
  out along +Y). The strike's frames are the attack timing's (peak,
  load, windup start); the *settle* is the first frame after the peak
  with the tip under 15% of its peak speed. Parts:
  - *body change*: the largest `1 - IoU` of the body alone (no blade)
    against the clip's first frame, from the windup's start to the
    settle: does the body take part? (*pose change*, reported too, is
    the same with the blade, over load to settle.)
  - *sweep*: the area the blade covers over load to settle, over the
    first frame's body area.
  - *arc*: the tip's screen path over load to settle in body heights,
    times its turn consistency (`|sum of signed turns| / sum of turns`,
    each turn weighted by its shorter step; 1 for an arc or a straight
    cut, 0 for a zigzag).
  - *load hold*: the longest run of frames in the second before the peak
    whose silhouette changes under 4% from the frame before.
  - *score*: each part over what the readable references reach (body
    0.7, sweep 0.8, arc 2.0, hold 4 frames), capped at 1, averaged; the
    hold counts only as far as the arc reads (idle holds still all
    clip long).
  `--silhouettes DIR` writes each frame's mask as a PGM (body black,
  blade grey).

Clips are sampled at 30 fps whatever rate they were keyed at (`--fps`
to change), the rate the renders use, so frame numbers match the
pictures. Output is deterministic.

## Tests with known answers

`rust/core/tests/readcam.rs` (2026-11-13): a triangle fills exactly the
55 pixel centres inside it, either winding; a 6x4 px thick segment is 24
pixels; two 4x4 squares 2 px apart have IoU 1/3; a 90 deg camera 100 px
high has a 50 px focal length and puts a point 1 m up and right at 2 m
25 px off centre; the game camera's eye is `3 sin 0.32` up and `3 cos
0.32` behind the focus, which lands on the centre; half a circle in 20
steps is `20 * 2r sin(4.5 deg)` long with consistency 1 and 171 deg of
turn, a zigzag of eight equal turns has consistency 0, a straight cut 1;
the strike timing on the synthetic swing below; on the fixture
humanoid, standing still has no pose or body change and no arc, an arm
swung up 150 deg reads higher, the same twice byte for byte.

2026-11-13 additions: back-and-forth on the same +-0.02 rad forearm
jitter is exactly `4 L sin(theta)` at its peak (half that as the chain
mean), every other chain 0; a fast even swing (7.5 rad/s) has jerk but
no back-and-forth; an elbow or knee bent 40 deg backward is 30 deg past
its limit (and bent 100 or 90 deg the natural way, none); a hand turned
100 deg is 10 past; both feet pushed back together at 0.6 m/s under
still hips is a 0.6 m/s belt with no skate; a held T-pose is fully
clear with zero contrast.


`rust/core/tests/motion_check.rs`, on the procedural fixture humanoid:

- a still clip: two plants, no skate, no sink or float, nothing inside
  the body, zero jerk;
- the left ankle slid 8 cm by leg IK over 20 frames: one skating plant,
  20 skate frames, worst plant L, slide = 8 cm (1 mm); slid 2 cm instead:
  no skate; moved at 1.3 m/s, 1 cm up: not a plant;
- the hips 4 cm down: every frame sinks, by 4 cm; 20 cm up: every frame
  floats, by 20 cm;
- the right hand driven into the chest and back (the contact pass's own
  test motion): mesh frames inside, the worst at the deepest frame, a
  `.R>` contact;
- the right forearm jittered +-0.02 rad every frame: the arm.R chain's
  peak jerk is exactly `8 L sin(0.02) / dt^3` (L = forearm length; to
  0.1%), its mean half that (the elbow is still), every other chain and
  the root zero;
- a synthetic swing (hand 10 cm back over 10 frames, 40 cm forward in 4,
  still for 10): peak at frames 11-14, load at 9-11, windup about
  0.3 s, strike under 5 frames, hold 9+ frames, hand peak 3.0 m/s;
- a retargeted fixture walk scores the same twice, byte for byte.

## Commands

```bash
motionforge motion check hero.glb                 # table + worst three of each
motionforge motion check hero.glb --clip attack_1 --json
python3 tools/bench/motion_sheet.py strip OUT.png 'a=hero.glb:attack_1@10,11,12!11' --view 3q
python3 tools/bench/motion_sheet.py reel OUT.mp4 'orig=x.glb:attack_1' 'new=x.glb:attack_1_new' --trail DEF-hand.R \
    --prop sword.glb@Socket_Hand_R
python3 tools/bench/motion_compare.py base=before.json new=after.json [--suffix _new]
motionforge motion check hero.glb --camera game --json > check.json   # + read_cam per clip
python3 tools/bench/motion_video.py OUT.mp4 'attack_1=hero.glb:attack_1' 'attack_3=hero.glb:attack_3' \
    --cam game --prop sword.glb@Socket_Hand_R --prop-pose game --marks check.json --cols 2
```

**Videos at the game camera** (2026-11-13; the owner: "Wouldn't video
be better for me to evaluate?"): `motion_video.py` renders each row as
a panel at HLH's follow camera (the numbers above; `--cam lock` is the
locked-on camera over his right shoulder, `--cam close` a 3/4 front
view framing the whole clip with a 34 deg lens), EEVEE with the
character's own materials, a plain floor, one soft sun (55 deg up, 45
deg left of the view, 3 deg wide) and sky light, so the shadow falls
short beside him; `--engine workbench` where EEVEE will not run. Panels
play in step: twice at game speed, then once at half speed (marked
"x0.5 speed"); `--marks` takes `motion check --json` and labels the
load and strike frames. The prop's origin is its grip (the props kit's
convention) and it is held as the game holds the sword (`--prop-pose
game`; `socket` lays the blade along the socket's +Y, which on HLH's
rigs crosses the body). The camera is moved, not the character, to
show him at 2 m (`stature`). Also writes OUT.strip.png; OUT.png with
`--only 0,12` renders stills.

Renders are painted (the character's own colour texture, Workbench
studio light, shadows) since 2026-11-13; `--grey` for the old flat
colour. The floor is a 10 cm grid texture. `--prop` holds a model (the
hero's sword) at a socket. `motion_compare.py` totals each measure over
the set and lists every clip's changes, marking the ones that got worse.

`motion_sheet.py` (stdlib, Blender + ffmpeg) renders frames on one fixed
camera over a 10 cm grid (a sliding foot moves against the grid), with
an optional red trail of a joint's whole path; `strip` tiles chosen
frames per row (marked frames are labelled), `reel` plays rows side by
side and writes a matching 8-frame strip.

## The game's clips (2026-11-12)

The hero's rig with its 31 clips (2.0 m tall, so `k = 1.143`), scored on
f (Linux, release build) in 1 min 39 s, nearly all of it the exact mesh
check: `motionforge motion check <hero rig>.glb`. Frame strips of the
three worst of each measure: `tools/bench/motion_sheet.py strip`
(kept outside the repo with the game's files).

| measure | worst three | what the strips show |
|---|---|---|
| skate | dodge, land, sf_idle (100% of frames) | sf_idle stands sunk to the waist (below); land and dodge drift 7-14 cm over a plant, too small to see at strip scale |
| ground | sf_cut_left 93 cm, sf_cut_right 89, sf_sheathe 88 (all 7 `sf_*` clips 86-93 cm) | real: these clips put the hips about 1 m lower than every other clip (hips local y about -1.0 under a root at +1.0), so the body is in the floor to the waist |
| floating | the enemy's death 41 cm, a second swordsman's death 29 and third attack 23 (every frame of all ten of those two characters' clips) | `worst_float.png`: the retargeted Mixamo clips hold the feet 7-23 cm up; the shadows sit off the feet, consistent with that, but at strip scale not unmistakable |
| through the body | attack_1, attack_2, hit, idle: every frame, 3-5 cm | `worst_through_standing.png`: real (the hand sinks into the belly in hit, the forearm cuts across the chest in attack_1 frame 12 and attack_2 frames 5-6); the sunk `sf_*` clips also score 100% at 13-14 cm, where the strips show little but a torso in the floor |
| jerk | twirl 8,700, attack_1 3,650, attack_2 3,150 m/s^3 (arm.R mean) | fast moves, not noise: the twirl is a spin and the attacks are keyed at 24 fps with sharp corners in the hand's path (visible in the trail) |

Where the numbers mislead:

- **In-place locomotion** (walk, jog, run): until 2026-11-13 these scored
  0 skate (the stance foot slides back faster than the plant cut-off).
  Now measured on their own treadmill (above): walk 74% of frames in a
  skating plant but 5.4 cm in all (worst plant 2.8 cm), jog 58% / 14.4
  cm, run 0. The belt is a median, so a gait whose stance speed varies
  shows a little skate even when it matches the game's speed.
- **Jerk ranks speed** as much as roughness. Compare a clip's jerk with
  clips of its own kind (attacks with attacks), and read the jerk strip
  before calling a clip jittery.
- **Ground and through-body interact**: a clip sunk into the floor puts
  the arms "inside" the thigh and hip mesh too.

