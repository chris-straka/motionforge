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

Clips are sampled at 30 fps whatever rate they were keyed at (`--fps`
to change), the rate the renders use, so frame numbers match the
pictures. Output is deterministic.

## Tests with known answers

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
python3 tools/bench/motion_sheet.py reel OUT.mp4 'orig=x.glb:attack_1' 'new=x.glb:attack_1_new' --trail DEF-hand.R
```

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

- **In-place locomotion** (walk, jog, run): the clip
  carries no travel, so a planted foot slides backwards at walking speed.
  That is faster than the plant cut-off (0.75 m/s), so it is a step, not
  a skate, and those clips score 0. Skate on them needs the game's travel
  speed added; not done.
- **Jerk ranks speed** as much as roughness. Compare a clip's jerk with
  clips of its own kind (attacks with attacks), and read the jerk strip
  before calling a clip jittery.
- **Ground and through-body interact**: a clip sunk into the floor puts
  the arms "inside" the thigh and hip mesh too.

