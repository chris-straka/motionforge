# Pro bench: correctness against professional clips

`tools/bench/pro_bench.py` (stdlib; drives the CLI) scores what the
pipeline does to a clip against the professional clip it came from. The
owner's rule (2026-11-13): the main score comes from professional ground
truth, correctness only. Feet plant when the pro's plant and stay put, the
body is on the floor when the pro's is, arms stay out of the body, joints
stay inside their limits. Style (exaggeration, attack timing,
readability) is never scored against the reference: the pros' style need
not suit our characters.

Source: Quaternius Universal Animation Library 1 and 2 [Standard], the
root-motion GLBs (`UAL1_Standard_RM.glb`, `UAL2_Standard_RM.glb`), CC0 1.0
(License.txt in each zip), from the official free itch.io pages
(quaternius.itch.io/universal-animation-library and -2; no account, no
payment; sha256 pinned in `tools/bench/fetch.sh`). 84 clips (A_TPose
skipped), keyed at 30 fps. Benchmark use only, never training.

## Commands

```bash
tools/bench/fetch.sh                                    # into ~/.cache/forge-bench
python3 tools/bench/pro_bench.py calibrate              # pros on their own mannequin
python3 tools/bench/pro_bench.py run --target hero=RIG.glb --mannequin-f
python3 tools/bench/pro_bench.py report bench/pro/<date>
python3 tools/bench/pro_bench.py why bench/pro/<date> hero clean   # failing clips per check, with numbers
python3 tools/bench/pro_bench.py rescore bench/pro/<date>          # after a measure changed (through-body kept)
# on f, through fleet (stage GLBs stay on f, tables come back):
fleet run --on f --alongside --cores 4 --snapshot -- bash tools/bench/pro_fleet.sh NAME RIG [RIG_JOB]
```

`motion check --tracks` (new) adds per-frame tracks to its JSON: per foot
planted or not, the contact point, its height and its slide step; the
hips; the lowest foot and body points; arms inside the body; degrees past
each joint limit. The bench compares those frame by frame.

## Stages

| stage | what | timing kept |
|---|---|---|
| raw | `animate --no-pin --no-contact` (plain retarget, reference) | yes |
| retarget | `animate` as shipped (feet pinned, contact pass) | yes |
| clean | `clean` on the retarget | yes |
| contact | `clean --contact` on the retarget | yes |
| stylize | `stylize` (body mode) on the clean result, attack clips only | no |

## Checks (per clip and stage, lengths on a 1.75 m body)

| check | passes when |
|---|---|
| plants | planted frames agree with the pro's: IoU per foot >= 0.75 |
| hold | during the pro's plants the contact point moves no more than the pro's own + 2.5 cm per plant |
| skate | skate no more than the pro's + 2.5 cm |
| sink | at most 2 frames with a foot >1 cm under the floor where the pro's is not |
| float | floating frames (whole body off the floor) at most the pro's + 10% of the clip |
| through | arm inside the body where the pro's is not: under 5% of frames |
| limits | a joint 5+ deg further past a limit than the pro's same joint: under 5% of frames |
| rough | back-and-forth (worst chain mean) at most the pro's + 0.5 mm |
| timing | the body's speed curve lines up with the pro's within 1 frame (key poses, its minima, are shallow on loops and idles and moved up to 25 frames for nothing) |
| pose | knee and elbow bends, wrists and ankles within 15 deg of the pro's on average, each joint |

A clip passes a stage when every check passes; the main score is the share
of clips that pass. Stylize retimes on purpose, so it is compared by
totals (defect frames against the pro's) and plants, hold, timing and pose
are not scored there.

## Calibration: the pros on their own mannequin (2026-11-13)

Owner rule: when score and pictures disagree, fix the benchmark first. A
pro clip should show no defect; before this round 71 of the 84 did.

| measure | clips flagged before | after | what was wrong, what changed |
|---|---|---|---|
| back-and-forth (peak > 5 mm) | 53 | 23 | one turn of the acceleration counted, so every fast move's speed peak (a punch speeds up, then stops) read as 100-950 mm. Now a zigzag must turn back twice running. Test: `a_fast_stop_is_not_back_and_forth` |
| joint limits | 58 | 40 | elbows "bent sideways" (36 + 39 clips): where an elbow bends is set by the upper arm's twist, which joint positions do not show and the pros do not keep (bent elbow frames: median 11-19 deg out of the upper arm's plane, 1 in 10 past 50, a best-fit fixed axis still 17-20 deg rms). Dropped. Knees: the hinge now turns with the foot (knee and ankle axes run across the leg together), not with the thigh, whose twist is the hip's rotation: knee-sideways frames 1024 -> 579. Tests: `an_elbow_out_of_its_plane_is_not_a_limit_a_knee_is`, `a_knee_bends_with_its_foot_whatever_the_thigh_twist` |
| floating | 14 | 13 | "both feet 5 cm up" flagged lying and rolling frames (Death01 10 frames). Now the whole body's lowest skinned vertex must be off the floor too. Test: `a_body_lying_on_the_floor_does_not_float` |
| skate | 30 | 30 | see below |
| sink | 18 | 18 | see below |
| through the body | 20 | 20 | see below |
| any | 71 | 57 | |

Tried and dropped: contact points on the sole under the ankle and the
ball, carried by the foot bones (the joint above a sole that rolls
moves). On the pros it was worse (Walk_Loop 5.4 -> 13.9 cm): the pros
pivot the foot about the joints (the walk's heel slap turns the foot
about a still ankle), so the sole under the ankle slides 4 cm per heel
strike and sweeps 2.5 cm into the floor. The joints stay the contact
points. Measuring sink on the mesh (lowest skinned vertex) dropped for
the same reason (sink clips 18 -> 24). Plant rules tried on the tracks
(slide as the furthest point from where the foot went down instead of
the path; a moving foot counted as planted only within 1-2 cm of the
floor) changed the skate count by at most one clip.

What the pictures (`bench/pro/f/pics1/`, painted strips on f) show of
what is left:

- **Skate is real in the pros.** The walk's toe slides 4-5 cm forward on
  the floor at toe-off (frames 21-22: the ball joint at floor height moves
  2.5 + 1.4 cm); at strip scale the foot reads planted. The crouch walk
  drags the leading foot along the floor (its ball's trail is a flat line,
  15-22 cm at 0.68 m/s). Sword attacks slide-step (Sword_Regular_A's
  front foot moves 7 cm at 0.43 m/s, 0.5 cm off the floor). Idle_FoldArms
  wobbles its flat foot 0.3-0.4 cm a frame (9.6 cm of path, 4.6 cm at the
  most from where it stood). These are the pros' own foot slides; a
  smaller one than 5 cm is not visible at strip scale.
- **Sink and float are mostly no-floor clips:** the swims (the body is
  under the water line, 29 and 129 cm), Jump_Loop and the NinjaJump clips
  (played in the air), ClimbUp_1m (ends on a ledge 1 m up); the rest are
  1-3 cm dips of the ball joint as the toes press down (sword combos,
  slides, the zombie walk).
- **Through the body is resting contact:** the sitting clips' hands rest
  on the thighs (fingers 7 cm into the low-poly thigh), the kneel rests a
  forearm on the knee, the folded arms touch the chest (2 cm).
- **Limits left:** knees sideways in deep lunges and kneels (Sword_Regular_A_Rec
  frame 0, Fixing_Kneeling), which read as plausible poses:
  the knee measure is still strict where the shin turns under a deeply bent
  knee (tibial rotation grows with flexion). Elbow "straighter than
  straight" on whip strikes (TreeChopping 56 deg past, OverhandThrow,
  Sword_Attack): style (snap), not scored against the pro.

So the absolute score cannot reach zero on the pros without blinding it to
real slides; the bench therefore scores every stage against the pro's own
numbers, frame by frame.

## First run: every UAL clip onto the hero rig and the female mannequin (2026-11-13)

On f through fleet (`pro_fleet.sh`), 2 h 41 min on a loaded machine,
nearly all of it the exact through-body check on the hero's mesh. Commit
445df22 (before the two fixes below; checks then: keys instead of timing,
no pose check). Clips passing every check, of 84 (stylize: 5 attack
clips):

| stage | hero rig (2.0 m) | female mannequin (1.81 m) |
|---|---|---|
| raw | 6 | 53 |
| retarget | 21 | 73 |
| clean | 35 | 72 |
| contact | 35 | 73 |
| stylize | 0 of 5 | 0 of 5 |

Hero rig, clips failing each check:

| stage | plants | hold | skate | sink | float | through | limits | rough | keys |
|---|---|---|---|---|---|---|---|---|---|
| raw | 8 | 34 | 37 | 44 | 3 | 69 | 35 | 0 | 3 |
| retarget | 7 | 5 | 10 | 44 | 3 | 2 | 36 | 8 | 11 |
| clean | 8 | 1 | 0 | 0 | 4 | 2 | 38 | 9 | 14 |
| contact | 8 | 1 | 0 | 0 | 4 | 1 | 38 | 9 | 14 |
| stylize (5) | - | - | 0 | 0 | 1 | 3 | 5 | 2 | - |

Where each pass helps and hurts (hero rig):

- **`animate`'s pinning and contact pass** (retarget against raw): fixes
  72 clips' checks, breaks 16. Arms through the body 2200 -> 65 frames (69
  clips -> 2), hold 34 -> 5, skate 37 -> 10. Cost: back-and-forth on 8
  clips (Hit_Chest 0 -> 1.8 mm, Hit_Knockback 1.3 -> 2.7, ClimbUp_1m 0.6
  -> 2.2, Chest_Open, Sitting_Exit, Pistol_Reload, Swim_Fwd_Loop,
  Sword_Block).
- **`clean`** (against the retarget): fixes 46, breaks 7. Every sink gone
  (44 clips, 1017 frames: the retarget leaves the hero's feet 2-11 cm
  under the floor), skate 10 -> 0, hold 3.7 -> 1.5 cm a plant (the pros'
  own: 3.3). Worse on: Crouch_Fwd_Loop (ankle past its limit on 16
  frames: the lock holds the dragged foot), Slide_Exit (ankle, 1 frame),
  Sword_Regular_B (plants IoU 0.41, 7 more floating frames: the whole-clip
  floor offset), Jump_Start (back-and-forth 0 -> 2.3 mm).
- **`clean --contact`** (against the retarget): same clips as `clean`;
  through-body frames 70 -> 55, limit frames +6. No clip changes side:
  `animate` had already run the contact pass.
- **`stylize`** (attacks, against clean): none of the 5 passes. Punch_Cross
  and Punch_Jab gain through-body and limit frames, the two sword combos
  back-and-forth (4.7 mm against the pros' 2.5) and through-body frames.

On the mannequin (same proportions and rest pose as the source) the
retarget passes 73 of 84 and the limits check fails 0-1 clips, against
35-38 on the hero rig. That gap was the benchmark's and the retarget's,
not the motion's:

1. **A retarget bug.** A hand whose fingers the two rigs do not share by
   name got no rest alignment, so it kept the source's rest direction.
   The hero rig rests with its arms down and has one finger bone; the
   pros rest in a T-pose: every clip bent the hero's wrists by the
   difference. Wrist bend against the pro's on four clips (Idle_Loop,
   Push_Loop, Sitting_Talking_Loop, Sword_Idle): 62-70 deg off before,
   5-8 after (`animate::rest_alignment`: toward the hand's own children,
   else as its forearm; test `retarget_aligns_a_hand_without_shared_
   fingers`). The hand sockets take the same alignment, so the sword's
   grip changes with it: the owner should see it.
2. **Limits read each rig's rest pose and bone roll.** The knee hinge
   carried by a bone read 13-17 deg more sideways on the hero (it stands
   toed out) than on the pros' mannequin for the same joint positions;
   wrists and ankles were rotations from the rest pose. Now from joint
   positions alone: the knee's hinge is across the shin and the foot's
   direction, wrists and ankles are the bend between the two bones
   against the same bend at rest. Knee bend and tilt agree to 0.1 deg
   between a pro clip and its raw retarget; ankles within 5.

The run with both fixes (and the timing and pose checks) is fleet job
`5je56jqj` on f: stage GLBs in `~/.cache/forge-bench/pro/2026-11-13b/run`,
tables in its artifacts. Its numbers replace the hero column above.

## Open

- Knee limits still flag the pros' deep lunges and kneels (tibial
  rotation grows with flexion).
- The pros' own foot slides (2.5-9 cm in 30 of 84 clips) are scored as
  skate by `motion check`; the bench scores against them instead.
- The whole-clip floor offset in `clean` on a clip with few grounded
  frames (Sword_Regular_B).
- MPFB2 bodies as targets (the weightforge corpus was not built on f).
- Pictures of the pros' flags: `bench/pro/f/pics1/` (not tracked).
