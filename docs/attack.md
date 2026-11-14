# Attack authoring: from the sword's path backward

`motionforge attack --rig hero.glb --preset cleave|backcut|sweep|thrust|all
--output out.glb [--spec my.json] [--name clip] [--prefix p] [--report r.json]`
(`rust/core/src/attack/`). It adds each attack to the rig as a clip.

## Why

The owner could not read the game's attacks ("I have no idea what he's
doing"). The same keys read the same on the old rig, today's hero and two
professionally rigged bodies, so the fault was the clips: an arm moving
over an upright body. Polishing those keys (stylizer rounds 1-3) did not
change that. This tool authors an attack the other way round: where the
sword goes first, the body solved under it.

## What professional attacks do (measured)

Six CC0 sword attacks (Quaternius Universal Animation Library 1 and 2,
benchmark use only), retargeted exactly onto the hero and read at the
game camera (`motion check --camera game`) and from their joints, beside
the game's `attack_1` and `attack_3`:

| | pro clips (6) | game attack_1 / attack_3 |
|---|---|---|
| hips turn through the strike | 60-120 deg | 0 |
| chest twist on top of that | 30-60 deg | 50 / 0 |
| lean into the cut | up to 30-57 deg | 0 / 12 |
| hips drop (body heights) | 0.15-0.2 | 0.03 / 0.09 |
| feet apart at the strike (heights) | 0.5-0.95 | 0.24 / 0.39 |
| strike, load to fastest frame | 2-5 frames | 3 |
| tip speed | 40-50 m/s | 40-45 m/s |
| body change at the game camera | 0.56-0.88 | 0.29 / 0.40 |
| readability score | 0.41-0.67 | 0.42 / 0.49 |

The sword already moved as fast and as far as a pro's. The body did not
take part. On the correctness measures the pro clips are not clean
either (back-and-forth 4-32 mm, 10-107 frames past a joint limit, 3-51 cm
of skate, up to 30 frames with an arm in the body), which is the range an
authored attack is judged against.

## A spec

`motionforge-attack` v1, JSON: 4-8 keys. Lengths are body heights in the
character's frame (x to his left, y up, z forward, from the floor under
his standing hips); angles are degrees, a positive turn is to his left. A
key inherits what it leaves out from the key before.

- `grip`, and `tip` or `blade` (a direction): where the sword is.
- `hips`: `turn`, `drop`, `shift` [left, forward] (weight shift).
- `torso`: `twist` (chest past the hips), `lean`, `side`.
- `feet`: `L` / `R` as [left, forward, turn, heel]: where the ball of the
  foot is from standing, the foot turned about its ball.
- `off`: the free hand: `free` (out for balance), `grip` (on the hilt), or
  a point.
- `t` seconds to arrive, `hold` seconds held, `ease` (`in` for a strike
  into contact, `out` for a follow-through, `inout`, `linear`).

## How a frame is made

Nothing is keyed as bone rotations. Every frame is solved from the
interpolated key values:

1. Hips placed and turned; the spine twisted and leaned bone by bone;
   the head kept toward the target (never more than 70 deg off the chest).
2. Each foot planted by its ball where the key stands it, with the game's
   two-bone solver mirrored (`attack/ik.rs`: `two_bone`, `soft_reach` from
   `hlh/bevy/src/ik/solve.rs`), written as a true hinge and aimed again
   until the foot is on its point. The knee tracks the toes. The hips come
   down by exactly what the legs need to reach. A foot whose place changes
   steps (it lifts); one that only turns pivots on its ball, heel just up;
   a heel comes up when the ankle would fold more than 45 deg.
3. The sword hand turned so the blade, held as the game holds it in the
   hand socket (`tracks::held_pose`), runs from the grip to the tip with
   the wrist as straight as that allows (50 deg at most; past that the
   blade gives, and the report says by how much); the forearm takes the
   hand's twist up to 80 deg; the arm solved to put the grip on its point.
4. The free hand to its place.

Between keys the body values ease; the grip follows a Hermite curve
through the keys (stopping at a held key or a turn-round, passing through
a contact key at speed); the blade turns along the great arc between its
directions (a `swing` key in the middle of a strike says which way round).

## The presets, on the hero (2026-11-13)

`presets/attacks/*.json`. Game camera (3 m behind, 18 deg down), the
hero's best body, read by `motion check --camera game`:

| clip | readability | body change | tip arc (body heights x one-way share) | skate / sink / float | frames past a limit | through body |
|---|---|---|---|---|---|---|
| sweep (level slash, right to left) | 0.78 | 0.69 | 2.25 x 1.00 | 0 / 0 / 0 | 9 (wrist 12 deg) | 0 |
| cleave (forehand diagonal) | 0.68 | 0.79 | 2.31 x 0.69 | 0 / 0 / 0 | 11 (back ankle 18 deg) | 1 fr, 6 cm |
| backcut (backhand diagonal) | 0.65 | 0.74 | 2.22 x 0.65 | 0 / 0 / 0 | 2 | 11 fr, 9 cm |
| thrust (lunge) | 0.32 | 0.64 | 0.38 x 1.00 | 0 / 0 / 0 | 8 | 1 fr, 6 cm |
| game attack_1 | 0.42 | 0.29 | 1.82 x 0.88 | 12 cm / 4 cm / 0 | 0 | 11 fr, 3 cm |
| game attack_3 | 0.49 | 0.40 | 1.93 x 1.00 | 28 cm / 10 cm / 0 | 0 | 3 fr, 2 cm |
| key-pose cut (round 2) | 0.52 | 0.81 | 1.84 x 0.94 | 13 cm / 1 cm / 0 | 15 | 10 fr, 8 cm |
| pro clips (6) | 0.41-0.67 | 0.56-0.88 | 1.8-2.9 | 3-51 cm | 10-107 | 0-30 fr |

Each is 1.1 s: a coil (the back foot steps back), a windup held 2 frames
(the front foot steps out), a strike of 4 frames over planted feet, a
follow-through held 5 frames, a recover in two steps.

Where the numbers fall short:

- **thrust** scores low by design of the measure and of the view: a blade
  pointing away from a camera behind him sweeps nothing. It reads by the
  lunge. An attack for this camera should cut across it.
- **backcut**: the sword arm crosses the chest on the way to the left
  shoulder (11 frames, 9 cm), and its grip is 2% short of two keys.
- The back ankle in **cleave** is past its limit for 11 frames (the leg
  stretched out behind over a flat foot).
- The readability score is not yet checked against the owner's own picks.

## Round 2 (2026-11-13): the grip, weight, phone clips

The owner on round 1: "clearly sword attacks. They need refinement. And
he's not holding the sword properly (maybe not gripping it at all)".

**The grip** (`attack/grip.rs`). The game's hold constants
(`tracks::held_pose`) put the handle across the back of the wrist on a
hero with real hands, the hand open. On a hand with finger bones the
hold now comes from the hand: the handle across the palm under the
knuckles, from the little finger's side out past the index (a hammer
grip, canted 0.25 toward the fingers), a handle's and a finger's
thickness to the palm side; the fingers close to 0.9 of the fist
(`--grips` reads a hand-tools `grips.ron` for the fist's bend per joint:
the starred hero's); the thumb comes round. Closing "until a finger
touches the handle" stopped at once (straight fingers already lie within
touching distance of a handle under the knuckles), so the default is the
fixed share, checked close up on the hero (`--cam hand`). The hand socket
is rewritten into the palm (origin at the grip's middle, +Y the blade),
so a sword hung on it with no offset sits in the fist (`motion check
--blade-pose socket` reads it that way). A hand without finger bones
keeps the game's hold and its socket (test). The free hand curls 0.35
of a fist, 0.85 on the hilt.

**Weight: the hips lead, the blade lags, then snaps.** Each move's keys
are reached on three clocks: the hips (and feet) finish 35% early, the
chest 20% early, the sword starts 20% late and catches up; all land on
the key together (test). Measured frame by frame against the six pro
attacks (the frame each part turns fastest, against the tip's fastest):

| | hips fastest, frames before the tip | hips turn speed | still after the strike |
|---|---|---|---|
| pro clips (6) | 0-2 (one 11) | 700-1,700 deg/s | 1-15 frames |
| round 1 (no overlap) | 0 | 510-650 deg/s | 4-5 |
| round 2 | 1 | 730-1,060 deg/s | 7 |

The hips turn further through the strike (windup 45-60 deg the other
way, follow-through 55-65 deg), the follow-through holds 0.3 s (pros:
up to 0.5 s), the recovery ends on a 0.1 s hold in the ready pose, and
the ready pose holds the sword at the waist pointing forward (it had
stood up in front of his face once the palm turned the blade).

| clip (round 2, starred hero) | readability | body change | tip arc | skate / sink / float | through body | past a limit |
|---|---|---|---|---|---|---|
| sweep | 0.78 | 0.72 | 2.25 x 1.00 | 0 / 0 / 0 | 0 | 22 fr (wrist 26 deg) |
| backcut | 0.75 | 0.75 | 2.21 x 0.89 | 0 / 0 / 0 | 8 fr, 11 cm | 19 fr (wrist 30 deg) |
| cleave | 0.72 | 0.79 | 2.32 x 0.78 | 0 / 0 / 0 | 1 fr, 2 cm | 17 fr (back ankle 24 deg) |

The palm grip bends the wrist further than the old hold: the wrist limit
is passed on 17-22 frames per attack (the pros: 10-107 frames). By eye
(close on the hand) the hand reads as gripping through the swing.

**Clips the owner can read on a phone** (`tools/bench/phone_clip.py`,
the standard for every motion clip sent to the owner): 720x1080, the
figure fitted to about 80% of the height over the whole move (the blade
may leave the frame at full stretch), the close 3/4 view first, then the
game camera's angle (`--cam gamefit`: its direction and pitch, moved
in), each once at speed and once at 0.4 speed, labelled.

## Tests

`core/tests/attack.rs` on the procedural fixture: the grip is on each
key's point (1e-6 of the stature) and the blade along its direction, or
off by exactly what the report says; a foot that does not move keeps its
ball within 1e-9 while the hips drop, shift and turn over it; a foot
turned 30 deg pivots about its ball; a stepping foot lifts; every preset
builds twice to the same bytes; building again replaces the clip; specs
inherit and refuse nonsense. `attack::ik` unit tests: the knee goes to the
pole's side, a target out of reach is never stretched to.

## Commands

```bash
motionforge attack --rig hero.glb --preset all --output hero_attacks.glb --report attacks.json
motionforge motion check hero_attacks.glb --camera game --attack-clips cleave,backcut,sweep
python3 tools/bench/motion_video.py out.mp4 'cleave=hero_attacks.glb:cleave' --cam game \
    --prop sword.glb@Socket_Hand_R --prop-pose game
```
