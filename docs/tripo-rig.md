# From a generated model to a rigged GLB: `tools/tripo_rig.py`

One command takes a generated model (a Tripo FBX, or a GLB or OBJ) to a
rigged, skinned GLB with clips, a pose sheet and a report:

    python3 tools/tripo_rig.py MODEL.fbx OUT_DIR --name NAME
        [--kind auto|human|creature] [--class hero|npc|monster]
        [--seeds 1,2,3] [--clips LIBRARY.glb --pick walk=walk,attack_1=attack]

It needs `skintokens`, `weights` (weightforge), `rfcheck`, this repo's
`motionforge` and Blender; each is found by flag, environment
(`SKINTOKENS_BIN`, `WEIGHTS_BIN`, `RFCHECK_BIN`, `MOTIONFORGE_BIN`,
`BLENDER`), `PATH`, then the sibling repo's `target/release`. Exit 0: the
best try passes weightforge's gate (no failing finding) and rfcheck's
contract. Exit 1: it doesn't; the GLB is still written and `report.json`
says what failed. Exit 2: a step could not run.

## The steps

1. **Clean** (`tools/tripo_to_glb.py`, Blender): one mesh, on the ground
   at the origin, facing glTF +Z.
   - Facing is measured (`tools/facing.py`): the best upright mirror plane
     gives side to side; a biped's toes lead its ankles, a quadruped
     carries its head end higher and heavier. A weak measurement (a cloak
     to the ground, a weapon in one hand) gives way to the source's
     convention: Tripo's exports face +X. Rigged sideways, a humanoid's
     arms split front and back and skintokens returns a rig without its
     core bones.
   - Holes are counted, and those with a rim no longer than 8% of the
     model's largest side are filled, the new faces taking the colour
     beside them. Longer rims are taken for openings that are meant and
     left; the report lists found, filled and left.
   - Triangles are cut to the class budget (hero 14,000, npc 7,500,
     monster 18,000), keeping UVs. Textures leave as PNG, 1024 a side at
     most: Tripo's are JPEG, which the glTF exporter passes through and
     the game's Bevy build doesn't decode (rfcheck `B_IMAGE_CODEC`).
2. **Rig**: `skintokens rig`, once per seed.
3. **Standardize** (`motionforge standardize`), then clips:
   - a human gets the canonical names, twist helpers and hand sockets, and
     with `--clips` its clips retargeted from a library (`animate`);
   - a four-legged creature keeps its skeleton. `creature-profile` reads
     it, `creature-gait` names its parts (four legs, spine, neck, tail),
     makes a walk and a lunge on it (there is no four-legged clip library
     to retarget from) and writes its range of motion as a weightforge
     pose set.
4. **Fix and gate**: `weights fix`, then `weights check`. A human is posed
   through weightforge's humanoid set, a creature through its own set,
   both through their clips.
5. **Contract**: `rfcheck --class`.

The best try is the one that passes, else the one with a clean contract
and the fewest failing findings, then the highest score.

## What it does on real models (2026-11-11)

Eight Tripo exports, about 95,000 triangles each; three seeds; skintokens
on an M1 Pro (Metal), 36 to 90 s a rig. Names are left out: the owner's
models are not named in this repo.

| Model                         | Holes          | Raw  | After fix | Failing findings              |
| ----------------------------- | -------------- | ---- | --------- | ----------------------------- |
| Woman, fitted clothes         | 22             | 59.7 | 59.9      | 2 (one thigh, one upper arm)  |
| Woman, shirt and belt pouches | 8              | 70.6 | 80.0      | 3 (both thighs, an upper arm) |
| Child                         | 19             | 68.3 | 68.4      | 3                             |
| Man, long coat                | 17             | 55.0 | 55.9      | 6 (thighs, arms, cuffs)       |
| Wolf, long fur                | 35 (24 filled) | 45.3 | 62.8      | 9                             |
| Wolf, ragged fur              | 54 (32 filled) | 41.1 | 61.5      | 4                             |
| Horse, saddled                | 23 (17 filled) | 55.6 | 61.2      | 7                             |
| Cloaked figure                | 36 (29 filled) | none | none      | no rig (below)                |

The four humans were run before the hole filling was added, so theirs
were only counted.
None passes the gate. For scale, the hero rig the game shipped with scores
6.8 with about twenty failing findings on the same check.

What the failures are:

- **Humans: the same few vertices every seed.** Each model fails at the
  crotch in `hip_forward` (the strip between the legs, 10 to 54 vertices
  of about 7,000) and at an armpit in `arm_up` or `arm_forward` (4 to 6).
  Three seeds of the first model score 59.9, 59.1 and 59.6 with the same
  two findings, and no `weights fix` method or candidate
  (`--all-regions`, `--method optimize|smooth|geodesic`, `--skintokens`)
  changes them. A generated model is one fused shell (15,963 of 16,004
  vertices in one piece on the first): the legs meet in a bridge of skin
  two or three centimetres wide, and a 90 degree hip turn stretches it
  past the 2.5 times limit whatever the weights. A coat or a pouch across
  the hip adds its own.
- **Standardize first.** Without it the unnamed finger and hair joints
  stay in the skin and score as noise: the same four humans at 24,000
  triangles, checked straight off skintokens, scored 16.9 to 50.3 with
  10 to 16 failing findings.
- **Creatures were judged on the wrong poses.** With no pose set of its
  own a creature got weightforge's fallback, every bone bent 45 degrees
  each way and twisted: 24.3 with 13 failing findings for the first wolf,
  62.8 with 9 on its own range of motion.
- **`creature-profile` failed on every one** ("non-finite float in
  output"): `drop_debris` counted each UV island as a loose part and
  dropped the whole mesh. Fixed (parts are joined by position first).
- **The cloaked figure has no rig.** `skintokens rig` went past 12 GB of
  memory on it within seconds, twice (fleet stopped both jobs at 14.3
  and 13.7 GB; the other seven models rig in about 3.5 GB). Why is not
  found. The route for it is open: rigforge's fitter, or a body under
  the cloak.

## Tripo's own Auto Rig, scored (2026-11-11)

Three of the same models rigged by Tripo Studio's Auto Rig (20 credits
each; humans on its Mixamo preset, the wolf on "Other") and scored with
the same check. Each export was welded, turned to face +Z from its own
foot bones, then standardized; "at our budget" is the same triangle cut
as our route, made after their rig.

| Model                         | Tripo, as exported | Tripo, at our budget, raw to fixed | Ours, raw to fixed      |
| ----------------------------- | ------------------ | ---------------------------------- | ----------------------- |
| Woman, fitted clothes         | 38.9, 10 failing   | 13.0 to 15.1, 7 failing            | 59.7 to 59.9, 2 failing |
| Woman, shirt and belt pouches | 60.9, 12 failing   | 9.2 to 44.9, 2 failing             | 70.6 to 80.0, 3 failing |
| Wolf, long fur                | not scored         | 27.8 to 38.5, 16 failing           | 45.3 to 62.8, 9 failing |

- It doesn't pass the gate, and by score our route leads on all three.
  On the full mesh it fails where ours does (the thighs and spine in
  `hip_forward`, the arms in `arm_up`), which fits: the geometry is
  what's fused.
- Pose by pose it is closer than the scores (`weights compare --zoom`,
  both at our budget, failing vertices over every pose). First woman:
  ours 103, Tripo's 83; its crotch is the better one (9 against 49 in
  `hip_forward.R`), and its arms drag a dagger on the belt with them (27
  in `arm_up.R`, 22 in `arm_forward.R`, where ours has none). Second
  woman: ours 30, Tripo's 9, with no crotch failure at all against our
  11. Wolf: ours 136, Tripo's 565. The score counts every vertex's
  deformation, the speckled finger weights included, not only the
  vertices past the fail line; that is where Tripo's humans lose it.
- Its humans keep finger bones (65 joints against our 26 after
  standardize), so they are also posed through finger curls, and fail
  some. Cutting triangles after rigging speckles the finger weights,
  which is most of the low raw scores at our budget; our route cuts
  first.
- Its wolf skeleton has 35 joints and no left-right mirror among its 22
  side joints; `creature-profile` still finds four legs.
- What it adds: fingers on a Mixamo skeleton for a few credits. What it
  changes in the plan: nothing.

## Not done

- Nothing passes the gate. The crotch and armpit strips need either
  geometry that isn't fused there (a model generated with the legs apart
  and the arms away, or a wrap onto a base body: `wrapforge`) or a fix
  candidate made for a fused crease.
- The creature clips are forward kinematics: feet slide, and the lunge is
  one shape for every body.
- Faces and fingers: standardize folds a generated hand's fingers into the
  hand; nothing here rigs a face.
