# Make-It-Animatable against our rigger (2026-11-12)

Make-It-Animatable (CVPR 2025; code MIT, weights Apache-2.0) predicts a
fixed 52-joint Mixamo skeleton with three bones a finger, and skin weights,
for humanoids. It was run on the four human reference models from
`tripo-rig.md` and scored with the same checker as our route (skintokens,
then `standardize`, then `weights fix`). Names are left out: the owner's
models are not named in this repo.

## Verdict

- **Body: drop.** Our route scores higher on three of the four models and
  passes the gate on two; the rival passes on none. On three of the four
  `arm_up` tears the rival's underarm where ours holds (13 to 17 failing
  vertices a side on the coated man, where ours has 0 to 7; 4 on the
  fitted woman and on the child, where ours has 0). Its hip joint is sometimes better on adults (smaller crotch strips on
  the fitted woman and the coated man) but goes wrong under belts and
  pouches (123 failing vertices in `hip_forward.R` on the second woman,
  where ours has none) and on the child (27 against 2).
- **Fingers: worth grafting.** Our rig has no finger bones. The rival gives
  all five fingers their own bones and weights, and on these models (which
  do have five modelled fingers) each finger's weight stays on that finger:
  no fused mitten. A spread hand fans cleanly. A full fist (75 degrees at
  every finger joint) folds but crumples into jagged facets at the knuckles
  and tips, because the fingers have few edge loops at the 14,000-triangle
  budget and the finger weights fade well into the palm. Its raw finger
  weights are speckled (13 to 19 `D_NOISE` findings a model) and
  `weights fix` clears them (0 to 2 left).
- **Its crotch doesn't carry over.** Offered to `weights fix` as a
  `--candidate` for our rig, its weights were taken only for some arm
  regions (fitted woman: right forearm 71.6 to 93.3), never a thigh; scores
  moved by 0.1 at most and crotch counts not at all.

## Results

`weights check` of each rig as made ("raw") and after `weights fix`. Ours is
the best seed of `tripo-rig.md`'s run, re-fixed and re-checked with today's
checker (weightforge's stretch line moved from 2.5x to 3.5x on 2026-11-11,
so two of ours now pass). The rival's column is its best of four variants
(below) after fix. Crotch is failing vertices in `hip_forward` left/right,
armpit in `arm_up` left/right, both after fix.

| Model                         | Ours: raw to fixed, failing | Rival: raw to fixed, failing      | Crotch, ours / rival | Armpit, ours / rival |
| ----------------------------- | --------------------------- | --------------------------------- | -------------------- | -------------------- |
| Man, long coat                | 55.0 to 55.9, 2             | 38.5 to 44.1, 5 (no fingers, normals) | 16/20 / 11/14    | 7/0 / 17/13          |
| Woman, shirt and belt pouches | 70.6 to 74.6, 0 (passes)    | 13.8 to 57.8, 3 (no fingers)      | 0/0 / 8/123          | 0/0 / 0/0            |
| Woman, fitted clothes         | 59.7 to 59.8, 1             | 54.6 to 64.2, 2 (no fingers)      | 8/12 / 4/5           | 0/0 / 4/0            |
| Child                         | 61.7 to 69.5, 0 (passes)    | 29.9 to 59.6, 4 (no fingers, normals) | 2/2 / 0/27       | 0/0 / 4/0            |

The rival with its finger bones kept, after fix: 43.3 (5 failing), 39.2
(2), 39.7 (4) and 36.5 (3) in the same order. With fingers kept the body
scores lower too: the model's post-processing spends its four weights a
vertex differently, and on the fitted woman `hip_forward.L` goes from 4 to
61 failing vertices.

The fitted woman is the one model where the rival scores higher (64.2
against 59.8). It also fails one more finding, and the sheet shows a trade:
a smaller crotch strip, a new underarm tear.

Time and hardware: the rival needs CUDA; on the RTX 2060 SUPER (8 GB) it
loads in 4 s and rigs a model in 1.3 to 2.4 s at 2.1 GB of VRAM (2.8 GB
with normals), plus 1.5 to 3.7 s for Blender to write the GLB. `weights fix`
took 4 to 36 s on its no-finger rigs and 50 to 229 s on its finger rigs
(M1 Pro, shared with other jobs), against 2 to 11 s on ours. Ours: 36 to
90 s a seed for skintokens on the M1 Pro, three seeds.

## Method

- Set up on the CUDA machine in its own folder and Python environment
  (torch 2.1.2 + CUDA 12.1, Python 3.11), with a note there recording
  sources, commit, weight checksums and licences. Main (v1) branch only:
  the v2 branch is built on a component whose licence doesn't allow
  shipping everywhere. It runs as a separate program; nothing of it is in
  this repo.
- The repo's demo is a Gradio app. A small runner replays its steps
  (`prepare_input`, `preprocess`, `infer`, `vis` with weight
  post-processing on) using the repo's own functions, and skips the
  pose-to-rest model: the rig is bound in the pose the mesh is in. The
  repo reads its skeleton's parent map from a Mixamo file in a gated
  dataset; the runner gives the standard Mixamo hierarchy instead.
- Four variants a model, two demo settings: "No Fingers" (the demo's
  default; finger weights go to the hand and the finger bones are left
  out) on and off, "Use Normal" (meant for limbs close together) on and
  off. Best per model is kept, as ours keeps its best of three seeds.
- Blender builds the armature from the predicted joints (a bone ends at its
  child, a leaf at its predicted tail) and moves the weights onto the same
  cleaned mesh our route rigs, by exact vertex position.
- `motionforge standardize` reads its Mixamo names as they are: 52 joints
  in, 56 out (the four twist helpers added), all 30 finger bones kept, so
  no converter was needed.
- Each rig: `weights check --json`; `weights fix`; `weights check` again;
  and both checks once more with a pose set holding only `hip_forward` and
  `arm_up` (same specs as weightforge's humanoid set, `--no-clips`) for the
  crotch and armpit counts, since the full report lists only the twelve
  worst poses. `weights compare <ours> <rival> --zoom --clips` per model,
  both after fix: these counts agree with the sheets to within one vertex.
- Hands: Blender close-ups of the rival's fixed finger rigs, at rest with
  each vertex coloured by its finger, in a fist (every finger joint bent
  75 degrees toward the palm, the thumb 45) and spread (index and ring 18
  degrees off the middle finger, pinky 36). The palm side is found from the
  knuckle line, since the hands hang by the thighs and weightforge's
  `finger_curl` direction ("down") is nearly along the fingers there.

The sheets (zoom sheets of ours against the rival's best, against its
finger variant, and the hand close-ups) are kept untracked in the game's
own `tmp/rig-rivals/`.

## Not done

- The finger graft itself: the rival's 30 finger bones and the hand and
  finger weights onto our rig, then `weights fix`. Two things to watch: a
  sibling run found that adding finger bones whose tails point at their
  children can flip weightforge's one-vote guess of how the whole rig's
  bone tails run (and score the thighs wrongly); and a full fist will
  crumple on these meshes whatever the weights, so clips should keep finger
  curls partial or the hands need more edge loops.
- The rival's pose-to-rest model was not used (our models are already in
  an A-pose), nor its Gaussian-splat input.
