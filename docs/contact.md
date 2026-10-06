# Contact pass, hand sockets, grip and clip picks

Built 2026-10-06 for the HLL character chain (genforge `gen character`,
stage `animate`). Code: `rust/core/src/contact.rs`, `animate.rs`
(`retarget_picked`, `sync_sockets`), `standardize.rs` (sockets),
`glb.rs` (`compact`); CLI `animate`, `grip`; renderer
`blender/tools/clip_sheet.py`.

## Why

FK clips know nothing about the body they play on. A swing keyed or
captured on a slim rig cuts through a broad character's chest, and a
library clip's arms hang inside a bulkier hero's hips. The pass fixes
that per character, at retarget time, so every clip from every source
gets it and the game needs no code.

## What runs in `animate` (default on; `--no-contact` skips it)

1. **Proxies**, fitted to the character's skinned rest mesh (no tuning
   per character): per spine segment a lozenge of two capsules (depth
   sets the radius, extra width the side offset), a head capsule, thigh
   capsules; per arm a forearm and a hand capsule; on the weapon hand a
   blade capsule from `Socket_Hand_R` along its +Y (`--weapon-length`,
   in character heights, default 0.75: HLL's sword tip sits 1.5 units
   from the grip on the 2.0-unit Andras). Body surface points (torso,
   head, thigh vertices no arm bone moves) catch shallow contacts the
   capsules miss.
2. **Per frame**: find the wrist offset that clears every contact (push
   along the deepest contact's normal, solve two-bone arm IK, measure
   again; the shoulder stays put, the elbow keeps its bend plane, the
   hand keeps its world rotation). A blade that cuts the body is swung
   clear about the wrist first (up to 40 deg), then the wrist moves.
   Caps keep the swing's intent: the wrist moves at most 12% of the
   character's height.
3. **Over time**: corrections are smoothed with a max envelope and a
   smoothstep falloff (`--contact-ramp`, 0.15 s), so a one-frame contact
   eases in and out; timing and the arc's shape are untouched where
   nothing touches.
4. **Exact check**: forearm and hand vertices (and blade samples) are
   tested against the body mesh with the generalized winding number
   (Jacobson et al. 2013; robust to the open cuts at neck, shoulders and
   hips). An arm that still cuts the mesh is re-solved from the original
   with fatter proxies (x1.15 ... x3) and the best result is kept.
5. Pairs that already touch in the rest pose keep that depth as slack.

Weapon proxies apply to clips whose name contains
attack/sword/slash/combo/block/parry/stab/swing (`--weapon-clips`).

Each clip's report (`RESULT.json` `animate.clips[].contact`) lists per
arm: frames in contact and deepest contact before/after (proxies and
mesh), the worst frame's time, frames changed, the largest wrist shift
and blade turn, and which proxies touched. `--proxies FILE` writes the
fitted capsules.

## Hand sockets and grip

`standardize` adds `Socket_Hand_L/R` (plain nodes under the hands, not
joints): 45% along the hand, blade (+Y) across the palm from the little
finger's knuckle to the index finger's when the rig has fingers, else
the character's front (the hand-built Andras rule). The HLL game holds
its sword at `Socket_Hand_R`.

A clip set implies a grip: `motionforge grip --input LIB.glb` scores 98
blade directions over the library's own weapon clips (blade points inside
the mesh or below the floor are hits; fewer hits win, then more
clearance) and writes the winner into the library's socket. `animate`
then turns the target's sockets to match the clips' source
(`S_t = align^-1 * yaw * S_s`, the same rest transfer the hand gets), so a
sword sits in the hand the way the animator's prop did.

## Clip picks and compaction

`animate --pick "lib.glb:Sword_Regular_A=attack_1,Idle_Loop=idle"`
chooses, renames and orders clips (a bare name keeps its name).
`standardize` and `animate` drop the buffer data of replaced animations
(`Document::compact`): the CC0 mannequin standardizes from 7.6 MB to
0.6 MB.

## Measurements (2026-10-06, M4)

Tests: `cargo test --locked --release` = 75 core unit + 4 contact
(`core/tests/contact.rs`: a hand driven through the chest is cleared and
eased, clean clips are untouched, sockets, picks) + 13 rig-adapter +
3 adapter contract + 10 CLI goldens.

Andras after the weights gate (genforge rehearsal, SkinTokens rig,
4,459 verts), the 13-clip CC0 library (`adapter animate`, contact on):
22 clip-arms cut the mesh before (1 to 76 frames, up to 12.3 cm deep);
21 are clear after; attack_2 keeps one blade frame at 2.0 cm (wrist at
its cap). On the 11 extra clips, attack_dash keeps one frame at 2.4 cm.
The pass with its exact mesh check takes about 15 s for the 13 clips on
the M4 (most of it the winding-number check).

| clip | arm | frames inside | deepest | after | wrist moved |
|---|---|---|---|---|---|
| attack_1 | L | 7 | 11.0 cm | 0 | 11.9 cm |
| attack_2 | R | 2 | 6.6 cm | 1 (2.0 cm) | 23.9 cm, blade 32 deg |
| attack_3 | R | 27 | 7.5 cm | 0 | 23.9 cm |
| idle | R | 76 | 8.3 cm | 0 | 10.1 cm |
| walk | R | 32 | 12.1 cm | 0 | 13.5 cm |
| hit | R | 11 | 12.3 cm | 0 | 14.4 cm |

(full list in `RESULT.json`). The library's slim mannequin arms hang
inside Andras's hips; the pass opens them about 10 cm.

Grip on the CC0 library's eight sword clips (209 frames): the
knuckle-line socket hit the body 42 times, the found grip 0.

## Limits

- Hands, forearms and one held weapon against torso, head and thighs.
  Upper arms (the armpit), the legs and the other arm are not proxies.
- Locomotion clips with a drawn weapon get no blade proxy unless their
  names match `--weapon-clips`.
- The deformation gate is separate: weightforge's recheck measures skin
  stretch/volume in the same clips (see the README's genforge note).
