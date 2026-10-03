# Licensing: what may train AutoPose, what may ship

Hard constraint from `PLAN.md`, verified at build time (2026-10-01).
The manifest gate (`tools/manifest.py --check`) enforces the training
half; the shipping half is owner procedure (provenance log).

## Training data

| Source | Status | Terms (as verified 2026-10-01) |
|---|---|---|
| CMU Graphics Lab mocap DB | OK | "This data is free for use in research and commercial projects worldwide." Acknowledgment requested: cite `http://mocap.cs.cmu.edu` + "created with funding from NSF EIA-0196217". Not a formal CC0 grant — record the terms text as it stood on the download date in the manifest's `terms_snapshot` field. Fetched via the `una-dinosauria/cmu-mocap` BVH mirror or direct. |
| Owner's own animations/recordings | OK | Owner-authored; no third-party terms. |
| AMASS / anything SMPL-based | BLOCKED | SMPL is non-commercial without a paid license; HumanML3D inherits it. |
| Research-only / NC-licensed data | BLOCKED | Any license with a non-commercial or research-only clause. |
| Mixamo clips | Default EXCLUDE from training | Royalty-free in games (retargeting for in-game use is fine), but training on it needs a terms check first. Pass `--allow-mixamo-training` only after the owner approves. |
| Hunyuan 3D output | BLOCKED everywhere | Territory exclusion (EU/UK/South Korea); reference/placeholder only, never shipped or trained on. |

## Shipping (game content)

- Tripo textures/meshes: shipping assets only while subscribed (paid
  plan = commercial rights); free plan is CC BY non-commercial.
- Suno: commercial rights only for songs made while subscribed
  (Pro/Premier), per `asset-pipeline.md`.
- Steam AI disclosure + provenance log: one row per shipped asset in
  the game's log (see `~/Games/tools/asset-pipeline.md`).

## Tool licenses

This repo: MIT (`rust/`, `python/`, `tools/`, `tests/`, `docs/`),
GPL-3.0-or-later (`blender/`). Content made with the tools is
unaffected. rigforge is GPL (tools only, same deal).
