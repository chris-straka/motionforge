#!/usr/bin/env python3
"""Dataset manifest gate (stdlib only).

A manifest is the ONLY form in which mocap enters this repo: clip id +
source + license + checksum, never the motion itself. Training runs
record the manifest's sha256 in their run log.

Manifest format (paths relative to the manifest file):
    {"format": "motionforge-dataset", "version": 1, "clips": [
      {"id": "cmu_01_01", "path": "cmu/01_01.json",
       "source": "cmu", "license": "cmu-mocap",
       "terms_snapshot": "free for research+commercial (mocap.cs.cmu.edu, read 2026-10-01)",
       "split": "train", "sha256": "..."},
      {"id": "owner_attack_01", "path": "owner/attack_01.json",
       "source": "owner", "license": "owner-recorded", "split": "heldout",
       "sha256": "..."}]}

Usage:
    python3 tools/manifest.py --check data/manifest.json [--allow-mixamo-training]

Exit 0 when every clip is licensed-clean and every file verifies.
"""

import argparse
import hashlib
import json
import os
import sys

# Licenses allowed for training. cmu-mocap additionally requires a
# terms_snapshot (terms text as it stood on the download date).
ALLOW = {"cmu-mocap", "owner-recorded"}

# Explicitly blocked: non-commercial or research-only lineage, plus the
# Hunyuan territory exclusion. Anything not in ALLOW is rejected too;
# this set exists to name the known traps in error messages.
BLOCKED = {
    "amass", "smpl", "smpl-x", "humanml3d", "research-only",
    "non-commercial", "hunyuan", "mixamo-training",
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_manifest(path, allow_mixamo_training=False):
    """Returns (ok, lines). ok False on any violation or mismatch."""
    lines = []
    ok = True
    try:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, ValueError) as exc:
        return False, [f"ERROR: cannot read manifest: {exc}"]
    if doc.get("format") != "motionforge-dataset" or doc.get("version") != 1:
        return False, ["ERROR: not a motionforge-dataset v1 manifest"]
    clips = doc.get("clips", [])
    if not clips:
        return False, ["ERROR: manifest has no clips"]
    base = os.path.dirname(os.path.abspath(path))
    seen_ids = set()
    for i, clip in enumerate(clips):
        tag = f"clip {i} ({clip.get('id', '?')})"
        cid = clip.get("id", "")
        if not cid or cid in seen_ids:
            lines.append(f"{tag}: ERROR: missing or duplicate id")
            ok = False
            continue
        seen_ids.add(cid)
        lic = clip.get("license", "")
        if lic == "mixamo-training" and allow_mixamo_training:
            pass
        elif lic not in ALLOW:
            why = "explicitly blocked" if lic in BLOCKED else "not in the allowlist"
            lines.append(f"{tag}: BLOCKED: license '{lic}' {why} (see docs/licensing.md)")
            ok = False
            continue
        if lic == "cmu-mocap" and not clip.get("terms_snapshot"):
            lines.append(f"{tag}: ERROR: cmu-mocap needs a terms_snapshot")
            ok = False
            continue
        rel = clip.get("path", "")
        full = os.path.normpath(os.path.join(base, rel))
        if not rel or not os.path.isfile(full):
            lines.append(f"{tag}: ERROR: file missing: {rel}")
            ok = False
            continue
        want = clip.get("sha256", "")
        got = sha256_file(full)
        if want != got:
            lines.append(f"{tag}: ERROR: sha256 mismatch (manifest {want[:12]}.., file {got[:12]}..)")
            ok = False
            continue
        split = clip.get("split", "train")
        lines.append(f"{tag}: OK [{lic}] split={split} sha={got[:12]}")
    return ok, lines


def main(argv):
    parser = argparse.ArgumentParser(description="dataset manifest license gate")
    parser.add_argument("--check", required=True, help="manifest JSON to validate")
    parser.add_argument("--allow-mixamo-training", action="store_true",
                        help="allow mixamo-training rows (owner approval required)")
    args = parser.parse_args(argv)
    ok, lines = check_manifest(args.check, args.allow_mixamo_training)
    for line in lines:
        print(line)
    print("MANIFEST " + ("OK" if ok else "REJECTED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
