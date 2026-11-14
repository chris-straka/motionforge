#!/usr/bin/env python3
"""A motion clip the owner can read on a phone (the standard for every
motion clip sent to the owner, 2026-11-13): tall framing with the figure
filling most of the height; the close 3/4 view first, then the game
camera's angle; each at normal speed, then in slow motion, labelled.

    phone_clip.py OUT.mp4 LABEL=path.glb:clip [--prop sword.glb@Socket_Hand_R]
        [--prop-pose socket|game] [--size 720x1080] [--slow 0.4]

Several LABEL=... rows play one after another in the same video.
Needs Blender and ffmpeg (motion_video.py does the rendering).
"""
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))


def ff(*args):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-nostdin", *args], check=True,
                   stdin=subprocess.DEVNULL)


def main():
    out, rest = sys.argv[1], sys.argv[2:]
    opts = {"prop": None, "prop-pose": "socket", "size": "720x1080", "slow": "0.4"}
    rows, i = [], 0
    while i < len(rest):
        if rest[i].startswith("--"):
            opts[rest[i][2:]] = rest[i + 1]
            i += 2
        else:
            rows.append(rest[i])
            i += 1
    if not rows:
        sys.exit(__doc__)
    tmp = tempfile.mkdtemp(prefix="phone")
    parts = []
    for ri, row in enumerate(rows):
        label, src = row.split("=", 1)
        for cam, what in (("close", "close"), ("gamefit", "game camera")):
            part = os.path.join(tmp, f"p{ri}_{cam}.mp4")
            cmd = [sys.executable, os.path.join(HERE, "motion_video.py"), part,
                   f"{label} - {what}={src}", "--cam", cam, "--size", opts["size"],
                   "--loops", "1", "--slow", opts["slow"], "--cols", "1",
                   "--prop-pose", opts["prop-pose"], "--fit", "tight"]
            if opts["prop"]:
                cmd += ["--prop", opts["prop"]]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0 or not os.path.exists(part):
                sys.exit(r.stdout[-2000:] + r.stderr[-2000:])
            parts.append(part)
    lst = os.path.join(tmp, "list.txt")
    with open(lst, "w") as fh:
        for p in parts:
            fh.write(f"file '{p}'\n")
    ff("-f", "concat", "-safe", "0", "-i", lst, "-c:v", "libx264", "-pix_fmt", "yuv420p",
       "-crf", "18", "-movflags", "+faststart", out)
    shutil.rmtree(tmp, ignore_errors=True)
    print(out)


if __name__ == "__main__":
    main()
