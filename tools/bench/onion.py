#!/usr/bin/env python3
"""Before/after onion-skin sheet of a retargeted clip (feet against a grid).

    python3 tools/bench/onion.py OUT.png CLIP LABEL=anim.glb [LABEL=anim.glb ...]

Renders each GLB's clip at 8 frames with tools/bench/filmstrip.py (fixed
side camera, 10 cm ground grid) and overlays them: a planted foot stays one
crisp shoe, a sliding foot leaves a trail of shoes. One labelled row per
GLB, stacked.
"""

import os
import shutil
import subprocess
import sys

from PIL import Image, ImageChops, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))


def blender():
    for c in (os.environ.get("BLENDER"), shutil.which("blender"), "/Applications/Blender.app/Contents/MacOS/Blender"):
        if c and os.path.exists(c):
            return os.path.realpath(c)
    sys.exit("set $BLENDER")


def main():
    out, clip, rows = sys.argv[1], sys.argv[2], [a.split("=", 1) for a in sys.argv[3:]]
    tmp = out + ".tiles"
    p = subprocess.run([blender(), "-b", "--factory-startup", "--python", os.path.join(HERE, "filmstrip.py"), "--",
                        tmp, clip] + [f"{k}={v}" for k, v in rows], capture_output=True, text=True)
    if "FILMSTRIP" not in p.stdout:
        sys.exit(p.stdout[-2000:] + p.stderr[-2000:])
    strips = []
    for r in range(len(rows)):
        tiles = [Image.open(f"{tmp}.{r}.{k}.png").convert("RGB") for k in range(8)]
        o = tiles[0]
        for t in tiles[1:]:
            o = ImageChops.darker(o, t)
        strips.append(o)
    w, h = strips[0].size
    sheet = Image.new("RGB", (w, (h + 24) * len(strips)), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    for r, ((label, _), s) in enumerate(zip(rows, strips)):
        y = r * (h + 24)
        d.text((8, y + 6), f"{label}  ({clip}, 8 frames overlaid, fixed camera)", fill=(0, 0, 0))
        sheet.paste(s, (0, y + 24))
    sheet.save(out)
    for f in os.listdir(os.path.dirname(os.path.abspath(out))):
        if f.startswith(os.path.basename(tmp)):
            os.remove(os.path.join(os.path.dirname(os.path.abspath(out)), f))
    print(out)


if __name__ == "__main__":
    main()
