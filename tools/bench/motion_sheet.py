#!/usr/bin/env python3
"""Frame strips and side-by-side reels of clips on rigged GLBs (stdlib only;
needs Blender and ffmpeg).

    motion_sheet.py strip OUT.png ROW [ROW ...] [--view 3q|front|side]
        [--trail DEF-hand.R]
    motion_sheet.py reel OUT.mp4 ROW [ROW ...] [--view ...] [--trail ...]
        (also writes OUT.strip.png: 8 evenly spaced frames per row)

ROW = label=path.glb:clip[@f1,f2,...][!m1,m2]  (frames to show; frames to
mark). A strip shows the listed frames (default 8 evenly spaced) in one
row per ROW; a reel plays each ROW's whole clip at 30 fps, side by side,
on one shared fixed camera.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))


def blender():
    for c in (os.environ.get("BLENDER"), shutil.which("blender"),
              "/Applications/Blender.app/Contents/MacOS/Blender"):
        if c and os.path.exists(c):
            return os.path.realpath(c)
    sys.exit("set $BLENDER")


def parse_row(s):
    label, rest = s.split("=", 1)
    mark = []
    if "!" in rest:
        rest, m = rest.split("!", 1)
        mark = [int(x) for x in m.split(",") if x]
    frames = "even8"
    if "@" in rest:
        rest, fr = rest.split("@", 1)
        frames = [int(x) for x in fr.split(",") if x]
    glb, clip = rest.rsplit(":", 1)
    return {"label": label, "glb": glb, "clip": clip, "frames": frames, "mark": mark}


def clip_frames(glb, clip):
    """Frame count of a clip at 30 fps (from the GLB's JSON chunk)."""
    import struct
    b = open(glb, "rb").read()
    n = struct.unpack("<I", b[12:16])[0]
    j = json.loads(b[20:20 + n])
    for a in j.get("animations", []):
        if a.get("name") == clip:
            acc = j["accessors"][a["samplers"][0]["input"]]
            return int(round(acc["max"][0] * 30)) + 1
    sys.exit(f"no clip {clip} in {glb}")


def render(spec, tmp):
    path = os.path.join(tmp, "spec.json")
    json.dump(spec, open(path, "w"))
    p = subprocess.run([blender(), "-b", "--factory-startup", "--python",
                        os.path.join(HERE, "motion_render.py"), "--", path],
                       capture_output=True, text=True, stdin=subprocess.DEVNULL)
    if "MOTION_RENDER" not in p.stdout:
        sys.exit(p.stdout[-3000:] + p.stderr[-3000:])


def ff(*args):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-nostdin", *args], check=True,
                   stdin=subprocess.DEVNULL)


def main():
    kind, out = sys.argv[1], sys.argv[2]
    rest = sys.argv[3:]
    opts = {"view": "3q", "trail": None}
    rows = []
    i = 0
    while i < len(rest):
        if rest[i].startswith("--"):
            opts[rest[i][2:]] = rest[i + 1]
            i += 2
        else:
            rows.append(parse_row(rest[i]))
            i += 1
    tmp = tempfile.mkdtemp(prefix="msheet")
    for r in rows:
        n = clip_frames(r["glb"], r["clip"])
        r["n"] = n
        if kind == "reel":
            r["frames"] = "all"
        elif r["frames"] == "even8":
            r["frames"] = [round((n - 1) * k / 7) for k in range(8)]
    spec = {"out": os.path.join(tmp, "t"), "view": opts["view"], "trail": opts["trail"],
            "shared_camera": True, "size": [480, 540] if kind == "strip" else [480, 600],
            "rows": rows}
    render(spec, tmp)
    if kind == "strip":
        strips = []
        for ri, r in enumerate(rows):
            k = len(r["frames"])
            s = os.path.join(tmp, f"s{ri}.png")
            ff("-i", os.path.join(tmp, f"t.r{ri}.%04d.png"), "-vf", f"tile={k}x1",
               "-frames:v", "1", s)
            strips.append(s)
        if len(strips) == 1:
            shutil.copy(strips[0], out)
        else:
            ins = sum((["-i", s] for s in strips), [])
            ff(*ins, "-filter_complex", f"vstack=inputs={len(strips)}", out)
    else:
        vids = []
        nmax = max(r["n"] for r in rows)
        for ri, r in enumerate(rows):
            v = os.path.join(tmp, f"v{ri}.mp4")
            # Hold the last frame so every row lasts as long as the longest.
            pad = (nmax - r["n"]) / 30.0 + 0.5
            ff("-framerate", "30", "-i", os.path.join(tmp, f"t.r{ri}.%04d.png"),
               "-vf", f"tpad=stop_mode=clone:stop_duration={pad:.3f}",
               "-pix_fmt", "yuv420p", "-c:v", "libx264", "-crf", "18", v)
            vids.append(v)
        ins = sum((["-i", v] for v in vids), [])
        ff(*ins, "-filter_complex", f"hstack=inputs={len(vids)}", "-pix_fmt", "yuv420p",
           "-c:v", "libx264", "-crf", "20", out)
        # Matching frame strips: 8 evenly spaced frames per row.
        strips = []
        for ri, r in enumerate(rows):
            n = r["n"]
            pick = [round((n - 1) * k / 7) for k in range(8)]
            lst = os.path.join(tmp, f"l{ri}.txt")
            with open(lst, "w") as fh:
                for f in pick:
                    fh.write(f"file '{os.path.join(tmp, f't.r{ri}.{f:04d}.png')}'\n")
            s = os.path.join(tmp, f"s{ri}.png")
            ff("-f", "concat", "-safe", "0", "-i", lst, "-vf", "tile=8x1", "-frames:v", "1", s)
            strips.append(s)
        base = out.rsplit(".", 1)[0]
        ins = sum((["-i", s] for s in strips), [])
        if len(strips) == 1:
            shutil.copy(strips[0], base + ".strip.png")
        else:
            ff(*ins, "-filter_complex", f"vstack=inputs={len(strips)}", base + ".strip.png")
    shutil.rmtree(tmp, ignore_errors=True)
    print(out)


if __name__ == "__main__":
    main()
