#!/usr/bin/env python3
"""Short MP4s of clips on rigged GLBs, seen as the game shows them
(stdlib only; needs Blender and ffmpeg).

    motion_video.py OUT.mp4 ROW [ROW ...] [--cam game|lock|close]
    motion_video.py OUT.png ROW [ROW ...] --only 0,12 [...]   (stills: a row
        per ROW, a column per frame)
        [--yaw DEG] [--prop sword.glb@Socket_Hand_R] [--prop-pose socket|game]
        [--engine eevee|workbench] [--size 960x540] [--loops 2]
        [--slow 0.5] [--marks check.json] [--cols 3]

ROW = label=path.glb:clip. Each row is one panel, side by side (wrapping
into rows of `--cols`), all panels the same length: each clip plays
`--loops` times at game speed with a short hold between, then once at
`--slow` speed (0 for none; the panels say so). `--marks` takes
`motionforge motion check --json` output and tags each attack's load and
strike frames in the label. Also writes OUT.strip.png (8 frames a row).

Camera, lights and floor: `motion_video_render.py`'s module doc (the game
camera is HLH's follow camera: 3.0 m from a focus 1.535 m up, pitch
-18.3 deg, straight behind him, 75 deg vertical field of view).
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
FPS = 30
HOLD_S = 0.35


def blender():
    for c in (os.environ.get("BLENDER"), shutil.which("blender"),
              "/Applications/Blender.app/Contents/MacOS/Blender"):
        if c and os.path.exists(c):
            return os.path.realpath(c)
    sys.exit("set $BLENDER")


def ff(*args):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-nostdin", *args], check=True,
                   stdin=subprocess.DEVNULL)


def has_drawtext():
    p = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True)
    return " drawtext " in p.stdout


def font():
    for f in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(f):
            return f
    return None


def parse_row(s):
    label, rest = s.split("=", 1)
    glb, clip = rest.rsplit(":", 1)
    return {"label": label, "glb": os.path.abspath(glb), "clip": clip}


def marks_from(path):
    """{clip: {frame: tag}} from `motion check --json`."""
    out = {}
    j = json.load(open(path))
    for c in j.get("clips", []):
        a = c.get("attack")
        m = {}
        if a:
            m[str(a["load_frame"])] = "load"
            m[str(a["peak_frame"])] = "strike"
        rc = c.get("read_cam")
        if rc and "key_frame" in rc and str(rc["key_frame"]) not in m:
            m[str(rc["key_frame"])] = "key pose"
        out[c["name"]] = m
    return out


def main():
    out = sys.argv[1]
    rest = sys.argv[2:]
    opts = {"cam": "game", "yaw": "0", "prop": None, "prop-pose": "socket", "engine": "eevee",
            "size": "960x540", "loops": "2", "slow": "0.5", "marks": None, "cols": "3", "only": None, "fit": "sphere"}
    rows = []
    i = 0
    while i < len(rest):
        if rest[i].startswith("--"):
            opts[rest[i][2:]] = rest[i + 1]
            i += 2
        else:
            rows.append(parse_row(rest[i]))
            i += 1
    if not rows:
        sys.exit(__doc__)
    w, h = (int(x) for x in opts["size"].split("x"))
    if opts["marks"]:
        mk = marks_from(opts["marks"])
        for r in rows:
            r["marks"] = mk.get(r["clip"], {})
    tmp = tempfile.mkdtemp(prefix="mvideo")
    spec = {"out": os.path.join(tmp, "t"), "size": [w, h], "cam": opts["cam"],
            "yaw_deg": float(opts["yaw"]), "engine": opts["engine"], "fit": opts["fit"],
            "prop": None, "prop_pose": opts["prop-pose"], "rows": rows,
            "only": [int(x) for x in opts["only"].split(",")] if opts["only"] else None}
    if opts["prop"]:
        p, sock = opts["prop"].rsplit("@", 1)
        spec["prop"] = os.path.abspath(p) + "@" + sock
    path = os.path.join(tmp, "spec.json")
    json.dump(spec, open(path, "w"))
    p = subprocess.run([blender(), "-b", "--factory-startup", "--python",
                        os.path.join(HERE, "motion_video_render.py"), "--", path],
                       capture_output=True, text=True, stdin=subprocess.DEVNULL)
    if "MOTION_VIDEO" not in p.stdout:
        sys.exit(p.stdout[-3000:] + p.stderr[-3000:])
    counts = json.loads(p.stdout.split("MOTION_VIDEO", 1)[1].strip().splitlines()[0])
    if out.endswith(".png"):
        only = spec["only"] or [0]
        tiles = []
        for ri, (_, n) in enumerate(counts):
            row = [os.path.join(tmp, f"t.r{ri}.{min(k, n - 1):04d}.png") for k in only]
            row = [t for t in row if os.path.exists(t)]
            s = os.path.join(tmp, f"still{ri}.png")
            if len(row) == 1:
                shutil.copy(row[0], s)
            else:
                ff(*sum((["-i", t] for t in row), []), "-filter_complex",
                   f"hstack=inputs={len(row)}", s)
            tiles.append(s)
        if len(tiles) == 1:
            shutil.copy(tiles[0], out)
        else:
            ff(*sum((["-i", t] for t in tiles), []), "-filter_complex",
               f"vstack=inputs={len(tiles)}", out)
        shutil.rmtree(tmp, ignore_errors=True)
        print(out)
        return
    loops, slow = int(opts["loops"]), float(opts["slow"])
    hold = int(round(HOLD_S * FPS))
    # Per panel: the frame sequence (loops at speed, then the slow pass).
    seqs = []
    for ri, (_, n) in enumerate(counts):
        fr = [os.path.join(tmp, f"t.r{ri}.{k:04d}.png") for k in range(n)]
        seq = []
        for _ in range(loops):
            seq += fr + [fr[-1]] * hold
        fast_len = len(seq)
        if slow > 0:
            rep = max(1, int(round(1 / slow)))
            seq += [f for f in fr for _ in range(rep)] + [fr[-1]] * hold
        seqs.append((seq, fast_len))
    # Panels play in step: the fast part and the slow part each padded to
    # the longest panel's.
    fast = max(fl for _, fl in seqs)
    slow_len = max(len(s) - fl for s, fl in seqs)
    vids = []
    for ri, (seq, fl) in enumerate(seqs):
        a, b = seq[:fl], seq[fl:]
        a += [a[-1]] * (fast - len(a))
        if slow_len:
            b = b or [a[-1]]
            b += [b[-1]] * (slow_len - len(b))
        d = os.path.join(tmp, f"seq{ri}")
        os.makedirs(d)
        for k, src in enumerate(a + b):
            os.symlink(src, os.path.join(d, f"{k:05d}.png"))
        v = os.path.join(tmp, f"v{ri}.mp4")
        ff("-framerate", str(FPS), "-i", os.path.join(d, "%05d.png"), "-pix_fmt", "yuv420p",
           "-c:v", "libx264", "-crf", "16", v)
        vids.append(v)
    cols = max(1, min(int(opts["cols"]), len(vids)))
    nrow = (len(vids) + cols - 1) // cols
    ins = sum((["-i", v] for v in vids), [])
    blank = nrow * cols - len(vids)
    filt = []
    if blank:
        dur = (fast + slow_len) / FPS
        ins += ["-f", "lavfi", "-i", f"color=c=black:s={w}x{h}:r={FPS}:d={dur:.3f}"] * blank
    total = len(vids) + blank
    if total == 1:
        stack = "[0:v]null[s]"
    else:
        lay = "|".join(f"{(k % cols) * w}_{(k // cols) * h}" for k in range(total))
        stack = "".join(f"[{k}:v]" for k in range(total)) + f"xstack=inputs={total}:layout={lay}[s]"
    filt.append(stack)
    last = "[s]"
    if slow_len and has_drawtext():
        f = font()
        t0 = fast / FPS
        txt = f"drawtext=text='x{slow:g} speed':x=w-tw-20:y=h-th-16:fontsize={max(14, h // 22)}" \
              f":fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=8:enable='gte(t,{t0:.3f})'"
        if f:
            txt += f":fontfile='{f}'"
        filt.append(f"{last}{txt}[o]")
        last = "[o]"
    ff(*ins, "-filter_complex", ";".join(filt), "-map", last, "-pix_fmt", "yuv420p",
       "-c:v", "libx264", "-crf", "20", "-movflags", "+faststart", out)
    # Strip: 8 evenly spaced frames per row.
    strips = []
    for ri, (_, n) in enumerate(counts):
        pick = [round((n - 1) * k / 7) for k in range(8)]
        lst = os.path.join(tmp, f"l{ri}.txt")
        with open(lst, "w") as fh:
            for fr in pick:
                fh.write(f"file '{os.path.join(tmp, f't.r{ri}.{fr:04d}.png')}'\n")
        s = os.path.join(tmp, f"s{ri}.png")
        ff("-f", "concat", "-safe", "0", "-i", lst, "-vf", f"scale={w // 2}:-1,tile=8x1",
           "-frames:v", "1", s)
        strips.append(s)
    base = out.rsplit(".", 1)[0]
    if len(strips) == 1:
        shutil.copy(strips[0], base + ".strip.png")
    else:
        ff(*sum((["-i", s] for s in strips), []), "-filter_complex",
           f"vstack=inputs={len(strips)}", base + ".strip.png")
    if os.environ.get("MOTION_VIDEO_KEEP"):
        print("frames kept in", tmp)
    else:
        shutil.rmtree(tmp, ignore_errors=True)
    print(out)


if __name__ == "__main__":
    main()
