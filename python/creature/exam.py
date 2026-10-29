"""The exam: for each creature, a random motion within its limits is
filmed through the same two cameras, run through the whole tool (joint
finder -> triangulate -> solve) and compared bone by bone with the truth.
It is the gate for every change: a change that makes any creature's exam
worse is out.

    python python/creature/exam.py --creature quad_wolf=DIR [--creature biped_golem=DIR ...] \
        --out exam/ [--seeds 1,2,3] [--frames 48] [--check tests/fixtures/creature/exam_baseline.json] \
        [--video] [--blender /path/to/blender] [--motionforge PATH]

Each creature DIR holds `creature.json` (the profile), `clean.glb` (the
rig the profile came from) and `finder.pt` (its trained joint finder).
Per creature and seed the exam writes truth.json, the film frames, the
detections, the solved clip (+ GLB), the scores, and with --video a
side-by-side MP4 (the film on top, our model doing the move below).

--check compares the fresh scores with a committed baseline: a creature
whose mean joint error (mm) or mean inner bone error (deg) grows by more
than 5% (or 2 mm / 0.5 deg, whichever is larger) fails the exam.
`--update BASELINE` writes the new numbers instead.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))


def find_motionforge(explicit):
    if explicit:
        return explicit
    for p in [shutil.which("motionforge"),
              os.path.join(REPO, "rust", "target", "release", "motionforge"),
              os.path.join(os.environ.get("CARGO_TARGET_DIR", ""), "release", "motionforge")]:
        if p and os.path.exists(p):
            return p
    raise SystemExit("motionforge binary not found; build it or pass --motionforge")


def run(cmd, log=None):
    t = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if log is not None:
        log.append({"cmd": [os.path.basename(cmd[0])] + cmd[1:], "seconds": round(time.time() - t, 2),
                    "exit": r.returncode, "stdout": r.stdout[-4000:], "stderr": r.stderr[-4000:]})
    if r.returncode != 0:
        raise SystemExit(f"command failed ({r.returncode}): {' '.join(cmd)}\n{r.stderr[-2000:]}")
    return r.stdout


def exam_one(name, cdir, out, seed, frames, mf, blender, python, video, log):
    os.makedirs(out, exist_ok=True)
    creature = os.path.join(cdir, "creature.json")
    glb = os.path.join(cdir, "clean.glb")
    model = os.path.join(cdir, "finder.pt")
    for p in (creature, glb, model):
        if not os.path.exists(p):
            raise SystemExit(f"{name}: missing {p}")
    truth = os.path.join(out, "truth.json")
    run([mf, "creature-motion", "--creature", creature, "--output", truth, "--frames", str(frames), "--seed", str(seed)], log)
    film = os.path.join(out, "film")
    run([blender, "--background", "--factory-startup", "--python", os.path.join(HERE, "render.py"), "--",
         "--glb", glb, "--creature", creature, "--clip", truth, "--out", film, "--mode", "motion"], log)
    det = os.path.join(out, "detections.json")
    run([python, os.path.join(HERE, "detect.py"), "--model", model, "--creature", creature, "--frames", film, "--out", det], log)
    solved = os.path.join(out, "solved.json")
    solve_report = os.path.join(out, "solve.json")
    run([mf, "creature-solve", "--creature", creature, "--detections", det, "--output", solved,
         "--glb-in", glb, "--glb-out", os.path.join(out, "solved.glb"), "--clip-name", f"exam_{seed}",
         "--report", solve_report], log)
    scores_path = os.path.join(out, "scores.json")
    run([mf, "creature-compare", "--creature", creature, "--truth", truth, "--solved", solved, "--output", scores_path], log)
    with open(scores_path) as f:
        scores = json.load(f)
    with open(solve_report) as f:
        scores["solve"] = json.load(f)
    if video:
        ours = os.path.join(out, "ours")
        run([blender, "--background", "--factory-startup", "--python", os.path.join(HERE, "render.py"), "--",
             "--glb", glb, "--creature", creature, "--clip", solved, "--out", ours, "--mode", "motion"], log)
        make_video(film, ours, os.path.join(out, f"{name}_seed{seed}_side_by_side.mp4"), creature, log)
    return scores


def make_video(film, ours, path, creature, log):
    """Top row: the film (front | side). Bottom row: our model doing the
    solved move through the same cameras. No sound."""
    with open(creature) as f:
        views = [v["name"] for v in json.load(f)["views"]]
    with open(os.path.join(film, "truth.json")) as f:
        fps = json.load(f).get("fps", 24)
    inputs = []
    for d in (film, ours):
        for v in views:
            inputs += ["-framerate", str(fps), "-i", os.path.join(d, f"f%05d_{v}.png")]
    n = len(views)
    top = "".join(f"[{i}:v]" for i in range(n)) + f"hstack=inputs={n}[top]"
    bottom = "".join(f"[{n + i}:v]" for i in range(n)) + f"hstack=inputs={n}[bottom]"
    graph = f"{top};{bottom};[top][bottom]vstack=inputs=2,drawtext=text='film':x=10:y=10:fontsize=28:fontcolor=white," \
            f"drawtext=text='motionforge':x=10:y=h/2+10:fontsize=28:fontcolor=white[v]"
    cmd = ["ffmpeg", "-y", "-loglevel", "error"] + inputs + ["-filter_complex", graph, "-map", "[v]", "-an",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", path]
    try:
        run(cmd, log)
    except SystemExit:
        # No drawtext (ffmpeg built without freetype): plain stack.
        graph = f"{top};{bottom};[top][bottom]vstack=inputs=2[v]"
        cmd = ["ffmpeg", "-y", "-loglevel", "error"] + inputs + ["-filter_complex", graph, "-map", "[v]", "-an",
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", path]
        run(cmd, log)


def worse(new, old, key, rel=0.05, abs_floor=0.0):
    return new[key] > old[key] * (1 + rel) + abs_floor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--creature", action="append", default=[], help="name=DIR (creature.json, clean.glb, finder.pt)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--check", help="baseline JSON to compare against (fails on regression)")
    ap.add_argument("--update", help="write the new scores as this baseline")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--blender", default=os.environ.get("BLENDER", "blender"))
    ap.add_argument("--motionforge", default="")
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()
    if not args.creature:
        raise SystemExit("give at least one --creature name=DIR")
    mf = find_motionforge(args.motionforge)
    seeds = [int(s) for s in args.seeds.split(",") if s]
    results = {"format": "motionforge-exam", "version": 1, "frames": args.frames, "seeds": seeds, "creatures": {}}
    log = []
    t0 = time.time()
    for spec in args.creature:
        name, cdir = spec.split("=", 1)
        per_seed = {}
        for seed in seeds:
            out = os.path.join(args.out, name, f"seed{seed}")
            print(f"[{name} seed {seed}] ...", flush=True)
            s = exam_one(name, cdir, out, seed, args.frames, mf, args.blender, args.python, args.video, log)
            per_seed[str(seed)] = s
            print(f"[{name} seed {seed}] mean {s['mean_mm']:.1f} mm, inner bones {s['mean_deg_inner']:.2f} deg, "
                  f"max {s['max_mm']:.0f} mm, foot slide {s['foot_slide_mm']:.1f} mm, "
                  f"views agree {s['solve']['consistent']} (gap {s['solve']['median_gap_ratio'] * 100:.2f}%)", flush=True)
        n = len(per_seed)
        summary = {k: sum(v[k] for v in per_seed.values()) / n for k in ("mean_mm", "max_mm", "mean_deg", "mean_deg_inner", "max_deg", "foot_slide_mm")}
        summary["seeds"] = per_seed
        results["creatures"][name] = summary
        print(f"[{name}] over {n} seeds: mean {summary['mean_mm']:.1f} mm, inner bones {summary['mean_deg_inner']:.2f} deg, "
              f"max {summary['max_mm']:.0f} mm, foot slide {summary['foot_slide_mm']:.1f} mm", flush=True)
    results["seconds"] = round(time.time() - t0, 1)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "exam.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(args.out, "exam.log.json"), "w") as f:
        json.dump(log, f, indent=1)
    ok = True
    if args.check:
        with open(args.check) as f:
            base = json.load(f)
        for name, s in results["creatures"].items():
            b = base.get("creatures", {}).get(name)
            if b is None:
                print(f"EXAM {name}: no baseline (new creature)")
                continue
            bad = []
            if worse(s, b, "mean_mm", abs_floor=2.0):
                bad.append(f"mean_mm {b['mean_mm']:.1f} -> {s['mean_mm']:.1f}")
            if worse(s, b, "mean_deg_inner", abs_floor=0.5):
                bad.append(f"mean_deg_inner {b['mean_deg_inner']:.2f} -> {s['mean_deg_inner']:.2f}")
            if bad:
                ok = False
                print(f"EXAM {name}: WORSE ({'; '.join(bad)})")
            else:
                print(f"EXAM {name}: ok (mean_mm {b['mean_mm']:.1f} -> {s['mean_mm']:.1f}, "
                      f"mean_deg_inner {b['mean_deg_inner']:.2f} -> {s['mean_deg_inner']:.2f})")
    if args.update:
        slim = {"format": "motionforge-exam-baseline", "version": 1, "frames": args.frames, "seeds": seeds,
                "creatures": {n: {k: round(v, 3) for k, v in s.items() if k != "seeds"} for n, s in results["creatures"].items()}}
        with open(args.update, "w") as f:
            json.dump(slim, f, indent=2)
            f.write("\n")
        print(f"baseline written: {args.update}")
    print("EXAM PASS" if ok else "EXAM FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
