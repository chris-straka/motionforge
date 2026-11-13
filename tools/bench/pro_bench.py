#!/usr/bin/env python3
"""Correctness benchmark against professional clips (stdlib only; drives the
motionforge CLI).

    pro_bench.py calibrate [--out DIR] [--libs ual1,ual2]
    pro_bench.py run --target andras=RIG.glb [--target name=RIG.glb ...]
                     [--libs ual1,ual2] [--clips a,b] [--stages ...] [--out DIR]
                     [--jobs N]
    pro_bench.py report DIR

Source: Quaternius Universal Animation Library 1 and 2 [Standard], root
motion GLBs (CC0; benchmark use only, never training). Fetch them with
tools/bench/fetch.sh (into $FORGE_BENCH, default ~/.cache/forge-bench).

calibrate: `motion check` on every pro clip on its own mannequin. A pro clip
should show no defect; every one it shows is listed (calib.md), so a measure
that flags the pros can be fixed (docs/pro-bench.md).

run: per target rig and pro clip, the clip goes through what the game's clips
go through, one stage at a time:
  raw      animate --no-pin --no-contact (the plain retarget, for reference)
  retarget animate (as shipped: feet pinned, contact pass)
  clean    clean on the retarget
  contact  clean --contact on the retarget
  stylize  stylize (body mode) on the clean result, attack clips only
and each stage's clips are scored by `motion check --tracks` and compared
with the pro clip frame by frame (both sampled at 30 fps). Correctness only:
style (exaggeration, attack timing, readability) is never scored against the
reference, since the pros' style need not suit our characters.

Checks per clip and stage (lengths on a 1.75 m body):
  plants   planted frames agree with the pro's: IoU per foot >= 0.75
  hold     during the pro's plants the contact point stays put: path no more
           than the pro's own + 2.5 cm per plant
  skate    skate distance no more than the pro's + 2.5 cm
  sink     no frame a foot is >1 cm under the floor where the pro's is not
           (2 frames allowed)
  float    no more floating frames (whole body off the floor) than the pro
           + 10% of the clip
  through  frames with an arm inside the body where the pro's is not: under
           5% of the clip
  limits   frames where a joint is 5+ deg further past a limit than the pro's
           same joint: under 5% of the clip
  rough    back-and-forth (worst chain mean) no more than the pro's + 0.5 mm
  keys     key poses (the body's speed minima) within 2 frames of the pro's
           on average (not for stylize, which retimes on purpose)
A clip passes a stage when every check passes. The main score is the share
of clips that pass.

Outputs under --out (default bench/pro/<date>, gitignored): the stage GLBs,
motion check JSON per stage, results.json and report.md.
"""

import argparse
import concurrent.futures as cf
import datetime
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
CLI = os.environ.get("MOTIONFORGE_BIN", os.path.join(ROOT, "rust", "target", "release", "motionforge"))
FORGE = os.environ.get("FORGE_BENCH", os.path.expanduser("~/.cache/forge-bench"))
LIBS = {
    "ual1": os.path.join(FORGE, "clips", "UAL1_Standard_RM.glb"),
    "ual2": os.path.join(FORGE, "clips", "ual2", "UAL2_Standard_RM.glb"),
}
MANNEQUIN_F = os.path.join(FORGE, "clips", "ual2", "Mannequin_F.glb")
STAGES = ["raw", "retarget", "clean", "contact", "stylize"]
# Clips that are a pose, not a motion.
SKIP = {"A_TPose"}
ATTACK_WORDS = ["attack", "slash", "cut", "combo", "stab", "swing", "strike", "punch", "kick"]

TOL = {
    "iou": 0.75, "hold_cm": 2.5, "skate_cm": 2.5, "sink_frames": 2, "float_share": 0.10,
    "through_share": 0.05, "limit_deg": 5.0, "limit_share": 0.05, "rough_mm": 0.5, "keys_frames": 2.0,
}
CHECKS = ["plants", "hold", "skate", "sink", "float", "through", "limits", "rough", "keys"]


def run(cmd, out=None):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)}: {(p.stderr or p.stdout)[-400:]}")
    if out:
        with open(out, "w") as fh:
            fh.write(p.stdout)
    return p.stdout


def check(glb, out, clips=None):
    cmd = [CLI, "motion", "check", glb, "--tracks"]
    if clips:
        cmd += ["--clip", ",".join(clips)]
    run(cmd, out)
    return json.load(open(out))


def is_attack(name):
    n = name.lower()
    return any(w in n for w in ATTACK_WORDS)


# ---------------------------------------------------------------- measures

def runs(bits):
    """[a, b) runs of 2+ set frames."""
    out, a = [], None
    for f, x in enumerate(list(bits) + [0]):
        if x and a is None:
            a = f
        if not x and a is not None:
            if f - a >= 2:
                out.append((a, f))
            a = None
    return out


def slide(foot, a, b):
    """The contact point's horizontal path over frames [a, b), metres: each
    step follows the point that was lowest on the frame before, as motion
    check's plants do (a heel-to-ball switch is not a slide)."""
    return sum(foot["step"][a + 1:b])


def worst_rough(c):
    return max((r["mean_mm"] for r in c["rough"]), default=0.0)


def compare(pro, tgt, stage):
    """Correctness of one target clip against its pro clip: the checks and
    the numbers behind them. A stage that retimes (stylize) is compared by
    totals, not frame by frame, and plants, hold and keys are not scored."""
    pd, td = pro["detail"], tgt["detail"]
    n = min(pro["frames"], tgt["frames"])
    retime = stage == "stylize"
    m = {}
    # Plants: IoU of planted frames per foot, and contact hold during the
    # pro's plants.
    ious, holds, holds_pro = [], [], []
    for pf in pd["feet"]:
        tf = next((t for t in td["feet"] if t["side"] == pf["side"]), None)
        if tf is None:
            continue
        a, b = pf["planted"][:n], tf["planted"][:n]
        union = sum(1 for x, y in zip(a, b) if x or y)
        if union:
            ious.append(sum(1 for x, y in zip(a, b) if x and y) / union)
        for s, e in runs(a):
            holds.append(slide(tf, s, e) * 100)
            holds_pro.append(slide(pf, s, e) * 100)
    m["plant_iou"] = round(min(ious), 3) if ious else None
    m["hold_cm"] = round(sum(holds) / len(holds), 2) if holds else None
    m["hold_pro_cm"] = round(sum(holds_pro) / len(holds_pro), 2) if holds_pro else None
    m["hold_worst_cm"] = round(max((h - p for h, p in zip(holds, holds_pro)), default=0.0), 2)
    m["skate_cm"] = tgt["feet"]["skate_cm"]
    m["skate_pro_cm"] = pro["feet"]["skate_cm"]
    # Ground against the pro's.
    sink = lambda d, f: d["lowest"][f] < -0.01
    if retime:
        m["sink_new"] = tgt["feet"]["sink_frames"] - pro["feet"]["sink_frames"]
    else:
        m["sink_new"] = sum(1 for f in range(n) if sink(td, f) and not sink(pd, f))
    m["float_extra"] = tgt["feet"]["float_frames"] - pro["feet"]["float_frames"]
    m["ground_cm"] = round(sum(abs(td["lowest"][f] - pd["lowest"][f]) for f in range(n)) / max(n, 1) * 100, 2)
    # Through the body, where the pro is clear.
    tt, pt = td.get("through") or [0] * n, pd.get("through") or [0] * n
    if retime:
        m["through_new"] = sum(tt) - sum(pt)
    else:
        m["through_new"] = sum(1 for f in range(n) if tt[f] and not pt[f])
    # Limits past the pro's own, same joint (by frame; retimed: past the
    # pro's worst for that joint).
    lim = 0
    for f in range(tgt["frames"] if retime else n):
        past = False
        for w, v in td["limits"].items():
            ref = pd["limits"].get(w)
            base = (max(ref) if ref else 0.0) if retime else (ref[f] if ref and f < len(ref) else 0.0)
            if f < len(v) and v[f] > base + TOL["limit_deg"]:
                past = True
        lim += past
    m["limits_new"] = lim
    m["rough_mm"] = round(worst_rough(tgt), 3)
    m["rough_pro_mm"] = round(worst_rough(pro), 3)
    # Key poses (speed minima): mean distance from each pro key to the
    # nearest target key.
    pk, tk = pro["read"]["keys"], tgt["read"]["keys"]
    inner = [k for k in pk if 0 < k < n - 1] or pk
    m["keys_frames"] = round(sum(min(abs(k - t) for t in tk) for k in inner) / len(inner), 2) if inner and tk else None
    # Pass/fail.
    fail = []
    if retime:
        m["plant_iou"] = m["hold_cm"] = m["hold_pro_cm"] = m["keys_frames"] = None
    if m["plant_iou"] is not None and m["plant_iou"] < TOL["iou"]:
        fail.append("plants")
    if m["hold_cm"] is not None and m["hold_cm"] > m["hold_pro_cm"] + TOL["hold_cm"]:
        fail.append("hold")
    if m["skate_cm"] > m["skate_pro_cm"] + TOL["skate_cm"]:
        fail.append("skate")
    if m["sink_new"] > TOL["sink_frames"]:
        fail.append("sink")
    if m["float_extra"] > TOL["float_share"] * n:
        fail.append("float")
    if m["through_new"] > TOL["through_share"] * n:
        fail.append("through")
    if m["limits_new"] > TOL["limit_share"] * n:
        fail.append("limits")
    if m["rough_mm"] > m["rough_pro_mm"] + TOL["rough_mm"]:
        fail.append("rough")
    if m["keys_frames"] is not None and m["keys_frames"] > TOL["keys_frames"]:
        fail.append("keys")
    m["fail"] = fail
    return m


# ---------------------------------------------------------------- calibrate

def defects(c):
    """The defects motion check reports on one clip (no reference)."""
    f, t = c["feet"], c.get("through_body") or {}
    out = {}
    if f["skating_plants"]:
        out["skate"] = f"{f['skate_cm']} cm in {f['skating_plants']} plants (worst {f['worst_plant_cm']} cm, " \
                       f"{f['worst_plant_foot']} from f{f['worst_plant_frame']})"
    if f["sink_frames"]:
        out["sink"] = f"{f['sink_frames']} fr, {f['sink_cm']} cm"
    if f["float_frames"]:
        out["float"] = f"{f['float_frames']} fr, {f['float_cm']} cm"
    if t.get("mesh_frames"):
        out["through"] = f"{t['mesh_frames']} fr, {t['mesh_depth_cm']} cm ({', '.join(t['contacts'][:3])})"
    if c["limits"]["frames"]:
        out["limits"] = f"{c['limits']['frames']} fr, worst {c['limits']['worst_deg']} deg ({c['limits']['worst']})"
    rw = max(c["rough"], key=lambda r: r["peak_mm"])
    if rw["peak_mm"] > 5.0:
        out["rough"] = f"peak {rw['peak_mm']} mm {rw['chain']} f{rw['peak_frame']}"
    return out


def calibrate(a):
    out = a.out or os.path.join(ROOT, "bench", "pro", "cal-" + datetime.date.today().isoformat())
    os.makedirs(out, exist_ok=True)
    res = {"commit": commit(), "libs": {}}
    kinds = ["skate", "sink", "float", "through", "limits", "rough"]
    lines = [f"# Pro clips on their own mannequin ({res['commit']})", "",
             "Defects `motion check` reports on the Quaternius clips (rough = back-and-forth peak over 5 mm).", ""]
    for lib in a.libs.split(","):
        j = check(LIBS[lib], os.path.join(out, f"{lib}.json"))
        clips = [c for c in j["clips"] if c["name"] not in SKIP]
        per = {c["name"]: defects(c) for c in clips}
        counts = {k: sum(1 for d in per.values() if k in d) for k in kinds}
        res["libs"][lib] = {"clips": len(clips), "counts": counts, "defects": per}
        lines += [f"## {lib}: {len(clips)} clips", "",
                  "| " + " | ".join(kinds) + " | any |", "|" + "---|" * (len(kinds) + 1),
                  "| " + " | ".join(str(counts[k]) for k in kinds) + f" | {sum(1 for d in per.values() if d)} |", ""]
        for name, d in per.items():
            if d:
                lines.append(f"- {name}: " + "; ".join(f"{k} {v}" for k, v in d.items()))
        lines.append("")
    json.dump(res, open(os.path.join(out, "calib.json"), "w"), indent=1)
    open(os.path.join(out, "calib.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


# ---------------------------------------------------------------- run

def commit():
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()


def clip_names(glb_json):
    return [c["name"] for c in glb_json["clips"]]


def stage_files(target, rig, lib, src, clips, stages, out):
    """Run the stages for one target and library; returns {stage: json path}."""
    w = os.path.join(out, target, lib)
    os.makedirs(w, exist_ok=True)
    pick = ",".join(clips)
    paths = {}
    g = lambda s: os.path.join(w, f"{s}.glb")
    log = open(os.path.join(w, "log.txt"), "w")
    if "raw" in stages:
        log.write(run([CLI, "animate", "--input", rig, "--clips", src, "--output", g("raw"), "--pick", pick,
                       "--no-pin", "--no-contact"]))
    log.write(run([CLI, "animate", "--input", rig, "--clips", src, "--output", g("retarget"), "--pick", pick]))
    if "clean" in stages or "stylize" in stages:
        log.write(run([CLI, "clean", "--input", g("retarget"), "--output", g("clean"), "--clip", pick,
                       "--report", os.path.join(w, "clean_report.json")]))
    if "contact" in stages:
        log.write(run([CLI, "clean", "--input", g("retarget"), "--output", g("contact"), "--clip", pick,
                       "--contact", "--report", os.path.join(w, "contact_report.json")]))
    attacks = [c for c in clips if is_attack(c)]
    if "stylize" in stages and attacks:
        log.write(run([CLI, "stylize", "--input", g("clean"), "--output", g("stylize"), "--clip", ",".join(attacks)]))
    jobs = []
    for s in stages:
        if s == "stylize" and not attacks:
            continue
        if os.path.exists(g(s)):
            jobs.append((s, g(s), os.path.join(w, f"{s}.json"), attacks if s == "stylize" else clips))
    return jobs


def main_run(a):
    out = a.out or os.path.join(ROOT, "bench", "pro", datetime.date.today().isoformat())
    os.makedirs(out, exist_ok=True)
    stages = a.stages.split(",")
    targets = dict(t.split("=", 1) for t in a.target)
    if a.mannequin_f and os.path.exists(MANNEQUIN_F):
        std = os.path.join(out, "mannequin_f_std.glb")
        run([CLI, "standardize", "--input", MANNEQUIN_F, "--output", std])
        targets["mannequin_f"] = std
    want = a.clips.split(",") if a.clips else None
    results = {"commit": commit(), "date": datetime.datetime.now().isoformat(timespec="seconds"),
               "stages": stages, "tol": TOL, "targets": {}}
    with cf.ThreadPoolExecutor(max_workers=a.jobs) as ex:
        pros = {}
        for lib in a.libs.split(","):
            pros[lib] = ex.submit(check, LIBS[lib], os.path.join(out, f"pro_{lib}.json"))
        pros = {lib: f.result() for lib, f in pros.items()}
        jobs = []
        for target, rig in targets.items():
            for lib, pj in pros.items():
                clips = [c for c in clip_names(pj) if c not in SKIP and (want is None or c in want)]
                if clips:
                    jobs.append((target, lib, ex.submit(stage_files, target, rig, lib, LIBS[lib], clips, stages, out)))
        checks = []
        for target, lib, f in jobs:
            for s, glb, js, clips in f.result():
                checks.append((target, lib, s, ex.submit(check, glb, js, clips)))
        for target, lib, s, f in checks:
            tj = f.result()
            pro = {c["name"]: c for c in pros[lib]["clips"]}
            T = results["targets"].setdefault(target, {"height_m": tj["height_m"], "clips": {}})
            for c in tj["clips"]:
                key = f"{lib}/{c['name']}"
                T["clips"].setdefault(key, {})[s] = compare(pro[c["name"]], c, s)
            print(f"{target} {lib} {s}: {len(tj['clips'])} clips", flush=True)
    json.dump(results, open(os.path.join(out, "results.json"), "w"), indent=1)
    report(out)


def report(out):
    R = json.load(open(os.path.join(out, "results.json")))
    L = [f"# Pro correctness bench ({R['commit']}, {R['date']})", "",
         "Each stage's clips against the professional clip they came from; correctness only.", ""]
    for target, T in R["targets"].items():
        clips = T["clips"]
        L += [f"## {target} ({T['height_m']} m), {len(clips)} clips", "",
              "| stage | clips | pass | " + " | ".join(CHECKS) + " | IoU (mean) | hold cm (pro) | sink fr | "
              "through fr | limit fr | rough mm (pro) |",
              "|" + "---|" * (len(CHECKS) + 9)]
        for s in R["stages"]:
            rows = [c[s] for c in clips.values() if s in c]
            if not rows:
                continue
            def mean(k, fmt):
                xs = [r[k] for r in rows if r[k] is not None]
                return format(sum(xs) / len(xs), fmt) if xs else "-"
            L.append(f"| {s} | {len(rows)} | {sum(1 for r in rows if not r['fail'])} | " +
                     " | ".join(str(sum(1 for r in rows if ch in r["fail"])) for ch in CHECKS) +
                     f" | {mean('plant_iou', '.2f')} | {mean('hold_cm', '.1f')} ({mean('hold_pro_cm', '.1f')}) | "
                     f"{sum(r['sink_new'] for r in rows)} | {sum(r['through_new'] for r in rows)} | "
                     f"{sum(r['limits_new'] for r in rows)} | {mean('rough_mm', '.2f')} ({mean('rough_pro_mm', '.2f')}) |")
        L.append("")
        # Where each pass helps or hurts: a stage against the one it runs on.
        for s, base in (("retarget", "raw"), ("clean", "retarget"), ("contact", "retarget"), ("stylize", "clean")):
            worse, better = [], []
            for name, c in sorted(clips.items()):
                if s in c and base in c:
                    nb, ns = set(c[base]["fail"]), set(c[s]["fail"])
                    if ns - nb:
                        worse.append(f"{name} (+{','.join(sorted(ns - nb))})")
                    if nb - ns:
                        better.append(f"{name} (-{','.join(sorted(nb - ns))})")
            if worse or better:
                L += [f"**{s} vs {base}**: fixes {len(better)}, breaks {len(worse)}", ""]
                if better:
                    L.append("- fixes: " + "; ".join(better))
                if worse:
                    L.append("- breaks: " + "; ".join(worse))
                L.append("")
    open(os.path.join(out, "report.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrate")
    c.add_argument("--out")
    c.add_argument("--libs", default="ual1,ual2")
    r = sub.add_parser("run")
    r.add_argument("--target", action="append", default=[], help="name=rigged.glb (repeatable)")
    r.add_argument("--mannequin-f", action="store_true", help="also Quaternius' female mannequin (UAL2)")
    r.add_argument("--libs", default="ual1,ual2")
    r.add_argument("--clips")
    r.add_argument("--stages", default=",".join(STAGES))
    r.add_argument("--out")
    r.add_argument("--jobs", type=int, default=4)
    p = sub.add_parser("report")
    p.add_argument("dir")
    a = ap.parse_args()
    if not os.path.exists(CLI):
        subprocess.run(["cargo", "build", "--locked", "--release"], cwd=os.path.join(ROOT, "rust"), check=True)
    if a.cmd == "calibrate":
        calibrate(a)
    elif a.cmd == "run":
        main_run(a)
    else:
        report(a.dir)


if __name__ == "__main__":
    main()
