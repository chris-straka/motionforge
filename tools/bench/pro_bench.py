#!/usr/bin/env python3
"""Correctness benchmark against professional clips (stdlib only; drives the
motionforge CLI).

    pro_bench.py calibrate [--out DIR] [--libs ual1,ual2]
    pro_bench.py run --target andras=RIG.glb [--target name=RIG.glb ...]
                     [--libs ual1,ual2] [--clips a,b] [--stages ...] [--out DIR]
                     [--jobs N]
    pro_bench.py report DIR | recompare DIR | why DIR TARGET STAGE
    pro_bench.py rescore DIR [--target NAME]   (GLBs as they are, through-body kept)

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
  timing   the body's speed curve lines up with the pro's within 1 frame
           (not for stylize, which retimes on purpose)
  pose     knee and elbow bends, wrists and ankles within 15 deg of the
           pro's on average, each joint (not for stylize)
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
    "through_share": 0.05, "limit_deg": 5.0, "limit_share": 0.05, "rough_mm": 0.5, "lag_frames": 1,
    "pose_deg": 15.0,
}
CHECKS = ["plants", "hold", "skate", "sink", "float", "through", "limits", "rough", "timing", "pose"]


def run(cmd, out=None):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)}: {(p.stderr or p.stdout)[-400:]}")
    if out:
        with open(out, "w") as fh:
            fh.write(p.stdout)
    return p.stdout


def check(glb, out, clips=None, keep_through=False):
    """`motion check --tracks` into `out`. keep_through: skip the slow
    limbs-through-the-body check and keep that part of the JSON already at
    `out` (for re-scoring the other measures on unchanged GLBs)."""
    cmd = [CLI, "motion", "check", glb, "--tracks"]
    if clips:
        cmd += ["--clip", ",".join(clips)]
    if not keep_through:
        run(cmd, out)
        return json.load(open(out))
    old = {c["name"]: c for c in json.load(open(out))["clips"]}
    new = json.loads(run(cmd + ["--no-through"]))
    for c in new["clips"]:
        o = old[c["name"]]
        if "through_body" in o:
            c["through_body"] = o["through_body"]
        c["detail"]["through"] = o["detail"]["through"]
    json.dump(new, open(out, "w"))
    return new


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


def lag(a, b, n, reach=6):
    """Frames b's curve runs behind a's (negative: ahead): the shift in
    -reach..reach with the best correlation; None for a flat curve."""
    if not a or not b or n < 2 * reach + 4:
        return None
    a, b = a[:n], b[:n]

    def corr(s):
        xs = [(a[f], b[f + s]) for f in range(max(0, -s), min(n, n - s))]
        ma = sum(x for x, _ in xs) / len(xs)
        mb = sum(y for _, y in xs) / len(xs)
        va = sum((x - ma) ** 2 for x, _ in xs)
        vb = sum((y - mb) ** 2 for _, y in xs)
        if va < 1e-9 or vb < 1e-9:
            return None
        return sum((x - ma) * (y - mb) for x, y in xs) / (va * vb) ** 0.5
    best = None
    for s in sorted(range(-reach, reach + 1), key=abs):
        c = corr(s)
        if c is not None and (best is None or c > best[0] + 1e-9):
            best = (c, s)
    return best[1] if best else None


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
    lim, which = 0, {}
    for f in range(tgt["frames"] if retime else n):
        past = False
        for w, v in td["limits"].items():
            ref = pd["limits"].get(w)
            base = (max(ref) if ref else 0.0) if retime else (ref[f] if ref and f < len(ref) else 0.0)
            if f < len(v) and v[f] > base + TOL["limit_deg"]:
                past = True
                which[w] = which.get(w, 0) + 1
        lim += past
    m["limits_new"] = lim
    m["limits_which"] = which
    m["sink_cm"] = tgt["feet"]["sink_cm"]
    m["rough_mm"] = round(worst_rough(tgt), 3)
    m["rough_pro_mm"] = round(worst_rough(pro), 3)
    # Key poses (speed minima): mean distance from each pro key to the
    # nearest target key.
    pk, tk = pro["read"]["keys"], tgt["read"]["keys"]
    inner = [k for k in pk if 0 < k < n - 1] or pk
    m["keys_frames"] = round(sum(min(abs(k - t) for t in tk) for k in inner) / len(inner), 2) if inner and tk else None
    # Timing: the shift (frames) that best lines the body's speed curve up
    # with the pro's. The key poses are that curve's minima; on a loop or
    # an idle they are shallow and move by many frames for nothing, the
    # curve's shift does not.
    m["lag_frames"] = lag(pd.get("speed"), td.get("speed"), n)
    # Pose: joint angles against the pro's (bends of knees and elbows,
    # wrists, ankles), mean difference per joint, the worst joint.
    worst, which_joint = 0.0, ""
    for w, v in td.get("angles", {}).items():
        ref = pd.get("angles", {}).get(w)
        if not ref or w.endswith(" side") or w in ("torso", "head"):
            continue
        d = sum(abs(v[f] - ref[f]) for f in range(n)) / max(n, 1)
        if d > worst:
            worst, which_joint = d, w
    m["pose_deg"] = round(worst, 1)
    m["pose_joint"] = which_joint
    # Pass/fail.
    fail = []
    if retime:
        m["plant_iou"] = m["hold_cm"] = m["hold_pro_cm"] = m["keys_frames"] = m["lag_frames"] = None
        m["pose_deg"] = None
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
    if m["lag_frames"] is not None and abs(m["lag_frames"]) > TOL["lag_frames"]:
        fail.append("timing")
    if m["pose_deg"] is not None and m["pose_deg"] > TOL["pose_deg"]:
        fail.append("pose")
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


def rescore(out, only, jobs):
    """Score a run's stage GLBs again (the through-body part kept), then
    compare: after a change to any other measure."""
    old = json.load(open(os.path.join(out, "results.json")))
    work = []
    for lib in LIBS:
        if os.path.exists(os.path.join(out, f"pro_{lib}.json")):
            work.append((LIBS[lib], os.path.join(out, f"pro_{lib}.json"), None, False))
            for target in old["targets"]:
                if only and target not in only:
                    continue
                for s in old["stages"]:
                    g, js = (os.path.join(out, target, lib, f"{s}.{e}") for e in ("glb", "json"))
                    if os.path.exists(g) and os.path.exists(js):
                        names = [c["name"] for c in json.load(open(js))["clips"]]
                        work.append((g, js, names, True))
    with cf.ThreadPoolExecutor(max_workers=jobs) as ex:
        for f in [ex.submit(check, *w) for w in work]:
            f.result()
    recompare(out)


def recompare(out):
    """Rebuild results.json from the motion check JSONs a run left in
    `out` (no CLI runs): for a changed check or tolerance."""
    old = json.load(open(os.path.join(out, "results.json")))
    results = {"commit": old["commit"], "date": old["date"], "stages": old["stages"], "tol": TOL, "targets": {}}
    for target in old["targets"]:
        for lib in LIBS:
            pj = os.path.join(out, f"pro_{lib}.json")
            if not os.path.exists(pj):
                continue
            pro = {c["name"]: c for c in json.load(open(pj))["clips"]}
            for s in old["stages"]:
                sj = os.path.join(out, target, lib, f"{s}.json")
                if not os.path.exists(sj):
                    continue
                tj = json.load(open(sj))
                T = results["targets"].setdefault(target, {"height_m": tj["height_m"], "clips": {}})
                for c in tj["clips"]:
                    T["clips"].setdefault(f"{lib}/{c['name']}", {})[s] = compare(pro[c["name"]], c, s)
    json.dump(results, open(os.path.join(out, "results.json"), "w"), indent=1)


def why(out, target, stage):
    """Per check, the clips that fail it at a stage, with the numbers."""
    R = json.load(open(os.path.join(out, "results.json")))
    clips = R["targets"][target]["clips"]
    show = {"plants": ["plant_iou"], "hold": ["hold_cm", "hold_pro_cm"], "skate": ["skate_cm", "skate_pro_cm"],
            "sink": ["sink_new", "sink_cm"], "float": ["float_extra"], "through": ["through_new"],
            "limits": ["limits_new", "limits_which"], "rough": ["rough_mm", "rough_pro_mm"],
            "timing": ["lag_frames", "keys_frames"], "pose": ["pose_deg", "pose_joint"]}
    for ch in CHECKS:
        rows = [(n, c[stage]) for n, c in sorted(clips.items()) if stage in c and ch in c[stage]["fail"]]
        print(f"{ch}: {len(rows)} clips")
        for n, m in rows:
            print(f"  {n}: " + ", ".join(f"{k} {m[k]}" for k in show[ch]))
    labels = {}
    for c in clips.values():
        if stage in c:
            for w, k in c[stage].get("limits_which", {}).items():
                labels[w] = labels.get(w, 0) + k
    print("limit frames past the pro's, by joint:", dict(sorted(labels.items(), key=lambda x: -x[1])))


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
    p = sub.add_parser("recompare", help="results.json again from a run's motion check JSONs, then the report")
    p.add_argument("dir")
    p = sub.add_parser("rescore", help="motion check again on a run's GLBs (through-body kept), then recompare")
    p.add_argument("dir")
    p.add_argument("--target", action="append", default=[])
    p.add_argument("--jobs", type=int, default=4)
    p = sub.add_parser("why", help="the clips failing each check at a stage, with numbers")
    p.add_argument("dir")
    p.add_argument("target")
    p.add_argument("stage")
    a = ap.parse_args()
    if a.cmd in ("calibrate", "run", "rescore") and not os.path.exists(CLI):
        subprocess.run(["cargo", "build", "--locked", "--release"], cwd=os.path.join(ROOT, "rust"), check=True)
    if a.cmd == "calibrate":
        calibrate(a)
    elif a.cmd == "run":
        main_run(a)
    elif a.cmd == "recompare":
        recompare(a.dir)
        report(a.dir)
    elif a.cmd == "rescore":
        rescore(a.dir, a.target, a.jobs)
        report(a.dir)
    elif a.cmd == "why":
        why(a.dir, a.target, a.stage)
    else:
        report(a.dir)


if __name__ == "__main__":
    main()
