#!/usr/bin/env python3
"""motionforge retarget benchmark (CC0 clips onto CC0 bodies). One command:

    python3 tools/bench/bench.py [--tag NAME] [--only target,target] [--clips a,b]

Source: Quaternius Universal Animation Library 1 (CC0), root-motion GLB.
Targets: the 8 MPFB2 bodies of the shared corpus (child .. tall, slim ..
heavy; built by weightforge's bench/corpus/fetch.sh) plus Quaternius'
female mannequin (UAL2, CC0). Fetching: tools/bench/fetch.sh.

Per target and clip it runs `motionforge standardize` + `animate` (timed)
and measures, from the GLBs alone (tools/bench/glbanim.py, no Blender):
  slide      foot (ankle) drift during the source's foot plants, % of the
             target's leg length per plant (0 = planted feet stay put)
  ground     ankle height error during plants vs the source's (scaled),
             % leg length (floating / sinking)
  dir        limb direction error vs the source, degrees (thigh, shin,
             upper arm, forearm, spine, neck; after facing alignment)
  travel     target hips travel / (source travel * leg ratio); 1 = feet and
             body agree
Baselines: "by-name" copies the clip's local rotations onto the target's
same-named bones with root motion unscaled (playing a clip on another
skeleton with matching names, in an engine or in Blender); "by-name
scaled" also scales root motion by the leg ratio (the usual manual fix).
"motionforge" is `animate` as shipped (contact pass on).

Writes bench/results/<tag>/{summary.json,scorecard.md,<target>_<clip>.png}
(gitignored) and copies summary.json + scorecard.md to docs/bench/.
"""

import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RESULTS = os.path.join(ROOT, "bench", "results")  # gitignored
HISTORY = os.path.join(ROOT, "docs", "bench")  # tracked
sys.path.insert(0, HERE)
import glbanim  # noqa: E402

CLI = os.environ.get("MOTIONFORGE_BIN", os.path.join(ROOT, "rust", "target", "release", "motionforge"))
FORGE = os.environ.get("FORGE_BENCH", os.path.expanduser("~/.cache/forge-bench"))
SRC = os.path.join(FORGE, "clips", "UAL1_Standard_RM.glb")
CLIPS = ["Walk_Loop", "Jog_Fwd_Loop", "Sprint_Loop", "Crouch_Fwd_Loop", "Walk_Formal_Loop", "Idle_Loop",
         "Jump_Start", "Roll", "Punch_Cross", "Sword_Attack", "PickUp_Table"]
SHEET_CLIPS = ["Walk_Loop", "Jog_Fwd_Loop", "Crouch_Fwd_Loop"]

# Unreal-style (source, MPFB) name -> canonical (motionforge standardize).
CANON = {"pelvis": "DEF-spine", "spine_01": "DEF-spine.001", "spine_03": "DEF-spine.003", "neck_01": "DEF-spine.004",
         "head": "DEF-spine.006", "thigh_l": "DEF-thigh.L", "calf_l": "DEF-shin.L", "foot_l": "DEF-foot.L",
         "ball_l": "DEF-toe.L", "thigh_r": "DEF-thigh.R", "calf_r": "DEF-shin.R", "foot_r": "DEF-foot.R",
         "ball_r": "DEF-toe.R", "upperarm_l": "DEF-upper_arm.L", "lowerarm_l": "DEF-forearm.L",
         "hand_l": "DEF-hand.L", "upperarm_r": "DEF-upper_arm.R", "lowerarm_r": "DEF-forearm.R",
         "hand_r": "DEF-hand.R"}
SEGS = [("DEF-thigh.L", "DEF-shin.L"), ("DEF-shin.L", "DEF-foot.L"), ("DEF-thigh.R", "DEF-shin.R"),
        ("DEF-shin.R", "DEF-foot.R"), ("DEF-upper_arm.L", "DEF-forearm.L"), ("DEF-forearm.L", "DEF-hand.L"),
        ("DEF-upper_arm.R", "DEF-forearm.R"), ("DEF-forearm.R", "DEF-hand.R"), ("DEF-spine", "DEF-spine.003"),
        ("DEF-spine.004", "DEF-spine.006")]


def canon_index(g):
    """canonical name -> node index, for raw Unreal-named or standardized rigs."""
    out = {}
    for i, n in enumerate(g.names):
        if n.startswith("DEF-"):
            out.setdefault(n, i)
        c = CANON.get(n.lower())
        if c:
            out.setdefault(c, i)
    return out


def positions(g, anim, times):
    t, T, R, S = g.sample(anim, times=times)
    W = g.world(T, R, S)
    return W[:, :, :3, 3]


def rest_positions(g):
    T, R, S = g.rest_trs()
    return g.world(T[None], R[None], S[None])[0, :, :3, 3]


def leg_len(P, c):
    return float(np.linalg.norm(P[c["DEF-thigh.L"]] - P[c["DEF-shin.L"]]) +
                 np.linalg.norm(P[c["DEF-shin.L"]] - P[c["DEF-foot.L"]]))


def facing(P, c):
    d = sum(P[c[f"DEF-toe.{s}"]] - P[c[f"DEF-foot.{s}"]] for s in "LR")
    d[1] = 0
    return d / np.linalg.norm(d)


def yaw_matrix(a, b):
    """Rotation about +Y taking horizontal a onto b (glTF, Y up)."""
    th = np.arctan2(b[0], b[2]) - np.arctan2(a[0], a[2])
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def plants(P, c, L, times):
    """Foot plants in a source clip: per side, frames where the ankle is
    within 6% of a leg of its lowest height and moves < 30% of a leg / s."""
    out = {}
    dt = np.diff(times).mean() if len(times) > 1 else 1 / 30
    for s in "LR":
        a = P[:, c[f"DEF-foot.{s}"]]
        h = a[:, 1]
        v = np.zeros(len(a))
        v[1:] = np.linalg.norm(np.diff(a[:, [0, 2]], axis=0), axis=1) / dt
        v[0] = v[1] if len(v) > 1 else 0
        on = (h - h.min() < 0.06 * L) & (v < 0.3 * L)
        # intervals of >= 3 frames
        iv, start = [], None
        for f, x in enumerate(list(on) + [False]):
            if x and start is None:
                start = f
            if not x and start is not None:
                if f - start >= 3:
                    iv.append((start, f))
                start = None
        out[s] = iv
    return out


def toe_plants(P, c, L, times):
    """Independent plant test on the toe (ball) joint: within 4% of a leg
    of its lowest height and slower than 20% of a leg / s, >= 3 frames."""
    out = {}
    dt = np.diff(times).mean() if len(times) > 1 else 1 / 30
    for s in "LR":
        a = P[:, c[f"DEF-toe.{s}"]]
        v = np.zeros(len(a))
        v[1:] = np.linalg.norm(np.diff(a[:, [0, 2]], axis=0), axis=1) / dt
        v[0] = v[1] if len(v) > 1 else 0
        on = (a[:, 1] - a[:, 1].min() < 0.04 * L) & (v < 0.2 * L)
        iv, start = [], None
        for f, x in enumerate(list(on) + [False]):
            if x and start is None:
                start = f
            if not x and start is not None:
                if f - start >= 3:
                    iv.append((start, f))
                start = None
        out[s] = iv
    return out


def toe_slide(Pt, ct, Lt, iv):
    xs = []
    for s in "LR":
        for a, b in iv[s]:
            p = Pt[a:b, ct[f"DEF-toe.{s}"]]
            xs.append(float(np.linalg.norm(np.diff(p[:, [0, 2]], axis=0), axis=1).sum()) / Lt * 100)
    return float(np.mean(xs)) if xs else 0.0


def measure(Pt, ct, Lt, rest_t, Ps, cs, Ls, rest_s, iv, tiv=None):
    """Metrics of a target motion Pt against the source motion Ps."""
    scale = Lt / Ls
    slide, ground, n = [], [], 0
    for s in "LR":
        ft, fs = ct[f"DEF-foot.{s}"], cs[f"DEF-foot.{s}"]
        for a, b in iv[s]:
            p = Pt[a:b, ft]
            slide.append(float(np.linalg.norm(p[-1, [0, 2]] - p[0, [0, 2]]) +
                               np.linalg.norm(np.diff(p[:, [0, 2]], axis=0), axis=1).sum()) / 2 / Lt * 100)
            want = rest_t[ft, 1] + (Ps[a:b, fs, 1] - rest_s[fs, 1]) * scale
            ground.append(float(np.abs(p[:, 1] - want).mean()) / Lt * 100)
            n += 1
    Y = yaw_matrix(facing(rest_s, cs), facing(rest_t, ct))
    dirs = []
    for a, b in SEGS:
        if a in ct and b in ct and a in cs and b in cs:
            dt_ = Pt[:, ct[b]] - Pt[:, ct[a]]
            ds = (Ps[:, cs[b]] - Ps[:, cs[a]]) @ Y.T
            cosv = (dt_ * ds).sum(1) / np.maximum(np.linalg.norm(dt_, axis=1) * np.linalg.norm(ds, axis=1), 1e-12)
            dirs.append(np.degrees(np.arccos(np.clip(cosv, -1, 1))).mean())
    hs, ht = cs["DEF-spine"], ct["DEF-spine"]
    trav_s = np.linalg.norm((Ps[-1, hs] - Ps[0, hs])[[0, 2]]) * scale
    trav_t = np.linalg.norm((Pt[-1, ht] - Pt[0, ht])[[0, 2]])
    toe = toe_slide(Pt, ct, Lt, tiv) if tiv else None
    return {"toe_slide": toe, "slide": float(np.mean(slide)) if slide else 0.0, "slide_max": float(np.max(slide)) if slide else 0.0,
            "ground": float(np.mean(ground)) if ground else 0.0, "plants": n,
            "dir": float(np.mean(dirs)) if dirs else float("nan"),
            "travel": float(trav_t / trav_s) if trav_s > 0.05 * Lt else None}


def by_name(src, tgt, anim, times, scale=1.0):
    """Baseline: copy local rotations by bone name; root motion unscaled, or
    scaled by the leg ratio (`scale`, the usual manual fix)."""
    t, T, R, S = src.sample(anim, times=times)
    Tt, Rt, St = tgt.rest_trs()
    F = len(times)
    T2, R2, S2 = np.repeat(Tt[None], F, 0), np.repeat(Rt[None], F, 0), np.repeat(St[None], F, 0)
    sidx = {n.lower(): i for i, n in enumerate(src.names)}
    for i, n in enumerate(tgt.names):
        j = sidx.get(n.lower())
        if j is None:
            continue
        R2[:, i] = R[:, j]
        if n.lower() in ("pelvis", "root"):
            T2[:, i] = T[:, j] * scale
    W = tgt.world(T2, R2, S2)
    return W[:, :, :3, 3]


def plot(path, title, rows, Lt):
    """Side view of both ankles over the clip: planted feet are dots that
    pile up; sliding shows as streaks. One row per method."""
    from PIL import Image, ImageDraw
    W, H, pad = 900, 170, 10
    img = Image.new("RGB", (W, 30 + H * len(rows)), (30, 30, 30))
    d = ImageDraw.Draw(img)
    d.text((8, 8), title + "  (side view of both ankles, every frame; planted = tight dot clusters)", fill=(255, 255, 255))
    allx = np.concatenate([np.concatenate([r[1][:, 0], r[2][:, 0]]) for r in rows])
    lo, hi = allx.min(), allx.max()
    span = max(hi - lo, 1e-6)
    for k, (label, A, B, plant_mask) in enumerate(rows):
        y0 = 30 + k * H
        d.text((8, y0 + 4), label, fill=(230, 230, 230))
        for P, col in ((A, (90, 170, 255)), (B, (255, 150, 80))):
            for f in range(len(P)):
                x = pad + (P[f, 0] - lo) / span * (W - 2 * pad)
                y = y0 + H - 20 - P[f, 1] / Lt * 0.5 * (H - 40)
                r = 2
                d.ellipse([x - r, y - r, x + r, y + r], fill=col)
        d.line([pad, y0 + H - 18, W - pad, y0 + H - 18], fill=(90, 90, 90))
    img.save(path)


def run(cmd):
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=datetime.date.today().isoformat())
    ap.add_argument("--only")
    ap.add_argument("--clips", default=",".join(CLIPS))
    ap.add_argument("--no-contact", action="store_true")
    a = ap.parse_args()
    if not os.path.exists(CLI):
        subprocess.run(["cargo", "build", "--locked", "--release", "-j2"], cwd=os.path.join(ROOT, "rust"), check=True)
    if not os.path.exists(SRC):
        subprocess.run([os.path.join(HERE, "fetch.sh")], check=True)
    clips = a.clips.split(",")
    targets = {}
    corpus = os.path.join(FORGE, "corpus")
    for n in sorted(os.listdir(corpus)) if os.path.isdir(corpus) else []:
        targets[n] = os.path.join(corpus, n, "ref.glb")
    mf = os.path.join(FORGE, "clips", "ual2", "Mannequin_F.glb")
    if os.path.exists(mf):
        targets["quaternius_mannequin_f"] = mf
    if a.only:
        targets = {k: v for k, v in targets.items() if k in a.only.split(",")}
    out = os.path.join(RESULTS, a.tag)
    os.makedirs(out, exist_ok=True)
    src = glbanim.Glb(SRC)
    cs = canon_index(src)
    rest_s = rest_positions(src)
    Ls = leg_len(rest_s, cs)
    summary = {"tag": a.tag, "date": datetime.datetime.now().isoformat(timespec="seconds"),
               "commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                                        text=True).stdout.strip(), "cli": CLI, "source": os.path.basename(SRC),
               "clips": clips, "targets": {}}
    pick = ",".join(clips)
    for name, path in targets.items():
        w = os.path.join(out, name)
        os.makedirs(w, exist_ok=True)
        std, anim = os.path.join(w, "std.glb"), os.path.join(w, "anim.glb")
        p, t_std = run([CLI, "standardize", "--input", path, "--output", std])
        if p.returncode != 0:
            summary["targets"][name] = {"error": "standardize: " + (p.stderr or p.stdout)[-300:]}
            continue
        cmd = [CLI, "animate", "--input", std, "--clips", SRC, "--output", anim, "--pick", pick]
        if a.no_contact:
            cmd.append("--no-contact")
        p, t_anim = run(cmd)
        if p.returncode != 0:
            summary["targets"][name] = {"error": "animate: " + (p.stderr or p.stdout)[-300:]}
            continue
        g = glbanim.Glb(anim)
        ct = canon_index(g)
        rest_t = rest_positions(g)
        Lt = leg_len(rest_t, ct)
        raw = glbanim.Glb(path)
        cr = canon_index(raw)
        rest_r = rest_positions(raw)
        res = {"leg_m": Lt, "leg_ratio": Lt / Ls, "standardize_s": round(t_std, 2), "animate_s": round(t_anim, 2),
               "clips": {}}
        for clip in clips:
            ai = g.anim_names().index(clip)
            times = g.sample(ai)[0]
            Ps = positions(src, clip, times)
            iv = plants(Ps, cs, Ls, times)
            tiv = toe_plants(Ps, cs, Ls, times)
            Pt = positions(g, ai, times)
            Pn = by_name(src, raw, clip, times)
            Lr = leg_len(rest_r, cr)
            Pns = by_name(src, raw, clip, times, Lr / Ls)
            r = {"motionforge": measure(Pt, ct, Lt, rest_t, Ps, cs, Ls, rest_s, iv, tiv),
                 "by-name": measure(Pn, cr, Lr, rest_r, Ps, cs, Ls, rest_s, iv, tiv),
                 "by-name scaled": measure(Pns, cr, Lr, rest_r, Ps, cs, Ls, rest_s, iv, tiv),
                 "source": measure(Ps, cs, Ls, rest_s, Ps, cs, Ls, rest_s, iv, tiv)}
            res["clips"][clip] = r
            if clip in SHEET_CLIPS:
                sc = Lt / Ls
                Y = yaw_matrix(facing(rest_s, cs), facing(rest_t, ct))
                to_side = lambda P, c: np.stack([(P[:, c] @ facing(rest_t, ct)), P[:, c, 1]], 1)
                Pss = Ps @ Y.T * sc
                rows = [("source (scaled)", to_side(Pss, cs["DEF-foot.L"]), to_side(Pss, cs["DEF-foot.R"]), None),
                        ("by-name", to_side(Pn, cr["DEF-foot.L"]), to_side(Pn, cr["DEF-foot.R"]), None),
                        ("by-name scaled", to_side(Pns, cr["DEF-foot.L"]), to_side(Pns, cr["DEF-foot.R"]), None),
                        ("motionforge", to_side(Pt, ct["DEF-foot.L"]), to_side(Pt, ct["DEF-foot.R"]), None)]
                plot(os.path.join(out, f"{name}_{clip}.png"), f"{name} {clip}  slide mf "
                     f"{r['motionforge']['slide']:.1f}% / by-name {r['by-name']['slide']:.1f}% of leg", rows, Lt)
        summary["targets"][name] = res
        print(f"{name}: leg x{Lt / Ls:.2f}, animate {t_anim:.1f}s, slide mf "
              f"{np.mean([c['motionforge']['slide'] for c in res['clips'].values()]):.2f}% by-name "
              f"{np.mean([c['by-name']['slide'] for c in res['clips'].values()]):.2f}%", flush=True)
        with open(os.path.join(out, "summary.json"), "w") as fh:
            json.dump(summary, fh, indent=1)
    write_scorecard(summary, os.path.join(out, "scorecard.md"))
    hist = HISTORY
    os.makedirs(hist, exist_ok=True)
    shutil.copy(os.path.join(out, "summary.json"), os.path.join(hist, f"{a.tag}.json"))
    shutil.copy(os.path.join(out, "scorecard.md"), os.path.join(hist, f"{a.tag}.md"))
    print(open(os.path.join(out, "scorecard.md")).read())


def write_scorecard(s, path):
    T = {k: v for k, v in s["targets"].items() if "clips" in v}

    def mean(method, key, clip=None):
        xs = [c[method][key] for t in T.values() for n, c in t["clips"].items()
              if (clip is None or n == clip) and c[method][key] is not None]
        return float(np.mean(xs)) if xs else float("nan")
    L = [f"# motionforge retarget bench `{s['tag']}` ({s['commit']}, {s['date']})", "",
         f"{len(s['clips'])} CC0 clips ({s['source']}) onto {len(T)} CC0 bodies (leg length x"
         f"{min(t['leg_ratio'] for t in T.values()):.2f}-{max(t['leg_ratio'] for t in T.values()):.2f} of the source). "
         "slide = ankle drift during the source's foot plants, % of leg length per plant; ground = ankle height "
         "error in plants, % leg; dir = limb direction error vs source, deg; travel = hips travel / scaled source "
         "travel (1 = ideal).", "",
         "toe slide = toe path length during the source's toe plants (an independent plant test on another "
         "joint), % leg.", "",
         "| method | slide % | worst slide % | toe slide % | ground % | dir deg | travel |", "|---|---|---|---|---|---|---|"]
    for m in ("source", "by-name", "by-name scaled", "motionforge"):
        worst = max((c[m]["slide_max"] for t in T.values() for c in t["clips"].values()), default=float("nan"))
        L.append(f"| {m} | {mean(m, 'slide'):.2f} | {worst:.1f} | {mean(m, 'toe_slide'):.2f} | {mean(m, 'ground'):.2f} | {mean(m, 'dir'):.2f} | "
                 f"{mean(m, 'travel'):.3f} |")
    L += ["", "Slide % per clip:", "", "| clip | motionforge | by-name scaled | by-name | source |",
          "|---|---|---|---|---|"]
    for clip in s["clips"]:
        L.append(f"| {clip} | {mean('motionforge', 'slide', clip):.2f} | {mean('by-name scaled', 'slide', clip):.2f} | "
                 f"{mean('by-name', 'slide', clip):.2f} | {mean('source', 'slide', clip):.2f} |")
    L += ["", "| target | leg ratio | animate s | slide % | ground % |", "|---|---|---|---|---|"]
    for n, t in T.items():
        L.append(f"| {n} | {t['leg_ratio']:.2f} | {t['animate_s']} | "
                 f"{np.mean([c['motionforge']['slide'] for c in t['clips'].values()]):.2f} | "
                 f"{np.mean([c['motionforge']['ground'] for c in t['clips'].values()]):.2f} |")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
