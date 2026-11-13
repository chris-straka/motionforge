#!/usr/bin/env python3
"""One command from a generated model to a game-ready rigged GLB
(stdlib only).

    python3 tools/tripo_rig.py MODEL.fbx OUT_DIR [--name NAME]
        [--kind auto|human|creature] [--class hero|npc|monster]
        [--seeds 1,2,3] [--tris N] [--height M] [--convention +x|...|none]
        [--clips LIBRARY.glb [--pick walk=walk,attack_1=attack]]
        [--no-fix] [--fix-args "..."] [--keep-tries]
        [--unfuse [--unfuse-args "..."]]
        [--creature-route auto|body|predicted]

Steps, each a tool of its own:

1. `tools/tripo_to_glb.py` (Blender): one mesh, facing forward, on the
   ground, cut to the class's triangle budget.
2. `skintokens rig`, once per seed: a skeleton and weights.
3. `motionforge standardize`: canonical bone names; a human gets the
   twist helpers and hand sockets, a creature keeps its own skeleton.
   Then its clips: a human's are retargeted from `--clips` (`motionforge
   animate`); a four-legged creature gets a walk and a lunge made on its
   own skeleton (`creature-profile`, `creature-gait`), and its range of
   motion written as a pose set.
   A creature skips steps 2 to 5 as written here when skintokens'
   `creature-rig` is installed: `skintokens/tools/creature_route.py`
   takes over after the clean step and rigs it from its own body (its
   parts read from the mesh, a skeleton along them, a skeleton check
   beside the bend gate). `--creature-route predicted` keeps the model's
   skeleton.
   With `--unfuse` (off by default) a human's rig is tried a second
   time with its legs cut apart: `meshkit/tools/unfuse.py` slits the lump
   that holds both thighs (a tunic or coat skirt, wide trousers, thighs
   fused by the generator) up to the hip joints and walls each half, and
   `weights fix` then gives each half to its own leg. The cut copy
   (`try_s<seed>.cut.*`) is ranked with the others, so it is kept only
   when it does better; `report.json` says which won (`best_cut`).
4. `weights fix` (weightforge): repairs what its range-of-motion check
   flags; then `weights check` is the gate. A human is posed through the
   humanoid set, a creature through its own; both through their clips.
5. `rfcheck`: the export contract and the class's budget.

The best try wins: one that passes the gate before one that doesn't,
then the higher score, then the lower seed. OUT_DIR gets `<name>.glb`,
`<name>.sheet.png` (before and after the fix, posed), `report.json`
(every try, every step) and the cleaned mesh. Exit 0: the winner passes
the gate and the contract. Exit 1: it doesn't (it is still written, and
the report says what failed). Exit 2: a step could not run.
"""

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time

# Triangle budgets a little under rfcheck's per-class caps (seams split
# vertices on export, and the cap is per mesh).
TRIS = {"hero": 14000, "npc": 7500, "monster": 18000}


def find(name, explicit, env, fallbacks, first=()):
    # `first`: places to trust before PATH (a `blender` on PATH that is a
    # link into the macOS app starts without its bundled Python).
    for p in [explicit, os.environ.get(env)] + list(first) + [shutil.which(name)] + fallbacks:
        if p and os.path.exists(p):
            return p
    raise SystemExit(f"{name} not found: pass --{name}, set {env}, or put it on PATH")


def run(cmd, timeout=None):
    t = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        code, out, err = r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired as e:
        code, out, err = 124, e.stdout or "", f"timed out after {timeout} s"
        if isinstance(out, bytes):
            out = out.decode(errors="replace")
    return {"code": code, "secs": round(time.time() - t, 1), "out": out, "err": err}


def tail(text, n=300):
    return (text or "").strip()[-n:]


def rank(t):
    """Sort key: gate passed, then contract clean, then fewer failing
    findings (the gate is "none"), then score, then seed, then the uncut
    model before its cut copy."""
    return (not t.get("pass", False), not t.get("contract", False), len(t.get("fails", [])),
            -t.get("score", 0.0), t.get("seed", 0), t.get("cut", False))


def fresh(path):
    """Remove `path` so a step that fails cannot leave an older run's file
    standing in for its output."""
    if os.path.exists(path):
        os.remove(path)


def apply_step_failures(t):
    """A try whose clips failed to retarget fails, whatever the weight gate
    said: the gate then judged the bare rig, not the clips (the run used to
    carry on and could report a pass)."""
    step = t.get("animate") or {}
    if step.get("error") is not None or step.get("code", 0) != 0:
        t["pass"] = False
        t["fails"] = ["animate failed: " + (step.get("error") or f"exit {step.get('code')}")] + list(t.get("fails", []))


def pick_best(tries):
    good = [t for t in tries if t.get("rigged")]
    return sorted(good, key=rank)[0] if good else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--name", default="")
    ap.add_argument("--kind", default="auto", choices=["auto", "human", "creature"])
    ap.add_argument("--class", dest="klass", default="", choices=["", "hero", "npc", "monster"])
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--tris", type=int, default=0)
    ap.add_argument("--height", type=float, default=0.0)
    ap.add_argument("--convention", default="+x")
    ap.add_argument("--no-fix", action="store_true")
    ap.add_argument("--fix-args", default="")
    ap.add_argument("--fix-timeout", type=int, default=1800)
    ap.add_argument("--keep-tries", action="store_true")
    ap.add_argument("--creature-route", default="auto", choices=["auto", "body", "predicted"],
                    help="a creature's rig: body = skintokens' creature-rig (skeleton and weights read from the "
                         "body), predicted = the model's skeleton (the steps below); auto = body when that tool is here")
    ap.add_argument("--creature-rig", default="", help="path to creature-rig (default: $CREATURE_RIG_BIN, PATH, the sibling skintokens repo)")
    ap.add_argument("--unfuse", action="store_true",
                    help="a human: also try each rig with fused legs cut apart (meshkit tools/unfuse.py)")
    ap.add_argument("--unfuse-args", default="")
    ap.add_argument("--unfuse-tool", default="", help="path to unfuse.py (default: the sibling meshkit repo, or $UNFUSE_PY)")
    ap.add_argument("--clips", action="append", default=[], help="a human's clip library (GLB or folder), repeatable")
    ap.add_argument("--pick", default="", help="motionforge animate --pick: which clips, and their names")
    for tool in ("blender", "skintokens", "motionforge", "weights", "rfcheck"):
        ap.add_argument(f"--{tool}", default="")
    a = ap.parse_args()

    home = os.path.expanduser("~")
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sib = os.path.dirname(repo)
    blender = find("blender", a.blender, "BLENDER", [], first=["/Applications/Blender.app/Contents/MacOS/Blender"])
    sk = find("skintokens", a.skintokens, "SKINTOKENS_BIN", [os.path.join(sib, "skintokens/target/release/skintokens")])
    mf = find("motionforge", a.motionforge, "MOTIONFORGE_BIN", [os.path.join(repo, "rust/target/release/motionforge")])
    wf = find("weights", a.weights, "WEIGHTS_BIN", [os.path.join(sib, "weightforge/rust/target/release/weights")])
    rf = find("rfcheck", a.rfcheck, "RFCHECK_BIN", [os.path.join(sib, "rfcheck/target/release/rfcheck"),
                                                     os.path.join(home, ".cargo/bin/rfcheck")])
    unfuse = a.unfuse_tool or os.environ.get("UNFUSE_PY") or os.path.join(sib, "meshkit/tools/unfuse.py")
    if a.unfuse and not os.path.exists(unfuse):
        raise SystemExit(f"unfuse.py not found at {unfuse}: pass --unfuse-tool or set UNFUSE_PY")
    name = a.name or os.path.basename(os.path.dirname(os.path.abspath(a.model))) or "model"
    os.makedirs(a.out, exist_ok=True)
    report = {"model": os.path.abspath(a.model), "name": name, "steps": {}, "tries": []}

    def save():
        with open(os.path.join(a.out, "report.json"), "w") as f:
            json.dump(report, f, indent=1)

    # 1. Clean mesh. The kind decides the class, the class the budget, so
    # a first pass measures and a budget is applied in the same run when
    # the kind was given.
    mesh = os.path.join(a.out, "mesh.glb")
    kind_arg = {"auto": "auto", "human": "biped", "creature": "auto"}[a.kind]

    def clean(tris):
        cmd = [blender, "-b", "--factory-startup", "-P", os.path.join(repo, "tools/tripo_to_glb.py"), "--",
               a.model, mesh, "--kind", kind_arg, "--convention", a.convention, "--tris", str(tris),
               "--height", str(a.height)]
        r = run(cmd, timeout=900)
        line = [ln for ln in r["out"].splitlines() if ln.startswith("TRIPO_TO_GLB ")]
        if r["code"] != 0 or not line:
            report["steps"]["clean"] = {"ok": False, "error": tail(r["err"] or r["out"])}
            save()
            raise SystemExit(f"clean failed: {tail(r['err'] or r['out'])}")
        return json.loads(line[-1][len("TRIPO_TO_GLB "):]), r["secs"]

    info, secs = clean(a.tris or (TRIS[a.klass] if a.klass else 0))
    human = a.kind == "human" or (a.kind == "auto" and info["kind"] == "biped")
    klass = a.klass or ("npc" if human else "monster")
    budget = a.tris or TRIS[klass]
    if info["tris"] > budget:
        info, secs = clean(budget)
    report["steps"]["clean"] = dict(info, ok=True, secs=secs)
    report.update(kind="human" if human else "creature", **{"class": klass})
    h = info.get("holes") or {}
    print(f"{name}: {report['kind']} ({klass}), facing decided by {info['decided_by']}, "
          f"{info['source_tris']} -> {info['tris']} tris; holes: {h.get('found', '?')} found, "
          f"{h.get('filled', 0)} filled, {h.get('left', '?')} left")
    save()

    # A creature is rigged from its own body when skintokens' creature-rig
    # is here: its parts read from the mesh, a skeleton laid along them,
    # weights through the body, its own range of motion and skeleton check
    # (skintokens `tools/creature_route.py`, same outputs as this script).
    # The model's skeleton, the steps below, stays as `--creature-route
    # predicted`.
    if not human and a.creature_route != "predicted":
        route = os.environ.get("CREATURE_ROUTE") or os.path.join(sib, "skintokens/tools/creature_route.py")
        cr = (a.creature_rig or os.environ.get("CREATURE_RIG_BIN") or shutil.which("creature-rig")
              or os.path.join(sib, "skintokens/target/release/creature-rig"))
        if os.path.exists(route) and os.path.exists(cr):
            cmd = [sys.executable, route, a.model, a.out, "--name", name, "--class", klass, "--clean-mesh", mesh,
                   "--creature-rig", cr, "--skintokens", sk, "--motionforge", mf, "--weights", wf, "--rfcheck", rf,
                   "--seed", (a.seeds.split(",")[0] or "1"), "--fix-timeout", str(a.fix_timeout)]
            if a.no_fix:
                cmd.append("--no-fix")
            if a.fix_args:
                cmd += ["--fix-args", a.fix_args]
            if a.keep_tries:
                cmd.append("--keep-tries")
            return subprocess.call(cmd)
        if a.creature_route == "body":
            raise SystemExit(f"--creature-route body: creature_route.py or creature-rig not found ({route}, {cr})")

    def finish(t, base, std):
        """Clips, the check, the fix, the gate and the contract for one
        rigged model (`std`); the result is `base + ".glb"`."""
        fixed = base + ".glb"
        label = f"seed {t['seed']}" + (" (cut)" if t.get("cut") else "")
        poses = []
        if not human:
            # A creature: its parts read from the skeleton, a walk and a
            # lunge made on it, and a quadruped's range of motion to judge
            # it by (without one, weightforge bends every bone 45 degrees
            # each way, which no creature's weights survive).
            prof, clean_glb, moving, rom = base + ".creature.json", base + ".clean.glb", base + ".moving.glb", base + ".rom.ron"
            r = run([mf, "creature-profile", "--rig", std, "--output", prof, "--clean-glb", clean_glb,
                     "--report", base + ".profile.json", "--name", name])
            t["profile"] = {"code": r["code"], "secs": r["secs"]}
            if r["code"] == 0:
                r = run([mf, "creature-gait", "--creature", prof, "--glb-in", clean_glb, "--glb-out", moving,
                         "--rom", rom])
                t["gait"] = {"code": r["code"], "parts": [ln for ln in r["out"].splitlines() if ": " in ln][:8]}
                if r["code"] == 0:
                    std, poses = moving, ["--poses", rom]
                else:
                    t["gait"]["error"] = tail(r["err"] or r["out"])
            else:
                t["profile"]["error"] = tail(r["err"] or r["out"])
        elif a.clips:
            # A human: clips retargeted from a library before the check, so
            # the gate sees them too.
            moving = base + ".moving.glb"
            fresh(moving)
            cmd = [mf, "animate", "--input", std, "--output", moving]
            for c in a.clips:
                cmd += ["--clips", c]
            if a.pick:
                cmd += ["--pick", a.pick]
            r = run(cmd, timeout=1800)
            t["animate"] = {"code": r["code"], "secs": r["secs"]}
            if r["code"] == 0 and os.path.exists(moving):
                std = moving
            else:
                t["animate"]["error"] = tail(r["err"] or r["out"])
        t["judged_on"] = "quadruped range of motion" if poses else ("humanoid range of motion" if human else "any-skeleton fallback")
        raw = check(wf, std, poses)
        t["raw"] = raw
        if a.no_fix or raw["pass"]:
            shutil.copyfile(std, fixed)
        else:
            # A file left by an earlier run must not pass for this fix.
            fresh(fixed)
            cmd = [wf, "fix", std, "--out", fixed, "--sheet", base + ".sheet.png"] + poses + shlex.split(a.fix_args)
            r = run(cmd, timeout=a.fix_timeout)
            t["fix"] = {"code": r["code"], "secs": r["secs"]}
            if r["code"] != 0 or not os.path.exists(fixed):
                t["fix"]["error"] = tail(r["err"] or r["out"])
                shutil.copyfile(std, fixed)
        final = check(wf, fixed, poses)
        if poses:
            shutil.copyfile(poses[1], os.path.join(a.out, name + ".rom.ron"))
        t.update(score=final["score"], **{"pass": final["pass"]}, fails=final["fails"])
        apply_step_failures(t)
        r = run([rf, "--json", "--class", klass, fixed])
        t["contract"] = r["code"] == 0
        t["rfcheck"] = contract_notes(r["out"])
        print(f"  {label}: {raw['score']} -> {t['score']} "
              f"({'passes' if t['pass'] else str(len(t['fails'])) + ' failing'}; "
              f"contract {'clean' if t['contract'] else 'broken: ' + ', '.join(t['rfcheck'][:4])})")
        save()

    # 2-5, once per seed.
    for seed in [int(s) for s in a.seeds.split(",") if s]:
        t = {"seed": seed, "rigged": False}
        report["tries"].append(t)
        base = os.path.join(a.out, f"try_s{seed}")
        rig, std = base + ".rig.glb", base + ".std.glb"
        r = run([sk, "rig", mesh, rig, "--report", base + ".skintokens.json",
                 "--class", "humanoid" if human else "quadruped", "--seed", str(seed)], timeout=3600)
        t["rig"] = {"code": r["code"], "secs": r["secs"]}
        if r["code"] != 0 or not os.path.exists(rig):
            try:
                with open(base + ".skintokens.json") as f:
                    t["rig"]["error"] = json.load(f).get("error") or tail(r["err"])
            except (OSError, ValueError):
                t["rig"]["error"] = tail(r["err"])
            print(f"  seed {seed}: no rig: {t['rig']['error']}")
            save()
            continue
        r = run([mf, "standardize", "--input", rig, "--output", std, "--class",
                 "humanoid" if human else "quadruped", "--report", base + ".standardize.json"])
        t["standardize"] = {"code": r["code"], "secs": r["secs"]}
        if r["code"] != 0 or not os.path.exists(std):
            t["standardize"]["error"] = tail(r["err"] or r["out"])
            print(f"  seed {seed}: standardize failed: {t['standardize']['error']}")
            save()
            continue
        t["rigged"] = True
        finish(t, base, std)
        if a.unfuse and human:
            # The same rig with fused legs cut apart (meshkit unfuse): a
            # second try, so the cut is kept only when it ranks better.
            cut = base + ".cut.std.glb"
            r = run([blender, "-b", "--factory-startup", "--python", unfuse, "--", std, cut,
                     "--report", base + ".unfuse.json"] + shlex.split(a.unfuse_args), timeout=900)
            line = [ln for ln in r["out"].splitlines() if ln.startswith("UNFUSE ")]
            info = json.loads(line[-1][len("UNFUSE "):]) if r["code"] == 0 and line else {}
            if info.get("cut") and os.path.exists(cut):
                t2 = {"seed": seed, "rigged": True, "cut": True, "rig": t["rig"], "standardize": t["standardize"],
                      "unfuse": {k: info[k] for k in ("legs", "verts_before", "verts") if k in info}}
                report["tries"].append(t2)
                finish(t2, base + ".cut", cut)
            else:
                t["unfuse"] = {"cut": False, "why": (info.get("legs") or {}).get("why_not") or tail(r["err"] or r["out"], 160)}
                print(f"  seed {seed}: not cut: {t['unfuse']['why']}")
                save()

    best = pick_best(report["tries"])
    if not best:
        save()
        print(f"{name}: no try gave a rig")
        return 2
    base = os.path.join(a.out, f"try_s{best['seed']}" + (".cut" if best.get("cut") else ""))
    shutil.copyfile(base + ".glb", os.path.join(a.out, name + ".glb"))
    if os.path.exists(base + ".sheet.png"):
        shutil.copyfile(base + ".sheet.png", os.path.join(a.out, name + ".sheet.png"))
    else:
        rom = os.path.join(a.out, name + ".rom.ron")
        run([wf, "sheet", os.path.join(a.out, name + ".glb"), "--out", os.path.join(a.out, name + ".sheet.png")]
            + (["--poses", rom] if os.path.exists(rom) else []))
    report["best_seed"] = best["seed"]
    report["best_cut"] = bool(best.get("cut"))
    report["ok"] = bool(best["pass"] and best["contract"])
    save()
    if not a.keep_tries:
        for t in report["tries"]:
            for ext in (".rig.glb", ".std.glb", ".clean.glb", ".moving.glb", ".glb"):
                p = os.path.join(a.out, f"try_s{t['seed']}{'.cut' if t.get('cut') else ''}{ext}")
                if os.path.exists(p):
                    os.remove(p)
    print(f"{name}: seed {best['seed']}, score {best['score']}, "
          f"{'PASSES' if report['ok'] else 'FAILS: ' + '; '.join(best['fails'][:6] + ([] if best['contract'] else best['rfcheck'][:4]))}"
          f" -> {os.path.join(a.out, name + '.glb')}")
    return 0 if report["ok"] else 1


def check(wf, glb, poses=()):
    r = run([wf, "check", glb, "--json"] + list(poses), timeout=1800)
    try:
        d = json.loads(r["out"])
    except ValueError:
        return {"score": 0.0, "pass": False, "fails": ["check did not run: " + tail(r["err"], 120)], "class": "?"}
    fails = [f"{f.get('code')} {f.get('region')}: {f.get('detail')}" for f in d.get("findings", [])
             if f.get("severity") == "fail"]
    return {"score": d.get("score", 0.0), "pass": bool(d.get("pass")), "fails": fails, "class": d.get("class"),
            "verts": d.get("verts"), "bones": d.get("bones"), "poses": d.get("poses")}


def contract_notes(out):
    try:
        d = json.loads(out)
    except ValueError:
        return [tail(out, 120)]
    d = d[0] if isinstance(d, list) and d else d
    return [f"{x.get('code')}: {x.get('detail')}" for x in (d.get("diags") or []) if not str(x.get("code", "")).startswith("P_")]


if __name__ == "__main__":
    sys.exit(main())
