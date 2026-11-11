#!/usr/bin/env python3
"""One command from a generated model to a game-ready rigged GLB
(stdlib only).

    python3 tools/tripo_rig.py MODEL.fbx OUT_DIR [--name NAME]
        [--kind auto|human|creature] [--class hero|npc|monster]
        [--seeds 1,2,3] [--tris N] [--height M] [--convention +x|...|none]
        [--no-fix] [--fix-args "..."] [--keep-tries]

Steps, each a tool of its own:

1. `tools/tripo_to_glb.py` (Blender): one mesh, facing forward, on the
   ground, cut to the class's triangle budget.
2. `skintokens rig`, once per seed: a skeleton and weights.
3. `motionforge standardize`: canonical bone names; a human gets the
   twist helpers and hand sockets, a creature keeps its own skeleton.
4. `weights fix` (weightforge): repairs what its range-of-motion check
   flags; then `weights check` is the gate.
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
    """Sort key: gate passed, then contract clean, then score, then seed."""
    return (not t.get("pass", False), not t.get("contract", False), -t.get("score", 0.0), t.get("seed", 0))


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

    # 2-5, once per seed.
    for seed in [int(s) for s in a.seeds.split(",") if s]:
        t = {"seed": seed, "rigged": False}
        report["tries"].append(t)
        base = os.path.join(a.out, f"try_s{seed}")
        rig, std, fixed = base + ".rig.glb", base + ".std.glb", base + ".glb"
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
        raw = check(wf, std)
        t["raw"] = raw
        if a.no_fix or raw["pass"]:
            shutil.copyfile(std, fixed)
        else:
            cmd = [wf, "fix", std, "--out", fixed, "--sheet", base + ".sheet.png"] + shlex.split(a.fix_args)
            r = run(cmd, timeout=a.fix_timeout)
            t["fix"] = {"code": r["code"], "secs": r["secs"]}
            if not os.path.exists(fixed):
                t["fix"]["error"] = tail(r["err"] or r["out"])
                shutil.copyfile(std, fixed)
        final = check(wf, fixed)
        t.update(score=final["score"], **{"pass": final["pass"]}, fails=final["fails"])
        r = run([rf, "--json", "--class", klass, fixed])
        t["contract"] = r["code"] == 0
        t["rfcheck"] = contract_notes(r["out"])
        print(f"  seed {seed}: {raw['score']} -> {t['score']} "
              f"({'passes' if t['pass'] else str(len(t['fails'])) + ' failing'}; "
              f"contract {'clean' if t['contract'] else 'broken: ' + ', '.join(t['rfcheck'][:4])})")
        save()

    best = pick_best(report["tries"])
    if not best:
        save()
        print(f"{name}: no try gave a rig")
        return 2
    base = os.path.join(a.out, f"try_s{best['seed']}")
    shutil.copyfile(base + ".glb", os.path.join(a.out, name + ".glb"))
    if os.path.exists(base + ".sheet.png"):
        shutil.copyfile(base + ".sheet.png", os.path.join(a.out, name + ".sheet.png"))
    else:
        run([wf, "sheet", os.path.join(a.out, name + ".glb"), "--out", os.path.join(a.out, name + ".sheet.png")])
    report["best_seed"] = best["seed"]
    report["ok"] = bool(best["pass"] and best["contract"])
    save()
    if not a.keep_tries:
        for t in report["tries"]:
            for ext in (".rig.glb", ".std.glb", ".glb"):
                p = os.path.join(a.out, f"try_s{t['seed']}{ext}")
                if os.path.exists(p):
                    os.remove(p)
    print(f"{name}: seed {best['seed']}, score {best['score']}, "
          f"{'PASSES' if report['ok'] else 'FAILS: ' + '; '.join(best['fails'][:6] + ([] if best['contract'] else best['rfcheck'][:4]))}"
          f" -> {os.path.join(a.out, name + '.glb')}")
    return 0 if report["ok"] else 1


def check(wf, glb):
    r = run([wf, "check", glb, "--json"], timeout=1800)
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
