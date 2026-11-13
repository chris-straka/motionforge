#!/usr/bin/env python3
"""Where do the two views of a creature video disagree? (stdlib only)

    python3 tools/creature_views.py --creature creature.json --detections det.json \
        [--lags -12,-8,-4,-2,0,2,4,8,12] [--group-lag 8]

Reads detections the way `motionforge creature-solve` does (the first two
of the creature's views; use `creature-solve --aligned-out` to look at
detections after the camera fit and the time slide) and prints:

- the solve's height-disagreement motion ratio at a range of constant
  lags of the second view (a clip out of step has its minimum away from 0);
- per leg (each foot's chain up to the body) and for the rest of the
  body: the ratio at lag 0, the best lag in +-`--group-lag` and its ratio,
  and the finder's mean confidence per view (a joint hidden in one view
  is guessed there: low confidence, or a guess that disagrees);
- the pooled ratio when every group takes its own best lag (if even that
  stays over the solve's limit, sliding in time cannot make the views
  agree).
"""

import argparse
import json
import math


def sub(a, b): return [a[i] - b[i] for i in range(3)]
def add(a, b): return [a[i] + b[i] for i in range(3)]
def sc(a, s): return [x * s for x in a]
def dot(a, b): return sum(a[i] * b[i] for i in range(3))
def cross(a, b): return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def norm(a):
    n = math.sqrt(dot(a, a))
    return sc(a, 1.0 / n)


def ray(view, u, v):
    """Same pinhole as motion_core::creature::camera::View::ray."""
    fwd = norm(sub(view["look_at"], view["position"]))
    right = cross(fwd, [0.0, 0.0, 1.0])
    right = norm(right) if dot(right, right) > 1e-18 else [1.0, 0.0, 0.0]
    up = norm(cross(right, fwd))
    th = math.tan(math.radians(view["fov_y_deg"]) * 0.5)
    tw = th * view["width"] / view["height"]
    xn = u / view["width"] * 2.0 - 1.0
    yn = 1.0 - v / view["height"] * 2.0
    d = add(add(sc(right, xn * tw), sc(up, yn * th)), fwd)
    return view["position"], norm(d)


def closest(a, b):
    p, u = a
    q, v = b
    w = sub(p, q)
    uu, uv, vv, uw, vw = dot(u, u), dot(u, v), dot(v, v), dot(u, w), dot(v, w)
    den = uu * vv - uv * uv
    if abs(den) < 1e-12:
        s, t = 0.0, (vw / vv if vv > 0 else 0.0)
    else:
        s, t = (uv * vw - vv * uw) / den, (uu * vw - uv * uw) / den
    return add(p, sc(u, s)), add(q, sc(v, t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--creature", required=True)
    ap.add_argument("--detections", required=True)
    ap.add_argument("--lags", default="-12,-8,-4,-2,-1,0,1,2,4,8,12")
    ap.add_argument("--group-lag", type=int, default=8)
    ap.add_argument("--min-conf", type=float, default=0.2)
    args = ap.parse_args()
    cre = json.load(open(args.creature))
    det = json.load(open(args.detections))
    views = {v["name"]: v for v in cre["views"]}
    pair = [(i, views[n]) for i, n in enumerate(det["views"]) if n in views][:2]
    if len(pair) < 2:
        raise SystemExit("detections need two of the creature's views")
    (ai, va), (bi, vb) = pair
    names = [b["name"] for b in cre["bones"]]
    parent = {b["name"]: b["parent"] for b in cre["bones"]}
    col = {n: det["joints"].index(n) for n in names if n in det["joints"]}
    frames = det["frames"]
    n = len(frames)

    # Groups: each foot's chain up to the first joint with other children.
    children = {nm: [c for c in names if parent[c] == nm] for nm in names}
    groups = {}
    taken = set()
    for foot in cre["feet"]:
        chain = [foot]
        cur = parent[foot]
        while cur is not None and len(children[cur]) == 1:
            chain.append(cur)
            cur = parent[cur]
        groups[f"leg {foot}"] = chain
        taken.update(chain)
    groups["body (rest)"] = [nm for nm in names if nm not in taken]

    def sums(nm, lag):
        """(var sum of disagreement, var sum of height, samples) for one joint."""
        j = col.get(nm)
        if j is None:
            return 0.0, 0.0, 0
        g, z = [], []
        for f in range(n):
            fb = f + lag
            if fb < 0 or fb >= n:
                continue
            a, b = frames[f][ai][j], frames[fb][bi][j]
            if a[2] < args.min_conf or b[2] < args.min_conf:
                continue
            pa, pb = closest(ray(va, a[0], a[1]), ray(vb, b[0], b[1]))
            g.append(pa[2] - pb[2])
            z.append(0.5 * (pa[2] + pb[2]))
        if len(g) < 2:
            return 0.0, 0.0, 0
        mg, mz = sum(g) / len(g), sum(z) / len(z)
        return sum((x - mg) ** 2 for x in g), sum((x - mz) ** 2 for x in z), len(g)

    def ratio(joints, lag):
        s = [sums(nm, lag) for nm in joints]
        gz = sum(x[1] for x in s)
        return (sum(x[0] for x in s) / gz if gz > 0 else 0.0), s

    print(f"views {det['views'][ai]} / {det['views'][bi]}, {n} frames; lag = frames the second view runs late")
    print("motion ratio at a constant lag:")
    print("  " + "  ".join(f"{lag:+d}: {ratio(names, lag)[0]:.3f}" for lag in [int(x) for x in args.lags.split(",")]))
    print(f"per group (ratio at 0 | best lag in +-{args.group_lag} | mean confidence {det['views'][ai]} / {det['views'][bi]}):")
    tg = tz = 0.0
    for gname, joints in groups.items():
        r0, _ = ratio(joints, 0)
        best = min(range(-args.group_lag, args.group_lag + 1), key=lambda l: (ratio(joints, l)[0], abs(l)))
        rb, s = ratio(joints, best)
        tg += sum(x[0] for x in s)
        tz += sum(x[1] for x in s)
        js = [col[nm] for nm in joints if nm in col]
        ca = sum(fr[ai][j][2] for fr in frames for j in js) / max(1, n * len(js))
        cb = sum(fr[bi][j][2] for fr in frames for j in js) / max(1, n * len(js))
        print(f"  {gname:22s} {r0:6.2f} | {rb:6.2f} at {best:+d} | {ca:.2f} / {cb:.2f}")
    print(f"pooled with every group at its own best lag: {tg / tz if tz else 0.0:.3f}")


if __name__ == "__main__":
    main()
