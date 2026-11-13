#!/usr/bin/env python3
"""Compare `motionforge motion check --json` runs clip by clip and over the
whole set (stdlib only).

    motion_compare.py base=before.json new=after.json [--suffix _new]
        [--clips a,b,c] [--md]

Each run's clips are matched by name (with --suffix, a clip `x_new` in a
later run is matched to `x` in the first). Per measure, the set total and,
per clip, the change from the first run. Lower is better for every defect
(skate, sink, float, through the body, back-and-forth, joint limits);
higher is better for readability (clear share, contrast). A change is kept
only when no defect total rises and the pictures agree (docs/log.md).
"""

import json
import sys

# (key, label, getter, lower_is_better)
MEASURES = [
    ("skate_cm", "skate cm", lambda c: c["feet"]["skate_cm"], True),
    ("sink_cm", "sink cm (deepest)", lambda c: c["feet"]["sink_cm"], True),
    ("float_fr", "floating frames", lambda c: c["feet"]["float_frames"], True),
    ("through_fr", "through-body frames", lambda c: (c.get("through_body") or {}).get("mesh_frames", 0), True),
    ("through_cm", "through-body cm (deepest)", lambda c: (c.get("through_body") or {}).get("mesh_depth_cm", 0), True),
    ("rough_mm", "back-and-forth mm (worst chain mean)", lambda c: max([r["mean_mm"] for r in c.get("rough", [])] or [0]), True),
    ("limit_fr", "joint-limit frames", lambda c: c.get("limits", {}).get("frames", 0), True),
    ("limit_deg", "joint-limit worst deg", lambda c: c.get("limits", {}).get("worst_deg", 0), True),
    ("clear", "key poses clear of the body (share)", lambda c: c.get("read", {}).get("clear", 0), False),
    ("contrast", "key-pose contrast deg", lambda c: c.get("read", {}).get("contrast_deg", 0), False),
]


def main():
    runs, suffix, only, md = [], None, None, False
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--suffix":
            suffix = args[i + 1]
            i += 2
            continue
        if a == "--clips":
            only = set(args[i + 1].split(","))
            i += 2
            continue
        if a == "--md":
            md = True
            i += 1
            continue
        label, path = a.split("=", 1)
        clips = {c["name"]: c for c in json.load(open(path))["clips"]}
        runs.append((label, clips))
        i += 1
    if len(runs) < 2:
        sys.exit(__doc__)
    base = runs[0][1]
    names = sorted(n for n in base if only is None or n in only)

    def find(clips, n):
        if n in clips:
            return clips[n]
        if suffix and n + suffix in clips:
            return clips[n + suffix]
        return None

    sep = " | " if md else "  "
    head = ["measure"] + [lab for lab, _ in runs]
    lines = []
    if md:
        lines.append("| " + " | ".join(head) + " |")
        lines.append("|" + "---|" * len(head))
    else:
        lines.append(sep.join(f"{h:>14}" if k else f"{h:<38}" for k, h in enumerate(head)))
    for key, label, get, lower in MEASURES:
        tot = []
        for _, clips in runs:
            vals = [get(c) for c in (find(clips, n) for n in names) if c is not None]
            t = sum(vals)
            if key in ("clear", "contrast"):
                t = t / max(len(vals), 1)
            tot.append(t)
        cells = [f"{t:.2f}" for t in tot]
        row = [label + (" (mean)" if key in ("clear", "contrast") else " (sum)")] + cells
        if md:
            lines.append("| " + " | ".join(row) + " |")
        else:
            lines.append(sep.join(f"{c:>14}" if k else f"{c:<38}" for k, c in enumerate(row)))
    print("\n".join(lines))
    print()
    # Per clip: which defects rose or fell against the first run.
    for n in names:
        changes = []
        for key, label, get, lower in MEASURES:
            b = get(base[n])
            for lab, clips in runs[1:]:
                c = find(clips, n)
                if c is None:
                    continue
                v = get(c)
                if abs(v - b) > 1e-9:
                    worse = (v > b) if lower else (v < b)
                    changes.append(f"{key} {b:g}->{v:g}{' WORSE' if worse else ''}" + (f" [{lab}]" if len(runs) > 2 else ""))
        if changes:
            print(f"{n}: " + ", ".join(changes))


if __name__ == "__main__":
    main()
