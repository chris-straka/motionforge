"""Draw the fitted rig back over the key-pose drawings (the "are the dots
in the right place" check).

    python python/keypose/overlay.py --report kp.report.json \
        --landmarks keys.json --front a_front.jpg [--side a_side.jpg] \
        --out overlay.png [--width 1600]

Green dots and thin lines: MediaPipe's reading of the drawing. Thick red
(his right) and blue (his left) lines: the rig's joints after the fit,
projected back into the picture with the same view mapping. Yellow: the
fitted sword socket's blade axis. Each figure is labelled with the mean
joint distance as a share of the figure's height (`motionforge keypose`
report, `overlay.mean_err_share`).
"""

import argparse
import json

import cv2
import numpy as np

LIMBS = [(11, 13), (13, 15), (12, 14), (14, 16), (11, 12), (23, 24), (11, 23), (12, 24),
         (23, 25), (25, 27), (24, 26), (26, 28)]


def draw_view(img, figures, keys, view):
    for k, fig in enumerate(figures):
        p = fig["image"]
        for a, b in LIMBS:
            cv2.line(img, (int(p[a][0]), int(p[a][1])), (int(p[b][0]), int(p[b][1])), (60, 170, 60), 2)
        for q in p[11:29]:
            cv2.circle(img, (int(q[0]), int(q[1])), 5, (40, 200, 40), -1)
        ov = keys[k]["overlay"].get(view)
        if not ov:
            continue
        j = {int(r[0]): (int(r[1]), int(r[2])) for r in ov["joints"]}
        for a, b in LIMBS:
            if a in j and b in j:
                right = a % 2 == 0 and b % 2 == 0
                cv2.line(img, j[a], j[b], (40, 40, 220) if right else (220, 120, 30), 6)
        for bl in ov["blades"][1:2]:
            cv2.line(img, (int(bl[0]), int(bl[1])), (int(bl[2]), int(bl[3])), (0, 215, 255), 6)
        hx = int(np.mean([q[0] for q in p[11:29]]))
        cv2.putText(img, f"{keys[k]['role']} {ov['mean_err_share'] * 100:.1f}%", (hx - 90, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (20, 20, 20), 3)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--landmarks", required=True)
    ap.add_argument("--front")
    ap.add_argument("--side")
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=1600)
    a = ap.parse_args()
    rep = json.load(open(a.report))
    lms = json.load(open(a.landmarks))
    rows = []
    for view, path in (("front", a.front), ("side", a.side)):
        if not path or view not in lms["views"]:
            continue
        img = cv2.imread(path)
        img = draw_view(img, lms["views"][view]["figures"], rep["keys"], view)
        cv2.putText(img, view, (20, img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (20, 20, 20), 3)
        s = a.width / img.shape[1]
        rows.append(cv2.resize(img, (a.width, int(img.shape[0] * s)), interpolation=cv2.INTER_AREA))
    cv2.imwrite(a.out, np.vstack(rows))


if __name__ == "__main__":
    main()
