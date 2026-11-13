"""Read the body (and the blade) in key-pose drawings; write landmarks JSON
for `motionforge keypose`.

    python python/keypose/landmarks.py --model pose_landmarker_heavy.task \
        --front a_front.jpg [--side a_side.jpg] --poses 3 --out keys.json \
        [--debug DIR]

A sheet is one picture with `--poses` figures side by side, left to right
in time (for an attack: load, strike, follow-through).
`--front` is drawn facing the viewer, `--side` from the character's left
(he faces image-left). Either may be missing; a key pose then has one view.

Per figure: MediaPipe Pose Landmarker's 33 image landmarks (pixels, y
down) with visibility, its 33 world landmarks (metres, hip-centred,
MediaPipe's camera axes), and the blade if one can be read: the longest
dark, grey, thin region next to the wrists, as a 2D segment from the end
nearer the hands (`hilt`) to the far end (`tip`).

MediaPipe (Apache-2.0, PyPI `mediapipe`) and its model file
(https://storage.googleapis.com/mediapipe-models/pose_landmarker/
pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task, Apache-2.0)
are not vendored: `docs/keypose.md` has the install lines.
"""

import argparse
import json
import os
import sys

import numpy as np


def read_rgb(path):
    # mp.Image.numpy_view() rows did not line up with the file's (a padded
    # stride); the picture analysis reads the file itself.
    import cv2

    return cv2.imread(path)[:, :, ::-1].copy()


def detect(model, path, n_poses):
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    img = mp.Image.create_from_file(path)
    opts = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=n_poses,
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3,
    )
    with vision.PoseLandmarker.create_from_options(opts) as lm:
        res = lm.detect(img)
    w, h = img.width, img.height
    rgb = read_rgb(path)
    figs = []
    for i, pts in enumerate(res.pose_landmarks):
        world = res.pose_world_landmarks[i]
        figs.append(
            {
                "image": [[round(p.x * w, 2), round(p.y * h, 2), round(p.visibility or 0.0, 3)] for p in pts],
                "world": [[round(p.x, 4), round(p.y, 4), round(p.z, 4)] for p in world],
            }
        )
    return figs, rgb, (w, h)


def split_detect(model, path, n_poses):
    """Fallback: cut the sheet into equal panels and read one body in each."""
    rgb = read_rgb(path)
    h, w = rgb.shape[:2]
    figs = []
    tmp_dir = os.environ.get("TMPDIR", "/tmp")
    for k in range(n_poses):
        x0 = k * w // n_poses
        x1 = (k + 1) * w // n_poses
        crop = np.ascontiguousarray(rgb[:, x0:x1])
        p = os.path.join(tmp_dir, f"keypose_panel_{os.getpid()}_{k}.png")
        import cv2

        cv2.imwrite(p, crop[:, :, ::-1])
        f, _, _ = detect(model, p, 1)
        os.remove(p)
        for fig in f:
            for row in fig["image"]:
                row[0] = round(row[0] + x0, 2)
            figs.append(fig)
    return figs, rgb, (w, h)


def dark_mask(rgb):
    """Blade pixels: dark and grey (not the blue embroidery, not skin, hair, boots)."""
    f = rgb.astype(np.float32)
    lum = 0.299 * f[:, :, 0] + 0.587 * f[:, :, 1] + 0.114 * f[:, :, 2]
    sat = f.max(axis=2) - f.min(axis=2)
    mask = ((lum < 120) & (sat < 40)).astype(np.uint8)
    # Line art outlines are dark too but thin; the blade is a filled
    # shape. An opening a little wider than an outline keeps only filled
    # shapes (blade, guard, pommel).
    import cv2

    k = max(3, int(round(rgb.shape[1] * 0.004)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) > 0


def components(mask):
    import cv2

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return n, labels, stats


def read_blade(rgb, fig, labels, stats, height_px):
    """The longest thin dark component whose near end is by the wrists."""
    pts = fig["image"]
    wrists = np.array([pts[15][:2], pts[16][:2], pts[19][:2], pts[20][:2]])
    hand = wrists.mean(axis=0)
    best = None
    for c in range(1, len(stats)):
        if stats[c][4] < 80:
            continue
        ys, xs = np.nonzero(labels == c)
        p = np.stack([xs, ys], axis=1).astype(np.float64)
        mean = p.mean(axis=0)
        cov = np.cov((p - mean).T)
        evals, evecs = np.linalg.eigh(cov)
        axis = evecs[:, 1]
        proj = (p - mean) @ axis
        length = proj.max() - proj.min()
        width = 4.0 * np.sqrt(max(evals[0], 1e-9))
        if length < 0.15 * height_px or width > 0.25 * length:
            continue
        a = mean + axis * proj.min()
        b = mean + axis * proj.max()
        da = np.linalg.norm(a - hand)
        db = np.linalg.norm(b - hand)
        near, far = (a, b) if da < db else (b, a)
        if min(da, db) > 0.22 * height_px:
            continue
        score = length
        if best is None or score > best[0]:
            best = (score, near, far, width)
    if best is None:
        return None
    _, near, far, width = best
    return {"hilt": [round(float(near[0]), 1), round(float(near[1]), 1)],
            "tip": [round(float(far[0]), 1), round(float(far[1]), 1)],
            "width_px": round(float(width), 1)}


def read_sheet(model, path, n_poses):
    figs, rgb, (w, h) = detect(model, path, n_poses)
    how = "multi"
    if len(figs) != n_poses:
        figs, rgb, (w, h) = split_detect(model, path, n_poses)
        how = "panels"
    if len(figs) != n_poses:
        raise SystemExit(f"{path}: read {len(figs)} bodies, expected {n_poses}")
    figs.sort(key=lambda f: (f["image"][23][0] + f["image"][24][0]))
    mask = dark_mask(rgb)
    _, labels, stats = components(mask)
    for f in figs:
        ys = [p[1] for p in f["image"]]
        height_px = max(ys) - min(ys)
        f["blade"] = read_blade(rgb, f, labels, stats, height_px)
    floor = ground_line(rgb)
    return {"file": os.path.basename(path), "size": [w, h], "read": how, "floor_y": floor, "figures": figs}


def ground_line(rgb):
    """The drawn ground line: the lowest image row that is dark across much of the width."""
    f = rgb.astype(np.float32).mean(axis=2)
    h, w = f.shape
    dark_share = (f < 140).mean(axis=1)
    rows = [y for y in range(h // 2, h) if dark_share[y] > 0.5]
    return float(rows[-1]) if rows else None


def draw_debug(path, sheet, out_png):
    import cv2

    img = cv2.imread(path)
    bones = [(11, 13), (13, 15), (12, 14), (14, 16), (11, 12), (23, 24), (11, 23), (12, 24),
             (23, 25), (25, 27), (24, 26), (26, 28), (27, 29), (29, 31), (28, 30), (30, 32),
             (15, 19), (16, 20), (0, 7), (0, 8)]
    for f in sheet["figures"]:
        p = f["image"]
        for a, b in bones:
            col = (60, 60, 230) if (a % 2 == 0 and b % 2 == 0 and a > 10) else (230, 120, 40)
            cv2.line(img, (int(p[a][0]), int(p[a][1])), (int(p[b][0]), int(p[b][1])), col, 3)
        for q in p:
            cv2.circle(img, (int(q[0]), int(q[1])), 4, (20, 200, 20), -1)
        if f["blade"]:
            hl, tp = f["blade"]["hilt"], f["blade"]["tip"]
            cv2.line(img, (int(hl[0]), int(hl[1])), (int(tp[0]), int(tp[1])), (0, 220, 255), 4)
    if sheet.get("floor_y"):
        y = int(sheet["floor_y"])
        cv2.line(img, (0, y), (img.shape[1], y), (255, 0, 255), 2)
    cv2.imwrite(out_png, img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--front")
    ap.add_argument("--side")
    ap.add_argument("--poses", type=int, default=3)
    ap.add_argument("--name", default="keypose")
    ap.add_argument("--out", required=True)
    ap.add_argument("--debug", help="folder for overlay pictures")
    a = ap.parse_args()
    if not a.front and not a.side:
        sys.exit("need --front and/or --side")
    out = {"format": "motionforge-keypose-landmarks", "version": 1, "name": a.name,
           "landmarks": "mediapipe-pose-33", "views": {}}
    for view, path in (("front", a.front), ("side", a.side)):
        if not path:
            continue
        sheet = read_sheet(a.model, path, a.poses)
        out["views"][view] = sheet
        if a.debug:
            os.makedirs(a.debug, exist_ok=True)
            draw_debug(path, sheet, os.path.join(a.debug, f"{a.name}_{view}.png"))
    with open(a.out, "w") as fh:
        json.dump(out, fh, separators=(",", ":"))
        fh.write("\n")


if __name__ == "__main__":
    main()
