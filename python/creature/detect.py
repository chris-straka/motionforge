"""Run the joint finder on a two-view video (or rendered frames) and
write motionforge-detections JSON for `motionforge creature-solve`.

    python python/creature/detect.py --model models/<creature>/finder.pt \
        --creature creature.json --out det.json \
        (--video clip.mp4 | --frames DIR) [--fps 24] [--layout side-by-side]

A video is one composite per frame: the front view on the left, the side
view on the right (the layout the start frame was rendered in); each half
is resized to the model's input. `--frames DIR` takes render.py --mode
motion output (f{i}_{view}.png) directly, which is what the exam does.
"""

import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import finder  # noqa: E402


def read_video(path):
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = bgr[:, :, ::-1].copy()
        frames.append(torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0)
    cap.release()
    if not frames:
        raise SystemExit(f"no frames in {path}")
    return frames, fps


def split_views(frame, n_views):
    """Composite (3, H, W) -> list of (3, H, W/n) halves left to right."""
    c, h, w = frame.shape
    step = w // n_views
    return [frame[:, :, i * step:(i + 1) * step] for i in range(n_views)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--creature", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--video")
    ap.add_argument("--frames")
    ap.add_argument("--fps", type=float, default=0.0, help="override the video's fps")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if bool(args.video) == bool(args.frames):
        raise SystemExit("give --video or --frames")
    with open(args.creature) as f:
        creature = json.load(f)
    views = [v["name"] for v in creature["views"]]
    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    if ckpt["views"] != views:
        raise SystemExit(f"model views {ckpt['views']} differ from the creature's {views}")
    model = finder.JointFinder(len(ckpt["joints"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(args.device).eval()

    # Per frame: one image per view (original view size kept for the output scale).
    per_frame = []  # list of [(img, (w, h)) per view]
    fps = args.fps
    if args.video:
        frames, vfps = read_video(args.video)
        fps = fps or vfps
        for fr in frames:
            halves = split_views(fr, len(views))
            per_frame.append([(h_, (h_.shape[2], h_.shape[1])) for h_ in halves])
    else:
        i = 0
        while True:
            paths = [os.path.join(args.frames, f"f{i:05d}_{v}.png") for v in views]
            if not all(os.path.exists(p) for p in paths):
                break
            imgs = [finder.load_image(p) for p in paths]
            per_frame.append([(im, (im.shape[2], im.shape[1])) for im in imgs])
            i += 1
        truth = os.path.join(args.frames, "truth.json")
        if not fps and os.path.exists(truth):
            with open(truth) as f:
                fps = json.load(f).get("fps", 24.0)
    if not per_frame:
        raise SystemExit("no frames")
    fps = fps or 24.0

    flat = [(fi, vi, img, size) for fi, vs in enumerate(per_frame) for vi, (img, size) in enumerate(vs)]
    results = {}
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.device.startswith("cuda")):
        for k in range(0, len(flat), args.batch):
            chunk = flat[k:k + args.batch]
            x = torch.stack([finder.to_input(c[2]) for c in chunk]).to(args.device)
            u, v, conf = finder.decode(model(x).float())
            for n, (fi, vi, _, (w, h)) in enumerate(chunk):
                su, sv = w / finder.INPUT, h / finder.INPUT
                results[(fi, vi)] = [[float(u[n, j]) * su, float(v[n, j]) * sv, float(conf[n, j])] for j in range(u.shape[1])]
    out = {"format": "motionforge-detections", "version": 1, "fps": fps, "views": views, "joints": ckpt["joints"],
           "frames": [[results[(fi, vi)] for vi in range(len(views))] for fi in range(len(per_frame))]}
    with open(args.out, "w") as f:
        json.dump(out, f)
    confs = [d[2] for fr in out["frames"] for v in fr for d in v]
    print(f"detections: {len(per_frame)} frames x {len(views)} views x {len(ckpt['joints'])} joints -> {args.out} "
          f"(mean confidence {sum(confs) / len(confs):.3f}, fps {fps})")


if __name__ == "__main__":
    main()
