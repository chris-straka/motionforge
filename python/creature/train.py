"""Fine-tune the joint finder for one creature on its rendered poses.

    python python/creature/train.py --renders DIR --out models/<creature> \
        [--epochs 12] [--batch 16] [--lr 1e-4] [--seed 1] [--val 0.1]

DIR is what render.py --mode poses wrote: f{i}_{view}.png and truth.json.
Every view image is one sample (front and side share the model). Images
are degraded on the fly (finder.degrade: light, colour, blur, noise,
blocks, drift) and the 2D targets follow. Writes out/finder.pt (weights +
joint names + input size), out/run.json (settings, timings, validation
error in pixels of the 512 view), and prints the validation numbers.

Sized for an 8 GB card: ViT-S/14 at 448, batch 16, fp16 autocast.
"""

import argparse
import json
import os
import random
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import finder  # noqa: E402


def load_truth(renders):
    with open(os.path.join(renders, "truth.json")) as f:
        t = json.load(f)
    items = []
    for s in t["samples"]:
        for v in t["views"]:
            path = os.path.join(renders, f"f{s['index']:05d}_{v}.png")
            if os.path.exists(path):
                items.append((path, s["views"][v]))
    return t["joints"], t["views"], items


class Data:
    """Images cached in RAM as uint8 at INPUT size (3000 x 2 x 448² x 3 = 3.6 GB)."""

    def __init__(self, items, device):
        self.items = items
        self.device = device
        self.cache = {}

    def get(self, i):
        if i not in self.cache:
            from PIL import Image
            import numpy as np

            path, _ = self.items[i]
            im = Image.open(path).convert("RGB").resize((finder.INPUT, finder.INPUT), Image.BILINEAR)
            self.cache[i] = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).contiguous()  # uint8
        return self.cache[i]

    def targets(self, i, view_size):
        _, joints = self.items[i]
        uv = torch.tensor([[j[0], j[1]] for j in joints], dtype=torch.float32) * (finder.INPUT / view_size)
        vis = torch.tensor([j[2] > 0 for j in joints])
        return uv, vis


def batch_iter(data, idx, batch, rng, degrade, view_size, device):
    for k in range(0, len(idx), batch):
        ids = idx[k:k + batch]
        imgs, uvs, viss = [], [], []
        for i in ids:
            img = data.get(i).to(device).float() / 255.0
            uv, vis = data.targets(i, view_size)
            uv = uv.to(device)
            if degrade:
                img, aff = finder.degrade(img, rng)
                uv = finder.apply_affine(uv, aff, finder.INPUT, finder.INPUT)
            imgs.append(img)
            uvs.append(uv)
            viss.append(vis.to(device))
        yield torch.stack(imgs), torch.stack(uvs), torch.stack(viss)


def evaluate(model, data, idx, batch, view_size, device, degrade=False, rng=None):
    model.eval()
    errs = []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
        for imgs, uvs, viss in batch_iter(data, idx, batch, rng or random.Random(0), degrade, view_size, device):
            u, v, conf = finder.decode(model(imgs).float())
            d = torch.sqrt((u - uvs[..., 0]) ** 2 + (v - uvs[..., 1]) ** 2) * (view_size / finder.INPUT)
            errs.append(d[viss].cpu())
    e = torch.cat(errs)
    return {"mean_px": float(e.mean()), "median_px": float(e.median()),
            "pck_10px": float((e < 10).float().mean()), "n": int(e.numel())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--renders", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--backbone-lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--limit", type=int, default=0, help="use at most N images (smoke tests)")
    ap.add_argument("--view-size", type=int, default=512)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    torch.set_num_threads(4)
    finder.seed_all(args.seed)
    rng = random.Random(args.seed)
    joints, views, items = load_truth(args.renders)
    if args.limit:
        items = items[: args.limit]
    if not items:
        raise SystemExit("no rendered images found")
    data = Data(items, args.device)
    idx = list(range(len(items)))
    rng.shuffle(idx)
    n_val = max(1, int(len(idx) * args.val))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    model = finder.JointFinder(len(joints)).to(args.device)
    params = [
        {"params": model.backbone.parameters(), "lr": args.backbone_lr},
        {"params": model.decoder.parameters(), "lr": args.lr},
    ]
    opt = torch.optim.AdamW(params, weight_decay=0.01)
    steps = args.epochs * ((len(train_idx) + args.batch - 1) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.backbone_lr, args.lr], total_steps=max(1, steps), pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=args.device.startswith("cuda"))
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    log = []
    step = 0
    for epoch in range(args.epochs):
        model.train()
        rng.shuffle(train_idx)
        total, count = 0.0, 0
        for imgs, uvs, viss in batch_iter(data, train_idx, args.batch, rng, True, args.view_size, args.device):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.device.startswith("cuda")):
                logits = model(imgs)
                loss = finder.heatmap_loss(logits.float(), uvs, viss)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            if step < steps:
                sched.step()
            step += 1
            total += float(loss) * imgs.shape[0]
            count += imgs.shape[0]
        val = evaluate(model, data, val_idx, args.batch, args.view_size, args.device)
        val_deg = evaluate(model, data, val_idx, args.batch, args.view_size, args.device, degrade=True, rng=random.Random(99))
        entry = {"epoch": epoch + 1, "train_loss": total / max(1, count), "val": val, "val_degraded": val_deg,
                 "seconds": round(time.time() - t0, 1)}
        log.append(entry)
        print(f"epoch {epoch + 1}/{args.epochs} loss {entry['train_loss']:.4f} "
              f"val {val['mean_px']:.2f} px (median {val['median_px']:.2f}, pck10 {val['pck_10px']:.3f}) "
              f"degraded {val_deg['mean_px']:.2f} px  [{entry['seconds']}s]", flush=True)
    peak = torch.cuda.max_memory_allocated() // 2 ** 20 if args.device.startswith("cuda") else 0
    torch.save({"state_dict": model.state_dict(), "joints": joints, "views": views, "input": finder.INPUT,
                "backbone": finder.BACKBONE, "backbone_license": finder.BACKBONE_LICENSE,
                "view_size": args.view_size}, os.path.join(args.out, "finder.pt"))
    run = {"renders": os.path.abspath(args.renders), "images": len(items), "train": len(train_idx), "val": len(val_idx),
           "joints": len(joints), "epochs": args.epochs, "batch": args.batch, "lr": args.lr, "backbone_lr": args.backbone_lr,
           "seed": args.seed, "backbone": finder.BACKBONE, "backbone_license": finder.BACKBONE_LICENSE,
           "device": args.device, "gpu": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else "",
           "peak_vram_mb": peak, "seconds": round(time.time() - t0, 1), "log": log}
    with open(os.path.join(args.out, "run.json"), "w") as f:
        json.dump(run, f, indent=2)
    print(f"saved {os.path.join(args.out, 'finder.pt')} ({run['seconds']}s, peak {peak} MB)")


if __name__ == "__main__":
    main()
