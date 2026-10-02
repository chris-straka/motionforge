#!/usr/bin/env python3
# MIT (see LICENSE-MIT). AutoPose training (needs torch except --dry-run).
"""Train the AutoPose MLP on retargeted clip JSON listed by a manifest.

Pipeline: manifest gate (tools/manifest.py) -> load clips -> shared
skeleton check -> per-epoch random effector subsets (seeded) -> MLP ->
cosine quat loss + foot penetration penalty -> weights JSON + run log.

    python3 python/autopose/train.py --manifest data/manifest.json \\
        --out-dir models/run1 --epochs 50 --seed 1

--dry-run validates the manifest + data and prints shapes without torch.
Gates (see docs/autopose.md): held-out mean joint error under --gate-mm
(default 50 mm) AND zero foot penetrations; exit 0 on pass, 1 otherwise
(weights are still written so a miss is debuggable).

Determinism: --seed drives Python + torch RNG; CPU training sets
torch.use_deterministic_algorithms(True). MPS/CUDA are allowed via
--device with a nondeterminism warning.
"""

import argparse
import hashlib
import json
import math
import os
import random
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "python"))
sys.path.insert(0, os.path.join(REPO, "tools"))

import manifest as manifest_gate  # noqa: E402
from autopose import dataset, inference  # noqa: E402


def manifest_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def load_data(manifest_path):
    """(skeleton, train, heldout, manifest_sha). Frames are dataset poses."""
    with open(manifest_path, encoding="utf-8") as f:
        doc = json.load(f)
    base = os.path.dirname(os.path.abspath(manifest_path))
    skeletons, train, heldout = [], [], []
    for clip in doc.get("clips", []):
        skel, frames = dataset.load_clip(os.path.join(base, clip["path"]))
        skeletons.append(skel)
        if clip.get("split", "train") == "heldout":
            heldout.extend(frames)
        else:
            train.extend(frames)
    skeleton = dataset.check_shared_skeleton(skeletons)
    return skeleton, train, heldout, manifest_sha(manifest_path)


def build_parser():
    parser = argparse.ArgumentParser(description="train the AutoPose MLP")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--feet", default="DEF-foot.L,DEF-foot.R",
                        help="comma-separated foot bones for the penetration term")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden", default="256,256",
                        help="comma-separated hidden widths")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--foot-weight", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gate-mm", type=float, default=50.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-mixamo-training", action="store_true")
    return parser


def main(argv):
    args = build_parser().parse_args(argv)

    ok, lines = manifest_gate.check_manifest(args.manifest, args.allow_mixamo_training)
    for line in lines:
        print(line)
    if not ok:
        print("MANIFEST REJECTED: refusing to train")
        return 1

    skeleton, train, heldout, msha = load_data(args.manifest)
    n = len(skeleton.names)
    try:
        feet = [skeleton.names.index(name.strip())
                for name in args.feet.split(",") if name.strip()]
    except ValueError as exc:
        print(f"unknown foot bone in --feet: {exc}")
        return 1
    print(f"skeleton: {n} bones, train frames: {len(train)}, heldout frames: {len(heldout)}")
    print(f"manifest sha256: {msha}")
    if not train:
        print("no train frames")
        return 1
    if args.dry_run:
        hidden = tuple(int(h) for h in args.hidden.split(","))
        params = sum(
            (a + 1) * b for a, b in zip([5 * n, *hidden], [*hidden, 4 * n])
        )
        print(f"model: MLP 5n={5 * n} -> {hidden} -> 4n={4 * n} ({params} params)")
        print("DRY RUN OK")
        return 0

    try:
        import torch
    except ImportError:
        print("torch is not installed: pip install -r python/requirements.txt")
        return 1
    from autopose import model as model_mod

    device = torch.device(args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cpu":
        torch.use_deterministic_algorithms(True)
    else:
        print(f"WARNING: nondeterministic device {args.device}; use --device cpu to reproduce")

    hidden = tuple(int(h) for h in args.hidden.split(","))
    net = model_mod.AutoPoseMLP(n, hidden).to(device).double()
    kin = model_mod.RestKinematics(skeleton, device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    # Precompute heads/targets/locs once (effector subsets resample per epoch).
    with torch.no_grad():
        train_heads = torch.tensor(
            [dataset.fk(skeleton, p) for p in train], dtype=torch.float64)
        train_quats = torch.tensor(
            [[list(q) for (_l, q) in p] for p in train], dtype=torch.float64)
        train_locs = torch.tensor(
            [[list(l) for (l, _q) in p] for p in train], dtype=torch.float64)
        ground = train_heads[:, feet, 2].min().item() if feet else 0.0

    rng = random.Random(args.seed)
    batches = max(1, (len(train) + args.batch_size - 1) // args.batch_size)
    for epoch in range(args.epochs):
        order = list(range(len(train)))
        rng.shuffle(order)
        total = 0.0
        for b in range(batches):
            idx = order[b * args.batch_size : (b + 1) * args.batch_size]
            ins = []
            for i in idx:
                heads = [tuple(v) for v in train_heads[i].tolist()]
                ins.append(dataset.build_input(
                    skeleton, heads, dataset.sample_effectors(rng, n)))
            xb = torch.tensor(ins, dtype=torch.float64, device=device)
            pred = net(xb)
            loss = model_mod.angular_loss(pred, train_quats[idx].to(device))
            if feet and args.foot_weight > 0:
                pred_heads = kin.fk(train_locs[idx].to(device), pred)
                pen = torch.relu(ground - pred_heads[:, feet, 2]).mean()
                loss = loss + args.foot_weight * pen
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        print(f"epoch {epoch + 1}/{args.epochs} loss {total / len(train):.6f}", flush=True)

    # Held-out eval with fixed (seeded) effector sets.
    metrics = None
    if heldout:
        with torch.no_grad():
            eval_rng = random.Random(args.seed + 999)
            sets = [dataset.sample_effectors(eval_rng, n) for _ in heldout]
            held_heads = [dataset.fk(skeleton, p) for p in heldout]
            ins = [dataset.build_input(skeleton, h, s)
                   for h, s in zip(held_heads, sets)]
            pred = net(torch.tensor(ins, dtype=torch.float64, device=device))
            pred_q = pred.cpu().tolist()
            true_q = [[list(q) for (_l, q) in p] for p in heldout]
            pos_errs, ang_errs, pen_count = [], [], 0
            for p, t, true_p in zip(pred_q, true_q, heldout):
                ph = dataset.fk(skeleton, [(l, tuple(q)) for (l, _), q in zip(true_p, p)])
                th = dataset.fk(skeleton, true_p)
                for a, c in zip(ph, th):
                    pos_errs.append(math.dist(a, c))
                for qp, qt in zip(p, t):
                    ang_errs.append(inference.quat_angle(tuple(qp), tuple(qt)))
                for f in feet:
                    if ph[f][2] < ground - 1e-9:
                        pen_count += 1
            metrics = {
                "mean_joint_mm": 1000.0 * sum(pos_errs) / len(pos_errs),
                "max_joint_mm": 1000.0 * max(pos_errs),
                "mean_angle_deg": sum(ang_errs) / len(ang_errs) * 180.0 / math.pi,
                "foot_penetrations": pen_count,
            }
            print("heldout: mean {:.2f} mm, max {:.2f} mm, angle {:.2f} deg, "
                  "penetrations {}".format(
                      metrics["mean_joint_mm"], metrics["max_joint_mm"],
                      metrics["mean_angle_deg"], pen_count))

    os.makedirs(args.out_dir, exist_ok=True)
    doc = inference.export_weights(skeleton.names, model_mod.to_plain_lists(net))
    inference.write_weights(os.path.join(args.out_dir, "weights.json"), doc)
    gate_pass = (
        metrics is not None
        and metrics["mean_joint_mm"] <= args.gate_mm
        and metrics["foot_penetrations"] == 0
    )
    run = {
        "manifest": os.path.abspath(args.manifest),
        "manifest_sha256": msha,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "hidden": list(hidden),
        "lr": args.lr,
        "foot_weight": args.foot_weight,
        "device": str(device),
        "train_frames": len(train),
        "heldout_frames": len(heldout),
        "torch": torch.__version__,
        "metrics": metrics,
        "gate_mm": args.gate_mm,
        "gate": "PASS" if gate_pass else "FAIL",
    }
    with open(os.path.join(args.out_dir, "run.json"), "w", encoding="utf-8") as f:
        json.dump(run, f, indent=1)
    print(f"wrote {args.out_dir}/weights.json + run.json")
    print("GATE " + ("PASS" if gate_pass else "FAIL"))
    return 0 if gate_pass else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
