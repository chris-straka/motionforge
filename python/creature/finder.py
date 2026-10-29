"""The joint finder's model and image pipeline (torch).

A DINOv2 ViT-S/14 backbone (facebook/dinov2-small, Apache-2.0) fine-tuned
per creature to light up one heatmap per joint. Input: one camera view
resized to 448x448 (32x32 patches). Head: a small conv decoder from the
patch tokens to J heatmaps at 112x112 (4x the patch grid). Each joint's
position is the soft-argmax of its heatmap; its confidence is the
heatmap's peak after a softmax over the map, so a joint the model cannot
see still gets a guess (what we want for far legs) but a low score.

Shared by train.py (fit) and detect.py (run on a video).
"""

import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

BACKBONE = "facebook/dinov2-small"
BACKBONE_LICENSE = "Apache-2.0"
INPUT = 448
PATCH = 14
GRID = INPUT // PATCH  # 32
HEAT = GRID * 4  # 128 (heatmap side)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class JointFinder(nn.Module):
    def __init__(self, n_joints, backbone=BACKBONE, pretrained=True):
        super().__init__()
        from transformers import AutoConfig, AutoModel

        if pretrained:
            self.backbone = AutoModel.from_pretrained(backbone)
        else:
            self.backbone = AutoModel.from_config(AutoConfig.from_pretrained(backbone))
        dim = self.backbone.config.hidden_size
        self.n_joints = n_joints
        self.decoder = nn.Sequential(
            nn.Conv2d(dim, 256, 3, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 64
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 128
            nn.GELU(),
            nn.Conv2d(64, n_joints, 1),
        )

    def forward(self, x):
        """x: (B, 3, INPUT, INPUT) in [0, 1]. Returns heatmap logits (B, J, HEAT, HEAT)."""
        x = (x - MEAN.to(x.device, x.dtype)) / STD.to(x.device, x.dtype)
        out = self.backbone(pixel_values=x).last_hidden_state  # (B, 1+G*G[+regs], D)
        tokens = out[:, -GRID * GRID:, :]
        b, n, d = tokens.shape
        feat = tokens.transpose(1, 2).reshape(b, d, GRID, GRID)
        return self.decoder(feat)


def decode(logits):
    """Heatmap logits -> (u, v) in input pixels and a confidence per joint.

    Position: soft-argmax of the softmax over the map (sub-pixel).
    Confidence: the softmax peak scaled so a sharp one-cell peak is 1.
    """
    b, j, h, w = logits.shape
    flat = logits.reshape(b, j, h * w)
    prob = torch.softmax(flat, dim=-1)
    peak = prob.max(dim=-1).values
    ys = torch.arange(h, device=logits.device, dtype=prob.dtype)
    xs = torch.arange(w, device=logits.device, dtype=prob.dtype)
    grid_y = ys.view(1, 1, h, 1).expand(b, j, h, w).reshape(b, j, -1)
    grid_x = xs.view(1, 1, 1, w).expand(b, j, h, w).reshape(b, j, -1)
    u = (prob * grid_x).sum(-1)
    v = (prob * grid_y).sum(-1)
    scale = INPUT / w
    conf = (peak * 20.0).clamp(0.0, 1.0)  # a tight peak holds ~5% of the mass
    return (u + 0.5) * scale, (v + 0.5) * scale, conf


def gaussian_targets(uv, sigma_px=6.0):
    """uv: (B, J, 2) in input pixels -> (B, J, HEAT, HEAT) targets summing ~1 per map."""
    b, j, _ = uv.shape
    scale = HEAT / INPUT
    cx = (uv[..., 0] * scale - 0.5).view(b, j, 1, 1)
    cy = (uv[..., 1] * scale - 0.5).view(b, j, 1, 1)
    ys = torch.arange(HEAT, device=uv.device, dtype=uv.dtype).view(1, 1, HEAT, 1)
    xs = torch.arange(HEAT, device=uv.device, dtype=uv.dtype).view(1, 1, 1, HEAT)
    s = sigma_px * scale
    g = torch.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * s * s))
    return g / g.sum(dim=(-1, -2), keepdim=True).clamp_min(1e-9)


def heatmap_loss(logits, uv, visible):
    """Cross-entropy between the softmax heatmap and the Gaussian target
    (visible joints), plus nothing for joints behind the camera."""
    b, j, h, w = logits.shape
    target = gaussian_targets(uv).reshape(b, j, -1)
    logp = torch.log_softmax(logits.reshape(b, j, -1), dim=-1)
    ce = -(target * logp).sum(-1)  # (B, J)
    mask = visible.to(ce.dtype)
    return (ce * mask).sum() / mask.sum().clamp_min(1.0)


# --- degradation: what video models and codecs do to a picture ------------

def degrade(img, rng, strength=1.0):
    """img: (3, H, W) float in [0, 1] on any device. Light, colour, blur,
    noise, blockiness, a little scale/shift. Returns the image and the
    affine (scale, dx, dy) applied so targets can follow."""
    c, h, w = img.shape
    s = strength
    # Exposure / contrast / gamma / colour cast.
    img = img * rng.uniform(1 - 0.35 * s, 1 + 0.35 * s)
    mean = img.mean()
    img = (img - mean) * rng.uniform(1 - 0.3 * s, 1 + 0.3 * s) + mean
    img = img.clamp(1e-4, 1) ** rng.uniform(1 - 0.3 * s, 1 + 0.3 * s)
    cast = torch.tensor([rng.uniform(1 - 0.12 * s, 1 + 0.12 * s) for _ in range(3)], device=img.device, dtype=img.dtype)
    img = img * cast.view(3, 1, 1)
    # Saturation.
    gray = img.mean(0, keepdim=True)
    img = gray + (img - gray) * rng.uniform(0.3, 1.3)
    # Blur (gaussian) and sometimes directional blur.
    if rng.random() < 0.7 * s:
        k = rng.choice([3, 5, 7])
        sig = rng.uniform(0.5, 2.0)
        xs = torch.arange(k, device=img.device, dtype=img.dtype) - (k - 1) / 2
        g = torch.exp(-xs ** 2 / (2 * sig * sig))
        g = g / g.sum()
        img = F.conv2d(img.unsqueeze(0), g.view(1, 1, 1, k).expand(3, 1, 1, k), padding=(0, k // 2), groups=3)
        img = F.conv2d(img, g.view(1, 1, k, 1).expand(3, 1, k, 1), padding=(k // 2, 0), groups=3).squeeze(0)
    # Down-up (codec softness) and block noise.
    if rng.random() < 0.5 * s:
        f = rng.uniform(0.4, 0.9)
        small = F.interpolate(img.unsqueeze(0), scale_factor=f, mode="bilinear", align_corners=False, recompute_scale_factor=False)
        img = F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False).squeeze(0)
    if rng.random() < 0.6 * s:
        img = img + torch.randn_like(img) * rng.uniform(0.0, 0.05 * s)
    if rng.random() < 0.3 * s:
        bs = rng.choice([8, 16])
        blocks = torch.rand(3, max(1, h // bs), max(1, w // bs), device=img.device, dtype=img.dtype)
        blocks = F.interpolate(blocks.unsqueeze(0), size=(h, w), mode="nearest").squeeze(0)
        img = img + (blocks - 0.5) * rng.uniform(0.0, 0.08 * s)
    img = img.clamp(0, 1)
    # Scale / shift (the video model drifts the framing a little).
    scale = rng.uniform(1 - 0.08 * s, 1 + 0.08 * s)
    dx = rng.uniform(-0.04, 0.04) * s * w
    dy = rng.uniform(-0.04, 0.04) * s * h
    theta = torch.tensor([[1 / scale, 0, -2 * dx / w / scale], [0, 1 / scale, -2 * dy / h / scale]], device=img.device, dtype=img.dtype)
    grid = F.affine_grid(theta.unsqueeze(0), (1, 3, h, w), align_corners=False)
    img = F.grid_sample(img.unsqueeze(0), grid, mode="bilinear", padding_mode="border", align_corners=False).squeeze(0)
    return img, (scale, dx, dy)


def apply_affine(uv, affine, w, h):
    """Move targets with the image: p' = (p - c) * scale + c + (dx, dy)."""
    scale, dx, dy = affine
    cx, cy = w / 2, h / 2
    out = uv.clone()
    out[..., 0] = (uv[..., 0] - cx) * scale + cx + dx
    out[..., 1] = (uv[..., 1] - cy) * scale + cy + dy
    return out


def load_image(path, device="cpu"):
    """PNG/JPEG -> (3, H, W) float [0, 1]."""
    from PIL import Image
    import numpy as np

    im = Image.open(path).convert("RGB")
    arr = torch.from_numpy(np.asarray(im).copy()).to(device)
    return arr.permute(2, 0, 1).float() / 255.0


def to_input(img):
    """(3, H, W) -> (3, INPUT, INPUT) by resize (views are square)."""
    return F.interpolate(img.unsqueeze(0), size=(INPUT, INPUT), mode="bilinear", align_corners=False).squeeze(0)


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
