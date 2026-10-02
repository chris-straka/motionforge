# MIT (see LICENSE-MIT). AutoPose model + losses (needs torch).
"""Small MLP: effector positions + bone lengths in, full-body quats out.

Input dim 5n ([x, y, z, mask] per bone, then n bone lengths), output 4n
(one quat per bone, normalized). Start small per PLAN.md: two hidden
layers of 256 ReLU; grow to a small transformer only if the MLP fails
its gate. Loss is cosine distance 1 - |dot| (smooth; the acos angle is
reported, not backpropped, since its gradient explodes at zero).
"""

import torch
import torch.nn as nn


class AutoPoseMLP(nn.Module):
    def __init__(self, n_bones, hidden=(256, 256)):
        super().__init__()
        self.n_bones = n_bones
        dims = [5 * n_bones, *hidden, 4 * n_bones]
        layers = []
        for a, b in zip(dims, dims[1:]):
            layers.append(nn.Linear(a, b))
            layers.append(nn.ReLU())
        layers.pop()  # last layer is linear
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        y = self.net(x)
        y = y.view(-1, self.n_bones, 4)
        return y / y.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def angular_loss(pred, target):
    """Mean 1 - |dot| over batch and bones (pred/target: (B, n, 4))."""
    return (1.0 - (pred * target).sum(dim=-1).abs()).mean()


@torch.no_grad()
def mean_angle_deg(pred, target):
    d = (pred * target).sum(dim=-1).abs().clamp(0.0, 1.0)
    return (2.0 * torch.acos(d)).mean().item() * 180.0 / torch.pi.item()


class RestKinematics:
    """Torch FK tables for one skeleton (from dataset.Skeleton)."""

    def __init__(self, skeleton, device):
        n = len(skeleton.names)
        self.parents = list(skeleton.parents)
        self.rel_rot = torch.tensor(skeleton.rel_rot, dtype=torch.float64, device=device)
        self.rel_off = torch.tensor(skeleton.rel_off, dtype=torch.float64, device=device)
        self.lengths = torch.tensor(skeleton.lengths, dtype=torch.float64, device=device)
        self.n = n

    def fk(self, locs, quats):
        """Heads (B, n, 3) from locs (B, n, 3), quats (B, n, 4)."""
        w, x, y, z = quats.unbind(-1)
        one = torch.ones_like(w)
        two = 2.0 * torch.ones_like(w)
        brot = torch.stack(
            [
                one - (y * y + z * z) * two,
                (x * y - z * w) * two,
                (x * z + y * w) * two,
                (x * y + z * w) * two,
                one - (x * x + z * z) * two,
                (y * z - x * w) * two,
                (x * z - y * w) * two,
                (y * z + x * w) * two,
                one - (x * x + y * y) * two,
            ],
            dim=-1,
        ).view(-1, self.n, 3, 3)
        rots = [None] * self.n
        heads = [None] * self.n
        for i in range(self.n):
            local_rot = self.rel_rot[i] @ brot[:, i]
            local_off = (self.rel_rot[i] @ locs[:, i].unsqueeze(-1)).squeeze(-1) + self.rel_off[i]
            p = self.parents[i]
            if p is None:
                rots[i], heads[i] = local_rot, local_off
            else:
                rots[i] = rots[p] @ local_rot
                heads[i] = (rots[p] @ local_off.unsqueeze(-1)).squeeze(-1) + heads[p]
        return torch.stack(heads, dim=1)


def to_plain_lists(model):
    """Extract [(weights_rows, bias)] plain lists for weights export."""
    layers = []
    for module in model.net:
        if isinstance(module, nn.Linear):
            layers.append((module.weight.detach().cpu().tolist(),
                           module.bias.detach().cpu().tolist()))
    return layers
