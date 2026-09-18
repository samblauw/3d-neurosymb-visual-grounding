"""LaSP default-view formulas, retaining the existing numerical edge cases."""

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def side_scores(centers, sign):
    offset = centers.unsqueeze(1) - centers.unsqueeze(0)
    origin = centers.unsqueeze(1)
    # The cross-product sign tells us which side of the anchor we're on.
    cross_z = origin[..., 0] * offset[..., 1] - origin[..., 1] * offset[..., 0]
    scores = torch.where(cross_z * sign > 0, 1.0, 0.0)
    # Keep the full 3D distance on this path, as in the baseline.
    scores *= 1 / (1 + torch.norm(offset, dim=-1) + 1e-6)
    scores.fill_diagonal_(0)
    return scores.to(DEVICE)


def front_scores(centers, sizes):
    # The original formula doesn't distinguish front from behind.
    offset = centers.unsqueeze(1) - centers.unsqueeze(0)
    # No epsilon here: coincident centres keep the baseline's NaNs.
    direction = offset / offset.norm(dim=2, keepdim=True)
    dx, dy = offset[:, :, 0], offset[:, :, 1]
    distance = torch.sqrt(dx**2 + dy**2)
    alignment = (direction[:, :, :2] * torch.stack((dx, dy), dim=2)
                 / distance.unsqueeze(2)).sum(dim=2)
    size = sizes[:, :2].norm(dim=1)
    influence = (size.unsqueeze(1) + size.unsqueeze(0)) / (distance + 1e-6)
    scores = torch.relu(alignment) * influence
    scores.fill_diagonal_(0)
    return scores


def behind_scores(centers, sizes):
    # LaSP puts epsilon in different places here than in Front; keep that.
    offset = centers.unsqueeze(0) - centers.unsqueeze(1)
    direction = offset / (offset.norm(dim=2, keepdim=True) + 1e-6)
    distance = offset[..., :2].norm(dim=2) + 1e-6
    alignment = (direction[..., :2] * offset[..., :2]).sum(dim=2) / distance
    size = sizes[:, :2].norm(dim=1)
    influence = (size.unsqueeze(1) + size.unsqueeze(0)) / distance
    scores = torch.relu(alignment) * influence
    scores.fill_diagonal_(0)
    return scores
