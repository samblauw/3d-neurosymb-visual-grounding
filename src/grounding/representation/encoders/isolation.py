"""Score isolation from the nearest same-category object's bounding box."""

from __future__ import annotations

import numpy as np
import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

__all__ = ["Isolated", "DEVICE"]


class Isolated:
    """Without category labels, measure distance to the nearest object of any kind."""

    DISTANCE_SCALE = 1.0  # Hand-set decay length in metres.

    def __init__(
        self,
        object_locations: torch.Tensor,
        scene_point_cloud: torch.Tensor = None,
        object_labels: list = None,
    ) -> None:
        """Locations are (N, 6): centre xyz, size xyz. Point cloud is unused."""
        self.object_locations = object_locations.to(DEVICE)
        self.object_labels = object_labels
        self._init_params()

    def _init_params(self) -> None:
        """Pairwise gaps between boxes: 0 where two boxes overlap."""
        centers = self.object_locations[:, :3]
        sizes = self.object_locations[:, 3:].abs()
        mins = centers - sizes / 2
        maxs = centers + sizes / 2
        separation = torch.maximum(
            mins.unsqueeze(1) - maxs.unsqueeze(0),
            mins.unsqueeze(0) - maxs.unsqueeze(1),
        ).clamp(min=0)
        self.distances = separation.norm(dim=-1)
        self.distances.fill_diagonal_(float('inf'))

    def forward(self) -> torch.Tensor:
        """Return (N,) in [0, 1]; 1 means far from everything of its kind."""
        num_objects = self.object_locations.shape[0]
        if num_objects < 2:
            return torch.zeros(num_objects, device=DEVICE)

        distances = self.distances
        if self.object_labels is not None and len(self.object_labels) == num_objects:
            labels = np.asarray(self.object_labels)
            same = torch.from_numpy(labels[:, None] == labels[None, :]).to(DEVICE)
            same.fill_diagonal_(False)
            distances = distances.masked_fill(~same, float('inf'))

        # No peer gives an infinite distance and therefore a score of 1.
        nearest = distances.min(dim=1).values
        score = 1 - torch.exp(-nearest / self.DISTANCE_SCALE)
        return torch.where(torch.isfinite(score), score, torch.ones_like(score))
