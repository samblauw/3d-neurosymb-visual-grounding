"""Score where each object sits relative to the others.

With no viewer or look direction, keep LaSP's original scores, quirks included.
The optional frames are for diagnostics, not the main evaluation.
"""

from __future__ import annotations

import torch

from grounding._vendor.lasp.encoders.view import side_scores, front_scores, behind_scores

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

__all__ = ["Left", "Right", "Front", "Behind", "DEVICE"]


class _SideRelation:
    def __init__(self, object_locations: torch.Tensor, viewer=None, look=None) -> None:
        self.object_locations = object_locations.to(DEVICE)
        self.N = object_locations.shape[0]
        # Height doesn't change which way the viewer is facing.
        self.viewer = None if viewer is None else torch.as_tensor(
            viewer, dtype=object_locations.dtype, device=DEVICE).flatten()[:2]
        self.look = None if look is None else torch.as_tensor(
            look, dtype=object_locations.dtype, device=DEVICE).flatten()[:2]


    def _by_direction(self, sign: float) -> torch.Tensor:
        eps = 1e-6
        # Here all anchors share one look direction.
        direction = self.look / (self.look.norm() + eps)
        left = torch.stack([-direction[1], direction[0]])
        centers = self.object_locations[:, :2]
        offset = centers.unsqueeze(1) - centers.unsqueeze(0)
        projection = (offset * left).sum(-1) * sign
        scores = torch.where(projection > 0, 1.0, 0.0) / (
            1.0 + offset.norm(dim=2) + eps)
        scores.fill_diagonal_(0)
        return scores

    def forward(self) -> torch.Tensor:
        if self.look is not None:
            return self._by_direction(self._SIGN)

        centers = self.object_locations[:, :3]
        if self.viewer is not None:
            # Move the viewer to the origin without changing the input boxes.
            centers = centers.clone()
            centers[:, :2] = centers[:, :2] - self.viewer
        return side_scores(centers, self._SIGN)


class Left(_SideRelation):
    _SIGN = 1.0


class Right(_SideRelation):
    _SIGN = -1.0


class _DepthRelation:
    def __init__(self, object_locations: torch.Tensor, viewer=None, look=None) -> None:
        self.object_locations = object_locations.to(DEVICE)
        self.viewer = None if viewer is None else torch.as_tensor(
            viewer, dtype=object_locations.dtype, device=DEVICE).flatten()[:2]
        self.look = None if look is None else torch.as_tensor(
            look, dtype=object_locations.dtype, device=DEVICE).flatten()[:2]
        self._init_params()

    def _init_params(self) -> None:
        self.centers = self.object_locations[:, :3]
        self.sizes = self.object_locations[:, 3:]

    def _directional(self, front: bool) -> torch.Tensor:
        eps = 1e-6
        centers = self.centers[:, :2]
        if self.look is not None:
            direction = self.look / (self.look.norm() + eps)
            look = direction.unsqueeze(0).expand(centers.shape[0], 2)
        else:
            # With a viewer position, look towards each anchor separately.
            look = centers - self.viewer
            look = look / (look.norm(dim=1, keepdim=True) + eps)
        offset = centers.unsqueeze(1) - centers.unsqueeze(0)
        distance = offset.norm(dim=2) + eps
        cosine = (offset * look.unsqueeze(0)).sum(dim=2) / distance
        # "In front" means back towards the viewer along their line of sight.
        signed = -cosine if front else cosine
        # Larger horizontal sizes count more at the same distance.
        size = self.sizes[:, :2].norm(dim=1)
        influence = (size.unsqueeze(1) + size.unsqueeze(0)) / distance
        scores = torch.relu(signed) * influence
        scores.fill_diagonal_(0)
        return scores


class Front(_DepthRelation):
    def forward(self) -> torch.Tensor:
        if self.viewer is not None or self.look is not None:
            return self._directional(front=True)

        return front_scores(self.centers, self.sizes)



class Behind(_DepthRelation):
    def forward(self) -> torch.Tensor:
        if self.viewer is not None or self.look is not None:
            return self._directional(front=False)

        return behind_scores(self.centers, self.sizes)
