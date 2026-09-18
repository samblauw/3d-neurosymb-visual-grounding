"""Score how strongly each object is against annotated room-wall geometry."""

from __future__ import annotations

import numpy as np
import torch

from grounding.representation.encoders.wall_geometry import hull_halfplanes

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

__all__ = ["AgainstTheWall", "DEVICE", "wall_centreline_endpoints"]


def wall_centreline_endpoints(wall_boxes):
    """Ends of each wall box's centreline, also used by the reference frame."""
    if wall_boxes is None or not len(wall_boxes):
        return None
    w = np.asarray(wall_boxes, dtype=np.float64)
    lo, hi = w[:, :2], w[:, 2:]
    ax = np.argmin(hi - lo, axis=1)
    other = 1 - ax
    rows = np.arange(len(w))
    mid = (lo[rows, ax] + hi[rows, ax]) / 2
    a, b = np.empty((len(w), 2)), np.empty((len(w), 2))
    a[rows, ax] = b[rows, ax] = mid
    a[rows, other] = lo[rows, other]
    b[rows, other] = hi[rows, other]
    return np.concatenate([a, b], axis=0)


class AgainstTheWall:
    """Score against annotated walls, which are not themselves candidates.

    The wall boxes supply the surfaces, and the hull of their centreline
    endpoints the room outline; neither needs a point cloud.
    """

    #: Decay length of the gap term, in metres. A box touching a wall scores
    #: 1.0, one standing ~25 cm off it scores ~0.37.
    GAP_SCALE = 0.25
    #: Smallest extent used to normalise the overlap term, so thin objects
    #: (a picture, a whiteboard) are not judged on a near-zero denominator.
    MIN_EXTENT = 0.2

    def __init__(
        self,
        object_locations: torch.Tensor,
        scene_point_cloud: torch.Tensor = None,
        wall_boxes: torch.Tensor = None,
    ) -> None:
        """
        Args:
            object_locations: (N, 6), centre xyz then size xyz per object.
            wall_boxes: (W, 4) annotated wall boxes as [min_x, min_y, max_x, max_y].
            scene_point_cloud: (M, 3) points, or a 2x3 [min; max] extent. Read
                only by the room-boundary fallback, for a scan with no wall.
        """
        self._set_inputs(object_locations, scene_point_cloud, wall_boxes)
        self._init_params()

    def _set_inputs(self, object_locations, scene_point_cloud=None,
                    wall_boxes=None) -> None:
        self.object_locations = object_locations.to(DEVICE)
        self.scene_point_cloud = (None if scene_point_cloud is None
                                  else scene_point_cloud.to(DEVICE))
        self.wall_boxes = wall_boxes

    @classmethod
    def over_surfaces(cls, object_locations, wall_boxes):
        """Score against the wall rectangles alone, without the room outline."""
        encoder = cls.__new__(cls)
        encoder._set_inputs(object_locations, wall_boxes=wall_boxes)
        encoder.surfaces = encoder._surfaces()
        encoder.hull_A = encoder.hull_b = None
        return encoder

    def _surfaces(self):
        """Wall rectangles [min_x, min_y, max_x, max_y], or None."""
        if self.wall_boxes is not None and len(self.wall_boxes):
            return torch.as_tensor(np.asarray(self.wall_boxes),
                                   dtype=torch.float32, device=DEVICE)
        return None

    def _wall_centreline_endpoints(self):
        return wall_centreline_endpoints(self.wall_boxes)

    def _init_params(self) -> None:
        self.surfaces = self._surfaces()
        self.hull_A = self.hull_b = None
        ends = self._wall_centreline_endpoints()
        if ends is not None and len(ends) >= 3:
            A, b = hull_halfplanes(ends)
            to_t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=DEVICE)
            self.hull_A, self.hull_b = to_t(A), to_t(b)

    def _corners(self):
        centres = self.object_locations[:, :3]
        sizes = self.object_locations[:, 3:].abs()
        mins, maxs = centres - sizes / 2, centres + sizes / 2
        corners = torch.stack([
            torch.stack([mins[:, 0], mins[:, 1]], dim=-1),
            torch.stack([mins[:, 0], maxs[:, 1]], dim=-1),
            torch.stack([maxs[:, 0], mins[:, 1]], dim=-1),
            torch.stack([maxs[:, 0], maxs[:, 1]], dim=-1),
        ], dim=1)
        return mins, maxs, corners

    def forward(self) -> torch.Tensor:
        """Return (N,) in [0, 1]; 1 means flush against a wall."""
        parts = []
        if self.surfaces is not None:
            parts.append(self._surface_score())
        if self.hull_A is not None and len(self.hull_A):
            parts.append(self._hull_score())
        if not parts:
            return self._room_boundary_fallback()
        score = parts[0]
        for extra in parts[1:]:
            score = torch.maximum(score, extra)
        return score.clamp(0.0, 1.0).reshape(len(self.object_locations))

    def per_surface_terms(self, surfaces=None):
        """Return (scores, gaps, laterals), each (N, W), before choosing a wall."""
        surfaces = self.surfaces if surfaces is None else surfaces
        mins, maxs, _ = self._corners()
        n = mins.shape[0]
        if surfaces is None or not len(surfaces):
            empty = torch.zeros((n, 0), device=DEVICE)
            return empty, empty.clone(), empty.clone()

        gaps, laterals = [], []
        for w in surfaces:
            lo, hi = w[:2], w[2:]
            ax = int(torch.argmin(hi - lo))
            other = 1 - ax
            a_lo, a_hi = mins[:, ax], maxs[:, ax]
            # The gap to the nearest face of the wall box.
            gap = torch.minimum(
                torch.minimum((a_lo - lo[ax]).abs(), (a_lo - hi[ax]).abs()),
                torch.minimum((a_hi - lo[ax]).abs(), (a_hi - hi[ax]).abs()))
            gaps.append(gap)
            laterals.append(self._overlap_ratio(mins[:, other], maxs[:, other],
                                                lo[other], hi[other]))
        gaps = torch.stack(gaps, dim=1)
        laterals = torch.stack(laterals, dim=1)
        return torch.exp(-gaps / self.GAP_SCALE) * laterals, gaps, laterals

    def _surface_score(self, surfaces=None) -> torch.Tensor:
        """Best gap-and-overlap score over the given walls."""
        scores, _, _ = self.per_surface_terms(surfaces)
        if not scores.shape[1]:
            return torch.zeros(scores.shape[0], device=DEVICE)
        return scores.max(dim=1).values

    def _hull_score(self) -> torch.Tensor:
        # Interior distance to the nearest hull edge. No lateral term: every edge
        # of a room outline bounds the whole room, so it would be 1 throughout.
        _, _, corners = self._corners()
        d = -(corners @ self.hull_A.t() + self.hull_b)
        gap = d.clamp(min=0).min(dim=1).values.min(dim=1).values
        return torch.exp(-gap / self.GAP_SCALE)

    def _overlap_ratio(self, a_min, a_max, b_min, b_max):
        """Fraction of the object's interval covered by the wall."""
        overlap = torch.minimum(a_max, b_max) - torch.maximum(a_min, b_min)
        extent = (a_max - a_min).clamp(min=self.MIN_EXTENT)
        return (overlap / extent).clamp(0.0, 1.0)

    def _room_boundary_fallback(self) -> torch.Tensor:
        """Distance to the extent of whatever geometry we have."""
        mins, maxs, _ = self._corners()
        if self.scene_point_cloud is not None:
            lo = self.scene_point_cloud.min(dim=0).values[:2]
            hi = self.scene_point_cloud.max(dim=0).values[:2]
        else:
            lo, hi = mins[:, :2].min(dim=0).values, maxs[:, :2].max(dim=0).values
        gap = torch.stack([
            mins[:, 0] - lo[0], hi[0] - maxs[:, 0],
            mins[:, 1] - lo[1], hi[1] - maxs[:, 1],
        ], dim=1).clamp(min=0).min(dim=1).values
        return torch.exp(-gap / self.GAP_SCALE)
