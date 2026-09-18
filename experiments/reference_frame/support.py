"""Recover supporting-wall identity and inward orientation for frame diagnostics.

Read AgainstTheWall.per_surface_terms on annotated wall rectangles. Sign the
wall's thin-axis normal using the wall-endpoint centroid, with floor centre as
fallback. The named thresholds gate support, corner ambiguity and sign evidence;
they are hand-set abstention criteria, not learned parameters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = ["SupportingWall", "supporting_wall", "MIN_SUPPORT_SCORE",
           "MIN_LATERAL", "MAX_RIVAL_RATIO", "MIN_INTERIOR_OFFSET"]


# Support exp(-gap / 0.25) * overlap: 0.5 permits about 17 cm at full overlap
MIN_SUPPORT_SCORE = 0.5

# Require half the object extent along the wall to reject corner-only contact
MIN_LATERAL = 0.5

# Abstain when a differently oriented rival wall has >80% of the best support
MAX_RIVAL_RATIO = 0.8

#: Metres the room interior must lie off the wall plane for the inward sign to mean anything
MIN_INTERIOR_OFFSET = 0.1

#: Two wall normals within this angle count as one orientation, so split rectangles agree
SAME_NORMAL_DEG = 20.0


@dataclass(frozen=True)
class SupportingWall:
    """The wall an instance is confirmed against, oriented."""

    wall_index: int
    inward: np.ndarray      # (2,) unit, pointing into the room
    score: float            # AgainstTheWall's own term for this object/wall
    gap: float              # metres to the nearest face of the wall box
    lateral: float          # share of the object's extent running along it
    rival_ratio: float      # best differently-oriented wall, as a share of score
    interior_offset: float  # metres from the wall plane to the room interior

    @property
    def outward(self) -> np.ndarray:
        """Away from the room -- the direction an observer LOOKING at something
        on this wall is facing."""
        return -self.inward


def _wall_normal_axes(wall_boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per wall: the thin axis (0 or 1) and the plane's coordinate on it."""
    w = np.asarray(wall_boxes, dtype=float)
    lo, hi = w[:, :2], w[:, 2:]
    ax = np.argmin(hi - lo, axis=1)
    rows = np.arange(len(w))
    mid = (lo[rows, ax] + hi[rows, ax]) / 2
    return ax, mid


def _interior_point(scene) -> Optional[np.ndarray]:
    """The point that says which side of a wall the room is on."""
    ends = scene.wall_centreline_endpoints()
    if ends is not None and len(ends) >= 3:
        return ends.mean(axis=0)
    return scene.room_centre


def supporting_wall(
    scene,
    obj_index: int,
    *,
    min_score: float = MIN_SUPPORT_SCORE,
    min_lateral: float = MIN_LATERAL,
    max_rival_ratio: float = MAX_RIVAL_RATIO,
    min_interior_offset: float = MIN_INTERIOR_OFFSET,
) -> tuple[Optional[SupportingWall], str]:
    """``(wall, code)`` -- the confirmed supporting wall, or None and why not.

    ``code`` is one of ``ok``, ``no_wall_annotation``, ``not_against_wall``,
    ``insufficient_lateral_overlap``, ``ambiguous_corner``,
    ``interior_side_unknown``. The caller turns it into a reason; keeping it
    separate is what lets the diagnostic count them.
    """
    import torch

    from grounding.representation.encoders.against_wall import AgainstTheWall

    if scene.wall_boxes is None or not len(scene.wall_boxes):
        return None, "no_wall_annotation"

    boxes = torch.as_tensor(scene.boxes, dtype=torch.float32)
    encoder = AgainstTheWall.over_surfaces(boxes, scene.wall_boxes)
    scores, gaps, laterals = encoder.per_surface_terms()
    row = scores[obj_index].detach().cpu().numpy()
    gap_row = gaps[obj_index].detach().cpu().numpy()
    lat_row = laterals[obj_index].detach().cpu().numpy()

    # Lateral first, then the combined score. Both must pass either way, but the
    # encoder's score already has the overlap multiplied into it, so testing the
    # score first would report "not against a wall" for an object flush against
    # a wall fragment shorter than itself -- true only in the encoder's sense,
    # and not the reason a reader needs
    best = int(np.argmax(row))
    if float(lat_row[best]) < min_lateral:
        return None, "insufficient_lateral_overlap"
    if float(row[best]) < min_score:
        return None, "not_against_wall"

    ax, mid = _wall_normal_axes(scene.wall_boxes)
    normal = np.zeros(2)
    normal[int(ax[best])] = 1.0

    # A rival is a wall this object is nearly as against, whose normal points elsewhere
    rival = 0.0
    for j in range(len(row)):
        if j == best or float(row[j]) <= 0:
            continue
        other = np.zeros(2)
        other[int(ax[j])] = 1.0
        cos = abs(float(other @ normal))
        if cos > np.cos(np.deg2rad(SAME_NORMAL_DEG)):
            continue
        rival = max(rival, float(row[j]))
    rival_ratio = rival / float(row[best])
    if rival_ratio > max_rival_ratio:
        return None, "ambiguous_corner"

    interior = _interior_point(scene)
    if interior is None:
        return None, "interior_side_unknown"
    offset = float(interior[int(ax[best])] - mid[best])
    if abs(offset) < min_interior_offset:
        return None, "interior_side_unknown"

    return SupportingWall(
        wall_index=best,
        inward=normal * np.sign(offset),
        score=float(row[best]),
        gap=float(gap_row[best]),
        lateral=float(lat_row[best]),
        rival_ratio=rival_ratio,
        interior_offset=abs(offset),
    ), "ok"


def best_supported_instance(scene, indices, **kwargs):
    """The instance of a class with the strongest confirmed support.

    A plural anchor ("the windows") names several instances that may sit on
    different walls. Rather than averaging normals -- two opposite walls average
    to zero -- this picks the best-supported instance and reports how many of
    the others agree with it, which the caller can gate on.
    """
    hits = []
    for i in indices:
        wall, code = supporting_wall(scene, i, **kwargs)
        if wall is not None:
            hits.append((i, wall))
    if not hits:
        # Report the last code seen, or the only one if the class is singular
        codes = [supporting_wall(scene, i, **kwargs)[1] for i in indices]
        return None, None, 0, (codes[0] if len(set(codes)) == 1 else "no_supported_instance")
    idx, wall = max(hits, key=lambda t: t[1].score)
    agree = sum(1 for _, w in hits
                if float(w.inward @ wall.inward) > np.cos(np.deg2rad(SAME_NORMAL_DEG)))
    return idx, wall, agree, "ok"
