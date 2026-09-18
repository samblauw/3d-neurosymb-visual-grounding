"""The room outline used by ``AgainstTheWall``.

The candidate list has `wall`/`floor`/`ceiling` removed, so wall geometry comes
from the annotated wall boxes: the hull of their centreline endpoints bounds the
room, and needs no point cloud.
"""

import numpy as np

__all__ = ["hull_halfplanes"]


def hull_halfplanes(pts_xy: np.ndarray):
    """(A, b) with ``A @ x + b <= 0`` inside the convex hull of ``pts_xy``.

    Taken over the endpoints of the annotated wall centrelines, which needs no
    point cloud at all.
    """
    pts = np.unique(np.asarray(pts_xy, dtype=np.float64), axis=0)
    if len(pts) < 3:
        return np.zeros((0, 2)), np.zeros(0)

    order = np.lexsort((pts[:, 1], pts[:, 0]))
    p = pts[order]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def chain(seq):
        out = []
        for q in seq:
            while len(out) >= 2 and cross(out[-2], out[-1], q) <= 0:
                out.pop()
            out.append(q)
        return out[:-1]

    hull = np.array(chain(p) + chain(p[::-1]))
    if len(hull) < 3:
        return np.zeros((0, 2)), np.zeros(0)

    edge = np.roll(hull, -1, axis=0) - hull
    length = np.linalg.norm(edge, axis=1)
    keep = length > 1e-9
    hull, edge, length = hull[keep], edge[keep], length[keep]
    # Counter-clockwise hull, so the outward normal is the edge rotated by -90.
    normals = np.stack([edge[:, 1], -edge[:, 0]], axis=1) / length[:, None]
    offsets = -(normals * hull).sum(axis=1)
    return normals, offsets
