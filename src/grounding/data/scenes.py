"""Wall and floor geometry for one scan.

This module used to also assemble scans without their point cloud, from
benchmark/boxes plus a 2x3 extents cache. That is retired: the extents were
exact, but the surrogate's candidate boxes came from benchmark/boxes, which
stores each object's centroid, where load_pc centres on the midpoint of the point
extent -- up to 0.36 m apart, so the filter and the evaluator disagreed on the
target for ~7% of programs. Scans are now loaded
through the baseline tree's own load_one_scene -- see
grounding.resolution.engine.Scenes.
"""

from __future__ import annotations


import numpy as np

__all__ = ["wall_geometry"]




# ---------------------------------------------------------------------------
# Wall geometry for AgainstTheWall
# ---------------------------------------------------------------------------
# The candidate list drops wall/floor/ceiling instances (load_pc does), but the
# ANNOTATION keeps them -- so wall surfaces are read from there rather than
# recovered from geometry. The baseline instead selects from foreground boxes
# using max(dx,dy,dz)>10 m and min(dx,dy)<0.5 m. diagnose_wall_vectors.py reports
# its selected inputs separately from zero outputs; these are different claims.
#
# Extracted from the harness' load_one_scene so there is one derivation and the
# against-wall experiments can call it directly. Pure function of the scene's
# canonical boxes plus its point cloud; no torch, no dataset object.

def wall_geometry(canonical_boxes, inst_labels):
    """``{wall_boxes, floor_box}`` for one scan.

    ``canonical_boxes``   the benchmark box dicts, indexed by instance id
    ``inst_labels``       per-instance label, same indexing

    Each value is ``None`` when the scan cannot supply it, which is what the
    callers' fallbacks already expect:

    * ``wall_boxes``   the wall instances' own boxes. Each is already a segment
      -- thin axis is the normal, the other its extent -- so no fitting is
      needed at all. ``None`` when no annotated wall instance is available.
    * ``floor_box``    the first annotated floor, when present. The
      reference-frame diagnostic reads it for the room centre.
    """
    walls = [i for i, n in enumerate(inst_labels) if n == "wall"]
    floors = [i for i, n in enumerate(inst_labels) if n == "floor"]

    wall_boxes = (
        np.array([canonical_boxes[i]["bbox_min"][:2] + canonical_boxes[i]["bbox_max"][:2]
                  for i in walls], dtype=float)
        if walls else None)

    floor_box = (
        np.array(canonical_boxes[floors[0]]["bbox_min"][:2]
                 + canonical_boxes[floors[0]]["bbox_max"][:2], dtype=float)
        if floors else None)

    return {"wall_boxes": wall_boxes, "floor_box": floor_box}
