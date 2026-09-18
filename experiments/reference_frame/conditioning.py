"""Per-row reference-frame intervention on the left/right scene features."""

from __future__ import annotations

from typing import Mapping, Sequence

__all__ = ["LATERAL_RELATIONS", "frame_conditioned_concepts"]

# Only left/right are re-framed: front/behind have no axis, and `look` changes their formula
LATERAL_RELATIONS = ("left", "right")


def frame_conditioned_concepts(
    concepts: Mapping,
    object_locations,
    frame,
    view_relations: Mapping,
    relation_names: Sequence[str] = LATERAL_RELATIONS,
):
    """Rebuild existing left/right tensors using ``frame.forward`` and N×6 boxes.

    Scene features are shared across rows, so a resolved frame uses a shallow
    copy. An unresolved frame, or no applicable relation, returns the original
    dictionary by identity. Unmentioned relations are never added.
    """
    if frame is None or not frame.resolved or frame.forward is None:
        return concepts
    names = [n for n in relation_names
             if n in concepts and n in view_relations]
    if not names:
        return concepts
    out = dict(concepts)
    for name in names:
        encoder = view_relations[name]
        out[name] = encoder(object_locations, look=frame.forward).forward().float()
    return out
