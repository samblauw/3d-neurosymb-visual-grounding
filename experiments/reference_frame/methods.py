"""Five independently switchable frame-resolution methods.

Each maps (parsed_frame, scene) to ReferenceFrame without falling back to another
method. Only point_to_point supplies a stated observer origin; surface methods
supply directions. Missing or ambiguous evidence produces a coded abstention."""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

import numpy as np

from experiments.reference_frame import lexicon
from experiments.reference_frame.frame import (MIN_BASELINE,
                                                            FrameSource,
                                                            ParsedFrame,
                                                            ReferenceFrame,
                                                            resolved,
                                                            unresolved)
from experiments.reference_frame.support import best_supported_instance

__all__ = ["point_to_point", "surface_normal", "doorway_inward",
           "away_from_surface", "conditional_support", "METHODS", "METHOD_NAMES"]


_SUPPORT_REASONS = {
    "no_wall_annotation": "the scan annotates no wall instance",
    "not_against_wall": "the instance is not against any annotated wall",
    "insufficient_lateral_overlap":
        "the instance touches a wall but does not run along it",
    "ambiguous_corner":
        "the instance is equally against two differently-oriented walls",
    "interior_side_unknown":
        "which side of the wall the room lies on could not be determined",
    "no_supported_instance":
        "no instance of the class has a confirmed supporting wall",
}


# ---------------------------------------------------------------------------
# 1. direct point to point -- at_toward, from_toward
# ---------------------------------------------------------------------------

def _point_for(scene, phrase: Optional[str]):
    """``(point, how, detail)`` for an observer-position or look-target phrase."""
    if not phrase:
        return None, "missing", {}
    if lexicon.is_room_centre(phrase):
        centre = scene.room_centre
        if centre is None:
            return None, "no_room_geometry", {}
        return centre, "room_centre", {}
    if lexicon.is_vague_room_region(phrase):
        return None, "vague_region", {}
    if lexicon.normalise(phrase) in {"them", "it", "those", "you", "your", "there"}:
        return None, "unresolved_pronoun", {}
    if lexicon.spatial_part(phrase):
        # "front of the windows" is a position relative to an anchor whose orientation we lack
        return None, "spatial_part", {"part": lexicon.spatial_part(phrase)}
    hit = scene.anchor(phrase)
    if hit is None:
        return None, "label_not_in_scene", {}
    return hit.point, "centroid", {"label": hit.label, "n": hit.n,
                                   "match": hit.match_mode}


def point_to_point(frame: ParsedFrame, scene) -> ReferenceFrame:
    """Both endpoints stated: ``forward = normalize(toward - source)``.

    The one method that needs no assumption about the scene beyond where things
    are. "From the couch, facing the cabinets" and "in the middle of the room
    facing the sink" are the same computation; the source is whichever of
    ``at_anchor`` / ``from_anchor`` the parser filled.
    """
    name = "point_to_point"
    source_phrase = frame.at_anchor or frame.from_anchor
    if not source_phrase or not frame.toward_anchor:
        return unresolved(name, "not_a_two_point_frame",
                          "The frame does not state both a source position and "
                          "a look target.")

    src, how_src, d_src = _point_for(scene, source_phrase)
    tgt, how_tgt, d_tgt = _point_for(scene, frame.toward_anchor)
    if src is None:
        return unresolved(name, f"source_{how_src}",
                          f"The stated source '{source_phrase}' is not a "
                          f"resolvable scene point ({how_src}).", **d_src)
    if tgt is None:
        return unresolved(name, f"target_{how_tgt}",
                          f"The stated look target '{frame.toward_anchor}' is "
                          f"not a resolvable scene point ({how_tgt}).", **d_tgt)

    look = np.asarray(tgt, dtype=float) - np.asarray(src, dtype=float)
    baseline = float(np.linalg.norm(look))
    if baseline < MIN_BASELINE:
        return unresolved(name, "degenerate_baseline",
                          f"Source and target are {baseline:.2f} m apart, under "
                          f"the {MIN_BASELINE} m needed for a direction.",
                          baseline=round(baseline, 3))

    # Confidence drops on a scattered endpoint: the mean of five chairs is a place no speaker stood
    spread = max(d_src.get("n", 1), d_tgt.get("n", 1))
    confidence = 0.9 if spread == 1 else max(0.5, 0.9 - 0.1 * (spread - 1))
    return resolved(
        name, look,
        f"Source '{source_phrase}' ({how_src}) and target "
        f"'{frame.toward_anchor}' ({how_tgt}) are both scene points, "
        f"{baseline:.2f} m apart.",
        FrameSource.DIRECT_POINTS, confidence, origin=src,
        source_how=how_src, target_how=how_tgt, baseline=round(baseline, 3),
        source_detail=d_src, target_detail=d_tgt)


# ---------------------------------------------------------------------------
# 2. surface / wall normal -- toward_only, strong surface anchor
# ---------------------------------------------------------------------------

def _wall_frame(scene, phrase, name, *, facing: bool, source: FrameSource,
                base_confidence: float, require_class=None) -> ReferenceFrame:
    """Shared geometry for the three wall-normal methods.

    ``facing=True`` means the observer looks AT the surface, so forward is the
    wall's outward normal; ``facing=False`` means they came through it or have
    their back to it, so forward is the inward normal. Every other difference
    between the three is which frame slot is read and what is required of it,
    which is why they stay separate functions.
    """
    if require_class is not None and lexicon.anchor_class(phrase) != require_class:
        return unresolved(name, "wrong_anchor_class",
                          f"'{phrase}' is not a {require_class} surface anchor.")
    if lexicon.intrinsic_part(phrase):
        return unresolved(name, "needs_intrinsic_orientation",
                          f"'{phrase}' names an object's own front/back, which "
                          "the representation does not encode.")
    hit = scene.anchor(phrase)
    if hit is None:
        return unresolved(name, "label_not_in_scene",
                          f"No instance in the scan matches '{phrase}'.")

    idx, wall, agree, code = best_supported_instance(scene, hit.indices)
    if wall is None:
        return unresolved(name, f"support_{code}",
                          f"'{phrase}' resolved to {hit.n} '{hit.label}' "
                          f"instance(s), but {_SUPPORT_REASONS.get(code, code)}.",
                          label=hit.label, n=hit.n)
    if agree < hit.n:
        # Windows on two different walls: the class does not name one direction
        return unresolved(name, "support_directions_disagree",
                          f"{hit.n} '{hit.label}' instances are supported by "
                          f"walls facing different ways ({agree} agree).",
                          label=hit.label, n=hit.n, agree=agree)

    forward = wall.outward if facing else wall.inward
    # The support score is the encoder's own confidence, so it carries straight through
    confidence = base_confidence * wall.score
    direction = "away from the room" if facing else "into the room"
    return resolved(
        name, forward,
        f"'{hit.label}' is confirmed against wall {wall.wall_index} "
        f"(against-wall score {wall.score:.2f}, gap {wall.gap:.2f} m, lateral "
        f"{wall.lateral:.2f}); forward is that wall's normal {direction}.",
        source, confidence,
        label=hit.label, n=hit.n, instance=int(idx),
        wall_index=wall.wall_index, support_score=round(wall.score, 3),
        gap=round(wall.gap, 3), lateral=round(wall.lateral, 3),
        rival_ratio=round(wall.rival_ratio, 3))


def surface_normal(frame: ParsedFrame, scene) -> ReferenceFrame:
    """Facing a window / whiteboard / door: forward is the wall's outward normal.

    "Facing the windows" does not say where the speaker stood, and this method
    does not guess: it returns a direction only. That is enough for left/right,
    which needs an axis, not a position.

    The category being a strong surface term is a precondition, not the
    evidence. The evidence is that THIS instance is confirmed against a wall --
    a "door" instance that turns out to be a cupboard door standing in open
    space fails the same check a bookshelf would.
    """
    return _wall_frame(scene, frame.toward_anchor, "surface_normal",
                       facing=True, source=FrameSource.SURFACE_NORMAL,
                       base_confidence=0.85, require_class="strong")


# ---------------------------------------------------------------------------
# 3. doorway entry -- from_only on a door / doorway / entrance
# ---------------------------------------------------------------------------

def doorway_inward(frame: ParsedFrame, scene) -> ReferenceFrame:
    """"As you walk in the door": forward is the doorway wall's INWARD normal.

    Kept apart from the ordinary facing-a-door case on purpose. The geometry is
    the same wall and the opposite sign, and the two are distinguished only by
    which frame slot the door landed in: a door in ``toward_anchor`` is being
    looked at from inside the room, a door in ``from_anchor`` is being walked
    through. Fusing them would make that sign a heuristic.
    """
    name = "doorway_inward"
    phrase = frame.from_anchor
    if not phrase:
        return unresolved(name, "no_from_anchor",
                          "The frame states no source the observer entered from.")
    if not any(t in lexicon.normalise(phrase).split()
               for t in lexicon.DOORWAY_TERMS):
        return unresolved(name, "source_is_not_a_doorway",
                          f"Source '{phrase}' is not a door, doorway or entrance.")
    return _wall_frame(scene, phrase, name, facing=False,
                       source=FrameSource.SURFACE_NORMAL, base_confidence=0.8)


# ---------------------------------------------------------------------------
# 4. away from a surface -- "with your back to the door"
# ---------------------------------------------------------------------------

def away_from_surface(frame: ParsedFrame, scene) -> ReferenceFrame:
    """Back to a surface: forward is the direction AWAY from it.

    For a wall-mounted anchor that is the wall's inward normal -- the same
    vector :func:`doorway_inward` computes, from different language and a
    different argument, which is why it is a separate method with its own count.

    When the frame also names a look target, the target is NOT used to place an
    observer (there is none) but IS checked for consistency: if the target lies
    behind the observer, the two cues contradict and the method abstains rather
    than picking the one it likes.
    """
    name = "away_from_surface"
    phrase = frame.away_from_anchor
    if not phrase:
        return unresolved(name, "no_away_anchor",
                          "The frame states nothing the observer's back is to.")

    result = _wall_frame(scene, phrase, name, facing=False,
                         source=FrameSource.SURFACE_NORMAL, base_confidence=0.8)
    if not result.resolved or not frame.toward_anchor:
        return result

    target, how, detail = _point_for(scene, frame.toward_anchor)
    if target is None:
        return replace(result,
                       detail={**result.detail, "toward_check": f"unresolved:{how}"})
    hit = scene.anchor(phrase)
    surface_point = hit.point if hit is not None else None
    if surface_point is None:
        return result
    to_target = np.asarray(target, float) - np.asarray(surface_point, float)
    agree = float(to_target @ result.forward)
    if agree <= 0:
        return unresolved(name, "toward_contradicts_away",
                          f"The stated look target '{frame.toward_anchor}' lies "
                          f"behind the direction away from '{phrase}'.",
                          projection=round(agree, 3))
    return replace(result, detail={**result.detail, "toward_check": "consistent",
                                   "toward_projection": round(agree, 3)})


# ---------------------------------------------------------------------------
# 5. conditional wall-supported object -- sink, shelves, counter, toilet
# ---------------------------------------------------------------------------

def conditional_support(frame: ParsedFrame, scene) -> ReferenceFrame:
    """Facing a sink / shelves / counter / toilet, IF this instance is supported.

    Identical geometry to :func:`surface_normal` and a deliberately different
    method, because the licence is different. There the category implies the
    support and the scene confirms it; here the category implies nothing -- a
    sink can sit in an island, shelves can stand free in the middle of a lab --
    and the scene is the only evidence there is. An instance that fails the
    check returns unresolved, never a category-based guess.
    """
    return _wall_frame(scene, frame.toward_anchor, "conditional_support",
                       facing=True, source=FrameSource.CONFIRMED_SUPPORT,
                       base_confidence=0.6, require_class="conditional")


METHODS = {
    "point_to_point": point_to_point,
    "surface_normal": surface_normal,
    "doorway_inward": doorway_inward,
    "away_from_surface": away_from_surface,
    "conditional_support": conditional_support,
}

METHOD_NAMES = tuple(METHODS)
