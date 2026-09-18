"""Classify frame anchors as strong surfaces, conditional supports or generic objects.

Strong and conditional anchors still require geometric confirmation. A generic
object category alone does not determine which side the observer faces."""

from __future__ import annotations

import re
from typing import Optional

__all__ = ["STRONG_SURFACE", "CONDITIONAL_SURFACE", "GENERIC_UNORIENTED",
           "DOORWAY_TERMS", "anchor_class", "normalise", "is_room_centre",
           "is_vague_room_region", "spatial_part", "intrinsic_part"]


STRONG_SURFACE = {
    "wall", "window", "windows", "whiteboard", "white board", "blackboard",
    "board", "boards", "door", "doors", "doorway", "entrance", "picture",
    "pictures", "painting", "paintings", "frame", "frames", "mirror",
    "screen on the wall", "curtain", "curtains", "blinds",
}

CONDITIONAL_SURFACE = {
    "sink", "sinks", "kitchen sink", "counter", "counters", "kitchen counter",
    "cabinet", "cabinets", "kitchen cabinet", "kitchen cabinets", "bookshelf",
    "bookshelves", "shelf", "shelves", "bookcase", "fridge", "refrigerator",
    "stove", "oven", "toilet", "toilets", "vanity", "dresser", "wardrobe",
    "radiator", "heater",
}

GENERIC_UNORIENTED = {
    "chair", "chairs", "armchair", "armchairs", "sofa", "sofas", "couch",
    "couches", "sofa chair", "sofa chairs", "stool", "stools", "cart", "carts",
    "box", "boxes", "ottoman", "ottomans", "printer", "printers", "table",
    "tables", "desk", "desks", "bed", "beds", "lamp", "lamps", "bin", "bins",
}

#: Anchors whose "from" reading is an entry, not a look target
DOORWAY_TERMS = {"door", "doors", "doorway", "doorways", "entrance", "entry"}

#: Word-level synonyms tried when a phrase names no scan label directly
#: Deliberately tiny and one-directional. ScanNet has no "doorway" or
#: "entrance" class -- the opening a speaker walks through is annotated "door"
#: or "doorframe" -- so without this the doorway method can never fire on the
#: phrasing that licenses it. Nothing else is mapped: a synonym list is a way of
#: guessing, and this one is here because the annotation vocabulary demonstrably
#: lacks the word, not because the match felt close
LABEL_SYNONYMS = {
    "doorway": ("door", "doorframe"),
    "doorways": ("door", "doorframe"),
    "entrance": ("door", "doorframe"),
    "entry": ("door", "doorframe"),
}

_ROOM_CENTRE = re.compile(r"\b(?:middle|center|centre)\s+of\s+(?:the\s+)?room\b")
_VAGUE_REGION = re.compile(
    r"\b(?:other\s+)?side\s+of\s+(?:the\s+)?room\b|\bsomewhere\s+in\s+(?:the\s+)?room\b"
    r"|\b(?:back|front|corner|end)\s+of\s+(?:the\s+)?room\b")
_SPATIAL_PART = re.compile(
    r"\b(front of|back of|side of|left of|right of|foot of|head of|top of|"
    r"other side of)\b")
_INTRINSIC_PART = re.compile(r"\b(?:front|back)\s+of\b")


def normalise(text: Optional[str]) -> str:
    """Lowercase, de-hyphenate, drop articles. Same shape as the study's."""
    if not text:
        return ""
    x = re.sub(r"[_-]+", " ", text.lower().strip())
    x = re.sub(r"\b(?:the|a|an)\b", " ", x)
    return re.sub(r"\s+", " ", x).strip()


def _contains(phrase: str, terms: set[str]) -> bool:
    a = normalise(phrase)
    if not a:
        return False
    if a in terms:
        return True
    for term in sorted(terms, key=len, reverse=True):
        t = normalise(term)
        if t and re.search(rf"(?<!\w){re.escape(t)}(?!\w)", a):
            return True
    return False


def anchor_class(phrase: Optional[str]) -> str:
    """``strong`` | ``conditional`` | ``generic`` | ``unknown``."""
    a = normalise(phrase)
    if not a:
        return "unknown"
    if re.search(r"\bon (?:the )?wall\b", a):
        return "strong"
    if _contains(a, STRONG_SURFACE):
        return "strong"
    if _contains(a, CONDITIONAL_SURFACE):
        return "conditional"
    if _contains(a, GENERIC_UNORIENTED):
        return "generic"
    return "unknown"


def is_room_centre(phrase: Optional[str]) -> bool:
    return bool(_ROOM_CENTRE.search(normalise(phrase)))


def is_vague_room_region(phrase: Optional[str]) -> bool:
    """A region of the room, not a point. Not resolvable to a centroid."""
    a = normalise(phrase)
    return bool(a and not _ROOM_CENTRE.search(a) and _VAGUE_REGION.search(a))


def spatial_part(phrase: Optional[str]) -> Optional[str]:
    """"front of the windows" -> "front of". None when the phrase is a plain noun."""
    m = _SPATIAL_PART.search(normalise(phrase))
    return m.group(1) if m else None


def intrinsic_part(phrase: Optional[str]) -> bool:
    """Does the phrase name an object's own front/back ("back of the sofa chairs")?

    Distinct from :func:`spatial_part`: this asks whether answering the phrase
    needs the ANCHOR's intrinsic orientation, which the representation does not
    carry at all.
    """
    return bool(_INTRINSIC_PART.search(normalise(phrase)))
