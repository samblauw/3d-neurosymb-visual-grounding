"""The scene API the resolver needs, over the geometry the pipeline already has.

Everything here is derived from ONE source -- the benchmark boxes
(:mod:`grounding.data.benchmark`) plus the wall/floor rectangles
:func:`grounding.data.scenes.wall_geometry` reads out of the same file. No point
cloud, no second box convention, and no new wall detector: the supporting-wall
question is answered by ``AgainstTheWall`` itself, in :mod:`support`.

The context is a value object built once per scan and reused across rows, since
a probe run touches the same handful of scans many times over.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

__all__ = ["AnchorHit", "SceneFrameContext"]

# Tokens excluded from longest-trailing-phrase matching against scene labels
_STOP = {"the", "a", "an", "you", "your", "it", "and", "of", "in", "on", "at",
         "is", "that", "this", "from", "to", "toward", "towards", "some",
         "two", "three", "four", "five", "both", "all"}


def _tokens(phrase: str) -> list[str]:
    import re
    return [t for t in re.split(r"\W+", (phrase or "").lower())
            if t and t not in _STOP]


def _singular(word: str) -> str:
    """Enough English plurals to reach a ScanNet label, and no more.

    ``-ves -> -f`` is here because it is the difference between "the shelves"
    and the scan's ``shelf``/``bookshelf``, which is not a rare phrasing in
    these utterances.
    """
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("ves") and len(word) > 4:
        return word[:-3] + "f"
    if word.endswith("ses") or word.endswith("hes") or word.endswith("xes"):
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


@dataclass(frozen=True)
class AnchorHit:
    """A phrase resolved against the scene."""

    label: str
    indices: tuple           # instance rows in the context's arrays
    point: np.ndarray        # (2,) the xy point the phrase denotes
    match_mode: str          # exact | singular | head_noun

    @property
    def n(self) -> int:
        return len(self.indices)


class SceneFrameContext:
    """One scan's geometry, in the shape the reference-frame methods ask for.

    Attributes:
        labels: per-instance category name, background included.
        boxes: (N, 6) centre xyz then size xyz -- the same layout every encoder
            in this package takes, so ``AgainstTheWall`` can be constructed from
            it directly.
        wall_boxes: (W, 4) annotated wall rectangles [min_x, min_y, max_x, max_y].
        floor_box: (4,) the floor instance's rectangle, or None.
    """

    def __init__(self, labels: Sequence[str], boxes: np.ndarray,
                 wall_boxes=None, floor_box=None,
                 object_ids: Optional[Sequence[int]] = None,
                 scan_id: Optional[str] = None) -> None:
        self.scan_id = scan_id
        self.labels = [str(x) for x in labels]
        self.boxes = np.asarray(boxes, dtype=float).reshape(len(self.labels), 6)
        self.wall_boxes = None if wall_boxes is None else np.asarray(wall_boxes, dtype=float)
        self.floor_box = None if floor_box is None else np.asarray(floor_box, dtype=float)
        self.object_ids = (list(range(len(self.labels))) if object_ids is None
                           else [int(i) for i in object_ids])
        self._label_set = set(self.labels)

    # --- construction ------------------------------------------------------

    @classmethod
    def from_scan(cls, scan_id: str, boxes_dir=None) -> "SceneFrameContext":
        """Load from the benchmark boxes, background instances included.

        Background is kept because the wall and floor rectangles ARE the wall
        geometry, and because "door" and "window" are foreground instances whose
        supporting wall is a background one.
        """
        from grounding.data.benchmark import load_boxes
        from grounding.data.scenes import wall_geometry
        from grounding.paths import BOXES_DIR

        objects = load_boxes(scan_id, boxes_dir or BOXES_DIR, include_background=True)
        # load_boxes preserves instance order, so row i IS instance i, as wall_geometry indexes
        labels = [o["label"] for o in objects]
        geom = wall_geometry(objects, labels)
        boxes = np.array([list(o["centre"]) + list(o["size"]) for o in objects],
                         dtype=float)
        return cls(labels, boxes, wall_boxes=geom["wall_boxes"],
                   floor_box=geom["floor_box"],
                   object_ids=[o["object_id"] for o in objects], scan_id=scan_id)

    # --- geometry ----------------------------------------------------------

    @property
    def centres(self) -> np.ndarray:
        return self.boxes[:, :2]

    @property
    def room_centre(self) -> Optional[np.ndarray]:
        """Floor-rectangle centre, falling back to the mean wall-centreline endpoint.

        Used for an explicitly stated room-centre position or to sign a wall
        normal. It is not an assumed observer location."""
        if self.floor_box is not None and self.floor_box.size == 4:
            f = self.floor_box
            return np.array([(f[0] + f[2]) / 2, (f[1] + f[3]) / 2])
        hull = self.wall_centreline_endpoints()
        if hull is not None and len(hull):
            return hull.mean(axis=0)
        return None

    def wall_centreline_endpoints(self) -> Optional[np.ndarray]:
        """Two points per annotated wall: the ends of its centreline.

        The same construction ``AgainstTheWall`` uses for its default
        ``wallbox_wallhull`` source, called as that module's own function so
        there is one derivation, not a copy.
        """
        from grounding.representation.encoders.against_wall import \
            wall_centreline_endpoints
        return wall_centreline_endpoints(self.wall_boxes)

    # --- language -> scene -------------------------------------------------

    def match_label(self, phrase: Optional[str]) -> tuple[Optional[str], str]:
        """Longest trailing phrase naming a class present in the scan.

        Four modes, tried in order and reported so a caller can weigh them:
        exact ("sink"), singular ("shelves" -> "shelf"), compound ("white board"
        -> the scan's "whiteboard"), head noun ("bookshelves" -> the scan's one
        "bookshelf"). Nothing beyond that: a looser match invents an anchor, and
        inventing is what this resolver is for not doing.
        """
        toks = _tokens(phrase or "")
        if not toks:
            return None, "none"

        for n in (3, 2, 1):
            for i in range(len(toks) - n + 1):
                cand = " ".join(toks[i:i + n])
                if cand in self._label_set:
                    return cand, "exact"
                sing = " ".join(_singular(t) for t in toks[i:i + n])
                if sing in self._label_set:
                    return sing, "singular"

        # ScanNet closes some compounds speakers write open, so compare with spaces removed
        squashed = {l.replace(" ", ""): l for l in self._label_set}
        for n in (3, 2):
            for i in range(len(toks) - n + 1):
                for cand in ("".join(toks[i:i + n]),
                             "".join(_singular(t) for t in toks[i:i + n])):
                    if cand in squashed:
                        return squashed[cand], "compound"

        head = _singular(toks[-1])
        hits = {l for l in self._label_set if _singular(l.split()[-1]) == head}
        if len(hits) == 1:
            return hits.pop(), "head_noun"

        from experiments.reference_frame.lexicon import \
            LABEL_SYNONYMS
        for tok in reversed(toks):
            for alt in LABEL_SYNONYMS.get(tok, ()):
                if alt in self._label_set:
                    return alt, "synonym"
        return None, "none"

    def anchor(self, phrase: Optional[str]) -> Optional[AnchorHit]:
        """Resolve a phrase to a point in the scene, or None.

        A class with several instances gives the MEAN of their centres, which is
        what a plural anchor ("the cabinets", "the windows") denotes. For a
        singular phrase matching several instances the mean is a compromise and
        the caller can see ``n`` and decide.
        """
        label, mode = self.match_label(phrase)
        if label is None:
            return None
        idx = tuple(i for i, l in enumerate(self.labels) if l == label)
        if not idx:
            return None
        point = self.centres[list(idx)].mean(axis=0)
        return AnchorHit(label=label, indices=idx, point=point, match_mode=mode)
