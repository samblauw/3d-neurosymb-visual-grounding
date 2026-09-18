"""Shared Nr3D populations and evaluation bookkeeping.

Official test totals include missing parses as misses; ``parses`` totals use
the supplied rows. Dataset indices refer to the selected split's annotation
CSV, so development and test tables must not be interchanged. Missing training
strata retain the existing False/False convention."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from grounding.data.splits import load_splits
from grounding.paths import SPLITS_DEV_PATH, SPLITS_PATH

__all__ = ["OFFICIAL_TOTALS", "OFFICIAL_DENOMINATOR", "TOTALS_MODES",
           "SPLIT_TABLES", "load_splits", "splits_for", "split_flags",
           "empty_result", "row_key", "count_result"]

#: The fixed Nr3D test-set counts. easy+hard and view_dependent+view_independent
#: each sum to overall; changing any of these changes what "accuracy" means.
OFFICIAL_TOTALS = {
    "easy": 3669,
    "hard": 3816,
    "view_dependent": 2478,
    "view_independent": 5007,
    "overall": 7485,
}
OFFICIAL_DENOMINATOR = OFFICIAL_TOTALS["overall"]
TOTALS_MODES = ("official", "parses")


#: The strata table for each split that has one, by the tree's ``--split`` name.
SPLIT_TABLES = {"test": SPLITS_PATH, "dev": SPLITS_DEV_PATH}


@lru_cache(maxsize=len(SPLIT_TABLES) + 1)
def splits_for(split: str = "test") -> dict[str, dict]:
    """The strata table for one split, or an empty one where there is none."""
    path = SPLIT_TABLES.get(split)
    if path is None or not Path(path).exists():
        return {}
    return load_splits(path)


def split_flags(dataset_idx, splits: dict[str, dict] | None = None,
                split: str = "test"):
    """``(is_hard, is_view_dependent)`` for one row index of one split.

    The index is a position in that split's own annotation csv, so the table has
    to be the one for the split being scored -- see the module docstring on why
    the wrong table answers instead of failing.

    Rows that are not in the table -- train indices, custom samples -- fall back
    to ``(False, False)`` rather than raising, which is what makes
    ``--totals parses`` usable on a probe file.
    """
    table = splits_for(split) if splits is None else splits
    entry = table.get(str(int(dataset_idx)))
    if entry is None:
        return False, False
    return entry["is_hard"], entry["is_view_dependent"]


def empty_result(totals_mode: str = "official") -> dict[str, dict]:
    """The result accumulator, with denominators set by the totals mode."""
    if totals_mode not in TOTALS_MODES:
        raise ValueError(totals_mode)
    return {
        k: {"correct": 0, "total": v if totals_mode == "official" else 0}
        for k, v in OFFICIAL_TOTALS.items()
    }


def row_key(data):
    """Annotation row ID, accepting the serialized dev and test field names."""
    for field in ("dataset_idx", "row_idx"):
        if field in data:
            return data[field]
    raise KeyError("parses row has neither dataset_idx nor row_idx")


def count_result(result, field, is_hard, is_view_dependent):
    """Increment a row's overall, difficulty and view strata together."""
    for name in ("overall", "hard" if is_hard else "easy",
                 "view_dependent" if is_view_dependent else "view_independent"):
        result[name][field] += 1
