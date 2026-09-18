"""Types for the three program formats used in parser adaptation.

FlatProgram is the model/SFT format (stored as parsed, program or samples).
CleanedProgram has the same shape after repair; ExecutableProgram nests
category/relations and is stored as json_obj. TypedDicts do not validate inputs."""

from __future__ import annotations

from typing import Any, Optional, TypedDict

__all__ = [
    "FlatAnchor", "FlatAttribute", "FlatRelation", "FlatProgram",
    "CleanedProgram", "ExecutableNode", "ExecutableRelation",
    "ExecutableProgram", "FLAT_KEYS", "EXECUTABLE_KEY",
    "flat_program_of", "is_executable", "is_flat",
]

# Row keys that hold a FlatProgram, in the order callers should prefer them.
FLAT_KEYS = ("parsed", "program")
# The one row key that holds an ExecutableProgram.
EXECUTABLE_KEY = "json_obj"


class FlatAttribute(TypedDict, total=False):
    name: str
    negated: bool
    rank: Optional[int]


class FlatAnchor(TypedDict, total=False):
    category: str
    attributes: list[FlatAttribute]
    color: Optional[str]
    # Accepted for compatibility with nested outputs, not requested by v4/v5.
    relations: list["FlatRelation"]


class FlatRelation(TypedDict, total=False):
    predicate: str
    negated: bool
    rank: Optional[int]
    anchors: list[FlatAnchor]


class FlatProgram(TypedDict, total=False):
    """One level deep, keyed by ``predicate``/``anchors``. Never executable."""
    head: str
    attributes: list[FlatAttribute]
    color: Optional[str]
    relations: list[FlatRelation]
    viewpoint: Optional[dict[str, Any]]


#: A FlatProgram after clean_program. Same shape; repaired content.
CleanedProgram = FlatProgram


class ExecutableRelation(TypedDict, total=False):
    relation_name: str
    objects: list["ExecutableNode"]
    negative: bool
    rank: int


class ExecutableNode(TypedDict, total=False):
    category: str
    relations: list[ExecutableRelation]
    color: str
    viewpoint_anchor: str
    viewpoint_away: bool


#: The nested tree the engines execute. Lives under ``json_obj`` and nowhere else.
ExecutableProgram = ExecutableNode


def is_flat(obj) -> bool:
    """True for a FlatProgram: a dict keyed by ``head``, not by ``category``."""
    return isinstance(obj, dict) and "head" in obj and "category" not in obj


def is_executable(obj) -> bool:
    """True for an ExecutableProgram: a node keyed by ``category``."""
    return isinstance(obj, dict) and "category" in obj and "head" not in obj


def flat_program_of(row: dict) -> Optional[FlatProgram]:
    """The FlatProgram a row carries, whatever key it calls it.

    ``raw`` is deliberately not handled here -- pulling JSON out of a generation
    is :func:`grounding.parsing.parse.extract_json_objects`' job, and doing it
    here would hide a text-extraction step behind a dict lookup.
    """
    for key in FLAT_KEYS:
        obj = row.get(key)
        if isinstance(obj, dict):
            return obj
    return None
