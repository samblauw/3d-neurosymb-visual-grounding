"""Shared executable-program traversal for the resolver and thesis diagnostics.

Retained children/relations/nodes traversal accepts the serialized anchor alias
without mutating programs. Feature-specific lexical selection lives in the
representation probe commands.
"""

from __future__ import annotations

ISOLATION = {"isolated", "isolation", "alone", "by itself", "lone"}
LATERAL = ("left", "right")

def relation_objects(relation):
    """Return a relation's executable children without changing its input.

    ``objects`` is the executable schema. ``anchors`` is accepted only for old
    serialized executable programs that predate the schema boundary; when both
    exist, the canonical ``objects`` field wins.
    """
    if not isinstance(relation, dict):
        return ()
    children = relation.get("objects")
    if children is None:
        children = relation.get("anchors", ())
    return children if isinstance(children, list) else ()


def nodes(obj):
    if not isinstance(obj, dict):
        return
    yield obj
    for r in obj.get("relations") or []:
        for o in relation_objects(r):
            yield from nodes(o)


def relations(obj):
    if not isinstance(obj, dict):
        return
    for r in obj.get("relations") or []:
        yield r
        for o in relation_objects(r):
            yield from relations(o)
