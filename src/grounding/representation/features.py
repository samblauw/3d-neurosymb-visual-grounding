"""Collect colour concepts recursively from executable programs.

Feature caches must include the colour keys requested by the evaluated parses."""

from __future__ import annotations

from grounding.parsing.fields import relation_objects

__all__ = ["get_all_colors"]


# Colour names anywhere in a program. Colour is an appearance modifier, not a
# relation, so it is keyed separately as "color:<name>".
def get_all_colors(json_obj):
    out = set()
    if isinstance(json_obj, dict):
        if json_obj.get("color"):
            out.add(str(json_obj["color"]).strip().lower())
        for r in json_obj.get("relations") or []:
            for o in relation_objects(r):
                out |= get_all_colors(o)
    return out
