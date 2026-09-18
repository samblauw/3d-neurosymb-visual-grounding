"""Lower flat parser output into executable category/relation trees.

Predicates are matched against the selected vocabulary; unsupported ones are dropped.
"""

from __future__ import annotations

from typing import Any

from grounding.parsing.cleaning import clean_program

__all__ = [
    "clean_program", "lower_program", "lower_one", "raw_predicates",
    "normalise_predicate",
]


_PREDICATE_ALIAS: dict[str, str] = {
    "large": "larger",
    "small": "smaller",
    "high": "higher",
    "low": "lower",
    "at_corner": "corner",
    "close to": "close",
    "closest to": "close",
    "next to": "next to",
    "on top of": "on top",
    "far from": "far",
    "far away from": "far",
    "away from": "far",
    "adjacent to": "next to",
    "by": "beside",
    "at the back of": "behind",
    "in back of": "behind",
    "in front of": "front",
    "front of": "front",
    "to the left of": "left",
    "on the left of": "left",
    "on the left side of": "left",
    "left of": "left",
    "to the right of": "right",
    "on the right of": "right",
    "on the right side of": "right",
    "right of": "right",
    "across from": "across",
    "underneath": "under",
    "over": "above",
    "within": "within",
    "connected": "attached",
    "touching": "attached",
    "at": "near",
    "near to": "near",
}


def _norm_predicate(pred: str, rel_num: dict[str, int]) -> str | None:
    """Try exact names, aliases, spaced names, then preposition suffixes."""
    p = (pred or "").strip().lower()
    if p in rel_num:
        return p
    if (alias := _PREDICATE_ALIAS.get(p)) and alias in rel_num:
        return alias

    if "_" in p:
        spaced = p.replace("_", " ")
        if spaced in rel_num:
            return spaced
        if (alias := _PREDICATE_ALIAS.get(spaced)) and alias in rel_num:
            return alias

    # Strip suffixes from the original name, not the underscore-expanded form.
    for tail in (" of", " to", " from"):
        if p.endswith(tail) and (base := p[: -len(tail)]) in rel_num:
            return base

    return None


# Accept stored nested anchors, but omit relations below this depth.
_MAX_DEPTH = 3


def _lower_node(anchor: dict, rel_num: dict[str, int], depth: int = 1) -> dict[str, Any]:
    node: dict[str, Any] = {
        "category": str(anchor.get("category", "")).strip(),
        "relations": [],
    }
    if anchor.get("color"):
        node["color"] = str(anchor["color"]).strip().lower()

    for a in anchor.get("attributes") or []:
        name = _norm_predicate(str(a.get("name", "")), rel_num)
        if not name:
            continue
        rel: dict[str, Any] = {"relation_name": name, "objects": []}
        if a.get("negated"):
            rel["negative"] = True
        if isinstance(a.get("rank"), int) and a["rank"] >= 1:
            rel["rank"] = a["rank"]
        node["relations"].append(rel)

    if depth < _MAX_DEPTH:
        _lower_relations(anchor, node, rel_num, depth)
    return node


def _lower_relations(
    spec: dict, node: dict[str, Any], rel_num: dict[str, int], depth: int,
) -> None:
    for rel in spec.get("relations") or []:
        name = _norm_predicate(rel.get("predicate", ""), rel_num)
        if name is None:
            continue
        rank = rel.get("rank")

        if name in ("middle", "center") and not (rel.get("anchors") or []):
            node["relations"].append(
                {
                    "relation_name": "in",
                    "objects": [{"category": "middle", "relations": []}],
                }
            )
            continue

        out: dict[str, Any] = {
            "relation_name": name,
            "negative": bool(rel.get("negated", False)),
            "objects": [
                _lower_node(a, rel_num, depth + 1)
                for a in rel.get("anchors") or []
            ],
        }
        if isinstance(rank, int) and rank >= 1:
            out["rank"] = rank
        node["relations"].append(out)


def lower_program(expr: dict, rel_num: dict[str, int]) -> dict[str, Any]:
    """Convert attributes and anchor relations into a vocabulary-bound tree."""
    head = _lower_node(
        {
            "category": expr.get("head", ""),
            "attributes": expr.get("attributes"),
            "color": expr.get("color"),
        },
        rel_num,
    )
    _lower_relations(expr, head, rel_num, depth=1)

    if isinstance(vp := expr.get("viewpoint"), dict) and vp.get("anchor"):
        head["viewpoint_anchor"] = str(vp["anchor"]).strip().lower()
        if vp.get("away"):
            head["viewpoint_away"] = True

    return head


def raw_predicates(expr: dict) -> list[str]:
    """List top-level relation predicates before vocabulary matching."""
    out = []
    for rel in expr.get("relations") or []:
        if isinstance(rel, dict) and rel.get("predicate"):
            out.append(str(rel["predicate"]).strip().lower())
    return out


def lower_one(flat, sentence, rel_num, *, recover, outcomes=None, dropped=None):
    """Clean and lower a parse, returning None for missing or invalid input.

    Callers must choose whether to recover omitted constraints. Optional counters
    record outcomes and unsupported top-level predicates before they are dropped.
    """
    def bump(counter, key):
        if counter is not None:
            counter[key] += 1

    if flat is None:
        bump(outcomes, "parse_null")
        return None
    try:
        cleaned = clean_program(flat, sentence, recover=recover)
        if dropped is not None:
            for p in raw_predicates(cleaned):
                if _norm_predicate(p, rel_num) is None:
                    dropped[p] += 1
        lowered = lower_program(cleaned, rel_num)
    except Exception:  # Invalid parses remain null rows in the evaluation population.
        bump(outcomes, "lower_error")
        return None
    bump(outcomes, "ok" if lowered is not None else "lower_null")
    return lowered


normalise_predicate = _norm_predicate
