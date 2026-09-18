"""Shared normalisation and validation for v4 parser-training targets."""

from __future__ import annotations

import re
from collections import Counter

from grounding.parsing import normalise_predicate, sft_user_turn


SCHEMA_KEYS = ("head", "attributes", "color", "relations")
ATTR_KEYS = ("name", "negated", "rank")
RELATION_KEYS = ("predicate", "negated", "rank", "anchors")
ANCHOR_KEYS = ("category", "attributes", "color")
COLOR_ALIAS = {"grey": "gray", "wood": "wooden"}


def allowed(prompt: str, label: str) -> frozenset[str]:
    match = re.search(rf"^{label}: (.+)$", prompt, re.M)
    if match is None:
        raise SystemExit(f"{label!r} not found in the system prompt")
    return frozenset(value.strip() for value in match.group(1).split(","))


def attribute_for_predicate(attrs_ok, relation_numbers) -> dict[str, str]:
    """Map an executable predicate to its prompt-declared attribute spelling."""
    attributes = {}
    for name in sorted(attrs_ok):
        predicate = normalise_predicate(name, relation_numbers)
        if predicate is None:
            continue
        if attributes.setdefault(predicate, name) != name:
            raise SystemExit(f"predicate {predicate!r} has two v4 attributes")
    return attributes


class Reject(Exception):
    """A row that cannot be expressed in the v4 vocabulary without guessing."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


def _attribute(attribute: dict, relation_numbers, attrs_ok, predicates_ok,
               moves: Counter, drops: Counter, keep_inert: bool,
               attributes_for_predicates: dict) -> tuple[dict | None, dict | None]:
    name = str(attribute.get("name", ""))
    predicate = normalise_predicate(name, relation_numbers)
    negated = bool(attribute.get("negated", False))
    rank = attribute.get("rank") if isinstance(attribute.get("rank"), int) else None
    if predicate is None:
        if keep_inert and name in attrs_ok:
            return {"name": name, "negated": negated, "rank": rank}, None
        drops[f"attribute {name!r} (executor ignored)"] += 1
        return None, None
    if name in attrs_ok:
        return {"name": name, "negated": negated, "rank": rank}, None
    if predicate in predicates_ok:
        moves[f"attribute {name!r} -> relation {predicate!r}"] += 1
        return None, {
            "predicate": predicate,
            "negated": negated,
            "rank": rank,
            "anchors": [],
        }
    if declared := attributes_for_predicates.get(predicate):
        moves[f"attribute {name!r} -> attribute {declared!r}"] += 1
        return {"name": declared, "negated": negated, "rank": rank}, None
    raise Reject("unexpressible_attribute", f"{name!r} -> {predicate!r}")


def _color(value, colors_ok, moves: Counter):
    if value in (None, ""):
        return None
    color = str(value).strip().lower()
    if color in colors_ok:
        return color
    if (alias := COLOR_ALIAS.get(color)) in colors_ok:
        moves[f"color {color!r} -> {alias!r}"] += 1
        return alias
    raise Reject("unexpressible_color", repr(color))


def _anchor(anchor: dict, relation_numbers, attrs_ok, predicates_ok, colors_ok,
            moves: Counter, drops: Counter, keep_inert: bool,
            attributes_for_predicates: dict) -> dict:
    attributes = []
    for attribute in anchor.get("attributes") or []:
        kept, relation = _attribute(
            attribute, relation_numbers, attrs_ok, predicates_ok, moves, drops,
            keep_inert, attributes_for_predicates,
        )
        if relation is not None:
            raise Reject("unexpressible_anchor_attribute", str(attribute.get("name")))
        if kept is not None:
            attributes.append(kept)
    return {
        "category": str(anchor.get("category", "")).strip().lower(),
        "attributes": attributes,
        "color": _color(anchor.get("color"), colors_ok, moves),
    }


def canonicalise(program: dict, relation_numbers, vocabulary, moves: Counter,
                 drops: Counter, keep_inert: bool = False,
                 attributes_for_predicates: dict | None = None) -> dict:
    """Rewrite an accepted program into the declared v4 schema without guessing."""
    attrs_ok, predicates_ok, colors_ok = vocabulary
    attributes_for_predicates = attributes_for_predicates or {}
    attributes, relations = [], []

    for attribute in program.get("attributes") or []:
        kept, relation = _attribute(
            attribute, relation_numbers, attrs_ok, predicates_ok, moves, drops,
            keep_inert, attributes_for_predicates,
        )
        if kept is not None:
            attributes.append(kept)
        if relation is not None:
            relations.append(relation)

    for relation in program.get("relations") or []:
        source_predicate = str(relation.get("predicate", ""))
        predicate = normalise_predicate(source_predicate, relation_numbers)
        if predicate is None:
            drops[f"predicate {source_predicate!r} (executor dropped)"] += 1
            continue

        anchors = relation.get("anchors") or []
        if predicate not in predicates_ok:
            declared = attributes_for_predicates.get(predicate)
            if declared is not None and not anchors:
                moves[f"relation {source_predicate!r} -> attribute {declared!r}"] += 1
                attributes.append({
                    "name": declared,
                    "negated": bool(relation.get("negated", False)),
                    "rank": relation.get("rank")
                    if isinstance(relation.get("rank"), int) else None,
                })
                continue
            raise Reject("unexpressible_predicate", f"{source_predicate!r} -> {predicate!r}")

        if predicate != source_predicate:
            moves[f"predicate {source_predicate!r} -> {predicate!r}"] += 1
        relations.append({
            "predicate": predicate,
            "negated": bool(relation.get("negated", False)),
            "rank": relation.get("rank")
            if isinstance(relation.get("rank"), int) else None,
            "anchors": [
                _anchor(
                    anchor, relation_numbers, attrs_ok, predicates_ok, colors_ok,
                    moves, drops, keep_inert, attributes_for_predicates,
                )
                for anchor in anchors
            ],
        })

    head = str(program.get("head", "")).strip().lower()
    if not head:
        raise Reject("empty_head")
    return {
        "head": head,
        "attributes": attributes,
        "color": _color(program.get("color"), colors_ok, moves),
        "relations": relations,
    }


def violations(program: dict, vocabulary) -> list[str]:
    """Return strict v4 schema and vocabulary violations."""
    attrs_ok, predicates_ok, colors_ok = vocabulary
    violations_found = []
    if tuple(program) != SCHEMA_KEYS:
        violations_found.append(f"top-level keys {tuple(program)}")
    if not isinstance(program.get("head"), str) or not program["head"]:
        violations_found.append("head must be a non-empty string")
    if program.get("color") is not None and program["color"] not in colors_ok:
        violations_found.append(f"color {program.get('color')!r}")
    for attribute in program.get("attributes") or []:
        if tuple(attribute) != ATTR_KEYS:
            violations_found.append(f"attribute keys {tuple(attribute)}")
        if attribute.get("name") not in attrs_ok:
            violations_found.append(f"attribute {attribute.get('name')!r}")
        if not isinstance(attribute.get("negated"), bool):
            violations_found.append("attribute negated must be bool")
        if attribute.get("rank") is not None and not isinstance(attribute["rank"], int):
            violations_found.append("attribute rank must be int or null")
    for relation in program.get("relations") or []:
        if tuple(relation) != RELATION_KEYS:
            violations_found.append(f"relation keys {tuple(relation)}")
        if relation.get("predicate") not in predicates_ok:
            violations_found.append(f"predicate {relation.get('predicate')!r}")
        if not isinstance(relation.get("negated"), bool):
            violations_found.append("relation negated must be bool")
        if relation.get("rank") is not None and not isinstance(relation["rank"], int):
            violations_found.append("relation rank must be int or null")
        anchors = relation.get("anchors")
        if not isinstance(anchors, list):
            violations_found.append("anchors must be a list")
            continue
        if relation["predicate"] == "between" and len(anchors) != 2:
            violations_found.append(f"between takes 2 anchors, got {len(anchors)}")
        for anchor in anchors:
            if tuple(anchor) != ANCHOR_KEYS:
                violations_found.append(f"anchor keys {tuple(anchor)}")
            if not isinstance(anchor.get("category"), str) or not anchor["category"]:
                violations_found.append("anchor category must be a non-empty string")
            if anchor.get("color") is not None and anchor["color"] not in colors_ok:
                violations_found.append(f"anchor color {anchor.get('color')!r}")
            for attribute in anchor.get("attributes") or []:
                if attribute.get("name") not in attrs_ok:
                    violations_found.append(f"anchor attribute {attribute.get('name')!r}")
    return violations_found


def size(program: dict) -> int:
    """Program complexity used for the minimality tie-break."""
    return (len(program["attributes"]) + len(program["relations"])
            + sum(len(relation["anchors"]) for relation in program["relations"]))


user_turn = sft_user_turn
