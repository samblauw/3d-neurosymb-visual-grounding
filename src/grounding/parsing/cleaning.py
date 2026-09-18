"""Repair flat parser output using attribute cues and structural-anchor rules.

Optional recovery adds omitted constraints. Output remains flat for lowering.
"""

from __future__ import annotations

import re

__all__ = ["clean_program"]


_AGAINST_WALL = (
    r"\b(?:up\s+)?against\s+(?:the|a)\s+wall|\balong\s+the\s+wall|"
    r"\bflush\s+against|\bpushed\s+against|\battached\s+to\s+the\s+wall|"
    r"\btouching\s+the\s+wall|\bnext\s+to\s+the\s+wall|\bby\s+the\s+wall|"
    r"\bon\s+the\s+wall\b|\bwall\s*-?\s*mounted\b"
)

# Attributes without a trigger rule pass through unchanged.
_ATTR_TRIGGER: dict[str, str] = {
    "large": r"\b(large|larger|largest|big|bigger|biggest|long|longer|wide|wider)\b",
    "small": r"\b(small|smaller|smallest|tiny|little|narrow|narrower|short|shorter)\b",
    "high": r"\b(high|higher|highest|tall|taller|upper|top)\b",
    "low": r"\b(low|lower|lowest|bottom)\b",
    "on_the_floor": r"\b(floor|ground)\b",
    "against_wall": _AGAINST_WALL,
    "at_corner": r"\b(corner|end)\b",
}

# Single structural anchors become attributes on the target.
_CONCEPT_ANCHOR: dict[str, str] = {
    "bottom": "low", "top": "high", "end": "at_corner",
    "floor": "on_the_floor", "ground": "on_the_floor",
    "corner": "at_corner", "wall": "against_wall",
}

_SUBPART = frozenset({"foot", "feet", "end", "head", "side", "edge", "base", "tip"})
_ORDINAL_WORD = {
    "first", "second", "third", "fourth", "fifth", "sixth",
    "1st", "2nd", "3rd", "4th", "5th", "6th",
}
_ORDINAL_FORMS = {
    1: ("first", "1st"), 2: ("second", "2nd"), 3: ("third", "3rd"),
    4: ("fourth", "4th", "forth"), 5: ("fifth", "5th"), 6: ("sixth", "6th"),
}
_LATERAL = {"left", "right", "front", "back", "behind"}
_SUPERLATIVE = r"\b(\w+est|most)\b"
_NEG_CUE = re.compile(r"\b(without|not|no|never|isn't|aren't|doesn't|don't|lacks?)\b|n't")


def _negated_context(utterance: str, word: str, window: int = 5) -> bool:
    """Look for a negation cue before the final token of a phrase."""
    if not word:
        return False
    toks = re.findall(r"[a-z']+", utterance.lower())
    w = word.lower().split()[-1]
    for i, t in enumerate(toks):
        if t == w:
            slice_tokens = toks[max(0, i - window):i]
            if any(_NEG_CUE.fullmatch(x) or _NEG_CUE.search(x) for x in slice_tokens):
                return True
    return False


def _gate_attrs(attrs: list[dict] | None, utterance: str) -> list[dict]:
    kept = []
    for a in attrs or []:
        name = str(a.get("name", "")).lower()
        pat = _ATTR_TRIGGER.get(name)
        ordinal_forms = _ORDINAL_FORMS.get(a.get("rank"), (name,))
        mentioned = (
            any(
                re.search(rf"\b{re.escape(form)}\b", utterance)
                for form in ordinal_forms
            )
            if name in _ORDINAL_WORD
            else pat is None or re.search(pat, utterance)
        )
        if mentioned:
            kept.append(a)
    return kept


def _rehome_ordinal_attrs(e: dict) -> dict:
    """Move the first ordinal rank to a lateral relation or attribute."""
    attrs = e.get("attributes") or []
    carried = next(
        (
            a["rank"] for a in attrs
            if str(a.get("name", "")).lower() in _ORDINAL_WORD
            and isinstance(a.get("rank"), int) and a["rank"] >= 2
        ),
        None,
    )
    if carried is None:
        return e

    e["attributes"] = [
        a for a in attrs if str(a.get("name", "")).lower() not in _ORDINAL_WORD
    ]
    for r in e.get("relations") or []:
        if str(r.get("predicate", "")).lower() in _LATERAL:
            r["rank"] = carried
            return e

    for a in e["attributes"]:
        if str(a.get("name", "")).lower() in _LATERAL:
            a["rank"] = carried
            break
    return e


def clean_program(expr: dict, utterance: str, recover: bool = True) -> dict:
    """Gate attributes, relocate ordinals and rewrite structural anchors.

    Recovery also infers missing high/low attributes and relation negation.
    """
    u = utterance.lower()
    e = dict(expr)
    # Copy records before editing so repeated cleaning preserves the input.
    e["attributes"] = [dict(a) for a in expr.get("attributes") or []]
    e["relations"] = [dict(r) for r in expr.get("relations") or []]
    e["attributes"] = _gate_attrs(e.get("attributes"), u)
    e = _rehome_ordinal_attrs(e)

    if not re.search(_SUPERLATIVE, u):
        e["attributes"] = [
            a if isinstance(a.get("rank"), int) and a["rank"] >= 2 else dict(a, rank=None)
            for a in e["attributes"]
        ]

    def _add(name: str) -> None:
        if name not in {a["name"] for a in e["attributes"]}:
            e["attributes"].append({"name": name, "negated": False, "rank": None})

    if (
        recover
        and re.search(r"\b(bottom|lower|lowest)\b", u)
        and "low" not in {a["name"] for a in e["attributes"]}
    ):
        _add("low")
    if (
        recover
        and re.search(r"\b(top|upper|highest)\b", u)
        and "on top" not in u
        and "high" not in {a["name"] for a in e["attributes"]}
    ):
        _add("high")

    for a, b in (("large", "small"), ("high", "low")):
        cur = {x["name"] for x in e["attributes"]}
        if a in cur and b in cur:
            e["attributes"] = [x for x in e["attributes"] if x["name"] not in (a, b)]

    rels = []
    for r in e.get("relations") or []:
        anchors = r.get("anchors") or []
        cat = anchors[0]["category"].lower() if anchors else ""

        if len(anchors) == 1 and cat in _SUBPART:
            if m_sub := re.search(
                rf"\b{re.escape(cat)}\s+of\s+(?:the\s+|an?\s+)?([a-z]+)", u
            ):
                anchors = [dict(anchors[0], category=m_sub.group(1))]

        cat = anchors[0]["category"].lower() if anchors else ""
        if len(anchors) == 1 and cat in _CONCEPT_ANCHOR:
            attr_name = _CONCEPT_ANCHOR[cat]
            pat = _ATTR_TRIGGER.get(attr_name)
            if pat is not None and not re.search(pat, u):
                continue
            attr = {
                "name": attr_name,
                "negated": _negated_context(u, cat),
                "rank": None,
            }
            if attr["name"] not in {a["name"] for a in e["attributes"]}:
                e["attributes"].append(attr)
            continue

        r2 = dict(r)
        if recover and not r2.get("negated") and cat and _negated_context(u, cat):
            r2["negated"] = True
        r2["anchors"] = [
            dict(a, attributes=_gate_attrs(a.get("attributes"), u)) for a in anchors
        ]
        rels.append(r2)

    e["relations"] = rels
    return e
