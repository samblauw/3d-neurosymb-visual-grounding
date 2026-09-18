from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from grounding.paths import BASELINE_TREE, REPO_ROOT
from grounding.parsing.schema import FlatProgram


__all__ = [
    "REPRESENTATIONS",
    "normalise_representation",
    "representation_tree",
    "load_rel_num",
    "extract_json_objects",
]


REPRESENTATIONS = ("baseline", "extended")

_REPRESENTATION_TREES = {
    "baseline": BASELINE_TREE,
    "extended": REPO_ROOT,
}


def normalise_representation(name: str) -> str:
    """Validate and normalise a representation name."""
    name = str(name).strip().lower()

    if name not in REPRESENTATIONS:
        raise SystemExit(
            f"unknown representation {name!r}; "
            f"expected one of {REPRESENTATIONS}"
        )

    return name


def representation_tree(name: str) -> Path:
    """Working directory for the baseline CLI or the package-native runtime."""
    name = normalise_representation(name)
    return _REPRESENTATION_TREES[name].resolve()


def load_rel_num(representation: str = "extended") -> dict[str, int]:
    """Load the predicate arities defined by an execution tree."""
    if normalise_representation(representation) == "extended":
        from grounding.representation.pipeline import rel_num
    else:
        tree = representation_tree(representation)
        if str(tree) not in sys.path:
            sys.path.insert(0, str(tree))
        from src.relation_encoders.compute_features import rel_num
    return dict(rel_num)


def extract_json_objects(text: str) -> FlatProgram | None:
    """Return the first valid JSON object found in generated text."""
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    candidate_text = fence.group(1) if fence else text

    start = candidate_text.find("{")

    while start != -1:
        depth = 0
        in_string = False
        escaped = False

        for i in range(start, len(candidate_text)):
            char = candidate_text[i]

            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1

                if depth == 0:
                    blob = candidate_text[start:i + 1]

                    try:
                        obj = json.loads(blob)
                    except json.JSONDecodeError:
                        break

                    if isinstance(obj, dict):
                        return obj

                    break

        start = candidate_text.find("{", start + 1)

    return None
