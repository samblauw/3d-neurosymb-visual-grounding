"""Work through a parsed description and give each candidate object a score.

Colour goes with the category score; relations are scored separately.
Viewpoint anchors aren't used here, so relations stay in the fixed frame.
"""

from __future__ import annotations

import torch

from grounding.parsing.fields import relation_objects
from grounding.representation.encoders.ordinal import (
    is_row, membership, ordinal_count_peak, ordinal_peak, rank_of,
)
from grounding.resolution.config import DEFAULT_RESOLVER
from grounding.resolution.scoring import normalize_concept
from grounding.resolution.tracing import bind1 as _bind1
from grounding.resolution.tracing import bind2 as _bind2
from grounding.resolution.tracing import factor as _factor
from grounding.resolution.tracing import nested as _nested

__all__ = ["parse", "DEVICE"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse(scan_id, json_obj, all_concepts, valid_ids, *,
          rel_num, valid_relations, config=DEFAULT_RESOLVER):
    # Anchors are descriptions too, so they go through the same resolver.
    def resolve(obj):
        with _nested():
            return parse(scan_id, obj, all_concepts, valid_ids,
                         rel_num=rel_num, valid_relations=valid_relations, config=config)

    final_concept = torch.ones(len(valid_ids), device=DEVICE)
    category = json_obj["category"]
    appearance = all_concepts[category][valid_ids]

    if category in ("corner", "middle", "room", "center"):
        return final_concept
    if category in valid_relations:
        raise NotImplementedError

    # The trace hook can leave a factor out to measure how much it matters.
    if _factor("cat", category, appearance):
        final_concept = torch.minimum(final_concept, appearance)

    # Colour belongs with "chair", not with relations like "next to".
    color = json_obj.get("color")
    if color:
        color = str(color).strip().lower()
        key = f"color:{color}"
        if key in all_concepts:
            value = all_concepts[key][valid_ids]
            if _factor("color", color, value):
                product = config.color_mode == "product"
                final_concept = (final_concept * value if product
                                 else torch.minimum(final_concept, value))

    for relation in json_obj.get("relations", []):
        name = relation["relation_name"]
        if name not in valid_relations:
            continue
        features = all_concepts[name]
        arity = rel_num[name]
        concept = torch.ones(len(valid_ids), device=DEVICE)
        rank = rank_of(relation, config)
        ranked = False

        if arity == 0:
            # "Largest", "highest", etc. already have one score per object.
            concept = features[valid_ids]
            if rank:
                concept = ordinal_peak(concept, rank, config)
                ranked = True

        elif arity == 1:
            features = features[valid_ids, :][:, valid_ids]
            anchors = relation_objects(relation)
            if anchors:
                anchor = anchors[0]
            else:
                # "Leftmost red chair" still compares with chairs of every colour.
                anchor = dict(json_obj, relations=[])
                anchor.pop("color", None)
                anchor.pop("shape", None)
            anchor_score = resolve(anchor)

            if anchors:
                concept = concept * _bind1(features, anchor_score, config)
                if rank:
                    # Let the head category weigh in before choosing a rank.
                    concept = ordinal_peak(concept * appearance, rank, config)
                    ranked = True
            else:
                if rank:
                    # Count who is left/right of whom; distance weights can
                    # scramble the row order. Only use the rank if it's clear.
                    weights = membership(anchor_score)
                    counts = (features > 0).float() @ weights
                    if is_row(counts, weights):
                        concept = ordinal_count_peak(counts, weights.sum(), rank, config)
                        ranked = True
                if not ranked:
                    concept = concept * _bind1(features, anchor_score, config)

        elif arity == 2:
            features = features[valid_ids, :, :][:, valid_ids, :][:, :, valid_ids]
            anchors = relation_objects(relation)
            anchor = anchors[0] if anchors else dict(json_obj, relations=[])
            # Reuse a lone anchor; ignore extras after the first two, like LaSP.
            first = resolve(anchor)
            second = resolve(anchors[1]) if len(anchors) >= 2 else first
            # "Between" has no axis to count along, so there's no rank step.
            concept = _bind2(features, first, second, config)
        else:
            raise NotImplementedError

        # Keep the raw scores so diagnostics can show what normalisation changed.
        raw = concept
        concept = normalize_concept(concept, config)
        if relation.get("negative") == True:
            concept = concept.max() - concept
        label = name + ("|ranked" if ranked else "")
        if _factor("rel", label, concept, raw=raw):
            final_concept = final_concept * concept

    return final_concept
