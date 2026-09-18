"""Tie policies for evaluation and probes.

Headline prediction uses top-k ordering; expected_rank gives probes fractional
credit over ties. Checkpoint selection separately credits every top tie."""

from __future__ import annotations

__all__ = ["expected_rank", "predicted_index", "OFFICIAL_TOTALS"]


# a k-way tie at the top is worth 1/k, not 1.0 or 0.0 -- ties are common here
# (gate a unary and every survivor is equal) so the convention decides the number
def expected_rank(scores, target, tol=1e-9):
    s = scores[target]
    better = sum(1 for x in scores if x > s + tol)
    tied = sum(1 for x in scores if abs(x - s) <= tol)
    return {
        "top1": 1.0 / tied if better == 0 else 0.0,
        "rr": sum(1.0 / j for j in range(better + 1, better + tied + 1)) / tied,
        "rank": better + (tied + 1) / 2,
        "tied": tied,
    }


def predicted_index(final_concept, top_k: int = 5) -> int:
    """Predict using torch.topk; argmax would change the tie policy."""
    import torch
    return int(torch.topk(final_concept,
                          min(top_k, len(final_concept))).indices[0])


#: Re-exported for convenience; the definition lives in `evaluate`.
from grounding.resolution.evaluate import OFFICIAL_TOTALS  # noqa: E402
