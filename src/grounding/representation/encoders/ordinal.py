"""Raw ordinal scores: exp(-distance from the requested rank / config.ordinal_tau).

The resolver normalises these just like any other relation's raw scores.
"""

from __future__ import annotations

import torch

from grounding.resolution.config import DEFAULT_RESOLVER

__all__ = ["rank_of", "membership", "is_row", "ordinal_count_peak", "ordinal_peak"]


def membership(sub_concept):
    """Recover candidate membership from a sum-normalised category vector."""
    peak = sub_concept.max()
    return sub_concept / peak if float(peak) > 0 else sub_concept


def is_row(counts, w, thresh=0.5):
    """Whether candidate relation counts define an unambiguous row order."""
    values = [round(float(count), 4) for count, weight in zip(counts, w)
              if float(weight) > thresh]
    return len(values) >= 2 and len(set(values)) == len(values)


def _rank_score(offset, config):
    return torch.exp(-offset.abs().float() / config.ordinal_tau)


def ordinal_count_peak(counts, mass, rank, config=DEFAULT_RESOLVER):
    """Raw rank score from peer counts, without ranking non-candidates."""
    k = min(float(rank), float(mass)) if float(mass) > 0 else float(rank)
    return _rank_score(counts - (mass - k), config)


def ordinal_peak(key, rank, config=DEFAULT_RESOLVER):
    """Raw rank score from a descending order over `key`."""
    n = key.shape[0]
    order = torch.argsort(key, descending=True)
    pos = torch.empty(n, dtype=torch.long, device=key.device)
    pos[order] = torch.arange(n, device=key.device)
    k = min(int(rank), n) - 1
    return _rank_score(pos - k, config)


def rank_of(relation_item, config=DEFAULT_RESOLVER):
    # Rank 1 already means the unranked extreme.
    if not config.use_rank:
        return None
    rank = relation_item.get("rank")
    return rank if isinstance(rank, int) and rank >= 2 else None
