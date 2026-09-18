"""Thesis calibration and binding, with LaSP's softmax kept for comparison."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from grounding.resolution.config import DEFAULT_RESOLVER

__all__ = ["normalize_concept", "bind1", "bind2"]


def bind1(R, s, config=DEFAULT_RESOLVER):
    """Bind R[i, j] to anchor scores s[j]."""
    if config.combine == "sum":
        return R @ s
    return (R * s.unsqueeze(0)).amax(dim=1)


def bind2(R, s1, s2, config=DEFAULT_RESOLVER):
    """Bind R[i, j, k] to two anchors."""
    if config.combine == "sum":
        return torch.einsum("ijk,j,k->i", R, s1, s2)
    return (R * s1[None, :, None] * s2[None, None, :]).amax(dim=(1, 2))


def normalize_concept(concept, config=DEFAULT_RESOLVER):
    """Calibrate a relation before accumulation.

    ``zscore`` applies softmax after population-standardization;
    ``zscore_sigmoid`` uses sigmoid. Below std=1e-6 both skip standardization
    but still squash raw scores. ``softmax`` is LaSP's own, kept for comparison.
    """
    if config.norm_mode == "softmax":
        return F.softmax(concept, dim=0)
    if config.norm_mode in ("zscore", "zscore_sigmoid"):
        sd = concept.std(unbiased=False)
        # Do not amplify noise in a nearly constant vector.
        if not float(sd) < 1e-6:
            concept = (concept - concept.mean()) / (sd + 1e-6)
        if config.norm_mode == "zscore":
            return F.softmax(concept, dim=0)
        return torch.sigmoid(concept)
    raise ValueError(config.norm_mode)
