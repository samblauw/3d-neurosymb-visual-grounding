"""Influence diagnostics for the resolver.

Both hooks are OFF by default, so the scored path runs the same arithmetic it
would without them. ``start_trace`` collects every factor the resolver folds
into the product, with the recursion depth it was folded in at: 0 is the
description itself, and each anchor sits one level below the relation naming
it. ``skip`` names factors to hold out, which is how the leave-one-factor-out
swing is measured. ``binds``, when given a list, also receives the anchor score
vectors every relation binds against.

Factor ids are POSITIONAL, so the same traversal on a held-out re-run names the
same node. Every call site must therefore run whether or not the factor is
applied -- holding one out must not change control flow, only whether the value
is multiplied in.
"""

from __future__ import annotations

from contextlib import contextmanager

from grounding.resolution import scoring
from grounding.resolution.config import DEFAULT_RESOLVER

__all__ = ["start_trace", "stop_trace", "factor", "nested", "bind1", "bind2"]


_TRACE = None
_BINDS = None
_SKIP = frozenset()
_FIDX = 0
_DEPTH = 0


def start_trace(skip=(), binds=None):
    global _TRACE, _BINDS, _SKIP, _FIDX
    _TRACE, _BINDS, _SKIP, _FIDX = [], binds, frozenset(skip), 0
    return _TRACE


def stop_trace():
    global _TRACE, _BINDS, _SKIP
    trace, _TRACE, _BINDS, _SKIP = _TRACE, None, None, frozenset()
    return trace


@contextmanager
def nested():
    """Resolve an anchor one level below the description that names it."""
    global _DEPTH
    _DEPTH += 1
    try:
        yield
    finally:
        _DEPTH -= 1


def factor(kind, name, vec, raw=None):
    """Capture a factor and return whether it should be applied on this run."""
    global _FIDX
    if _TRACE is None:
        return True
    key = f"{kind}:{name}#{_FIDX}"
    _FIDX += 1
    _TRACE.append({"key": key, "kind": kind, "name": name, "depth": _DEPTH,
                   "vec": vec.detach().clone(),
                   "raw": None if raw is None else raw.detach().clone()})
    return key not in _SKIP


def _record_bind(*anchors):
    if _BINDS is not None:
        _BINDS.append({"depth": _DEPTH,
                       "anchors": [a.detach().clone() for a in anchors]})


def bind1(R, s, config=DEFAULT_RESOLVER):
    """``scoring.bind1``, recording the anchor scores while tracing."""
    _record_bind(s)
    return scoring.bind1(R, s, config)


def bind2(R, s1, s2, config=DEFAULT_RESOLVER):
    """``scoring.bind2``, recording both anchor score vectors while tracing."""
    _record_bind(s1, s2)
    return scoring.bind2(R, s1, s2, config)
