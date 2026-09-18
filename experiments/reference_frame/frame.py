"""Parsed frames and geometry-resolved horizontal directions.

Resolved forward/right vectors lie in the xy plane (z up). An observer origin
is supplied only when the utterance states one; unresolved frames carry codes."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Sequence

import numpy as np

__all__ = ["FrameSource", "ParsedFrame", "ReferenceFrame", "right_of",
           "unresolved", "MIN_BASELINE"]


# Minimum endpoint separation in metres; shorter baselines give unstable axes
MIN_BASELINE = 0.5


class FrameSource(str, Enum):
    """Evidence used to resolve a frame."""

    DIRECT_POINTS = "direct_points"
    SURFACE_NORMAL = "surface_normal"
    CONFIRMED_SUPPORT = "confirmed_support"
    NONE = "none"


@dataclass(frozen=True)
class ParsedFrame:
    """One row of the `parse` stage's output, as a value object.

    The parser is not this task's concern: this reads its schema and never
    writes it. Unknown keys are ignored so a future slot cannot break loading.
    """

    frame_type: str = "none"
    has_viewpoint: bool = False
    from_anchor: Optional[str] = None
    toward_anchor: Optional[str] = None
    away_from_anchor: Optional[str] = None
    at_anchor: Optional[str] = None
    body_pose: Optional[str] = None
    observer_deictic: bool = False
    evidence: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "ParsedFrame":
        data = data or {}
        clean = lambda v: v.strip() or None if isinstance(v, str) else None
        return cls(
            frame_type=str(data.get("frame_type") or "none"),
            has_viewpoint=bool(data.get("has_viewpoint", False)),
            from_anchor=clean(data.get("from_anchor")),
            toward_anchor=clean(data.get("toward_anchor")),
            away_from_anchor=clean(data.get("away_from_anchor")),
            at_anchor=clean(data.get("at_anchor")),
            body_pose=clean(data.get("body_pose")),
            observer_deictic=bool(data.get("observer_deictic", False)),
            evidence=clean(data.get("evidence")),
        )


@dataclass(frozen=True)
class ReferenceFrame:
    """A horizontal observer frame, or a reasoned abstention.

    ``forward``/``right`` are unit (2,) arrays in the scene's xy plane, or None
    when ``resolved`` is False. ``origin`` is filled in only by methods that
    actually locate the observer -- the surface methods constrain a DIRECTION
    and leave it None, because "facing the window" says nothing about where in
    the room the speaker stands.
    """

    resolved: bool
    method: str
    reason: str
    source: FrameSource = FrameSource.NONE
    confidence: float = 0.0
    forward: Optional[np.ndarray] = None
    right: Optional[np.ndarray] = None
    origin: Optional[np.ndarray] = None
    detail: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        vec = lambda v: None if v is None else [float(v[0]), float(v[1])]
        return {
            "resolved": self.resolved,
            "method": self.method,
            "reason": self.reason,
            "source": self.source.value,
            "confidence": round(float(self.confidence), 4),
            "forward": vec(self.forward),
            "right": vec(self.right),
            "origin": vec(self.origin),
            "detail": self.detail,
        }


def right_of(forward: Sequence[float]) -> np.ndarray:
    """The right-hand axis for a z-up forward.

    Matched to the sign convention already in the left/right encoders, where
    ``left = (-u_y, u_x)`` (a +90 degree turn about +z). Right is its negation,
    so a frame handed to those encoders and this ``right`` agree.
    """
    u = np.asarray(forward, dtype=float)[:2]
    return np.array([u[1], -u[0]])


def _unit(vec) -> Optional[np.ndarray]:
    v = np.asarray(vec, dtype=float)[:2]
    n = float(np.linalg.norm(v))
    return None if n < 1e-9 else v / n


def resolved(method: str, forward, reason: str, source: FrameSource,
             confidence: float, *, origin=None, **detail) -> ReferenceFrame:
    """Build a resolved frame, or abstain if the direction degenerates."""
    u = _unit(forward)
    if u is None:
        return unresolved(method, "degenerate_direction",
                          "The geometry produced a zero-length forward vector.")
    return ReferenceFrame(
        resolved=True, method=method, reason=reason, source=source,
        confidence=float(confidence), forward=u, right=right_of(u),
        origin=None if origin is None else np.asarray(origin, dtype=float)[:2],
        detail=detail,
    )


def unresolved(method: str, code: str, reason: str, **detail) -> ReferenceFrame:
    """Abstain with a machine-readable code and an explanation."""
    return ReferenceFrame(resolved=False, method=method, reason=reason,
                          source=FrameSource.NONE, confidence=0.0,
                          detail={"code": code, **detail})
