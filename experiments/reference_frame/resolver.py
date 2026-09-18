"""Route a parsed frame to one geometry method using frame type and anchor class.

A disabled or abstaining method leaves the frame unresolved; no fallback changes
which method receives credit in the diagnostic."""

from __future__ import annotations

from typing import Iterable, Optional, Union

from experiments.reference_frame import lexicon
from experiments.reference_frame.frame import (ParsedFrame,
                                                            ReferenceFrame,
                                                            unresolved)
from experiments.reference_frame.methods import (METHOD_NAMES,
                                                              METHODS)

__all__ = ["resolve_reference_frame", "method_for", "METHOD_NAMES"]


#: Why a frame type carries no method at all. These are abstentions decided by
#: the language alone -- no scene can rescue them -- and they are separated from
#: the method-level ones so the report can tell "nothing to compute" from
#: "computed and declined"
_NO_METHOD = {
    "none": ("no_explicit_frame",
             "The utterance states no observer or reference frame."),
    "egocentric_only": ("missing_observer_pose",
                        "'your left/right' with no stated position or facing: "
                        "the representation carries no observer pose."),
    "at_only": ("position_without_facing",
                "An observer position is stated but no facing direction. "
                "Position alone does not define left and right."),
}


def method_for(frame: ParsedFrame) -> Optional[str]:
    """Choose a method from the parse, before consulting scene geometry."""
    t = frame.frame_type

    if t in ("at_toward", "from_toward"):
        return "point_to_point"

    if t == "toward_only":
        cls = lexicon.anchor_class(frame.toward_anchor)
        if cls == "strong":
            return "surface_normal"
        if cls == "conditional":
            return "conditional_support"
        return None

    if t == "from_only":
        phrase = lexicon.normalise(frame.from_anchor).split()
        if any(term in phrase for term in lexicon.DOORWAY_TERMS):
            return "doorway_inward"
        return None

    if t in ("away_only", "away_toward"):
        return "away_from_surface"

    return None


def _language_only_abstention(frame: ParsedFrame) -> Optional[ReferenceFrame]:
    if frame.frame_type in _NO_METHOD:
        code, reason = _NO_METHOD[frame.frame_type]
        return unresolved("none", code, reason)

    if frame.frame_type == "toward_only":
        phrase = frame.toward_anchor
        if lexicon.intrinsic_part(phrase):
            return unresolved("none", "needs_intrinsic_orientation",
                              f"'{phrase}' names an object's own front or back; "
                              "object orientation is not represented.")
        return unresolved("none", "toward_anchor_side_unknown",
                          f"Facing '{phrase}' gives a target but not which side "
                          "of it the observer stood on, and the category implies "
                          "no surface.")

    if frame.frame_type == "from_only":
        return unresolved("none", "source_without_target",
                          f"Source '{frame.from_anchor}' is stated with no look "
                          "target and no entry geometry.")

    return unresolved("none", "unsupported_frame_type",
                      f"Frame type '{frame.frame_type}' has no resolution rule.")


def resolve_reference_frame(
    parsed_frame: Union[ParsedFrame, dict, None],
    scene,
    *,
    enabled: Optional[Iterable[str]] = None,
) -> ReferenceFrame:
    """Resolve ParsedFrame or the frame-pass dictionary against SceneFrameContext.

    ``enabled`` limits the allowed methods. Missing geometric evidence returns
    an abstention with detail["code"]; malformed inputs can still raise."""
    frame = (parsed_frame if isinstance(parsed_frame, ParsedFrame)
             else ParsedFrame.from_dict(parsed_frame))
    allowed = set(METHOD_NAMES if enabled is None else enabled)

    name = method_for(frame)
    if name is None:
        return _language_only_abstention(frame)
    if name not in allowed:
        return unresolved(name, "method_disabled",
                          f"Frame type '{frame.frame_type}' routes to '{name}', "
                          "which is switched off in this run.")
    return METHODS[name](frame, scene)
