"""Reference-frame diagnostic: five geometry-based methods and coded abstentions.

The dispatcher chooses one method from the parsed frame; conditioning replaces
per-row directional features. This diagnostic is separate from the final system."""

from experiments.reference_frame.frame import (MIN_BASELINE,
                                                            FrameSource,
                                                            ParsedFrame,
                                                            ReferenceFrame,
                                                            right_of)
from experiments.reference_frame.methods import (METHOD_NAMES,
                                                              METHODS)
from experiments.reference_frame.resolver import (
    method_for, resolve_reference_frame)
from experiments.reference_frame.scene import (AnchorHit,
                                                            SceneFrameContext)
from experiments.reference_frame.support import (SupportingWall,
                                                              supporting_wall)

__all__ = [
    "resolve_reference_frame", "method_for", "METHODS", "METHOD_NAMES",
    "ReferenceFrame", "ParsedFrame", "FrameSource", "right_of", "MIN_BASELINE",
    "SceneFrameContext", "AnchorHit", "SupportingWall", "supporting_wall",
]
