"""Four representation scales and their one-at-a-time ±50% sweeps.

Wall, isolation and colour scales require rebuilding features; ordinal tau acts
at resolution time. GUARDS documents fixed numerical tolerances, which are not
swept."""

from __future__ import annotations

from grounding.representation.encoders import color as _color
from grounding.resolution.config import DEFAULT_RESOLVER
from grounding.representation.encoders.against_wall import AgainstTheWall
from grounding.representation.encoders.isolation import Isolated

__all__ = ["PARAMETERS", "GUARDS", "REBUILDS_FEATURES", "get", "set",
           "snapshot", "sweep_values"]


#: name -> (getter, setter, shipped value, units, what the scale means).
#: The shipped value is recorded here as a literal so a sweep that forgets to
#: restore state cannot silently redefine what "shipped" was.
PARAMETERS = {
    "wall.gap_scale": (
        lambda: AgainstTheWall.GAP_SCALE,
        lambda v: setattr(AgainstTheWall, "GAP_SCALE", float(v)),
        0.25, "metres",
        "gap at which an object stops counting as against the wall"),
    "isolation.distance_scale": (
        lambda: Isolated.DISTANCE_SCALE,
        lambda v: setattr(Isolated, "DISTANCE_SCALE", float(v)),
        1.0, "metres",
        "gap to the nearest same-category peer at which an object is isolated"),
    "color.scale": (
        lambda: _color.SCALE,
        lambda v: setattr(_color, "SCALE", float(v)),
        0.35, "RGB distance in [0,1]^3",
        "RGB distance at which a colour stops matching"),
    "ordinal.tau": (
        None, None,
        1.0, "rank positions",
        "offset from the requested rank at which a candidate stops being it"),
}

#: Which parameters are read by an encoder that runs at feature-build time. A
#: sweep of these is only valid against a feature tensor rebuilt under them.
REBUILDS_FEATURES = frozenset(
    {"wall.gap_scale", "isolation.distance_scale", "color.scale"})

#: NOT parameters. Each keeps arithmetic well-defined; none sets the scale of
#: anything the method claims to measure, and sweeping one measures the
#: numerics, not the model.
GUARDS = {
    "against_wall.MIN_EXTENT": "0.2 m floor on the overlap denominator, so a "
                               "picture is not judged on a near-zero extent",
    "scoring.zscore sd threshold": "1e-6; below it a vector is constant and "
                                   "zscore declines to invent a preference",
    "scoring.zscore epsilon": "1e-6 added to the standard deviation",
    "ordinal.is_row thresh": "0.5 membership cut, and round(counts, 4); both "
                             "read a hard 0/1 signal, not a scale",
    "color.weight epsilon": "1e-9 when renormalising the GMM mixture weights",
}


def get(name, config=DEFAULT_RESOLVER):
    return config.ordinal_tau if name == "ordinal.tau" else PARAMETERS[name][0]()


def set(name, value, config=DEFAULT_RESOLVER):
    """Set an encoder scale, or return updated settings for resolver-time tau."""
    if name == "ordinal.tau":
        return config.updated(ordinal_tau=float(value))
    PARAMETERS[name][1](value)
    return config


def snapshot(config=DEFAULT_RESOLVER):
    """Every parameter's current value, for a run manifest."""
    return {name: get(name, config) for name in PARAMETERS}


def sweep_values(name, fraction=0.5):
    """The one-at-a-time arms for `name`: (low, shipped, high)."""
    shipped = PARAMETERS[name][2]
    return (shipped * (1.0 - fraction), shipped, shipped * (1.0 + fraction))
