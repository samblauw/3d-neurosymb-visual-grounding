"""One definition of resolver settings recorded by experiment scripts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import json
import sys

__all__ = ["BASELINE_RESOLVER", "COLOR_MODES", "COMBINE_MODES", "CONTROL_FLAGS",
           "FROZEN_RESOLVER", "NORM_MODES", "add_resolver_flags",
           "ResolverConfig", "DEFAULT_RESOLVER", "resolver_metadata",
           "resolver_from_args", "resolver_suffix", "shared_resolver"]


BASELINE_RESOLVER = {"norm_mode": "softmax", "combine": "sum",
                     "use_rank": False}

# Target configuration for new runs. Recorded results retain their own settings.
FROZEN_RESOLVER = {
    "norm_mode": "zscore_sigmoid", "combine": "sum", "color_mode": "product",
    "use_rank": True, "ordinal_tau": 1.0,
}

NORM_MODES = ("softmax", "zscore", "zscore_sigmoid")
COMBINE_MODES = ("sum", "max")
COLOR_MODES = ("min", "product")

#: Flag defaults for the diagnostics that run against LaSP's arithmetic. LaSP
#: has no colour factor, so colour takes the retained product.
CONTROL_FLAGS = {**BASELINE_RESOLVER, "color_mode": FROZEN_RESOLVER["color_mode"]}


@dataclass(frozen=True)
class ResolverConfig:
    """Numerical settings for one resolver run, including recursive anchors."""

    norm_mode: str = FROZEN_RESOLVER["norm_mode"]
    combine: str = FROZEN_RESOLVER["combine"]
    color_mode: str = FROZEN_RESOLVER["color_mode"]
    use_rank: bool = FROZEN_RESOLVER["use_rank"]
    ordinal_tau: float = FROZEN_RESOLVER["ordinal_tau"]

    def __post_init__(self):
        if self.norm_mode not in NORM_MODES:
            raise ValueError(self.norm_mode)
        if self.combine not in COMBINE_MODES:
            raise ValueError(self.combine)
        if self.color_mode not in COLOR_MODES:
            raise ValueError(self.color_mode)
        if not self.ordinal_tau > 0:
            raise ValueError(f"ordinal_tau must be positive, got {self.ordinal_tau}")

    def updated(self, **changes):
        return replace(self, **changes)


DEFAULT_RESOLVER = ResolverConfig()


def resolver_metadata(config=DEFAULT_RESOLVER, **metadata) -> dict:
    """Serialize the settings used for a run, preserving the result schema."""
    values = {**metadata, **asdict(config)}
    values["baseline_resolver_arithmetic"] = all(
        values[key] == value for key, value in BASELINE_RESOLVER.items())
    return values


def resolver_suffix(config: dict) -> str:
    """Filename suffix for the differences from baseline resolver arithmetic."""
    if isinstance(config, ResolverConfig):
        config = resolver_metadata(config)
    if config["baseline_resolver_arithmetic"]:
        return ""
    changed = [
        config["norm_mode"] if config["norm_mode"] != BASELINE_RESOLVER["norm_mode"] else None,
        config["combine"] if config["combine"] != BASELINE_RESOLVER["combine"] else None,
        "rank" if config["use_rank"] else None,
    ]
    return "_" + "_".join(part for part in changed if part)


def add_resolver_flags(parser, defaults):
    """Add --norm-mode, --combine, --color-mode and --use-rank/--no-use-rank."""
    parser.add_argument("--norm-mode", dest="norm_mode", choices=NORM_MODES,
                        default=defaults["norm_mode"],
                        help="per-condition normalisation")
    parser.add_argument("--combine", choices=COMBINE_MODES,
                        default=defaults["combine"], help="anchor binding")
    parser.add_argument("--color-mode", dest="color_mode", choices=COLOR_MODES,
                        default=defaults["color_mode"],
                        help="how colour combines with the category term")
    parser.add_argument("--use-rank", dest="use_rank", action="store_true",
                        default=defaults["use_rank"],
                        help="honour a relation's rank")
    parser.add_argument("--no-use-rank", dest="use_rank", action="store_false",
                        help="ignore ranks")
    return parser


def resolver_from_args(args) -> ResolverConfig:
    """Build settings from CLI flags without changing another run's state."""
    return ResolverConfig(norm_mode=args.norm_mode, combine=args.combine,
                          color_mode=args.color_mode, use_rank=args.use_rank,
                          ordinal_tau=(DEFAULT_RESOLVER.ordinal_tau
                                       if getattr(args, "ordinal_tau", None) is None
                                       else args.ordinal_tau))


def shared_resolver(configs: dict, hint: str) -> dict:
    """The one resolver configuration every named run recorded, or exit."""
    signatures = {json.dumps(c, sort_keys=True) for c in configs.values()}
    if len(signatures) != 1 or None in configs.values():
        sys.exit("runs were scored under different resolver configurations, or "
                 f"one records none: {json.dumps(configs, indent=1)}\n  {hint}")
    return next(iter(configs.values()))
