#!/usr/bin/env python3
"""Stage II (§5.5) experiments.

  norm    which per-condition normalisation: 12 mechanisms, product fixed
  final   which rule combines conditions: product, mean, harmonic, min,
          entropy-weighted product; --calibration crosses it with `norm`
  anchor  sum vs max anchor binding, optionally per normalisation

`norm` and `final` recombine the factors the resolver folds in, after first
rebuilding the engine's own prediction from them; a run where any row disagrees
exits non-zero. `anchor` re-runs the resolver once per arm. `norm` and `final`
use the C>=2 dev pool, `anchor` the C=1 pool, both written by
build_stage2_subsets.py. Commands are in README.md.
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

from grounding.resolution import engine
from grounding.resolution.config import DEFAULT_RESOLVER
from grounding.resolution.stage2 import (ANCHOR_C1_PARSES, ANCHOR_STRATA,
                                         CGE2_PARSES, DIAGNOSTICS_OUT, STAGE2_OUT,
                                         accuracy_table, anchor_eligible,
                                         anchor_stratum, executed_rows, guard_split,
                                         load_rows, load_tree, predicted_id, row_key,
                                         scene_batches, traced_parse, write_report)

NORM_MODES = ("released", "center_only", "std_only", "zscore", "zscore_sigmoid",
              "temp:0.01", "temp:0.05", "temp:0.1", "temp:0.25", "temp:0.5",
              "temp:2.0", "range")
FACTOR_MODES = ("product", "mean", "harmonic", "min", "entropy_weighted_product")

#: The resolver's own epsilon and constant-vector bypass, so zscore arms rebuild the engine
_ZS_EPS = 1e-6
_ZS_BYPASS = 1e-6


def comma_list(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def arms_for(modes, calibrations):
    return [(f"{cal}+{mode}" if cal else mode, cal, mode)
            for cal in calibrations or [None] for mode in modes]


def renormalize(mode, c, torch, F):
    """One condition's raw score vector under one normalisation mechanism."""
    if mode == "released":
        return F.softmax(c, dim=0)
    if mode == "center_only":
        return F.softmax(c - c.mean(), dim=0)
    if mode in ("std_only", "zscore", "zscore_sigmoid"):
        squash = torch.sigmoid if mode == "zscore_sigmoid" else (
            lambda x: F.softmax(x, dim=0))
        sd = c.std(unbiased=False)
        if float(sd) < _ZS_BYPASS:
            return squash(c)
        centred = c if mode == "std_only" else c - c.mean()
        return squash(centred / (sd + _ZS_EPS))
    if mode.startswith("temp:"):
        return F.softmax(c / float(mode.split(":", 1)[1]), dim=0)
    if mode == "range":
        mn, mx = c.min(), c.max()
        return F.softmax((c - mn) / (mx - mn + _ZS_EPS), dim=0)
    raise ValueError(mode)


def factor_is_negated(factor, mode, torch, F):
    """Negation is not traced, so infer it from which reading rebuilds `vec`."""
    if factor["raw"] is None:
        return False
    normalized = renormalize(mode, factor["raw"], torch, F)
    direct = (normalized - factor["vec"]).abs().max()
    negated = (normalized.max() - normalized - factor["vec"]).abs().max()
    return bool(direct > negated)


def recalibrate(factors, negated, mode, torch, F):
    """Re-normalise traced factors under `mode`, re-applying each negation."""
    out = []
    for f, neg in zip(factors, negated):
        if f["raw"] is None:
            out.append(f["vec"])
            continue
        vec = renormalize(mode, f["raw"], torch, F)
        out.append(vec.max() - vec if neg else vec)
    return out


def aggregate(mode, factors, eps):
    """Combine C per-condition (M,) factors into one (M,) score."""
    import torch

    k = len(factors)
    if k == 1:
        return factors[0]
    stack = torch.stack(factors)                        # (C, M)
    if mode == "product":
        return stack.prod(dim=0)
    if mode == "mean":
        return stack.mean(dim=0)
    if mode == "harmonic":
        return k / (1.0 / (stack + eps)).sum(dim=0)
    if mode == "min":
        return stack.min(dim=0).values
    if mode == "entropy_weighted_product":
        m = stack.shape[1]
        if m < 2:
            return stack.prod(dim=0)
        p = stack / stack.sum(dim=1, keepdim=True).clamp(min=eps)
        # A peaked condition (entropy near 0) gets weight ~1, a flat one ~0
        h = -(p * (p + eps).log()).sum(dim=1)
        w = (1.0 - h / math.log(m)).clamp(min=0.0)
        if float(w.sum()) < eps:
            w = torch.ones_like(w)      # every condition flat: equal weights
        return ((w.unsqueeze(1) * (stack + eps).log()).sum(dim=0) / w.sum()).exp()
    raise ValueError(mode)


def control(arm, rebuilt, mismatched):
    """Report whether the arm matching the engine reproduces its predictions."""
    print(f"\ncontrol: {arm} rebuilt from the trace matches the engine on "
          f"{rebuilt - mismatched}/{rebuilt} rows")
    if mismatched:
        print(f"FAILED: {mismatched} rows differ, so the tables measure the "
              "capture, not the arms", file=sys.stderr)
    return {"arm": arm, "rebuilt": rebuilt, "agree": rebuilt - mismatched}


def cmd_norm(args):
    """Every normalisation mechanism on the same traced factors, product fixed."""
    tree = load_tree(args.tree)
    import torch
    import torch.nn.functional as F

    shipped = DEFAULT_RESOLVER.norm_mode
    modes = comma_list(args.modes)
    if unknown := set(modes) - set(NORM_MODES):
        sys.exit(f"unknown --modes {sorted(unknown)}")
    if shipped not in modes:
        sys.exit(f"include the engine's mode {shipped!r} in --modes; it is the control")
    if args.base not in modes:
        sys.exit(f"--base {args.base!r} must be one of --modes")

    stats, results, mismatched = {"skipped": 0}, {}, 0
    for rec in executed_rows(tree, load_rows(args.parses), stats):
        target = int(rec.row["target_id"])
        negated = [factor_is_negated(f, shipped, torch, F) for f in rec.factors]
        results[rec.key] = {}
        for mode in modes:
            score = rec.appearance
            for vec in recalibrate(rec.factors, negated, mode, torch, F):
                score = score * vec
            pred = predicted_id(rec.obj_ids, score)
            results[rec.key][mode] = pred == target
            mismatched += mode == shipped and pred != rec.engine_pred

    report = {"parses": args.parses, "modes": modes, "paired_baseline": args.base,
              "n_rows": len(results), "skipped": stats["skipped"],
              "control": control(shipped, len(results), mismatched),
              "overall": {}}
    accuracy_table(f"OVERALL (paired against {args.base})",
                   [("all rows", list(results))], report["overall"],
                   results=results, arms=modes, base=args.base)
    write_report(args, "normalization_mechanism", report)
    return 1 if mismatched else 0


def cmd_final(args):
    """Combination rules over the same traced factors, optionally per normalisation."""
    tree = load_tree(args.tree)
    import torch
    import torch.nn.functional as F

    dtype = getattr(torch, args.dtype)
    shipped = DEFAULT_RESOLVER.norm_mode
    modes, calibrations = comma_list(args.modes), comma_list(args.calibration)
    if not modes or modes[0] != "product" or not set(modes) <= set(FACTOR_MODES):
        sys.exit(f"--modes must start with product and use: {', '.join(FACTOR_MODES)}")
    if unknown := set(calibrations) - set(NORM_MODES):
        sys.exit(f"unknown --calibration {sorted(unknown)}")
    arms = arms_for(modes, calibrations)
    names = [name for name, _, _ in arms]
    control_arm = f"{shipped}+product" if calibrations else "product"
    if control_arm not in names:
        sys.exit(f"include the engine's mode {shipped!r} in --calibration; "
                 "it is the control")

    stats, results, n_factors, mismatched = {"skipped": 0}, {}, {}, 0
    for rec in executed_rows(tree, load_rows(args.parses), stats):
        target = int(rec.row["target_id"])
        if calibrations:
            negated = [factor_is_negated(f, shipped, torch, F) for f in rec.factors]
            factors = {cal: recalibrate(rec.factors, negated, cal, torch, F)
                       for cal in calibrations}
        else:
            factors = {None: [f["vec"] for f in rec.factors]}
        appearance = rec.appearance.to(dtype)
        results[rec.key], n_factors[rec.key] = {}, len(rec.factors)
        for name, cal, mode in arms:
            vecs = [v.to(dtype) for v in factors[cal]]
            score = appearance * aggregate(mode, vecs, args.eps) if vecs else appearance
            pred = predicted_id(rec.obj_ids, score)
            results[rec.key][name] = pred == target
            mismatched += name == control_arm and pred != rec.engine_pred

    report = {"parses": args.parses, "modes": modes, "calibrations": calibrations,
              "arms": names, "eps": args.eps, "dtype": args.dtype,
              "n_rows": len(results), "skipped": stats["skipped"],
              "control": control(control_arm, len(results), mismatched),
              "overall": {}, "by_predicate_count": {}}
    keys = list(results)

    def bucket(c):
        return str(c) if c < 4 else "4+"

    accuracy_table("OVERALL", [("all rows", keys)], report["overall"],
                   results=results, arms=names)
    accuracy_table("BY NUMBER OF CONDITIONS (C)",
                   [(b, [k for k in keys if bucket(n_factors[k]) == b])
                    for b in ("0", "1", "2", "3", "4+")],
                   report["by_predicate_count"], results=results, arms=names)
    write_report(args, "final_aggregation_calibration" if calibrations
                 else "final_predicate_aggregation", report)
    return 1 if mismatched else 0


def cmd_anchor(args):
    """Anchor binding modes on the C=1 pool, re-running the resolver per arm."""
    tree = load_tree(args.tree)
    modes, calibrations = comma_list(args.modes), comma_list(args.calibration)
    arms = arms_for(modes, calibrations)
    names = [name for name, _, _ in arms]

    stats, results, strata, drifted = {"skipped": 0}, defaultdict(dict), {}, {}
    batches = scene_batches(tree, engine.Scenes(), load_rows(args.parses), stats)
    for group, concepts, valid, obj_ids in batches:
        for row in group:
            key = row_key(row)
            for name, cal, mode in arms:
                config = DEFAULT_RESOLVER.updated(
                    combine=mode, norm_mode=cal or DEFAULT_RESOLVER.norm_mode)
                arm_tree = engine.tree(config)
                score, top, root_binds = traced_parse(arm_tree, row["json_obj"],
                                                      concepts, valid)
                pred = None if score is None else predicted_id(obj_ids, score)
                results[key][name] = pred == int(row["target_id"])
                if name == names[0]:
                    # The pool's selection rule, checked on this arm's trace
                    rec = SimpleNamespace(
                        row=row, factors=[f for f in top if f["kind"] == "rel"],
                        root_binds=root_binds)
                    relation, why = anchor_eligible(rec, tree.rel_num)
                    if relation is None:
                        drifted[key] = why
                    else:
                        strata[key] = anchor_stratum(relation)

    reasons = dict(Counter(drifted.values()))
    if drifted and not args.allow_predicate_drift:
        sys.exit(f"{len(drifted)} of {len(results)} rows no longer satisfy the "
                 f"pool's selection rule {reasons}; rebuild the pool with "
                 "build_stage2_subsets.py or pass --allow-predicate-drift")
    if drifted:
        print(f"\npredicate drift allowed: {len(drifted)} rows are in 'all rows' "
              f"but have no stratum {reasons}")
    report = {"parses": args.parses, "modes": modes,
              "calibration": calibrations or None, "arms": names,
              "n": len(results), "skipped": stats["skipped"],
              "predicate_drift": ({"n": len(drifted), "rows": sorted(drifted),
                                   "reasons": reasons} if drifted else None),
              "strata": {}}
    accuracy_table("BY ROOT-CONDITION STRATUM",
                   [("all rows", list(results))]
                   + [(s, [k for k, v in strata.items() if v == s])
                      for s in ANCHOR_STRATA],
                   report["strata"], results=results, arms=names)
    write_report(args, "anchor_calibration" if calibrations else "anchor_binding",
                 report)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="experiment", required=True)
    norm = commands.add_parser("norm", help="compare normalisation mechanisms")
    final = commands.add_parser("final", help="compare combination rules")
    anchor = commands.add_parser("anchor", help="compare anchor binding")
    for command, run, pool in ((norm, cmd_norm, CGE2_PARSES),
                               (final, cmd_final, CGE2_PARSES),
                               (anchor, cmd_anchor, ANCHOR_C1_PARSES)):
        command.set_defaults(run=run)
        command.add_argument("--parses", default=pool)
        command.add_argument("--tree", default="extended")
        command.add_argument("--out", default=STAGE2_OUT)
        command.add_argument("--confirm-on-test", action="store_true",
                             help="allow test parses, only to confirm a dev choice")

    norm.add_argument("--modes", default=",".join(NORM_MODES))
    norm.add_argument("--base", default="released",
                      help="mode every other mode is paired against")
    final.add_argument("--modes", default=",".join(FACTOR_MODES),
                       help="product first: it is the control")
    final.add_argument("--calibration", default="",
                       help="also cross with these normalisations; must include "
                            "the engine's own")
    final.add_argument("--eps", type=float, default=1e-9,
                       help="floor inside the log and reciprocal rules")
    final.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    anchor.add_argument("--modes", default="sum,max", help="binding modes")
    anchor.add_argument("--calibration", default="",
                        help="also cross with these normalisations")
    anchor.add_argument("--allow-predicate-drift", action="store_true",
                        help="score rows that no longer satisfy the pool's "
                             "selection rule, and list them in the report")

    args = parser.parse_args()
    guard_split(args.parses, args.confirm_on_test)
    if args.confirm_on_test and Path(args.out) == Path(STAGE2_OUT):
        args.out = DIAGNOSTICS_OUT      # keep test diagnostics out of the dev results
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
