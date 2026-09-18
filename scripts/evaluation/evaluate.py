#!/usr/bin/env python3
"""Score parses through one or both representations.

Features must have been built from the same parses file first.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from grounding.parsing.parse import normalise_representation
from grounding.paths import REPO_ROOT
from grounding.resolution.artifacts import (
    git_commit, parse_scores, provenance, relative_path as _rel,
)
from grounding.resolution.evaluate import TOTALS_MODES
from grounding.resolution.harness import (DATASET_SPLITS, FEATURE_FILES,
                                          REPRESENTATIONS, SPLIT_NAMES,
                                          anno_csv, dataset_split,
                                          feature_relpath, run_in_tree,
                                          tree_dir)

def run_representation(rep: str, features: str, extra: list[str]) -> str:
    args = ["--features_path", features, "--top_k", "5", "--threshold", "0.9",
            "--label_type", "gt", *extra]
    if rep == "baseline" and "--tie_break" not in extra:
        # The same single tie-break compare_baseline.py runs both arms under:
        # baseline's `legacy` IS the extended tree's unconditional `topk`, while
        # baseline's own default is a stable sort. Reporting a baseline number
        # here under a different tie-break from the one in the paired
        # comparison would put two different "baseline" numbers in the thesis.
        # An explicit --tie_break still wins, for reproducing the shipped
        # default deliberately.
        args += ["--tie_break", "legacy"]
    print(f"--- {rep} ---", flush=True)
    return run_in_tree(rep, "src.eval.eval_nr3d", args)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--representation", dest="representation",
                    action="append",
                    help="baseline | extended. "
                         "Repeatable; default is both")
    ap.add_argument("--parses", type=Path, default=None,
                    help="parses jsonl to execute")
    ap.add_argument("--totals", choices=list(TOTALS_MODES), default="official",
                    help="official: the fixed Nr3D test-set denominators "
                         "(7485 overall), comparable to the paper. parses: "
                         "counted off --parses, for subset files")
    ap.add_argument("--split", choices=["auto", *DATASET_SPLITS],
                    default="auto",
                    help="which annotation csv the tree's scene loader reads. "
                         "auto (default) picks the split whose scenes cover "
                         "--parses; without --parses it is test, the split the "
                         "released tensors cover")
    ap.add_argument("--features", default=None,
                    help="override the per-representation default "
                         "(requires exactly one --representation)")
    ap.add_argument("--out", type=Path, default=None,
                    help="write the split table plus provenance (parses and "
                         "tensor digests, tie-break, denominator) as JSON, "
                         "and a per-row dump beside it as <out>_<rep>.jsonl")
    args, extra = ap.parse_known_args()

    reps = ([normalise_representation(r) for r in args.representation]
            if args.representation else list(REPRESENTATIONS))
    if args.features and len(reps) != 1:
        sys.exit("--features needs exactly one --representation")
    if args.totals == "parses" and not args.parses:
        sys.exit("--totals parses requires a --parses file")
    if args.parses:
        if not args.parses.exists():
            sys.exit(f"No such parses file: {args.parses}")
        extra += ["--parses", str(args.parses.resolve())]
    extra += ["--totals", args.totals]

    # Which scenes the tree loads. Resolved from the parses file, because the
    # tree's own default is the held-out test csv: dev rows run against it are
    # skipped (baseline) or raise KeyError (extended) rather than failing loudly.
    split = (dataset_split(args.parses, args.split) if args.parses
             else ("test" if args.split == "auto" else args.split))
    if "--split" not in extra:
        extra += ["--split", split]
    print(f"split: {split} ({anno_csv(split).name})")

    # With --parses the features must be the ones build_features.py wrote FROM
    # that file. The released baseline tensor covers only the upstream rule
    # parses and lacks concepts a supplied parser emits (KeyError at score
    # time), so it is the default only for the rule-parse comparison.
    default_features = ({rep: feature_relpath(rep, split)
                         for rep in REPRESENTATIONS} if args.parses
                        else FEATURE_FILES)
    out = (args.out if args.out is None or args.out.is_absolute()
           else REPO_ROOT / args.out)
    results, prov = {}, {}
    for rep in reps:
        features = args.features or default_features[rep]
        if not (tree_dir(rep) / features).exists():
            print(f"--- {rep} ---\n  missing features: {features}")
            continue
        rep_extra = list(extra)
        if out is not None:
            out.parent.mkdir(parents=True, exist_ok=True)
            rep_extra += ["--dump", str(out.with_name(f"{out.stem}_{rep}.jsonl"))]
        results[rep] = parse_scores(run_representation(rep, features, rep_extra))
        if out is not None and results[rep]:
            # What the baseline was actually run with, not what its default is.
            if "--tie_break" in extra:
                tie = extra[extra.index("--tie_break") + 1]
            else:
                tie = "legacy"      # the aligned policy, both arms; see above
            prov[rep] = provenance(rep, args.parses, tree_dir(rep) / features,
                                   tie, args.totals)

    scored = {r: v for r, v in results.items() if v}
    if not scored:
        return 1

    if out is not None:
        out.write_text(json.dumps({
            "producer": "scripts/evaluation/evaluate.py",
            "git_commit": git_commit(),
            "command": " ".join(sys.argv),
            "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
            # Record both the population path and its digest under provenance.
            "parses": (prov[next(iter(scored))]["parses"]["path"]),
            "dataset_split": split,
            "totals": args.totals,
            "scores": {rep: {name: {"correct": c, "total": n}
                             for name, (c, n) in table.items()}
                       for rep, table in scored.items()},
            "provenance": prov,
        }, indent=2) + "\n")
        print(f"\nwrote {_rel(out)}  (+ {out.stem}_<rep>.jsonl)")

    width = max(max(len(r) for r in scored), 15) + 3
    print(f"\n{'split':<18}" + "".join(f"{r:>{width}}" for r in scored))
    for name in SPLIT_NAMES:
        row = f"{name:<18}"
        for r in scored:
            if name in scored[r]:
                c, n = scored[r][name]
                row += f"{c}/{n} {c / n:.3f}".rjust(width)
            else:
                row += "-".rjust(width)
        print(row)

    src = str(args.parses) if args.parses else "the trees' rule-based nr3d.jsonl"
    print(f"\nparses:      {src}")
    print(f"denominator: {args.totals}" +
          ("  (fixed Nr3D test-set counts)" if args.totals == "official"
           else "  (counted off parses file)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
