#!/usr/bin/env python3
"""Compare baseline and extended results over one parses file.

Both arms use features built from that file and the same legacy/top-k tie-break.
Per-row dumps and a paired summary are written beside the score table.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from grounding.paths import REPO_ROOT, RESULTS, RESULTS_FINAL
from grounding.resolution.config import FROZEN_RESOLVER, NORM_MODES
from grounding.resolution.evaluate import TOTALS_MODES
from grounding.resolution.harness import (DATASET_SPLITS, REPRESENTATIONS,
                                          SPLIT_NAMES, dataset_split,
                                          anno_csv, feature_relpath,
                                          run_in_tree, tree_dir)

from grounding.resolution.artifacts import (
    comparison_path, git_commit, parse_scores, provenance, relative_path as _rel,
)
from grounding.resolution.statistics import mcnemar


TIE_BREAK = "legacy"
TIE_BREAK_NOTE = "legacy (topk) in both arms"

#: The baseline tree has no normalisation flag: it is the released LaSP
#: arithmetic and hardcodes a softmax. Saying so in the output is the point --
#: the two arms are not on one calibration and never were.
BASELINE_NORM_MODE = "softmax (hardcoded; the baseline tree has no --norm_mode)"


def run_representation(rep: str, parses: Path, split: str, totals: str,
                       dump: Path, norm_mode: str) -> dict:
    features = feature_relpath(rep, split)
    if not (tree_dir(rep) / features).exists():
        print(f"--- {rep} ---\n  missing features: {features}\n"
              f"  build them first: scripts/evaluation/build_features.py "
              f"--parses {parses}")
        return {}

    print(f"--- {rep} ---", flush=True)
    args = [
        "--features_path", features,
        "--top_k", "5",
        "--threshold", "0.9",
        "--label_type", "gt",
        # Passed, never defaulted: the trees default to the HELD-OUT test csv,
        # and a dev file run against it has every row skipped.
        "--split", split,
        "--totals", totals,
        "--parses", str(parses.resolve()),
        "--dump", str(dump.resolve()),
    ]
    if rep == "baseline":
        args += ["--tie_break", TIE_BREAK]
    else:
        # Passed, not inherited: the extended tree's default has moved before
        # and the summary has to record what actually ran.
        args += ["--norm_mode", norm_mode]
    return parse_scores(run_in_tree(rep, "src.eval.eval_nr3d", args))


def load_dump(path: Path) -> dict[int, bool]:
    if not path.exists():
        return {}
    rows = {}
    with path.open() as input_file:
        for line in input_file:
            if line.strip():
                row = json.loads(line)
                rows[int(row["dataset_idx"])] = bool(row["correct"])
    return rows


def output_dir(parses: Path, requested: Path | None, test_diagnostic: bool) -> Path:
    """Choose the final or diagnostic output directory for a parses file."""
    on_test_split = "parses/test/" in parses.as_posix()
    if requested is not None:
        if on_test_split and not test_diagnostic:
            sys.exit("a test-split parse outside final/ must be declared with "
                     "--test-diagnostic; the final number lives in final/")
        return requested if requested.is_absolute() else REPO_ROOT / requested
    if test_diagnostic:
        sys.exit("--test-diagnostic needs --out-dir: final/ holds the final "
                 "number, and a diagnostic beside it would be read as one")
    return RESULTS_FINAL if on_test_split else RESULTS


def score_arms(parses: Path, split: str, totals: str, out_dir: Path,
               norm_mode: str) -> tuple[dict, dict]:
    tables, dumps = {}, {}
    for rep in REPRESENTATIONS:
        dump = comparison_path(out_dir, parses, rep)
        tables[rep] = run_representation(rep, parses, split, totals, dump,
                                         norm_mode)
        dumps[rep] = load_dump(dump)
    return tables, dumps


def print_score_table(scored: dict[str, dict[str, tuple[int, int]]]) -> None:
    width = 18
    print(f"\n{'split':<18}" + "".join(f"{rep:>{width}}" for rep in scored))
    for name in SPLIT_NAMES:
        cells = []
        for table in scored.values():
            correct, total = table.get(name, (0, 0))
            cells.append((f"{correct}/{total} {correct / total:.3f}" if total else "-").rjust(width))
        print(f"{name:<18}" + "".join(cells))


def paired_stats(baseline: dict[int, bool], extended: dict[int, bool]) -> dict | None:
    shared = set(baseline) & set(extended)
    if not shared:
        return None

    both = sum(baseline[key] and extended[key] for key in shared)
    extended_only = sum(extended[key] and not baseline[key] for key in shared)
    baseline_only = sum(baseline[key] and not extended[key] for key in shared)
    return {
        "n": len(shared),
        "both": both,
        "extended_only": extended_only,
        "baseline_only": baseline_only,
        "neither": len(shared) - both - extended_only - baseline_only,
        "net_to_extended": extended_only - baseline_only,
        "mcnemar_p": mcnemar(extended_only, baseline_only),
        "unpaired": {
            "baseline_only": len(set(baseline) - set(extended)),
            "extended_only": len(set(extended) - set(baseline)),
        },
    }


def print_paired(paired: dict | None, parses: Path, out_dir: Path) -> None:
    if paired is None:
        print("\nno rows scored by both arms -- were features built for both?")
        return

    print(f"\npaired over {paired['n']} rows scored by both arms")
    print(f"  both correct        {paired['both']}")
    print(f"  extended only       {paired['extended_only']}")
    print(f"  baseline only       {paired['baseline_only']}")
    print(f"  neither             {paired['neither']}")
    print(f"  net to extended     {paired['net_to_extended']:+d}")
    print(f"  exact McNemar p     {paired['mcnemar_p']:.3g}"
          f"   ({paired['extended_only']} wins vs {paired['baseline_only']} losses)")
    unpaired = paired["unpaired"]
    if unpaired["baseline_only"] or unpaired["extended_only"]:
        print(f"  UNPAIRED            {unpaired['baseline_only']} baseline-only, "
              f"{unpaired['extended_only']} extended-only row(s) -- excluded from the test")
    print(f"\ndumps: {comparison_path(out_dir, parses, 'baseline')}, "
          f"{comparison_path(out_dir, parses, 'extended')}")


def write_summary(summary: Path, parses: Path, split: str, totals: str,
                  test_diagnostic: bool, scored: dict, paired: dict | None,
                  norm_mode: str) -> None:
    payload = {
        "producer": "scripts/evaluation/compare_baseline.py",
        "git_commit": git_commit(),
        "command": " ".join(sys.argv),
        "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "parses": _rel(parses.resolve()),
        "dataset_split": split,
        "totals": totals,
        "tie_break": TIE_BREAK_NOTE,
        "calibration": {"extended": norm_mode, "baseline": BASELINE_NORM_MODE,
                        "passed_explicitly": True},
        "test_diagnostic": bool(test_diagnostic),
        "scores": {
            rep: {
                name: {"correct": correct, "total": total}
                for name, (correct, total) in table.items()
            }
            for rep, table in scored.items()
        },
        "paired": paired,
        "provenance": {
            rep: provenance(rep, parses, tree_dir(rep) / feature_relpath(rep, split),
                            TIE_BREAK, totals)
            for rep in scored
        },
    }
    summary.write_text(json.dumps(payload, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parses", type=Path, default="data/corpora/nr3d.jsonl")
    parser.add_argument("--totals", choices=list(TOTALS_MODES), default="official")
    parser.add_argument("--split", choices=["auto", *DATASET_SPLITS],
                        default="auto",
                        help="which annotation csv the trees' scene loader "
                             "reads. auto (default) picks the split whose "
                             "scenes cover --parses; the trees' own default is "
                             "the held-out test csv, so this is always passed")
    parser.add_argument("--out-dir", type=Path,
                        help="where to write dumps and the summary")
    parser.add_argument("--test-diagnostic", action="store_true",
                        help="label a test-split result outside final/")
    parser.add_argument("--norm-mode", dest="norm_mode",
                        choices=NORM_MODES,
                        default=FROZEN_RESOLVER["norm_mode"],
                        help="per-relation-node normalisation for the EXTENDED "
                             "arm, passed to its evaluator and recorded in the "
                             "summary. The baseline arm has no such knob")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    parses = Path(args.parses)
    if not parses.exists():
        sys.exit(f"No such parses file: {parses}")
    split = dataset_split(parses, args.split)
    out_dir = output_dir(parses, args.out_dir, args.test_diagnostic)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"split:       {split} ({anno_csv(split).name})")
    print(f"calibration: extended={args.norm_mode}, baseline={BASELINE_NORM_MODE}")
    tables, dumps = score_arms(parses, split, args.totals, out_dir,
                               args.norm_mode)
    scored = {rep: table for rep, table in tables.items() if table}
    if not scored:
        return 1

    print_score_table(scored)
    paired = paired_stats(dumps.get("baseline", {}), dumps.get("extended", {}))
    print_paired(paired, parses, out_dir)

    summary = comparison_path(out_dir, parses)
    write_summary(summary, parses, split, args.totals, args.test_diagnostic,
                  scored, paired, args.norm_mode)
    print(f"\nparses:      {parses}")
    print(f"split:       {split}")
    print(f"denominator: {args.totals}")
    print("tie-break:   legacy (topk) in BOTH arms")
    print(f"summary:     {_rel(summary)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
