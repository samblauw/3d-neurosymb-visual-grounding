#!/usr/bin/env python3
"""Decompose the baseline-to-extended gap on one parses file.

The walk isolates extended encoders (A -> B), colour (B -> B2), and resolver
calibration/rank (B2 -> C). Arm A is read from the recorded baseline dump;
every other arm is scored on the same rows.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

from grounding.paths import REPO_ROOT, RESULTS_FINAL
from grounding.resolution.harness import (REGENERATED_FEATURE_FILES,
                                          run_in_tree, tree_dir)

from grounding.resolution.artifacts import (
    comparison_path, git_commit, parse_scores, relative_path as _rel, sha256,
)
from grounding.resolution.config import FROZEN_RESOLVER
from grounding.resolution.statistics import mcnemar


# key, label, resolver flags, and whether programs retain their colour factor.
ARMS = (
    ("B", "extended encoders, control arithmetic",
     ["--norm_mode", "softmax", "--no_use_rank"], False),
    ("B2", "+ colour wired in",
     ["--norm_mode", "softmax", "--no_use_rank"], True),
    ("C", f"+ {FROZEN_RESOLVER['norm_mode']} calibration and rank", [], True),
)
ARM_ORDER = ("A", "B", "B2", "C")
BASELINE_LABEL = "baseline (control encoders and arithmetic)"


def remove_colour(value: object) -> bool:
    """Remove every ``color`` key below value and report whether any existed."""
    if isinstance(value, dict):
        changed = "color" in value
        value.pop("color", None)
        for child in value.values():
            changed |= remove_colour(child)
        return changed
    if isinstance(value, list):
        changed = False
        for item in value:
            changed |= remove_colour(item)
        return changed
    return False


def strip_colour(source: Path, destination: Path) -> int:
    """Write a colour-free parse copy and return the number of changed rows."""
    touched = 0
    with source.open() as input_file, destination.open("w") as output_file:
        for line in input_file:
            if not line.strip():
                continue
            row = json.loads(line)
            touched += remove_colour(row.get("json_obj"))
            output_file.write(json.dumps(row) + "\n")
    return touched


def run_extended(parses: Path, flags: list[str], dump: Path) -> dict:
    features = REGENERATED_FEATURE_FILES["extended"]
    if not (tree_dir("extended") / features).exists():
        sys.exit(f"missing extended features: {features}")
    output = run_in_tree("extended", "src.eval.eval_nr3d", [
        "--features_path", features,
        "--top_k", "5",
        "--threshold", "0.9",
        "--label_type", "gt",
        "--totals", "official",
        "--parses", str(parses.resolve()),
        "--dump", str(dump.resolve()),
        *flags,
    ])
    return parse_scores(output)


def correctness(dump: Path) -> dict[int, bool]:
    rows = {}
    with dump.open() as input_file:
        for line in input_file:
            if line.strip():
                row = json.loads(line)
                rows[int(row["dataset_idx"])] = bool(row["correct"])
    return rows


def contrast(before: dict[int, bool], after: dict[int, bool], before_name: str,
             after_name: str) -> dict:
    keys = before.keys() & after.keys()
    gained = sum(after[key] and not before[key] for key in keys)
    lost = sum(before[key] and not after[key] for key in keys)
    correct_before = sum(before[key] for key in keys)
    correct_after = sum(after[key] for key in keys)
    return {
        "from": before_name,
        "to": after_name,
        "n": len(keys),
        "correct_from": correct_before,
        "correct_to": correct_after,
        "gained": gained,
        "lost": lost,
        "net": gained - lost,
        "delta_pp": (correct_after - correct_before) / len(keys) * 100 if keys else 0.0,
        "mcnemar_p": mcnemar(gained, lost),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parses", type=Path, required=True,
                        help="the parses both arms of the headline comparison used")
    parser.add_argument("--baseline-dump", type=Path,
                        help="per-row baseline dump (default: compare_baseline.py output)")
    parser.add_argument("--out", type=Path,
                        help="default: artifacts/results/final/decompose_<stem>.json")
    return parser.parse_args()


def validate_inputs(parses: Path, baseline_dump: Path) -> None:
    if not parses.exists():
        sys.exit(f"no such parses file: {parses}")
    if not baseline_dump.exists():
        sys.exit(
            f"no baseline dump at {_rel(baseline_dump)}.\n"
            "  Arm A is read from the recorded comparison rather than re-run.\n"
            "  Produce it with scripts/evaluation/compare_baseline.py first."
        )

    summary = comparison_path(RESULTS_FINAL, parses)
    if not summary.exists():
        return
    recorded = json.loads(summary.read_text())
    expected = recorded.get("provenance", {}).get("baseline", {}).get("parses", {}).get("sha256")
    actual = sha256(parses)
    if expected and expected != actual:
        sys.exit(
            f"{_rel(summary)} was computed on a different parses file "
            f"(sha256 {expected[:12]}... vs {actual[:12]}...)"
        )


def score_arms(parses: Path, baseline_dump: Path) -> tuple[dict, dict, dict, int]:
    arms = {"A": correctness(baseline_dump)}
    labels = {"A": BASELINE_LABEL}
    scores = {}

    with tempfile.TemporaryDirectory() as temporary_dir:
        temporary_dir = Path(temporary_dir)
        no_colour = temporary_dir / f"{parses.stem}_nocolour.jsonl"
        touched = strip_colour(parses, no_colour)
        print(f"colour stripped from {touched} rows for arm B")

        for key, label, flags, keep_colour in ARMS:
            print(f"\n--- {key}: {label} ---", flush=True)
            dump = temporary_dir / f"{key}.jsonl"
            scores[key] = run_extended(parses if keep_colour else no_colour, flags, dump)
            arms[key] = correctness(dump)
            labels[key] = label

    return arms, labels, scores, touched


def arm_summary(rows: dict[int, bool], label: str, official_total: int) -> dict:
    correct = sum(rows.values())
    return {
        "label": label,
        "correct": correct,
        "n_paired": len(rows),
        "accuracy_paired": correct / len(rows),
        "accuracy_official": correct / official_total,
    }


def print_report(payload: dict) -> None:
    print(f"\n{'arm':<4} {'label':<46} {'correct':>8} {'official':>9} {'paired':>8}")
    for key in ARM_ORDER:
        arm = payload["arms"][key]
        print(f"{key:<4} {arm['label']:<46} {arm['correct']:>8} "
              f"{arm['accuracy_official'] * 100:8.2f}% {arm['accuracy_paired'] * 100:7.2f}%")

    print(f"\n{'step':<52} {'net':>5} {'gained':>7} {'lost':>5} {'p':>10}")
    for step in payload["steps"] + [payload["headline"]]:
        name = f"{step['from'].split('(')[0].strip()[:22]} -> {step['to'].split('(')[0].strip()[:24]}"
        print(f"{name:<52} {step['net']:>+5} {step['gained']:>7} "
              f"{step['lost']:>5} {step['mcnemar_p']:>10.3g}")


def main() -> int:
    args = parse_args()
    parses = args.parses
    baseline_dump = (args.baseline_dump
                     or comparison_path(RESULTS_FINAL, parses, "baseline"))
    out = args.out or RESULTS_FINAL / f"decompose_{parses.stem}.json"
    out = out if out.is_absolute() else REPO_ROOT / out
    validate_inputs(parses, baseline_dump)

    arms, labels, scores, touched = score_arms(parses, baseline_dump)
    official_total = scores["C"]["overall"][1]
    steps = [
        contrast(arms[before], arms[after], labels[before], labels[after])
        for before, after in zip(ARM_ORDER, ARM_ORDER[1:])
    ]
    headline = contrast(arms["A"], arms["C"], labels["A"], labels["C"])

    payload = {
        "producer": "scripts/evaluation/decompose_representation.py",
        "git_commit": git_commit(),
        "command": " ".join(sys.argv),
        "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "parses": {"path": _rel(parses.resolve()), "sha256": sha256(parses)},
        "baseline_dump": _rel(baseline_dump),
        "colour_rows_stripped": touched,
        "official_total": official_total,
        "arms": {
            key: arm_summary(arms[key], labels[key], official_total)
            for key in ARM_ORDER
        },
        "official_totals": scores,
        "steps": steps,
        "headline": headline,
        "note": "B - A is the representation-only contrast; C - B is what the "
                "resolver change buys. The headline comparison reports C - A.",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print_report(payload)
    print(f"\nwrote {_rel(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
