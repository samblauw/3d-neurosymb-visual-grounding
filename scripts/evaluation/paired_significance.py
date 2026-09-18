#!/usr/bin/env python3
"""Paired sign tests between stored per-expression results.

Each comparison pairs two committed result files scored on one split. Under the
official denominator an expression absent from a dump is a miss, exactly as in
the accuracy reported for that run; the test over expressions present in both
dumps is written beside it. Only result files are read: no tree, parse file or
split csv.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from grounding.paths import (RESULTS_BASELINE, RESULTS_FINAL,
                             RESULTS_STAGE3, resolve_under_root)
from grounding.resolution.artifacts import git_commit, relative_path, sha256
from grounding.resolution.statistics import mcnemar


ABSENT = "<not recorded>"

FULL_SYSTEM = {
    "label": "full system",
    "kind": "compare",
    "rows": RESULTS_FINAL / "test_predictions_extended.jsonl",
    "summary": RESULTS_FINAL / "test_comparison.json",
    "representation": "extended",
}
EXPERIMENTAL_BASELINE = {
    "label": "experimental baseline",
    "kind": "compare",
    "rows": RESULTS_BASELINE / "teacher_parser/test_predictions.jsonl",
    "summary": RESULTS_BASELINE / "teacher_parser/test_summary.json",
    "representation": "baseline",
}
REPRODUCED_LASP = {
    "label": "reproduced LaSP",
    "kind": "compare",
    "rows": RESULTS_BASELINE / "released_lasp/test_predictions.jsonl",
    "summary": RESULTS_BASELINE / "released_lasp/test_summary.json",
    "representation": "baseline",
}
# The adapter whose test parses the full-system row above was scored on.
FINAL_ADAPTER = {
    "label": "adapted parser (random_size_matched seed 2)",
    "kind": "checkpoint",
    "rows": RESULTS_STAGE3 / "random_size_matched_seed_2.json",
}
ZERO_SHOT = {
    "label": "zero-shot parser",
    "kind": "checkpoint",
    "rows": RESULTS_STAGE3 / "qwen3_1p7b_zero_shot.json",
}

#: name -> (split, arm, reference). Wins are expressions only the arm gets right.
COMPARISONS = {
    "combined_full_vs_experimental_baseline": ("test", FULL_SYSTEM, EXPERIMENTAL_BASELINE),
    "reproduced_lasp_vs_experimental_baseline": ("test", REPRODUCED_LASP, EXPERIMENTAL_BASELINE),
    "parser_adaptation_vs_zero_shot": ("dev", FINAL_ADAPTER, ZERO_SHOT),
}


def load_arm(spec: dict) -> tuple[dict, dict[int, tuple[bool, str | None]]]:
    """Read one arm's rows and the totals its own summary records."""
    if spec["kind"] == "compare":
        summary = json.loads(spec["summary"].read_text())
        rep = spec["representation"]
        overall = summary["scores"][rep]["overall"]
        rows = {}
        with spec["rows"].open() as source:
            for line in source:
                if line.strip():
                    row = json.loads(line)
                    rows[int(row["dataset_idx"])] = (bool(row["correct"]), row.get("scan_id"))
        settings = {key: summary.get(key, ABSENT)
                    for key in ("producer", "git_commit", "parses", "totals", "tie_break")}
        settings["calibration"] = (summary.get("calibration") or {}).get(rep, ABSENT)
        files = [spec["rows"], spec["summary"]]
        split, total, correct = summary.get("dataset_split"), overall["total"], overall["correct"]
    else:
        data = json.loads(spec["rows"].read_text())
        rows = {int(row["dataset_idx"]): (bool(row["model"]), row.get("scan_id"))
                for row in data["rows"]}
        settings = {key: data.get(key, ABSENT)
                    for key in ("model", "adapter", "prompt", "resolver", "generations_from")}
        files = [spec["rows"]]
        split, total, correct = data.get("split"), data["summary"]["n"], data["summary"]["model_top1"]

    scored = sum(ok for ok, _ in rows.values())
    if scored != correct or len(rows) > total:
        sys.exit(f"{relative_path(spec['rows'])}: {scored} correct over {len(rows)} rows, "
                 f"but its summary records {correct}/{total}")
    record = {
        "label": spec["label"],
        "split_recorded": split,
        "correct": correct,
        "total": total,
        "accuracy": correct / total,
        "rows_in_dump": len(rows),
        "files": [{"path": relative_path(path), "sha256": sha256(path)} for path in files],
        "settings": settings,
    }
    return record, rows


def discordant(arm: dict[int, bool], reference: dict[int, bool], keys) -> tuple[int, int]:
    wins = sum(arm.get(key, False) and not reference.get(key, False) for key in keys)
    losses = sum(reference.get(key, False) and not arm.get(key, False) for key in keys)
    return wins, losses


def compare(split: str, arm_spec: dict, reference_spec: dict) -> dict:
    (arm, arm_rows), (reference, reference_rows) = load_arm(arm_spec), load_arm(reference_spec)
    for record in (arm, reference):
        if record["split_recorded"] not in (None, split):
            sys.exit(f"{record['label']} records split {record['split_recorded']!r}, "
                     f"not {split!r}")
    if arm["total"] != reference["total"]:
        sys.exit(f"denominators differ: {arm['label']} {arm['total']}, "
                 f"{reference['label']} {reference['total']}")

    shared = arm_rows.keys() & reference_rows.keys()
    moved = sorted(key for key in shared
                   if None not in (arm_rows[key][1], reference_rows[key][1])
                   and arm_rows[key][1] != reference_rows[key][1])
    if moved:
        sys.exit(f"dataset_idx names different scans in the two dumps (first: {moved[:5]}); "
                 "the rows are not the same expressions")
    scored = arm_rows.keys() | reference_rows.keys()
    if len(scored) > arm["total"]:
        sys.exit(f"the dumps name {len(scored)} distinct rows, more than the "
                 f"denominator {arm['total']}")

    arm_ok = {key: ok for key, (ok, _) in arm_rows.items()}
    reference_ok = {key: ok for key, (ok, _) in reference_rows.items()}
    wins, losses = discordant(arm_ok, reference_ok, scored)
    both = sum(arm_ok.get(key, False) and reference_ok.get(key, False) for key in scored)
    shared_wins, shared_losses = discordant(arm_ok, reference_ok, shared)
    report = {
        "split": split,
        "arm": arm,
        "reference": reference,
        "official": {
            "n": arm["total"],
            "wins": wins,
            "losses": losses,
            "both": both,
            "neither": arm["total"] - wins - losses - both,
            "delta_pp": 100 * (arm["correct"] - reference["correct"]) / arm["total"],
            "mcnemar_p": mcnemar(wins, losses),
        },
        "shared_rows": {
            "n": len(shared),
            "wins": shared_wins,
            "losses": shared_losses,
            "mcnemar_p": mcnemar(shared_wins, shared_losses),
        },
    }

    arm_resolver = arm["settings"].get("resolver")
    reference_resolver = reference["settings"].get("resolver")
    if isinstance(arm_resolver, dict) and isinstance(reference_resolver, dict):
        report["resolver_differences"] = {
            key: {"arm": arm_resolver.get(key, ABSENT),
                  "reference": reference_resolver.get(key, ABSENT)}
            for key in sorted(arm_resolver.keys() | reference_resolver.keys())
            if arm_resolver.get(key, ABSENT) != reference_resolver.get(key, ABSENT)
        }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", nargs="*", choices=list(COMPARISONS),
                        default=list(COMPARISONS),
                        help="comparisons to run (default: all)")
    parser.add_argument("--out", default=str(RESULTS_FINAL / "paired_significance.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reports = {name: compare(*COMPARISONS[name]) for name in args.comparison}

    for name, report in reports.items():
        official, shared = report["official"], report["shared_rows"]
        print(f"\n=== {name} ({report['split']}) ===")
        for role in ("arm", "reference"):
            record = report[role]
            print(f"  {role:<10} {record['label']:<45} "
                  f"{record['correct']}/{record['total']} {100 * record['accuracy']:.2f}%")
        print(f"  official   n={official['n']}  W={official['wins']}  L={official['losses']}"
              f"  p={official['mcnemar_p']:.3g}")
        print(f"  shared     n={shared['n']}  W={shared['wins']}  L={shared['losses']}"
              f"  p={shared['mcnemar_p']:.3g}")
        if report.get("resolver_differences"):
            print(f"  RESOLVER RECORDS DIFFER: {json.dumps(report['resolver_differences'])}")

    payload = {
        "producer": "scripts/evaluation/paired_significance.py",
        "git_commit": git_commit(),
        "command": " ".join(sys.argv),
        "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "test": "two-sided exact sign test on discordant pairs "
                "(grounding.resolution.statistics.mcnemar)",
        "wins": "expressions the arm gets right and the reference gets wrong",
        "official": "every expression in the denominator; a row absent from a dump is a miss",
        "shared_rows": "expressions present in both dumps only",
        "comparisons": reports,
    }
    out = resolve_under_root(Path(args.out))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {relative_path(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
