#!/usr/bin/env python3
"""Build feature tensors for a parses file."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from grounding.parsing.parse import REPRESENTATIONS
from grounding.paths import resolve_under_root
from grounding.resolution.artifacts import git_commit, relative_path as _rel, sha256
from grounding.resolution.harness import (
    DATASET_SPLITS,
    dataset_split,
    feature_dir,
    feature_relpath,
    run_in_tree,
    tree_dir,
)


def split_of(parses: Path) -> str:
    """Which population the parses file covers, from its location.

    A feature tensor is only interpretable next to the rows it was built from,
    and both arms' ``output/`` links share ONE directory -- so the split is
    recorded rather than inferred from a filename later.
    """
    p = _rel(parses.resolve())
    for marker, name in (("data/parses/test/", "test"),
                         ("data/parses/dev/", "dev"),
                         ("data/parses/stage1/", "stage1"),
                         ("data/parses/reference_frame/", "reference_frame"),
                         ("data/probes/", "probe"),
                         ("data/corpora/", "corpora")):
        if p.startswith(marker):
            return name
    return "unknown"


def write_meta(rep: str, parses: Path, tensor: Path, split: str, label: str,
               scales: dict[str, float | None]) -> Path:
    """Record what decides this tensor, next to it as ``<tensor>.meta.json``.

    Same convention as the ``<out>.meta.json`` parse_split.py writes. Features
    must be rebuilt from the exact parses file being evaluated; without a record
    of WHICH file, a stale tensor and a fresh one are the same bytes on disk to
    everything downstream, and the two arms write into one shared directory.
    """
    meta = {
        "producer": "scripts/evaluation/build_features.py",
        "git_commit": git_commit(),
        "command": " ".join(sys.argv),
        "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "representation": rep,
        "split": split_of(parses),
        # The tree's --split: which annotation csv supplied the scenes. Coarser
        # than the population above -- every probe population is `train` here.
        "dataset_split": split,
        "parses": {"path": _rel(parses), "sha256": sha256(parses)},
        "tensor": {"path": _rel(tensor), "sha256": sha256(tensor),
                   "bytes": tensor.stat().st_size},
        "label": label,
        # The extended encoder scales that change the tensor's contents. None
        # means the tree's own default was used.
        "settings": dict(scales),
    }
    meta_path = Path(str(tensor) + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return meta_path


def build(
    rep: str,
    parses: Path,
    split: str,
    label: str,
    scales: dict[str, float | None],
) -> bool:
    print(f"--- {rep} ---", flush=True)

    args = [
        "--label", label,
        "--parses", str(parses),
        # Which annotation csv the tree's scene loader reads, and a matching
        # output directory: both arms' output/ is one shared directory, so a dev
        # tensor written at the default path would replace the test one.
        "--split", split,
        "--output", feature_dir(split),
    ]

    if rep == "extended":
        for name, value in scales.items():
            if value is not None:
                args += [f"--{name}", str(value)]

    if not run_in_tree(
        rep,
        "src.relation_encoders.compute_features",
        args,
        capture=False,
    ):
        return False

    relative = feature_relpath(rep, split)
    path = tree_dir(rep) / relative

    if not path.exists():
        print(f"  expected output not found: {relative}")
        return False

    meta_path = write_meta(rep, parses, path.resolve(), split, label, scales)
    print(f"  wrote {relative} ({path.stat().st_size / 1e6:.1f} MB)"
          f"  (+ {meta_path.name})")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--representation",
        dest="representation",
        action="append",
        choices=REPRESENTATIONS,
        help="baseline or extended; repeat to build both",
    )
    parser.add_argument(
        "--parses",
        type=Path,
        default=Path("data/corpora/nr3d.jsonl"),
    )
    parser.add_argument("--label", choices=["gt", "pred"], default="gt")
    parser.add_argument(
        "--split",
        choices=["auto", *DATASET_SPLITS],
        default="auto",
        help="which annotation csv the trees' scene loader reads. auto "
             "(default) picks the split whose scenes cover --parses, and "
             "refuses a file no split covers",
    )

    parser.add_argument("--gap_scale", type=float)
    parser.add_argument("--distance_scale", type=float)
    parser.add_argument("--color_scale", type=float)

    args = parser.parse_args()

    parses = resolve_under_root(args.parses).resolve()
    if not parses.exists():
        sys.exit(f"No such parses file: {parses}")

    split = dataset_split(parses, args.split)
    print(f"parses:  {_rel(parses)}\nsplit:   {split} "
          f"({split_of(parses)} rows)\noutput:  {feature_dir(split)}/")

    reps = args.representation or REPRESENTATIONS
    scales = {
        "gap_scale": args.gap_scale,
        "distance_scale": args.distance_scale,
        "color_scale": args.color_scale,
    }

    success = [
        build(rep, parses, split, args.label, scales)
        for rep in reps
    ]

    return 0 if all(success) else 1


if __name__ == "__main__":
    raise SystemExit(main())
