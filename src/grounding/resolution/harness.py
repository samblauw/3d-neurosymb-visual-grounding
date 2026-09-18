"""Run the baseline or extended execution tree as a subprocess."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

from grounding.parsing.parse import REPRESENTATIONS, representation_tree
from grounding.paths import RAW, FEATURES


__all__ = [
    "REPRESENTATIONS",
    "FEATURE_FILES",
    "REGENERATED_FEATURE_FILES",
    "SPLIT_NAMES",
    "DATASET_SPLITS",
    "AUTO_SPLITS",
    "tree_dir",
    "run_in_tree",
    "anno_csv",
    "dataset_split",
    "feature_dir",
    "feature_relpath",
]


FEATURE_FILES = {
    "baseline": str(FEATURES / "nr3d_features_per_scene_gt_label_dl.pth"),
    "extended": str(FEATURES / "nr3d_features_per_scene_gt_label_extended.pth"),
}

REGENERATED_FEATURE_FILES = {
    "baseline": str(FEATURES / "nr3d_features_per_scene_gt_label.pth"),
    "extended": str(FEATURES / "nr3d_features_per_scene_gt_label_extended.pth"),
}

SPLIT_NAMES = (
    "easy",
    "hard",
    "view_dependent",
    "view_independent",
    "overall",
)


#: The trees' `--split`: which annotation csv their scene loader reads. A tree
#: knows only the scenes in that csv, so running dev parses against `test` is
#: not a smaller run -- it is a run whose rows are all skipped (baseline) or a
#: KeyError (extended). Resolved per parses file by `dataset_split` below rather
#: than defaulted, because the default opens the HELD-OUT split.
DATASET_SPLITS = ("test", "dev", "train")

#: What `auto` will choose between. `train` is deliberately not here: its csv is
#: the whole corpus, so it covers every scene and would absorb any file the other
#: two do not -- loading every scan and answering a question nobody asked.
AUTO_SPLITS = ("test", "dev")

# Where both trees look: each has data/ -> data/raw, and their SCAN_FAMILY_BASE
# is data/referit3d, so this is the same file the subprocess will open.
_REFER_DIR = RAW / "referit3d" / "annotations" / "refer"
_ANNO_STEMS = {"test": "nr3d_test", "dev": "nr3d_dev", "train": "nr3d"}


def anno_csv(split: str) -> Path:
    """The annotation csv a tree reads for ``--split <split>``."""
    if split not in _ANNO_STEMS:
        raise ValueError(f"unknown split {split!r}")
    return _REFER_DIR / f"{_ANNO_STEMS[split]}.csv"


@lru_cache(maxsize=len(_ANNO_STEMS))
def split_scenes(split: str) -> frozenset[str]:
    """The scan ids a tree would know under ``--split <split>``."""
    path = anno_csv(split)
    if not path.exists():
        return frozenset()
    with path.open(newline="") as handle:
        return frozenset(row["scan_id"] for row in csv.DictReader(handle))


def parses_scenes(parses: Path) -> set[str]:
    """The scan ids a parses file asks for."""
    scenes = set()
    with Path(parses).open() as handle:
        for line in handle:
            if line.strip():
                scenes.add(json.loads(line)["scan_id"])
    return scenes


def dataset_split(parses: Path, requested: str = "auto") -> str:
    """Which `--split` covers this parses file's scenes.

    Measured, not inferred from the path: the split is whichever annotation csv
    actually contains every scene the file refers to. `test` is tried first so a
    test parse keeps reading the test csv and nothing else does; a file no single
    split covers is refused by name rather than run with rows silently dropped.
    """
    scenes = parses_scenes(parses)
    if requested != "auto":
        missing = scenes - split_scenes(requested)
        if missing:
            raise SystemExit(
                f"--split {requested} reads {anno_csv(requested).name}, which "
                f"has {len(missing)} of this file's {len(scenes)} scenes "
                f"missing (e.g. {sorted(missing)[:3]}). Those rows would be "
                "skipped, not evaluated.")
        return requested

    for split in AUTO_SPLITS:
        if scenes and scenes <= split_scenes(split):
            return split

    name = Path(parses).name
    if scenes and scenes <= split_scenes("train"):
        raise SystemExit(
            f"{name} covers {len(scenes)} TRAIN scenes -- the probe populations "
            "are not one of the evaluated splits. Pass --split train to load "
            "the whole corpus, or run the file through grounding.resolution."
            "engine, which builds its own scenes.")

    covered = {s: len(scenes & split_scenes(s)) for s in DATASET_SPLITS}
    raise SystemExit(
        f"no annotation csv covers the {len(scenes)} scenes in {name} "
        f"(covered: {covered}). One population per parses file: the trees load "
        "their scenes from one csv.")


def feature_dir(split: str) -> str:
    """Where a split's feature tensors live inside a tree's output/.

    Both trees' output/ is one shared directory, and a tensor's filename says
    only which ARM built it -- so a dev tensor written beside the test one would
    silently replace it. Per-split subdirectories keep both on disk.
    """
    if split not in _ANNO_STEMS:
        raise ValueError(f"unknown split {split!r}")
    return str(FEATURES if split == "test" else FEATURES / split)


def feature_relpath(representation: str, split: str = "test") -> str:
    """The regenerated feature tensor for one arm on one split."""
    name = Path(REGENERATED_FEATURE_FILES[representation]).name
    return f"{feature_dir(split)}/{name}"


def tree_dir(representation: str) -> Path:
    """Return the execution directory for a representation."""
    return representation_tree(representation)


def run_in_tree(
    representation: str,
    module: str,
    args: list[str],
    capture: bool = True,
) -> str:
    """Run a Python module inside a representation tree."""

    tree = tree_dir(representation)

    if representation == "extended":
        module = {
            "src.eval.eval_nr3d": "grounding.resolution.cli",
            "src.relation_encoders.compute_features": "grounding.representation.pipeline",
        }.get(module, module)
    cmd = [sys.executable, "-m", module, *args]

    process = subprocess.run(
        cmd,
        cwd=tree,
        capture_output=capture,
        text=True,
    )

    if process.returncode != 0:
        if capture:
            output = process.stderr or process.stdout
            for line in output.strip().splitlines()[-6:]:
                print(f"    {line}")

        print("  FAILED")
        return ""

    return process.stdout if capture else "ok"
