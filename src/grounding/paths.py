"""Repository paths resolved from the directory containing ``pyproject.toml``."""

from __future__ import annotations

from pathlib import Path

__all__ = [
    "REPO_ROOT",
    "SRC",
    "DATA",
    "RAW",
    "BENCHMARK",
    "CORPORA",
    "ROLLOUTS",
    "DISTILLED",
    "PARSES",
    "PARSES_STAGE1",
    "PARSES_REFFRAME",
    "PARSES_DEV",
    "PARSES_TEST",
    "PROBES",
    "PROBES_STAGE1",
    "PROBES_REFFRAME",
    "ARTIFACTS",
    "FEATURES",
    "MODELS",
    "RESULTS",
    "RESULTS_STAGE1",
    "RESULTS_REFFRAME",
    "RESULTS_STAGE2",
    "RESULTS_STAGE3",
    "RESULTS_FINAL",
    "RESULTS_BASELINE",
    "BASELINES",
    "BASELINE_TREE",
    "EXPERIMENTS",
    "SCRIPTS",
    "BOXES_DIR",
    "SPLITS_PATH",
    "SPLITS_DEV_PATH",
    "SCENE_EXTENTS",
    "resolve_under_root",
]


def _find_root(start: Path) -> Path:
    """Find the repository root by locating ``pyproject.toml``."""
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate

    # Fallback for installations where the repository root is unavailable.
    parents = start.parents
    return parents[2] if len(parents) > 2 else start


REPO_ROOT = _find_root(Path(__file__).resolve().parent)
SRC = REPO_ROOT / "src"


# Datasets and pipeline intermediates
DATA = REPO_ROOT / "data"
RAW = DATA / "raw"
BENCHMARK = DATA / "benchmark"
CORPORA = DATA / "corpora"
ROLLOUTS = DATA / "rollouts"
DISTILLED = DATA / "distilled"
PARSES = DATA / "parses"

# One directory per parse population, because a parse is only interpretable
# together with the row set it covers. `stage1` is the cue-filtered train rows
# behind the encoder probes, `reference_frame` the frame diagnostic's two
# passes, `dev` the Stage II resolver parent, `test` the frozen final split.
PARSES_STAGE1 = PARSES / "stage1"
PARSES_REFFRAME = PARSES / "reference_frame"
PARSES_DEV = PARSES / "dev"
PARSES_TEST = PARSES / "test"

# Frozen probe row sets: what an experiment is run ON, as opposed to the parse
# it was carved from. `data/selections/` was retired on 2026-09-03 -- the draw
# is a property of the seed now, not a worklist on disk.
PROBES = DATA / "probes"
PROBES_STAGE1 = PROBES / "stage1"
PROBES_REFFRAME = PROBES / "reference_frame"

BOXES_DIR = BENCHMARK / "boxes"
SPLITS_PATH = BENCHMARK / "splits.json"
# One strata table per split, keyed by that split's own row indices: the
# test table by nr3d_test.csv position, the dev table by nr3d_train.csv
# position. The keys overlap numerically, so the tables are never shared.
SPLITS_DEV_PATH = BENCHMARK / "splits_dev.json"
SCENE_EXTENTS = BENCHMARK / "scene_extents.json"


# Generated artifacts
ARTIFACTS = REPO_ROOT / "artifacts"
FEATURES = ARTIFACTS / "features"
MODELS = ARTIFACTS / "models"
RESULTS = ARTIFACTS / "results"

# Results grouped by contribution, split, and evaluation protocol.
RESULTS_STAGE1 = RESULTS / "representation"
RESULTS_REFFRAME = RESULTS / "reference_frame"
RESULTS_STAGE2 = RESULTS / "calibration" / "dev"
RESULTS_STAGE3 = RESULTS / "parser_adaptation" / "dev" / "top_tie"
RESULTS_FINAL = RESULTS / "final"
RESULTS_BASELINE = RESULTS / "baselines"


# Execution trees
BASELINES = REPO_ROOT / "baselines"
BASELINE_TREE = BASELINES / "lasp"


EXPERIMENTS = REPO_ROOT / "experiments"
SCRIPTS = REPO_ROOT / "scripts"


def resolve_under_root(path: str | Path) -> Path:
    """Resolve an absolute path directly or a relative path from the repo root."""
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path
