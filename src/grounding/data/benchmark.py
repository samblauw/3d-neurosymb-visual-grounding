"""Shared point-derived boxes, scene extents and split tables.

Boxes use centre=(max+min)/2 and size=max-min, matching the inherited loader.
Scene extents cover the complete point cloud, including background instances.
The cache is produced by scripts/prepare/build_benchmark.py."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from grounding.paths import BOXES_DIR, RAW, SCENE_EXTENTS


# LaSP excludes these from the candidate set, but they are retained for
# geometric predicates that depend on room surfaces.
BACKGROUND_LABELS = {"wall", "floor", "ceiling"}

# Raw ReferIt3D scan data used to construct the benchmark geometry.
SOURCE_DIR = RAW / "referit3d" / "scan_data"


def scan_geometry(
    scan_id: str,
    source_dir: Path = SOURCE_DIR,
) -> tuple[list[dict], dict]:
    """Build object bounding boxes and the scene extent for one scan."""

    locs = np.load(
        source_dir / "instance_id_to_loc" / f"{scan_id}.npy"
    )
    names = json.loads(
        (
            source_dir
            / "instance_id_to_name"
            / f"{scan_id}.json"
        ).read_text()
    )

    if len(locs) != len(names):
        raise ValueError(
            f"{scan_id}: {len(locs)} boxes vs {len(names)} labels"
        )

    pcds, _, _, instance_labels = torch.load(
        source_dir
        / "pcd_with_global_alignment"
        / f"{scan_id}.pth",
        weights_only=False,
    )

    xyz = np.asarray(pcds)[:, :3]
    instance_labels = np.asarray(instance_labels)

    objects = []

    for i, label in enumerate(names):
        pts = xyz[instance_labels == i]

        if len(pts) == 0:
            raise ValueError(
                f"{scan_id}: instance {i} ({label}) has no points"
            )

        lo = pts.min(axis=0)
        hi = pts.max(axis=0)

        centre = ((hi + lo) / 2).tolist()
        size = (hi - lo).tolist()
        min_pt = lo.tolist()
        max_pt = hi.tolist()

        objects.append(
            {
                "instance_id": i,
                "object_id": i,
                "label": label,
                "centre": centre,
                "size": size,
                "bbox_center": centre,
                "bbox_size": size,
                "bbox_min": min_pt,
                "bbox_max": max_pt,
                "is_background": label in BACKGROUND_LABELS,
            }
        )

    extent = {
        "min": xyz.min(axis=0).tolist(),
        "max": xyz.max(axis=0).tolist(),
    }

    return objects, extent


@lru_cache(maxsize=1)
def scene_extents(path: Path = SCENE_EXTENTS) -> dict:
    """Load the cached per-scene minimum and maximum coordinates."""

    path = Path(path)

    if not path.exists():
        raise SystemExit(
            f"{path} missing. Build it:\n"
            "  python scripts/prepare/build_benchmark.py"
        )

    return json.loads(path.read_text())


def load_boxes(
    scan_id: str,
    boxes_dir: Path = BOXES_DIR,
    include_background: bool = False,
) -> list[dict]:
    """Load the cached bounding boxes for one scan."""

    path = Path(boxes_dir) / f"{scan_id}_bboxes.json"
    objects = json.loads(path.read_text())

    if include_background:
        return objects

    return [
        obj
        for obj in objects
        if not obj["is_background"]
    ]


def available_scans(
    source_dir: Path = SOURCE_DIR,
) -> list[str]:
    """Return the scan IDs available in the raw ReferIt3D data."""

    loc_dir = source_dir / "instance_id_to_loc"
    return sorted(path.stem for path in loc_dir.glob("*.npy"))
