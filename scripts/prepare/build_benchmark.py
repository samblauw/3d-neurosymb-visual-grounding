#!/usr/bin/env python3
"""Build benchmark geometry and evaluation splits.

IN   data/raw/referit3d/scan_data/
     data/raw/nr3d_test.csv
     data/corpora/nr3d_dev.jsonl + data/corpora/nr3d_train.csv   (dev strata)
OUT  data/benchmark/boxes/<scan_id>_bboxes.json
     data/benchmark/scene_extents.json
     data/benchmark/splits.json        keyed by nr3d_test.csv position
     data/benchmark/splits_dev.json    keyed by nr3d_train.csv position

    python scripts/prepare/build_benchmark.py
    python scripts/prepare/build_benchmark.py --splits-only

The baseline and extended pipelines use the same generated benchmark files.
Rebuild these files only when the source geometry or split definitions change.
"""

from __future__ import annotations

import argparse
import json

from grounding.data import benchmark as boxes_mod
from grounding.paths import SCENE_EXTENTS
from grounding.data import splits as splits_mod


def verify() -> int:
    """Verify cached bounding boxes against boxes returned by load_pc."""
    import sys

    import numpy as np

    from grounding.paths import BASELINE_TREE
    from grounding.resolution import engine

    engine.bind("baseline")
    sys.path.insert(0, str(BASELINE_TREE))
    from grounding._vendor.lasp.eval_helper import load_pc

    scans = boxes_mod.available_scans()
    bad, checked = [], 0
    for scan_id in scans:
        try:
            with engine.inside_tree():
                _, ids, locs = load_pc(scan_id)[:3]
        except Exception as exc:
            bad.append(f"{scan_id}: load_pc failed: {exc!r}")
            continue
        fg = boxes_mod.load_boxes(scan_id)
        if [int(i) for i in ids] != [o["object_id"] for o in fg]:
            bad.append(f"{scan_id}: object ids differ")
            continue
        cached = np.array([o["centre"] + o["size"] for o in fg], dtype=float)
        live = np.asarray(locs, dtype=float)
        if not np.array_equal(cached, live):
            bad.append(f"{scan_id}: max |delta| {np.abs(cached - live).max():.3e}")
            continue
        checked += 1
    print(f"verified {checked}/{len(scans)} scans exactly equal to load_pc")
    for line in bad[:20]:
        print(f"  MISMATCH {line}")
    if bad:
        print(f"  ... {len(bad)} scan(s) failed")
    return 1 if bad else 0


def report(name: str, table: dict, path) -> None:
    hard = sum(v["is_hard"] for v in table.values())
    vd = sum(v["is_view_dependent"] for v in table.values())
    missing = sum(not v["has_boxes"] for v in table.values())

    print(f"{name:<9}{len(table)} rows -> {path}")
    print(f"         hard {hard}  easy {len(table)-hard}  "
          f"view_dependent {vd}  view_independent {len(table)-vd}")
    if missing:
        print(f"         {missing} rows have no box file for their scan")


def write_splits() -> int:
    """Write one strata table per evaluated split, from the boxes on disk."""
    table = splits_mod.build()
    splits_mod.SPLITS_PATH.write_text(json.dumps(table, indent=0))
    report("splits", table, splits_mod.SPLITS_PATH)

    # The dev draw has its own table because row indices are per-csv positions:
    # a dev index read out of the test table returns another utterance's flags.
    # Absent before the dev set is frozen, which is not an error.
    if splits_mod.DEV_PATH.exists():
        dev = splits_mod.build_dev()
        splits_mod.SPLITS_DEV_PATH.write_text(json.dumps(dev, indent=0))
        report("dev", dev, splits_mod.SPLITS_DEV_PATH)
    else:
        print(f"dev      no {splits_mod.DEV_PATH.name}; skipped")

    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", action="store_true",
                    help="do not build; assert every cached box equals a live "
                         "load_pc call, over every scan")
    ap.add_argument("--splits-only", action="store_true",
                    help="rebuild only the split tables, from the boxes already "
                         "on disk; geometry is untouched")
    args = ap.parse_args()
    if args.verify:
        return verify()

    if args.splits_only:
        return write_splits()

    boxes_mod.BOXES_DIR.mkdir(parents=True, exist_ok=True)

    scans = boxes_mod.available_scans()
    n_obj = n_bg = 0

    extents = {}
    for scan_id in scans:
        objects, extents[scan_id] = boxes_mod.scan_geometry(scan_id)
        n_obj += len(objects)
        n_bg += sum(o["is_background"] for o in objects)
        (boxes_mod.BOXES_DIR / f"{scan_id}_bboxes.json").write_text(json.dumps(objects))

    print(f"boxes    {len(scans)} scans, {n_obj} instances "
          f"({n_bg} background flagged, {n_obj - n_bg} candidates)")

    # Scene extents are derived from the same geometry pass as the boxes.
    SCENE_EXTENTS.write_text(json.dumps(extents))
    print(f"extents  {len(extents)} scans -> {SCENE_EXTENTS}")

    return write_splits()


if __name__ == "__main__":
    raise SystemExit(main())
