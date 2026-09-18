#!/usr/bin/env python3
"""Write the frozen development draw in the engines' annotation schema.

Index the training CSV with each dev dataset_idx and verify scan/utterance
alignment. Preserve all columns needed by the inherited Nr3D loaders."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

from grounding.paths import CORPORA, RAW, REPO_ROOT

DEV_JSONL = CORPORA / "nr3d_dev.jsonl"
TRAIN_CSV = CORPORA / "nr3d_train.csv"
DEV_CSV = CORPORA / "nr3d_dev.csv"

# Where the trees look: both have data/ -> data/raw, and SCAN_FAMILY_BASE is
# data/referit3d, so one symlink here serves both.
REFER_DIR = RAW / "referit3d" / "annotations" / "refer"


def link_into_refer(target: Path, link: Path) -> None:
    """Point ``link`` at ``target``, the way nr3d.csv is already linked."""
    relative = Path("../../../..") / target.relative_to(RAW.parent)
    if link.is_symlink() or link.exists():
        if link.is_symlink() and Path(link).readlink() == relative:
            return
        link.unlink()
    link.symlink_to(relative)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dev", default=str(DEV_JSONL))
    ap.add_argument("--train", default=str(TRAIN_CSV))
    ap.add_argument("--out", default=str(DEV_CSV))
    args = ap.parse_args()

    dev_path = REPO_ROOT / args.dev
    train_path = REPO_ROOT / args.train
    rows = list(csv.DictReader(open(train_path, newline="")))
    dev = [json.loads(line) for line in open(dev_path) if line.strip()]

    for item in dev:
        idx = item["dataset_idx"]
        if idx >= len(rows):
            sys.exit(f"dev dataset_idx {idx} is past the end of {args.train} "
                     f"({len(rows)} rows); the two files are not the pair the "
                     "dev draw was made from")
        row = rows[idx]
        if (row["scan_id"] != item["scan_id"]
                or row["utterance"].strip() != item["utterance"]):
            sys.exit(f"dev row {idx} does not match {args.train}: "
                     f"{row['scan_id']} / {row['utterance'].strip()!r} vs "
                     f"{item['scan_id']} / {item['utterance']!r}. The dev draw "
                     "indexes that csv by position; rebuild the draw.")

    out = Path(args.out)
    out = out if out.is_absolute() else REPO_ROOT / out
    fieldnames = list(rows[0])
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for item in dev:
            writer.writerow(rows[item["dataset_idx"]])

    scenes = sorted({item["scan_id"] for item in dev})
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    meta = {
        "producer": "scripts/prepare/build_dev_csv.py",
        "purpose": "the split=dev annotation csv both engine trees read",
        "rows_from": {"path": args.train, "rows": len(rows)},
        "row_set_from": {"path": args.dev,
                         "sha256": hashlib.sha256(
                             dev_path.read_bytes()).hexdigest()},
        "n": len(dev),
        "scans": len(scenes),
        "sha256": sha,
    }
    Path(str(out) + ".meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    link = REFER_DIR / out.name
    if REFER_DIR.is_dir():
        link_into_refer(out, link)
        linked = f"{link.relative_to(REPO_ROOT)} -> {Path(link).readlink()}"
    else:
        linked = f"NOT LINKED: {REFER_DIR.relative_to(REPO_ROOT)} does not exist"

    print(f"wrote {out.relative_to(REPO_ROOT)}  "
          f"({len(dev)} rows over {len(scenes)} scans)")
    print(f"sha256  {sha}")
    print(f"link    {linked}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
