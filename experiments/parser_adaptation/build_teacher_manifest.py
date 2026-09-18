#!/usr/bin/env python3
"""Record the frozen R=8 teacher selection and verify its disjointness from dev.

Reuse generate_rollouts.load_rows: shuffle train rows at seed 42 and take an
8750-row prefix. The teacher pool was selected before shared dev; this command
checks that existing draw and does not redraw it after dev exclusion."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_rollouts import load_rows
from grounding.paths import CORPORA, REPO_ROOT, ROLLOUTS
from grounding.resolution.artifacts import sha256

TRAIN_CSV = CORPORA / "nr3d_train.csv"
DEV = CORPORA / "nr3d_dev.jsonl"
POOL = ROLLOUTS / "nr3d_rollouts_fft_v4.jsonl"
OUT = CORPORA / "nr3d_train_teacher_r8.manifest.jsonl"

#: The seed load_rows shuffles under, named here so the manifest records the generator's value
SEED = 42


def row_idx(path: Path) -> set[int]:
    out = set()
    with path.open() as f:
        for line in f:
            if line.strip():
                out.add(json.loads(line)["row_idx"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=TRAIN_CSV)
    ap.add_argument("--n", type=int, default=8750,
                    help="prompts the teacher ran on. A prefix of one shuffle, "
                         "so raising it extends the same sequence")
    ap.add_argument("--pool", type=Path, default=POOL,
                    help="the R=8 rollout file to check the selection against")
    ap.add_argument("--dev", type=Path, default=DEV,
                    help="the shared dev set to check disjointness against")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    with args.input.open(newline="") as f:
        eligible = [i for i, r in enumerate(csv.DictReader(f))
                    if (r.get("utterance") or "").strip()]
    selected = load_rows(args.input, args.n)
    if len(selected) != args.n:
        raise SystemExit(f"asked for {args.n} prompts, the draw yields "
                         f"{len(selected)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for r in selected:
            f.write(json.dumps({k: r[k] for k in
                                ("row_idx", "stimulus_id", "scan_id",
                                 "target_id", "instance_type", "utterance")},
                               ensure_ascii=False) + "\n")

    chosen = {r["row_idx"] for r in selected}
    stats = {
        "producer": "experiments/parser_adaptation/build_teacher_manifest.py",
        "input": str(args.input.relative_to(REPO_ROOT)),
        "rule": f"csv order, random.Random({SEED}).shuffle, first {args.n}",
        "seed": SEED,
        "n_selected": len(chosen),
        "eligible_pool": {
            "rule": "every nr3d_train.csv row with a non-empty utterance",
            "n": len(eligible),
        },
        "manifest": str(args.out.relative_to(REPO_ROOT)),
        "manifest_sha256": sha256(args.out),
        "scans": len({r["scan_id"] for r in selected}),
    }

    # Checked, not stated: the rollout file covers this selection and dev is disjoint from it
    if args.pool.exists():
        pool = row_idx(args.pool)
        stats["rollouts"] = {
            "path": str(args.pool.relative_to(REPO_ROOT)),
            "sha256": sha256(args.pool),
            "prompts": len(pool),
            "selected_not_in_rollouts": len(chosen - pool),
            "rollouts_not_in_selection": len(pool - chosen),
        }
    if args.dev.exists():
        dev = row_idx(args.dev)
        stats["shared_dev"] = {
            "path": str(args.dev.relative_to(REPO_ROOT)),
            "rows": len(dev),
            "overlap_with_selection": len(dev & chosen),
        }

    sidecar = Path(str(args.out) + ".stats.json")
    sidecar.write_text(json.dumps(stats, indent=2) + "\n")

    print(f"eligible pool {len(eligible)} rows -> selected {len(chosen)} "
          f"prompts over {stats['scans']} scans (seed {SEED})")
    if "rollouts" in stats:
        r = stats["rollouts"]
        print(f"  rollouts {r['prompts']} prompts | selected not rolled out "
              f"{r['selected_not_in_rollouts']} | rolled out not selected "
              f"{r['rollouts_not_in_selection']}")
    if "shared_dev" in stats:
        print(f"  shared dev {stats['shared_dev']['rows']} rows, overlap "
              f"{stats['shared_dev']['overlap_with_selection']}")
    print(f"wrote {args.out}\nwrote {sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
