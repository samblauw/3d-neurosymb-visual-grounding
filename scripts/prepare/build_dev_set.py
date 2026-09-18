#!/usr/bin/env python3
"""Draw shared development rows from eligible Nr3D training utterances.

Exclude teacher-pool IDs and SFT train/validation IDs. Also exclude utterance
strings used in SFT; teacher-only strings are counted but not excluded.
Shuffle sorted survivors at the declared seed and take a prefix, preserving
membership when the requested size grows. Scans may overlap training: this is
utterance-held-out development; the official test split is scene-held-out.
Existing frozen draws should be reused, not regenerated during evaluation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.parser_adaptation.build_winner_pool import difficulty_table                      # noqa: E402
from grounding.paths import CORPORA, REPO_ROOT                      # noqa: E402

DEV_PATH = CORPORA / "nr3d_dev.jsonl"


def training_row_idxs(sets_dir: Path) -> tuple[set[int], list[str]]:
    """Every prompt any arm saw, in training or in its validation split."""
    seen: set[int] = set()
    names: list[str] = []
    for split in sorted(sets_dir.glob("*/")):
        used = False
        for part in ("train.jsonl", "valid.jsonl"):
            path = split / part
            if not path.exists():
                continue
            used = True
            for line in open(path):
                if line.strip():
                    seen.add(json.loads(line)["row_idx"])
        if used:
            names.append(split.name)
    return seen, names


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=str(CORPORA / "nr3d_train.csv"))
    ap.add_argument("--pool",
                    default="data/rollouts/nr3d_rollouts_fft_v4.jsonl",
                    help="the teacher rollout pool; every row_idx in it is "
                         "excluded, whether or not it produced a keeper")
    ap.add_argument("--sets", default="data/training_sets",
                    help="directory of built training sets, for the "
                         "belt-and-braces row_idx exclusion")
    ap.add_argument("--out", default=str(DEV_PATH))
    ap.add_argument("--n", type=int, default=3000,
                    help="dev rows to draw (0 = every eligible row). The draw "
                         "nests, so a larger value extends a smaller one")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.input, newline="")))
    pool = {json.loads(l)["row_idx"]
            for l in open(REPO_ROOT / args.pool) if l.strip()}
    trained, arm_names = training_row_idxs(REPO_ROOT / args.sets)
    print(f"{len(rows)} train rows | {len(pool)} in the rollout pool | "
          f"{len(trained)} in {len(arm_names)} built sets ({', '.join(arm_names)})")
    if not trained:
        sys.exit(f"no training sets found under {args.sets}; refusing to build "
                 "a dev set that cannot be shown to exclude them")
    leaked = trained - pool
    if leaked:
        sys.exit(f"{len(leaked)} trained row_idx are outside the rollout pool "
                 f"{args.pool}: {sorted(leaked)[:5]}\n"
                 "  The pool and the sets disagree about what was trained on; "
                 "pass the pool those sets were actually built from.")

    excluded = pool | trained
    train_utterances = {rows[i]["utterance"].strip() for i in trained
                        if i < len(rows)}

    difficulty = difficulty_table(Path(args.input))
    counts = Counter()
    eligible = []
    for idx, row in enumerate(rows):
        utt = (row.get("utterance") or "").strip()
        if not utt:
            counts["empty utterance"] += 1
        elif idx in excluded:
            counts["in the rollout pool or a training set"] += 1
        elif utt in train_utterances:
            counts["utterance string seen in training"] += 1
        elif not difficulty[idx]["has_boxes"]:
            counts["scan has no boxes"] += 1
        else:
            eligible.append(idx)

    print(f"\n{len(eligible)} eligible rows")
    for why, n in counts.most_common():
        print(f"  excluded  {why:<42}{n:>7}")

    # Shuffle the sorted survivors and take a prefix, rather than sampling: see
    # the module docstring for why sample() cannot be trusted to nest. Sorted
    # again afterwards so the FILE is in row_idx order whatever --n was.
    picks = sorted(eligible)
    random.Random(args.seed).shuffle(picks)
    if args.n and args.n < len(picks):
        picks = picks[:args.n]
    picks = sorted(picks)

    out = Path(args.out)
    out = out if out.is_absolute() else REPO_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        for idx in picks:
            row, meta = rows[idx], difficulty[idx]
            fh.write(json.dumps({
                "row_idx": idx,
                "dataset_idx": idx,
                "stimulus_id": row.get("stimulus_id", ""),
                "scan_id": row["scan_id"],
                "target_id": row["target_id"],
                "instance_type": row["instance_type"],
                "target_label": row["instance_type"],
                "utterance": row["utterance"].strip(),
                "is_hard": meta["is_hard"],
                "label_count": meta["label_count"],
            }, ensure_ascii=False) + "\n")

    # Dev rows whose utterance string also occurs on a POOL prompt that never
    # became training data. Kept (see the docstring on why the string cut is
    # scoped to trained utterances), but counted, and asserted to carry no
    # student exposure -- if one of these ever turns out to have been trained
    # on, the scoping argument is wrong and the build should fail here.
    pool_utterances: dict[str, list[int]] = {}
    for i in pool:
        pool_utterances.setdefault(rows[i]["utterance"].strip(), []).append(i)
    twins = [i for i in picks
             if rows[i]["utterance"].strip() in pool_utterances]
    exposed = [i for i in twins
               if any(j in trained
                      for j in pool_utterances[rows[i]["utterance"].strip()])]
    if exposed:
        sys.exit(f"{len(exposed)} dev rows share an utterance with a TRAINED "
                 f"prompt: {exposed[:5]}. The string exclusion above should "
                 "have removed them; it did not, so do not trust this set.")

    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    hard = sum(1 for i in picks if difficulty[i]["is_hard"])
    stats = {
        "source": args.input,
        "pool_excluded": args.pool,
        "sets_excluded": arm_names,
        "n": len(picks),
        "eligible": len(eligible),
        "excluded": dict(counts),
        "scans": len({rows[i]["scan_id"] for i in picks}),
        "hard": hard,
        "easy": len(picks) - hard,
        "utterance_twins": {
            "n": len(twins),
            "note": "dev rows sharing an utterance string with a rollout-pool "
                    "prompt that was never trained on; 0 of them share one "
                    "with a trained prompt, which the build asserts",
        },
        "seed": args.seed,
        "sha256": sha,
    }
    Path(str(out) + ".stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(f"\nwrote {out}  ({len(picks)} rows over {stats['scans']} scans, "
          f"{hard} hard / {len(picks) - hard} easy)")
    print(f"sha256  {sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
