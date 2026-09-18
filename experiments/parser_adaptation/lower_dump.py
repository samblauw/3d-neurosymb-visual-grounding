#!/usr/bin/env python3
# ----------------------------------------------------------------------------
# IN   data/rollouts/<pool>.jsonl               --input   (rows with `samples`)
#      or any flat dump                        --input   (parsed / program / raw)
# OUT  data/parses/<prefix>_sample_<i>.jsonl    --outdir --prefix   (rollout pool)
#      or one file                             --out               (flat dump)
#      jsonl: dataset_idx, row_idx, scan_id, target_id, target_label,
#      sentence, json_obj
# `json_obj` is LOWERED and runnable, null where nothing parsed. Only for a dump
# you ALREADY have -- rollouts/filter/parse all lower as they write
# ----------------------------------------------------------------------------
"""Lower an existing flat parse dump for an engine vocabulary.

Use this only for dumps not already lowered by their producing command.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


from grounding.paths import REPO_ROOT
from grounding.parsing import (REPRESENTATIONS, extract_json_objects, load_rel_num,
                               lower_one, normalise_representation)


# The flat JSON a row carries, whatever it calls it: parsed, program or raw
def flat_json(row: dict) -> dict | None:
    for key in ("parsed", "program"):
        obj = row.get(key)
        if isinstance(obj, dict):
            return obj
    raw = row.get("raw")
    return extract_json_objects(raw) if isinstance(raw, str) else None


def utterance_of(row: dict) -> str:
    return row.get("utterance") or row.get("sentence") or ""


# One line in the shape eval_nr3d reads. Both id keys are written because the
# evaluators look up splits by dataset_idx while the probe tooling keys on
# row_idx. target_id is an int, not the csv's string: eval_nr3d indexes gt_locs
# with it and compares against obj_ids, so a string scores a clean-looking zero
def out_row(row: dict, json_obj) -> str:
    idx = row.get("row_idx", row.get("dataset_idx"))
    sentence = utterance_of(row)
    label = row.get("target_label") or row.get("instance_type", "")
    return json.dumps({
        "dataset_idx": row.get("dataset_idx", idx),
        "row_idx": idx,
        "stimulus_id": row.get("stimulus_id", ""),
        "scan_id": row.get("scan_id", ""),
        "target_id": int(row["target_id"]),
        "target_label": label,
        "instance_type": label,
        "utterance": sentence,
        "sentence": sentence,
        "json_obj": json_obj,
    }, ensure_ascii=False) + "\n"


# A row that produced nothing is written with a null json_obj, not
# dropped: eval_nr3d counts the line and then skips it, so the failure lands in
# the denominator as a miss. Dropping it would divide a sample out of its own
# failure rate and leave the files at different lengths
def split_pool(rows, rel_num, recover, outdir: Path, prefix: str) -> int:
    outdir.mkdir(parents=True, exist_ok=True)
    outcomes: Counter = Counter()
    files: dict[int, object] = {}
    for row in rows:
        for i, sample in enumerate(row.get("samples", [])):
            if i not in files:
                files[i] = open(outdir / f"{prefix}_sample_{i}.jsonl", "w")
            json_obj = lower_one(flat_json(sample), utterance_of(row), rel_num,
                                 recover=recover, outcomes=outcomes)
            files[i].write(out_row(row, json_obj))
    for f in files.values():
        f.close()
    print(f"Wrote {len(files)} files to {outdir}/{prefix}_sample_*.jsonl")
    report(outcomes)
    return 0


def lower_flat(rows, rel_num, recover, out: Path) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    outcomes: Counter = Counter()
    with open(out, "w") as fout:
        for row in rows:
            json_obj = lower_one(flat_json(row), utterance_of(row), rel_num,
                                 recover=recover, outcomes=outcomes)
            fout.write(out_row(row, json_obj))
    print(f"Wrote {out}")
    report(outcomes)
    return 0


def report(outcomes: Counter) -> None:
    total = sum(outcomes.values())
    for k, v in outcomes.most_common():
        print(f"  {k:12s} {v}  ({100 * v / total:.1f}%)")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="jsonl to lower")
    ap.add_argument("--out", default=None,
                    help="single output file (flat dumps)")
    ap.add_argument("--outdir", default="data/parses",
                    help="output directory (rollout pools)")
    ap.add_argument("--prefix", default="nr3d",
                    help="output prefix (rollout pools)")
    ap.add_argument("--representation", dest="representation",
                    default="extended", choices=list(REPRESENTATIONS),
                    help="whose rel_num vocabulary lower_program lowers against: "
                         "baseline | extended")
    # Off by default, as the RFT filter has it: recovery scores the repair, not the parse
    ap.add_argument("--recover", action="store_true", default=False,
                    help="let clean_program infer constraints the sample omitted")
    args = ap.parse_args()

    src = REPO_ROOT / args.input if not Path(args.input).is_absolute() else Path(args.input)
    rows = [json.loads(l) for l in open(src) if l.strip()]
    if not rows:
        print(f"{src} is empty")
        return 1

    rep = normalise_representation(args.representation)
    rel_num = load_rel_num(rep)
    print(f"{len(rows)} rows | {len(rel_num)} relations in {rep}'s vocabulary")

    if isinstance(rows[0].get("samples"), list):
        outdir = Path(args.outdir)
        return split_pool(rows, rel_num, args.recover,
                          outdir if outdir.is_absolute() else REPO_ROOT / outdir,
                          args.prefix)

    if not args.out:
        return print("flat dump: --out is required") or 1
    out = Path(args.out)
    return lower_flat(rows, rel_num, args.recover,
                      out if out.is_absolute() else REPO_ROOT / out)


if __name__ == "__main__":
    raise SystemExit(main())
