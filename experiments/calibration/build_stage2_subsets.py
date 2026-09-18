#!/usr/bin/env python3
"""Build calibration/aggregation and anchor subsets from one parent execution.

C counts executed top-level conditions (K in the thesis). C>=2 supplies the
aggregation comparison; C=1 with structural restrictions supplies anchor
binding. Selected lines are copied byte-for-byte in source order, without
sampling. The manifest records selection rules and input/output hashes.
Generate the parent parses separately; see README.md."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

from grounding.paths import REPO_ROOT as _ROOT
from grounding.resolution.artifacts import sha256
from grounding.resolution.stage2 import (ANCHOR_C1_PARSES, ANCHOR_STRATA,
                                         CGE2_PARSES, STAGE2_PARENT, anchor_eligible,
                                         anchor_stratum, executed_rows, guard_split,
                                         load_rows, load_tree, row_key)

MANIFEST = "data/parses/dev/stage2_subsets.manifest.json"

#: One pool, three strata; the strata group those rows, they are not three files
STRATA = (("C=2", lambda c: c == 2), ("C=3", lambda c: c == 3),
          ("C>=4", lambda c: c >= 4))


def copy_selected(parses, out_file, keep):
    """Copy the selected source lines byte-for-byte, in source order."""
    out = _ROOT / out_file
    out.parent.mkdir(parents=True, exist_ok=True)
    seen, written = set(), 0
    with open(out, "w") as fh:
        for line in open(_ROOT / parses):
            if not line.strip():
                continue
            key = row_key(json.loads(line))
            if key in seen:
                sys.exit(f"duplicate row key {key} in {parses}")
            seen.add(key)
            if key in keep:
                fh.write(line if line.endswith("\n") else line + "\n")
                written += 1
    if written != len(keep):
        sys.exit(f"wrote {written} lines for {len(keep)} selected rows")
    print(f"wrote {out}\nsha256  {sha256(out)}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parses", default=STAGE2_PARENT,
                    help="the Stage II parent: one 30B zero-shot v4 parse per "
                         "shared-dev row")
    ap.add_argument("--tree", default="extended")
    ap.add_argument("--min-predicates", type=int, default=2,
                    help="C threshold for the main set")
    ap.add_argument("--cge2-out", default=CGE2_PARSES)
    ap.add_argument("--anchor-out", default=ANCHOR_C1_PARSES)
    ap.add_argument("--c1-control-out", default=None,
                    help="also write the plain C==1 control. Off by default: "
                         "it is optional and nothing reads it")
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--confirm-on-test", action="store_true",
                    help="allow --parses under data/parses/test/")
    args = ap.parse_args()

    guard_split(args.parses, args.confirm_on_test)
    tree = load_tree(args.tree)
    rows_raw = load_rows(args.parses)
    print(f"{len(rows_raw)} parsed rows in {args.parses}")

    stats = {"skipped": 0}
    c_hist: Counter[int] = Counter()
    cge2, c1, anchor = set(), set(), set()
    anchor_strata: Counter[str] = Counter()
    refused: Counter[str] = Counter()

    # ONE execution, so a subset cannot drift from the C the ablation computes
    for rec in executed_rows(tree, rows_raw, stats):
        c = len(rec.factors)
        c_hist[c] += 1
        if c >= args.min_predicates:
            cge2.add(rec.key)
        if c == 1:
            c1.add(rec.key)
        relation, mult = anchor_eligible(rec, tree.rel_num)
        if relation is None:
            refused[mult] += 1
        else:
            anchor.add(rec.key)
            anchor_strata[anchor_stratum(relation)] += 1

    scored = sum(c_hist.values())
    print(f"\nscored {scored} rows, skipped {stats['skipped']}")
    print("  C  rows")
    for c in sorted(c_hist):
        print(f"  {c:<3}{c_hist[c]:>6}")

    overlap = cge2 & anchor
    if overlap:
        raise SystemExit(f"C>=2 and the anchor pool overlap on {len(overlap)} "
                         "rows; they select on C>=2 and C==1 and cannot")

    outputs = {}
    print(f"\nC >= {args.min_predicates}: {len(cge2)} rows")
    copy_selected(args.parses, args.cge2_out, cge2)
    outputs["cge2"] = args.cge2_out

    print("\nanchor refusals")
    for why, n in sorted(refused.items(), key=lambda kv: -kv[1]):
        print(f"  {why:<40}{n:>6}")
    print(f"\nanchor C=1 structural: {len(anchor)} rows")
    for name in ANCHOR_STRATA:
        print(f"  {name:<40}{anchor_strata[name]:>6}")
    copy_selected(args.parses, args.anchor_out, anchor)
    outputs["anchor_c1"] = args.anchor_out

    if args.c1_control_out:
        print(f"\nC == 1 control: {len(c1)} rows")
        copy_selected(args.parses, args.c1_control_out, c1)
        outputs["c1_control"] = args.c1_control_out

    parent = _ROOT / args.parses
    manifest = {
        "parent": {"path": args.parses, "rows": len(rows_raw),
                   "scored": scored, "skipped": stats["skipped"],
                   "sha256": sha256(parent)},
        "c_histogram": {str(c): c_hist[c] for c in sorted(c_hist)},
        "subsets": {},
        # A grouping of the C>=2 rows, recorded so a results table can be checked against its pool
        "cge2_strata": {name: sum(n for c, n in c_hist.items()
                                  if c >= args.min_predicates and test(c))
                        for name, test in STRATA},
        "anchor_strata": dict(anchor_strata),
        "seed": None,
        "note": "deterministic function of the parent and the engine; "
                "nothing is sampled",
    }
    rules = {
        "cge2": f"C >= {args.min_predicates} executed top-level conditions",
        "anchor_c1": "C == 1, that condition relational, no rank anywhere in "
                     "the program, more than one anchor candidate "
                     "(ordinal.membership > 0.5)",
        "c1_control": "C == 1",
    }
    for key, rel in outputs.items():
        path = _ROOT / rel
        manifest["subsets"][key] = {
            "path": rel, "rule": rules[key],
            "rows": sum(1 for line in path.open() if line.strip()),
            "sha256": sha256(path),
        }

    out = _ROOT / args.manifest
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
