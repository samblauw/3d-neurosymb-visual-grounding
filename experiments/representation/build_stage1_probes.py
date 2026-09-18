#!/usr/bin/env python3
"""Construct mixed and sole representation probes from cached v4 parses.

Select lexical cues in raw train utterances, then require the feature in the
parsed program. Mixed and sole are drawn from the same feature-positive parent;
sole applies its structural filter before sampling, not to the mixed sample.
Shuffle scene groups at seed 0 and trim to 100 rows, including the final group.
Partial parse coverage is reported and affects the available population."""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from grounding.paths import CORPORA, DATA
from grounding.paths import REPO_ROOT as _ROOT
from grounding.representation.lexical_cues import DEFAULT_KINDS, cue_of
from probe_feature import FEATURES, _is_sole as is_sole

TRAIN_CSV = CORPORA / "nr3d_train.csv"
PARSE_DIR = DATA / "parses" / "stage1"
PROBE_DIR = DATA / "probes" / "stage1"

# Reference-frame selection is separate: python -m experiments.reference_frame
STAGE1_FEATURES = ["againstwall", "isolation", "ordinal", "color"]


# Compute budget for a prefix of the seeded order; extending frozen parses moves membership
DEFAULT_MAX_PARSE = 800

#: Rows in nr3d_train.csv. Cached because every counter is a fraction of it
_N_RAW: int | None = None


def n_raw() -> int:
    global _N_RAW
    if _N_RAW is None:
        with open(TRAIN_CSV, newline="") as f:
            _N_RAW = sum(1 for _ in csv.DictReader(f))
    return _N_RAW


def cued_rows(feature: str) -> list[dict]:
    """Every train row whose RAW utterance carries a cue for `feature`.

    The regex decides only what is eligible. Whether the feature is the sole
    constraint, whether it is negated, and whether the sentence means it at all
    are read off the parse, which is the only way the parser's own failures stay
    visible.
    """
    kinds = DEFAULT_KINDS.get(feature)
    picks = []
    with open(TRAIN_CSV, newline="") as f:
        for idx, row in enumerate(csv.DictReader(f)):
            utt = row["utterance"].lower().strip()
            cue, kind, _, _ = cue_of(feature, utt, kinds)
            if cue is None:
                continue
            picks.append({
                "row_idx": idx,
                "stimulus_id": row.get("stimulus_id", ""),
                "scan_id": row["scan_id"],
                "target_id": int(row["target_id"]),
                "target_label": row["instance_type"],
                "sentence": row["utterance"],
                "json_obj": None,
                "src": f"stage1_{feature}_cued",
                "cue": cue,
                "cue_kind": kind,
            })
    return picks


def draw(rows: list[dict], n: int, seed: int) -> list[dict]:
    """Take up to `n` rows, drawing whole scenes in a seeded scene order."""
    by_scan: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_scan[r["scan_id"]].append(r)
    scans = sorted(by_scan)
    random.Random(seed).shuffle(scans)
    picked: list[dict] = []
    for s in scans:
        if len(picked) >= n:
            break
        picked.extend(by_scan[s])
    return picked[:n]


def cmd_select(args) -> int:
    rows = cued_rows(args.feature)
    # Order the candidates by the SAME seeded scene shuffle the probe draw uses,
    # so --max-parse has the prefix property: raising it extends the same
    # sequence and never changes which rows the smaller parse already covered
    # 800 -> 1200 therefore adds rows and moves none, which is what makes the
    # budget a compute decision, not a selection one
    ordered = draw(rows, args.max_parse or len(rows), args.seed)
    out = _ROOT / args.out if args.out else PARSE_DIR / f"{args.feature}_cued_rows.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in ordered:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{args.feature}: {len(rows)} cued rows over "
          f"{len({r['scan_id'] for r in rows})} scans "
          f"-> {len(ordered)} to parse -> {out.relative_to(_ROOT)}")
    print(f"  cue kinds: {dict(Counter(r['cue_kind'] for r in ordered))}")
    return 0


def cmd_build(args) -> int:
    feature = args.feature
    parse_path = _ROOT / (args.parse or PARSE_DIR / f"{feature}_cued_v4.jsonl")
    if not parse_path.exists():
        raise SystemExit(
            f"no v4 parse at {parse_path}. Run `select` and parse those rows "
            f"first -- see README.md, representation probes.")

    cued = {r["row_idx"]: r for r in cued_rows(feature)}

    parsed, seen = [], set()
    for line in open(parse_path):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        idx = row.get("row_idx")
        if idx in seen or not isinstance(row.get("json_obj"), dict):
            continue
        # A parse row with no cue means the cue table moved; refusing keeps the population
        if idx not in cued:
            raise SystemExit(
                f"{parse_path.name} row_idx {idx} is not in the current "
                f"{feature} cue selection. The cue table changed since the "
                f"parse; re-run `select` and re-parse.")
        seen.add(idx)
        row["cue"] = cued[idx]["cue"]
        row["cue_kind"] = cued[idx]["cue_kind"]
        parsed.append(row)

    test = FEATURES[feature]
    parent = [r for r in parsed if test(r["json_obj"])]
    sole = [r for r in parent if is_sole(r["json_obj"], feature)]

    # How deep into the candidate order the parse reached. `select` writes a
    # PREFIX of this order, so a parse of a budgeted slice is that prefix and
    # nothing else; reading the depth back off the parse means the budget is
    # recoverable from the files on disk without a manifest to go stale
    order = [r["row_idx"] for r in draw(list(cued.values()), len(cued), args.seed)]
    rank = {idx: i for i, idx in enumerate(order)}
    depth = max((rank[r["row_idx"]] for r in parsed), default=-1) + 1
    contiguous = {r["row_idx"] for r in parsed} == set(order[:depth])

    mixed_rows = draw(parent, args.n, args.seed)
    sole_rows = draw(sole, args.n, args.seed)

    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, rows in (("mixed", mixed_rows), ("sole", sole_rows)):
        out = PROBE_DIR / f"{feature}_{name}.jsonl"
        with open(out, "w") as f:
            for r in rows:
                f.write(json.dumps({
                    "row_idx": r["row_idx"],
                    "scan_id": r["scan_id"],
                    "target_id": r["target_id"],
                    "target_label": r["target_label"],
                    "sentence": r.get("sentence") or r.get("utterance"),
                    "json_obj": r["json_obj"],
                    "cue": r["cue"],
                    "cue_kind": r["cue_kind"],
                    "src": f"stage1_{feature}_{name}",
                }, ensure_ascii=False) + "\n")
        written[name] = out

    stats = {
        "feature": feature,
        "seed": args.seed,
        "n_requested": args.n,
        # The design's counters, in lineage order
        "n_raw": n_raw(),
        "n_lexical": len(cued),
        "n_preparse": depth,
        "n_parsed": len(parsed),
        "n_feature_positive": len(parent),
        "n_mixed": len(mixed_rows),
        "n_sole": len(sole_rows),
        # A parse of the pre-parse slice is that prefix; False makes n_preparse an upper bound
        "preparse_is_prefix": contiguous,
        "parse_coverage": round(len(parsed) / depth, 4) if depth else 0.0,
        "sole_eligible": len(sole),
        "mixed_scans": len({r["scan_id"] for r in mixed_rows}),
        "sole_scans": len({r["scan_id"] for r in sole_rows}),
        "overlap_mixed_sole": len({r["row_idx"] for r in mixed_rows}
                                  & {r["row_idx"] for r in sole_rows}),
        "parse": str(parse_path.relative_to(_ROOT)),
        "cue_kinds_mixed": dict(Counter(r["cue_kind"] for r in mixed_rows)),
    }
    (PROBE_DIR / f"{feature}.stats.json").write_text(json.dumps(stats, indent=2) + "\n")

    print(f"{feature:<12} raw {n_raw()} | lexical {len(cued):>5} | pre-parse "
          f"{depth:>5}{'' if contiguous else ' (NOT a prefix)'} | parsed "
          f"{len(parsed):>5} ({stats['parse_coverage']:.0%}) | feature-positive "
          f"{len(parent):>4} | sole-eligible {len(sole):>4}")
    print(f"  -> {written['mixed'].relative_to(_ROOT)}  {len(mixed_rows)} rows / "
          f"{stats['mixed_scans']} scans")
    print(f"  -> {written['sole'].relative_to(_ROOT)}   {len(sole_rows)} rows / "
          f"{stats['sole_scans']} scans  (overlap with mixed: "
          f"{stats['overlap_mixed_sole']})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("select", help="rows to hand parse_split.py, in draw order")
    s.add_argument("--feature", required=True, choices=STAGE1_FEATURES)
    s.add_argument("--max-parse", type=int, default=DEFAULT_MAX_PARSE,
                   help="cap the rows sent to the parser (0 = every cued row). "
                        "The order is the probe draw order, so raising this "
                        "extends the same sequence and never moves a row an "
                        f"earlier slice already had. Default {DEFAULT_MAX_PARSE}")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", default=None)
    s.set_defaults(run=cmd_select)

    b = sub.add_parser("build", help="carve mixed + sole from the v4 parse")
    b.add_argument("--feature", required=True, choices=STAGE1_FEATURES)
    b.add_argument("--parse", default=None,
                   help="default: data/parses/stage1/<feature>_cued_v4.jsonl")
    b.add_argument("--n", type=int, default=100)
    b.add_argument("--seed", type=int, default=0)
    b.set_defaults(run=cmd_build)

    args = ap.parse_args()
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
