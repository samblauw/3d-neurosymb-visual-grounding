#!/usr/bin/env python3
# Select Nr3D train utterances containing a feature cue. The regex only chooses
# candidates; parsing decides semantics, negation, and sole-feature eligibility
# The cue table itself lives in grounding.representation.lexical_cues
#
#   lexical_probe.py <feature> [--out FILE]   report the cue table's yield
#
# Rows carry the selecting cue, not a program; parsing fills `json_obj` later
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

from grounding.paths import CORPORA
from grounding.paths import REPO_ROOT as _ROOT
from grounding.representation.lexical_cues import CUES, DEFAULT_KINDS, cue_of

TRAIN_CSV = CORPORA / "nr3d_train.csv"


def select(feature: str, kinds: set[str] | None) -> list[dict]:
    picks: list[dict] = []
    funnel: Counter = Counter()
    with open(TRAIN_CSV, newline="") as f:
        for idx, row in enumerate(csv.DictReader(f)):
            utt = row["utterance"].lower().strip()
            if cue_of(feature, utt)[0] is None:
                continue
            funnel["cued"] += 1
            cue, kind, at, end = cue_of(feature, utt, kinds)
            if cue is None:
                funnel["wrong kind"] += 1
                continue
            picks.append({
                "row_idx": idx,
                "stimulus_id": row.get("stimulus_id", ""),
                "scan_id": row["scan_id"],
                "target_id": int(row["target_id"]),
                "target_label": row["instance_type"],
                "sentence": row["utterance"],
                "json_obj": None,
                "src": f"lexical_{feature}_cue",
                "cue": cue,
                "cue_kind": kind,
            })
    dropped = ", ".join(f"{v} {k}" for k, v in funnel.items() if k != "cued")
    print(f"{funnel['cued']} train utterances carry a {feature} cue"
          f"{' | dropped: ' + dropped if dropped else ''} -> {len(picks)} to parse",
          file=sys.stderr)
    return picks


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("feature", choices=sorted(CUES))
    ap.add_argument("--kind", default=None,
                   help="comma-separated cue kinds to keep. Defaults to all "
                        "kinds except where DEFAULT_KINDS narrows it (see CUES)")
    ap.add_argument("--out", default=None,
                   help="write the cued rows here. Off by default: this CLI "
                        "inspects a cue table's yield; the pre-parse slice the "
                        "pipeline parses is written by build_stage1_probes.py "
                        "select, in probe-draw order")
    args = ap.parse_args()

    kinds = (set(args.kind.split(",")) if args.kind
             else DEFAULT_KINDS.get(args.feature))
    picks = select(args.feature, kinds)
    where = ""
    if args.out:
        out = Path(args.out)
        if not out.is_absolute():
            out = _ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            for pick in picks:
                f.write(json.dumps(pick, ensure_ascii=False) + "\n")
        where = f"  -> {out.relative_to(_ROOT)}"

    print(f"  {len(picks)} picks over {len({p['scan_id'] for p in picks})} scans{where}")
    print(f"  cue kinds: {dict(Counter(p['cue_kind'] for p in picks))}")
    print(f"  cues:      {dict(Counter(p['cue'] for p in picks).most_common())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
