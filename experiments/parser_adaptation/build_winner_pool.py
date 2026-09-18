"""Construct training winners from the R=8 keeper pool.

Deduplicate canonical programs per prompt, require valid v4 targets and
successful re-execution, then rank by cue coverage. Store margins, program size
and difficulty for downstream sampling. Recovery rollouts do not enter SFT."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from filter_rollouts import canonical, rank_and_margin              
from grounding.data.benchmark import BOXES_DIR, load_boxes         
from grounding.parsing import PROMPTS, clean_program, lower_program      
from grounding.parsing import cues                                  
from grounding.parsing.training_set import (Reject, allowed,
                                            attribute_for_predicate,
                                            canonicalise, size, violations)
from grounding.paths import CORPORA, REPO_ROOT                      
from grounding.resolution.config import DEFAULT_RESOLVER              
from grounding.resolution import engine                    
from grounding.resolution.engine import Scenes, rel_num, score      


def difficulty_table(csv_path: Path) -> dict[int, dict]:
    """``is_hard`` per training row, by the benchmark's own definition.

    splits.py calls a row hard when the scan holds more than two instances of
    the target's label; that table is built for the test split only, so the same
    rule is applied here to the train csv. Verified against the 2700-row
    recovery manifest, which agrees on every row.
    """
    counts: dict[str, Counter] = {}
    out: dict[int, dict] = {}
    with open(csv_path, newline="") as f:
        for idx, row in enumerate(csv.DictReader(f)):
            scan_id = row["scan_id"]
            if scan_id not in counts:
                try:
                    counts[scan_id] = Counter(
                        o["label"] for o in load_boxes(scan_id, BOXES_DIR))
                except FileNotFoundError:
                    counts[scan_id] = Counter()
            label = row["instance_type"]
            out[idx] = {
                "stimulus_id": row.get("stimulus_id", ""),
                "target_label": label,
                "label_count": counts[scan_id][label],
                "is_hard": counts[scan_id][label] > 2,
                "has_boxes": bool(counts[scan_id]),
            }
    return out


def read_keepers(path: Path, source: str, exclude: set[int]) -> list[dict]:
    rows = []
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        if r["row_idx"] in exclude:
            continue
        r["source"] = source
        rows.append(r)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keepers",
                    default="data/distilled/nr3d_keepers_fft_v4_ext.jsonl")
    ap.add_argument("--output",
                    default="data/distilled/nr3d_winners_fft_v4_ext.jsonl")
    ap.add_argument("--prompt", default="v4", choices=sorted(PROMPTS))
    ap.add_argument("--margin", type=float, default=0.0,
                    help="re-execution margin floor; 0.0 keeps every correct "
                         "program, which is the point of this pool")
    ap.add_argument("--keep-declared-inert", action="store_true", default=False)
    ap.add_argument("--representation", dest="representation",
                    default=None, help="override; defaults to the accepting arm")
    ap.add_argument("--train-csv", default=str(CORPORA / "nr3d_train.csv"))
    args = ap.parse_args()

    system = PROMPTS[args.prompt]
    vocab = (allowed(system, "Allowed attribute names"),
             allowed(system, "Allowed predicates"),
             allowed(system, "Allowed colors"))

    src = REPO_ROOT / args.keepers
    keeper_stats = json.loads(Path(str(src) + ".stats.json").read_text())
    rep = args.representation or keeper_stats.get("representation", "baseline")
    engine.bind(rep)

    # R=8 keepers only: mixing recovery keepers would make coverage and budget one variable
    rows = read_keepers(src, "original", set())

    difficulty = difficulty_table(Path(args.train_csv))

    rn = rel_num()
    attr_for_pred = attribute_for_predicate(vocab[0], rn)
    scenes = Scenes()
    n: Counter = Counter()
    moves: Counter = Counter()
    drops: Counter = Counter()
    rejects: Counter = Counter()
    n["keeper programs in"] = len(rows)
    n["keeper prompts in"] = len({r["row_idx"] for r in rows})

    by_scan: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_scan[r["scan_id"]].append(r)

    winners: dict[int, list[dict]] = defaultdict(list)
    seen: dict[int, set[str]] = defaultdict(set)

    for si, (scan_id, group) in enumerate(sorted(by_scan.items()), 1):
        for row in group:
            try:
                cand = canonicalise(row["program"], rn, vocab, moves, drops,
                                    args.keep_declared_inert, attr_for_pred)
            except Reject as e:
                rejects[e.reason] += 1
                continue
            if violations(cand, vocab):
                n["failed canonical schema validation"] += 1
                rejects["schema_validation"] += 1
                continue

            key = canonical(cand)
            if key in seen[row["row_idx"]]:
                # Two samples differing only in spelling collapse here; they are one trajectory
                n["duplicate canonical programs removed"] += 1
                continue

            try:
                lowered = lower_program(clean_program(json.loads(key), row["utterance"],
                                            recover=False), rn)
                ids, scores = score(scenes, scan_id, lowered)
            except Exception as exc:                        # noqa: BLE001
                n["failed re-execution"] += 1
                rejects[f"reexec_{type(exc).__name__}"] += 1
                continue
            rank, margin = rank_and_margin(scores, ids, row["target_id"])
            if rank is None or rank > 1:
                n["failed re-execution target criterion"] += 1
                rejects["reexec_target"] += 1
                continue
            if margin < args.margin:
                n["failed re-execution margin criterion"] += 1
                rejects["reexec_margin"] += 1
                continue

            seen[row["row_idx"]].add(key)
            meta = difficulty.get(row["row_idx"], {})
            winners[row["row_idx"]].append({
                "row_idx": row["row_idx"], "scan_id": scan_id,
                "target_id": row["target_id"], "utterance": row["utterance"],
                "stimulus_id": meta.get("stimulus_id", ""),
                "program": cand, "canonical_key": key,
                "margin": margin, "size": size(cand),
                "coverage": cues.coverage(cand, row["utterance"]),
                "cue_count": cues.cue_count(row["utterance"]),
                "predicate_count": cues.predicate_count(cand),
                "word_count": len(row["utterance"].split()),
                "is_hard": bool(meta.get("is_hard", False)),
                "label_count": meta.get("label_count", 0),
                "source": row["source"],
                "sample_index": row["sample_index"],
                "keeper_margin": row["margin"],
                "cleaned": bool(row["cleaned"]),
            })
        scenes.drop(scan_id)
        if si % 50 == 0 or si == len(by_scan):
            print(f"  {si}/{len(by_scan)} scans, "
                  f"{sum(len(v) for v in winners.values())} winners", flush=True)

    n["winner programs"] = sum(len(v) for v in winners.values())
    n["covered prompts"] = len(winners)

    out_path = REPO_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fout:
        for row_idx in sorted(winners):
            # Rank once, here, so every consumer sees the same order
            ordered = sorted(winners[row_idx],
                             key=lambda c: (-c["coverage"], -c["margin"],
                                            c["size"], c["canonical_key"]))
            for rank, c in enumerate(ordered):
                fout.write(json.dumps({**c, "winner_rank": rank},
                                      ensure_ascii=False) + "\n")

    per_prompt = Counter(len(v) for v in winners.values())
    stats = {
        "keepers": args.keepers, "recovery": None,
        "recovery_note": "R=8 only; the recovery diagnostic is archived",
        "prompt": args.prompt, "representation": engine.bound(),
        "resolver": {"norm_mode": DEFAULT_RESOLVER.norm_mode,
                     "color_mode": DEFAULT_RESOLVER.color_mode,
                     "combine": DEFAULT_RESOLVER.combine,
                     "ordinal_tau": DEFAULT_RESOLVER.ordinal_tau},
        "margin_threshold": args.margin,
        "counts": dict(n),
        "winners_per_prompt": dict(sorted(per_prompt.items())),
        "by_source": dict(Counter(c["source"] for v in winners.values() for c in v)),
        "hard_prompts": sum(1 for v in winners.values() if v[0]["is_hard"]),
        "rejections": dict(rejects),
        "transformations": dict(moves.most_common(40)),
        "removals": dict(drops.most_common(40)),
    }
    Path(str(out_path) + ".stats.json").write_text(json.dumps(stats, indent=2))

    print(f"\n--- winner pool ({engine.bound()} arm, {DEFAULT_RESOLVER.norm_mode}, "
          f"margin floor {args.margin}) ---")
    for k, v in n.most_common():
        print(f"  {k:48s} {v}")
    print("\nwinners per prompt")
    for k, v in sorted(per_prompt.items()):
        print(f"  {k:2d}  {v}")
    print("\nrejection reasons")
    for k, v in rejects.most_common():
        print(f"  {v:6d}  {k}")
    print(f"\nWrote {out_path} (+ .stats.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
