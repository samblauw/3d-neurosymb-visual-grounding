"""Which rows each reference-frame diagnostic runs on.

Selection reads the utterance and, after the parse, the executable program; the
frame pass gates the doorway rows. Nothing here scores geometry.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Optional

from experiments.reference_frame import ParsedFrame
from experiments.reference_frame.conditioning import LATERAL_RELATIONS
from experiments.reference_frame.lexicon import DOORWAY_TERMS, normalise
from experiments.reference_frame.rows import (
    REFFRAME_ROWS,
    has_lateral_relation,
    load_jsonl,
    selection_key,
)
from grounding.parsing.fields import relation_objects
from grounding.paths import CORPORA, resolve_under_root
from grounding.representation.lexical_cues import cue_of


# Entry language requires a named doorway, avoiding arbitrary-wall assignments
ENTRY_RE = re.compile(
    r"(?:"
    r"\bwalk(?:ing)?\s+(?:in(?:to)?|through)\b[^.]{0,80}\b(?:door|doors|doorway|entrance|entry)\b"
    r"|\benter(?:ing|ed)?\b[^.]{0,80}\b(?:through|from)\b[^.]{0,40}\b(?:door|doors|doorway|entrance|entry)\b"
    r"|\bcom(?:e|ing|es)\s+(?:in|through)\b[^.]{0,80}\b(?:door|doors|doorway|entrance|entry)\b"
    r"|\blook(?:ing)?\s+(?:in|into)\b[^.]{0,50}\bfrom\s+(?:the\s+)?(?:door|doors|doorway|entrance|entry)\b"
    r"|\bfacing\s+in\s+from\s+(?:the\s+)?(?:door|doors|doorway|entrance|entry)\b"
    r")", re.IGNORECASE)


DOOR_NOUN_RE = re.compile(r"\b(doorway|entrance|entry|doors|door)\b", re.IGNORECASE)


def parsed_relation_names(node: dict) -> set[str]:
    """All relation names in the fixed executable parse, recursively."""
    out: set[str] = set()
    if not isinstance(node, dict):
        return out
    for rel in node.get("relations") or []:
        if isinstance(rel, dict):
            name = rel.get("relation_name")
            if isinstance(name, str):
                out.add(name.lower())
            for child in relation_objects(rel):
                out |= parsed_relation_names(child)
    return out


def doorway_entry_source(sentence: str, json_obj: dict) -> str | None:
    """Return a canonical doorway source iff this is an axis-relevant entry row.

    The noun is canonicalised and colours/determiners are ignored: the scene
    matcher operates on ScanNet labels, not language-level instance identity.
    This makes plural / coloured wording deterministic and exposes ambiguity to
    the geometry gate instead of pretending the adjective picked an instance.
    """
    if not isinstance(sentence, str) or not isinstance(json_obj, dict):
        return None
    if not ENTRY_RE.search(sentence):
        return None
    if not (parsed_relation_names(json_obj) & set(LATERAL_RELATIONS)):
        return None
    noun = DOOR_NOUN_RE.search(sentence)
    return noun.group(1).lower() if noun else None


#: Observer-origin slots only; `toward_anchor` has the opposite orientation
DOORWAY_FRAME_SLOTS = ("from_anchor", "at_anchor", "away_from_anchor")


def doorway_observer_anchor(frame: Optional[dict]) -> tuple[Optional[str], str]:
    """The doorway/entry anchor the VIEWPOINT parse states, or why there is none.

    The doorway analogue of the `has_viewpoint` test `join` applies for the
    general diagnostic, and the reason the two pipelines are now the same shape:
    a row is kept only if the frame pass -- not the selector, not the grounding
    parse -- put the observer at a doorway.

    Reads the frame pass and nothing else. The utterance and the grounding
    program are tested separately by the caller, so the funnel can say which of
    the three dropped a row instead of reporting one fused count.

    The doorway term test is `resolver.method_for`'s, word-level over
    `lexicon.normalise`, so language this accepts is language the resolver
    routes to `doorway_inward`, not a second, looser rule.
    """
    if not isinstance(frame, dict):
        return None, "no_viewpoint_frame"
    if not ParsedFrame.from_dict(frame).has_viewpoint:
        return None, "frame_states_no_viewpoint"
    for slot in DOORWAY_FRAME_SLOTS:
        value = frame.get(slot)
        if not value:
            continue
        if any(term in normalise(str(value)).split() for term in DOORWAY_TERMS):
            return str(value), slot
    return None, "no_doorway_observer_anchor"


def cmd_select_rows(args) -> int:
    """Pick the reference-frame rows from RAW train utterances.

    The retired chain drew this pool from `nr3d_fft_v4_sample_0.jsonl`, teacher
    rollout sample 0, which is the parser-training pool -- so the frame probe
    was conditioned on a parse of rows the student had been trained on, and its
    eligibility test (a lateral relation in the program) was applied BEFORE the
    parse that the diagnostic is supposed to be measuring. Selecting on the
    utterance alone removes both. The program-level restrictions still apply,
    but in `join`, against the v4 parse of THESE rows.
    """
    import csv

    rows = []
    with open(CORPORA / "nr3d_train.csv", newline="") as f:
        for idx, row in enumerate(csv.DictReader(f)):
            cue, kind, _, _ = cue_of("refframe", row["utterance"].lower().strip())
            if cue is None:
                continue
            rows.append({
                "row_idx": idx,
                "stimulus_id": row.get("stimulus_id", ""),
                "scan_id": row["scan_id"],
                "target_id": int(row["target_id"]),
                "target_label": row["instance_type"],
                "sentence": row["utterance"],
                "json_obj": None,
                "src": "refframe_cued",
                "cue": cue,
                "cue_kind": kind,
            })

    # Same sha256-of-row_idx order every other frozen refframe set uses, so a
    # larger --limit extends the same sequence without reshuffling
    # sha256(row_idx) order, the same one every other frozen refframe set uses
    # --max-parse is a prefix of it, so raising the budget extends the sequence
    # and never moves a row an earlier slice already sent to the parser
    rows.sort(key=selection_key)
    selected = rows[:args.max_parse] if args.max_parse else rows

    out = resolve_under_root(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{len(rows)} train utterances state a frame -> {len(selected)} selected")
    print(f"  cue kinds: {dict(Counter(r['cue_kind'] for r in selected))}")
    print(f"  wrote {out}")
    return 0


def cmd_join(args) -> int:
    """Join the two 30B passes over the SAME rows, by row_idx.

    The v4 parse controls grounding: its `json_obj` is what the resolver
    executes, and the frame pass contributes `viewpoint_frame` and nothing else.
    Keeping them in separate files until here is what makes that checkable --
    a frame pass cannot quietly edit the program it is supposed to re-frame.
    """
    v4 = {r["row_idx"]: r for r in load_jsonl(resolve_under_root(args.v4))}
    frames = {r["row_idx"]: r for r in load_jsonl(resolve_under_root(args.frames))}

    shared = sorted(set(v4) & set(frames))
    if not shared:
        raise SystemExit("the two passes share no row_idx -- different inputs?")

    joined, dropped = [], Counter()
    for idx in shared:
        program = v4[idx].get("json_obj")
        frame = frames[idx].get("viewpoint_frame")
        if not isinstance(program, dict):
            dropped["v4 parse failed"] += 1
            continue
        if frame is None:
            dropped["no frame field"] += 1
            continue
        if not ParsedFrame.from_dict(frame).has_viewpoint:
            dropped["frame states no viewpoint"] += 1
            continue
        # LATERAL_RELATIONS only, no rank >= 2: front/back have no axis to re-frame
        if not has_lateral_relation(program):
            dropped["no lateral relation"] += 1
            continue
        if _has_rank(program):
            dropped["rank >= 2"] += 1
            continue
        row = dict(v4[idx])
        row["viewpoint_frame"] = frame
        joined.append(row)

    joined.sort(key=selection_key)
    selected = joined[:args.limit] if args.limit else joined

    out = resolve_under_root(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"v4 {len(v4)} | frames {len(frames)} | shared {len(shared)}")
    print(f"  dropped: {dict(sorted(dropped.items()))}")
    print(f"  eligible {len(joined)} -> selected {len(selected)}")
    print(f"  wrote {out}")
    return 0


def _has_rank(node) -> bool:
    if not isinstance(node, dict):
        return False
    for relation in node.get("relations") or []:
        rank = relation.get("rank")
        if isinstance(rank, int) and rank >= 2:
            return True
        if any(_has_rank(child) for child in relation_objects(relation)):
            return True
    return False


def cmd_select_doorway(args) -> int:
    """Pick the doorway-entry rows from RAW train utterances.

    This stage exists because the ablation used to select its rows out of
    `data/parses/test/nr3d_test_v4.jsonl`. That put a held-out split under a
    Phase I diagnostic for no gain: entry language is a property of the
    utterance, so the rows can be drawn from the same population every other
    reference-frame stage uses.

    The gate is `select-rows`' frame cue AND the entry predicate. Requiring the
    cue is what makes this set a SUBSET of the reference-frame candidate pool
    and not a second, differently-defined population -- and it is also the
    stricter test: the one train utterance that matches ENTRY_RE without stating
    a frame ("the door that comes in by a window") is a predicate false positive
    the cue correctly drops.

    The lateral-relation half of the selector is NOT applied here. It reads the
    program, so it belongs after the parse, in `doorway` itself -- the same
    order `select-rows` and `join` keep, and for the same reason: an eligibility
    test applied before the parse would condition the row set on a parse the
    diagnostic is supposed to be measuring.
    """
    import csv

    rows, n_entry, n_cued = [], 0, 0
    with open(CORPORA / "nr3d_train.csv", newline="") as f:
        for idx, row in enumerate(csv.DictReader(f)):
            utterance = row["utterance"]
            entry = bool(ENTRY_RE.search(utterance))
            cue, kind, _, _ = cue_of("refframe", utterance.lower().strip())
            n_entry += entry
            n_cued += cue is not None
            if not (entry and cue is not None):
                continue
            rows.append({
                "row_idx": idx,
                "stimulus_id": row.get("stimulus_id", ""),
                "scan_id": row["scan_id"],
                "target_id": int(row["target_id"]),
                "target_label": row["instance_type"],
                "sentence": utterance,
                "json_obj": None,
                "src": "doorway_entry_cued",
                "cue": cue,
                "cue_kind": kind,
            })

    # The frozen sha256(row_idx) order, so a --max-parse prefix extends without reshuffling
    rows.sort(key=selection_key)
    selected = rows[:args.max_parse] if args.max_parse else rows

    out = resolve_under_root(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Few of these fall in the frozen 400-row parse, which is why this is its own selection
    overlap = 0
    if REFFRAME_ROWS.exists():
        already = {r["row_idx"] for r in load_jsonl(REFFRAME_ROWS)}
        overlap = len({r["row_idx"] for r in selected} & already)

    stats = {
        "producer": "python -m experiments.reference_frame select-doorway",
        "parent": "data/corpora/nr3d_train.csv",
        "rule": "refframe frame cue AND entry predicate; lateral relation is "
                "applied later, in doorway, against the parse",
        "entry_pattern": ENTRY_RE.pattern,
        "n_entry_lexical": n_entry,
        "n_frame_cued": n_cued,
        "n_selected": len(selected),
        "max_parse": args.max_parse,
        "n_also_in_refframe_rows": overlap,
        "cue_kinds": dict(Counter(r["cue_kind"] for r in selected)),
    }
    Path(str(out) + ".stats.json").write_text(json.dumps(stats, indent=2) + "\n")

    print(f"{n_entry} train utterances use entry language, {n_cued} state a "
          f"frame -> {len(rows)} do both -> {len(selected)} selected")
    print(f"  cue kinds: {stats['cue_kinds']}")
    print(f"  {overlap} of them are already in {REFFRAME_ROWS.name}")
    print(f"  wrote {out}")
    return 0
