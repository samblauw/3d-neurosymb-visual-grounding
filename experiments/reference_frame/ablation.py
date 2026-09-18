"""Scoring the reference-frame arms: baseline, and the conditioned frames.

`ablate` compares the parsed frame with its exact negation; `doorway` compares a
doorway's inward normal with its outward one. Both score every arm on the same
rows through the extended tree, replacing only the left/right tensors.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

from experiments.reference_frame import (ParsedFrame, SceneFrameContext,
                                         method_for, resolve_reference_frame)
from experiments.reference_frame.conditioning import (LATERAL_RELATIONS,
                                                      frame_conditioned_concepts)
from experiments.reference_frame.frame import resolved
from experiments.reference_frame.frame_pass import SYSTEM_PROMPT
from experiments.reference_frame.rows import (DOORWAY_ROWS,
                                              DOORWAY_V4, REFFRAME_FRAMES,
                                              REFFRAME_PROBE, REPO, digest,
                                              load_jsonl, read_json,
                                              repo_relative)
from experiments.reference_frame.selection import (DOOR_NOUN_RE,
                                                   DOORWAY_FRAME_SLOTS,
                                                   ENTRY_RE,
                                                   doorway_entry_source,
                                                   doorway_observer_anchor)
from grounding.paths import ARTIFACTS, resolve_under_root
from grounding.resolution import engine
from grounding.resolution.config import DEFAULT_RESOLVER, resolver_metadata
from grounding.resolution.statistics import mcnemar as mcnemar_exact


def load_engine(config=DEFAULT_RESOLVER):
    """The extended tree's entry points, and its left/right encoder table."""
    engine.bind("extended")
    tree = engine.tree(config)
    return tree, {name: tree.concepts[name] for name in LATERAL_RELATIONS}


def predict(tree, scan, concepts, program):
    """One arm's prediction for one row: ``(object_id, scores, error)``.

    A scoring failure is a result, not a crash: the row stays in the denominator
    of every arm, which is what keeps the arms paired.
    """
    import torch

    valid_ids = torch.arange(len(scan["pred_locs"])).to(tree.device)
    try:
        scores = tree.parse("", program, concepts, valid_ids)
    except (NotImplementedError, ValueError, RuntimeError, KeyError,
            IndexError) as exc:
        return None, None, type(exc).__name__
    pred = int(scan["obj_ids"][int(torch.argmax(scores).item())])
    return pred, scores, None


def score_frame_arms(tree, view_relations, scan, concepts, program, arm_frames):
    """Score one row once per arm, each arm under its own frame.

    ``arm_frames`` maps an arm name to a resolved frame, or None for the
    unconditioned baseline. Only the left/right tensors are ever replaced, so an
    arm whose frame did not resolve scores the baseline's own dictionary.
    """
    import numpy as np
    import torch

    locs = torch.from_numpy(np.vstack(scan["pred_locs"])).to(tree.device)
    scored = {}
    for name, frame in arm_frames.items():
        conditioned = (concepts if frame is None else
                       frame_conditioned_concepts(concepts, locs, frame,
                                                  view_relations, LATERAL_RELATIONS))
        prediction, scores, error = predict(tree, scan, conditioned, program)
        scored[name] = SimpleNamespace(frame=frame, concepts=conditioned,
                                       prediction=prediction, scores=scores,
                                       error=error)
    return scored


def suffixed(path, suffix, default):
    """Apply `suffix` only to a default output path.

    An explicit --out is the caller's choice and is left exactly as given; the
    defaults get the suffix so a softmax run cannot silently overwrite the
    shipped-resolver file.
    """
    if suffix and str(path) == str(default):
        stem, dot, ext = str(path).rpartition(".")
        return f"{stem}{suffix}{dot}{ext}" if dot else f"{path}{suffix}"
    return path


def score_dict(scan, scores) -> dict[str, float] | None:
    if scores is None:
        return None
    return {str(int(obj_id)): float(score)
            for obj_id, score in zip(scan["obj_ids"], scores)}


#: The general diagnostic's frame pass, read only to compare invocations
REFFRAME_FRAMES_META = Path(str(REFFRAME_FRAMES) + ".meta.json")


DOORWAY_DEFAULT_ROWS = str(DOORWAY_V4)


DOORWAY_DEFAULT_CACHE = ARTIFACTS / "features" / "reference_frame_doorway_gt_label.pth"


DOORWAY_DEFAULT_OUT = "artifacts/results/reference_frame/doorway_entry/ablation.json"


DOORWAY_DEFAULT_ROWS_OUT = "artifacts/results/reference_frame/doorway_entry/predictions.jsonl"


#: join combines utterance selection, a v4 program and the frame pass; eligibility ignores geometry
ABLATE_DEFAULT_ROWS = str(REFFRAME_PROBE)


ABLATE_DEFAULT_CACHE = (ARTIFACTS / "features"
                        / "reference_frame_probe_v4_cued_gt_label.pth")

ABLATE_DEFAULT_OUT = "artifacts/results/reference_frame/observer_frame/ablation.json"

ABLATE_DEFAULT_ROWS_OUT = ("artifacts/results/reference_frame/observer_frame/"
                           "predictions.jsonl")


def inverted_frame(frame):
    """The sign control: preserve every resolved-frame fact except orientation."""
    if not frame.resolved:
        return frame
    # Right derives from forward, so it reverses too: the parsed frame through the origin
    from dataclasses import replace
    return replace(frame, forward=-frame.forward, right=-frame.right,
                   reason="Inverted-frame control: negated parsed forward vector.",
                   detail={**frame.detail, "inverted_frame_control": True})


def feature_cache_for(rows, cache_path, tree, scenes, rebuild=False):
    """The per-scene concept tensors these rows need, built on demand.

    NOT `scripts/evaluation/build_features.py`. That runs the extended tree's
    `compute_features.py`, which builds `Nr3DDataset(label_type=...)` -- and
    that constructor defaults to `split="test"`, so `dataset.scans` holds the
    130 test scans and nothing else. Every reference-frame stage selects from
    nr3d_train, so every one of its scans is a KeyError there. It also writes
    the SHARED extended tensor, which would have to be rebuilt afterwards.

    Both problems go away by going through `Scenes`, the tree's own one-scan
    loader, and writing to a cache named for this diagnostic. `get_all_features`
    only ever reads `.scans`, so it cannot tell the difference.

    A scan already in the cache is reused only if it also carries every concept
    THESE rows need: the cache is keyed by scan, but its contents depend on the
    parse, so a scan cached for one parse can be short a concept for another.
    """
    import torch

    cache = {}
    if cache_path.exists() and not rebuild:
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        print(f"feature cache: {len(cache)} scans from {cache_path}")

    needed = defaultdict(set)
    for r in rows:
        needed[r["scan_id"]] |= set(engine.concept_names(r["json_obj"]))

    missing = [s for s in needed if s not in cache
               or not needed[s] <= set(cache[s])]
    if missing:
        print(f"building features for {len(missing)} scans ...")
        for i, scan_id in enumerate(missing, 1):
            scenes.load(scan_id)
            with engine.inside_tree():
                cache[scan_id] = tree.get_all_features(
                    scan_id, sorted(needed[scan_id]), scenes, "gt")
            print(f"  [{i}/{len(missing)}] {scan_id}", flush=True)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cache, cache_path)
        print(f"wrote {cache_path}")
    return cache


def target_margin(scan, scores, target_id):
    """Target score minus the best competing candidate score for one arm."""
    if scores is None:
        return None
    try:
        target_idx = list(scan["obj_ids"]).index(int(target_id))
    except ValueError:
        return None
    target = float(scores[target_idx])
    others = [float(score) for i, score in enumerate(scores) if i != target_idx]
    return target - max(others) if others else target


def only_named_concepts_changed(baseline_concepts, conditioned_concepts, names):
    """Identity-level invariant for the sign comparison's non-axis features."""
    if set(baseline_concepts) != set(conditioned_concepts):
        return False
    governed = set(names)
    return all(conditioned_concepts[key] is baseline_concepts[key]
               for key in baseline_concepts if key not in governed)


def pairwise_confusion(rows, source, destination):
    """Paired accuracy changes when moving from ``source`` to ``destination``."""
    win = loss = same_ok = same_bad = 0
    for row in rows:
        before = row["arms"][source]["correct"]
        after = row["arms"][destination]["correct"]
        if after and not before:
            win += 1
        elif before and not after:
            loss += 1
        elif before:
            same_ok += 1
        else:
            same_bad += 1
    return {"wins": win, "losses": loss, "unchanged_correct": same_ok,
            "unchanged_wrong": same_bad, "n_discordant": win + loss,
            "mcnemar_p": round(mcnemar_exact(win, loss), 4)}


def mean_target_margin_change(rows, arm):
    """Mean of margin(arm) - margin(baseline), excluding scoring failures."""
    deltas = []
    for row in rows:
        before = row["arms"]["baseline"].get("target_margin")
        after = row["arms"][arm].get("target_margin")
        if before is not None and after is not None:
            deltas.append(after - before)
    return {"mean": sum(deltas) / len(deltas) if deltas else None,
            "n": len(deltas)}


def inversion_report(rows):
    """Baseline, parsed and inverted accuracy with their paired tests."""
    n = len(rows)
    correct = {arm: sum(row["arms"][arm]["correct"] for row in rows)
               for arm in ("baseline", "parsed", "inverted")}
    return {
        "n": n,
        "baseline": {"correct": correct["baseline"],
                     "accuracy": correct["baseline"] / n if n else 0.0},
        "parsed": {"correct": correct["parsed"],
                   "accuracy": correct["parsed"] / n if n else 0.0},
        "inverted": {"correct": correct["inverted"],
                     "accuracy": correct["inverted"] / n if n else 0.0},
        "baseline_to_parsed": pairwise_confusion(rows, "baseline", "parsed"),
        "baseline_to_inverted": pairwise_confusion(rows, "baseline", "inverted"),
        "parsed_to_inverted": pairwise_confusion(rows, "parsed", "inverted"),
        "mean_target_margin_change": {
            "parsed_minus_baseline": mean_target_margin_change(rows, "parsed"),
            "inverted_minus_baseline": mean_target_margin_change(rows, "inverted"),
        },
    }


def arm_entry(scan, target_id, pred, scores, frame=None):
    """One arm's outcome on one row, and the frame it was scored under."""
    entry = {
        "prediction": None if pred is None else int(pred),
        "correct": pred is not None and int(pred) == target_id,
        "error": pred is None,
        "target_margin": target_margin(scan, scores, target_id),
    }
    if frame is not None:
        vec = lambda v: None if v is None else [round(float(v[0]), 4),
                                                round(float(v[1]), 4)]
        entry.update({
            "resolved": frame.resolved,
            "method": frame.method,
            "reason": frame.reason,
            "code": frame.detail.get("code"),
            "confidence": round(float(frame.confidence), 4),
            "forward": vec(frame.forward),
            "right": vec(frame.right),
        })
    return entry


def positive_scores(scan, scores):
    return {str(int(o)): round(float(s), 6)
            for o, s in zip(scan["obj_ids"], scores) if float(s) > 0}


def cmd_ablate(args) -> int:
    rows = load_jsonl(args.rows)
    print(f"{len(rows)} probe rows over "
          f"{len({r['scan_id'] for r in rows})} scans")

    tree, view_relations = load_engine(args._resolver)
    scenes = engine.Scenes()
    cache = feature_cache_for(rows, resolve_under_root(args.features_cache),
                              tree, scenes, rebuild=args.rebuild_features)

    contexts: dict[str, SceneFrameContext] = {}
    out_rows = []
    violations = defaultdict(list)

    for r in rows:
        if not r.get("json_obj"):
            continue                    # an unparsed row scores nothing in any arm
        scan_id = r["scan_id"]
        scan = scenes.load(scan_id)
        if scan_id not in contexts:
            contexts[scan_id] = SceneFrameContext.from_scan(scan_id)
        concepts = cache[scan_id]
        target_id = int(r["target_id"])
        row_id = r.get("row_idx", r.get("dataset_idx"))
        frame = ParsedFrame.from_dict(r.get("viewpoint_frame"))

        parsed = resolve_reference_frame(frame, contexts[scan_id])
        inverted = inverted_frame(parsed)

        scored = score_frame_arms(tree, view_relations, scan, concepts,
                                  r["json_obj"],
                                  {"baseline": None, "parsed": parsed,
                                   "inverted": inverted})
        base = scored["baseline"]
        record = {
            "row_idx": row_id,
            "scan_id": scan_id,
            "sentence": r.get("sentence", ""),
            "target_id": target_id,
            "target_label": r.get("target_label"),
            "parsed_frame": r.get("viewpoint_frame"),
            "routed_method": method_for(frame),
            "n_candidates": len(scan["obj_ids"]),
            "arms": {"baseline": arm_entry(scan, target_id, base.prediction,
                                           base.scores)},
        }
        for arm in ("parsed", "inverted"):
            result = scored[arm]
            # Unresolved must pass the ORIGINAL dictionary through untouched
            if not result.frame.resolved and result.concepts is not concepts:
                violations["unresolved_passes_original_dict"].append((row_id, arm))
            if not only_named_concepts_changed(concepts, result.concepts,
                                               LATERAL_RELATIONS):
                violations["only_left_right_tensors_change"].append((row_id, arm))
            entry = arm_entry(scan, target_id, result.prediction, result.scores,
                              result.frame)
            if (result.scores is not None and base.scores is not None
                    and entry["prediction"] != record["arms"]["baseline"]["prediction"]):
                entry["candidate_scores"] = positive_scores(scan, result.scores)
                record["baseline_candidate_scores"] = positive_scores(scan, base.scores)
            record["arms"][arm] = entry

        if parsed.resolved and not (float(parsed.forward @ inverted.forward) < -0.999999
                                    and float(parsed.right @ inverted.right) < -0.999999):
            violations["inverted_is_exact_frame_negation"].append(row_id)
        record["changed"] = any(
            record["arms"][arm]["prediction"] != record["arms"]["baseline"]["prediction"]
            for arm in ("parsed", "inverted"))
        out_rows.append(record)

    resolved_rows = [r for r in out_rows if r["arms"]["parsed"]["resolved"]]
    drift = [r["row_idx"] for r in out_rows
             if not r["arms"]["parsed"]["resolved"] and r["changed"]]
    checks = {"unresolved_matches_baseline": {
        "pass": not drift, "n_unresolved": len(out_rows) - len(resolved_rows),
        "mismatched_rows": drift}}
    for name in ("unresolved_passes_original_dict", "only_left_right_tensors_change",
                 "inverted_is_exact_frame_negation"):
        checks[name] = {"pass": not violations[name],
                        "violations": violations[name][:10]}

    subsets = {"all_rows": inversion_report(out_rows),
               "resolved_rows": inversion_report(resolved_rows)}
    changed = [r["row_idx"] for r in out_rows if r["changed"]]

    print("\n" + "=" * 72)
    print("GROUNDING ABLATION -- reference frame (left/right)")
    print("=" * 72)
    for name, rep in subsets.items():
        n = rep["n"]
        print(f"\n{name}  (n = {n})")
        for arm in ("baseline", "parsed", "inverted"):
            print(f"  {arm:10s} {rep[arm]['correct']:3d}/{n:<3d} "
                  f"{rep[arm]['accuracy']:6.1%}")
        for pair in ("baseline_to_parsed", "baseline_to_inverted",
                     "parsed_to_inverted"):
            stat = rep[pair]
            print(f"  {pair:22s} +{stat['wins']} / -{stat['losses']}"
                  f"  p={stat['mcnemar_p']:.3f}")
    print("\nverification")
    for name, c in checks.items():
        print(f"  {'PASS' if c['pass'] else 'FAIL'}  {name}"
              + ("" if c["pass"] else f"   {c}"))
    print(f"\n{len(changed)} rows changed prediction under parsed or inverted")

    result = {
        "rows_file": str(args.rows),
        "relations": list(LATERAL_RELATIONS),
        "resolver": resolver_metadata(args._resolver),
        "arms": ["baseline", "parsed", "inverted"],
        "subsets": subsets,
        "checks": checks,
        "changed_rows": changed,
    }
    out = resolve_under_root(suffixed(args.out, args._suffix, ABLATE_DEFAULT_OUT))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"\nWrote: {out}")

    rows_out = resolve_under_root(
        suffixed(args.rows_out, args._suffix, ABLATE_DEFAULT_ROWS_OUT))
    rows_out.parent.mkdir(parents=True, exist_ok=True)
    with rows_out.open("w") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote: {rows_out}")
    return 0 if all(c["pass"] for c in checks.values()) else 1


def viewpoint_pass_checks(frames_meta: Optional[dict],
                          general_meta: Optional[dict]) -> dict[str, bool]:
    """Did the doorway rows go through the general diagnostic's frame pass?

    The point of the change is that both diagnostics extract the frame the same
    way, which is only true if the two passes were the same prompt, the same
    model and the same decoding. Comparing the two sidecars is what makes that
    reproducible from the artifacts, not from the command someone
    remembers running.
    """
    out: dict[str, bool] = {}
    live = hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()
    if frames_meta is None:
        return out
    out["viewpoint_prompt_is_the_frame_schema"] = (
        frames_meta.get("prompt_sha256") == live)
    out["viewpoint_producer_is_the_parse_stage"] = (
        any(command in str(frames_meta.get("producer", "")) for command in (
            "reference_frame.py parse", "experiments.reference_frame parse")))
    if general_meta is None:
        return out
    out["viewpoint_model_matches_general"] = (
        frames_meta.get("model") == general_meta.get("model"))
    # Transitive through the digest above, but stated so an unrecorded prompt digest stays visible
    if general_meta.get("prompt_sha256"):
        out["viewpoint_prompt_matches_general"] = (
            frames_meta.get("prompt_sha256") == general_meta["prompt_sha256"])
    keys = ("sampler", "temperature", "max_tokens", "thinking")
    mine, theirs = frames_meta.get("decoding") or {}, general_meta.get("decoding") or {}
    out["viewpoint_decoding_matches_general"] = all(
        mine.get(k) == theirs.get(k) for k in keys)
    return out


def outward_frame(inward):
    """The wrong-sign control: identical evidence and gates, negated forward."""
    if not inward.resolved:
        return inward
    return resolved("doorway_outward_ablation", -inward.forward,
                    "Wrong-sign control: the same doorway wall's outward normal.",
                    inward.source, inward.confidence, **inward.detail)


def correctness(pred, target) -> bool:
    return pred is not None and int(pred) == int(target)


def arm_summary(rows, name):
    return sum(bool(r["arms"][name]["correct"]) for r in rows)

def select_doorway_rows(rows, frames):
    """The doorway pool, and why every dropped row fell out.

    Three independent gates -- the utterance, the frame pass and the grounding
    program -- are counted separately, each a subset of the one above it, so a
    drop is attributable to one of them.
    """
    selected, drops, lexical, viewpoint_ok, accepted = [], Counter(), 0, 0, 0
    for row in rows:
        idx = int(row.get("row_idx", row.get("dataset_idx")))
        sentence = str(row.get("sentence", ""))
        if ENTRY_RE.search(sentence):
            lexical += 1
        frame_row = frames.get(idx)
        if frame_row is None:
            drops["no_frame_pass_row"] += 1
            continue
        if frame_row.get("viewpoint_parse_error"):
            drops["viewpoint_parse_error"] += 1
            continue
        viewpoint_ok += 1
        frame_obj = frame_row.get("viewpoint_frame")
        anchor, slot = doorway_observer_anchor(frame_obj)
        if anchor is None:
            drops[slot] += 1
            continue
        accepted += 1
        # The anchor is the canonical door noun: the frame pass gates the row, it does not re-aim it
        source = doorway_entry_source(sentence, row.get("json_obj") or {})
        if not source:
            # Both halves are named, not fused: a no_entry_language drop would mean --rows was wrong
            drops["no_lateral_relation" if ENTRY_RE.search(sentence)
                  else "no_entry_language"] += 1
            continue
        selected.append((row, source, frame_obj, anchor, slot))
    selected.sort(key=lambda item: int(item[0].get("row_idx", item[0].get("dataset_idx"))))
    return SimpleNamespace(rows=selected, drops=drops, lexical=lexical,
                           viewpoint_ok=viewpoint_ok, accepted=accepted)


def print_doorway_report(report):
    """The funnel, the arms and the controls, exactly as the run reports them."""
    selection, counts = report["selection"], report["counts"]
    print("DOORWAY ENTRY SIGN ABLATION")
    print("funnel:", " -> ".join(
        f"{k} {v}" for k, v in report["funnel"].items() if k != "dropped"))
    print("  dropped:", report["funnel"]["dropped"] or "none")
    print(f"selected {selection['n_selected']}/{selection['n_input']} rows; "
          f"lexical entry candidates {selection['n_entry_lexical']}")
    print("row ids:", selection["row_ids"])
    for name, value in counts.items():
        print(f"{name:10s} {value['correct']}/{value['n']} ({value['accuracy']:.1%})")
    print(f"resolved {report['resolution']['resolved']}/{selection['n_selected']}")
    for name, block in report["paired_stats"].items():
        print(f"paired ({name}, n={block['n']}):")
        for pair, stat in block.items():
            if pair == "n":
                continue
            print(f"  {pair:22s} +{stat['wins']} / -{stat['losses']}"
                  f"  discordant {stat['n_discordant']}"
                  f"  p={stat['mcnemar_p']:.4f}")
    for name, check in report["checks"].items():
        print(f"{'PASS' if check['pass'] else 'FAIL'}  {name} (n={check['n']})")
    print("changed rows:", report["changed_rows"])


def cmd_doorway(args) -> int:
    rows_path = resolve_under_root(args.rows)
    if not rows_path.exists():
        raise SystemExit(
            f"missing parse: {rows_path}\n"
            "Run `select-doorway`, then parse_split.py over its output "
            "(see README.md, reference-frame usage).")
    frames_path = resolve_under_root(args.frames)
    if not frames_path.exists():
        raise SystemExit(
            f"missing frame pass: {frames_path}\n"
            "The doorway rows go through the SAME viewpoint prompt as the "
            "general diagnostic. Run `parse` over the select-doorway rows "
            "(see README.md, reference-frame usage).")
    rows = load_jsonl(rows_path)
    frames = {int(r["row_idx"]): r for r in load_jsonl(frames_path)}

    funnel = select_doorway_rows(rows, frames)
    selected, drops = funnel.rows, funnel.drops
    if not selected:
        raise SystemExit(
            f"selection is empty: no row in {rows_path} states a doorway "
            "observer frame in the frame pass AND parses to a lateral relation")
    ids = [int(r.get("row_idx", r.get("dataset_idx"))) for r, *_ in selected]
    if len(ids) != len(set(ids)):
        raise SystemExit("selector produced duplicate row IDs")

    tree, view_relations = load_engine(args._resolver)
    scenes = engine.Scenes()
    cache_path = resolve_under_root(args.features_cache)
    cache = feature_cache_for([r for r, *_ in selected], cache_path, tree,
                              scenes, rebuild=args.rebuild_features)
    contexts: dict[str, SceneFrameContext] = {}
    records = []
    controls = defaultdict(list)
    frames_meta = read_json(Path(str(frames_path) + ".meta.json"))
    general_meta = read_json(REFFRAME_FRAMES_META)
    # The parse's meta names its probe, so the chain holds even when --rows is overridden
    rows_meta = read_json(Path(str(rows_path) + ".meta.json")) or {}
    probe_path = ((rows_meta.get("rows") or {}).get("path")) or repo_relative(DOORWAY_ROWS)
    selection_stats = read_json(REPO / f"{probe_path}.stats.json")
    run_config = {
        "selector": ENTRY_RE.pattern,
        "lateral_relations": list(LATERAL_RELATIONS),
        "scoring": resolver_metadata(args._resolver),
        "feature_cache": str(cache_path),
        "viewpoint_pass": {
            "path": repo_relative(frames_path),
            "sha256": hashlib.sha256(frames_path.read_bytes()).hexdigest(),
            "prompt": "reference_frame.SYSTEM_PROMPT (the general diagnostic's)",
            "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
            "model": (frames_meta or {}).get("model"),
            "decoding": (frames_meta or {}).get("decoding"),
            "accepted_slots": list(DOORWAY_FRAME_SLOTS),
            "compared_against": repo_relative(REFFRAME_FRAMES_META),
        },
    }

    for row, source, frame_obj, viewpoint_anchor, viewpoint_slot in selected:
        scan_id = row["scan_id"]
        scan = scenes.load(scan_id)
        ctx = contexts.setdefault(scan_id, SceneFrameContext.from_scan(scan_id))
        concepts = cache[scan_id]
        target = int(row["target_id"])
        frame = ParsedFrame(frame_type="from_only", has_viewpoint=True,
                            from_anchor=source, evidence="deterministic doorway-entry selector")
        inward = resolve_reference_frame(frame, ctx)
        outward = outward_frame(inward)

        scored = score_frame_arms(tree, view_relations, scan, concepts,
                                  row["json_obj"],
                                  {"baseline": None, "outward": outward,
                                   "inward": inward})
        arms = {name: {
            "prediction": arm.prediction,
            "correct": correctness(arm.prediction, target), "error": arm.error,
            "scores": score_dict(scan, arm.scores),
            "frame": None if arm.frame is None else arm.frame.as_dict(),
            "concept_is_baseline": arm.concepts is concepts,
            "left_digest": digest(arm.concepts["left"]) if "left" in arm.concepts else None,
            "right_digest": digest(arm.concepts["right"]) if "right" in arm.concepts else None,
        } for name, arm in scored.items()}

        # Both treatment arms may only replace existing lateral tensors
        for name in ("outward", "inward"):
            changed = {k for k in concepts
                       if scored[name].concepts.get(k) is not concepts[k]}
            controls["only_lateral_tensors_change"].append(changed <= set(LATERAL_RELATIONS))
        if not inward.resolved:
            for name in ("outward", "inward"):
                controls["unresolved_is_baseline_identity"].append(
                    scored[name].concepts is concepts)
                controls["unresolved_scores_match_baseline"].append(
                    arms[name]["scores"] == arms["baseline"]["scores"])

        endpoints = ctx.wall_centreline_endpoints()
        sign_source = "wall_centreline_endpoints" if endpoints is not None and len(endpoints) >= 3 \
            else "room_centre_fallback"
        if inward.resolved:
            controls["no_room_centre_sign_fallback"].append(sign_source != "room_centre_fallback")

        record = {
            "row_idx": int(row.get("row_idx", row.get("dataset_idx"))),
            "scan_id": scan_id, "sentence": row.get("sentence"),
            "target_id": target, "target_label": row.get("target_label"),
            "entry_source": source, "parse": row["json_obj"],
            "viewpoint_frame": frame_obj,
            "viewpoint_anchor": viewpoint_anchor,
            "viewpoint_slot": viewpoint_slot,
            "viewpoint_frame_type": (frame_obj or {}).get("frame_type"),
            "viewpoint_method": method_for(ParsedFrame.from_dict(frame_obj)),
            # Whether the frame pass's own anchor canonicalises to the same door
            # noun the deterministic selector took from the sentence. Reported,
            # not enforced: the geometry reads `entry_source` either way, so a
            # disagreement is information about the parse, not a failure
            "viewpoint_anchor_matches_selector": bool(
                (m := DOOR_NOUN_RE.search(viewpoint_anchor or ""))
                and m.group(1).lower() == source),
            "parse_digest": digest(row["json_obj"]),
            "candidate_ids": [int(x) for x in scan["obj_ids"]],
            "candidate_digest": digest([int(x) for x in scan["obj_ids"]]),
            "base_feature_digests": {key: digest(concepts[key]) for key in LATERAL_RELATIONS if key in concepts},
            "wall_sign_source": sign_source,
            "arms": arms,
        }
        records.append(record)

    counts = {name: arm_summary(records, name) for name in ("baseline", "outward", "inward")}
    unresolved = [r for r in records if not r["arms"]["inward"]["frame"]["resolved"]]
    resolved_rows = [r for r in records if r["arms"]["inward"]["frame"]["resolved"]]

    # The paired tests, in `refframe_ablation.json`'s shape and through its own
    # `pairwise_confusion`, so the two reference-frame reports are read the same
    # way. Derived from the row-level `correct` flags already in `records`:
    # nothing here selects, resolves or scores anything
    #
    # The arms are scored on the SAME rows, so only the discordant cells carry
    # information -- and on an unresolved row every arm is baseline-identical
    # (see the `unresolved_*` checks), contributing nothing to either cell
    # `all_rows` and `resolved_rows` therefore agree on wins/losses and differ
    # only in the concordant cells and n
    paired_stats = {
        name: {"n": len(subset), **{
            f"{a}_to_{b}": pairwise_confusion(subset, a, b)
            for a, b in (("baseline", "inward"), ("baseline", "outward"),
                         ("inward", "outward"))}}
        for name, subset in (("all_rows", records), ("resolved_rows", resolved_rows))
    }
    changed = [r["row_idx"] for r in records if any(
        r["arms"][name]["prediction"] != r["arms"]["baseline"]["prediction"]
        for name in ("outward", "inward"))]
    checks = {name: {"pass": all(values), "n": len(values)}
              for name, values in controls.items()}
    if not args.allow_room_centre_sign:
        checks.setdefault("no_room_centre_sign_fallback", {"pass": True, "n": 0})

    # The frame pass covers exactly the pool, with the general diagnostic's prompt and model
    checks["viewpoint_pass_covers_every_row"] = {
        "pass": {int(r.get("row_idx", r.get("dataset_idx"))) for r in rows} <= set(frames),
        "n": len(rows)}
    for name, ok in viewpoint_pass_checks(frames_meta, general_meta).items():
        checks[name] = {"pass": bool(ok), "n": 1}
    checks["every_selected_row_states_a_doorway_frame"] = {
        "pass": all(r["viewpoint_anchor"] for r in records), "n": len(records)}

    # Normally the reference-frame population: `select-doorway` draws the rows
    # from nr3d_train and `doorway_30b_v4.jsonl` is the parse of exactly those
    # Test input overrides must remain explicitly labelled diagnostics. Store
    # repository-relative paths so absolute and relative CLI inputs agree
    pool = (str(rows_path.relative_to(REPO)) if rows_path.is_relative_to(REPO)
            else str(rows_path))
    on_test = "data/parses/test/" in pool.replace("\\", "/")
    report = {
        "experiment": "doorway_entry_sign_ablation_v1",
        "rows": pool,
        "split": "test" if on_test else "train",
        **({"test_diagnostic": True} if on_test else {}),
        "selection": {"n_input": len(rows), "n_entry_lexical": funnel.lexical,
                      "n_selected": len(records), "row_ids": ids,
                      "rule": "ENTRY_RE AND the frame pass states a doorway "
                              "observer frame AND the parsed relation contains "
                              "left or right"},
        # Every stage of the funnel, each a subset of the one above it
        "funnel": {
            "entry_lexical_in_train": (selection_stats or {}).get("n_entry_lexical"),
            "frame_cued_in_train": (selection_stats or {}).get("n_frame_cued"),
            "entry_and_frame_cued": len(rows),
            "viewpoint_parsed": funnel.viewpoint_ok,
            "viewpoint_accepted": funnel.accepted,
            "lateral_eligible": len(records),
            "geometry_resolved": len(records) - len(unresolved),
            "dropped": dict(sorted(drops.items())),
        },
        "viewpoint_anchor_agrees_with_selector": sum(
            r["viewpoint_anchor_matches_selector"] for r in records),
        "viewpoint_frame_types": dict(Counter(
            r["viewpoint_frame_type"] for r in records)),
        "config": run_config,
        "counts": {name: {"correct": value, "n": len(records),
                          "accuracy": value / len(records)} for name, value in counts.items()},
        "paired_stats": paired_stats,
        "resolution": {"resolved": len(records) - len(unresolved), "unresolved": len(unresolved),
                       "unresolved_rows": [r["row_idx"] for r in unresolved],
                       "unresolved_codes": dict(Counter(
                           r["arms"]["inward"]["frame"]["detail"].get("code") for r in unresolved))},
        "checks": checks, "changed_rows": changed,
    }

    print_doorway_report(report)

    if not args.no_write:
        out = resolve_under_root(
            suffixed(args.out, args._suffix, DOORWAY_DEFAULT_OUT))
        rows_out = resolve_under_root(
            suffixed(args.rows_out, args._suffix, DOORWAY_DEFAULT_ROWS_OUT))
        out.parent.mkdir(parents=True, exist_ok=True)
        rows_out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n")
        with rows_out.open("w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
        print(f"wrote {out}\nwrote {rows_out}")
    return 0 if all(c["pass"] for c in checks.values()) else 1
