"""Stage II plumbing: traced execution, scene batching and pool rules.

The experiments are ``experiments/calibration/score_aggregation.py``; the pools they
run on are written by ``experiments/calibration/build_stage2_subsets.py``. Both
count C, the number of top-level relation factors, through ``executed_rows``,
so a pool cannot drift from the C an experiment computes.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch

from grounding.parsing.fields import relation_objects
from grounding.paths import REPO_ROOT
from grounding.representation.encoders.ordinal import membership
from grounding.representation.features import get_all_colors
from grounding.resolution import engine, tracing
from grounding.resolution.config import DEFAULT_RESOLVER
from grounding.resolution.statistics import mcnemar

STAGE2_PARENT = "data/parses/dev/nr3d_dev_30b_v4.jsonl"
CGE2_PARSES = "data/parses/dev/nr3d_dev_30b_v4_cge2.jsonl"
ANCHOR_C1_PARSES = "data/parses/dev/nr3d_dev_30b_v4_anchor_c1.jsonl"
STAGE2_OUT = "artifacts/results/calibration/dev"
DIAGNOSTICS_OUT = "artifacts/results/calibration/test_diagnostics"

#: Peak-relative category membership threshold, matching `ordinal.is_row`.
ANCHOR_MEMBERSHIP_THRESH = 0.5

DIRECTIONAL = {"left", "right", "front", "back", "behind", "middle", "center"}
ANCHOR_STRATA = ("explicit anchor", "anchorless directional",
                 "anchorless non-directional")

_EXEC_ERRORS = (NotImplementedError, ValueError, RuntimeError, KeyError, IndexError)


# --- rows and engine ----------------------------------------------------------

def guard_split(parses, allow_test):
    if "test" in Path(parses).parts and not allow_test:
        sys.exit(f"refusing test parses: {parses}; pass --confirm-on-test only "
                 "to confirm a choice already made on dev")


def load_rows(parses):
    """Parsed rows; a row without a program cannot be scored by any arm."""
    path = REPO_ROOT / parses
    if not path.exists():
        sys.exit(f"no parses at {parses}; see README.md")
    rows = (json.loads(line) for line in open(path) if line.strip())
    return [row for row in rows if row.get("json_obj")]


def row_key(row):
    return row.get("row_idx", row.get("dataset_idx"))


def load_tree(representation):
    """Bind the engine to one arm and return that tree's entry points."""
    engine.bind(representation)
    return engine.tree()


def predicted_id(obj_ids, score):
    return int(obj_ids[int(torch.argmax(score).item())])


# --- traced execution ---------------------------------------------------------

def bind_multiplicity(bind):
    """How many objects could fill the most ambiguous anchor role of one bind."""
    return max(int((membership(s) > ANCHOR_MEMBERSHIP_THRESH).sum())
               for s in bind["anchors"])


def traced_parse(tree, json_obj, concepts, valid):
    """Score one program: (score or None, top-level factors, top-level binds)."""
    binds = []
    trace = tracing.start_trace(binds=binds)
    try:
        score = tree.parse("", json_obj, concepts, valid)
    except _EXEC_ERRORS:
        score = None
    finally:
        tracing.stop_trace()
    top = [f for f in trace if f["depth"] == 0]
    root_binds = [{"multiplicity": bind_multiplicity(b)}
                  for b in binds if b["depth"] == 0]
    return score, top, root_binds


def appearance_of(top, color_mode):
    """The category and colour terms, combined as the resolver combines them."""
    out = None
    for f in top:
        if f["kind"] == "cat":
            out = torch.minimum(torch.ones_like(f["vec"]), f["vec"])
        elif f["kind"] == "color" and out is not None:
            out = (out * f["vec"] if color_mode == "product"
                   else torch.minimum(out, f["vec"]))
    return out


def scene_batches(tree, scenes, rows, stats, progress_every=50):
    """Load each scene and the features its rows need once, then release it."""
    by_scan = defaultdict(list)
    for row in rows:
        by_scan[row["scan_id"]].append(row)
    for n, (scan_id, group) in enumerate(sorted(by_scan.items()), 1):
        try:
            scene = scenes.load(scan_id)
        except (FileNotFoundError, KeyError):
            stats["skipped"] += len(group)
            continue
        wanted = set()
        for row in group:
            wanted |= set(tree.get_all_relations(row["json_obj"]))
            wanted |= set(tree.get_all_categories(row["json_obj"]))
            wanted |= {f"color:{c}" for c in get_all_colors(row["json_obj"])}
        concepts = tree.get_all_features(scan_id, sorted(wanted), scenes, "gt")
        valid = torch.arange(len(scene["pred_locs"])).to(tree.device)
        yield group, concepts, valid, list(scene["obj_ids"])
        scenes.drop(scan_id)
        if progress_every and n % progress_every == 0:
            print(f"  [{n}/{len(by_scan)}] scans", flush=True)


def executed_rows(tree, rows, stats):
    """Run the engine on each row and yield its top-level trace.

    C, the number of top-level relation factors (``len(rec.factors)``), is
    defined here, so the pools and the experiments count it the same way.
    """
    for group, concepts, valid, obj_ids in scene_batches(tree, engine.Scenes(),
                                                         rows, stats):
        for row in group:
            score, top, root_binds = traced_parse(tree, row["json_obj"],
                                                  concepts, valid)
            appearance = (None if score is None
                          else appearance_of(top, DEFAULT_RESOLVER.color_mode))
            if appearance is None:
                stats["skipped"] += 1
                continue
            yield SimpleNamespace(
                row=row, key=row_key(row),
                factors=[f for f in top if f["kind"] == "rel"],
                root_binds=root_binds, appearance=appearance, obj_ids=obj_ids,
                engine_pred=predicted_id(obj_ids, score))


# --- anchor pool eligibility --------------------------------------------------

def program_relations(node):
    """Visit relations at every level of a program."""
    if isinstance(node, dict):
        for relation in node.get("relations") or []:
            yield relation
            for child in relation_objects(relation):
                yield from program_relations(child)


def carries_rank(node, operative_only=False):
    for relation in program_relations(node):
        rank = relation.get("rank")
        if (isinstance(rank, int) and rank >= 2) if operative_only else rank is not None:
            return True
    return False


def anchor_stratum(relation):
    if relation_objects(relation):
        return "explicit anchor"
    return ("anchorless directional"
            if relation.get("relation_name") in DIRECTIONAL
            else "anchorless non-directional")


def anchor_eligible(rec, rel_num):
    """(root relation, multiplicity) for a C=1 anchor-pool row, else (None, reason)."""
    if len(rec.factors) != 1:
        return None, "C != 1"
    if len(rec.root_binds) != 1:
        return None, "root condition is not relational"
    program = rec.row["json_obj"]
    # Rank 1 also disqualifies a row, even though the resolver ignores it.
    if carries_rank(program):
        return None, ("carries an ordinal rank (operative)"
                      if carries_rank(program, operative_only=True)
                      else "carries an ordinal rank (rank 1, inoperative)")
    multiplicity = rec.root_binds[0]["multiplicity"]
    if multiplicity < 2:
        return None, "single possible anchor"
    name = rec.factors[0]["name"].split("|")[0]
    for relation in program.get("relations") or []:
        if relation.get("relation_name") == name and name in rel_num:
            return relation, multiplicity
    return None, "root relation not recoverable"


# --- reporting ----------------------------------------------------------------

def accuracy_table(title, groups, into, *, results, arms, base=None):
    """Print and record accuracy per group, pairing every arm against `base`."""
    base = arms[0] if base is None else base
    print(f"\n{title}")
    print(f"  {'group':<28}{'n':>6}" + "".join(f"{arm:>30}" for arm in arms))
    for name, keys in groups:
        keys = [k for k in keys if k in results]
        if not keys:
            continue
        base_correct = sum(results[k][base] for k in keys)
        entry, cells = {"n": len(keys)}, []
        for arm in arms:
            correct = sum(results[k][arm] for k in keys)
            value = {"correct": correct, "accuracy": correct / len(keys)}
            if arm == base:
                cells.append(f"{correct}/{len(keys)} {value['accuracy']:.3f}")
            else:
                wins = sum(results[k][arm] and not results[k][base] for k in keys)
                losses = sum(results[k][base] and not results[k][arm] for k in keys)
                p = mcnemar(wins, losses)
                value.update(wins=wins, losses=losses, mcnemar_p=round(p, 4),
                             delta_pp=round(100 * (correct - base_correct) / len(keys), 2))
                cells.append(f"{value['accuracy']:.3f} +{wins}/-{losses} p={p:.3f}")
            entry[arm] = value
        into[str(name)] = entry
        print(f"  {str(name):<28}{len(keys):>6}" + "".join(f"{c:>30}" for c in cells))


def write_report(args, name, report):
    if args.confirm_on_test:
        report = {**report, "test_diagnostic": True}
    from grounding.resolution.artifacts import calibration_result_name
    path = REPO_ROOT / args.out / calibration_result_name(name, Path(args.parses))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {path}")
