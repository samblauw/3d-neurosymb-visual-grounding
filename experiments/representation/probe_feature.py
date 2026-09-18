# Per-feature scoring, ablations, and ±50% sensitivity sweeps. Probe sets come
# from build_stage1_probes.py or reference_frame.py
#
#   probe_feature.py score  --probe P               run the full program through the engine
#   probe_feature.py ablate --probe P --feature F   same, twice, with F stripped the second time
#   probe_feature.py sensitivity --probe P --feature F
#                                                   ±50% one-at-a-time encoder sweep
#
# Probes are selected from parser output, not utterance regexes. That preserves
# relation attachment but includes parser errors; lexical probes measure the
# complementary parser-independent ceiling
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from grounding.paths import REPO_ROOT as _ROOT
from grounding.parsing.fields import relation_objects
from grounding.resolution import engine
from grounding.resolution.config import (DEFAULT_RESOLVER, CONTROL_FLAGS, add_resolver_flags,
                                         resolver_metadata,
                                         resolver_from_args, resolver_suffix)
from grounding.resolution.artifacts import relative_path
from grounding.resolution.ranking import expected_rank
from grounding.resolution.statistics import mcnemar

# Shared traversal keeps nested-program interpretation consistent
from grounding.parsing.fields import (ISOLATION, LATERAL,
                                      nodes, relations)


def _is_ordinal(obj):
    return any(isinstance(r.get("rank"), int) and r["rank"] >= 2
               for r in relations(obj))


# Which programs exercise each feature; keys minus "all" are the --feature choices for ablate
FEATURES = {
    "againstwall": lambda o: any(r.get("relation_name") == "against wall"
                                 for r in relations(o)),
    "isolation": lambda o: any(r.get("relation_name") in ISOLATION
                               for r in relations(o)),
    # Only ranks the engine acts on: rank 1 is the default the unranked score computes
    "ordinal": _is_ordinal,
    "color": lambda o: any(n.get("color") for n in nodes(o)),
    # `viewdep` was here. It priced the explicit frame anchor -- strip dropped
    # viewpoint_anchor and kept the relation -- and both halves of that are now
    # gone: v4 emits no viewpoint slot, so the probes carry no anchor to drop,
    # and the resolver ignores the field regardless (resolver.py, "Viewpoint is
    # RETIRED from this path"). The two arms were therefore the same program and
    # scored identically. `refframe` below is the live successor, and asks a
    # different question
    # The reference-frame population. LATERAL only, not all of VIEWDEP: the
    # frame governs an axis, and front/back have none to re-frame -- their
    # unsupplied encoder branch collapses to a symmetric proximity matrix, so a
    # row whose only directional relation is `front` can never show a frame
    # effect and would only dilute the sample. rank >= 2 is excluded for the
    # reason the retired `viewdep` excluded it: those rows are what the ordinal
    # probe measures, and a miss there is unattributable between the frame and
    # the rank
    #
    # reference_frame.py join applies this together with the lexical frame cue:
    # the predicate alone says the program can USE a frame; the cue says the
    # utterance STATES one, and the experiment needs both
    "refframe": lambda o: any(r.get("relation_name") in LATERAL
                              for r in relations(o)) and not _is_ordinal(o),
    # `corner` and `floor` were here too. AtTheCorner and OnTheFloor are
    # UNMODIFIED LaSP -- they only ever had probe sets because the optuna sweep
    # needed a pool per fitted constant, and that work is retired. Probing them
    # measured upstream's encoders, not this tree's, on populations far too
    # small to conclude from either way
    #
    # The whole pool, unstratified. The per-feature sets over-sample the
    # properties under study, which is right for probing an encoder and wrong
    # for asking how the executor should weight the vocabulary as a whole
    "all": lambda o: True,
}
ABLATABLE = [k for k in FEATURES if k != "all"]


# Stage I's four thesis-owned continuous encoder parameters.  The public labels
# are the ones used in the Results section; ``setting`` is the single source of
# truth in grounding.sensitivity.  Do not add scoring or normalisation knobs
# here: this command is specifically the representation-scale robustness check
SENSITIVITY_FEATURES = {
    "againstwall": {"parameter": "delta_wall", "setting": "wall.gap_scale",
                    "arg": "delta_wall"},
    "isolation": {"parameter": "delta_iso", "setting": "isolation.distance_scale",
                  "arg": "delta_iso"},
    "color": {"parameter": "sigma_col", "setting": "color.scale",
              "arg": "sigma_col"},
    "ordinal": {"parameter": "tau_ord", "setting": "ordinal.tau",
                "arg": "tau_ord"},
}


#: Which arm of the thesis comparison a feature's ablation can be attributed to
#: `againstwall` and `isolation` are encoders whose CONDITION the control can
#: still score, so their contribution can be measured under the control's own
#: arithmetic -- a representation-only claim. `color` and `ordinal` cannot be:
#: the control resolver has no colour factor and no rank, so their WITHOUT arm
#: is not a program the control could express. Stripping them changes the
#: resolver as well as the representation, and the label says so
FEATURE_ATTRIBUTION = {
    "againstwall": "representation",
    "isolation": "representation",
    "color": "representation+resolver",
    "ordinal": "representation+resolver",
}

ATTRIBUTION_NOTE = {
    "representation": "encoder only; the same contrast is expressible under the "
                      "control's resolver arithmetic",
    "representation+resolver": "the control resolver has no such term (colour: "
                               "no colour factor; ordinal: no rank), so this is "
                               "not a representation-only contrast",
}


def arm_of(feature, cfg):
    """Which arm of the thesis comparison THIS run is, as opposed to which arm
    the feature could in principle be attributed to.

    A `representation` feature scored under the control's arithmetic is the
    representation-only arm; the same feature under the shipped resolver is not,
    and a `representation+resolver` feature has no representation-only arm at
    any setting.
    """
    if FEATURE_ATTRIBUTION[feature] != "representation":
        return "representation+resolver"
    return ("representation_only" if cfg["baseline_resolver_arithmetic"]
            else "shipped_resolver")


def apply_parameter_overrides(args):
    """Apply explicitly supplied encoder-scale flags, leaving defaults intact."""
    from grounding import sensitivity

    for spec in SENSITIVITY_FEATURES.values():
        value = getattr(args, spec["arg"], None)
        if value is not None:
            args._resolver = sensitivity.set(spec["setting"], value, args._resolver)


def sensitivity_metrics(rows, skipped):
    """Summary for one full-program probe arm, preserving expected-rank ties."""
    n = len(rows)
    if not n:
        raise SystemExit("nothing scored")
    top1 = sum(r["with"]["top1"] for r in rows)
    mrr = sum(r["with"]["rr"] for r in rows) / n
    return {"n": n, "skipped": skipped, "top1_expected_correct": top1,
            "top1_accuracy": top1 / n, "mrr_in_class": mrr}


def strip(obj, feat):
    """Return the program with `feat`'s contribution removed, recursively.

    Stripping, not zeroing: the constraint leaves the program entirely, which is
    what "the parser never emitted it" would have looked like.
    """
    if not isinstance(obj, dict):
        return obj
    out = dict(obj)
    if feat == "color":
        out.pop("color", None)
    rels = []
    for r in obj.get("relations") or []:
        name = r.get("relation_name")
        if feat == "againstwall" and name == "against wall":
            continue
        if feat == "isolation" and name in ISOLATION:
            continue
        r2 = dict(r)
        if feat == "ordinal":
            # Drop the rank, not the relation: "second chair from the left" is still "leftmost"
            r2.pop("rank", None)
        r2["objects"] = [strip(o, feat) for o in relation_objects(r)]
        rels.append(r2)
    out["relations"] = rels
    return out


# --------------------------------------------------------------------------
# engine

_EXEC_ERRORS = (NotImplementedError, ValueError, RuntimeError, KeyError, IndexError)


def load_tree(representation, label_type, config=DEFAULT_RESOLVER):
    """Bind the shared engine to one arm: its entry points, and a scene cache."""
    engine.bind(representation)
    return engine.tree(config), engine.Scenes(label_type)


def score_picks(picks, tree, scenes, label_type, arms):
    """Run every pick through the engine once per arm.

    `arms` maps an arm name to a program transform; "with" must be present and
    is the arm the top-1/class fields are read off. Any arm raising drops the
    whole pick, so the arms stay paired. Returns (rows, skipped).
    """
    import torch

    by_scan = defaultdict(list)
    for p in picks:
        by_scan[p["scan_id"]].append(p)

    rows, skipped = [], 0
    for scan_id, group in sorted(by_scan.items()):
        try:
            sd = scenes.load(scan_id)
        except (FileNotFoundError, KeyError):
            skipped += len(group)
            continue
        obj_ids = list(sd["obj_ids"])
        labels = list(sd["inst_labels"])
        valid = torch.arange(len(sd["pred_locs"]))

        for p in group:
            if p.get("json_obj") is None:
                skipped += 1    # utterance the parser could not handle
                continue
            tgt = int(p["target_id"])
            if tgt not in obj_ids:
                skipped += 1
                continue
            same = [i for i, l in enumerate(labels) if l == p["target_label"]]
            if not same:
                skipped += 1
                continue
            tgt_pos = obj_ids.index(tgt)

            row, ok = {}, True
            for arm, transform in arms.items():
                prog = transform(p["json_obj"])
                try:
                    feats = tree.get_all_features(
                        scan_id, engine.concept_names(prog), scenes, label_type)
                    concept = tree.parse(scan_id, prog, feats, valid)
                except _EXEC_ERRORS:
                    ok = False
                    break
                # Stripping the only discriminator leaves candidates equal: a 1/k guess
                row[arm] = expected_rank([float(concept[i]) for i in same],
                                         same.index(tgt_pos))
                if arm == "with":
                    top = torch.argsort(concept, descending=True,
                                        stable=True)[0].item()
                    row["pred_id"] = int(obj_ids[top])
                    row["pred_is_same_class"] = bool(
                        labels[top] == p["target_label"])
            if not ok:
                skipped += 1
                continue
            row.update(row_idx=p.get("row_idx"), scan_id=scan_id,
                       sentence=p.get("sentence"),
                       target_label=p["target_label"], target_id=tgt,
                       n_in_class=len(same))
            rows.append(row)
        scenes.drop(scan_id)
    return rows, skipped


def write_out(path, payload):
    out = _ROOT / path
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"  wrote {out}")


# --------------------------------------------------------------------------
# score

def cmd_score(args):
    picks = [json.loads(l) for l in open(_ROOT / args.probe) if l.strip()]
    print(f"{len(picks)} picks over {len({p['scan_id'] for p in picks})} scans")
    tree, scenes = load_tree(args.tree, args.label_type, args._resolver)
    rows, skipped = score_picks(picks, tree, scenes, args.label_type,
                                {"with": lambda o: o})

    n = len(rows)
    if not n:
        print("nothing scored")
        return 1
    resolved = sum(r["pred_id"] == r["target_id"] for r in rows)
    top_in_class = sum(r["with"]["top1"] for r in rows)
    right_class = sum(r["pred_is_same_class"] for r in rows)
    mrr = sum(r["with"]["rr"] for r in rows) / n
    chance_top1 = sum(1.0 / r["n_in_class"] for r in rows) / n
    chance_mrr = sum(sum(1.0 / k for k in range(1, r["n_in_class"] + 1)) / r["n_in_class"]
                     for r in rows) / n

    print(f"\n=== {Path(args.probe).stem}: full program through the engine "
          f"({n} scored, {skipped} skipped) ===")
    print(f"  top-1 overall            {resolved}/{n} ({resolved / n:.0%})")
    print(f"  top-1 within class       {top_in_class:.1f}/{n} ({top_in_class / n:.0%})"
          f"   mean tie block {sum(r['with']['tied'] for r in rows) / n:.1f}")
    print(f"  winner was right class   {right_class}/{n}")
    print(f"  mean reciprocal rank     {mrr:.3f}  (random permutation {chance_mrr:.3f})")
    print(f"  random top-1 rate        {chance_top1:.3f}  <- baseline for top-1, NOT MRR")
    print(f"  mean class size          {sum(r['n_in_class'] for r in rows) / n:.1f}")

    if args.out:
        write_out(args.out, {
            "probe": args.probe, "n": n, "skipped": skipped,
            "resolver": resolver_metadata(args._resolver, tree=args.tree,
                                                label_type=args.label_type),
            "top1_overall": resolved, "top1_in_class": top_in_class,
            "mrr_in_class": mrr, "chance_mrr": chance_mrr,
            "chance_top1": chance_top1,
            "rows": [{"row_idx": r["row_idx"], "scan_id": r["scan_id"],
                      "sentence": r["sentence"], "target_label": r["target_label"],
                      "pred_id": r["pred_id"], "target_id": r["target_id"],
                      "resolved": bool(r["pred_id"] == r["target_id"]),
                      "pred_is_same_class": r["pred_is_same_class"],
                      "rank_in_class": r["with"]["rank"],
                      "n_in_class": r["n_in_class"],
                      "exp_top1": r["with"]["top1"], "exp_rr": r["with"]["rr"],
                      "tied": r["with"]["tied"]} for r in rows]})
    return 0


# --------------------------------------------------------------------------
# sensitivity

def cmd_sensitivity(args):
    """Run the pre-specified ±50% sweep for one feature on one fixed probe."""
    from grounding import sensitivity

    spec = SENSITIVITY_FEATURES[args.feature]
    # A sensitivity run is one-at-a-time, so the other three flags would make a result ambiguous
    for other, other_spec in SENSITIVITY_FEATURES.items():
        if other != args.feature and getattr(args, other_spec["arg"], None) is not None:
            raise SystemExit(f"--{other_spec['arg'].replace('_', '-')} is not used by "
                             f"the {args.feature} sensitivity run")

    nominal = sensitivity.PARAMETERS[spec["setting"]][2]
    centre = getattr(args, spec["arg"], None)
    if centre is not None and centre != nominal:
        raise SystemExit(
            f"{spec['parameter']} sensitivity is defined around its fixed nominal "
            f"value {nominal}; do not replace it with --{spec['arg'].replace('_', '-')}")

    picks = [json.loads(l) for l in open(_ROOT / args.probe) if l.strip()]
    tree, scenes = load_tree(args.tree, args.label_type, args._resolver)
    cfg = resolver_metadata(args._resolver, tree=args.tree, label_type=args.label_type)
    suffix = resolver_suffix(cfg)
    stem = Path(args.probe).stem.removeprefix("probe_")
    # The nominal arm verifies against an ablation of the SAME resolver, so it must be local
    results_dir = Path(args.out).parent if args.out else Path("artifacts/results/representation/ablations")
    existing_path = _ROOT / results_dir / f"{stem.replace('againstwall', 'against_wall')}{suffix}.json"
    if not existing_path.exists():
        raise SystemExit(
            f"missing existing ablation for nominal verification: {existing_path}"
            + (f"\n  this sweep runs a non-shipped resolver ({cfg['norm_mode']}, "
               f"use_rank={cfg['use_rank']}), so it verifies against that "
               "resolver's ablation, not the shipped one. Run `ablate` with the "
               "same flags and --out that path first." if suffix else ""))
    existing = json.loads(existing_path.read_text())

    arms = (("minus50", 0.5), ("nominal", 1.0), ("plus50", 1.5))
    results = []
    try:
        for name, multiplier in arms:
            value = nominal * multiplier
            config = sensitivity.set(spec["setting"], value, args._resolver)
            tree = engine.tree(config)
            rows, skipped = score_picks(picks, tree, scenes, args.label_type,
                                        {"with": lambda o: o})
            metrics = sensitivity_metrics(rows, skipped)
            results.append({"name": name, "feature": args.feature,
                            "parameter": spec["parameter"], "probe": args.probe,
                            "multiplier": multiplier, "parameter_value": value,
                            **metrics})
    finally:
        sensitivity.set(spec["setting"], nominal, args._resolver)

    observed = next(r for r in results if r["name"] == "nominal")
    expected_top1 = float(existing["top1_with"])
    expected_mrr = float(existing["mrr_with"])
    reproduced = (observed["n"] == existing["n"]
                  and math.isclose(observed["top1_expected_correct"], expected_top1,
                                   rel_tol=0.0, abs_tol=1e-12)
                  and math.isclose(observed["mrr_in_class"], expected_mrr,
                                   rel_tol=0.0, abs_tol=1e-12))
    # The metric check below already catches a resolver change, because the
    # metrics move when it changes -- but it reports that as "did not reproduce",
    # which reads like a data problem. Comparing the recorded resolvers first
    # names the actual cause, and refuses an ablation that predates the field
    cited_resolver = existing.get("resolver")
    compared = ("norm_mode", "combine", "color_mode", "use_rank", "ordinal_tau")
    if cited_resolver is None:
        raise SystemExit(
            f"{relative_path(existing_path)} records no resolver, so this "
            "sweep cannot state which one its nominal arm verifies against. "
            "Re-run that ablation first.")
    resolver_drift = {k: (cited_resolver.get(k), cfg[k]) for k in compared
                      if cited_resolver.get(k) != cfg[k]}
    verification = {
        "existing_ablation": relative_path(existing_path),
        "expected": {"n": existing["n"], "top1_expected_correct": expected_top1,
                     "mrr_in_class": expected_mrr},
        "observed": {k: observed[k] for k in ("n", "top1_expected_correct",
                                                "mrr_in_class")},
        "cited_resolver": {k: cited_resolver.get(k) for k in compared},
        "resolver_matches_cited_ablation": not resolver_drift,
        "reproduced": reproduced,
    }
    if resolver_drift:
        raise SystemExit(
            "this sweep runs a different resolver than the ablation it verifies "
            f"against ({relative_path(existing_path)}): "
            + ", ".join(f"{k} {a!r} -> {b!r}" for k, (a, b) in sorted(resolver_drift.items())))
    if not reproduced:
        raise SystemExit("nominal sensitivity arm did not reproduce the existing "
                         f"ablation: {json.dumps(verification)}")

    payload = {
        "feature": args.feature,
        "parameter": spec["parameter"],
        "setting": spec["setting"],
        "nominal_parameter_value": nominal,
        "probe": args.probe,
        "results": results,
        "nominal_verification": verification,
        "attribution": FEATURE_ATTRIBUTION[args.feature],
        "attribution_note": ATTRIBUTION_NOTE[FEATURE_ATTRIBUTION[args.feature]],
        "arm": arm_of(args.feature, cfg),
        "resolver": cfg,
    }
    out = args.out or ("artifacts/results/representation/sensitivity/"
                       f"{args.feature.replace('againstwall', 'against_wall')}_{spec['parameter']}_"
                       f"{Path(args.probe).stem.split('_')[-1]}{suffix}.json")
    write_out(out, payload)
    print(f"\n=== {args.feature} / {spec['parameter']} on {Path(args.probe).stem} ===")
    for result in results:
        print(f"  {result['name']:<8} {result['parameter_value']:.6g}  "
              f"top-1 {result['top1_expected_correct']:.3f}/{result['n']} "
              f"({result['top1_accuracy']:.2%})  MRR {result['mrr_in_class']:.3f}")
    print("  nominal arm reproduced the existing ablation")
    return 0


# --------------------------------------------------------------------------
# ablate

def cmd_ablate(args):
    picks = [json.loads(l) for l in open(_ROOT / args.probe) if l.strip()]
    tree, scenes = load_tree(args.tree, args.label_type, args._resolver)
    rows, _ = score_picks(picks, tree, scenes, args.label_type, {
        "with": lambda o: o,
        "without": lambda o: strip(o, args.feature)})

    n = len(rows)
    if not n:
        print("nothing scored")
        return 1
    # The two families answer different questions and are referenced
    # differently. A MIXED probe asks what the feature CONTRIBUTES to a program
    # that resolves without it, which is a paired comparison of two systems and
    # is what the sign test below is for. A SOLE probe asks for the feature's
    # CEILING (_is_sole: "nothing else can resolve the target"), and a ceiling
    # is read against the analytic expected accuracy of an uninformative
    # program -- sum(1/k) -- not against a second system. So a sole run reports
    # chance_top1 and NO wins, losses or McNemar p-value. The stripped arm is
    # still scored and written, as the diagnostic that shows how far the program
    # actually collapses, but it is not a comparator
    programs = [p["json_obj"] for p in picks if p.get("json_obj")]
    family = ("sole" if programs and all(_is_sole(o, args.feature) for o in programs)
              else "mixed")

    with_ok = sum(r["with"]["top1"] for r in rows)
    without_ok = sum(r["without"]["top1"] for r in rows)
    mrr_w = sum(r["with"]["rr"] for r in rows) / n
    mrr_o = sum(r["without"]["rr"] for r in rows) / n
    tied_o = sum(r["without"]["tied"] for r in rows) / n
    chance_top1 = sum(1.0 / r["n_in_class"] for r in rows)

    attribution = FEATURE_ATTRIBUTION[args.feature]
    cfg = resolver_metadata(args._resolver, tree=args.tree, label_type=args.label_type)
    payload = {"feature": args.feature, "probe": args.probe, "family": family,
               "n": n, "top1_with": with_ok, "top1_without": without_ok,
               "mrr_with": mrr_w, "mrr_without": mrr_o,
               "attribution": attribution,
               "attribution_note": ATTRIBUTION_NOTE[attribution],
               "arm": arm_of(args.feature, cfg),
               "resolver": cfg}

    question = "ceiling" if family == "sole" else "contribution"
    print(f"\n=== {args.feature}: {question} on {Path(args.probe).stem} "
          f"({family}, n={n}) ===")
    print(f"  with    the feature   {with_ok:.1f}/{n} ({with_ok / n:.0%})   MRR {mrr_w:.3f}")
    print(f"  without the feature   {without_ok:.1f}/{n} ({without_ok / n:.0%})   MRR {mrr_o:.3f}"
          f"   mean tie block {tied_o:.1f}")

    if family == "sole":
        payload["reference"] = "analytic_expected_accuracy"
        payload["chance_top1"] = chance_top1
        print(f"  analytic reference    {chance_top1:.1f}/{n} "
              f"({chance_top1 / n:.0%})   expected top-1 of an uninformative program")
        print("  no paired test: a sole probe is a ceiling against that "
              "reference, not a contribution")
    else:
        # Sign test on expected top-1: a row moves when its tie-shared credit does
        won = sum(1 for r in rows if r["with"]["top1"] > r["without"]["top1"] + 1e-12)
        lost = sum(1 for r in rows if r["with"]["top1"] < r["without"]["top1"] - 1e-12)
        p_val = mcnemar(won, lost)
        payload.update(won=won, lost=lost, p=p_val, paired_metric="expected_top1")
        print(f"  feature better {won}, stripped better {lost}, p={p_val:.4g}")

    if args.out:
        payload["rows"] = [{"row_idx": r["row_idx"], "sentence": r["sentence"],
                            "n_in_class": r["n_in_class"], "rank_with": r["with"],
                            "rank_without": r["without"]} for r in rows]
        write_out(args.out, payload)
    return 0


# --------------------------------------------------------------------------
# the sole test, shared with build_stage1_probes.py

def _is_sole(obj, feat):
    """The feature is the ONLY spatial constraint, so nothing else can resolve
    the target -- a ceiling, not a contribution."""
    rels = list(relations(obj))
    if feat == "color":
        return not rels and any(n.get("color") for n in nodes(obj))
    if len(rels) != 1:
        return False
    r = rels[0]
    if feat == "againstwall":
        return r.get("relation_name") == "against wall"
    if feat == "isolation":
        return r.get("relation_name") in ISOLATION
    if feat == "refframe":
        # Defined, but NOT the right population for the reference frame, and the
        # reason is structural, not a matter of taste. v4 lowers "facing
        # X" to an ordinary relation, so STATING a frame is what adds the second
        # relation this test forbids: it keeps a minority of the eligible rows,
        # and fewer still of those whose phrasing states a frame outright. It
        # drops "standing in the middle of the room facing the windows" -- the
        # richest frame type there is -- for having said so
        # Sole is a ceiling for an encoder; here it selects against the
        # treatment. Report it as a stratum, not as the sample
        return (r.get("relation_name") in LATERAL
                and not (isinstance(r.get("rank"), int) and r["rank"] >= 2))
    # ordinal. LATERAL as well as the rank, which FEATURES["ordinal"] does not
    # require: it accepts a rank on ANY relation, so a vertical clarifier or a
    # ranked proximity counts as ordinal there. Sole-cue sets have to be
    # homogeneous to be a ceiling, and "second from the left" is the question --
    # ordinal_peak orders by the relation's own score vector, so a ranked
    # `lower` asks for the 2nd lowest and a ranked `front` for the 2nd nearest
    # Both are answerable, neither is the lateral ordering being probed
    return (isinstance(r.get("rank"), int) and r["rank"] >= 2
            and r.get("relation_name") in LATERAL)


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Score and ablate per-feature probe sets (built by build_stage1_probes.py).",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def engine_flags(p):
        p.add_argument("--tree", default="extended")
        add_resolver_flags(p, CONTROL_FLAGS)
        p.add_argument("--label-type", default="gt", choices=["gt", "pred"])
        p.add_argument("--delta-wall", type=float, default=None,
                       help="override delta_wall / wall.gap_scale (nominal: 0.25 m)")
        p.add_argument("--delta-iso", type=float, default=None,
                       help="override delta_iso / isolation.distance_scale (nominal: 1.0 m)")
        p.add_argument("--sigma-col", type=float, default=None,
                       help="override sigma_col / color.scale (nominal: 0.35)")
        p.add_argument("--tau-ord", type=float, default=None,
                       help="override tau_ord / ordinal.tau (nominal: 1.0)")
        p.add_argument("--out", default=None)


    s = sub.add_parser("score", help="run the full program through the engine")
    s.add_argument("--probe", required=True)
    engine_flags(s)
    s.set_defaults(fn=cmd_score)

    a = sub.add_parser("ablate", help="score with and without one feature")
    a.add_argument("--probe", required=True)
    a.add_argument("--feature", required=True, choices=ABLATABLE)
    engine_flags(a)
    a.set_defaults(fn=cmd_ablate)

    q = sub.add_parser("sensitivity", help="run the fixed +/-50%% Stage I encoder sweep")
    q.add_argument("--probe", required=True)
    q.add_argument("--feature", required=True, choices=sorted(SENSITIVITY_FEATURES))
    engine_flags(q)
    q.set_defaults(fn=cmd_sensitivity)

    args = ap.parse_args()
    # build takes no engine flags, so only set it where the engine actually runs
    if getattr(args, "color_mode", None):
        args._resolver = resolver_from_args(args)
    if args.cmd in {"score", "ablate"}:
        apply_parameter_overrides(args)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
