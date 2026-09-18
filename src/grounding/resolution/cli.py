"""LaSP-derived evaluation harness delegating scoring to the thesis resolver."""

from functools import partial
import json
import os
import torch
from tqdm import tqdm
from openai import OpenAI

from grounding.resolution.evaluate import count_result, row_key as _row_key
from grounding.representation.pipeline import ALL_VALID_RELATIONS, rel_num
from grounding._vendor.lasp.eval_helper import eval_ref_one_sample, construct_bbox_corners
from grounding.data.dataset import Nr3DDataset
from grounding._vendor.lasp.vlm import query_vlm

from grounding.resolution import evaluate as _evaluate
from grounding.resolution.resolver import parse as _resolver_parse
from grounding.paths import CORPORA, RAW
from grounding.resolution.config import (
    COLOR_MODES, DEFAULT_RESOLVER, FROZEN_RESOLVER, NORM_MODES,
)


client = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#: Which split's strata table the row indices below are keyed by. Set from
#: --split in eval_nr3d(): the tables' index ranges overlap, so the wrong one
#: returns another utterance's flags instead of nothing.
_SPLIT = "test"


def _split_flags(dataset_idx):
    return _evaluate.split_flags(dataset_idx, split=_SPLIT)


parse = partial(_resolver_parse, rel_num=rel_num, valid_relations=ALL_VALID_RELATIONS)

def eval_nr3d(args):
    config = DEFAULT_RESOLVER.updated(
        norm_mode=args.norm_mode, color_mode=args.color_mode, use_rank=args.use_rank,
        ordinal_tau=(DEFAULT_RESOLVER.ordinal_tau if args.ordinal_tau is None
                     else args.ordinal_tau))
    global _SPLIT
    _SPLIT = getattr(args, "split", "test")
    dataset = Nr3DDataset(split=_SPLIT, label_type=args.label_type)
    all_features = torch.load(args.features_path, map_location='cpu', weights_only=False)
    top_k = args.top_k
    threshold = args.threshold
    use_vlm = args.use_vlm
    if use_vlm:
        global client
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    scan_data = {}
    _dump = open(args.dump, "w") if getattr(args, "dump", None) else None
    # "official": the fixed Nr3D test-set counts, so accuracies stay comparable
    # to the paper. "parses": denominators counted off the file actually being
    # run, for subset files (probes, ablations) where the official totals would
    # divide by utterances that are not in the file at all.
    totals_mode = getattr(args, "totals", "official")
    result = _evaluate.empty_result(totals_mode)

    def _count_attempt(dataset_idx):
        """Credit one line of the parses file to its split denominators."""
        if totals_mode != "parses":
            return
        _h, _v = _split_flags(dataset_idx)
        count_result(result, "total", _h, _v)


    for line in tqdm(open(getattr(args, "parses", None)
                          or CORPORA / "nr3d.jsonl")):
        data = json.loads(line)
        # Counted before the skips below: an utterance the parser or executor
        # could not handle is a miss, not an exclusion, under either mode.
        _count_attempt(_row_key(data))
        scan_id = data["scan_id"]
        json_obj = data["json_obj"]
        if json_obj is None:
            continue
        target_obj_id = data["target_id"]
        dataset_idx = _row_key(data)
        sentence = data.get("sentence", data.get("utterance", ""))
        features_this_scene = all_features[scan_id]
        scan_data = dataset.scans[scan_id]
        is_hard, is_view_dependent = _split_flags(dataset_idx)
        # Preserve target-label validation when a row omits its label.
        if not data.get("target_label"):
            scan_data["inst_labels"][scan_data["obj_ids"].index(int(target_obj_id))]
        all_locations = scan_data["pred_locs"]
        valid_ids = torch.arange(len(all_locations)).to(DEVICE)
        try:
            final_concept = parse(scan_id, json_obj, features_this_scene, valid_ids, config=config)
        except (NotImplementedError, ValueError, RuntimeError):
            continue

        max_bbox = torch.argmax(final_concept).item()
        top_k_ids = torch.topk(final_concept, min(top_k, len(final_concept))).indices

        gt_locs = scan_data["gt_locs"]
        pred_locs = scan_data["pred_locs"]
        obj_ids = scan_data["obj_ids"]
        assert len(final_concept) == len(obj_ids)
        target_obj_id = data["target_id"]
        obj_ids = scan_data["obj_ids"]
        gt_center, gt_size = gt_locs[target_obj_id][:3],gt_locs[target_obj_id][3:]
        answers = []
        best_iou_at_top_k = 0
        for top_k_id in top_k_ids:
            pred = pred_locs[top_k_id]
            pred_center, pred_size = pred[:3], pred[3:]
            iou = eval_ref_one_sample(
                construct_bbox_corners(pred_center, pred_size),
                construct_bbox_corners(gt_center, gt_size)
            )
            best_iou_at_top_k = max(best_iou_at_top_k, iou)

        pred = all_locations[max_bbox]
        pred_center, pred_size = pred[:3], pred[3:]
        iou = best_iou_at_top_k
        top_k_candidates = [obj_ids[top_k_id] for top_k_id in top_k_ids]
        if _dump is not None:
            _mx = float(final_concept.max())
            _dump.write(json.dumps({
                "dataset_idx": int(dataset_idx),
                "scan_id": scan_id,
                "target_id": int(target_obj_id),
                "pred": int(top_k_candidates[0]),
                "correct": bool(top_k_candidates[0] == target_obj_id),
                "n_tied_at_max": int((final_concept == _mx).sum()),
                "n_objects": int(len(final_concept)),
                "argmax_idx": int(torch.argmax(final_concept)),
                "sel_idx": int(top_k_ids[0]),
            }) + "\n")
        logits = final_concept[top_k_ids]
        logits /= logits.max()
        # is_hard and is_view_dependent already computed above
        if use_vlm:
            answers = [target_obj_id]
            filtered_candidates = [
                candidate for candidate, logit in zip(top_k_candidates, logits) if logit > threshold
            ]
            if all([candidate in answers for candidate in filtered_candidates]):
                count_result(result, "correct", is_hard, is_view_dependent)
            if all([candidate not in answers for candidate in filtered_candidates]):
                continue

            answer_from_vlm = query_vlm(scan_id, sentence, filtered_candidates, client=client, data_root=RAW)
            if answer_from_vlm in answers:
                count_result(result, "correct", is_hard, is_view_dependent)
        else:
            if top_k_candidates[0] == target_obj_id:
                count_result(result, "correct", is_hard, is_view_dependent)
    if _dump is not None:
        _dump.close()
    for k, v in result.items():
        if v["total"]:
            print(k, v["correct"], v["total"], v["correct"] / v["total"])

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_path", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--use_vlm", action="store_true")
    parser.add_argument("--label_type", choices=["pred", "gt"], type=str)
    parser.add_argument("--split", choices=["test", "dev", "train"],
                        default="test",
                        help="which annotation csv the scene loader reads: test (default) is the held-out split, dev the frozen dev draw written by scripts/prepare/build_dev_csv.py. Features and parses must be on the same split")
    parser.add_argument("--dump", type=str, default=None,
                        help="per-utterance jsonl, for comparing two runs")
    parser.add_argument("--parses", type=str, default=None,
                        help="parses jsonl to execute; features must be built from it")
    parser.add_argument("--totals", choices=["official", "parses"],
                        default="official",
                        help="official: fixed Nr3D test-set denominators "
                             "(3669/3816/2478/5007/7485), comparable to the "
                             "paper. parses: denominators counted off --parses, "
                             "for subset files.")
    parser.add_argument("--norm_mode",
                        choices=NORM_MODES,
                        default=FROZEN_RESOLVER["norm_mode"],
                        help="per-relation-node normalization in parse(). "
                             "zscore_sigmoid (default) is this tree's method; "
                             "zscore is the same standardisation under a "
                             "softmax; softmax reproduces the released "
                             "behaviour")
    parser.add_argument("--color_mode", choices=COLOR_MODES,
                        default=FROZEN_RESOLVER["color_mode"],
                        help="how the colour factor combines with the category "
                             "term. product (default) keeps the colour ordering; "
                             "min reproduces the released torch.minimum, which "
                             "discards it wherever every candidate scores above "
                             "the category floor")
    parser.add_argument("--use_rank", action="store_true", default=True,
                        help="honour a relation's rank ('the second from the left')")
    parser.add_argument("--no_use_rank", dest="use_rank", action="store_false",
                        help="drop rank, reproducing the released behaviour")
    parser.add_argument("--ordinal_tau", type=float, default=None,
                        help="decay length of the ordinal raw score, in rank "
                             "positions (default 1.0). Resolution-time, so a "
                             "sweep of it needs no feature rebuild -- unlike "
                             "the wall/isolation/colour scales. see "
                             "grounding.sensitivity")
    args = parser.parse_args()
    eval_nr3d(args)


if __name__ == "__main__":
    main()
