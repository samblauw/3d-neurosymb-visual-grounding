"""Execute the independent baseline or package-native extended system."""
from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from functools import lru_cache, partial
from types import SimpleNamespace

import torch

from grounding.resolution.config import DEFAULT_RESOLVER
from grounding.paths import BASELINE_TREE
from grounding.parsing.parse import normalise_representation

__all__ = ["Scenes", "bind", "bound", "concept_names", "inside_tree", "rel_num",
           "score", "target_rank", "tree"]

_REP = "baseline"


def bind(representation: str) -> None:
    """Choose the default representation for subsequent calls."""
    global _REP
    _REP = normalise_representation(representation)


def bound() -> str:
    return _REP


@contextmanager
def inside_tree(representation=None):
    """Only the independent baseline requires its historical working directory."""
    rep = representation or _REP
    if rep == "extended":
        yield
        return
    cwd = os.getcwd()
    os.chdir(BASELINE_TREE)
    try:
        yield
    finally:
        os.chdir(cwd)


def tree(config=DEFAULT_RESOLVER, representation=None) -> SimpleNamespace:
    """Bind a scorer to its representation and immutable numerical settings."""
    return _tree(normalise_representation(representation or _REP), config)


@lru_cache(maxsize=None)
def _tree(representation, config):
    from grounding._vendor.lasp.eval_helper import get_all_categories, get_all_relations

    if representation == "extended":
        from grounding.data.dataset import load_one_scene
        from grounding.representation import pipeline as compute_features
        from grounding.resolution.resolver import parse, DEVICE
        scorer = partial(parse, rel_num=compute_features.rel_num,
                         valid_relations=compute_features.ALL_VALID_RELATIONS,
                         config=config)
    else:
        if str(BASELINE_TREE) not in sys.path:
            sys.path.insert(0, str(BASELINE_TREE))
        with inside_tree("baseline"):
            from src.eval.eval_nr3d import parse, DEVICE
            from src.dataset.datasets import load_one_scene
            from src.relation_encoders import compute_features
        scorer = parse
    return SimpleNamespace(
        parse=scorer, device=DEVICE,
        load_one_scene=load_one_scene,
        get_all_features=compute_features.get_all_features,
        rel_num=compute_features.rel_num,
        get_all_categories=get_all_categories,
        get_all_relations=get_all_relations,
        get_all_colors=getattr(compute_features, "get_all_colors", None),
        concepts=getattr(compute_features, "CONCEPTS", None))


class Scenes:
    """Cache scans by ID through the bound arm's loader.

    Feature building consumes ``.scans``."""

    def __init__(self, label_type: str = "gt", representation=None) -> None:
        self.representation = normalise_representation(representation or _REP)
        self.label_type = label_type
        self.scans: dict[str, dict] = {}

    def load(self, scan_id: str) -> dict:
        if scan_id not in self.scans:
            load_one_scene = tree(representation=self.representation).load_one_scene
            with inside_tree(self.representation):
                _, scan = load_one_scene(scan_id, self.label_type)
            self.scans[scan_id] = scan
        return self.scans[scan_id]

    def drop(self, scan_id: str) -> None:
        # Release the full point cloud when a caller finishes with this scan.
        self.scans.pop(scan_id, None)


def concept_names(json_obj: dict, representation=None) -> list[str]:
    # The baseline has no colour keys, so categories + relations is the whole
    # set. The extended arm also scores colour, and its parse() skips any colour
    # key that was not requested, so those must be asked for by name.
    t = tree(representation=representation)
    names = t.get_all_categories(json_obj) | t.get_all_relations(json_obj)
    if t.get_all_colors is not None:
        names |= {f"color:{c}" for c in t.get_all_colors(json_obj)}
    return sorted(names)


def rel_num() -> dict[str, int]:
    return tree().rel_num


def score(scenes: Scenes, scan_id: str, json_obj: dict, config=DEFAULT_RESOLVER):
    """(object_ids, score_vector) for one program, exactly as this arm would."""
    t = tree(config, scenes.representation)
    scan = scenes.load(scan_id)
    feats = t.get_all_features(scan_id, concept_names(json_obj, scenes.representation), scenes,
                               scenes.label_type)
    valid = torch.arange(len(scan["pred_locs"]))
    return list(scan["obj_ids"]), t.parse(scan_id, json_obj, feats, valid)


def target_rank(scores, obj_ids: list, target_id) -> int | None:
    """One-based strictly-greater rank; every top tie receives rank 1.

    Return None for an absent target. IDs may be integers or serialized strings.
    """
    target_id = int(target_id)
    ids = [int(i) for i in obj_ids]
    if target_id not in ids:
        return None
    t = float(scores[ids.index(target_id)])
    return int(sum(1 for v in scores if float(v) > t)) + 1
