"""Inherited Nr3D loader with shared geometry-cache and split adaptations."""

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from grounding._vendor.lasp.eval_helper import (
    convert_pc_to_box,
    is_explicitly_view_dependent,
    load_pc,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCAN_FAMILY_BASE = Path("data/referit3d")
PC_PATH = SCAN_FAMILY_BASE / "scan_data" / "pcd_with_global_alignment"
# The tree's own benchmark/ symlink, NOT data/benchmark -- data/ resolves to
# data/raw, which has no benchmark dir. Single source of truth for boxes,
# shared with the extended arm (see grounding/data/benchmark.py).
BENCHMARK_DIR = Path("benchmark")

_EXTENTS = None


def scene_extents():
    """Per-scan [min; max] of the point cloud, from the same benchmark/ dir the
    boxes come from. Read here rather than imported so the control tree keeps
    depending on nothing outside itself."""
    global _EXTENTS
    if _EXTENTS is None:
        with open(BENCHMARK_DIR / "scene_extents.json") as f:
            _EXTENTS = json.load(f)
    return _EXTENTS

BACKGROUND_LABELS = {"wall", "floor", "ceiling"}

_FEATS = None


# One annotation csv per split, so a run declares which rows it is on instead of
# having the test csv redirected under it. `dev` is the held-out dev draw written
# by scripts/prepare/build_dev_csv.py; `train` is the whole corpus, filtered to
# the train scan list in Nr3DDataset.
SPLIT_STEM = {"test": "{anno}_test", "dev": "{anno}_dev", "train": "{anno}"}


def resolve_anno_file(stem: str) -> Path:
    """Find a referit3d annotation csv."""
    candidates = [SCAN_FAMILY_BASE / "annotations" / "refer" / f"{stem}.csv", Path("data") / f"{stem}.csv"]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"None of {[str(c) for c in candidates]} exist.")


def _get_feats():
    """Load the ZSVG3D feats_3d.pkl on demand."""
    global _FEATS
    if _FEATS is None:
        with open(SCAN_FAMILY_BASE / "feats_3d.pkl", "rb") as f:
            _FEATS = pickle.load(f)
    return _FEATS


def load_pc_(scan_id: str):
    feats = _get_feats()
    return feats[scan_id]["obj_ids"], feats[scan_id]["inst_locs"], feats[scan_id]["center"], feats[scan_id]["obj_embeds"]


def load_pc_gt(inst_labels, inst_locs, scan_id):
    """Reconstruct object list from ground-truth annotations alone."""
    obj_ids = [i for i, label in enumerate(inst_labels) if label not in BACKGROUND_LABELS]
    _, pc_obj_ids, pc_inst_locs, center, _ = load_pc(scan_id)
    assert list(pc_obj_ids) == obj_ids, f"{scan_id}: object indexing diverged"
    return obj_ids, pc_inst_locs, center, None


def load_one_scene(scan_id: str, label_type: str, load_points: bool = True):
    # Fixed to point to the correct benchmark script output path under data/benchmark/boxes
    with open(BENCHMARK_DIR / "boxes" / f"{scan_id}_bboxes.json") as f:
        _canon = json.load(f)
    
    inst_labels = [o["label"] for o in _canon]
    inst_locs = np.array([o["centre"] + o["size"] for o in _canon], dtype=float)
    inst_colors = json.load(open(SCAN_FAMILY_BASE / "scan_data" / "instance_id_to_gmm_color" / f"{scan_id}.json"))

    result = {"gt_locs": inst_locs, "gt_colors": inst_colors, "scan_id": scan_id}

    object_info = lambda: load_pc_gt(inst_labels, inst_locs, scan_id) if label_type == "gt" else load_pc_(scan_id)

    # PATCHED-IN, not upstream: read the cached geometry instead of the cloud on
    # the GT path. This is a frozen control, so the change is confined to WHERE
    # the numbers come from, never what they are:
    #
    #   pred_locs    load_pc's own boxes, cached by build_benchmark.py and
    #                asserted equal to a live load_pc by `--verify`. float32 is
    #                part of matching it -- the values agree in double precision
    #                too, but the ternary encoder's 3-D contraction accumulates
    #                differently and a few concepts drift.
    #   room_points  a 2x3 [min; max] tensor. AtTheCorner reduces the cloud to
    #                x/y min and max and OnTheFloor to z min, so this reproduces
    #                both exactly rather than approximately.
    #
    # Dropped on this path only, all three verified to have no reader anywhere in
    # the tree: 'room_corners', 'pcds', 'colors'. Verified inert end to end by
    # regenerating the control's tensor.
    if label_type == "gt" and load_points:
        ext = scene_extents()[scan_id]
        fg = [o for o in _canon if not o["is_background"]]
        obj_ids = [o["object_id"] for o in fg]
        result["room_points"] = np.array([ext["min"], ext["max"]], dtype=np.float32)
        result.update({
            "obj_ids": obj_ids,
            "pred_locs": np.array([o["centre"] + o["size"] for o in fg], dtype=np.float32),
            "inst_labels": [inst_labels[i] for i in obj_ids],
            "obj_embeds": None,
        })
        return scan_id, result

    if not load_points:
        obj_ids, pred_inst_locs, _, obj_embeds = object_info()
        result.update({
            "obj_ids": obj_ids,
            "pred_locs": pred_inst_locs,
            "inst_labels": [inst_labels[i] for i in obj_ids],
            "obj_embeds": obj_embeds,
        })
        return scan_id, result

    pcd_data = torch.load(PC_PATH / f"{scan_id}.pth", weights_only=False)
    points, colors, instance_labels = pcd_data[0], pcd_data[1], pcd_data[-1]
    
    result["room_points"] = points
    center, size = convert_pc_to_box(points)
    result["room_corners"] = np.array([
        [center[0] + size[0] / 2, center[1] + size[1] / 2],
        [center[0] + size[0] / 2, center[1] - size[1] / 2],
        [center[0] - size[0] / 2, center[1] - size[1] / 2],
        [center[0] - size[0] / 2, center[1] + size[1] / 2],
    ])

    obj_ids, pred_inst_locs, _, obj_embeds = object_info()
    result.update({
        "obj_ids": obj_ids,
        "pred_locs": pred_inst_locs,
        "inst_labels": [inst_labels[i] for i in obj_ids],
        "obj_embeds": obj_embeds,
        "pcds": [torch.from_numpy(points[instance_labels == i]).to(DEVICE) for i in range(instance_labels.max() + 1)],
        "colors": [torch.from_numpy(colors[instance_labels == i]).mean(dim=0) for i in range(instance_labels.max() + 1)]
    })
    return scan_id, result


def load_scannet(scan_ids, label_type, load_points=True):
    return {scan_id: load_one_scene(scan_id, label_type, load_points=load_points)[1] for scan_id in tqdm(scan_ids)}


class Nr3DDataset:
    def __init__(self, anno_type="nr3d", split="test", label_type="pred", load_points=True):
        assert anno_type in ["nr3d", "sr3d"] and label_type in ["gt", "pred"]
        assert split in SPLIT_STEM, f"unknown split {split!r}"
        
        df = pd.read_csv(resolve_anno_file(SPLIT_STEM[split].format(anno=anno_type)))
        
        # Optionally load the precomputed splits table if it exists
        splits_file = BENCHMARK_DIR / "splits.json"
        self.splits_meta = json.load(open(splits_file, "r")) if splits_file.exists() else {}

        training_ids = set((SCAN_FAMILY_BASE / "annotations/splits/scannetv2_train.txt").read_text().splitlines())
        
        self.scan_ids = set()
        self.data = []
        
        for _, item in df.iterrows():
            if not item["mentions_target_class"]:
                continue
            if split == "train" and (item["uses_color_lang"] or item["uses_shape_lang"] or item["scan_id"] not in training_ids):
                continue
            self.scan_ids.add(item["scan_id"])
            self.data.append(item)

        self.scans = load_scannet(self.scan_ids, label_type, load_points=load_points)
        
        for scan_id in self.scan_ids:
            self.scans[scan_id]["label_count"] = Counter(self.scans[scan_id]["inst_labels"])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data.iloc[idx] if isinstance(self.data, pd.DataFrame) else self.data[idx]
        item_id = item["stimulus_id"]
        scan_id = item["scan_id"]
        tgt_object_id = int(item["target_id"])
        
        obj_labels = deepcopy(self.scans[scan_id]["inst_labels"])
        tgt_object_label = obj_labels[self.scans[scan_id]["obj_ids"].index(tgt_object_id)]
        
        tokens = eval(item["tokens"]) if isinstance(item["tokens"], str) else item["tokens"]
        
        # Pull precomputed values from splits.json if available, fallback to dynamic computation
        if item_id in self.splits_meta:
            meta = self.splits_meta[item_id]
            is_multiple = meta.get("is_multiple", self.scans[scan_id]["label_count"][tgt_object_label] > 1)
            is_hard = meta.get("is_hard", self.scans[scan_id]["label_count"][tgt_object_label] > 2)
            is_view_dependent = meta.get("is_view_dependent", is_explicitly_view_dependent(tokens))
        else:
            is_multiple = self.scans[scan_id]["label_count"][tgt_object_label] > 1
            is_hard = self.scans[scan_id]["label_count"][tgt_object_label] > 2
            is_view_dependent = is_explicitly_view_dependent(tokens)
        
        return {
            "scan_id": scan_id,
            "sentence": item["utterance"],
            "tgt_object_id": tgt_object_id,
            "tgt_object_label": tgt_object_label,
            "data_idx": item_id,
            "distractors": list(map(int, item_id.split("-")[4:])),
            "is_multiple": is_multiple,
            "is_view_dependent": is_view_dependent,
            "is_hard": is_hard,
        }
