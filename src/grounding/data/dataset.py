"""LaSP-derived loader with thesis wall/colour geometry and shared split plumbing."""

from copy import deepcopy
from collections import Counter
import json
import numpy as np
import os
import pandas as pd
import pickle
import torch
from tqdm import tqdm
from grounding._vendor.lasp.eval_helper import is_explicitly_view_dependent, load_pc

from grounding.data.benchmark import scene_extents
from grounding.paths import RAW, BENCHMARK
from grounding.data.scenes import wall_geometry

SCAN_FAMILY_BASE = RAW / "referit3d"

# One annotation csv per split, so a run declares which rows it is on instead of
# having the test csv redirected under it. `dev` is the held-out dev draw written
# by scripts/prepare/build_dev_csv.py; `train` is the whole corpus, filtered to
# the train scan list below.
SPLIT_CSV = {
    "test": "{anno}_test.csv",
    "dev": "{anno}_dev.csv",
    "train": "{anno}.csv",
}

_feats_path = SCAN_FAMILY_BASE / 'feats_3d.pkl'
if os.path.exists(_feats_path):
    with open(_feats_path, 'rb') as f:
        feats = pickle.load(f)
else:
    # The ZSVG3D feats_3d.pkl (obj_ids/inst_locs/center/obj_embeds) is not
    # available -- the only copy obtainable was a 116KB per-scene category-name
    # list, which is not that format. Only the `pred` label path needs it, for
    # obj_embeds; load_pc_ below already falls back to raw scan data, so the
    # `gt` path is unaffected. Loading it at import made every entry point
    # depend on a file most of them never read.
    feats = None


def load_pc_(scan_id):
    entry = feats.get(scan_id) if isinstance(feats, dict) else None
    if isinstance(entry, dict) and 'obj_ids' in entry:
        return entry['obj_ids'], entry['inst_locs'], entry['center'], entry['obj_embeds']
    # feats_3d.pkl not in ZSVG3D format; fall back to raw scan data
    _, obj_ids, inst_locs, center, _ = load_pc(scan_id, scan_dir=SCAN_FAMILY_BASE / "scan_data")
    return obj_ids, inst_locs, center, None

def load_one_scene(scan_id, label_type):
    # Single source of truth for boxes: benchmark/boxes, shared with both trees
    # (see benchmark/boxes.py). Identical values to the instance_id_to_{name,loc}
    # arrays this used to read -- verified over 200 scans -- but read from one
    # file so all three setups provably ground against the same geometry.
    with open(os.path.join(BENCHMARK, 'boxes', '%s_bboxes.json' % scan_id)) as _f:
        _canon = json.load(_f)
    inst_labels = [o['label'] for o in _canon]
    inst_locs = np.array([o['centre'] + o['size'] for o in _canon], dtype=float)
    # Per-instance colour GMM, consumed by ColorConcept. Indexed by instance id,
    # so it is NOT filtered to obj_ids here.
    result = {
        'gt_locs': inst_locs,
        'gt_colors': json.load(open(os.path.join(
            SCAN_FAMILY_BASE, 'scan_data', 'instance_id_to_gmm_color', '%s.json' % scan_id))),
        "scan_id": scan_id,
    }
    # No point cloud on the GT path. Everything the encoders take from it is
    # cached by scripts/prepare/build_benchmark.py, exactly:
    #
    #   room_points  AtTheCorner reads x/y min and max, OnTheFloor reads z min.
    #                A 2x3 [min; max] tensor reproduces those by construction.
    #   pred_locs    load_pc's own boxes, verified equal for all 1513 scans by
    #                `build_benchmark.py --verify`. NOT the annotated array --
    #                that pairs a centroid with an extent and differs by 0.36 m.
    #                float32 is part of matching it: the values agree in double
    #                precision too, but the ternary encoder's 3-D contraction
    #                accumulates differently and 44 concepts drift by ~1e-3.
    # label_type="pred" still needs the cloud: it classifies objects from
    # obj_embeds, which no cache carries.
    if label_type == "gt":
        ext = scene_extents()[scan_id]
        result["room_points"] = np.array([ext["min"], ext["max"]], dtype=np.float32)
        result.update(wall_geometry(_canon, inst_labels))
        fg = [o for o in _canon if not o["is_background"]]
        obj_ids = [o["object_id"] for o in fg]
        pred_inst_locs = np.array([o["centre"] + o["size"] for o in fg], dtype=np.float32)
        obj_embeds = None
        result["obj_ids"] = obj_ids
        result["pred_locs"] = pred_inst_locs
        result["inst_labels"] = [inst_labels[i] for i in obj_ids]
        result["obj_embeds"] = obj_embeds
        return scan_id, result

    pcd_data = torch.load(os.path.join(SCAN_FAMILY_BASE, "scan_data", "pcd_with_global_alignment", '%s.pth'% scan_id), weights_only=False)

    points, _, _ = pcd_data[0], pcd_data[1], pcd_data[-1]

    result["room_points"] = points
    # Wall/floor geometry for AgainstTheWall. One derivation, in the package, so
    # the against-wall experiments call the same code the features are built
    # from. see grounding/data/scenes.py
    result.update(wall_geometry(_canon, inst_labels))
    info = load_pc_(scan_id)
    obj_ids, pred_inst_locs, center, obj_embeds = info
    result["obj_ids"] = obj_ids
    if label_type == "gt":
        result["pred_locs"] = pred_inst_locs # USING GT LABEL
        result["inst_labels"] = [inst_labels[i] for i in obj_ids]
    else:
        result["pred_locs"] = pred_inst_locs
        result["inst_labels"] = [inst_labels[i] for i in obj_ids]
    result["obj_embeds"] = obj_embeds
    # Dropped here: 'room_corners', 'pcds' and 'colors'. None had a consumer in
    # src/ or scripts/, and building them held a large amount of memory for
    # nothing across a train load.
    #
    # 'gt_colors' and 'wall_boxes' are both set above and both live: colour
    # feeds ColorConcept, and the wall boxes feed AgainstTheWall.
    return scan_id, result

def load_scannet(scan_ids, label_type):
    scans = {}
    for scan_id in tqdm(scan_ids):
        scan_id, result = load_one_scene(scan_id, label_type)
        scans[scan_id] = result
    return scans

class Nr3DDataset:
    def __init__(self, 
            anno_type='nr3d', 
            split="test",
            label_type="pred"
        ):
        assert anno_type in ['nr3d', 'sr3d']
        assert label_type in ['gt', 'pred']
        assert split in SPLIT_CSV, f"unknown split {split!r}"
        anno_file = SCAN_FAMILY_BASE / 'annotations/refer' / SPLIT_CSV[split].format(anno=anno_type)
        self.scan_ids = set() # scan ids in data
        self.data = [] 
        training_ids = open(SCAN_FAMILY_BASE / "annotations/splits/scannetv2_train.txt").read().splitlines()
        if split == "train":
            df = pd.read_csv(anno_file)
            for i in range(len(df)):
                mention = df.iloc[i]['mentions_target_class']
                use_color = df.iloc[i]['uses_color_lang']
                use_shape = df.iloc[i]['uses_shape_lang']
                if not mention:
                    continue
                if split == "train" and (use_color or use_shape):
                    continue
                if split == "train" and df.iloc[i]['scan_id'] not in training_ids:
                    continue
                self.scan_ids.add(df.iloc[i]['scan_id'])
                item = df.iloc[i]
                self.data.append(item)
        else:
            df = pd.read_csv(anno_file)
            for i in range(len(df)):
                self.scan_ids.add(df.iloc[i]['scan_id'])
                item = df.iloc[i]
                self.data.append(item)

        # load category file
        self.scans = load_scannet(self.scan_ids, label_type)
        
        # build unique multiple look up
        for scan_id in self.scan_ids:
            inst_labels = self.scans[scan_id]['inst_labels']
            self.scans[scan_id]['label_count'] = Counter([l for l in inst_labels])
        

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        item_id = item['stimulus_id']
        distractor_ids = list(map(int, item_id.split('-')[4:]))
        scan_id = item['scan_id']
        tgt_object_id = int(item['target_id'])
        item['instance_type']  # Preserve the required annotation-field check.
        obj_labels = deepcopy(self.scans[scan_id]['inst_labels']) # N
        tgt_object_label = obj_labels[self.scans[scan_id]['obj_ids'].index(tgt_object_id)]
        sentence = item['utterance']
        if type(item["tokens"]) == str:
            item["tokens"] = eval(item["tokens"])        
        is_view_dependent = is_explicitly_view_dependent(item['tokens'])
            
        
        is_multiple = self.scans[scan_id]['label_count'][tgt_object_label] > 1
        is_hard = self.scans[scan_id]['label_count'][tgt_object_label] > 2
        
        
        data_dict = {
            "scan_id": scan_id,
            "sentence": sentence,
            "tgt_object_id": int(tgt_object_id), # 1
            "tgt_object_label": tgt_object_label,
            "data_idx": item_id,
            "distractors": distractor_ids,
            'is_multiple': is_multiple,
            'is_view_dependent': is_view_dependent,
            'is_hard': is_hard,
        }
    
        return data_dict
