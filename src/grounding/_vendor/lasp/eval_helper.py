# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import os
import json


def get_box3d_min_max(corner):
    ''' Compute min and max coordinates for 3D bounding box
        Note: only for axis-aligned bounding boxes

    Input:
        corners: numpy array (8,3), assume up direction is Z (batch of N samples)
    Output:
        box_min_max: an array for min and max coordinates of 3D bounding box IoU

    '''

    min_coord = corner.min(axis=0)
    max_coord = corner.max(axis=0)
    x_min, x_max = min_coord[0], max_coord[0]
    y_min, y_max = min_coord[1], max_coord[1]
    z_min, z_max = min_coord[2], max_coord[2]
    
    return x_min, x_max, y_min, y_max, z_min, z_max

def box3d_iou(corners1, corners2):
    ''' Compute 3D bounding box IoU.

    Input:
        corners1: numpy array (8,3), assume up direction is Z
        corners2: numpy array (8,3), assume up direction is Z
    Output:
        iou: 3D bounding box IoU

    '''

    x_min_1, x_max_1, y_min_1, y_max_1, z_min_1, z_max_1 = get_box3d_min_max(corners1)
    x_min_2, x_max_2, y_min_2, y_max_2, z_min_2, z_max_2 = get_box3d_min_max(corners2)
    xA = np.maximum(x_min_1, x_min_2)
    yA = np.maximum(y_min_1, y_min_2)
    zA = np.maximum(z_min_1, z_min_2)
    xB = np.minimum(x_max_1, x_max_2)
    yB = np.minimum(y_max_1, y_max_2)
    zB = np.minimum(z_max_1, z_max_2)
    inter_vol = np.maximum((xB - xA), 0) * np.maximum((yB - yA), 0) * np.maximum((zB - zA), 0)
    box_vol_1 = (x_max_1 - x_min_1) * (y_max_1 - y_min_1) * (z_max_1 - z_min_1)
    box_vol_2 = (x_max_2 - x_min_2) * (y_max_2 - y_min_2) * (z_max_2 - z_min_2)
    iou = inter_vol / (box_vol_1 + box_vol_2 - inter_vol + 1e-8)

    return iou

def eval_ref_one_sample(pred_bbox, gt_bbox):
    """ Evaluate one reference prediction

    Args:
        pred_bbox: 8 corners of prediction bounding box, (8, 3)
        gt_bbox: 8 corners of ground truth bounding box, (8, 3)
    Returns:
        iou: intersection over union score
    """

    iou = box3d_iou(pred_bbox, gt_bbox)

    return iou

def construct_bbox_corners(center, box_size):
    sx, sy, sz = box_size
    x_corners = [sx/2, sx/2, -sx/2, -sx/2, sx/2, sx/2, -sx/2, -sx/2]
    y_corners = [sy/2, -sy/2, -sy/2, sy/2, sy/2, -sy/2, -sy/2, sy/2]
    z_corners = [sz/2, sz/2, sz/2, sz/2, -sz/2, -sz/2, -sz/2, -sz/2]
    corners_3d = np.vstack([x_corners, y_corners, z_corners])
    corners_3d[0,:] = corners_3d[0,:] + center[0];
    corners_3d[1,:] = corners_3d[1,:] + center[1];
    corners_3d[2,:] = corners_3d[2,:] + center[2];
    corners_3d = np.transpose(corners_3d)

    return corners_3d

def convert_pc_to_box(obj_pc):
    xmin = np.min(obj_pc[:,0])
    ymin = np.min(obj_pc[:,1])
    zmin = np.min(obj_pc[:,2])
    xmax = np.max(obj_pc[:,0])
    ymax = np.max(obj_pc[:,1])
    zmax = np.max(obj_pc[:,2])
    center = [(xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2]
    box_size = [xmax-xmin, ymax-ymin, zmax-zmin]
    return center, box_size

def is_explicitly_view_dependent(tokens):
    """
    :return: a boolean mask
    """
    target_words = {'front', 'behind', 'back', 'right', 'left', 'facing', 'leftmost', 'rightmost',
                    'looking', 'across'}
    for token in tokens:
        if token in target_words:
            return True
    return False


def get_all_categories(json_obj):
    appearances = set()
    appearances.add(json_obj["category"])
    if "relations" in json_obj:
        for relation in json_obj["relations"]:
            if "anchors" in relation:
                relation["objects"] = relation["anchors"]
            for obj in relation["objects"]:
                appearances |= get_all_categories(obj)
    return appearances

def get_all_relations(json_obj):
    relations = set()
    if "relations" in json_obj:
        for relation in json_obj["relations"]:
            relations.add(relation["relation_name"])
            if "anchors" in relation:
                relation["objects"] = relation["anchors"]
            for obj in relation["objects"]:
                relations |= get_all_relations(obj)
    return relations


def load_pc(scan_id, keep_background = False, scan_dir = 'data/referit3d/scan_data'):
    pcds, colors, _, instance_labels = torch.load(
        os.path.join(scan_dir, 'pcd_with_global_alignment', '%s.pth' % scan_id), weights_only=False)
    obj_labels = json.load(open(os.path.join(scan_dir, 'instance_id_to_name', '%s.json' % scan_id)))

    batch_labels = []
    inst_locs = []
    obj_ids = []
    for i, obj_label in enumerate(obj_labels):
        if (not keep_background) and obj_label in ['wall', 'floor', 'ceiling']:
            continue
        mask = instance_labels == i
        assert np.sum(mask) > 0, 'scan: %s, obj %d' % (scan_id, i)
        obj_pcd = pcds[mask]

        obj_center = (obj_pcd[:, :3].max(0) + obj_pcd[:, :3].min(0)) / 2
        obj_size = obj_pcd[:, :3].max(0) - obj_pcd[:, :3].min(0)
        inst_locs.append(np.concatenate([obj_center, obj_size], 0))


        batch_labels.append(obj_label)
        obj_ids.append(i)

    batch_pcds = None
    center = (pcds.max(0) + pcds.min(0)) / 2

    return batch_labels, obj_ids, inst_locs, center, batch_pcds
