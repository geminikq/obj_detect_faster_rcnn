# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
#
#
# -----------------------------------------------------------------------------
import numpy as np
from model.utils.bbox import bbox_overlaps


def get_ap_image_class(bbox_pred, gt_boxes, class_ind):
    scores = bbox_pred[:, -1]
    sorted_ind = np.argsort(-scores)

    bbox_pred = bbox_pred[sorted_ind, :]
    gt_boxes = gt_boxes[sorted_ind, :]

    nd = len(sorted_ind)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        overlaps = bbox_overlaps(bbox_pred[d], gt_boxes[d])
        ovmax = np.max(overlaps)
        jmax = np.argmax(overlaps)



