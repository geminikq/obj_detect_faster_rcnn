# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
#
#
# -----------------------------------------------------------------------------
import numpy as np
import cv2
from model.utils.bbox import bbox_overlaps
from trainval.evaluate import get_detect_results


def draw_bboxes(image, bboxes, name, threshold):
    if len(bboxes) == 0:
        return
    inds = np.where(bboxes[:, -1] >= threshold)[0]
    if len(inds) == 0:
        return

    color_digit = (0, 255, 0)
    for i in inds:
        bbox = bboxes[i, :4]
        score = bboxes[i, -1]

        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color_digit, 1)
        text = name if isinstance(name, str) else name.decode()
        text = text + ': ' + str(score)
        cv2.putText(image, text, (bbox[0], bbox[3]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color_digit, 2)


def draw_gt_bboxes(image, gt_boxes, classes):
    color_gt = (255, 255, 0)
    for box in gt_boxes:
        x1, y1, x2, y2 = box[:4]
        cv2.rectangle(image, (x1, y1), (x2, y2), color_gt, 1)
        name = classes[int(box[4])]
        name = name if isinstance(name, str) else name.decode()
        cv2.putText(image, name, (x1, y2), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color_gt, 2)


def draw_bboxes_classes_probs(image, scores, rois, bbox_deltas, im_info, gt_boxes, classes):
    # pred_bboxes = _get_bboxes_predicted(rois, bbox_deltas, im_info[0])

    # ind_overlaps, inds = _get_max_overlaps(pred_bboxes, gt_boxes, 0.5)
    detects = get_detect_results(classes, scores, rois, bbox_deltas, im_info)

    for idx, dets in enumerate(detects):
        draw_bboxes(image, dets, classes[idx], 0.8)

    draw_gt_bboxes(image, gt_boxes, classes)
    # cls_pred = np.argmax(scores[ind_overlaps[inds]], axis=1)
    # bboxes = pred_bboxes[ind_overlaps[inds]]

    return image
