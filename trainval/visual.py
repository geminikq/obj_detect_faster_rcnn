# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
#
#
# -----------------------------------------------------------------------------
import numpy as np
import cv2
from model.utils.bbox import bbox_overlaps
from model.utils.bbox_trans import bbox_transform_inv, clip_boxes
from model.utils.nms import nms


def _get_bboxes_predicted(rois, bbox_deltas, im_info):
    pred_bboxes = bbox_transform_inv(rois, bbox_deltas)
    pred_bboxes = clip_boxes(pred_bboxes, im_info)
    return pred_bboxes


def _get_max_overlaps(pred_bboxes, gt_boxes, threshold):
    overlaps = bbox_overlaps(
        np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float32),
        np.ascontiguousarray(np.reshape(pred_bboxes, (-1, 4)), dtype=np.float32))
    num_classes = int(pred_bboxes.shape[-1] / 4)
    overlaps = np.reshape(overlaps, (-1, num_classes))

    max_overlaps = overlaps.max(axis=1)
    ind_overlaps = overlaps.argmax(axis=1)
    cls_max_overlaps = max_overlaps.argmax()

    inds = np.where(max_overlaps >= threshold)[0]

    return ind_overlaps, inds


def recover_bboxes(rois, bbox_deltas, im_info, gt_boxes, threshold):
    pred_bboxes = _get_bboxes_predicted(rois, bbox_deltas, im_info[0])
    return _get_max_overlaps(pred_bboxes, gt_boxes, threshold)


def _get_detect_results(classes, scores, bboxes):
    detect_classes = [[]]
    # detect_classes.append([])
    for idx, cls in enumerate(classes[1:]):
        idx += 1
        cls_boxes = bboxes[:, 4*idx:4*(idx+1)]
        cls_scores = scores[:, idx]
        dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, 0.3)
        dets = dets[keep, :]
        detect_classes.append(dets)
        # draw_bboxes(image, bboxes, 0.5)

    return detect_classes


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


def get_bboxes_classes_probs(image, scores, rois, bbox_deltas, im_info, gt_boxes, classes):
    pred_bboxes = _get_bboxes_predicted(rois, bbox_deltas, im_info[0])

    # ind_overlaps, inds = _get_max_overlaps(pred_bboxes, gt_boxes, 0.5)
    detects = _get_detect_results(classes, scores, pred_bboxes)

    for idx, dets in enumerate(detects):
        draw_bboxes(image, dets, classes[idx], 0.8)

    draw_gt_bboxes(image, gt_boxes, classes)
    # cls_pred = np.argmax(scores[ind_overlaps[inds]], axis=1)
    # bboxes = pred_bboxes[ind_overlaps[inds]]

    return image
