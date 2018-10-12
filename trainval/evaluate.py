# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
#
#
# -----------------------------------------------------------------------------
import numpy as np
from model.utils.bbox import bbox_overlaps
from model.utils.nms import nms
from model.utils.bbox_trans import bbox_transform_inv, clip_boxes


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


def get_detect_results(classes, scores, rois, bbox_deltas, im_info):
    bboxes = _get_bboxes_predicted(rois, bbox_deltas, im_info[0])
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


def _get_bboxes_by_class(all_bboxes, cls_inds):
    bboxes = []
    npos = 0
    for gt_bboxes_per_img in all_bboxes:
        bboxes_per_img = []
        for bbox in gt_bboxes_per_img:
            if bbox[4] == cls_inds:
                bboxes_per_img.append(bbox[:4])
                npos += 1
        bboxes.append(bboxes_per_img)
    return bboxes, npos


def _calculate_voc_ap(rec, prec):
    # 11 point metric
    ap = 0.
    for t in np.arange(0., 1.1, 0.1):
        if np.sum(rec >= t) == 0:
            p = 0
        else:
            p = np.max(prec[rec >= t])
        ap = ap + p / 11.

    return ap


def _evaluate_voc_class(bboxes_pred, gt_bboxes, num_bboxes, ovthresh=0.5):
    # flat bbox & calculate
    bboxes_pred_flat = []
    image_ids = []
    for img_idx in range(len(bboxes_pred)):
        for bbox in bboxes_pred[img_idx]:
            bboxes_pred_flat.append(bbox)
            image_ids.append(img_idx)

    bboxes_pred_flat = np.array(bboxes_pred_flat)
    image_ids = np.array(image_ids)
    bboxes_pred = np.array(bboxes_pred)

    scores = bboxes_pred_flat[:, -1]
    sorted_ind = np.argsort(-scores)

    bbox_pred = bboxes_pred_flat[:, :4][sorted_ind, :]
    image_ids = image_ids[sorted_ind]
    # gt_boxes = gt_boxes[sorted_ind, :]

    nd = len(sorted_ind)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        bbxgt = np.array(gt_bboxes[image_ids[d]])
        dets = [False] * len(bbxgt)
        ovmax = -np.inf
        if len(bbxgt) > 0:
            overlaps = bbox_overlaps(bbxgt, np.array([bbox_pred[d]]))
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not dets[jmax]:
                tp[d] = 1.
                dets[jmax] = 1
            else:
                fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(num_bboxes)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = _calculate_voc_ap(rec, prec)

    return rec, prec, ap


def detects_evaluation(bboxes_pred, gt_bboxes, all_classes):
    aps = []
    for cls_inds in range(len(all_classes)):
        if cls_inds == 0:
            continue
        cls_bboxes, num_bboxes = _get_bboxes_by_class(gt_bboxes, cls_inds)
        rec, prec, ap = _evaluate_voc_class(bboxes_pred[cls_inds], cls_bboxes, num_bboxes)
        aps += [ap]
        print('AP for {} = {:.4f}'.format(all_classes[cls_inds], ap))
    print('Mean AP = {:.4f}'.format(np.mean(aps)))

