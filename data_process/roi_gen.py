# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
#
#
# ------------------------------------------------------------------------------
import numpy as np
import numpy.random as npr
import cv2
from config.config_input import ImageInput
from config.config_model import DefaultModelConfig
from data_process.blob import prep_im_for_blob, im_list_to_blob


class ROIGenerator(object):
    """Fast R-CNN data layer used for training."""

    def __init__(self, roidb, num_classes, config=ImageInput()):
        """Set the roidb to be used by this layer during training."""
        self.cfg = config

        self._roidb = roidb
        self._num_classes = num_classes
        self._shuffle_roidb_inds()

    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        self._perm = np.random.permutation(np.arange(len(self._roidb)))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        #
        # if cfg.TRAIN.HAS_RPN:
        #     if self._cur + cfg.TRAIN.IMS_PER_BATCH >= len(self._roidb):
        #         self._shuffle_roidb_inds()
        #
        #     db_inds = self._perm[self._cur:self._cur + cfg.TRAIN.IMS_PER_BATCH]
        #     self._cur += cfg.TRAIN.IMS_PER_BATCH
        # else:
        #     # sample images
        #     db_inds = np.zeros((cfg.TRAIN.IMS_PER_BATCH), dtype=np.int32)
        #     i = 0
        #     while (i < cfg.TRAIN.IMS_PER_BATCH):
        #         ind = self._perm[self._cur]
        #         num_objs = self._roidb[ind]['boxes'].shape[0]
        #         if num_objs != 0:
        #             db_inds[i] = ind
        #             i += 1
        #
        #         self._cur += 1
        #         if self._cur >= len(self._roidb):
        #             self._shuffle_roidb_inds()

        if self._cur + self.cfg.image_per_batch >= len(self._roidb):
            self._shuffle_roidb_inds()

        db_inds = self._perm[self._cur:self._cur + self.cfg.image_per_batch]
        self._cur += self.cfg.image_per_batch

        return db_inds

    def get_minibatch(self, roidb, num_classes):
        """Given a roidb, construct a minibatch sampled from it."""
        num_images = len(roidb)
        # Sample random scales to use for each image in this batch
        random_scale_inds = npr.randint(0, high=len(self.cfg.train_scale),
                                        size=num_images)
        assert (self.cfg.batch_size_rois % num_images == 0), \
            'num_images ({}) must divide BATCH_SIZE ({})'. \
                format(num_images, self.cfg.batch_size_rois)
        rois_per_image = self.cfg.batch_size_rois / num_images
        fg_rois_per_image = np.round(self.cfg.batch_fg_fraction * rois_per_image)

        # Get the input image blob, formatted for caffe
        im_blob, im_scales = self._get_image_blob(roidb, random_scale_inds)

        blobs = {'data': im_blob}

        # if cfg.TRAIN.HAS_RPN:
        assert len(im_scales) == 1, "Single batch only"
        assert len(roidb) == 1, "Single batch only"
        # gt boxes: (x1, y1, x2, y2, cls)
        gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
        gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
        gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :] * im_scales[0]
        gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds]
        blobs['gt_boxes'] = gt_boxes
        blobs['im_info'] = np.array(
            [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]],
            dtype=np.float32)
        blobs['image'] = roidb[0]['image']

        # else: # not using RPN
        #     # Now, build the region of interest and label blobs
        #     rois_blob = np.zeros((0, 5), dtype=np.float32)
        #     labels_blob = np.zeros((0), dtype=np.float32)
        #     bbox_targets_blob = np.zeros((0, 4 * num_classes), dtype=np.float32)
        #     bbox_inside_blob = np.zeros(bbox_targets_blob.shape, dtype=np.float32)
        #     # all_overlaps = []
        #     for im_i in range(num_images):
        #         labels, overlaps, im_rois, bbox_targets, bbox_inside_weights = _sample_rois(
        #             roidb[im_i], fg_rois_per_image, rois_per_image, num_classes, cfg)
        #
        #         # Add to RoIs blob
        #         rois = _project_im_rois(im_rois, im_scales[im_i])
        #         batch_ind = im_i * np.ones((rois.shape[0], 1))
        #         rois_blob_this_image = np.hstack((batch_ind, rois))
        #         rois_blob = np.vstack((rois_blob, rois_blob_this_image))
        #
        #         # Add to labels, bbox targets, and bbox loss blobs
        #         labels_blob = np.hstack((labels_blob, labels))
        #         bbox_targets_blob = np.vstack((bbox_targets_blob, bbox_targets))
        #         bbox_inside_blob = np.vstack((bbox_inside_blob, bbox_inside_weights))
        #         # all_overlaps = np.hstack((all_overlaps, overlaps))
        #
        #     # For debug visualizations
        #     # _vis_minibatch(im_blob, rois_blob, labels_blob, all_overlaps)
        #
        #     blobs['rois'] = rois_blob
        #     blobs['labels'] = labels_blob
        #
        #     if cfg.TRAIN.BBOX_REG:
        #         blobs['bbox_targets'] = bbox_targets_blob
        #         blobs['bbox_inside_weights'] = bbox_inside_blob
        #         blobs['bbox_outside_weights'] = \
        #             np.array(bbox_inside_blob > 0).astype(np.float32)

        return blobs

    def _get_image_blob(self, roidb, scale_inds):
        """Builds an input blob from the images in the roidb at the specified
        scales.
        """
        num_images = len(roidb)
        processed_ims = []
        im_scales = []
        for i in range(num_images):
            im = cv2.imread(roidb[i]['image'])
            if roidb[i]['flipped']:
                im = im[:, ::-1, :]
            target_size = self.cfg.train_scale[scale_inds[i]]
            im, im_scale = prep_im_for_blob(
                im, np.array(self.cfg.pixel_means), target_size, self.cfg.train_max_size)
            im_scales.append(im_scale)
            processed_ims.append(im)

        # Create a blob to hold the input images
        blob = im_list_to_blob(processed_ims)

        return blob, im_scales

    def _get_next_minibatch(self):
        """Return the blobs to be used for the next minibatch.

        If cfg.TRAIN.USE_PREFETCH is True, then blobs will be computed in a
        separate process and made available through self._blob_queue.
        """
        db_inds = self._get_next_minibatch_inds()
        minibatch_db = [self._roidb[i] for i in db_inds]
        return self.get_minibatch(minibatch_db, self._num_classes)

    def forward(self):
        """Get blobs and copy them into this layer's top blob vector."""
        blobs = self._get_next_minibatch()
        return blobs

    def filter_roidb(self, cfg=DefaultModelConfig()):
        """Remove roidb entries that have no usable RoIs."""

        def is_valid(entry):
            # Valid images have:
            #   (1) At least one foreground RoI OR
            #   (2) At least one background RoI
            overlaps = entry['max_overlaps']
            # find boxes with sufficient overlap
            fg_inds = np.where(overlaps >= cfg.FG_THRESH)[0]
            # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
            bg_inds = np.where((overlaps < cfg.BG_THRESH_HI) &
                               (overlaps >= cfg.BG_THRESH_LO))[0]
            # image is only valid if such boxes exist
            valid = len(fg_inds) > 0 or len(bg_inds) > 0
            return valid

        num = len(self._roidb)
        filtered_roidb = [entry for entry in self._roidb if is_valid(entry)]
        num_after = len(filtered_roidb)
        print('Filtered {} roidb entries: {} -> {}'.format(num - num_after, num, num_after))

        self._roidb = filtered_roidb
        return filtered_roidb, num_after
