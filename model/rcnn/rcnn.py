# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
#
#
# -----------------------------------------------------------------------------
import tensorflow as tf
import tensorflow.contrib.slim as slim
from config.config_model import DefaultModelConfig
from model.cnn import CNNModel


class RCNNModel(CNNModel):
    def __init__(self, config=DefaultModelConfig()):
        super(CNNModel, self).__init__()
        self.cfg = config

    def add_losses(self, cls_score, roi_data, bbox_pred):
        # R-CNN
        # classification loss
        # cls_score = self.net.get_output('cls_score')
        label = tf.reshape(roi_data[1], [-1])
        cross_entropy = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_score, labels=label))
        tf.add_to_collection('rcnn_losses', cross_entropy)

        # bounding box regression L1 loss
        # bbox_pred = self.net.get_output('bbox_pred')
        bbox_targets = roi_data[2]
        bbox_inside_weights = roi_data[3]
        bbox_outside_weights = roi_data[4]

        smooth_l1 = self._modified_smooth_l1(
            1.0, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)
        loss_box = tf.reduce_mean(tf.reduce_sum(smooth_l1, reduction_indices=[1]))
        tf.add_to_collection('rcnn_losses', loss_box)

        tf.summary.scalar('loss_rcnn_cross_entropy', cross_entropy)
        tf.summary.scalar('loss_rcnn_box_pred', loss_box)

        return cross_entropy, loss_box

    def layer_roi_align(self, feature_maps, rois, img_shape, name='roi_align'):
        """use roi warping as roi_pooling ?? refer to
        https://github.com/DetectionTeamUCAS/Faster-RCNN_Tensorflow/blob/master/libs/networks/build_whole_network.py
        :param feature_maps:
        :param rois:
        :param img_shape:
        :param name:
        :return:
        """
        with tf.variable_scope(name) as scope:
            img_h, img_w = tf.cast(img_shape[0], tf.float32), tf.cast(img_shape[1], tf.float32)
            N = tf.shape(rois)[0]
            x1, y1, x2, y2 = tf.unstack(rois, axis=1)

            norm_x1, norm_x2 = x1 / img_w, x2 / img_w
            norm_y1, norm_y2 = y1 / img_h, y2 / img_h

            norm_rois = tf.transpose(tf.stack([norm_y1, norm_x1, norm_y2, norm_x2]), name='norm_rois')
            norm_rois = tf.stop_gradient(norm_rois)

            cropped_rois = tf.image.crop_and_resize(
                feature_maps, norm_rois, box_ind=tf.zeros(shape=[N, ], dtype=tf.int32),
                crop_size=[self.cfg.ROI_ALIGN_SIZE, self.cfg.ROI_ALIGN_SIZE], name='rois_crop_and_resize')

            roi_features = slim.max_pool2d(
                cropped_rois, [self.cfg.ROI_ALIGN_KERNEL_SIZE, self.cfg.ROI_ALIGN_KERNEL_SIZE],
                stride=self.cfg.ROI_ALIGN_KERNEL_SIZE)

            return roi_features

    def inference(self, feature_maps, roi_data, im_info, keep_prob, cfg_key=0):
        img_shape = im_info[0]
        rois = roi_data[0][:, 1:5]

        pool5 = self.layer_roi_align(feature_maps, rois, img_shape, 'pool5')

        flat, flat_depth = self.layer_flatten(tf.transpose(pool5, [0, 3, 1, 2]))
        fc6 = self.layer_fc(flat, flat_depth, 4096, name='fc6', stddev=0.01)
        drop6 = self.layer_dropout(fc6, keep_prob, name='drop6')

        fc7 = self.layer_fc(drop6, 4096, 4096, name='fc7', stddev=0.01)
        drop7 = self.layer_dropout(fc7, keep_prob, name='drop7')
        cls_score = self.layer_fc(
            drop7, 4096, self.cfg.num_classes, name='cls_score', relu=False, stddev=0.01)
        cls_prob = tf.nn.softmax(cls_score, name='cls_prob')

        bbox_pred = self.layer_fc(
            drop7, 4096, self.cfg.num_classes*4, name='bbox_pred', relu=False, stddev=0.001)

        if cfg_key == 0:
            self.add_losses(cls_score, roi_data, bbox_pred)

        return cls_prob, bbox_pred
