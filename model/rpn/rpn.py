# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
#
#
# -----------------------------------------------------------------------------
import tensorflow as tf
from config.config_model import DefaultModelConfig
from model.cnn import CNNModel
from model.rpn.anchor_target_layer import anchor_target_layer, get_anchor_target_cfg
from model.rpn.proposal_layer import proposal_layer, get_proposal_cfg
from model.rpn.proposal_target_layer import proposal_target_layer, get_proposal_target_cfg
from model.rpn.anchor_gen import generate_anchors, generate_shifted_anchors
from model.rpn.anchor_tensor_gen import generate_anchors_tensor


class RPNModel(CNNModel):
    def __init__(self, config=DefaultModelConfig()):
        super(CNNModel, self).__init__()
        self.cfg = config

    def add_losses(self, rpn_cls_score_reshape, rpn_data, rpn_bbox_pred):
        # RPN
        # classification loss
        rpn_cls_score = tf.reshape(rpn_cls_score_reshape, [-1, 2])
        rpn_label = tf.reshape(rpn_data[0], [-1])
        rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, tf.where(tf.not_equal(rpn_label, -1))), [-1, 2])
        rpn_label = tf.reshape(tf.gather(rpn_label, tf.where(tf.not_equal(rpn_label, -1))), [-1])
        rpn_cross_entropy = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_label))
        tf.add_to_collection('rpn_losses', rpn_cross_entropy)

        # bounding box regression L1 loss
        # rpn_bbox_pred = self.net.get_output('rpn_bbox_pred')
        rpn_bbox_targets = tf.transpose(rpn_data[1], [0, 2, 3, 1])
        rpn_bbox_inside_weights = tf.transpose(rpn_data[2], [0, 2, 3, 1])
        rpn_bbox_outside_weights = tf.transpose(rpn_data[3], [0, 2, 3, 1])
        # rpn_bbox_targets = rpn_data[1]
        # rpn_bbox_inside_weights = rpn_data[2]
        # rpn_bbox_outside_weights = rpn_data[3]

        rpn_smooth_l1 = self._modified_smooth_l1(
            3.0, rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights)
        rpn_loss_box = tf.reduce_mean(tf.reduce_sum(rpn_smooth_l1, reduction_indices=[1, 2, 3]))
        tf.add_to_collection('rpn_losses', rpn_loss_box)

        tf.summary.scalar('loss_rpn_cross_entropy', rpn_cross_entropy)
        tf.summary.scalar('loss_rpn_box_pred', rpn_loss_box)

        return rpn_cross_entropy, rpn_loss_box

    @staticmethod
    def layer_reshape_rpn(input_x, depth_out, name):
        with tf.variable_scope(name) as scope:
            input_shape = tf.shape(input_x)
            float_inshape = tf.cast(input_shape, tf.float32)
            if name == 'rpn_cls_prob_reshape':
                reshape_depth = float_inshape[1] / tf.cast(depth_out, tf.float32) * float_inshape[3]
            else:
                reshape_depth = float_inshape[1] * (float_inshape[3] / tf.cast(depth_out, tf.float32))

            reshape_to = [input_shape[0], int(depth_out), tf.cast(reshape_depth, tf.int32), input_shape[2]]
            reshape_ori = tf.reshape(tf.transpose(input_x, [0, 3, 1, 2]), reshape_to)
            reshape = tf.transpose(reshape_ori, [0, 2, 3, 1], name=scope.name)
            return reshape

    @staticmethod
    def layer_softmax_rpn(input_x, name):
        with tf.variable_scope(name) as scope:
            input_shape = tf.shape(input_x)
            sftmx_ori = tf.nn.softmax(tf.reshape(input_x, [-1, input_shape[3]]))
            softmax = tf.reshape(
                sftmx_ori, [-1, input_shape[1], input_shape[2], input_shape[3]], name=scope.name)
            return softmax

    def layer_anchor_target(self, rpn_cls_score, gt_boxes, im_info, name='anchor_target'):
        with tf.variable_scope(name) as scope:
            shape = tf.shape(rpn_cls_score)[1:3]
            all_anchors_t, num_anchors_t = generate_anchors_tensor(shape[0], shape[1], self.cfg)

            anchor_input = [rpn_cls_score, gt_boxes, im_info,
                            all_anchors_t, num_anchors_t, get_anchor_target_cfg(self.cfg)]
            anchor_ret_type = [tf.float32, tf.float32, tf.float32, tf.float32]
            rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = tf.py_func(
                anchor_target_layer, anchor_input, anchor_ret_type, name=scope.name)

            rpn_labels = tf.convert_to_tensor(tf.cast(rpn_labels, tf.int32), name='rpn_labels')
            rpn_bbox_targets = tf.convert_to_tensor(
                rpn_bbox_targets, name='rpn_bbox_targets')
            rpn_bbox_inside_weights = tf.convert_to_tensor(
                rpn_bbox_inside_weights, name='rpn_bbox_inside_weights')
            rpn_bbox_outside_weights = tf.convert_to_tensor(
                rpn_bbox_outside_weights, name='rpn_bbox_outside_weights')

            return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights

    def layer_proposal(self, rpn_cls_prob_reshape, rpn_bbox_pred, im_info, cfg_key=0, name='proposal'):
        with tf.variable_scope(name) as scope:
            shape = tf.shape(rpn_cls_prob_reshape)[1:3]
            all_anchors_t, num_anchors_t = generate_anchors_tensor(shape[0], shape[1], self.cfg)

            proposal_input = [rpn_cls_prob_reshape, rpn_bbox_pred, im_info, cfg_key,
                              all_anchors_t, num_anchors_t, get_proposal_cfg(self.cfg)]
            proposal_ret_type = [tf.float32]
            proposal = tf.reshape(
                tf.py_func(proposal_layer, proposal_input, proposal_ret_type),
                [-1, 5], name=scope.name)
            return proposal

    def layer_proposal_target(self, rpn_rois, gt_boxes, classes, name):
        with tf.variable_scope(name) as scope:
            proposal_input = [rpn_rois, gt_boxes, classes, get_proposal_target_cfg(self.cfg)]
            proposal_ret_type = [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32]
            rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = tf.py_func(
                proposal_target_layer, proposal_input, proposal_ret_type, name=scope.name)

            rois = tf.reshape(rois, [-1, 5], name='rois')
            labels = tf.convert_to_tensor(tf.cast(labels, tf.int32), name='labels')
            bbox_targets = tf.convert_to_tensor(
                bbox_targets, name='bbox_targets')
            bbox_inside_weights = tf.convert_to_tensor(
                bbox_inside_weights, name='bbox_inside_weights')
            bbox_outside_weights = tf.convert_to_tensor(
                bbox_outside_weights, name='bbox_outside_weights')

            return rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights

    def inference(self, feature_maps, gt_boxes, im_info, cfg_key=0):
        len_anchor = len(self.cfg.anchor_scales)

        rpn_conv_3x3 = self.layer_conv(
            feature_maps, 3, 1, 512, name='rpn_conv/3x3', stddev=0.01)
        rpn_cls_score = self.layer_conv(
            rpn_conv_3x3, 1, 1, len_anchor*3*2, 'VALID', name='rpn_cls_score',
            relu=False, stddev=0.01)

        # rpn_data = self.layer_anchor_target(rpn_cls_score, gt_boxes, im_info, name='rpn-data')

        rpn_bbox_pred = self.layer_conv(
            rpn_conv_3x3, 1, 1, len_anchor*3*4, 'VALID', name='rpn_bbox_pred',
            relu=False, stddev=0.01)

        rpn_cls_score_reshape = self.layer_reshape_rpn(
            rpn_cls_score, 2, name='rpn_cls_score_reshape')
        rpn_cls_prob = self.layer_softmax_rpn(
            rpn_cls_score_reshape, name='rpn_cls_prob')

        rpn_cls_prob_reshape = self.layer_reshape_rpn(
            rpn_cls_prob, len_anchor*3*2, name='rpn_cls_prob_reshape')

        rpn_rois = self.layer_proposal(
            rpn_cls_prob_reshape, rpn_bbox_pred, im_info, cfg_key=cfg_key, name='rpn_rois')

        if cfg_key == 0:
            rpn_data = self.layer_anchor_target(rpn_cls_score, gt_boxes, im_info, name='rpn-data')
            roi_data = self.layer_proposal_target(rpn_rois, gt_boxes, self.cfg.num_classes, name='roi-data')

            self.add_losses(rpn_cls_score_reshape, rpn_data, rpn_bbox_pred)

            return roi_data
        else:
            return [rpn_rois]
