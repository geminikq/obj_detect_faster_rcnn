# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
#
#
# -----------------------------------------------------------------------------
import tensorflow as tf
from config.config_model import DefaultModelConfig
from model.rpn.anchor_gen import generate_anchors


def generate_anchors_tensor(height, width, cfg=DefaultModelConfig()):

    # since anchors per feature pixel is predefined, so we don't need make it tf
    anchors = generate_anchors(cfg.anchor_base_size, cfg.anchor_ratios, cfg.anchor_scales)
    anchors_t = tf.convert_to_tensor(anchors, tf.float32)

    num_anchors_t = tf.convert_to_tensor(anchors.shape[0], tf.int32)
    # feature_maps_shape = tf.shape(feature_maps)
    all_anchors_t = generate_shifted_anchors_tensor(
        height, width, anchors_t, cfg.feature_stride)

    return all_anchors_t, num_anchors_t


def generate_shifted_anchors_tensor(height, width, anchors, feature_stride):
    shift_x = tf.cast(tf.range(width) * feature_stride, tf.float32)
    shift_y = tf.cast(tf.range(height) * feature_stride, tf.float32)
    shifted_x, shifted_y = tf.meshgrid(shift_x, shift_y)
    shifts = tf.transpose(tf.concat(
        [tf.reshape(shifted_x, (1, -1)), tf.reshape(shifted_y, (1, -1)),
         tf.reshape(shifted_x, (1, -1)), tf.reshape(shifted_y, (1, -1))], axis=0))
    shifts = tf.reshape(shifts, [-1, 4])
    A = anchors.get_shape().as_list()[0]
    # K = shifts.get_shape().as_list()[0]

    all_anchors = \
        tf.reshape(anchors, (1, A, 4)) + \
        tf.transpose(tf.reshape(shifts, (1, -1, 4)), [1, 0, 2])

    all_anchors = tf.reshape(all_anchors, (-1, 4))

    return all_anchors
