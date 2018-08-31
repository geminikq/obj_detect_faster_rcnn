# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
#
#
# -----------------------------------------------------------------------------
import tensorflow as tf
import numpy as np
import cv2
import os.path
from config.config_model import DefaultModelConfig
from config.config_input import PascalVOC2007Input

from model.feat_extract.vgg16 import VGGFeatureExtractor
from model.rpn.rpn import RPNModel
from model.rcnn.rcnn import RCNNModel
from trainval.visual import get_bboxes_classes_probs


class VGGNetFasterRCNNModel(object):
    TrainEnd2End = 0
    Evaluation = 1

    def __init__(self, config=DefaultModelConfig()):
        self.extractor = VGGFeatureExtractor(config.weights_file)
        self.rpn = RPNModel(config)
        self.rcnn = RCNNModel(config)

        self.weights_pre_trained = config.weights_file

        self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        self.im_info = tf.placeholder(tf.float32, shape=[None, 3])
        self.gt_boxes = tf.placeholder(tf.float32, shape=[None, 5])
        self.keep_prob = tf.placeholder(tf.float32)

    def load_weights_pre_trained(self, sess):
        data_dict = np.load(self.weights_pre_trained, encoding='latin1').item()
        for key in data_dict:
            with tf.variable_scope(key, reuse=True):
                for subkey in data_dict[key]:
                    try:
                        var = tf.get_variable(subkey)
                        sess.run(var.assign(data_dict[key][subkey]))
                        print("assign pretrain model " + subkey + " to " + key)
                    except ValueError:
                        print("ignore " + key)
                        # if not ignore_missing:
                        #     raise

    def set_train_start_point(self, sess, saver, ckpt_file):
        if ckpt_file is not None and os.path.exists(ckpt_file):
            step = int(ckpt_file.split('.')[-2].split('_')[-1])
            print('load ckpt and start from iter: {}'.format(step))
            saver.restore(sess, ckpt_file)
            return step
        else:
            self.load_weights_pre_trained(sess)
            return 0

    def visual_detects(self, image, cls_prob, rois, bbox_pred):
        bbox_input = [
            image, cls_prob, rois, bbox_pred,
            self.im_info, self.gt_boxes, PascalVOC2007Input().classes]
        result = tf.py_func(get_bboxes_classes_probs, bbox_input, [tf.float32])
        tf.summary.image('bbox_image', result)

    def inference(self, trainval):

        image = self.data[:, :, :, ::-1]
        # tf.summary.image('image', self.data[:, :, :, ::-1])

        feature_maps = self.extractor.inference(self.data)

        roi_data = self.rpn.inference(
            feature_maps, self.gt_boxes, self.im_info, cfg_key=trainval)

        cls_prob, bbox_pred = self.rcnn.inference(
            feature_maps, roi_data, self.im_info, self.keep_prob, cfg_key=trainval)

        self.visual_detects(image[0], cls_prob, roi_data[0][:, 1:5], bbox_pred)

        return cls_prob, bbox_pred, roi_data
