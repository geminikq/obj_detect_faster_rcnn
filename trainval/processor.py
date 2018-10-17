# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
#
#
# -----------------------------------------------------------------------------
import tensorflow as tf
import numpy as np
import time
import cv2
from skimage import io
from datetime import datetime
import _pickle as cPickle
import os

from config.config_input import ImageInput
from config.config_model import DefaultModelConfig
from config.config_train import DefaultTrainConfig, DefaultTestConfig

from data_process.data_reader import DataReader
from model.VGGnet_faster_rcnn import VGGNetFasterRCNNModel
from trainval.trainer import Trainer
from trainval.visual import draw_bboxes_classes_probs
from trainval.evaluate import get_detect_results, detects_evaluation


class Processor(object):
    def __init__(self):
        pass

    @staticmethod
    def train_once(trainer):
        loss = trainer.loss()
        train_op = trainer.train(loss)
        return loss, train_op

    def train_loop(self,
                   data=ImageInput(),
                   model=DefaultModelConfig(),
                   train=DefaultTrainConfig()):
        # prepare data
        reader = DataReader(data)
        roidb = reader.prepare_roidb()
        _, epe = roidb.filter_roidb(model)

        # build model
        model = VGGNetFasterRCNNModel(model)
        predict = model.inference(model.TrainEnd2End)
        # result = model.inference(model.Evaluation)

        # setup trainer
        trainer = Trainer(train)
        if epe != trainer.cfg.examples_per_epoch:
            print('set examples/epoch to {}'.format(epe))
            trainer.cfg.examples_per_epoch = epe
        if len(data.train_set.split('_')) > 1:
            name = data.train_set.split('_')[0]
            print('set output name to {}'.format(name))
            trainer.cfg.name = name

        loss, train_op = self.train_once(trainer)

        writer = tf.summary.FileWriter(trainer.cfg.train_dir)
        saver = tf.train.Saver()
        restorer = tf.train.Saver(write_version=tf.train.SaverDef.V1) \
            if trainer.cfg.restore_type == 'ckpt_v1' else saver

        max_iters = trainer.cfg.max_steps

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # model.load_weights_pre_trained(sess)
            start_step = model.set_train_start_point(sess, restorer, trainer.cfg.ckpt_file)

            writer.add_graph(sess.graph)
            merged = tf.summary.merge_all()

            for i in range(start_step, max_iters):
                start = time.time()

                blob = roidb.forward()
                feed = {model.data: blob['data'], model.gt_boxes: blob['gt_boxes'],
                        model.im_info: blob['im_info'], model.keep_prob: 0.5}

                train_loss, train_pred, _, summary = sess.run(
                    [loss, predict, train_op, merged], feed_dict=feed)

                end = time.time()
                duration = end - start

                format_str = '%s: step %d, loss = %.2f (%.3f sec/batch)'
                print(format_str % (datetime.now(), i, train_loss, duration))

                if i % 10 == 0:
                    writer.add_summary(summary, i)

                if i % (trainer.cfg.save_frequency / 10) == 0 and i > 0:
                    file_name = 'train_iter_' + str(i) + '.ckpt'
                    save_path = trainer.cfg.ckpt_path_base + file_name
                    saver.save(sess, save_path)
                    print('save checkpoint to : {}'.format(save_path))

    def test_loop(self,
                  data=ImageInput(),
                  model_cfg=DefaultModelConfig(),
                  test_cfg=DefaultTestConfig()):
        reader = DataReader(data)
        roidb = reader.prepare_roidb()
        _, epe = roidb.filter_roidb(model_cfg)

        # build model
        model = VGGNetFasterRCNNModel(model_cfg)
        pred_op = model.inference(model.Evaluation)

        saver = tf.train.Saver(write_version=tf.train.SaverDef.V1) \
            if test_cfg.restore_type == 'ckpt_v1' else tf.train.Saver()

        # epe = 20
        max_iters = epe

        all_boxes = [[[] for _ in range(epe)]
                     for _ in range(model_cfg.num_classes)]
        all_gt_boxes = [[] for _ in range(epe)]

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # model.set_train_start_point(sess, saver, weights_file_path)
            saver.restore(sess, test_cfg.ckpt_file)
            # saver.restore(sess, tf.train.latest_checkpoint(weights_file_path))

            for i in range(max_iters):
                start = time.time()

                blob = roidb.forward()

                scale = blob['im_info'][0][-1]
                image = cv2.imread(blob['image'])
                image = cv2.resize(image, None, None, scale, scale, interpolation=cv2.INTER_LINEAR)

                feed = {model.data: blob['data'], model.gt_boxes: blob['gt_boxes'],
                        model.im_info: blob['im_info'], model.keep_prob: 1.0}

                cls_prob, bbox_pred, rois = sess.run(pred_op, feed_dict=feed)

                dets = get_detect_results(
                    data.classes, cls_prob, np.array(rois)[0][:, 1:5], bbox_pred, blob['im_info'])

                for i_cls in range(1, len(dets)):
                    all_boxes[i_cls][i] = dets[i_cls]
                all_gt_boxes[i] = blob['gt_boxes']

                end = time.time()
                duration = end - start

                format_str = '%s: step %d, (%.3f sec/batch)'
                print(format_str % (datetime.now(), i, duration))

                if test_cfg.visual_while_test is True:
                    image = draw_bboxes_classes_probs(
                        image, cls_prob, np.array(rois)[0][:, 1:5], bbox_pred,
                        blob['im_info'], blob['gt_boxes'], data.classes)

                    # image = image / blob['im_info'][0][-1]
                    # scale = 1.0 / blob['im_info'][0][-1]
                    # image = cv2.resize(image, None, None, scale, scale, interpolation=cv2.INTER_LINEAR)
                    # image += data.pixel_means
                    cv2.imshow('result', image)
                    cv2.waitKey()

        det_file = os.path.join(data.cache_path, data.name + '_detections.pkl')
        with open(det_file, 'wb') as f:
            cPickle.dump(all_boxes, f, protocol=-1)
        print('wrote detections results to {}'.format(det_file))

        detects_evaluation(all_boxes, all_gt_boxes, data.classes)

    def demo_loop(self,
                  data=ImageInput(),
                  model_cfg=DefaultModelConfig(),
                  weights_file_path=None):

        reader = DataReader(data)
        roidb = reader.prepare_roidb()
        _, epe = roidb.filter_roidb(model_cfg)

        # build model
        model = VGGNetFasterRCNNModel(model_cfg)
        pred_op = model.inference(model.Evaluation)

        saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)
        # saver = tf.train.Saver()

        max_iters = epe
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # model.set_train_start_point(sess, saver, weights_file_path)
            saver.restore(sess, weights_file_path)
            # saver.restore(sess, tf.train.latest_checkpoint(weights_file_path))

            for i in range(max_iters):
                start = time.time()

                blob = roidb.forward()

                scale = blob['im_info'][0][-1]
                image = cv2.imread(blob['image'])
                image = cv2.resize(image, None, None, scale, scale, interpolation=cv2.INTER_LINEAR)

                feed = {model.data: blob['data'], model.gt_boxes: blob['gt_boxes'],
                        model.im_info: blob['im_info'], model.keep_prob: 1.0}

                cls_prob, bbox_pred, rois = sess.run(pred_op, feed_dict=feed)

                dets = get_detect_results(
                    data.classes, cls_prob, np.array(rois)[0][:, 1:5], bbox_pred, blob['im_info'])

                end = time.time()
                duration = end - start

                format_str = '%s: step %d, (%.3f sec/batch)'
                print(format_str % (datetime.now(), i, duration))

                image = draw_bboxes_classes_probs(
                    image, cls_prob, np.array(rois)[0][:, 1:5], bbox_pred,
                    blob['im_info'], blob['gt_boxes'], data.classes)
                cv2.imshow('result', image)
                cv2.waitKey()
