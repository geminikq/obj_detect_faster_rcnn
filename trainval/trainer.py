# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
#
#
# -----------------------------------------------------------------------------
import tensorflow as tf
import numpy as np
from config.config_train import DefaultTrainConfig
from trainval.visual import recover_bboxes


class Trainer(object):
    def __init__(self, config=DefaultTrainConfig()):
        self.cfg = config

    @staticmethod
    def loss():
        rpn_loss = tf.add_n(tf.get_collection('rpn_losses'))
        rcnn_loss = tf.add_n(tf.get_collection('rcnn_losses'))
        loss = rpn_loss + rcnn_loss
        tf.summary.scalar('loss_total', loss)

        return loss

    def accuracy(self, predict, blob):
        """

        :param predict:
        :param blob
        :return:
        """
        scores, bbox_deltas, rois = predict
        im_info, gt_boxes = blob['im_info'], blob['gt_boxes']
        rois = rois[0]

        scale = im_info[0][-1]
        boxes = rois[:, 1:5]

        ind_overlaps, inds = recover_bboxes(
            boxes, bbox_deltas, im_info, gt_boxes, 0.5)

        if len(inds) == 0:
            precision, accuracy = 0, 0
        else:
            positive = gt_boxes[:, 4][inds]
            cls_pred = np.argmax(scores[ind_overlaps[inds]], axis=1)
            true_pos = (positive == cls_pred)
            precision = np.sum(true_pos) / len(positive)

            cls_prob = np.max(scores[ind_overlaps[inds]], axis=1)
            prob_list = []
            for idx in range(len(true_pos)):
                if bool(true_pos[idx]) is True:
                    prob_list.append(cls_prob[idx])
                else:
                    prob_list.append(0)
            accuracy = np.array(prob_list).mean()

        return precision, accuracy

    # def accuracy(self, scores, bbox_deltas, rois):
    #     acc_input = [scores, bbox_deltas, rois]
    #     acc_ret_type = [tf.float32, tf.float32]
    #     precision, accuracy = tf.py_func(self._accuracy_eval, acc_input, acc_ret_type)
    #     tf.summary.scalar('precision', precision)
    #     return precision, accuracy

    def train(self, total_loss):
        # optimizer and learning rate
        global_step = tf.Variable(0, trainable=False)

        num_batches_per_epoch = self.cfg.examples_per_epoch / self.cfg.batch_size
        decay_steps = int(num_batches_per_epoch * self.cfg.epoches_every_decay)

        lr = tf.train.exponential_decay(
            self.cfg.initial_learning_rate, global_step, decay_steps,
            self.cfg.learning_rate_decay_factor, staircase=True)
        tf.summary.scalar('learning_rate', lr)

        momentum = self.cfg.momentum
        # train_op = tf.train.MomentumOptimizer(lr, momentum).minimize(loss, global_step=global_step)
        opt = tf.train.MomentumOptimizer(lr, momentum)
        grads = opt.compute_gradients(total_loss)
        train_op = opt.apply_gradients(grads, global_step=global_step)

        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

        # Add histograms for gradients.
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)

        # Track the moving averages of all trainable variables.
        # variable_averages = tf.train.ExponentialMovingAverage(
        #     0.9999, global_step)
        # with tf.control_dependencies([apply_gradient_op]):
        #     train_op = variable_averages.apply(tf.trainable_variables())

        return train_op
