# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
#
#
# -----------------------------------------------------------------------------
import tensorflow as tf


class DefaultTrainConfig(object):
    name = 'default'

    restore_type = 'ckpt_v2'

    examples_per_epoch = 5011
    examples_evaluate = 1000
    epoches_every_decay = 10

    initial_learning_rate = 0.001

    moving_average_decay = 0.9999
    learning_rate_decay_factor = 0.1

    batch_size = 1

    momentum = 0.9

    max_steps = 120000
    log_frequency = 10
    # save_frequency = examples_per_epoch

    @property
    def train_dir(self):
        return './tensorboard/' + self.name

    @property
    def ckpt_path_base(self):
        return self.train_dir + '/ckpt/'

    @property
    def ckpt_file(self):
        if self.restore_type == 'ckpt_v1':
            return './dataset/pretrain_model/VGGnet_fast_rcnn_iter_70000.ckpt'
        else:
            return tf.train.latest_checkpoint(self.ckpt_path_base)
        # return self.train_dir + '/ckpt/checkpoint'

    @property
    def save_frequency(self):
        return self.examples_per_epoch

