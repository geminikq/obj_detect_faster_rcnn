# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
#
#
# ------------------------------------------------------------------------------
import re
import numpy as np
import tensorflow as tf
# from skimage import io, transform
# from config_model import ConfigModel

TOWER_NAME = 'tower'


class CNNModel(object):
    name = 'cnn'
    skip_layers = []
    weights_file = ''

    def __init__(self):
        pass

    @staticmethod
    def _activation_summary(x):
        """Helper to create summaries for activations.
        Creates a summary that provides a histogram of activations.
        Creates a summary that measures the sparsity of activations.
        Args:
          x: Tensor
        Returns:
          nothing
        """
        # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
        # session. This helps the clarity of presentation on tensorboard.
        tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
        tf.summary.histogram(tensor_name + '/activations', x)
        tf.summary.scalar(tensor_name + '/sparsity',
                          tf.nn.zero_fraction(x))

    @staticmethod
    def _variable_on_cpu(name, shape, initializer):
        """Helper to create a Variable stored on CPU memory.
        Args:
          name: name of the variable
          shape: list of ints
          initializer: initializer for Variable
        Returns:
          Variable Tensor
        """
        with tf.device('/cpu:0'):
            # dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
            var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
        return var

    def _variable_with_weight_decay(self, name, shape, stddev, wd):
        """Helper to create an initialized Variable with weight decay.
        Note that the Variable is initialized with a truncated normal distribution.
        A weight decay is added only if one is specified.
        Args:
          name: name of the variable
          shape: list of ints
          stddev: standard deviation of a truncated Gaussian
          wd: add L2Loss weight decay multiplied by this float. If None, weight
              decay is not added for this Variable.
        Returns:
          Variable Tensor
        """
        # dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        var = self._variable_on_cpu(
            name, shape, tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))
        # var = tf.Variable(tf.truncated_normal(shape, stddev=stddev, dtype=tf.float32), name=name)
        if wd is not None:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return var

    @staticmethod
    def _modified_smooth_l1(sigma, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights):
        """
            ResultLoss = outside_weights * SmoothL1(inside_weights * (bbox_pred - bbox_targets))
            SmoothL1(x) = 0.5 * (sigma * x)^2,    if |x| < 1 / sigma^2
                          |x| - 0.5 / sigma^2,    otherwise
        """
        sigma2 = sigma * sigma

        inside_mul = tf.multiply(bbox_inside_weights, tf.subtract(bbox_pred, bbox_targets))

        smooth_l1_sign = tf.cast(tf.less(tf.abs(inside_mul), 1.0 / sigma2), tf.float32)
        smooth_l1_option1 = tf.multiply(tf.multiply(inside_mul, inside_mul), 0.5 * sigma2)
        smooth_l1_option2 = tf.subtract(tf.abs(inside_mul), 0.5 / sigma2)
        smooth_l1_result = tf.add(tf.multiply(smooth_l1_option1, smooth_l1_sign),
                                  tf.multiply(smooth_l1_option2, tf.abs(tf.subtract(smooth_l1_sign, 1.0))))

        outside_mul = tf.multiply(bbox_outside_weights, smooth_l1_result)

        return outside_mul

    def get_weight_and_freeze(self, layer_name, encoding='latin1', format='dict'):
        """

        :param layer_name:
        :param encoding:
        :param format:
        :return:
        """
        if layer_name in self.skip_layers:
            return None
        weights_dict = np.load(self.weights_file, encoding=encoding).item()
        if format == 'list':
            weights_list = weights_dict[layer_name]
            return {'weights': weights_list[0], 'biases': weights_list[1]}
        return weights_dict[layer_name]

    @staticmethod
    def conv(ix, ks, stride, padding):
        """lambda function of convolution"""
        return tf.nn.conv2d(ix, ks, strides=[1, stride, stride, 1], padding=padding)

    def layer_conv(self, input_x, ksize, stride, depth_out, padding='SAME', name='conv',
                   relu=True, stddev=5e-2, wd=None, const=0.0, groups=1, var_init=None):
        depth_in = int(input_x.shape[-1])
        with tf.variable_scope(name) as scope:
            if var_init is None:
                kernel = self._variable_with_weight_decay(
                    'weights', shape=[ksize, ksize, depth_in/groups, depth_out], stddev=stddev, wd=wd)
                biases = self._variable_on_cpu('biases', [depth_out], tf.constant_initializer(const))
            else:
                kernel = tf.get_variable(
                    'weights', shape=[ksize, ksize, depth_in/groups, depth_out],
                    initializer=tf.constant_initializer(var_init['weights']), trainable=False)
                biases = tf.get_variable(
                    'biases', shape=[depth_out],
                    initializer=tf.constant_initializer(var_init['biases']), trainable=False)

            if groups == 1:
                conv = self.conv(input_x, kernel, stride, padding)
            else:
                input_groups = tf.split(axis=3, num_or_size_splits=groups, value=input_x)
                weight_groups = tf.split(axis=3, num_or_size_splits=groups, value=kernel)
                output_groups = [self.conv(i, k, stride, padding) for i, k in zip(input_groups, weight_groups)]

                conv = tf.concat(axis=3, values=output_groups)

            # conv = tf.nn.conv2d(input_x, kernel, [1, stride, stride, 1], padding=padding)
            # biases = tf.Variable(tf.constant(value=const, shape=[depth], name='biases'))
            # pre_activation = tf.nn.bias_add(conv, biases, name=scope.name)
            # this_layer = tf.nn.relu(pre_activation)

            if relu is True:
                bias = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))
                relu = tf.nn.relu(bias, name=scope.name)
                self._activation_summary(relu)
                return relu
            else:
                bias = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv), name=scope.name)
                self._activation_summary(bias)
                return bias

    @staticmethod
    def layer_max_pool(input_x, ksize, stride, padding='SAME', name='pool'):
        with tf.variable_scope(name) as scope:
            pool = tf.nn.max_pool(
                input_x, ksize=[1, ksize, ksize, 1], strides=[1, stride, stride, 1],
                padding=padding, name=scope.name)
            return pool

    @staticmethod
    def layer_avg_pool(input_x, ksize, stride, padding, name):
        with tf.variable_scope(name) as scope:
            pool = tf.nn.avg_pool(
                input_x, ksize=[1, ksize, ksize, 1], strides=[1, stride, stride, 1],
                padding=padding, name=scope.name)
            return pool

    @staticmethod
    def layer_lrn(input_x, radius=4, bias=1.0, alpah=0.001 / 9.0, beta=0.75, name='lrn'):
        with tf.variable_scope(name) as scope:
            lrn = tf.nn.lrn(
                input_x, depth_radius=radius, bias=bias, alpha=alpah, beta=beta, name=scope.name)
            return lrn

    @staticmethod
    def layer_flatten(input_x, name='flatten'):
        with tf.variable_scope(name) as scope:
            last_shape = input_x.get_shape().as_list()
            this_depth = last_shape[1] * last_shape[2] * last_shape[3]
            this_layer = tf.reshape(input_x, [-1, this_depth], name=scope.name)
            return this_layer, this_depth

    def layer_fc(self, input_x, depth_in, depth_out, name='fc', relu=True,
                 stddev=0.04, wd=None, const=0.0, var_init=None):
        with tf.variable_scope(name) as scope:
            if var_init is None:
                weights = self._variable_with_weight_decay(
                    'weights', shape=[depth_in, depth_out], stddev=stddev, wd=wd)
                biases = self._variable_on_cpu(
                    'biases', [depth_out], tf.constant_initializer(const))
            else:
                weights = tf.get_variable(
                    'weights', shape=[depth_in, depth_out],
                    initializer=tf.constant_initializer(var_init['weights']), trainable=False)
                biases = tf.get_variable(
                    'biases', shape=[depth_out],
                    initializer=tf.constant_initializer(var_init['biases']), trainable=False)
            act = tf.nn.xw_plus_b(input_x, weights, biases, name=scope.name)

            if relu is True:
                relu = tf.nn.relu(act)
                self._activation_summary(relu)
                return relu
            else:
                self._activation_summary(act)
                return act

    @staticmethod
    def layer_dropout(input_x, keep_prob, name):
        with tf.variable_scope(name) as scope:
            drop = tf.nn.dropout(input_x, keep_prob, name=scope.name)
            return drop
