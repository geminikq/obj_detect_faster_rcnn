# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
#
#
# ------------------------------------------------------------------------------
import numpy as np
from model.cnn import CNNModel


class VGGFeatureExtractor(CNNModel):
    def __init__(self, weights_file=None):
        super(CNNModel, self).__init__()
        self.skip_layers = []
        self.weights_file = weights_file

    # def get_weight_and_freeze(self, layer_name):
    #     # assert self.weights_file is not None, 'No weights file defined'
    #     if layer_name in self.skip_layers:
    #         return None
    #     weights_dict = np.load(self.weights_file, encoding='latin1').item()
    #     return weights_dict[layer_name]

    def inference(self, images):
        conv1_1 = self.layer_conv(images, 3, 1, 64, name='conv1_1', stddev=0.01,
                                  var_init=self.get_weight_and_freeze('conv1_1'))
        conv1_2 = self.layer_conv(conv1_1, 3, 1, 64, name='conv1_2', stddev=0.01,
                                  var_init=self.get_weight_and_freeze('conv1_2'))
        pool1 = self.layer_max_pool(conv1_2, 2, 2, 'VALID', name='pool1')
        conv2_1 = self.layer_conv(pool1, 3, 1, 128, name='conv2_1', stddev=0.01,
                                  var_init=self.get_weight_and_freeze('conv2_1'))
        conv2_2 = self.layer_conv(conv2_1, 3, 1, 128, name='conv2_2', stddev=0.01,
                                  var_init=self.get_weight_and_freeze('conv2_2'))
        pool2 = self.layer_max_pool(conv2_2, 2, 2, 'VALID', name='pool2')
        conv3_1 = self.layer_conv(pool2, 3, 1, 256, name='conv3_1', stddev=0.01)
        conv3_2 = self.layer_conv(conv3_1, 3, 1, 256, name='conv3_2', stddev=0.01)
        conv3_3 = self.layer_conv(conv3_2, 3, 1, 256, name='conv3_3', stddev=0.01)
        pool3 = self.layer_max_pool(conv3_3, 2, 2, 'VALID', name='pool3')
        conv4_1 = self.layer_conv(pool3, 3, 1, 512, name='conv4_1', stddev=0.01)
        conv4_2 = self.layer_conv(conv4_1, 3, 1, 512, name='conv4_2', stddev=0.01)
        conv4_3 = self.layer_conv(conv4_2, 3, 1, 512, name='conv4_3', stddev=0.01)
        pool4 = self.layer_max_pool(conv4_3, 2, 2, 'VALID', name='pool4')
        conv5_1 = self.layer_conv(pool4, 3, 1, 512, name='conv5_1', stddev=0.01)
        conv5_2 = self.layer_conv(conv5_1, 3, 1, 512, name='conv5_2', stddev=0.01)
        conv5_3 = self.layer_conv(conv5_2, 3, 1, 512, name='conv5_3', stddev=0.01)

        return conv5_3
