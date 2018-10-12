# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
#
#
# -----------------------------------------------------------------------------
from config.config_input import DemoImageInput
from config.config_model import DefaultModelConfig
from config.config_train import DefaultTrainConfig

from trainval.processor import Processor


# define input data
test_data_cfg = DemoImageInput()

test_model_cfg = DefaultModelConfig(test_data_cfg)

proc = Processor()
weights_file = './tensorboard/default/ckpt/'
# weights_file = './dataset/pretrain_model/VGGnet_fast_rcnn_iter_70000.ckpt'
proc.demo_loop(test_data_cfg, test_model_cfg, weights_file)

print('test done')
