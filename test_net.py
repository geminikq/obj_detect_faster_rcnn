# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
#
#
# -----------------------------------------------------------------------------
from config.config_input import Pascal2007Test
from config.config_model import DefaultModelConfig
from config.config_train import DefaultTrainConfig

from trainval.processor import Processor


# define input data
test_data_cfg = Pascal2007Test()

test_model_cfg = DefaultModelConfig(test_data_cfg)

proc = Processor()
# weights_file = './tensorboard/horse/ckpt/'
weights_file = './dataset/pretrain_model/VGGnet_fast_rcnn_iter_70000.ckpt'
proc.test_loop(test_data_cfg, test_model_cfg, weights_file)

print('test done')
