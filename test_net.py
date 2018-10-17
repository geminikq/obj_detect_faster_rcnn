# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
#
#
# -----------------------------------------------------------------------------
from config.config_input import Pascal2007Test
from config.config_model import DefaultModelConfig
from config.config_train import DefaultTestConfig

from trainval.processor import Processor


# define input data
test_data_cfg = Pascal2007Test()

test_model_cfg = DefaultModelConfig(test_data_cfg)
test_cfg = DefaultTestConfig()

proc = Processor()
proc.test_loop(test_data_cfg, test_model_cfg, test_cfg)

print('test done')
