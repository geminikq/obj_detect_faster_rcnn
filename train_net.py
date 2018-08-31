# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
#
#
# -----------------------------------------------------------------------------
import argparse
from config.config_input import PascalVOC2007Input
from config.config_model import DefaultModelConfig
from config.config_train import DefaultTrainConfig

from trainval.processor import Processor


def train_net(class_name):
    # define input data
    pascal_data_cfg = PascalVOC2007Input()
    if class_name != 'all':
        if class_name not in pascal_data_cfg.classes:
            return
        else:
            print('only train specific class: {}'.format(class_name))
            pascal_data_cfg.name += '_' + class_name
            pascal_data_cfg.train_set = class_name + '_' + pascal_data_cfg.train_set

    # define model
    vgg_faster_rcnn_cfg = DefaultModelConfig(pascal_data_cfg)
    # define train method
    default_train_cfg = DefaultTrainConfig()

    proc = Processor()
    proc.train_loop(pascal_data_cfg, vgg_faster_rcnn_cfg, default_train_cfg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a Faster R-CNN network')
    parser.add_argument('--class', dest='class_name', help='specific class to train',
                        default='all', type=str)

    args = parser.parse_args()
    train_net(args.class_name)


print('train process end')
