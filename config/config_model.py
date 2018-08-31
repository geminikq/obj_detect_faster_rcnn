# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
#
#
# -----------------------------------------------------------------------------
from config.config_input import ImageInput


class DefaultModelConfig(object):
    EnumTrainSet = 0
    EnumTestSet = 1
    # A small number that's used many times
    EPS = 1e-14

    #
    feature_stride = [16, ]
    anchor_scales = [8, 16, 32]
    anchor_ratios = [0.5, 1, 2]
    anchor_base_size = 16

    num_classes = 2

    weights_file = './dataset/pretrain_model/VGG_imagenet.npy'

    # Minibatch size (number of regions of interest [ROIs])
    TRAIN_BATCH_SIZE = 128
    # Fraction of minibatch that is labeled foreground (i.e. class > 0)
    TRAIN_FG_FRACTION = 0.25

    # IOU >= thresh: positive example
    RPN_POSITIVE_OVERLAP = 0.7
    # IOU < thresh: negative example
    RPN_NEGATIVE_OVERLAP = 0.3
    # If an anchor statisfied by positive and negative conditions set to negative
    RPN_CLOBBER_POSITIVES = False
    # Max number of foreground examples
    RPN_FG_FRACTION = 0.5
    # Total number of examples
    RPN_BATCHSIZE = 256
    # NMS threshold used on RPN proposals
    # in [train, test] sequence
    RPN_NMS_THRESH = [0.7, 0.7]
    # Number of top scoring boxes to keep before apply NMS to RPN proposals,
    # in [train, test] sequence
    RPN_PRE_NMS_TOP_N = [12000, 6000]   # 12000, 6000
    # Number of top scoring boxes to keep after applying NMS to RPN proposals,
    # in [train, test] sequence
    RPN_POST_NMS_TOP_N = [2000, 300]    # 2000, 300
    # Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
    # in [train, test] sequence
    RPN_MIN_SIZE = [16, 16]
    # Deprecated (outside weights)
    RPN_BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
    # Give the positive RPN examples weight of p * 1 / {num positives}
    # and give negatives a weight of (1 - p)
    # Set to -1.0 to use uniform example weighting
    RPN_POSITIVE_WEIGHT = -1.0
    # Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)
    FG_THRESH = 0.5
    # Overlap threshold for a ROI to be considered background (class = 0 if
    # overlap in [LO, HI))
    BG_THRESH_HI = 0.5
    BG_THRESH_LO = 0.0
    # Normalize the targets using "precomputed" (or made up) means and stdevs
    # (BBOX_NORMALIZE_TARGETS must also be True)
    BBOX_NORMALIZE_TARGETS_PRECOMPUTED = True
    BBOX_NORMALIZE_MEANS = (0.0, 0.0, 0.0, 0.0)
    BBOX_NORMALIZE_STDS = (0.1, 0.1, 0.2, 0.2)
    # Normalize the targets (subtract empirical mean, divide by empirical stddev)
    BBOX_NORMALIZE_TARGETS = True
    # Deprecated (inside weights)
    BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)

    #
    ROI_ALIGN_SIZE = 14
    ROI_ALIGN_KERNEL_SIZE = 2

    def __init__(self, cfg=ImageInput()):
        self.num_classes = len(cfg.classes)

        self.TRAIN_BATCH_SIZE = cfg.batch_size_rois
        self.TRAIN_FG_FRACTION = cfg.batch_fg_fraction


cfg_fix = DefaultModelConfig()
