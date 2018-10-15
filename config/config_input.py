# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
#
#
# ------------------------------------------------------------------------------


class ImageInput(object):
    FileFormatPascal = 0

    name = 'default'
    file_path = './dataset/'
    cache_path = './dataset/cache'
    image_ext = '.jpg'

    file_format = FileFormatPascal

    # PASCAL specific config options
    data_set_cfg = {}

    classes = ('__background__', )  # background always index 0
    # data argument
    if_flipped = True

    # for rpn layer
    image_per_batch = 1

    train_set = 'trainval'
    # Scales to use during training (can list multiple scales)
    # Each scale is the pixel size of an image's shortest side
    train_scale = (600,)
    # Max pixel size of the longest side of a scaled input image
    train_max_size = 1000
    # Minibatch size (number of regions of interest [ROIs])
    batch_size_rois = 128
    # Fraction of minibatch that is labeled foreground (i.e. class > 0)
    batch_fg_fraction = 0.25
    # Pixel mean values (BGR order) as a (1, 1, 3) array
    # We use the same pixel mean for all networks even though it's not exactly what
    # they were trained with
    pixel_means = [[[102.9801, 115.9465, 122.7717]]]


class PascalVOC2007Input(ImageInput):
    name = 'pascal_voc_2007'
    file_path = './dataset/VOCdevkit/VOC2007'

    classes = ('__background__',
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor')

    # PASCAL specific config options
    data_set_cfg = {'cleanup': True,
                    'use_salt': True,
                    'use_diff': False,
                    'matlab_eval': False,
                    'rpn_file': None,
                    'min_size': 2}


class Pascal2007Test(PascalVOC2007Input):
    name = 'pascal_voc_2007_test'
    train_set = 'test'

    if_flipped = False


class DemoImageInput(PascalVOC2007Input):
    file_path = './dataset/demo/'
    name = 'demo_test'
    train_set = 'test'

    if_flipped = False
