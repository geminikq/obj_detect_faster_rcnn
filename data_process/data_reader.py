# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
#
#
# ------------------------------------------------------------------------------
import os
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
# import pickle as pk
import _pickle as cPickle
import PIL.Image

from config.config_input import ImageInput
from data_process.roi_gen import ROIGenerator


class DataReader:
    def __init__(self, config=ImageInput()):
        self.cfg = config

        self.num_classes = len(self.cfg.classes)
        self.image_index = self._load_image_set_index()
        # self.num_images = len(self.image_index)
        self._class_to_ind = dict(zip(self.cfg.classes, range(self.num_classes)))

        self.roi_data = None

    @property
    def num_images(self):
        return len(self.image_index)

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
        image_set_file = os.path.join(self.cfg.file_path, 'ImageSets', 'Main',
                                      self.cfg.train_set + '.txt')
        assert os.path.exists(image_set_file), \
            'Path does not exist: {}'.format(image_set_file)

        with open(image_set_file) as f:
            image_index = []
            for x in f.readlines():
                xdata = x.strip().split(' ')
                if len(xdata) == 1 or xdata[-1] == '1':
                    image_index.append(xdata[0])
            # image_index = [x.strip() for x in f.readlines()]
        return image_index

    def _load_pascal_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        filename = os.path.join(self.cfg.file_path, 'Annotations', index + '.xml')
        if not os.path.exists(filename):
            print('Can not find annotations')
            return {'boxes' : np.zeros((1, 4), dtype=np.uint16),
                    'gt_classes': np.zeros(1, dtype=np.int32),
                    'gt_overlaps': scipy.sparse.csr_matrix(
                        np.zeros((1, self.num_classes), dtype=np.float32)),
                    'flipped' : False,
                    'seg_areas' : None}
        tree = ET.parse(filename)
        objs = tree.findall('object')
        if not self.cfg.data_set_cfg['use_diff']:
            # Exclude the samples labeled as difficult
            non_diff_objs = [
                obj for obj in objs if int(obj.find('difficult').text) == 0]
            # if len(non_diff_objs) != len(objs):
            #     print 'Removed {} difficult objects'.format(
            #         len(objs) - len(non_diff_objs))
            objs = non_diff_objs
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            cls = self._class_to_ind[obj.find('name').text.lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'seg_areas' : seg_areas}

    def _get_widths(self):
        return [PIL.Image.open(self.image_path_at(i)).size[0]
                for i in range(self.num_images)]

    def append_flipped_images(self, roidb):
        num_images = self.num_images
        widths = self._get_widths()
        for i in range(num_images):
            boxes = roidb[i]['boxes'].copy()
            oldx1 = boxes[:, 0].copy()
            oldx2 = boxes[:, 2].copy()
            boxes[:, 0] = widths[i] - oldx2 - 1
            boxes[:, 2] = widths[i] - oldx1 - 1
            # assert (boxes[:, 2] >= boxes[:, 0]).all()
            entry = {'boxes': boxes,
                     'gt_overlaps': roidb[i]['gt_overlaps'],
                     'gt_classes': roidb[i]['gt_classes'],
                     'flipped': True}
            roidb.append(entry)
        self.image_index = self.image_index * 2
        return roidb

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cfg.cache_path, self.cfg.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self.cfg.name, cache_file))
            return roidb

        # self.image_index = self._load_image_set_index()
        gt_roidb = [self._load_pascal_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, protocol=-1)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self.image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self.cfg.file_path, 'JPEGImages',
                                  index + self.cfg.image_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def prepare_roidb(self):
        """Enrich the imdb's roidb by adding some derived quantities that
        are useful for training. This function precomputes the maximum
        overlap, taken over ground-truth boxes, between each ROI and
        each ground-truth box. The class with maximum overlap is also
        recorded.
        """
        # for pascal_voc dataset
        roidb = self.gt_roidb()
        # data argument
        if self.cfg.if_flipped is True:
            print('append flipped images to training')
            roidb = self.append_flipped_images(roidb)

        sizes = [PIL.Image.open(self.image_path_at(i)).size
                 for i in range(self.num_images)]

        for i in range(len(self.image_index)):
            roidb[i]['image'] = self.image_path_at(i)
            roidb[i]['width'] = sizes[i][0]
            roidb[i]['height'] = sizes[i][1]
            # need gt_overlaps as a dense array for argmax
            gt_overlaps = roidb[i]['gt_overlaps'].toarray()
            # max overlap with gt over classes (columns)
            max_overlaps = gt_overlaps.max(axis=1)
            # gt class that had the max overlap
            max_classes = gt_overlaps.argmax(axis=1)
            roidb[i]['max_classes'] = max_classes
            roidb[i]['max_overlaps'] = max_overlaps
            # sanity checks
            # max overlap of 0 => class should be zero (background)
            zero_inds = np.where(max_overlaps == 0)[0]
            assert all(max_classes[zero_inds] == 0)
            # max overlap > 0 => class should not be zero (must be a fg class)
            nonzero_inds = np.where(max_overlaps > 0)[0]
            assert all(max_classes[nonzero_inds] != 0)

        self.roi_data = ROIGenerator(roidb, self.num_classes, self.cfg)
        return self.roi_data

    def forward(self):
        assert self.roi_data is not None
        return self.roi_data.forward()

