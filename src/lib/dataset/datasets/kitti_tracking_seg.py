from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
import numpy as np
import torch
import json
import cv2
import os
import math

from ..generic_dataset import GenericDataset
from utils.ddd_utils import compute_box_3d, project_to_image

class KITTITrackingSeg(GenericDataset):
    num_categories = 3
    default_resolution = [384, 1280]
    class_name = ['Pedestrian', 'Car', 'Cyclist']
    # negative id is for "not as negative sample for abs(id)".
    # 0 for ignore losses for all categories in the bounding box region
    # ['Pedestrian', 'Car', 'Cyclist', 'Van', 'Truck',  'Person_sitting',
    #       'Tram', 'Misc', 'DontCare']
    cat_ids = {1: 1, 2: 2, 3: 3, 4: -2, 5: -2, 6: -1, 7: -9999, 8: -9999, 9: 0, 10:0 }
    max_objs = 50

    def __init__(self, opt, split):
        data_dir = os.path.join(opt.data_dir, 'kitti_tracking_seg')
        split_ = 'train' if opt.dataset_version != 'test' else 'test'  # 'test'
        img_dir = os.path.join(
            data_dir, 'data_tracking_image_2', '{}ing'.format(split_), 'image_02')
        ann_file_ = split_ if opt.dataset_version == '' else opt.dataset_version
        print('Warning! opt.dataset_version is not set')
        ann_path = os.path.join(
            data_dir, 'annotations', 'tracking_{}.json'.format(
                ann_file_))
        self.images = None
        super(KITTITrackingSeg, self).__init__(opt, split, ann_path, img_dir)
        self.alpha_in_degree = False
        self.num_samples = len(self.images)

        print('Loaded {} {} samples'.format(split, self.num_samples))

    def __len__(self):
        return self.num_samples

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def run_eval(self, results, save_dir):
        self.save_results(results, save_dir)
        os.system('python tools/eval_kitti_track/evaluate_tracking.py ' + \
                  '{}/results_kitti_tracking/ {}'.format(
                      save_dir, self.opt.dataset_version))

