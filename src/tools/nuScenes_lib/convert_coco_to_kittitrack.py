# -*- coding:utf-8 -*-
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
import xml.dom.minidom
import json
import os
from shutil import copy2
import cv2
from io import BytesIO
from PIL import Image
import pycocotools.mask as rletools
import pycocotools.coco as coco
from collections import defaultdict


save_dir = "/home/actl/data/liaocunyi/TraDeS-master/src/tools/eval_kitti_track/data/tracking"

results_dir = os.path.join(save_dir, 'label_02_val_new')
if not os.path.exists(results_dir):
    os.mkdir(results_dir)

ann_path = "/home/actl/data/liaocunyi/TraDeS-master/src/lib/../../data/kitti_tracking/annotations/tracking_val_new.json"
coco = coco.COCO(ann_path)

video_to_images = defaultdict(list)
for image in coco.dataset['images']:
    video_to_images[image['video_id']].append(image)

class_name = ['Pedestrian', 'Car', 'Cyclist']

for video in coco.dataset['videos']:
    video_id = video['id']
    file_name = video['file_name']
    out_path = os.path.join(results_dir, '{}.txt'.format(file_name))
    f = open(out_path, 'w')
    images = video_to_images[video_id]

    for image_info in images:
        img_id = image_info['id']

        frame_id = image_info['frame_id']
        results = coco.imgToAnns[img_id]
        for i in range(len(results)):
            item = results[i]
            class_name = item['category_id']
            if not ('alpha' in item):
                item['alpha'] = -1
            if not ('rot_y' in item):
                item['rot_y'] = -10
            if 'dim' in item:
                item['dim'] = [max(item['dim'][0], 0.01),
                               max(item['dim'][1], 0.01), max(item['dim'][2], 0.01)]
            if not ('dim' in item):
                item['dim'] = [-1, -1, -1]
            if not ('loc' in item):
                item['loc'] = [-1000, -1000, -1000]

            if not ('loc' in item):
                item['score'] = [-1000, -1000, -1000]

            track_id = item['track_id'] if 'track_id' in item else -1
            f.write('{} {} {} -1 -1'.format(frame_id - 1, track_id, class_name))
            f.write(' {:d}'.format(int(item['alpha'])))
            f.write(' {:.2f} {:.2f} {:.2f} {:.2f}'.format(
                item['bbox'][0], item['bbox'][1], item['bbox'][2] + item['bbox'][0], item['bbox'][3] + item['bbox'][1]))
            f.write(' {:d} {:d} {:d}'.format(
                int(item['loc'][0]), int(item['loc'][1]), int(item['loc'][2])))

            f.write(' {:d}'.format(int(item['rot_y'])))
            f.write(' {:d} {:d} {:d}\n'.format(
                int(item['dim'][0]), int(item['dim'][1]), int(item['dim'][2])))

    f.close()
