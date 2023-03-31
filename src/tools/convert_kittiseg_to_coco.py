from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import json
import numpy as np
import os
import cv2
import pycocotools.mask as rletools
DATA_PATH = '/home/actl/data/liaocunyi/TraDeS-master/data/kitti_seg/'
SPLITS = ['train_half', 'val_half', 'train', 'test']

#train文件中有21个txt文件夹,test文件中有29个文件夹
#train_half和val_half是指每个txt文件中,比如说:0000.txt,选一半来做训练集，一半来做预测集
VIDEO_SETS = {'train': range(21), 'test': range(29),
  'train_half': range(21), 'val_half': range(21)}
CREATE_HALF_LABEL = True
DEBUG = False

cats = ['Pedestrian', 'Car','DontCare']
cat_ids = {cat: i + 1 for i, cat in enumerate(cats)}

cat_info = []
for i, cat in enumerate(cats):
    if cat != 'DontCare':
        cat_info.append({'name': cat, 'id': i + 1})
    else:
        cat_info.append({'name': cat, 'id': 10})


def get_segmentation_bbox(height,width,code):
    size = [height,width]
    # code = bytes(code, encoding="utf8")
    message = {'size':size,'counts':code}
    data = rletools.decode(message)
    bbox = rletools.toBbox(message)
    bbox = bbox.tolist()
    return message,bbox

def seg_id_change(id):
    if id == 1:
        return 2
    elif id == 2:
        return 1
    else:
        return 10

def read_clib(calib_path):
  f = open(calib_path, 'r')
  for i, line in enumerate(f):
    if i == 2:
      calib = np.array(line.strip().split(' ')[1:], dtype=np.float32)
      calib = calib.reshape(3, 4)
      return calib

if __name__ == '__main__':
  for split in SPLITS:
    ann_dir = DATA_PATH + '/instance_txt/'
    ret = {'images': [], 'annotations': [], "categories": cat_info,
           'videos': []}

    # 加载图片的信息
    num_images = 0
    for i in VIDEO_SETS[split]:
      image_id_base = num_images
      # 0000.txt,0001.txt等文件名称
      video_name = '{:04d}'.format(i)
      ret['videos'].append({'id': i + 1, 'file_name': video_name})
      ann_dir = 'train' if not ('test' in split) else split
      video_path = DATA_PATH + \
                   '/data_tracking_image_2/{}ing/image_02/{}'.format(ann_dir, video_name)

      calib_path = DATA_PATH + 'data_tracking_calib/{}ing/calib/'.format(ann_dir) \
                   + '{}.txt'.format(video_name)
      calib = read_clib(calib_path)

      image_files = sorted(os.listdir(video_path))
      num_images_video = len(image_files)
      if CREATE_HALF_LABEL and 'half' in split:
        image_range = [0, num_images_video // 2 - 1] if split == 'train_half' else \
          [num_images_video // 2, num_images_video - 1]
      else:
        image_range = [0, num_images_video - 1]
      print('num_frames', video_name, image_range[1] - image_range[0] + 1)

      # # 需要加入图片的宽度和长度等信息
      # ann_path = DATA_PATH + 'instances_txt/{}.txt'.format(video_name)
      # anns = open(ann_path, 'r')

      for j, image_name in enumerate(image_files):
        if (j < image_range[0] or j > image_range[1]):
          continue
        num_images += 1
        image_info = {'file_name': '{}/{:06d}.png'.format(video_name, j),
                      'id': num_images,
                      'video_id': i + 1,
                      'height': 384,
                      'width': 1280,
                      'calib': calib.tolist(),
                      # 'prev_image_id': j if j > 0 else -1,
                      # 'next_image_id': j + 2 if j < num_images_video - 1 else -1,
                      'frame_id': j + 1 - image_range[0]}
        ret['images'].append(image_info)

      if split == 'test':
        continue

      ann_path = DATA_PATH + 'instances_txt/{}.txt'.format(video_name)
      anns = open(ann_path, 'r')

      if CREATE_HALF_LABEL and 'half' in split:
        label_out_folder = DATA_PATH + 'instances_txt_{}/'.format(split)
        label_out_path = label_out_folder + '{}.txt'.format(video_name)
        if not os.path.exists(label_out_folder):
          os.mkdir(label_out_folder)
        label_out_file = open(label_out_path, 'w')

      for ann_ind, txt in enumerate(anns):
          tmp = txt[:-1].split(' ')
          frame_id = int(tmp[0])
          track_id = int(tmp[1]) % 1000
          cat_id = seg_id_change(int(tmp[2]))
          height = int(tmp[3])
          width = int(tmp[4])
          segmentation,bbox = get_segmentation_bbox(height,width,tmp[5])
          if cat_id != 10:
              ann = {'image_id': frame_id + 1 - image_range[0] + image_id_base,
                     'id': int(len(ret['annotations']) + 1),
                     'iscrowd':0,
                     'category_id': cat_id,
                     'bbox': bbox,  # x1, y1, w, h
                     'segmentation': segmentation,
                     'track_id': track_id + 1}
              if CREATE_HALF_LABEL and 'half' in split:
                  if (frame_id < image_range[0] or frame_id > image_range[1]):
                      continue

                  # 把标签再全部写进另外一个文件里
                  out_frame_id = frame_id - image_range[0]
                  label_out_file.write('{} {}'.format(out_frame_id, txt[txt.find(' ') + 1:]))
              ret['annotations'].append(ann)

    print("# images: ", len(ret['images']))
    print("# annotations: ", len(ret['annotations']))
    out_dir = '{}/annotations/'.format(DATA_PATH)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    out_path = '{}/annotations/tracking_{}.json'.format(DATA_PATH, split)
    json.dump(ret, open(out_path, 'w'))


