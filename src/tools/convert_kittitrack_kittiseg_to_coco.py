from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import json
import numpy as np
import os
import cv2
import pycocotools.mask as rletools
DATA_PATH = '/mnt/sda/lcy/TraDeS-master/data/kitti_tracking_seg/'
SPLITS = ['train_half', 'val_half', 'train', 'test']

#train文件中有21个txt文件夹,test文件中有29个文件夹
#train_half和val_half是指每个txt文件中,比如说:0000.txt,选一半来做训练集，一半来做预测集
VIDEO_SETS = {'train': range(21), 'test': range(29),
  'train_half': range(21), 'val_half': range(21)}
CREATE_HALF_LABEL = True
DEBUG = False

cats = ['Pedestrian', 'Car', 'Cyclist', 'Van', 'Truck',  'Person_sitting',
        'Tram', 'Misc', 'DontCare' , 'DontCaretwo']

cat_ids = {cat: i + 1 for i, cat in enumerate(cats)}
cat_ids['Person'] = cat_ids['Person_sitting']

cat_info = []
for i, cat in enumerate(cats):
  cat_info.append({'name': cat, 'id': i + 1})

def get_segmentation_bbox(height,width,code):
    size = [height,width]
    # code = bytes(code, encoding="utf8")
    message = {'size':size,'counts':code}
    data = rletools.decode(message)
    bbox = rletools.toBbox(message)
    bbox = bbox.tolist()
    return message,bbox

def project_to_image(pts_3d, P):
  # pts_3d: n x 3
  # P: 3 x 4
  # return: n x 2
  pts_3d_homo = np.concatenate(
    [pts_3d, np.ones((pts_3d.shape[0], 1), dtype=np.float32)], axis=1)
  pts_2d = np.dot(P, pts_3d_homo.transpose(1, 0)).transpose(1, 0)
  pts_2d = pts_2d[:, :2] / pts_2d[:, 2:]
  return pts_2d

def read_clib(calib_path):
  f = open(calib_path, 'r')
  for i, line in enumerate(f):
    if i == 2:
      calib = np.array(line.strip().split(' ')[1:], dtype=np.float32)
      calib = calib.reshape(3, 4)
      return calib

def _bbox_to_coco_bbox(bbox):
  return [(bbox[0]), (bbox[1]),
          (bbox[2] - bbox[0]), (bbox[3] - bbox[1])]

def seg_id_change(id):
    if id == 1:
        return 2
    elif id == 2:
        return 1
    else:
        return 10

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
                # num_images += 1
                image_info = {'file_name': '{}/{:06d}.png'.format(video_name, j),
                              'id': j + 1,
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

            ann_2d_path = DATA_PATH + 'label_02/{}.txt'.format(video_name)
            ann_seg_path = DATA_PATH + 'instances_txt/{}.txt'.format(video_name)
            anns_2d = open(ann_2d_path, 'r')
            anns_seg = open(ann_seg_path, 'r')

            if CREATE_HALF_LABEL and 'half' in split:
                label_out_folder = DATA_PATH + 'label_02_{}/'.format(split)
                label_out_path = label_out_folder + '{}.txt'.format(video_name)
                if not os.path.exists(label_out_folder):
                    os.mkdir(label_out_folder)
                label_out_file = open(label_out_path, 'w')

            for ann_ind_2d, txt_2d in enumerate(anns_2d):
                tmp_2d = txt_2d[:-1].split(' ')
                frame_id_2d = int(tmp_2d[0])
                track_id_2d = int(tmp_2d[1])
                cat_id_2d = cat_ids[tmp_2d[2]]
                truncated = int(float(tmp_2d[3]))
                occluded = int(tmp_2d[4])
                alpha = float(tmp_2d[5])
                bbox = _bbox_to_coco_bbox([float(tmp_2d[6]), float(tmp_2d[7]), float(tmp_2d[8]), float(tmp_2d[9])])
                dim = [float(tmp_2d[10]), float(tmp_2d[11]), float(tmp_2d[12])]
                location = [float(tmp_2d[13]), float(tmp_2d[14]), float(tmp_2d[15])]
                rotation_y = float(tmp_2d[16])
                amodel_center = project_to_image(
                    np.array([location[0], location[1] - dim[0] / 2, location[2]],
                             np.float32).reshape(1, 3), calib)[0].tolist()

                for ann_ind_seg, txt_seg in enumerate(anns_seg):
                    tmp_seg = txt_seg[:-1].split(' ')
                    frame_id_seg = int(tmp_seg[0])
                    track_id_seg = int(tmp_seg[1]) % 1000
                    # cat_id_seg = int(tmp_seg[2])
                    cat_id_seg = seg_id_change(int(tmp_seg[2]))
                    height = int(tmp_seg[3])
                    width = int(tmp_seg[4])
                    segmentation, bbox_seg = get_segmentation_bbox(height, width, tmp_seg[5])

                    if frame_id_2d == frame_id_seg:
                        ann = {'image_id': frame_id_2d + 1 - image_range[0] + image_id_base,
                               'id': int(len(ret['annotations']) + 1),
                               'dim': dim,
                               'depth': location[2],
                               'alpha': alpha,
                               'truncated': truncated,
                               'occluded': occluded,
                               'location': location,
                               'rotation_y': rotation_y,
                               'amodel_center': amodel_center,
                               'category_id': cat_id_2d,
                               'bbox': bbox,
                               'bbox_seg':bbox_seg,
                               'track_id_seg':track_id_seg + 1,
                               # 只管2D目标检测的跟踪id
                               'track_id_2d': track_id_2d + 1,
                               'segmentation': segmentation}
                        ret['annotations'].append(ann)

            anns_2d = open(ann_2d_path, 'r')
            anns_seg = open(ann_seg_path, 'r')
            # 将标签信息写进一个txt文件中
            for ann_ind, txt in enumerate(anns_2d):
                tmp = txt[:-1].split(' ')
                frame_id = int(tmp[0])
                if CREATE_HALF_LABEL and 'half' in split:
                    if (frame_id < image_range[0] or frame_id > image_range[1]):
                        continue

                    # 把标签再全部写进另外一个文件里
                    out_frame_id = frame_id - image_range[0]
                    label_out_file.write('{} {}'.format(
                        out_frame_id, txt[txt.find(' ') + 1:]))

            for ann_ind, txt in enumerate(anns_seg):
                tmp = txt[:-1].split(' ')
                frame_id = int(tmp[0])
                track_id = int(tmp[1])
                txt = txt[txt.find(' ') + 1:]
                txt = txt[txt.find(' ') + 1:]
                txt = txt[txt.find(' ') + 1:]
                # 把车和行人的id进行转换
                cat_id= seg_id_change(int(tmp[2]))
                if CREATE_HALF_LABEL and 'half' in split:
                    if (frame_id < image_range[0] or frame_id > image_range[1]):
                        continue
                    # 把标签再全部写进另外一个文件里
                    out_frame_id = frame_id - image_range[0]
                    label_out_file.write('{} {} {} {}'.format(
                        out_frame_id,track_id,cat_id,txt))

        print("# images: ", len(ret['images']))
        print("# annotations: ", len(ret['annotations']))
        out_dir = '{}/annotations/'.format(DATA_PATH)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        out_path = '{}/annotations/tracking_{}.json'.format(DATA_PATH, split)
        json.dump(ret, open(out_path, 'w'))
