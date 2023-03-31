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



resFile = "/home/actl/data/liaocunyi/TraDeS-master/data/kitti_seg/annotations/tracking_train.json"
anns = json.load(open(resFile))


# # DATA_PATH = '/mnt/sda/lcy/CenterTrack-master1/data'
# DATA_PATH = '/mnt/sda/lcy/CenterTrack-master/data/hy/road'
# filePath = '/mnt/sda/lcy/CenterTrack-master/data/hy/road/training/'
# name = sorted(os.listdir(filePath))
#
# resFile_1 = "/mnt/sda/lcy/CenterTrack-master/data/hy/road/annotations/tracking_keypoint_train_new.json"
# anns_1 = json.load(open(resFile_1))
# # �?练集和测试集分开
# slice = 8

# 划分图片 or 划分标�??
# function = 'picture'
# function = 'label'

class_name = ['Pedestrian', 'Car','DontCare']

def get_category_id(category):
    if category == 'Pedestrian':
        return 1
    elif category == 'Car':
        return 2
    elif category == 'DontCare':
        return 10

def read_clib():
    calib_path = "/home/actl/data/liaocunyi/TraDeS-master/data/kitti_seg/calib/000672.txt"
    f = open(calib_path, 'r')
    for i, line in enumerate(f):
        if i == 2:
            calib = np.array(line.strip().split(' ')[1:], dtype=np.float32)
            calib = calib.reshape(3, 4)
            return calib

def get_bbox(seg,file):
    # picture_path = '/home/actl/data/liaocunyi/TraDeS-master/data/image_2/' + file
    points = []
    point = [0,0,0,0]
    x_all = []
    y_all = []
    x_ys = seg.split(';')
    for x_y in x_ys:
        x = int(float(x_y.split(',')[0]))
        y = int(float(x_y.split(',')[1]))
        x_all.append(x)
        y_all.append(y)
        points.append({'x':x,'y':y})

    x_min = min(x_all)
    x_max = max(x_all)
    y_min = min(y_all)
    y_max = max(y_all)
    point[0] = x_min / 2  # x1
    point[1] = y_min * 384 / 720 # y1
    point[2] = (x_max - x_min) / 2  # w
    point[3] = (y_max - y_min) * 384 / 720  # h
    # o_img = cv2.imread(picture_path)
    # draw_0 = cv2.rectangle(o_img, (x_min , y_min ), (x_max, y_max), (255, 0, 0), 2)
    #
    # cv2.imwrite('g.jpg',draw_0)
    # # p_img = o_img * X_2
    # # cv2.imwrite('gray_1.jpg', p_img)
    return point

def get_segmentation(segmentation,file):
    picture_path = '/home/actl/data/liaocunyi/TraDeS-master/data/kitti_tracking/real_data_tracking_image_2/training/image_02_new/0000/' + file
    x_all = []
    y_all = []
    x_ys = segmentation.split(';')
    for x_y in x_ys:
        x = float(x_y.split(',')[0])
        y = 720 - (float(x_y.split(',')[1]))
        x_all.append(x / 2)
        y_all.append(y * 384 / 720)

    fig = plt.figure(figsize=(12.8, 3.84))
    x_all = np.array(x_all)
    y_all = np.array(y_all)
    plt.fill(x_all,y_all,'black')
    plt.xlim(0,1280)
    plt.ylim(0,384)
    plt.savefig('./1.png',bbox_inches='tight',pad_inches=0)
    plt.close()
    # img = Image.open('./1.png')
    # img_1 = img.convert("1")
    img = cv2.imread('./1.png', cv2.IMREAD_GRAYSCALE)
    # cv2.imshow("image", img)
    # img = Image.open('./1.png')
    img = np.array(img)
    img = img[1:-24,37:]
    img = cv2.resize(img,(1280,384)) / 255

    # o_img = cv2.imread(picture_path)
    # o_img = cv2.cvtColor(o_img, code=cv2.COLOR_BGR2GRAY)
    # cv2.imwrite('gray.jpg',img * 255)
    # p_img = o_img * img
    # cv2.imwrite('gray_1.jpg', p_img)

    img[img > 0.5] = 1
    img[img < 0.5] = 0
    img = 1 - img

    # 编码为rle格式
    img = np.asfortranarray(img.astype('uint8'))
    rle = rletools.encode(img)
    # rle["counts"] = rle["counts"].decode("utf-8")
    rle["counts"] = rle["counts"].decode("utf-8")

    # size = [height, width]
    # # code = bytes(code, encoding="utf8")
    # message = {'size': size, 'counts': code}
    data = rletools.decode(rle)
    bbox = rletools.toBbox(rle)
    bbox = bbox.tolist()

    return rle


cat_info = []
for i, cat in enumerate(class_name):
    if cat == 'Pedestrian':
        cat_info.append({'name': cat, 'id': 1})
    elif cat == 'Car':
        cat_info.append({'name': cat, 'id': 2})
    elif cat == 'DontCare':
        cat_info.append({'name': cat, 'id': 10})

if __name__ == '__main__':
    ret = {'images': [], 'annotations': [], "categories": cat_info,
           'videos': []}

    xml_path = '/home/actl/data/liaocunyi/TraDeS-master/data/' + 'annotations.xml'
    ret['videos'].append({'id': 1, 'file_name': '0000'})
    pre_ann = ET.parse(xml_path)

    images = pre_ann.findall('image')
    images_len = len(images)
    image_id = 0
    id = 0
    for image in images:
        image_id = image_id + 1
        image_attrib = image.attrib

        calib = read_clib()
        image_info = {'file_name': '0000/' + image_attrib['name'],
                    'height': int(image_attrib['height']),
                    'width': int(image_attrib['width']),
                    'id': image_id,
                    'video_id': 1,
                    'calib': calib.tolist(),
                    'frame_id': int(image_attrib['id']) + 1}
        ret['images'].append(image_info)

        polygons = image.findall('polygon')
        for polygon in polygons:
            label = polygon.attrib

            track_id = polygon.findall('attribute')[0].text
            category_id = label['label']
            segmentation = label['points']
            bbox = get_bbox(segmentation,image_attrib['name'])
            id = id + 1

            ann = {'image_id': image_id,
                  'id': id,
                  'iscrowd': 0,
                  'category_id': get_category_id(category_id),
                  'bbox': bbox,
                  'segmentation': get_segmentation(segmentation,image_attrib['name']),
                  # 'segmentation': 1,
                  'track_id': track_id}
            ret['annotations'].append(ann)
        print(image_id)
    DATA_PATH = "/home/actl/data/liaocunyi/TraDeS-master/data/kitti_tracking/"
    out_dir = '{}/annotations/'.format(DATA_PATH)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    out_path = '{}/annotations/tracking_val_new.json'.format(
        DATA_PATH)

    json.dump(ret, open(out_path, 'w'))
    print("OK")
    # number = [0,0,0,0,0,0,0,0]
    # ret['categories'].append(cat_info)
    # image_id = 1
    # video_id = 1
    #
    # # �?抓出1类出�?
    # one_class = False
    #
    # for video in name:
    #     if video == '0830' or video == '0831':
    #         continue
    #     xml_path = filePath + video + '/annotations.xml'
    #     ret['videos'].append({'id': video_id, 'file_name': video})
    #
    #     # 打开xml文档
    #     pre_ann = ET.parse(xml_path)
    #     # 取出所有image的数�?
    #     images = pre_ann.findall('image')
    #     images_number = len(images)
    #
    #     frame_id = 1
    #     for image in images:
    #         image_attrib = image.attrib
    #         image_info = {'file_name': video + '/images/' + image_attrib['name'],
    #                       'height': int(image_attrib['height']),
    #                       'width': int(image_attrib['width']),
    #                       'id': image_id,
    #                       'video_id': video_id,
    #                       'frame_id': int(image_attrib['id']) + 1}
    #         ret['images'].append(image_info)
    #
    #         boxes = image.findall('box')
    #         points = image.findall('points')
    #         for box in boxes:
    #             box_attrib = box.attrib
    #             keypoints = []
    #             if one_class == True:
    #                 if box_attrib['label'] == 'car':
    #                     if "group_id" in box_attrib:
    #                         box_id = box_attrib['group_id']
    #                     else:
    #                         continue
    #                     # 得到关键点信�?
    #                     for point in points:
    #                         point_attrib = point.attrib
    #                         if "group_id" in point_attrib:
    #                             point_id = point_attrib['group_id']
    #                         else:
    #                             break
    #                         if box_id == point_id:
    #                             p = point_attrib['points']
    #                             p = p.split(',')
    #                             p_x = float(p[0])
    #                             p_y = float(p[1])
    #                             keypoints.append(p_x)
    #                             keypoints.append(p_y)
    #                             keypoints.append(2)
    #
    #                     if len(keypoints) != 12:
    #                         continue
    #
    #                     # 得到bbox信息
    #                     bbox = coco_bbox(box_attrib)
    #                     ann = {'image_id': image_id,
    #                            'id': int(len(ret['annotations']) + 1),
    #                            'category_id': 1,
    #                            'bbox': bbox,
    #                            'keypoints': keypoints,
    #                            'iscrowd': 0,
    #                            'num_keypoint': 4,
    #                            'track_id': int(box_id) + 1}
    #                     ret['annotations'].append(ann)
    #             else:
    #                 if "group_id" in box_attrib:
    #                     box_id = box_attrib['group_id']
    #                 else:
    #                     continue
    #                 # 得到关键点信�?
    #                 for point in points:
    #                     point_attrib = point.attrib
    #                     if "group_id" in point_attrib:
    #                         point_id = point_attrib['group_id']
    #                     else:
    #                         break
    #                     if box_id == point_id:
    #                         p = point_attrib['points']
    #                         p_debug = p
    #                         if ';' in p:
    #                             p = p.split(';')
    #                             p = p[0]
    #
    #                         p = p.split(',')
    #                         p_x = float(p[0])
    #                         p_y = float(p[1])
    #                         keypoints.append(p_x)
    #                         keypoints.append(p_y)
    #                         keypoints.append(2)
    #
    #                 if len(keypoints) != 12:
    #                     continue
    #
    #                 # 得到bbox信息
    #                 bbox = coco_bbox(box_attrib)
    #                 ann = {'image_id': image_id,
    #                        'id': int(len(ret['annotations']) + 1),
    #                        'category_id': get_category_id(box_attrib['label']),
    #                        'bbox': bbox,
    #                        'keypoints': keypoints,
    #                        'iscrowd': 0,
    #                        'num_keypoint': 4,
    #                        'track_id': int(box_id) + 1}
    #                 ret['annotations'].append(ann)
    #                 number[get_category_id(box_attrib['label']) - 1] = number[get_category_id(box_attrib['label']) - 1] + 1
    #
    #         image_id = image_id + 1
    #
    #         image = cv2.imread(filePath + video + '/images/' + image_attrib['name'])
    #         # cv2.rectangle(image,(int(ret['annotations'][i]['keypoints'][0]))
    #         for i in range(len(ret['annotations'])):
    #           cv2.circle(image, (int(ret['annotations'][i]['keypoints'][0]),int(ret['annotations'][i]['keypoints'][1])), 1, (0, 0, 255), -1)
    #           cv2.circle(image, (int(ret['annotations'][i]['keypoints'][3]),int(ret['annotations'][i]['keypoints'][4])), 1,
    #                      (0, 0, 255), -1)
    #           cv2.circle(image, (int(ret['annotations'][i]['keypoints'][6]),int(ret['annotations'][i]['keypoints'][7])), 1,
    #                      (0, 0, 255), -1)
    #           cv2.circle(image, (int(ret['annotations'][i]['keypoints'][9]), int(ret['annotations'][i]['keypoints'][10])), 1,
    #                      (0, 0, 255), -1)
    #         print('OK')
    #         cv2.imwrite(filePath + '1.png', image)
    #         # print('OK')
    #
    #     video_id = video_id + 1
    #     print('OK')
    #
    # print("# images: ", len(ret['images']))
    # print("# annotations: ", len(ret['annotations']))
    # out_dir = '{}/annotations/'.format(DATA_PATH)
    # if not os.path.exists(out_dir):
    #     os.mkdir(out_dir)
    #
    #
    # out_path = '{}/annotations/tracking_keypoint_train_new.json'.format(
    #     DATA_PATH)
    #
    # json.dump(ret, open(out_path, 'w'))
    #
    # # if function == 'picture':
    # #     for video in name:
    # #         # 获取每个包下面的图片
    # #         trainfiles = sorted(os.listdir(filePath + '/' + video + '/images/default'))
    # #         num_train = len(trainfiles)
    # #         print("num_train: " + str(num_train))
    # #         index_list = list(range(num_train))
    # #
    # #         num = 1
    # #         trainDir = filePath + '/training/' + video
    # #         validDir = filePath + '/testing/' + video
    # #         for i in index_list:
    # #             fileName = os.path.join(filePath + '/' + video + '/images/default', trainfiles[i])
    # #             if num < num_train * 0.8:
    # #                 print(str(fileName))
    # #                 copy2(fileName, trainDir)
    # #             else:
    # #                 copy2(fileName, validDir)
    # #
    # #             num += 1
