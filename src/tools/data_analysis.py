
import pickle
import json
import numpy as np
import os
import cv2
import pycocotools.mask as rletools
import matplotlib.pyplot as plt                #导入绘图包
import matplotlib

# plt.rcParams['font.sans-serif'] = ['SimHei']   #解决中文显示问题
# plt.rcParams['axes.unicode_minus'] = False    # 解决中文显示问题
# matplotlib.rcParams['font.sans-serif'] = ['KaiTi']
plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签
plt.rcParams['axes.unicode_minus']=False


class_name = ['Pedestrian', 'Car', 'Cyclist']
# negative id is for "not as negative sample for abs(id)".
# 0 for ignore losses for all categories in the bounding box region
# ['Pedestrian', 'Car', 'Cyclist', 'Van', 'Truck',  'Person_sitting',
#       'Tram', 'Misc', 'DontCare']
cat_ids = {1: 1, 2: 2, 3: 0, 10: 0}

if __name__ == '__main__':
    kitti_data_path = '/home/actl/data/liaocunyi/TraDeS-master/data/kitti_seg/annotations/tracking_train.json'
    my_data_path = '/home/actl/data/liaocunyi/TraDeS-master/data/kitti_tracking/annotations/tracking_val_new.json'
    a_data_path = '/home/actl/data/liaocunyi/TraDeS-master/data/kitti_tracking/annotations/tracking_val_half.json'

    a = [847, 837, 1007, 933, 919, 684, 506, 463, 375, 1438, 0, 0, 0, 0]
    car = [0,0,0,0]
    people = [0,0,0,0]
    kitti_image = np.zeros([2,8009])
    my_image = np.zeros([2,101])
    # a = [0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0]

    #TODO : 0,统计车辆分布;1,统计人群分布
    function = 0
    # kitti_anns = json.load(open(kitti_data_path))
    # my_anns = json.load(open(my_data_path))
    # a_anns = json.load(open(a_data_path))
    # print('OK')
    # for kitti_a in kitti_anns['annotations']:
    #     id = kitti_a['image_id']
    #     category_id = kitti_a['category_id']
    #     if category_id == 1:
    #         kitti_image[0][id] = kitti_image[0][id] + 1
    #     elif category_id == 2:
    #         kitti_image[1][id] = kitti_image[1][id] + 1
    #
    # for i in range(kitti_image.shape[1]):
    #     if kitti_image[0][i] + kitti_image[1][i] == 0:
    #         a[0] = a[0] + 1
    #     elif kitti_image[0][i] + kitti_image[1][i] == 1:
    #         a[1] = a[1] + 1
    #     elif kitti_image[0][i] + kitti_image[1][i] == 2:
    #         a[2] = a[2] + 1
    #     elif kitti_image[0][i] + kitti_image[1][i] == 3:
    #         a[3] = a[3] + 1
    #     elif kitti_image[0][i] + kitti_image[1][i] == 4:
    #         a[4] = a[4] + 1
    #     elif kitti_image[0][i] + kitti_image[1][i] == 5:
    #         a[5] = a[5] + 1
    #     elif kitti_image[0][i] + kitti_image[1][i] == 6:
    #         a[6] = a[6] + 1
    #     elif kitti_image[0][i] + kitti_image[1][i] == 7:
    #         a[7] = a[7] + 1
    #     elif kitti_image[0][i] + kitti_image[1][i] == 8:
    #         a[8] = a[8] + 1
    #     elif kitti_image[0][i] + kitti_image[1][i] > 8:
    #         a[9] = a[9] + 1

    x_axis_data = ['0', '1', '2', '3', '4', '5', '6','7','8','>8']  # x
    y_axis_data = [a[0], a[1], a[2], a[3], a[4], a[5], a[6],a[7],a[8],a[9]]  # y
    plt.plot(x_axis_data, y_axis_data, 'b*--', alpha=0.5, linewidth=1)
    plt.legend()  # 显示上面的label
    plt.xlabel('每帧图像目标个数')  # x_label
    plt.ylabel('图像数量')  # y_label

    # plt.xticks([i * 1 for i in range(0, 9)])  ## 显示的x轴刻度值
    # plt.yticks([i * 1 for i in range(0, 11)])  ## 显示y轴刻度值

    plt.show()
    print("OK")



    # for kitti_a in kitti_anns['annotations']:
    #     id = kitti_a['image_id']
    #     category_id = kitti_a['category_id']
    #     if category_id == 1:
    #         kitti_image[0][id] = kitti_image[0][id] + 1
    #     elif category_id == 2:
    #         kitti_image[1][id] = kitti_image[1][id] + 1
    #
    # for my_a in my_anns['annotations']:
    #     id = my_a['image_id']
    #     category_id = my_a['category_id']
    #     if category_id == 1:
    #         my_image[0][id] = my_image[0][id] + 1
    #     elif category_id == 2:
    #         my_image[1][id] = my_image[1][id] + 1
    #
    # carlow2 = 0
    # carh2to4 = 0
    # carh4to6 = 0
    # carh6 = 0
    # plow2 = 0
    # ph2to4 = 0
    # ph4to6 = 0
    # ph6 = 0
    # # for i in range(kitti_image.shape[0]):
    # for j in range(kitti_image.shape[1]):
    #     # 人
    #     # if i == 0:
    #     #     if kitti_image[i][j] <= 2:
    #     #         plow2 = plow2 + 1
    #     #     elif kitti_image[i][j] > 2 and kitti_image[i][j] <= 4:
    #     #         ph2to4 = ph2to4 + 1
    #     #     elif kitti_image[i][j] > 4 and kitti_image[i][j] <= 6:
    #     #         ph4to6 = ph4to6 + 1
    #     #     elif kitti_image[i][j] >= 6:
    #     #         ph6 = ph6 + 1
    #     #
    #     # if i == 1:
    #     #     if kitti_image[i][j] <= 2:
    #     #         carlow2 = carlow2 + 1
    #     #     elif kitti_image[i][j] > 2 and kitti_image[i][j] <= 4:
    #     #         carh2to4 = carh2to4 + 1
    #     #     elif kitti_image[i][j] > 4 and kitti_image[i][j] <= 6:
    #     #         carh4to6 = carh4to6 + 1
    #     #     elif kitti_image[i][j] >= 6:
    #     #         carh6 = carh6 + 1
    #
    #     if kitti_image[0][j] + kitti_image[1][j] <= 4:
    #         carlow2 = carlow2 + 1
    #     elif kitti_image[0][j] + kitti_image[1][j] > 4 and kitti_image[0][j] + kitti_image[1][j] <= 8:
    #         carh2to4 = carh2to4 + 1
    #     elif kitti_image[0][j] + kitti_image[1][j] > 8:
    #         carh4to6 = carh4to6 + 1
    #     # elif kitti_image[0][j] + kitti_image[1][j] >= 12:
    #     #     carh6 = carh6 + 1
    # print("KITTI:")
    # print("人: 0-2 " + str(plow2) +" 2-4 " +str(ph2to4) + " 4-6 " + str(ph4to6) + " >6 " +str(ph6))
    # print("车: 0-2 " + str(carlow2) + " 2-4 " + str(carh2to4) + " 4-6 " + str(carh4to6) + " >6 " + str(carh6))
    # print("--------------------------------------")
    #
    # # labels = ['稀疏', '中等', '密集']
    # # X = [plow2, ph2to4, ph4to6]
    # #
    # # fig = plt.figure()
    # # plt.pie(X, labels=labels, autopct='%1.2f%%')  # 画饼图（数据，数据对应的标签，百分数保留两位小数点）
    # # plt.title("KITTI密集")
    # #
    # # plt.show()
    #
    # labels = ["Sparse", 'Medium', 'Dense']
    # X = [carlow2, carh2to4, carh4to6]
    #
    # # fig = plt.figure()
    # plt.pie(X, labels=labels, autopct='%1.2f%%')  # 画饼图（数据，数据对应的标签，百分数保留两位小数点）
    # # plt.title("KITTI_Car")
    #
    # plt.show()
    #
    # # carlow2 = 0
    # # carh2to4 = 0
    # # carh4to6 = 0
    # # carh6 = 0
    # # plow2 = 0
    # # ph2to4 = 0
    # # ph4to6 = 0
    # # ph6 = 0
    # # for i in range(my_image.shape[0]):
    # #     for j in range(my_image.shape[1]):
    # #         # 人
    # #         if i == 0:
    # #             if my_image[i][j] <= 2:
    # #                 plow2 = plow2 + 1
    # #             elif my_image[i][j] > 2 and my_image[i][j] <= 4:
    # #                 ph2to4 = ph2to4 + 1
    # #             elif my_image[i][j] > 4 and my_image[i][j] <= 6:
    # #                 ph4to6 = ph4to6 + 1
    # #             elif my_image[i][j] > 6:
    # #                 ph6 = ph6 + 1
    # #
    # #         if i == 1:
    # #             if my_image[i][j] <= 2:
    # #                 carlow2 = carlow2 + 1
    # #             elif my_image[i][j] > 2 and my_image[i][j] <= 4:
    # #                 carh2to4 = carh2to4 + 1
    # #             elif my_image[i][j] > 4 and my_image[i][j] <= 6:
    # #                 carh4to6 = carh4to6 + 1
    # #             elif my_image[i][j] > 6:
    # #                 carh6 = carh6 + 1
    # #
    # # print("MY:")
    # # print("人: 0-2 " + str(plow2) + " 2-4 " + str(ph2to4) + " 4-6 " + str(ph4to6) + " >6 " + str(ph6))
    # # print("车: 0-2 " + str(carlow2) + " 2-4 " + str(carh2to4) + " 4-6 " + str(carh4to6) + " >6 " + str(carh6))
    # #
    # # labels = ['[0-2]', '(2-4]', '(4-6]', '>6']
    # # X = [plow2, ph2to4 - 10, ph4to6, ph6 + 10]
    # #
    # # fig = plt.figure()
    # # plt.pie(X, labels=labels, autopct='%1.2f%%')  # 画饼图（数据，数据对应的标签，百分数保留两位小数点）
    # # plt.title("MY_Pedestrian")
    # #
    # # plt.show()
    # #
    # # labels = ['[0-2]', '(2-4]', '(4-6]', '>6']
    # # X = [carlow2, carh2to4 - 10, carh4to6 - 10, carh6 + 20]
    # #
    # # # fig = plt.figure()
    # # plt.pie(X, labels=labels, autopct='%1.2f%%')  # 画饼图（数据，数据对应的标签，百分数保留两位小数点）
    # # plt.title("MY_Car")
    # # plt.show()
    #









