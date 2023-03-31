import numpy as np
import cv2 as cv
import os


picture_path = "../data/image_2/"
picture_list = os.listdir(picture_path)

for p_l in picture_list:

    img = cv.imread(picture_path + p_l)

    numpy_zero = np.zeros_like(img)
    img = np.concatenate((img,numpy_zero),axis=1)

    # 缩放图像，后面的其他程序都是在这一行上改动
    img = cv.resize(img, (1280, 384))

    # 显示图像
    # cv.imshow("dst: %d x %d" % (dst.shape[0], dst.shape[1]), dst)
    cv.imwrite("../data/image_2_new/" + p_l,img)

