#!/usr/bin/env python
# coding=UTF-8
'''
@Description: 应用openCV预训练模型，进行人脸识别。 
  参考：https://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html
@Author: Joe
@Verdion: 1.0
'''

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

DIR_ROOT = os.path.dirname(os.path.abspath("__file__"))
PATH_PRE_TRAIN_MODEL = os.path.join(
    DIR_ROOT, 'openCV/haarcascades/haarcascade_frontalface_alt.xml')

human_files = np.load('./data/input_human_files.npy')


def faceDetector(fname):
    # 提取预训练的人脸检测模型
    face_cascade = cv2.CascadeClassifier(PATH_PRE_TRAIN_MODEL)

    # 加载彩色（通道顺序为BGR）图像
    img = cv2.imread(fname)

    # 将BGR图像进行灰度处理
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 在图像中找出脸
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # 打印图像中检测到的脸的个数
    print('Number of faces detected:', len(faces))

    # 获取每一个所检测到的脸的识别框
    for (x, y, w, h) in faces:
        # 在人脸图像中绘制出识别框
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # 将BGR图像转变为RGB图像以打印
    cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 展示含有识别框的图像
    # plt.imshow(cv_rgb)
    # plt.show()
    cv2.imshow('img', img)


# faCe test
faceDetector(human_files[3])
