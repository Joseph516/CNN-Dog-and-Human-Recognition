#!/usr/bin/env python
# coding=UTF-8
'''
@Description: 采用keras ResNet50模型识别小狗种类。
  模型架构图：http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006
  keras说明：https://keras.io/zh/applications/#resnet50
  论文： https://arxiv.org/abs/1512.03385
@Author: Joe
@Verdion: 1.0
'''

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from tqdm import tqdm
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
from keras.models import load_model
import os
from process_function import *


if __name__ == "__main__":
    DIR_ROOT = os.path.dirname(os.path.abspath("__file__"))
    PATH_WEIGHTS = os.path.join(
        DIR_ROOT, 'model/resnet50_weights_tf_dim_ordering_tf_kernels.h5')
    #  ImageNet权重，下载地址：https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5

    # 加载图片
    dog_train_files = np.load('./data/input_dog_train_files.npy')

    # 加载ResNet50模型权重
    # 参考:https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py
    ResNet50_model = ResNet50(weights=PATH_WEIGHTS)
    # 图像识别
    dogDetector(dog_train_files[0], ResNet50_model)
    # Dog Predicted: 'kuvasz'
