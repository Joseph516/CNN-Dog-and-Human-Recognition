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
import json


def pathToTensor(img_path):
    """将单个图像转化为向量输入"""
    # 用PIL加载RGB图像为PIL.Image.Image类型
    # keras CNN输入维度：(nb_samples, rows, columns, channels)
    img = image.load_img(img_path, target_size=(224, 224))
    # 将PIL.Image.Image类型转化为格式为(224, 224, 3)的3维张量
    x = image.img_to_array(img)
    # 将3维张量转化为格式为(1, 224, 224, 3)的4维张量并返回
    return np.expand_dims(x, axis=0)


def pathsToTensor(img_paths):
    """获取图像向量集"""
    list_of_tensors = [pathToTensor(img_path)
                       for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)


def modelPredictLabels(img_path, model):
    """返回img_path路径的图像的预测向量"""

    # 将图像转化为输入向量
    img_vec = pathToTensor(img_path)
    # 张量归一化：各像素减去像素均值
    img = preprocess_input(img_vec)
    # 模型预测
    preds = model.predict(img)

    return preds


def dogDetector(img_path, model):
    """小狗种类预测"""
    preds = modelPredictLabels(img_path, model)

    isUseKerasLabel = 0
    if (isUseKerasLabel == 1):
        # 输出最大可能预测结果
        print('Predicted:', decode_predictions(preds, top=1))
    else:
        idx = np.argmax(preds)
        # 将idx与label文件中对应.
        label_dic = {}
        with open('./model/imagenet1000_clsidx_to_labels.txt', 'r') as f:
            text = f.read()
            text = text.split(',\n')
            for line in text:
                line = line.strip('{} ')  # 去除多余的符号
                line = line.split(': ')
                label_dic[int(line[0])] = line[1]
            print('\nDog Predicted:', label_dic[idx])
        f.close()


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
