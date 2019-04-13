#!/usr/bin/env python
# coding=UTF-8
'''
@Description: 将图像转化为向量，不包括图像增强。
@Author: Joe
@Verdion: 1.0
'''
from PIL import ImageFile
import numpy as np
from process_function import *
import os
from global_path import *


ImageFile.LOAD_TRUNCATED_IMAGES = True

# no image augementation
# load data
dog_train_files = np.load('./data/input_dog_train_files.npy')
dog_valid_files = np.load('./data/input_dog_valid_files.npy')
dog_test_files = np.load('./data/input_dog_test_files.npy')

# Keras中的数据预处理过程
dog_train_tensors = pathsToTensor(dog_train_files).astype('float32')/255
dog_valid_tensors = pathsToTensor(dog_valid_files).astype('float32')/255
dog_test_tensors = pathsToTensor(dog_test_files).astype('float32')/255

# save tensors
os.makedirs(DIR_TENSORS, exist_ok=True)
np.save(PATH_TRAIN_TENSORS, dog_train_tensors)
np.save(PATH_VALID_TENSORS, dog_valid_tensors)
np.save(PATH_TEST_TENSORS, dog_test_tensors)
