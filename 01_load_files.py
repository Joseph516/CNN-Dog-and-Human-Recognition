#!/usr/bin/env python
# coding=UTF-8
'''
@Description: 读取图片文件，并保存。
@Author: Joe
@Verdion: 1.0
'''
import random
from sklearn.datasets import load_files
from keras.utils import np_utils
import numpy as np
from glob import glob
import os


def loadDataset(path):
    # define function to load train, test, and validation datasets
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_kinds = 133
    dog_targets = np_utils.to_categorical(np.array(data['target']), dog_kinds)
    return dog_files, dog_targets


# load train, test, and validation datasets
DIR_ROOT = os.path.dirname(os.path.abspath("__file__"))
PATH_TRAIN = os.path.join(DIR_ROOT, 'data/dogImages/train')
PATH_VALID = os.path.join(DIR_ROOT, 'data/dogImages/valid')
PATH_TEST = os.path.join(DIR_ROOT, 'data/dogImages/test')
train_files, train_targets = loadDataset(PATH_TRAIN)
valid_files, valid_targets = loadDataset(PATH_VALID)
test_files, test_targets = loadDataset(PATH_TEST)

# load list of dog names
dog_names = [item[len(PATH_TRAIN)+1:-1]
             for item in sorted(glob(PATH_TRAIN + "/*"))]

# print statistics about the dataset
print('There are %d total dog categories.' % len(dog_names))
print('There are %s total dog images.\n' %
      len(np.hstack([train_files, valid_files, test_files])))
print('There are %d training dog images.' % len(train_files))
print('There are %d validation dog images.' % len(valid_files))
print('There are %d test dog images.' % len(test_files))

# save input dog data
np.save('./data/input_dog_' + 'train_files' + '.npy', train_files)
np.save('./data/input_dog_' + 'train_targets' + '.npy', train_targets)
np.save('./data/input_dog_' + 'valid_files' + '.npy', valid_files)
np.save('./data/input_dog_' + 'valid_targets' + '.npy', valid_targets)
np.save('./data/input_dog_' + 'test_files' + '.npy', test_files)
np.save('./data/input_dog_' + 'test_targets' + '.npy', test_targets)
np.save('./data/dog_names.npy', dog_names)

# load and save human image files
PATH_HUMAN = os.path.join(DIR_ROOT, 'data/lfw')
human_files = glob(PATH_HUMAN + '/*/*')
# 打乱数据
random.seed(818)
random.shuffle(human_files)

# 打印并保存数据集的数据量
print('\nThere are %d total human images.' % len(human_files))
np.save('./data/input_human_files.npy', human_files)
