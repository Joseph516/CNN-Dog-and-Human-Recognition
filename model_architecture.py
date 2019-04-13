from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.layers import Input
from keras.models import Sequential, Model
import tensorflow as tf
from keras.applications.mobilenetv2 import MobileNetV2
from keras.optimizers import Adam
import numpy as np
import os

class ModelArchitec():
    def __init__(self, size_row=224, size_column=224, labels=133):
        self.size_row = size_row  # 训练图像尺寸
        self.size_column = size_column
        self.labels = labels  # 分类总标签

    def modelCnn(self):
        print("Model building!")
        model = Sequential()

        model.add(Conv2D(filters=16, kernel_size=2, activation='relu',
                         input_shape=(self.size_row, self.size_column, 3)))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Conv2D(filters=32, kernel_size=2,
                         padding='valid', activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Conv2D(filters=64, kernel_size=2,
                         padding='valid', activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Flatten())

        model.add(Dense(self.labels, activation='softmax'))

        model.add(Dropout(0.2))

        model.summary()

        # 编译模型
        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy', metrics=['accuracy'])

        print("Model built!")
        return model


def getModelCnn(size_row, size_column, labels):
    print("Model building!")

    inp = Input(shape=(size_row, size_column, 3))

    x = Conv2D(filters=16, kernel_size=2, activation='relu')(inp)
    x = MaxPooling2D(pool_size=2)(x)
    x = Conv2D(filters=32, kernel_size=2,
               padding='valid', activation='relu')(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = Conv2D(filters=64, kernel_size=2,
               padding='valid', activation='relu')(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = Flatten()(x)
    outp = Dense(labels, activation='softmax')(x)
    outp = Dropout(0.2)(outp)

    model = Model(inputs=inp, outputs=outp)
    model.summary()

    # 编译模型
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy', metrics=['accuracy'])
    print("Model built!")
    return model


def getModelMobileNet0(size_row, size_column, labels):
    """重新训练mobileNet"""
    print("Model building!")

    inp = Input(shape=(size_row, size_column, 3))

    mn = MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_tensor=inp,
        input_shape=(size_row, size_column, 3),
        pooling='avg')

    for layer in mn.layers:
        layer.trainable = True  # trainable has to be false in order to freeze the layers

    mn_out = mn.output
    x = Dense(256, activation='relu')(mn_out)
    x = Dropout(0.2)(x)

    outp = Dense(labels, activation='softmax')(x)

    model = Model(inputs=inp, outputs=outp)
    # model.summary()

    # 编译模型
    model.compile(optimizer=Adam(),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    print("Model built!")
    return model

def getModelMobileNet(size_row, size_column, labels):
    print("Model building!")

    inp = Input(shape=(size_row, size_column, 3))

    mn = MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_tensor=inp,
        input_shape=(size_row, size_column, 3),
        pooling='avg')

    # 锁住0-151层。
    for layer in mn.layers[:152]:
        layer.trainable = False
    for layer in mn.layers[152:]:
        layer.trainable = True

    mn_out = mn.output
    x = Dense(256, activation='relu')(mn_out)
    x = Dropout(0.2)(x)

    outp = Dense(labels, activation='softmax')(x)

    model = Model(inputs=inp, outputs=outp)
    model.summary()
    
    # 编译模型
    model.compile(optimizer=Adam(),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    print("Model built!")
    return model

# 使用inceptionV3模型作为迁移学习，减少训练时间。
is_migrate_learn = False
if (is_migrate_learn):
    bottleneck_features = np.load('.model/DogInceptionV3Data.npz')
    train_InceptionV3 = bottleneck_features['train']
    valid_InceptionV3 = bottleneck_features['valid']
    test_InceptionV3 = bottleneck_features['test']

def getModelInceptionV3(size_row, size_column, labels):
    print("Model building!")

    inp = Input(shape=train_InceptionV3.shape[1:])

    x = GlobalAveragePooling2D(inp)
    x = Dropout(0.3)(x)

    x = Dense(256, activation='relu')(mn_out)
    x = Dropout(0.2)(x)

    outp = Dense(labels, activation='softmax')(x)

    model = Model(inputs=inp, outputs=outp)

    # for layer in model.layers[:25] 
    #     layer.trainable = False # 可设置模型指定层不训练
    # model.summary()

    # 编译模型
    model.compile(optimizer=Adam(),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    print("Model built!")
    return model


def getBestModelInDir(path):
    """获取文件夹中loss最小的模型"""
    model_files = os.listdir(path)

    best_model = ''
    mini_valid_loss = 100
    for model_file in model_files:
        # valid_loss = float(model_file.split('-')[1].split('.h')[0])
        valid_loss = float(model_file.split('-')[1])
        if (valid_loss < mini_valid_loss):
            mini_valid_loss = valid_loss
            best_model = model_file
    if (mini_valid_loss == 100):
            raise("No good model!")
    best_model = os.path.join(path, best_model)
    return best_model


if __name__ == "__main__":
    model = getModelMobileNet(224, 224, 133)
