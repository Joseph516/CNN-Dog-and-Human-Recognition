from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
import numpy as np
from tqdm import tqdm
import json

def pathToTensor(img_path, size_row=224, size_columns=224):
    """将单个图像转化为向量输入"""
    # 用PIL加载RGB图像为PIL.Image.Image类型
    # keras CNN输入维度：(nb_samples, rows, columns, channels)
    img = image.load_img(img_path, target_size=(size_row, size_columns))
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
