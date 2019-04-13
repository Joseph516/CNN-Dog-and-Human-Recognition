from keras.models import load_model
import os
import numpy as np
from global_path import *
from model_architecture import getBestModelInDir

# load test data
dog_test_tensors = np.load(PATH_TEST_TENSORS)
dog_test_targets = np.load(PATH_TEST_TARGETS)

# load minium loss model
best_model_path = getBestModelInDir(PATH_DOG_MODEL)

model = load_model(filepath=best_model_path)

print("Testing! The best model is:", best_model_path)
# 获取测试数据集中每一个图像所预测的狗品种的index
preds = [np.argmax(model.predict(np.expand_dims(tensor, axis=0)))
         for tensor in dog_test_tensors]
# preds_array = model.predict_generator(test_generator, steps=test_generator.samples//batch_size)
# preds = [np.argmax(pred) for pred in preds_array]

# 报告测试准确率
test_accuracy = 100*np.sum(np.array(preds) ==
                           np.argmax(dog_test_targets, axis=1))/len(preds)

print('Test accuracy: %.4f%%' % test_accuracy)