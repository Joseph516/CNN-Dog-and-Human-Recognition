import os

DIR_ROOT = os.path.dirname(os.path.abspath("__file__"))

# input data
PATH_TRAIN = os.path.join(DIR_ROOT, 'data/dogImages/train')
PATH_VALID = os.path.join(DIR_ROOT, 'data/dogImages/valid')
PATH_TEST = os.path.join(DIR_ROOT, 'data/dogImages/test')

# data tensors
DIR_TENSORS = './data/tensors'
PATH_TRAIN_TENSORS = os.path.join(DIR_TENSORS, 'dog_train_tensors.npy')
PATH_TRAIN_TARGETS = os.path.join(DIR_ROOT, 'data/input_dog_train_targets.npy')
PATH_VALID_TENSORS = os.path.join(DIR_TENSORS, 'dog_valid_tensors.npy')
PATH_VALID_TARGETS = os.path.join(DIR_ROOT, 'data/input_dog_valid_targets.npy')
PATH_TEST_TENSORS = os.path.join(DIR_TENSORS, 'dog_test_tensors.npy')
PATH_TEST_TARGETS = os.path.join(DIR_ROOT, 'data/input_dog_test_targets.npy')

# data generators for image augment
DIR_GENERATORS = './data/data_generators'
PATH_TRAIN_GENERATORS = os.path.join(DIR_GENERATORS, 'dog_train_generator.npy')
PATH_VALID_GENERATORS = os.path.join(DIR_GENERATORS, 'dog_valid_generator.npy')
PATH_TEST_GENERATORS = os.path.join(DIR_GENERATORS, 'dog_test_generator.npy')


# model weights
PATH_DOG_MODEL = os.path.join(DIR_ROOT, 'model/dog_detector/')
