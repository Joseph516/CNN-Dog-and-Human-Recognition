from keras.callbacks import ModelCheckpoint
import os
import numpy as np
from model_architecture import getModelMobileNet
from global_path import *
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# train data
# dog_train_tensors = np.load(PATH_TRAIN_TENSORS)
# dog_train_targets = np.load(PATH_TRAIN_TARGETS)

# valid data
# dog_valid_tensors = np.load(PATH_VALID_TENSORS)
# dog_valid_targets = np.load(PATH_VALID_TARGETS)

# imaage augment
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

batch_size = 16

# get augmented images
train_generator = train_datagen.flow_from_directory(
    directory=PATH_TRAIN,
    target_size=(224, 224),
    batch_size=batch_size,
    shuffle=True, seed=818,
    class_mode='categorical')

valid_generator = train_datagen.flow_from_directory(
    directory=PATH_VALID,
    target_size=(224, 224),
    batch_size=batch_size,
    shuffle=True, seed=818,
    class_mode='categorical')

# callbacks
os.makedirs(PATH_DOG_MODEL, exist_ok=True)
checkpointer = ModelCheckpoint(filepath=PATH_DOG_MODEL+'weights.{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}.hdf5',
                               monitor='val_loss', verbose=0, save_best_only=True)

# load train model
model = getModelMobileNet(size_row=224, size_column=224, labels=133)

# train
epochs = 20
print("Training!")

# model.fit(dog_train_tensors, dog_train_targets,
#           validation_data=(dog_valid_tensors, dog_valid_targets),
#           epochs=epochs, batch_size=20, callbacks=[checkpointer], verbose=1)

model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=valid_generator,
    validation_steps=valid_generator.samples // batch_size,
    epochs=epochs,
    callbacks=[checkpointer],
    verbose=1
)
print("Trained!")
