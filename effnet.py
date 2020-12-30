import tensorflow as tf

import pandas as pd
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.applications.resnet50 import ResNet50
from keras.layers import GaussianNoise
from keras.layers import Dropout
from keras.layers import BatchNormalization
from glob import glob
import numpy as np
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import os
import efficientnet.keras as efn
from keras import Model
from WarmUpLearningRateScheduler import WarmUpLearningRateScheduler

training_path = r'C:\Users\chinm\PycharmProjects\ISM\ISIC_2019_Training_Input\ISIC_2019_Training_Input'
validation_path = r'C:\Users\chinm\PycharmProjects\ISM\groundtruth_val.csv'

Train_path =  r'C:\Users\chinm\PycharmProjects\ISM\groundtruth_train.csv'
#train_data = Train_df['image']


train_data = pd.read_csv(Train_path, encoding='latin1', dtype={"image": str, "MEL": int, "NV": int, "BCC": int, "AK": int,"BKL": int, "": int, "DF": int, "VASC": int,"SCC": int, "UNK": int})
train_data['image'] =train_data['image']+'.jpg'
labels = np.argmax(np.array(train_data.iloc[:,1:10]), axis=1)
images = np.asarray(train_data.iloc[:, 0])
classes = list(train_data.columns.values[1:10])

print(classes)
print(images)
print(labels)


validation_data = pd.read_csv(validation_path, encoding='latin1', dtype={"image": str, "MEL": int, "NV": int, "BCC": int, "AK": int,"BKL": int, "": int, "DF": int, "VASC": int,"SCC": int, "UNK": int})
validation_data['image'] =validation_data['image']+'.jpg'
val_labels = np.argmax(np.array(validation_data.iloc[:,1:10]), axis=1)
val_images = np.asarray(validation_data.iloc[:, 0])
val_classes = list(validation_data.columns.values[1:10])
print(val_classes)
print(val_images)
print(val_labels)


base_model = efn.EfficientNetB6(input_shape=(256,256,3),
                                weights='imagenet',
                                include_top=False,
                                pooling='avg')
x = Dropout(0.3)(base_model.output)    # adding Droupout layer to the model.
prediction_efn = Dense(9, activation='softmax')(x)
model = Model(base_model.input, prediction_efn)
#compiling the CNN
model.compile(optimizer ='adam', loss ='categorical_crossentropy', metrics =['accuracy'])


callbacks_save = ModelCheckpoint('best isic.h5',
                                 monitor='val_loss',
                                 mode='min',
                                 save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=1e-6, cooldown=1, verbose=1)

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range=0.3, rotation_range=30,
                                   width_shift_range=0.3, height_shift_range=0.3,
                                   brightness_range=[0.2, 1.0], horizontal_flip=True,
                                   vertical_flip=True, fill_mode='nearest', zoom_range=0.4
                                   )

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_dataframe(train_data,
                                                 directory= training_path,
                                                 x_col= 'image',
                                                 y_col= list(classes),
                                                 target_size =(256,256),
                                                 batch_size = 4,
                                                 class_mode ='raw',
                                                 validate_filenames=False

)

test_set = test_datagen.flow_from_dataframe(validation_data,
                                                  directory= training_path,
                                                  x_col= 'image',
                                                  y_col= list(val_classes),
                                                  target_size =(256,256),
                                                  batch_size = 4,
                                                  class_mode ='raw',
                                                  validate_filenames=False

 )

model.fit_generator(training_set,
                         steps_per_epoch =len(training_set)/4,
                         epochs =15,
                         validation_data = test_set,
                         validation_steps =len(test_set)/4,
                         callbacks=[callbacks_save, reduce_lr])

model.save("effnettrial.h5")