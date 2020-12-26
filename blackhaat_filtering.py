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
import os


training_path = os.path.join(os.getcwd(),'C:\\Users\\chinm\\PycharmProjects\\ISM\\ISIC_2019_Training_Input\\')
validation_path =  os.path.join(os.getcwd(),'C:\\Users\\chinm\\PycharmProjects\\ISM\\groundtruth_val.csv')

Train_path =  os.path.join(os.getcwd(),'C:\\Users\\chinm\\PycharmProjects\\ISM\\groundtruth_train.csv')
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




resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=(512, 512, 3))

for layer in resnet_model.layers:
  layer.trainable = False

model = Sequential()

model.add(resnet_model)
model.add(Conv2D(32,(3,3),input_shape=(224,224,3),activation='relu'))
model.add(MaxPooling2D((2,2),strides=2))
model.add(Flatten())
model.add(GaussianNoise(0.1))
model.add(Dense(256, activation='relu'))
model.add(Dropout(.25))
model.add(BatchNormalization())
model.add(Dense(units =8, activation = 'sigmoid'))

#compiling the CNN
model.compile(optimizer ='adam', loss ='binary_crossentropy', metrics =['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range =0.2,
                                   horizontal_flip =True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_dataframe(train_data,
                                                 directory= training_path,
                                                 x_col= 'image',
                                                 y_col= list(classes),


                                                 target_size =(512,512),
                                                 batch_size = 4,
                                                 class_mode ='raw',
                                                 validate_filenames=False

)

test_set = test_datagen.flow_from_dataframe(validation_data,
                                                  directory= training_path,
                                                  x_col= 'image',
                                                  y_col= list(val_classes),


                                                  target_size =(512,512),
                                                  batch_size = 4,
                                                  class_mode ='raw',
                                                  validate_filenames=False

 )

model.fit_generator(training_set,
                         steps_per_epoch =8000,
                         epochs =15,
                         validation_data = test_set,
                         validation_steps =2000)

model.save("trial.h5")