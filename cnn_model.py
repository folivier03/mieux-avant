
#  Librairies Import

import os
import sys
import glob as gb
import pandas as pd
import numpy as np
import tensorflow as tf
import cv2
import keras

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from PIL import Image

from keras.models import Model
from keras.preprocessing import image

import matplotlib.pyplot as plt
from keras.models import load_model

import mlflow
import mlflow.keras

# getclasse function
val = 0
classes ={}
trainpath = './data/train' 
for folder in  os.listdir(trainpath) : 
    classes[folder] = val
    val = val+1
#fonction getclasses
def getclasses(n):
    for x , y in classes.items() : 
        if n == y : 
            return x

print(classes)


#  define the path and open folders

# train and validation folder
trainpath = './data/train' 
for folder in  os.listdir(trainpath) : 
    files = gb.glob(pathname= str( trainpath +'/' + folder + '/*.png'))
    print(f'For data , found {len(files)} in folder {folder}')


# prediction folder
predictpath = './data/prediction' 
files = gb.glob(pathname= str( predictpath +'/*.png'))
print(f'For Prediction data , found {len(files)}')
   

#train
size_train = []
for folder in  os.listdir(trainpath) : 
    files = gb.glob(pathname= str( trainpath +'/' + folder + '/*.png'))
    for file in files: 
        image = plt.imread(file)
        size_train.append(image.shape)
pd.Series(size_train).value_counts()


# prediction
# size_predict = []
# files = gb.glob(pathname= str( predictpath +'/*.png'))
# for file in files: 
#     image = plt.imread(file)
#     size_predict.append(image.shape)
# pd.Series(size_predict).value_counts()


#  Data Split

image_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(validation_split=0.2)
TRAIN_DATA_DIR = './data/train'
TRAIN_IMAGE_SIZE = 32
TRAIN_BATCH_SIZE = 32
train_generator = image_data_generator.flow_from_directory(
    TRAIN_DATA_DIR,
    target_size=(TRAIN_IMAGE_SIZE, TRAIN_IMAGE_SIZE),
    batch_size=TRAIN_BATCH_SIZE,
    class_mode='categorical',
    subset='training' )
validation_generator = image_data_generator.flow_from_directory(
    TRAIN_DATA_DIR, # same directory as training data
    target_size=(TRAIN_IMAGE_SIZE, TRAIN_IMAGE_SIZE),
    batch_size=TRAIN_BATCH_SIZE,
    class_mode='categorical',
    subset='validation')


#  Images reading and showing

X_train = []
y_train = []
for folder in  os.listdir(trainpath) : 
    files = gb.glob(pathname= str( trainpath +'/' + folder + '/*.png'))
    for file in files: 
        image = cv2.imread(file)
        X_train.append(list(image))



# prediction images
X_pred = []
predictpath = './data/prediction'
files = gb.glob(pathname= str( predictpath +'/*.png'))
for file in files: 
    image_array = cv2.imread(file)
    X_pred.append(list(image_array))
X_pred_array = np.array(X_pred)


# Building Model 


#  Model 3

model_3 = Sequential()
model_3.add(Conv2D(32, kernel_size=3, kernel_initializer = 'random_uniform', bias_initializer = 'zeros', activation='relu', padding='same', input_shape=(32,32,3))) # 5 ou 7 au dessus de 128
#model_2.add(MaxPooling2D(pool_size=(2,2)))
model_3.add(Conv2D(64, kernel_size=3, activation='relu', padding='same'))
model_3.add(MaxPooling2D(pool_size=(2,2)))

model_3.add(Conv2D(96, kernel_size=3, activation='relu', padding='same'))
model_3.add(Conv2D(128, kernel_size=3, activation='relu', padding='same'))
model_3.add(MaxPooling2D(pool_size=(2,2)))
model_3.add(Dense(128, activation='relu'))

#
model_3.add(Flatten())
#model_2.add(Dense(64, activation='relu'))
model_3.add(Dropout(0.2))
model_3.add(Dense(5, activation='softmax'))

model_3.summary()

# model compile and model training
model_3.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
#training = model_3.fit_generator(train_generator, epochs=50, callbacks=[es_callback], validation_data=(validation_generator))

#parameters MLflow

epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 1
optimizer_name = string(sys.argv[2]) if len(sys.argv) > 2 else 'Adam'
# #
#model MLflow
mlflow.keras.autolog()
#
def train(epochs, optimizer_name):
    with mlflow.start_run(run_name ='tracking cnn model') as run:
        model_3.compile(optimizer = optimizer_name , loss='categorical_crossentropy', metrics=['accuracy'])
        results = model_3.fit_generator(train_generator, epochs= epochs, callbacks=[es_callback], validation_data=(validation_generator))
        return(run.info.experiment_id, run.info.run_id)

# for epochs in [1, 2]:

train(epochs,optimizer_name)

# save model
model_3.save( 'model_3.h5')


test_model = load_model('model_3.h5')
image_test= "./data/prediction/0001.png"
u_pred= []
image_array = cv2.imread(image_test)
u_pred.append(list(image_array))
u_pred_array = np.array(u_pred)
result = test_model.predict(u_pred_array)
print('Prediction Shape is {}'.format(result.shape))
print(result)
#
plt.figure(figsize=(5,5))
plt.imshow(u_pred[0])
plt.title(getclasses(np.argmax(result[0])))










