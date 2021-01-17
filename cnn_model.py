mieux# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ## Librairies Import

# %%
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


# %%
val = 0
classes ={}
trainpath = '../data/train' 
for folder in  os.listdir(trainpath) : 
    classes[folder] = val
    val = val+1
#fonction getclasses
def getclasses(n):
    for x , y in classes.items() : 
        if n == y : 
            return x

print(classes)

# %% [markdown]
# ## define the path and open folders

# %%
# train and validation folder
trainpath = '../data/train' 
for folder in  os.listdir(trainpath) : 
    files = gb.glob(pathname= str( trainpath +'/' + folder + '/*.png'))
    print(f'For data , found {len(files)} in folder {folder}')


# %%
# prediction folder
predictpath = '../data/prediction' 
files = gb.glob(pathname= str( predictpath +'/*.png'))
print(f'For Prediction data , found {len(files)}')
   

# %% [markdown]
# ## Checking Images

# %%
#train
size_train = []
for folder in  os.listdir(trainpath) : 
    files = gb.glob(pathname= str( trainpath +'/' + folder + '/*.png'))
    for file in files: 
        image = plt.imread(file)
        size_train.append(image.shape)
pd.Series(size_train).value_counts()


# %%
# prediction
size_predict = []
files = gb.glob(pathname= str( predictpath +'/*.png'))
for file in files: 
    image = plt.imread(file)
    size_predict.append(image.shape)
pd.Series(size_predict).value_counts()

# %% [markdown]
# ## Data Split

# %%
image_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(validation_split=0.2)
TRAIN_DATA_DIR = '../data/train'
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

# %% [markdown]
# ## Images reading and showing

# %%
X_train = []
y_train = []
for folder in  os.listdir(trainpath) : 
    files = gb.glob(pathname= str( trainpath +'/' + folder + '/*.png'))
    for file in files: 
        image = cv2.imread(file)
        X_train.append(list(image))


# %%
plt.figure(figsize=(10,10))
for n , i in enumerate(list(np.random.randint(0,len(X_train),25))) : 
    plt.subplot(5,5,n+1)
    plt.imshow(X_train[i])   
    plt.axis('off')
    #plt.title(getclasses(y_train[i]))
   


# %%
# prediction images
X_pred = []
predictpath = '../data/prediction'
files = gb.glob(pathname= str( predictpath +'/*.png'))
for file in files: 
    image_array = cv2.imread(file)
    X_pred.append(list(image_array))
X_pred_array = np.array(X_pred)

# %% [markdown]
# ## Building Model 

# %%
model_1 = Sequential()
model_1.add(Conv2D(32, kernel_size=3, kernel_initializer = 'random_uniform', bias_initializer = 'zeros', activation='relu', padding='same', input_shape=(32,32,3))) 
model_1.add(MaxPooling2D(pool_size=(2,2)))
model_1.add(Conv2D(64, kernel_size=3,  activation='relu', padding='same'))
model_1.add(MaxPooling2D(pool_size=(2,2)))
model_1.add(Dense(128, activation='relu'))
#
model_1.add(Conv2D(72, kernel_size=5, activation='relu'))
#model_1.add(MaxPooling2D(pool_size=(2,2)))
model_1.add(Dense(130, activation='relu'))

#
model_1.add(Flatten())
model_1.add(Dense(64, activation='relu'))
model_1.add(Dense(5, activation='softmax'))

model_1.summary()


# %%
model_1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
training = model_1.fit_generator(train_generator, epochs=20, callbacks=[es_callback], validation_data=(validation_generator))

# %% [markdown]
# ## Loss and Accuracy Graphs

# %%
# summarize history for loss
plt.plot(training.history['loss'])
plt.plot(training.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
#
# summarize history for accuracy
plt.plot(training.history['accuracy'])
plt.plot(training.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

# %% [markdown]
# ## Model 2 

# %%
model_2 = Sequential()
model_2.add(Conv2D(32, kernel_size=3, kernel_initializer = 'random_uniform', bias_initializer = 'zeros', activation='relu', padding='same', input_shape=(32,32,3))) # 5 ou 7 au dessus de 128
model_2.add(Conv2D(64, kernel_size=3, activation='relu', padding='same'))
model_2.add(MaxPooling2D(pool_size=(2,2)))

model_2.add(Conv2D(76, kernel_size=3, activation='relu', padding='same'))
model_2.add(Conv2D(96, kernel_size=3, activation='relu', padding='same'))
model_2.add(MaxPooling2D(pool_size=(2,2)))

model_2.add(Conv2D(128, kernel_size=3, activation='relu', padding='same'))
model_2.add(Conv2D(150, kernel_size=3, activation='relu', padding='same'))
model_2.add(MaxPooling2D(pool_size=(2,2)))
model_2.add(Dense(128, activation='relu'))
model_2.add(Dropout(0.2))
#
# model_2.add(Conv2D(64, kernel_size=3, activation='relu'))
# model_2.add(MaxPooling2D(pool_size=(2,2)))
# model_2.add(Dense(128, activation='relu'))

#
model_2.add(Flatten())
model_2.add(Dense(64, activation='relu'))
model_2.add(Dense(5, activation='softmax'))

model_2.summary()


# %%
model_2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
training = model_1.fit_generator(train_generator, epochs=50, callbacks=[es_callback], validation_data=(validation_generator))


# %%
# summarize history for loss
plt.plot(training.history['loss'])
plt.plot(training.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
#
# summarize history for accuracy
plt.plot(training.history['accuracy'])
plt.plot(training.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

# %% [markdown]
# ## Model 3

# %%
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


# %%
model_3.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
training = model_3.fit_generator(train_generator, epochs=50, callbacks=[es_callback], validation_data=(validation_generator))


# %%
# summarize history for loss
plt.plot(training.history['loss'])
plt.plot(training.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
#
# summarize history for accuracy
plt.plot(training.history['accuracy'])
plt.plot(training.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')


# %%

model_3.save( 'model_3.h5')


# %%
test_model = load_model('model_3.h5')
image_test= "../data/prediction/0001.png"
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



# test_image_test=cv2.resize(test_image_test,(32,32))
# test_image_test = test_image_test.reshape(1,32,32,3)




# %%


