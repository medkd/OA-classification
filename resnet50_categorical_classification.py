# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 22:14:19 2022

@author: user
"""

import numpy as np
from tensorflow.keras.optimizers import Adam
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import ResNet50
import os
import tensorflow as tf
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow import keras
import coral_ordinal as coral

def resnet50():
    inp = tf.keras.layers.Input((224,224,3))
    res_out = ResNet50(include_top=False, pooling='avg', weights='imagenet')(inp)
    classification_out = tf.keras.layers.Dense(5, activation='softmax')(res_out)
    mod = tf.keras.models.Model(inputs = inp, outputs = classification_out)
    return mod



class noise_layer(tf.keras.layers.Layer):
    def __init__(self, sigma):
        super(noise_layer, self).__init__()
        self.sigma = sigma
    
    def call(self, x):
    
        noise_x = x + K.random_normal(shape = K.shape(x), mean = 0.0, stddev = self.sigma)
        return noise_x

class contrast_layer(tf.keras.layers.Layer):
    def __init__(self, level):
        super(contrast_layer, self).__init__()
        self.level = level
    def call(self,x):
        contrast_x = x*(1+self.level)
        return contrast_x


img_augmentation_noise = Sequential(
    [
        
        noise_layer(0.15), 
        
    ],
    name="img_augmentation_noise",
)


img_augmentation = Sequential(
    [
        preprocessing.RandomRotation(factor=0.05),
        preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),        
        preprocessing.RandomFlip('horizontal'),
        #preprocessing.RandomContrast(factor=0.5),
        #noise_layer(0.2)
    ],
    name="img_augmentation",
)

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(600,800), n_channels=3,
                 n_classes=29, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        
        
        Y = np.empty((self.batch_size))
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            
            
            
            if i%5 == 0:
            #f = gzip.GzipFile(ID,'r')
                arr_original = np.load(ID)['arr_0']/255.0 # 224,224,3
                arr_noise = np.squeeze(img_augmentation_noise(np.expand_dims(arr_original,0))) # 224,224,3
                
                
                
                X[i,] = arr_noise
                
                
                Y[i] = self.labels[ID]
            
            elif i%5 == 1:
                arr_original = np.load(ID)['arr_0']/255.0 # 224,224,3
                arr_noise = np.squeeze(img_augmentation(np.expand_dims(arr_original,0))) # 224,224,3
                X[i,] = arr_noise
                Y[i] = self.labels[ID]
            
            else:
                arr_original = np.load(ID)['arr_0']/255.0 # 224,224,3                
                X[i,] = arr_noise           
                
                Y[i] = self.labels[ID]
        
        return X, (keras.utils.to_categorical(Y, num_classes=self.n_classes))
    

params = {'dim': (224,224),
          'batch_size':8,
          'n_classes': 5,
          'n_channels': 3,
          'shuffle': True}

os.chdir('C:\\Users\\user\\Desktop\\OA_project\\KneeXrayData\\merge')
lst_merge = os.listdir('C:\\Users\\user\\Desktop\\OA_project\\KneeXrayData\\merge') 
lst_merge.sort()

lst_for_train = os.listdir('C:\\Users\\user\\Desktop\\OA_project\\KneeXrayData\\train') 
lst_for_train.sort()
len(lst_for_train) # 921



lst_for_val = os.listdir('C:\\Users\\user\\Desktop\\OA_project\\KneeXrayData\\val')
lst_for_val.sort()
len(lst_for_val) # 296

lst_for_test = os.listdir('C:\\Users\\user\\Desktop\\OA_project\\KneeXrayData\\test')
lst_for_test.sort()
len(lst_for_test) # 287


partition = {}
partition['train'] = lst_for_train
partition['val'] = lst_for_val
partition['test'] = lst_for_test

len(partition['train']) # 921




# 5 class
mr_dict = {}

for i in range(len(lst_merge)):
    mr_dict[lst_merge[i]] = lst_merge[i][0]

mr_dict['4_val_299_26.npz']





label = mr_dict
label

partition = partition
label = label

training_generator = DataGenerator(partition['train'], label, **params)
validation_generator = DataGenerator(partition['val'], label, **params)



mod = resnet50()

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,patience=6, min_lr=1e-7)
checkpoint_filepath = 'resnet_categorical.h5'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join('C:\\Users\\user\\Desktop\\OA_project\\KneeXrayData\\best_model',checkpoint_filepath),
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=False)


csv_logger = tf.keras.callbacks.CSVLogger(os.path.join('C:\\Users\\user\\Desktop\\OA_project\\KneeXrayData\\log',  'resnet_categorical.log'),append=False)


earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, verbose=0, mode='min')


mod.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-5),loss='categorical_crossentropy', metrics=['acc'])
mod.fit(training_generator,validation_data = validation_generator,epochs = 1000,callbacks = [reduce_lr,csv_logger, earlyStopping, model_checkpoint_callback])


































