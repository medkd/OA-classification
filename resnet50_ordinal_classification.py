# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 23:31:28 2022

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
from tensorflow.keras import layers
def resnet50_ordinal():
    inp = tf.keras.layers.Input((224,224,3))
    res_out = ResNet50(include_top=False, pooling='avg', weights='imagenet')(inp)
    res_out = tf.keras.layers.Dense(32, activation = "relu")(res_out)
    
    classification_out = coral.CoralOrdinal(5, name = 'ordinal')(res_out)
    mod = tf.keras.models.Model(inputs = inp, outputs = classification_out)
    return mod


class contrast_layer(tf.keras.layers.Layer):
    def __init__(self, level):
        super(contrast_layer, self).__init__()
        self.level = level
    def call(self,x):
        contrast_x = x*(1+self.level)
        return contrast_x


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
        preprocessing.RandomRotation(factor = 0.05),
        preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),        
        #preprocessing.RandomFlip('horizontal'),
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
        
        
        z = np.empty((self.batch_size))
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            
            
            if i%6 == 0:
            
                arr_original = np.load(ID)['arr_0']/255.0 # 128/128/10/7 # arr
                
                
                arr_noise = img_augmentation_noise(np.expand_dims(arr_original,0)) # 1,20,256,256,3
                arr_noise = np.squeeze(arr_noise) # 20,256,256,3
                
                
                X[i,] = arr_noise
                
                #y[i,] = cv2.resize(arr, (400,300), interpolation=cv2.INTER_CUBIC)
                z[i] = self.labels[ID]
            
            
            
            if i%6 == 1:
            
                arr_original = np.load(ID)['arr_0']/255.0 # 128/128/10/7 # arr
                
                
                
                arr_aug = img_augmentation(np.expand_dims(arr_original,0)) # 1,20,256,256,3
                arr_aug = np.squeeze(arr_aug) # 20,256,256,3
                
                
                X[i,] = arr_aug
                
                #y[i,] = cv2.resize(arr, (400,300), interpolation=cv2.INTER_CUBIC)
                z[i] = self.labels[ID]
            
            else:
                arr_original = np.load(ID)['arr_0']/255.0 # 128/128/10/7 # arr
                
                X[i,] = arr_original
               
                
                z[i] = self.labels[ID]
                
            
            
                       
        #return (X,keras.utils.to_categorical(z, num_classes=self.n_classes)), (keras.utils.to_categorical(z, num_classes=self.n_classes))
        return (X), (keras.utils.to_categorical(z, num_classes=self.n_classes))



params = {'dim': (224,224),
          'batch_size':8,
          'n_classes': 5,
          'n_channels': 3,
          'shuffle': True}

os.chdir('C:\\Users\\user\\Desktop\\OA_project')
lst_merge = os.listdir('C:\\Users\\user\\Desktop\\OA_project\\KneeXrayData\\merge') 
lst_merge.sort()

lst_for_train = os.listdir('C:\\Users\\user\\Desktop\\OA_project\\KneeXrayData\\train') 
lst_for_train.sort()
len(lst_for_train) # 921

file = open('./train_list.txt', 'w')
for fp in lst_for_train:
    file.write(str(fp))
    file.write('\n')
file.close()
print('保存问题文档成功！')


file = open('./train_list.txt', 'r')
lst_for_train = []
for line in file.readlines():
    line = line.strip('\n')
    lst_for_train.append(line)
file.close()
print('读取文档成功！')


lst_for_val = os.listdir('C:\\Users\\user\\Desktop\\OA_project\\KneeXrayData\\val')
lst_for_val.sort()
len(lst_for_val) # 296

file = open('./val_list.txt', 'w')
for fp in lst_for_val:
    file.write(str(fp))
    file.write('\n')
file.close()
print('保存问题文档成功！')

lst_for_test = os.listdir('C:\\Users\\user\\Desktop\\OA_project\\KneeXrayData\\test')
lst_for_test.sort()
len(lst_for_test) # 287

file = open('./test_list.txt', 'w')
for fp in lst_for_test:
    file.write(str(fp))
    file.write('\n')
file.close()
print('保存问题文档成功！')


partition = {}
partition['train'] = lst_for_train
partition['val'] = lst_for_val
partition['test'] = lst_for_test

len(partition['train']) # 921




# 5 class
mr_dict = {}

for i in range(len(lst_merge)):
    mr_dict[lst_merge[i]] = lst_merge[i][0]





label = mr_dict
label

partition = partition
label = label

training_generator = DataGenerator(partition['train'], label, **params)
validation_generator = DataGenerator(partition['val'], label, **params)

mod = resnet50_ordinal()

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,patience=20, min_lr=1e-5)
checkpoint_filepath = 'resnet_ordinal.h5'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join('C:\\Users\\user\\Desktop\\OA_project\\KneeXrayData\\best_model',checkpoint_filepath),
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=False)



def ordinal(y_true, y_pred):
    # coral的輸出已經是累積機率
    # 所以只需要
    

    
    tensor_probs = tf.keras.layers.Activation('sigmoid')(y_pred)
    tensor_probs = tf.where(tensor_probs>0.5,1,0)
    
    pred_labels = tf.math.count_nonzero(tensor_probs, axis = -1)
    
    
    y_pred_categorical = tf.one_hot(pred_labels, 5)
    
    
    score = tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred_categorical)
    
    
    return score


csv_logger = tf.keras.callbacks.CSVLogger(os.path.join('C:\\Users\\user\\Desktop\\OA_project\\KneeXrayData\\log',  'resnet_ordinal.log'),append=False)




mod.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3),loss=coral.OrdinalCrossEntropy(num_classes = 5), metrics=[coral.MeanAbsoluteErrorLabels(), ordinal])
mod.fit(training_generator,validation_data = validation_generator,epochs = 1000,callbacks = [reduce_lr,csv_logger, model_checkpoint_callback])


out = mod

# label prediction
import matplotlib.pyplot as plt
from scipy import special
import pandas as pd

len(lst_for_val)

print("Predict on test dataset")
test_dataset = np.zeros((len(lst_for_val),224,224,3))

for i in range(len(lst_for_val)):
    test_dataset[i] = np.load(lst_for_val[0])['arr_0']/255

plt.imshow(test_dataset[0])
# Note that these are ordinal (cumulative) logits, not probabilities or regular logits.
ordinal_logits = mod.predict(test_dataset)
cum_probs = pd.DataFrame(ordinal_logits).apply(special.expit)
cum_probs.head()

labels2 = cum_probs.apply(lambda x: x > 0.5).sum(axis = 1)
labels2.head()


# true
val_answer = []
for i in range(len(lst_for_val)):
    val_answer.append(int(lst_for_val[i][0]))
pd.DataFrame(val_answer)

np.mean(labels2 == val_answer)























