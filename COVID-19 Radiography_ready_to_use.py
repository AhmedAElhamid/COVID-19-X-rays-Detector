#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
from os import walk


# In[2]:


path = "COVID-19"


# In[5]:


import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Convolution2D,SeparableConv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.layers import Input, Lambda, Conv2D, BatchNormalization, LeakyReLU, ZeroPadding2D, UpSampling2D
IMG_SIZE=224
def conv_block(clf,filters,kernel_size):
    clf.add(SeparableConv2D(filters=filters, kernel_size=kernel_size, padding='same'))
    clf.add(BatchNormalization(epsilon=0.001))
    clf.add(LeakyReLU(alpha=0.1))
    clf.add(SeparableConv2D(filters=filters, kernel_size=kernel_size, padding='same'))
    clf.add(BatchNormalization(epsilon=0.001))
    clf.add(LeakyReLU(alpha=0.1))
    clf.add(MaxPooling2D(pool_size=2))
    return clf

def dense_block(clf,units, dropout_rate):
    
    clf.add(Dense(units,activation='relu'))
    clf.add(BatchNormalization(epsilon=0.001))
    clf.add(Dropout(dropout_rate))
    
    return clf


# In[7]:


def build_model():
    clf = Sequential()
    clf.add(Convolution2D(filters=16, kernel_size=5, padding='same', input_shape=(224,224,1)))
    clf.add(Convolution2D(filters=16, kernel_size=5, padding='same'))
    clf.add(MaxPooling2D(pool_size=2))
    clf=conv_block(clf,32,5)
    clf=conv_block(clf,64,5)
    
    clf=conv_block(clf,128,5)
    clf.add(Dropout(.2))
    
    clf=conv_block(clf,256,5)
    clf.add(Dropout(.2))
    
    clf=conv_block(clf,256,5)
    clf.add(Dropout(.2))
    
    clf.add(Flatten())
    clf=dense_block(clf,2304, 0.3)
    clf=dense_block(clf,512, 0.3)
    clf=dense_block(clf,128, 0.3)
    
    clf.add(Dense(3, activation='softmax'))

    return clf
model=build_model()


# In[9]:


from keras.models import load_model
model.load_weights('COVID-19_Radiography_Classifier.h5')


# In[10]:


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[21]:


IMG_SIZE=224
img_name=[]
images=[]
def test_images(path):
    for img in tqdm(os.listdir(path)): 
        try:
            img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  
            images.append(new_array)
            img_name.append(img)
        except Exception as e: 
            pass
        
    for n in range (len(images)):
        pred=model.predict(np.array(images[n]).reshape(1, IMG_SIZE, IMG_SIZE, 1))
        label=np.argmax(pred,axis=1)
        if label[0] == 0:
            print(img_name[n],'COVID-19')
        elif label[0] == 1:
            print(img_name[n],'Normal')
        elif label[0] == 2:
            print(img_name[n],'Viral Pneumonia') 


# In[22]:


test_images(path)


# In[ ]:




