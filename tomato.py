#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
import PIL
import pathlib


# In[3]:


get_ipython().system('cp -rf ../input/tomato-palnts/plant-village/plant-village/PlantVillage/Tomato_Early_blight ./Tomato_Early_blight')


# In[4]:


get_ipython().system('cp -rf ../input/tomato-palnts/plant-village/plant-village/PlantVillage/Tomato_Bacterial_spot ./Tomato_Bacterial_spot')


# In[5]:


get_ipython().system('cp -rf ../input/tomato-palnts/plant-village/plant-village/PlantVillage/Tomato_Late_blight ./Tomato_Late_blight')


# In[6]:


get_ipython().system('cp -rf ../input/tomato-palnts/plant-village/plant-village/PlantVillage/Tomato_healthy ./Tomato_healthy')


# In[7]:


CURRENT_DIR = os.getcwd()
dataset = pathlib.Path(CURRENT_DIR)
print(dataset)


# In[8]:


healthy_leaf = len(list(dataset.glob('Tomato_healthy/*')))
unhealthy_leaf = len(list(dataset.glob('Tomato_Late_blight/*')))
unhealthy1_leaf = len(list(dataset.glob('Tomato_Bacterial_spot/*')))
unhealthy2_leaf = len(list(dataset.glob('Tomato_Early_blight /*')))
print(f'Number of Tomato_healthy leaf: {healthy_leaf}')
print(f'Number of Tomato_unhealty leaf: {unhealthy_leaf,unhealthy1_leaf,unhealthy2_leaf}')


# In[10]:


IMAGE_SIZE=256
BATCH_SIZE=32
CHANNELS=3
EPOCH=30 
train_ds=tf.keras.preprocessing.image_dataset_from_directory(
    directory=dataset,
     batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset='training',
    seed=42,
    image_size=(IMAGE_SIZE, IMAGE_SIZE)
)


# In[12]:


# Validation dataset
valid_ds = tf.keras.utils.image_dataset_from_directory(
    directory=dataset,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset='validation',
    seed=42,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
)


# In[13]:


class_names= train_ds.class_names
class_names


# In[16]:


plt.figure(figsize=(20,20))
for image_batch,label_batch in train_ds.take(1):
    for i in range(20):
        plt.subplot(4,5,i+1)
        plt.imshow(image_batch[i].numpy().astype("uint8"))
        plt.title(class_names[label_batch[i]])
        plt.axis("off")


# In[14]:


train_ds=train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

valid_ds=valid_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)


# In[15]:


resize_rescale=tf.keras.Sequential([
    layers.experimental.preprocessing.Resizing(IMAGE_SIZE,IMAGE_SIZE),
    layers.experimental.preprocessing.Rescaling(1.0/255)
])


# In[17]:


data_augmentation=tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    layers.experimental.preprocessing.RandomRotation(0.2),
    
])

# In[18]:


input_shape=(BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,CHANNELS)
n_classes=10
model=Sequential([
    resize_rescale,
    data_augmentation,
    layers.Conv2D(32,(3,3),activation='relu',input_shape=input_shape),
    layers.MaxPooling2D((2,2,)),
    layers.Conv2D(64,kernel_size=(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,kernel_size=(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,kernel_size=(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,kernel_size=(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64,activation='relu'),
    layers.Dense(n_classes,activation='softmax')])

model.build(input_shape=input_shape)
    


# In[19]:


model.summary()


# In[20]:


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])


# In[21]:


callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    restore_best_weights=True,
    patience=5,
    min_delta=0.02
)


# In[22]:


history=model.fit(
    train_ds,
    epochs=30,
    batch_size=BATCH_SIZE,
    verbose=1,
    validation_data=valid_ds
)


# In[23]:


scores=model.evaluate(valid_ds)


# In[24]:


scores


# In[26]:


model.save("my_model.keras")


# In[ ]:
