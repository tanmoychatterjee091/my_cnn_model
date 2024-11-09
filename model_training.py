#!/usr/bin/env python
# coding: utf-8

# # CNN Model Training for Diabetic Retinopathy Detection

# In[9]:


get_ipython().system('pip install tensorflow')


# In[10]:


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint


# In[11]:


# Define model with Mirrored Strategy
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(5, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Save the model
tf.saved_model.save(
    model, 
    'C:/Users/tanmo/Downloads/My_Projects_in_Github/cnn_deployment_docker_latest/my_cnn_model/models/my_cnn_model/1'
    )



# In[ ]:




