
# coding: utf-8

# In[1]:

import keras
import h5py
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import os
from keras.models import Sequential
import warnings
warnings.filterwarnings('ignore')


# In[2]:

num_classes=3
input_shape=(264,264,3)


# In[3]:

model = Sequential()
model.add(Conv2D(16, kernel_size=(3, 3), activation='tanh', input_shape=input_shape))
model.add(Conv2D(32, (3, 3), activation='tanh'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='tanh'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='tanh'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='tanh'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(32, activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.summary()


# In[6]:

model.load_weights("C:/Users/Diganta/Desktop/Courses and Projects/Projects/Bennet/model.h5")


# In[38]:

test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(
        'Category',
        target_size=(264, 264),
        batch_size=64,
        class_mode='categorical')


# In[39]:

X,y=test_set.next()


# In[40]:

model.predict_classes(X, batch_size=32, verbose=1)


# In[ ]:



