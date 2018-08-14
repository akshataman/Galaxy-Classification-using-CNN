
# coding: utf-8

# # GALAXY CLASSIFIER
# 
# This IPython Notebook contains the Python code for implementing a Convolution Neural Network(CNN) Architecture for Classification of Galaxy Images into it's 3 corresponding classes- Spiral type, Ireegular type and Elliptical type. The model was trained on Nvidia 960MX GPU followed by an intensive training on the NVIDIA DGX 1 Octa Tesla V100 Supercomputer servers using technologies like Putty and WinSCP. On training for 40 epochs, it was observed the training accuracy was at 95.00% with training loss at 15.37% while Validation accuracy was at 94.75% and Validation loss at 15.31%. The Training set containing 3 classes were a total of 3232 images while the Validation set containing the same number of classes contained 1190 images. 

# In[1]:

import keras
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import h5py
import warnings
from keras import backend as K
import os
import timeit
warnings.filterwarnings('ignore')


# In[2]:

num_classes=3
input_shape=(256,256,3)


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


# In[27]:

start = timeit.default_timer()
model.compile(loss=keras.losses.categorical_crossentropy, optimizer = keras.optimizers.Adam(), metrics=['accuracy'])
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'training_set',
        target_size=(256, 256),
        batch_size=64,
        class_mode='categorical' )

test_set = test_datagen.flow_from_directory(
        'val_set',
        target_size=(256, 256),
        batch_size=64,
        class_mode='categorical')


classifier = model.fit_generator(
        training_set,
        steps_per_epoch=10,
        epochs=200,
        validation_data=test_set,
        validation_steps=100)

end = timeit.default_timer()
print("Time Taken to run the model:",end - start, "seconds") 


# In[29]:

model.save_weights('model.h5')


# In[ ]:



