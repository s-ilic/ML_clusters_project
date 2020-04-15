# -*- coding: utf-8 -*-

import os, sys
import numpy as np
from tqdm import tqdm
from PIL import Image

# import pdb
# import time
# import pickle
# from astropy.io import fits
# import matplotlib.pyplot as plt
# from healpy.projector import GnomonicProj as GP

# from sklearn import preprocessing
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import  BatchNormalization
from keras.layers.convolutional import Conv2D, Convolution2D, MaxPooling2D
# from keras.optimizers import rmsprop

# from keras.callbacks import EarlyStopping
# from keras.callbacks import ModelCheckpoint
# from keras.callbacks import TensorBoard

# from sklearn.metrics import roc_curve, precision_recall_curve, accuracy_score,auc



#################################################################################

# Path to imgs files
path_data = "/home/users/ilic/ML/imgs_clusters"
path_rand = "/home/users/ilic/ML/imgs_clusters/randoms"
path_train = "/home/users/ilic/ML/data/train"
path_valid = "/home/users/ilic/ML/data/validation"

# Read data in imgs files
list_data = [n for n in os.listdir(path_data) if '.jpeg' in n]
list_rand = [n for n in os.listdir(path_rand) if '.jpeg' in n]
n_data = len(list_data)
n_rand = len(list_rand)
n_tot = n_data + n_rand
np.random.shuffle(list_data)
np.random.shuffle(list_rand)
frac = 4./5. # fraction of data dedicated to training

'''
# Split training and testing datasets
os.system("rm -f %s/has_cluster/*" % path_train)
os.system("rm -f %s/no_cluster/*" % path_train)
os.system("rm -f %s/has_cluster/*" % path_valid)
os.system("rm -f %s/no_cluster/*" % path_valid)
max_ix = int(n_data * frac)
for i in tqdm(range(n_data)):
    if i < max_ix:
        cmd = "ln -s %s/%s %s/has_cluster/%s &" % (
            path_data,
            list_data[i],
            path_train,
            list_data[i]
        )
    else:
        cmd = "ln -s %s/%s %s/has_cluster/%s &" % (
            path_data,
            list_data[i],
            path_valid,
            list_data[i]
        )
    os.system(cmd)
for i in tqdm(range(n_rand)):
    if i < max_ix:
        cmd = "ln -s %s/%s %s/no_cluster/%s &" % (
            path_rand,
            list_rand[i],
            path_train,
            list_rand[i]
        )
    else:
        cmd = "ln -s %s/%s %s/no_cluster/%s &" % (
            path_rand,
            list_rand[i],
            path_valid,
            list_rand[i]
        )
    os.system(cmd)
'''

# Build image generators
batch_size = 8
target_size = (2048, 2048)
datagen = ImageDataGenerator()
train_it = datagen.flow_from_directory(
    path_train,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical',
)
val_it = datagen.flow_from_directory(
    path_valid,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical',
)
# test_it = datagen.flow_from_directory('data/test/', class_mode='binary')

# Build CNN model
dropoutpar = 0.5
nb_dense = 64

model=Sequential()

model.add(Conv2D(32, (3, 3), padding='same',input_shape=(target_size[0], target_size[1], 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(32, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dropout(dropoutpar))
# model.add(Dense(nb_dense, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

steps_per_epoch = int(np.ceil(n_tot * frac / batch_size))
validation_steps = int(np.ceil(n_tot * (1.-frac) / batch_size))
history = model.fit_generator(
    train_it,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_it,
    validation_steps=validation_steps
)
