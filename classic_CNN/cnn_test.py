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
# from keras.preprocessing.image import ImageDataGenerator
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

# Read data in imgs files
n_data = 1000
n_rand = 1000
n_tot = n_data + n_rand
pxsize = 2048
X_imgs = np.zeros((n_tot, pxsize, pxsize, 3))
Y_imgs = np.zeros((n_tot))
list_data = [n for n in os.listdir(path_data) if '.jpeg' in n]
list_rand = [n for n in os.listdir(path_rand) if '.jpeg' in n]
np.random.shuffle(list_data)
np.random.shuffle(list_rand)
ix = 0
for i in tqdm(range(n_data)):
    X_imgs[ix, :, :, :] = np.asarray(Image.open(path_data + '/' + list_data[i]))
    Y_imgs[ix] = 1
    ix += 1
for i in tqdm(range(n_rand)):
    X_imgs[ix, :, :, :] = np.asarray(Image.open(path_rand + '/' + list_rand[i]))
    Y_imgs[ix] = 0
    ix += 1

# Some shuffling and thinning
thin = 1
ix = np.random.permutation(n_tot)
X_imgs = X_imgs[ix[::thin], :, :, :]
Y_imgs = Y_imgs[ix[::thin]]

# Augment images

# Split training and testing datasets
frac = 4./5. # fraction of data dedicated to training
max_ix = int(n_tot * frac)
# X_train = X_imgs[:max_ix, :, :, :]
# Y_train = Y_imgs[:max_ix]
# X_test = X_imgs[max_ix:, :, :, :]
# Y_test = Y_imgs[max_ix:]


# Build CNN model
dropoutpar = 0.5
nb_dense = 16
model=Sequential()

model.add(Conv2D(32, (3, 3), padding='same',input_shape=(pxsize, pxsize, 3)))
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
model.add(Dense(1, activation='softmax'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

batch_size = 32
nb_epoch = 20

history = model.fit(
    X_imgs[:max_ix, :, :, :],
    Y_imgs[:max_ix],
    batch_size=batch_size,
    epochs=nb_epoch,
    validation_data=(X_imgs[max_ix:, :, :, :], Y_imgs[max_ix:]),
)

