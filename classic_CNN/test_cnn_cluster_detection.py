# -*- coding: utf-8 -*-

import numpy as np
from astropy.io import fits

from tqdm import tqdm

from healpy.projector import GnomonicProj as GP

import os, sys

from sklearn import preprocessing
import pickle
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import  BatchNormalization
from keras.layers.convolutional import Conv2D, Convolution2D, MaxPooling2D
from keras.optimizers import rmsprop
import pdb
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard

from sklearn.metrics import roc_curve, precision_recall_curve, accuracy_score,auc

import matplotlib.pyplot as plt

from PIL import Image

#################################################################################

# Path to imgs files
path_data = "/home/silic/Downloads/imgs_clusters"
path_rand = "/home/silic/Downloads/imgs_clusters/randoms"

# Read data in imgs files
n_data = 5
n_rand = 5
n_tot = n_data + n_rand
pxsize = 2048
X_imgs = np.zeros((n_tot, pxsize, pxsize, 3))
Y_imgs = np.zeros((n_tot))
list_data = os.listdir(path_data)
list_rand = [n for n in os.listdir(path_rand) if '.jpeg' in n]
np.random.shuffle(list_data)
np.random.shuffle(list_rand)
ix = 0
for i in range(n_data):
    X_imgs[ix, :, :, :] = np.asarray(Image.open(path_data + '/' + list_data[i]))
    Y_imgs[ix] = 1
    ix += 1
for i in range(n_rand):
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

sys.exit()

batch_size = max_ix
nb_epoch = 20
data_augmentation = True

history = model.fit(
    X_imgs[:max_ix, :, :, :],
    Y_imgs[:max_ix],
    batch_size=batch_size,
    epochs=nb_epoch,
    validation_data=(X_imgs[max_ix:, :, :, :], Y_imgs[max_ix:]),
)







#########################################################
#########################################################
#########################################################

import os
import numpy as np
import healpy as hp
from tqdm import tqdm
import astropy.units as u
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
from astropy.io import fits

pathData="/home/silic/Downloads/"
hdul1 = fits.open(pathData+'redmapper_dr8_public_v6.3_catalog.fits')
clu_full_data = hdul1[1].data
n_clu = len(clu_full_data)

ns = 256
m0 = np.zeros(hp.nside2npix(ns))
cat = SkyCoord(clu_full_data['RA'], clu_full_data['DEC'], frame='icrs', unit='deg')
cat2 = cat.transform_to('galactic')
ix = hp.ang2pix(ns, np.array(cat2.l), np.array(cat2.b), lonlat=True)
for i in tqdm(range(n_clu)):
    m0[ix[i]] += 1
hp.mollview(m0)

m = np.zeros(hp.nside2npix(ns))
ra = []
dec = []
for i in tqdm(range(20000)):
    path = "/home/silic/Downloads/imgs_clusters/randoms/%s.txt" % i
    if os.path.isfile(path):
        tmp = np.loadtxt(path)
        ra.append(tmp[0])
        dec.append(tmp[1])
cat = SkyCoord(ra, dec, frame='icrs', unit='deg')
cat2 = cat.transform_to('galactic')
ix = hp.ang2pix(ns, np.array(cat2.l), np.array(cat2.b), lonlat=True)
for i in tqdm(range(len(ra))):
    m[ix[i]] += 1
hp.mollview(m)

plt.show()
