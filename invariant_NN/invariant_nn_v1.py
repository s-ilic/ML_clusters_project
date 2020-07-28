import os, sys
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from astropy.io import fits
from multiprocessing import Pool

from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D, Convolution2D, MaxPooling2D

mirrored_strategy = tf.distribute.MirroredStrategy()

# Path to fits files
pathData="/home/users/ilic/ML/SDSS_fits_data/"

# Read data in fits files
hdul1 = fits.open(pathData+'redmapper_dr8_public_v6.3_catalog.fits')
hdul2 = fits.open(pathData+'redmapper_dr8_public_v6.3_members.fits')
clu_full_data = hdul1[1].data
mem_full_data = hdul2[1].data
print("Variables in cluster catalog:")
print(list(clu_full_data.dtype.names))
print("Variables in member catalog:")
print(list(mem_full_data.dtype.names))

# IDs of clusters and corresponding number of members
clu_ids, clu_num = np.unique(mem_full_data['ID'], return_counts=True)
n_clus = len(clu_num)
n_mem_max = clu_num.max()
np.random.shuffle(clu_ids)

# Build initial feature & label vectors
n_feat_per_gal = 3
n_labels_clus = 3
'''
allX = np.zeros((n_clus, n_feat_per_gal * n_mem_max + 1))
allY = np.zeros((n_clus, n_labels_clus))
for i in tqdm(range(n_clus)):
    g = mem_full_data['ID'] == clu_ids[i]
    n_mem = g.sum()
    allX[i, 0] = n_mem
    #####
    ras = mem_full_data['RA'][g] / 180. * np.pi
    decs = mem_full_data['DEC'][g] / 180. * np.pi
    x2s = (np.cos(decs) * np.cos(ras))**2.
    y2s = (np.cos(decs) * np.sin(ras))**2.
    z2s = (np.sin(decs))**2.
    tmp = np.vstack((x2s, y2s, z2s)).T.flatten()
    allX[i, 1:(n_mem * n_feat_per_gal)+1] = tmp
    #####
    g = clu_full_data['ID'] == clu_ids[i]
    rac = clu_full_data['RA'][g] / 180. * np.pi
    decc = clu_full_data['DEC'][g] / 180. * np.pi
    x2c = (np.cos(decc) * np.cos(rac))**2.
    y2c = (np.cos(decc) * np.sin(rac))**2.
    z2c = (np.sin(decc))**2.
    allY[i, :] = np.array([x2c[0], y2c[0], z2c[0]])

# Split training and validation datasets
frac = 4./5. # fraction of data dedicated to training
max_ix = int(n_clus * frac)
X_train = allX[:max_ix,:]
X_valid = allX[max_ix:,:]
Y_train = allY[:max_ix, :]
Y_valid = allY[max_ix:, :]
'''
tmp = np.load("/home/users/ilic/ML/data.npz")
X_train = tmp['X_train']
X_valid = tmp['X_valid']
Y_train = tmp['Y_train']
Y_valid = tmp['Y_valid']

# Define NN model
with mirrored_strategy.scope():

    model=Sequential()

    model.add(Dense(10, input_dim=X_train.shape[1]))
    model.add(Activation('relu'))
    # model.add(Dense(10))
    # model.add(Activation('relu'))
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='mean_absolute_error',optimizer='adam',metrics=['accuracy'])
    model.summary()

# ANN hyperparamters
nb_epoch=300
batch_size= X_train.shape[0]

history = model.fit(
    X_train,
    Y_train,
    epochs=nb_epoch,
    batch_size=batch_size,
    # shuffle=True,
    validation_data=(X_valid, Y_valid),
    # validation_batch_size=batch_size,
)
