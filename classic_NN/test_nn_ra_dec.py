# -*- coding: utf-8 -*-

import numpy as np
from astropy.io import fits

import os

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam, SGD
from keras.initializers import Constant

import matplotlib.pyplot as plt

# %pylab inline

np.random.seed(42)

'''
from google.colab import drive
drive.mount('/content/drive')
ls /content/drive/My\ Drive/redmapper*
'''
# Path to fits files
#pathData="/content/drive/My Drive/"
pathData="./"

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

# To be able to work with fixed NN structure, keep only clusters with
# the same (highest) number of members
tmp1, tmp2 = np.unique(clu_num, return_counts=True)
ids_to_keep = clu_ids[clu_num == tmp1[tmp2.argmax()]] # 813 clusters
n_mem = tmp1[tmp2.argmax()] # 55 members

# Multiplicative factor : new "fake" training will computed by shuffling
# because NN should be invariant by shuffling
mult_fact = 1000

# Features to use from member catalog
feat_from_mem = ['RA', 'DEC']

# Labels to recover from cluster catalog
labs_from_clu = ['RA', 'DEC']

# Build initial feature & label vectors
n_samp = len(ids_to_keep)
n_feat = n_mem * len(feat_from_mem)
n_labs = len(labs_from_clu)
iniX = np.zeros((n_samp, n_feat))
iniY = np.zeros((n_samp, n_labs))
iniYfake = np.zeros((n_samp, n_labs))
for i, ix in enumerate(ids_to_keep):
  g = mem_full_data['ID'] == ix
  for jx, feat in enumerate(feat_from_mem):
    iniX[i, jx*n_mem:(jx+1)*n_mem] = mem_full_data[feat][g]
    iniYfake[i, jx] = mem_full_data[feat][g].mean() # TESTING
  g = clu_full_data['ID'] == ix
  for jx, lab in enumerate(labs_from_clu):
    iniY[i, jx] = clu_full_data[lab][g]


# Build bigger final vectors by shuffling members for a given feature
X = np.zeros((n_samp * mult_fact, n_feat))
Y = np.zeros((n_samp * mult_fact, n_labs))
Yfake = np.zeros((n_samp * mult_fact, n_labs))
from tqdm import tqdm
for i in tqdm(range(mult_fact)):
  # Shuffle indices
  tmp_ix = np.arange(n_mem)
  np.random.shuffle(tmp_ix)
  shuf_ix = np.array([], dtype='int')
  for j in range(len(feat_from_mem)):
    shuf_ix = np.append(shuf_ix, tmp_ix + j * n_mem)
  # Fill final vectors
  X[i*n_samp:(i+1)*n_samp, :] = iniX[:, shuf_ix]
  Y[i*n_samp:(i+1)*n_samp, :] = iniY
  Yfake[i*n_samp:(i+1)*n_samp, :] = iniYfake

# Split training and testing datasets
frac = 4./5. # fraction of data dedicated to training 
max_ix = int(n_samp * mult_fact * frac)
print("%s samples for training out of %s" % (max_ix, X.shape[0]))
X_train = X[:max_ix,:]   
X_test = X[max_ix:,:]
Y_train = Y[:max_ix, :]
Y_test = Y[max_ix:, :]
Yfake_train = Yfake[:max_ix, :]
Yfake_test = Yfake[max_ix:, :]

# ANN hyperparamters
nb_epoch=300
batch_size= X_train.shape[0]

def my_init(shape, dtype=None):
    res = np.zeros(shape, dtype=dtype)
    res[:shape[0]/2, 0] = 1./55.
    res[shape[0]/2:, 1] = 1./55.
    return res

# NN with XX hidden layer of XX neurons
ann = Sequential()
ann.add(Dense(2, input_dim=n_feat, use_bias=False, kernel_initializer=my_init))
#ann.add(Activation('relu'))
#ann.add(Dense(2))
ann.add(Activation('linear'))
'''
ann.compile(
    loss='binary_crossentropy',
    optimizer=Adam(lr=0.00001),
    metrics=['accuracy']
)
'''
ann.compile(
    loss='mean_squared_error',
    optimizer=Adam(lr=0.0005),
    metrics=['accuracy']
)
#'''
ann.fit(X_train,Y_train,epochs=nb_epoch,batch_size=batch_size)#,verbose=0)

score = ann.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
Y_test_from_ann = ann.predict(X_test)
plt.subplot(1,2,1)
plt.plot(Y_test[:, 0], Y_test[:, 0]-Y_test_from_ann[:, 0], '+')
plt.plot(Y_test[:, 0], Y_test[:, 0]-Yfake_test[:, 0], '+')
plt.subplot(1,2,2)
plt.plot(Y_test[:, 1], Y_test[:, 1]-Y_test_from_ann[:, 1], '+')
plt.plot(Y_test[:, 1], Y_test[:, 1]-Yfake_test[:, 1], '+')
plt.show()



score = ann.evaluate(iniX, iniY, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
iniY_test_from_ann = ann.predict(iniX)
plt.subplot(1,2,1)
plt.plot(iniY[:, 0], iniY[:, 0]-iniY_test_from_ann[:, 0], '+')
plt.plot(iniY[:, 0], iniY[:, 0]-iniYfake[:, 0], '+')
plt.subplot(1,2,2)
plt.plot(iniY[:, 1], iniY[:, 1]-iniY_test_from_ann[:, 1], '+')
plt.plot(iniY[:, 1], iniY[:, 1]-iniYfake[:, 1], '+')
plt.show()

plt.plot(np.abs(iniY[:, 1]-iniY_test_from_ann[:, 1]), np.abs(iniY[:, 1]-iniYfake[:, 1]), '+')
plt.plot([-0.01, 0.04],[-0.01, 0.04])
plt.show()

