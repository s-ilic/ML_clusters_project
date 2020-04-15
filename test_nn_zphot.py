# -*- coding: utf-8 -*-

import numpy as np
from astropy.io import fits

import os, sys

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam, SGD
from keras.initializers import Constant

import matplotlib.pyplot as plt

'''
from google.colab import drive
drive.mount('/content/drive')
ls /content/drive/My\ Drive/redmapper*
'''
# Path to fits files
#pathData="/content/drive/My Drive/"
pathData="/home/silic/Downloads/"

# Read data in fits files
hdul2 = fits.open(pathData+'redmapper_dr8_public_v6.3_members.fits')
mem_full_data = hdul2[1].data
print("Variables in member catalog:")
print(list(mem_full_data.dtype.names))

g = mem_full_data['Z_SPEC'] != -1.
mem_full_data = mem_full_data[g]

# Features to use from member catalog
feat_from_mem = ['THETA_I', 'THETA_R', 'IMAG', 'IMAG_ERR', 'MODEL_MAG_U', 'MODEL_MAGERR_U', 'MODEL_MAG_G', 'MODEL_MAGERR_G', 'MODEL_MAG_R', 'MODEL_MAGERR_R', 'MODEL_MAG_I', 'MODEL_MAGERR_I', 'MODEL_MAG_Z', 'MODEL_MAGERR_Z']

# Labels to recover from member catalog
labs_from_mem = ['Z_SPEC']

# Build initial feature & label vectors
n_samp = mem_full_data.shape[0]
n_feat = len(feat_from_mem)
n_labs = len(labs_from_mem)
X = np.zeros((n_samp, n_feat))
Y = np.zeros((n_samp, n_labs))
for ix, feat in enumerate(feat_from_mem):
    X[:, ix] = mem_full_data[feat]
for ix, lab in enumerate(labs_from_mem):
    Y[:, ix] = mem_full_data[lab]

# Split training and testing datasets
frac = 4./5. # fraction of data dedicated to training 
max_ix = int(n_samp * frac)
print("%s samples for training out of %s" % (max_ix, X.shape[0]))
shuffle = False
if shuffle:
    np.random.seed(42)
    index = np.arange(n_samp)
    np.random.shuffle(index)
    X = X[index, :]
    Y = Y[index, :]
X_train = X[:max_ix,:]   
X_test = X[max_ix:,:]
Y_train = Y[:max_ix, :]
Y_test = Y[max_ix:, :]

sys.exit()

# ANN hyperparamters
nb_epoch=20000
batch_size= X_train.shape[0]

# NN with XX hidden layer of XX neurons
ann = Sequential()
def my_w1(shape, dtype=None):
    return np.load('z_l1.npz')['w']
def my_b1(shape, dtype=None):
    return np.load('z_l1.npz')['b']
ann.add(Dense(n_feat*2, input_dim=n_feat, kernel_initializer=my_w1, bias_initializer=my_b1))
ann.add(Activation('relu'))
def my_w2(shape, dtype=None):
    return np.load('z_l2.npz')['w']
def my_b2(shape, dtype=None):
    return np.load('z_l2.npz')['b']
ann.add(Dense(n_feat*2, kernel_initializer=my_w2, bias_initializer=my_b2))
ann.add(Activation('relu'))
def my_w3(shape, dtype=None):
    return np.load('z_l3.npz')['w']
def my_b3(shape, dtype=None):
    return np.load('z_l3.npz')['b']
ann.add(Dense(1, kernel_initializer=my_w3, bias_initializer=my_b3))
ann.add(Activation('linear'))
# ann.compile(
#     loss='binary_crossentropy',
#     # optimizer=Adam(lr=0.00001),
#     optimizer=Adam(),
#     metrics=['accuracy']
# )
ann.compile(
    # loss='mean_squared_error',
    loss='mean_absolute_percentage_error',
    # optimizer=Adam(),
    optimizer=Adam(lr=0.00001),
    metrics=['accuracy']
)
ann.fit(X_train,Y_train,epochs=nb_epoch,batch_size=batch_size,validation_data=(X_test,Y_test))#,verbose=0)

score = ann.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
Y_test_from_ann = ann.predict(X_test)
plt.plot(Y_test[:, 0], Y_test_from_ann[:, 0], '+')
plt.plot([Y_test[:, 0].min(), Y_test[:, 0].max()], [Y_test[:, 0].min(), Y_test[:, 0].max()])
plt.show()
