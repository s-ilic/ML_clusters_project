# -*- coding: utf-8 -*-

import numpy as np
from astropy.io import fits

import os, sys

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.initializers import Constant

import matplotlib.pyplot as plt

'''
from google.colab import drive
drive.mount('/content/drive')
ls /content/drive/My\ Drive/redmapper*
'''
# Path to fits files
#pathData="/content/drive/My Drive/"
pathData="/home/users/ilic/ML/SDSS_fits_data/"

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
shuffle = True
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

# ANN hyperparamters
nb_epoch=5000
batch_size= X_train.shape[0]

# NN with XX hidden layer of XX neurons
ann = Sequential()
ann.add(Dense(1000, input_dim=n_feat))
ann.add(Activation('relu'))
# ann.add(Dense(100))
# ann.add(Activation('relu'))
ann.add(Dense(1))
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
    optimizer=Adam(lr=0.0001),
    metrics=['accuracy']
)
# sys.exit()
ann.fit(X_train,Y_train,epochs=nb_epoch,batch_size=batch_size,validation_data=(X_test,Y_test))#,verbose=0)


score = ann.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
Y_test_from_ann = ann.predict(X_test)
plt.subplot(1,2,1)
plt.plot(Y_test[:, 0], Y_test_from_ann[:, 0], '+')
plt.plot([Y_test[:, 0].min(), Y_test[:, 0].max()], [Y_test[:, 0].min(), Y_test[:, 0].max()])

Y_train_from_ann = ann.predict(X_train)
plt.subplot(1,2,2)
plt.plot(Y_train[:, 0], Y_train_from_ann[:, 0], '+')
plt.plot([Y_train[:, 0].min(), Y_train[:, 0].max()], [Y_train[:, 0].min(), Y_train[:, 0].max()])

plt.savefig('z_phot_spec.pdf')
plt.show()
