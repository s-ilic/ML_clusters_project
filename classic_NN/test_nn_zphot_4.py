# -*- coding: utf-8 -*-

import numpy as np
from astropy.io import fits
from tqdm import tqdm

import os, sys

import tensorflow as tf
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

z_bins_file = np.loadtxt('/home/users/ilic/ML/SDSS_DR8_pz_cat/zbins-12.dat')
z_bins = np.zeros(len(z_bins_file[:, 0]) + 1)
z_bins[:-1] = z_bins_file[:, 0]
z_bins[-1] = z_bins_file[-1, 1]
ix_bins = np.digitize(mem_full_data['Z_SPEC'], z_bins) - 1

# Labels to recover from member catalog
labs_from_mem = ['Z_SPEC']

# Build initial feature & label vectors
n_samp = mem_full_data.shape[0]
n_feat = len(feat_from_mem)
n_labs = len(labs_from_mem)
X = np.zeros((n_samp, n_feat))
Y = np.zeros((n_samp, len(z_bins_file[:, 0])))
for ix, feat in enumerate(feat_from_mem):
    X[:, ix] = mem_full_data[feat]
for i in tqdm(range(n_samp)):
    Y[i, ix_bins[i]] += 1.

# Split training and testing datasets
frac = 4./5. # fraction of data dedicated to training
max_ix = int(n_samp * frac)
print("%s samples for training out of %s" % (max_ix, X.shape[0]))
shuffle = True
if shuffle:
#     np.random.seed(42)
    index = np.argsort(np.random.rand(n_samp))
    X = X[index, :]
    Y = Y[index, :]
X_train = X[:max_ix,:]
X_test = X[max_ix:,:]
Y_train = Y[:max_ix, :]
Y_test = Y[max_ix:, :]

# ANN hyperparamters
nb_epoch=50000
batch_size= X_train.shape[0]

# NN with XX hidden layer of XX neurons
ann = Sequential()
ann.add(Dense(1000, input_dim=n_feat))
ann.add(Activation('relu'))
# ann.add(Dense(100))
# ann.add(Activation('relu'))
ann.add(Dense(len(z_bins_file[:, 0])))
# ann.add(Activation('linear'))
ann.add(Activation('softmax'))
# ann.compile(
#     loss='binary_crossentropy',
#     # optimizer=Adam(lr=0.00001),
#     optimizer=Adam(),
#     metrics=['accuracy']
# )
ann.compile(
    # loss='mean_squared_error',
    # loss='mean_absolute_percentage_error',
    loss='categorical_crossentropy',
    # optimizer=Adam(),
    optimizer=Adam(lr=0.0002),
    metrics=['accuracy']
)
# sys.exit()

ann.summary()

log_filename = "zphot_v4.log"
log_cb = tf.keras.callbacks.CSVLogger(
    log_filename,
    separator=' ',
    append=True,
)


ann.fit(
    X_train,
    Y_train,
    epochs=nb_epoch,
    batch_size=batch_size,
    validation_data=(X_test, Y_test),
    shuffle=True,
    callbacks=[log_cb],
)


sys.exit()

#########################################################################

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

#########################################################################

fname = 'pofz-12-008158.dat'


# full_id = np.zeros(0, dtype=np.int64)
# full_pz = np.zeros((0, 35))

full_id = []
full_pz = []

for fname in tqdm(os.listdir('.')):
    if fname.startswith('pofz') and fname.endswith('.dat'):
        t = np.loadtxt(
            fname,
            dtype=np.dtype(
                [
                    ('objID', np.int64),
                    ('run', np.int32),
                    ('rerun', np.str),
                    ('camcol', np.int32),
                    ('field', np.int32),
                    ('id', np.int32),
                    ('ra', np.float64),
                    ('dec', np.float64),
                    ('cmodelmag_r', np.float32),
                    ('modelmag_umg', np.float32),
                    ('modelmag_gmr', np.float32),
                    ('modelmag_rmi', np.float32),
                    ('modelmag_imz', np.float32),
                ] +
                [
                    ('pofz_%s' % i, np.float32) for i in range(35)
                ]
            )
        )
        full_id = full_id.append(np.atleast_1d(t['objID']))
        try:
            tmp = np.zeros((len(t), 35))
        except:
            tmp = np.zeros((1, 35))
        for i in range(35):
            tmp[:, i] = t['pofz_%s' % i]
        full_pz.append(tmp.copy())

