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

# mirrored_strategy = tf.distribute.MirroredStrategy()

# Path to fits files
pathData="/home/silic/Downloads/"

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

# Fixed the NN structure to the highest number of members
ix = np.arange(len(clu_ids))
np.random.shuffle(ix)
clu_ids = clu_ids[ix][::10]
clu_num = clu_num[ix][::10]
n_mem_max = clu_num.max() # 55 members

# Multiplicative factor : new "fake" training will computed by shuffling
# because NN should be invariant by shuffling
mult_fact = 100

# Features to use from member catalog
feat_from_mem = ['RA', 'DEC']

# Labels to recover from cluster catalog
labs_from_clu = ['RA', 'DEC']

# Build initial feature & label vectors
n_samp = clu_ids.shape[0]
n_feat = n_mem_max * (len(feat_from_mem) + 1) # "+1" is "member boolean"
n_labs = len(labs_from_clu)
iniX = np.zeros((n_samp, n_feat))
iniY = np.zeros((n_samp, n_labs))
iniYfake = np.zeros((n_samp, n_labs))
for i in tqdm(range(n_samp)):
    ix = clu_ids[i]
    g = mem_full_data['ID'] == ix
    n_mem = g.sum()
    for jx, feat in enumerate(feat_from_mem):
        iniX[i, jx*n_mem_max:jx*n_mem_max+n_mem] = mem_full_data[feat][g]
        iniYfake[i, jx] = mem_full_data[feat][g].mean() # TESTING
    jx += 1
    iniX[i, jx*n_mem_max:jx*n_mem_max+n_mem] = 1.
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
  tmp_ix = np.arange(n_mem_max)
  np.random.shuffle(tmp_ix)
  shuf_ix = np.array([], dtype='int')
  for j in range(len(feat_from_mem)+1):
    shuf_ix = np.append(shuf_ix, tmp_ix + j * n_mem_max)
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

# NN with XX hidden layer of XX neurons
ann = Sequential()
# def my_w1(shape, dtype=None):
#     return np.load('z_l1b.npz')['w']
# def my_b1(shape, dtype=None):
#     return np.load('z_l1b.npz')['b']
# ann.add(Dense(10, input_dim=n_feat))#, kernel_initializer=my_w1, bias_initializer=my_b1))
# ann.add(Activation('relu'))
# def my_w2(shape, dtype=None):
#     return np.load('z_l2b.npz')['w']
# def my_b2(shape, dtype=None):
#     return np.load('z_l2b.npz')['b']
ann.add(Dense(2))#, kernel_initializer=my_w2, bias_initializer=my_b2))
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
    # optimizer=Adam(lr=0.0005),
    optimizer=Adam(),
    metrics=['accuracy']
)
#'''
ann.fit(X_train,Y_train,epochs=nb_epoch,batch_size=batch_size,validation_data=(X_test,Y_test))#,verbose=0)

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

