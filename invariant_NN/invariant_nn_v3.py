import os, sys
import numpy as np
from tqdm import tqdm
from astropy.io import fits
from multiprocessing import Pool

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten

import my_tf_layers

mirrored_strategy = tf.distribute.MirroredStrategy()

# Path to fits files
pathData="/home/users/ilic/ML/SDSS_fits_data/"

# Read data in fits files
hdul1 = fits.open(pathData+'redmapper_dr8_public_v6.3_catalog.fits')
hdul2 = fits.open(pathData+'redmapper_dr8_public_v6.3_members.fits')
# clu_full_data = hdul1[1].data
clu_full_data = {}
for n in hdul1[1].data.dtype.names:
    clu_full_data[n] = hdul1[1].data[n]
ras = clu_full_data['RA'] / 180. * np.pi
decs = clu_full_data['DEC'] / 180. * np.pi
clu_full_data['X'] = (np.cos(decs) * np.cos(ras))**2.
clu_full_data['Y'] = (np.cos(decs) * np.sin(ras))**2.
clu_full_data['Z'] = (np.sin(decs))**2.
# mem_full_data = hdul2[1].data
mem_full_data = {}
for n in hdul2[1].data.dtype.names:
    mem_full_data[n] = hdul2[1].data[n]
ras = mem_full_data['RA'] / 180. * np.pi
decs = mem_full_data['DEC'] / 180. * np.pi
mem_full_data['X'] = (np.cos(decs) * np.cos(ras))**2.
mem_full_data['Y'] = (np.cos(decs) * np.sin(ras))**2.
mem_full_data['Z'] = (np.sin(decs))**2.
print("Variables in cluster catalog:")
print(clu_full_data.keys())
print("Variables in member catalog:")
print(mem_full_data.keys())

# IDs of clusters and corresponding number of members
clu_ids, clu_num = np.unique(mem_full_data['ID'], return_counts=True)
n_clus = len(clu_ids)
n_mem_max = clu_num.max()
# np.random.shuffle(clu_ids)
clu_ids = clu_full_data['ID']

# Which feature to use
feat = ['X', 'Y', 'Z']

# Which labels to predicts
labs = ['X', 'Y', 'Z']

# Build feature array
n_feat = len(feat)
# allX = np.zeros((n_clus, n_mem_max * (n_feat + 1)))
# for i in tqdm(range(n_clus)):
#     ## Check number of member galaxies, fill with weights
#     g = mem_full_data['ID'] == clu_ids[i]
#     n_mem = g.sum()
#     allX[i, ::(n_feat + 1)][:n_mem] = 1.
#     ## Fill rest of array with features
#     for ix, f in enumerate(feat):
#         allX[i, (ix+1)::(n_feat + 1)][:n_mem] = mem_full_data[f][g]
def get_X(i):
    X = np.zeros(n_mem_max * (n_feat + 1))
    g = mem_full_data['ID'] == clu_ids[i]
    n_mem = g.sum()
    X[::(n_feat + 1)][:n_mem] = 1.
    for ix, f in enumerate(feat):
        X[(ix+1)::(n_feat + 1)][:n_mem] = mem_full_data[f][g]
    return X
if __name__ == '__main__':
    pool = Pool(32)
    allX = np.array(
        list(
            tqdm(
                pool.imap(get_X, range(n_clus)),
                total=n_clus,
                smoothing=0.,
            )
        )
    )

## Fill label array
n_labs = len(labs)
allY = np.zeros((n_clus, n_labs))
for ix, l in enumerate(labs):
    allY[:, ix] = clu_full_data[l]

# Shuffle and split training and validation datasets
ix_rand = np.argsort(np.random.rand(n_clus))
frac = 4./5. # fraction of data dedicated to training
max_ix = int(n_clus * frac)
X_train = allX[ix_rand, :][:max_ix,:]
X_valid = allX[ix_rand, :][max_ix:,:]
Y_train = allY[ix_rand, :][:max_ix, :]
Y_valid = allY[ix_rand, :][max_ix:, :]



# Define NN model
with mirrored_strategy.scope():

    model=Sequential()
    model.add(my_tf_layers.MSMM_Layer(nfeat=n_feat))
    model.add(Activation('linear'))
    model.add(Dense(3, activation='linear'))

    '''
    inputs = keras.Input(shape=(n_mem_max * (n_feat + 1),))
    lay1 = Dense(4 * n_feat, input_shape=(n_feat,), activation="relu")
    out_lay1 = []
    for i in range(n_mem_max):
        out_lay1.append(
            tf.concat(
                (
                    inputs[:, i*(n_feat + 1):(i*(n_feat + 1)+1)],
                    lay1(inputs[:, (i*(n_feat + 1)+1):((i+1)*(n_feat + 1))]),
                ),
                axis=-1,
            )
        )
    out_lay1 = tf.concat(
        out_lay1,
        axis=-1,
    )
    lays2 = [
        my_tf_layers.MSMM_Layer(nfeat=4 * n_feat),
        my_tf_layers.MSMM_Layer(nfeat=4 * n_feat),
        my_tf_layers.MSMM_Layer(nfeat=4 * n_feat),
        my_tf_layers.MSMM_Layer(nfeat=4 * n_feat),
        my_tf_layers.MSMM_Layer(nfeat=4 * n_feat),
        my_tf_layers.MSMM_Layer(nfeat=4 * n_feat),
        my_tf_layers.MSMM_Layer(nfeat=4 * n_feat),
        my_tf_layers.MSMM_Layer(nfeat=4 * n_feat),
        my_tf_layers.MSMM_Layer(nfeat=4 * n_feat),
        my_tf_layers.MSMM_Layer(nfeat=4 * n_feat),
    ]
    out_lays2 = [lay2(out_lay1) for lay2 in lays2]
    out_lays2 = tf.concat(
        out_lays2,
        axis=-1,
    )
    lay3 = Dense(n_labs, activation='linear')
    out_lay3 = lay3(out_lays2)
    model = keras.Model(inputs, out_lay3)
    '''

    model.compile(
        loss='mean_absolute_error',
        optimizer=Adam(
            learning_rate=0.00001,
        ),
        metrics=['accuracy'],
    )
    # model.summary()


# ANN hyperparamters
nb_epoch=1
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
