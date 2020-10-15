import os, sys
import numpy as np
from tqdm import tqdm
from astropy.io import fits
from multiprocessing import Pool
from itertools import combinations, permutations

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten

# mirrored_strategy = tf.distribute.MirroredStrategy()

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
clu_full_data['X'] = (np.cos(decs) * np.cos(ras))#**2.
clu_full_data['Y'] = (np.cos(decs) * np.sin(ras))#**2.
clu_full_data['Z'] = (np.sin(decs))#**2.
# mem_full_data = hdul2[1].data
mem_full_data = {}
for n in hdul2[1].data.dtype.names:
    mem_full_data[n] = hdul2[1].data[n]
ras = mem_full_data['RA'] / 180. * np.pi
decs = mem_full_data['DEC'] / 180. * np.pi
mem_full_data['X'] = (np.cos(decs) * np.cos(ras))#**2.
mem_full_data['Y'] = (np.cos(decs) * np.sin(ras))#**2.
mem_full_data['Z'] = (np.sin(decs))#**2.
print("Variables in cluster catalog:")
print(clu_full_data.keys())
print("Variables in member catalog:")
print(mem_full_data.keys())

# IDs of clusters and corresponding number of members
clu_ids, clu_num = np.unique(mem_full_data['ID'], return_counts=True)
n_clus_tot = len(clu_ids)
n_mem_max = 100
# n_mem_max = clu_num.max()
ix_keep_clu = [i for i in range(n_clus_tot) if clu_num[i] <= n_mem_max]
# clu_counts, clu_counts_num = np.unique(clu_num, return_counts=True)
# n_mem_max = clu_counts[clu_counts_num.argmax()]
# ix_keep_clu = [i for i in range(n_clus_tot) if clu_num[i] == n_mem_max]
np.random.shuffle(ix_keep_clu)
ix_keep_clu = ix_keep_clu[:6000]
n_clus = len(ix_keep_clu)

# Which feature to use
feat = ['X', 'Y', 'Z']
# feat = ['RA', 'DEC']
n_feat = len(feat)

# Which labels to predicts
labs = ['X', 'Y', 'Z']
# labs = ['RA', 'DEC']
n_labs = len(labs)

# Build permutation array
janossy_k = 2
# perm_arr = np.array(
#     list(permutations(range(n_mem_max), janossy_k))
# )
# n_perm = perm_arr.shape[0]
perm_dict = {
    n:np.array(list(permutations(range(n), janossy_k))) for n in np.unique(clu_num[ix_keep_clu])
}
n_perm_max = len(perm_dict[clu_num[ix_keep_clu].max()])

# arr size == (n_samp, n_perm, n_feat * janossy_k)
# n_perm = (n_feat * n_mem_max)! / (n_feat * n_mem_max - janossy_k)!


# Build feature array
def get_X(i):
    X = np.zeros((n_perm_max, janossy_k * n_feat + 1))
    g = mem_full_data['ID'] == clu_ids[i]
    n_mem = clu_num[i]
    # for ix, f in enumerate(feat):
    #     base = np.zeros(n_mem_max)
    #     base[:n_mem] = mem_full_data[f][g]
    #     for j in range(n_perm):
    #         X[j, ix*janossy_k:(ix+1)*janossy_k] = base[perm_arr[j]]
    n_perm = len(perm_dict[n_mem])
    X[:n_perm, -1] = 1. / n_perm
    for ix, f in enumerate(feat):
        base = mem_full_data[f][g]
        for j in range(n_perm):
            X[j, ix*janossy_k:(ix+1)*janossy_k] = base[perm_dict[n_mem][j]]

    return X
if __name__ == '__main__':
    pool = Pool(32)
    allX = np.array(
        list(
            tqdm(
                pool.imap(get_X, ix_keep_clu),
                total=n_clus,
                smoothing=0.,
            )
        )
    )

## Fill label array
allY = np.zeros((n_clus, n_labs))
for ix, l in enumerate(labs):
    allY[:, ix] = clu_full_data[l][ix_keep_clu]

# Shuffle and split training and validation datasets
frac = 4./5. # fraction of data dedicated to training
max_ix = int(n_clus * frac)
X_train = allX[:max_ix, :, :]
X_valid = allX[max_ix:, :, :]
Y_train = allY[:max_ix, :]
Y_valid = allY[max_ix:, :]



# Define NN model
num_neurons_in_f = 3
# num_neurons_in_f = 10
# num_layers_in_rho = 1
num_layers_in_rho = 0
num_neurons_in_rho = 100
# num_neurons_in_rho = 3
# with mirrored_strategy.scope():

inputs = keras.Input(shape=(X_train.shape[1], X_train.shape[2]))
# resh_inputs = tf.reshape(inputs, (-1, X_train.shape[2]))
lay1 = Dense(
    num_neurons_in_f,
    # input_shape=(X_train.shape[2],),
    input_shape=(X_train.shape[1], X_train.shape[2] - 1,),
    # activation="tanh",
    # activation="relu",
    activation="linear",
    # kernel_initializer=tf.keras.initializers.Ones(),
    kernel_initializer=tf.keras.initializers.Identity(gain=1.),
    # kernel_initializer=tf.keras.initializers.GlorotUniform(),
    # bias_initializer=tf.keras.initializers.Zeros(),
)
# out_lay1 = lay1(resh_inputs)
# resh_out_lay1 = tf.reshape(out_lay1, (-1, X_train.shape[1], num_neurons_in_f))
# out_lay1 = lay1(inputs)
# output = tf.math.reduce_mean(
    # # resh_out_lay1,
    # out_lay1,
    # axis=1,
# )
out_lay1 = lay1(inputs[:, :, :-1])
output = tf.math.reduce_sum(
    out_lay1 * inputs[:, :, -1][:, :, None],
    axis=1,
)
# lay2 = Dense(
#     num_neurons_in_f,
#     activation="tanh",
#     # activation="relu",
# )
# output = lay2(output)
rho_lays = []
for i in range(num_layers_in_rho):
    rho_lays.append(
        Dense(
            num_neurons_in_rho,
            # activation="relu",
            # activation="tanh",
            activation="linear",
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
        )
    )
    output = rho_lays[-1](output)
lay3 = Dense(
    n_labs,
    activation="linear",
)
output = lay3(output)

model = keras.Model(inputs, output)
model.compile(
    loss='mean_absolute_percentage_error',
    # loss='mean_squared_error',
    optimizer=Adam(
        learning_rate=0.001,
    ),
    metrics=['accuracy'],
)

model.summary()


# ANN hyperparamters
nb_epoch=1000
batch_size= X_train.shape[0]

history = model.fit(
    X_train,
    Y_train,
    epochs=nb_epoch,
    batch_size=batch_size,
    # shuffle=True,
    # validation_data=(X_valid, Y_valid),
    # validation_batch_size=batch_size,
    # use_multiprocessing=True,
    # workers=8,
)

'''
plt.clf()
plt.plot(history.history['loss'], label='Train loss')
plt.plot(history.history['val_loss'], label='Valid loss')
plt.xlabel("Epoch")
plt.ylabel("Losses")
plt.yscale('log')
plt.legend()
plt.savefig('loss_janop_clus_radecXYZ.pdf')
plt.clf()
'''