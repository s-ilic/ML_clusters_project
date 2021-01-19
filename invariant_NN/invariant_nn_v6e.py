# e : Learning to detect clusters with janossy Dense + Dense

import os, sys, gc
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
clu_full_data['X'] = np.cos(decs) * np.cos(ras)
clu_full_data['Y'] = np.cos(decs) * np.sin(ras)
clu_full_data['Z'] = np.sin(decs)
# mem_full_data = hdul2[1].data
mem_full_data = {}
for n in hdul2[1].data.dtype.names:
    mem_full_data[n] = hdul2[1].data[n]
ras = mem_full_data['RA'] / 180. * np.pi
decs = mem_full_data['DEC'] / 180. * np.pi
mem_full_data['X'] = np.cos(decs) * np.cos(ras)
mem_full_data['Y'] = np.cos(decs) * np.sin(ras)
mem_full_data['Z'] = np.sin(decs)
print("Variables in cluster catalog:")
print(clu_full_data.keys())
print("Variables in member catalog:")
print(mem_full_data.keys())
n_memgal_tot = len(mem_full_data['RA'])
# sys.exit()

# IDs of clusters and corresponding number of members
clu_ids, clu_num = np.unique(mem_full_data['ID'], return_counts=True)
n_clus_tot = len(clu_ids)
n_mem_max = 100
# batch_size = 1024
# n_batch = 5
batch_size = 1
n_batch = n_clus_tot
# n_mem_max = clu_num.max()
ix_keep_clu = [i for i in range(n_clus_tot) if clu_num[i] <= n_mem_max]
# clu_counts, clu_counts_num = np.unique(clu_num, return_counts=True)
# n_mem_max = clu_counts[clu_counts_num.argmax()]
# ix_keep_clu = [i for i in range(n_clus_tot) if clu_num[i] == n_mem_max]
np.random.seed(5678)
# np.random.seed(9012) # b3bis
np.random.shuffle(ix_keep_clu)
np.random.seed()
ix_keep_clu = ix_keep_clu[:(batch_size*n_batch)]
n_clus = len(ix_keep_clu)

# Which feature to use
feat = ['R', 'P', 'P_FREE', 'THETA_I', 'THETA_R', 'IMAG', 'IMAG_ERR', 'MODEL_MAG_U', 'MODEL_MAGERR_U', 'MODEL_MAG_G', 'MODEL_MAGERR_G', 'MODEL_MAG_R', 'MODEL_MAGERR_R', 'MODEL_MAG_I', 'MODEL_MAGERR_I', 'MODEL_MAG_Z', 'MODEL_MAGERR_Z', 'X', 'Y', 'Z']
n_feat = len(feat)

# Which labels to predicts
# labs = ['IS_CLUSTER']
n_labs = 1

# Build feature array
def get_X(p):
    i, false_clus = p
    n_mem = clu_num[i]
    X = np.zeros(n_mem * n_feat)
    if not false_clus:
        g = mem_full_data['ID'] == clu_ids[i]
        for ix, f in enumerate(feat):
            X[ix::n_feat] = mem_full_data[f][g]
    else:
        rand_ix = np.random.rand(n_memgal_tot).argsort()[:n_mem]
        for ix, f in enumerate(feat):
            X[ix::n_feat] = mem_full_data[f][rand_ix]
    return X
args_pool = []
for ix in ix_keep_clu:
    args_pool.append([ix, False])
    args_pool.append([ix, True])
n_clus *= 2

if __name__ == '__main__':
    pool = Pool(32)
    allX = list(
        tqdm(
            pool.imap(get_X, args_pool),
            total=n_clus,
            smoothing=0.,
        )
    )
    pool.close()
    pool.join()

## Fill label array
allY = np.zeros((n_clus, n_labs))
for ix, ap in enumerate(args_pool):
    if ap[1]:
        allY[ix, 0] = 0.
    else:
        allY[ix, 0] = 1.

# Shuffle and split training and validation datasets
frac = 4./5. # fraction of data dedicated to training
max_ix = int(n_clus * frac)
X_train = allX[:max_ix]
X_valid = allX[max_ix:]
Y_train = allY[:max_ix]
Y_valid = allY[max_ix:]
# ix_train = ix_keep_clu[:max_ix]
# ix_valid = ix_keep_clu[max_ix:]
ix_train = args_pool[:max_ix]
ix_valid = args_pool[max_ix:]
n_train = len(X_train)
n_valid = len(X_valid)


# Define NN model
n_rand_perm = 100 # b/b3
# n_rand_perm = 1000 # b2
# num_neurons_in_f = 100 # b/b2
# num_neurons_in_f = 30 # b3
num_neurons_in_f = 60 # b4
num_layers_in_rho = 1
num_neurons_in_rho = 100
# num_neurons_in_rho = 200 # b4

def generator(inputs, labels, n, batch_size, ixs):
    i = 0
    while True:
        inputs_batch = np.zeros((batch_size, n_rand_perm, n_mem_max * n_feat))
        labels_batch = np.zeros((batch_size, n_labs))
        for j in range(batch_size):
            n_mem = clu_num[ixs[i%n][0]]
            ix = np.random.rand(n_mem, n_rand_perm).argsort(axis=0)
            inputs_batch[j, :, :] = np.hstack((
                inputs[i%n].reshape(n_mem, n_feat).T[:,ix].T.reshape(n_rand_perm, -1),
                np.zeros((n_rand_perm, (n_mem_max - n_mem) * n_feat)),
            ))
            labels_batch[j, :] = labels[i%n]
            i += 1
        yield inputs_batch, labels_batch

# with mirrored_strategy.scope():
inputs = keras.Input(shape=(n_rand_perm, n_mem_max * n_feat))
lay1 = Dense(
    num_neurons_in_f,
    activation="relu", # b2/b3/b4
    # activation="tanh",   # b
    # activation="linear",
)
out_lay1 = lay1(inputs)
output = tf.math.reduce_mean(
    out_lay1,
    axis=1,
)
rho_lays = []
for i in range(num_layers_in_rho):
    rho_lays.append(
        Dense(
            num_neurons_in_rho,
            activation="relu", # b2/b3/b4
            # activation="tanh",  # b
            # activation="linear",
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
    # loss='mean_absolute_percentage_error',
    loss='mean_squared_error',
    optimizer=Adam(
        learning_rate=0.0001,
    ),
    metrics=['accuracy'],
)
model.summary()

model.load_weights('saved_models/inv_nn_v6e/inv_nn_v6e')
# sys.exit()

log_filename = "saved_models/inv_nn_v6e/inv_nn_v6e.log"
log_cb = tf.keras.callbacks.CSVLogger(
    log_filename,
    separator=' ',
    append=True,
)

chk_filename = "saved_models/inv_nn_v6e/inv_nn_v6e"
chk_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath=chk_filename,
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    save_weights_only=True,
)


# ANN hyperparamters
nb_epoch=10000

history = model.fit(
    generator(X_train, Y_train, n_train, batch_size, ix_train),
    validation_data=generator(X_valid, Y_valid, n_valid, batch_size, ix_valid),
    steps_per_epoch=n_train // batch_size,
    validation_steps=n_valid // batch_size,
    epochs=nb_epoch,
    callbacks=[log_cb, chk_cb],
)

sys.exit()

'''
plt.clf()
plt.plot(history.history['loss'], label='Train loss')
plt.plot(history.history['val_loss'], label='Valid loss')
plt.xlabel("Epoch")
plt.ylabel("Losses")
plt.yscale('log')
plt.legend()
plt.savefig('loss_piSGD_janop_clus_zlambda.pdf')
plt.clf()
'''

##########################################

inputs_test = keras.Input(shape=(n_mem_max * n_feat,))
lay1_test = Dense(
    num_neurons_in_f,
    activation="relu",
    # activation="tanh",
    # activation="linear",
)
output_test = lay1_test(inputs_test)
rho_lays_test = []
for i in range(num_layers_in_rho):
    rho_lays_test.append(
        Dense(
            num_neurons_in_rho,
            activation="relu",
            # activation="tanh",
            # activation="linear",
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
        )
    )
    output_test = rho_lays_test[-1](output_test)
lay3_test = Dense(
    n_labs,
    activation="linear",
)
output_test = lay3_test(output_test)
model_test = keras.Model(inputs_test, output_test)
model_test.compile(
    # loss='mean_absolute_percentage_error',
    loss='mean_squared_error',
    optimizer=Adam(
        learning_rate=0.0001,
    ),
    metrics=['accuracy'],
)
model_test.summary()
model.save_weights("tmp_weights.h5")
model_test.load_weights("tmp_weights.h5")

'''
target_tol = 1e-6
min_rand = 100
eval_out = np.zeros((n_valid, n_labs))
n_out = np.zeros(n_valid)
totos = []
for i in tqdm(range(10)):
    toto = []
    tata = []
    current_tol = np.inf
    n_mem = clu_num[ix_valid[i]]
    while (current_tol > target_tol) | (n_out[i] < min_rand):
        print(n_out[i], current_tol)
        ix = np.random.rand(n_mem).argsort()
        tmp_input = np.hstack((
            X_valid[i].reshape(n_mem, n_feat).T[:,ix].T.flatten(),
            np.zeros((n_mem_max - n_mem) * n_feat),
        ))
        old_eval_out = eval_out[i, :].copy()
        # eval_out[i, :] += model_test.predict(tmp_input[None, :])[0, :]
        res = model_test.predict(tmp_input[None, :])[0, :]
        tata.append(res)
        eval_out[i, :] += res
        n_out[i] += 1
        print(old_eval_out / (n_out[i]-1.))
        print(eval_out[i, :] / n_out[i])
        toto.append(eval_out[i, :] / n_out[i])
        if n_out[i] != 1.:
            current_tol = np.max(
                np.abs(
                    eval_out[i, :] / old_eval_out * (n_out[i]-1.) / n_out[i] - 1.
                )
            )
final_pred = eval_out / n_out[: ,None]
'''

eval_out = np.zeros((n_valid, n_labs))
n_rand_perm_test = 1000
for i in tqdm(range(n_valid)):
    n_mem = clu_num[ix_valid[i][0]]
    ix = np.random.rand(n_mem, n_rand_perm_test).argsort(axis=0)
    tmp_input = np.hstack((
        X_valid[i].reshape(n_mem, n_feat).T[:,ix].T.reshape(n_rand_perm_test, -1),
        np.zeros((n_rand_perm_test, (n_mem_max - n_mem) * n_feat)),
    ))
    out = model_test.predict(tmp_input)
    eval_out[i, :] = np.mean(out, axis=0)

eval_out_train = np.zeros((n_train, n_labs))
n_rand_perm_test = 1000
for i in tqdm(range(n_train)):
    n_mem = clu_num[ix_train[i][0]]
    ix = np.random.rand(n_mem, n_rand_perm_test).argsort(axis=0)
    tmp_input = np.hstack((
        X_train[i].reshape(n_mem, n_feat).T[:,ix].T.reshape(n_rand_perm_test, -1),
        np.zeros((n_rand_perm_test, (n_mem_max - n_mem) * n_feat)),
    ))
    out = model_test.predict(tmp_input)
    eval_out_train[i, :] = np.mean(out, axis=0)

########################

minangs = []
maxangs = []
for i in tqdm(range(n_clus_tot)):
    g = mem_full_data['ID'] == clu_ids[i]
    ng = mem_full_data['ID'] != clu_ids[i]
    ####
    x = mem_full_data['X'][g]
    y = mem_full_data['Y'][g]
    z = mem_full_data['Z'][g]
    v = np.vstack(
        (x, y, z)
    )
    dotprod = (v.T * v.T[:, None, :]).sum(axis=2)
    ang = np.arccos(dotprod)
    maxang = ang[np.isfinite(ang)].max()
    maxangs.append(maxang)
    ####
    mx = np.mean(x)
    my = np.mean(y)
    mz = np.mean(z)
    norm = np.sqrt(mx**2. + my**2. + mz**2.)
    mx /= norm
    my /= norm
    mz /= norm
    v = np.array([mx, my, mz])
    x2 = mem_full_data['X'][ng]
    y2 = mem_full_data['Y'][ng]
    z2 = mem_full_data['Z'][ng]
    v2 = np.vstack(
        (x2, y2, z2)
    )
    dotprod = (v * v2.T).sum(axis=1)
    ang = np.arccos(dotprod)
    minang = ang[np.isfinite(ang)].max()
    minangs.append(minang)

def get_angs(i):
    g = mem_full_data['ID'] == clu_ids[i]
    ng = mem_full_data['ID'] != clu_ids[i]
    ####
    x = mem_full_data['X'][g]
    y = mem_full_data['Y'][g]
    z = mem_full_data['Z'][g]
    v = np.vstack(
        (x, y, z)
    )
    dotprod = (v.T * v.T[:, None, :]).sum(axis=2)
    ang = np.arccos(dotprod)
    maxang = ang[np.isfinite(ang)].max()
    ####
    mx = np.mean(x)
    my = np.mean(y)
    mz = np.mean(z)
    norm = np.sqrt(mx**2. + my**2. + mz**2.)
    mx /= norm
    my /= norm
    mz /= norm
    v = np.array([mx, my, mz])
    x2 = mem_full_data['X'][ng]
    y2 = mem_full_data['Y'][ng]
    z2 = mem_full_data['Z'][ng]
    v2 = np.vstack(
        (x2, y2, z2)
    )
    dotprod = (v * v2.T).sum(axis=1)
    ang = np.arccos(dotprod)
    minang = ang[np.isfinite(ang)].min()
    return maxang, minang, np.sum(g), np.sum(ang[np.isfinite(ang)] <= (maxang / 2.))
if __name__ == '__main__':
    pool = Pool(32)
    allX = list(
        tqdm(
            pool.imap(get_angs, range(n_clus_tot)),
            total=n_clus_tot,
            smoothing=0.,
        )
    )
    pool.close()
    pool.join()
