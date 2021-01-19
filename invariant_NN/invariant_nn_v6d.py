# d : Learning Z_SPEC with janossy GRU + Dense
# d2 : Learning Z_SPEC with janossy LSTM + Dense
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
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, LSTM, GRU


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
clu_full_data['Z_SPEC'] = np.zeros(len(decs))
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

# IDs of clusters and corresponding number of members
clu_ids, clu_num = np.unique(mem_full_data['ID'], return_counts=True)
n_clus_tot = len(clu_ids)
n_mem_max = 100
# batch_size = 1024
# n_batch = 5
batch_size = 1
# n_batch = n_clus_tot
# n_mem_max = clu_num.max()
ix_keep_clu = []
for i in tqdm(range(n_clus_tot)):
    check1 = clu_num[i] <= n_mem_max
    g1 = mem_full_data['ID'] == clu_ids[i]
    g2 = clu_full_data['ID'] == clu_ids[i]
    gg = mem_full_data['Z_SPEC'][g1] != -1.
    check2 = gg.sum() > 1
    # check3 = np.std(mem_full_data['Z_SPEC'][g1][gg]) <= 5e-3 if check2 else False
    # if check1 & check2 & check3:
    if check1 & check2:
        ix_keep_clu.append(i)
        clu_full_data['Z_SPEC'][i] = np.mean(mem_full_data['Z_SPEC'][g1][gg])
n_batch = len(ix_keep_clu)
# clu_counts, clu_counts_num = np.unique(clu_num, return_counts=True)
# n_mem_max = clu_counts[clu_counts_num.argmax()]
# ix_keep_clu = [i for i in range(n_clus_tot) if clu_num[i] == n_mem_max]
np.random.seed(5678)
np.random.shuffle(ix_keep_clu)
np.random.seed()
ix_keep_clu = ix_keep_clu[:(batch_size*n_batch)]
n_clus = len(ix_keep_clu)

# Which feature to use
feat = ['R', 'P', 'P_FREE', 'THETA_I', 'THETA_R', 'IMAG', 'IMAG_ERR', 'MODEL_MAG_U', 'MODEL_MAGERR_U', 'MODEL_MAG_G', 'MODEL_MAGERR_G', 'MODEL_MAG_R', 'MODEL_MAGERR_R', 'MODEL_MAG_I', 'MODEL_MAGERR_I', 'MODEL_MAG_Z', 'MODEL_MAGERR_Z', 'X', 'Y', 'Z']
n_feat = len(feat)

# Which labels to predicts
# labs = ['Z_LAMBDA']
labs = ['Z_SPEC']
n_labs = len(labs)

# Build feature array
def get_X(i):
    g = mem_full_data['ID'] == clu_ids[i]
    n_mem = clu_num[i]
    X = np.zeros((n_mem, n_feat))
    for ix, f in enumerate(feat):
        X[:, ix] = mem_full_data[f][g]
    return X
if __name__ == '__main__':
    pool = Pool(32)
    allX = list(
        tqdm(
            pool.imap(get_X, ix_keep_clu),
            total=n_clus,
            smoothing=0.,
        )
    )
    pool.close()
    pool.join()

## Fill label array
allY = np.zeros((n_clus, n_labs))
for ix, l in enumerate(labs):
    allY[:, ix] = clu_full_data[l][ix_keep_clu]

# Shuffle and split training and validation datasets
frac = 4./5. # fraction of data dedicated to training
max_ix = int(n_clus * frac)
X_train = allX[:max_ix]
X_valid = allX[max_ix:]
Y_train = allY[:max_ix]
Y_valid = allY[max_ix:]
ix_train = ix_keep_clu[:max_ix]
ix_valid = ix_keep_clu[max_ix:]
n_train = len(X_train)
n_valid = len(X_valid)


# Define NN model
n_rand_perm = 128
num_neurons_in_f = 80
num_layers_in_rho = 1
num_neurons_in_rho = 100

def generator(inputs, labels, n, batch_size, ixs):
    i = 0
    while True:
        n_mem = clu_num[ixs[i%n]]
        inputs_batch = np.zeros((batch_size, n_rand_perm, n_mem, n_feat))
        labels_batch = np.zeros((batch_size, n_labs))
        for j in range(batch_size):
            ix = np.random.rand(n_rand_perm, n_mem).argsort(axis=1)
            inputs_batch[j, :, :, :] = inputs[i%n][ix, :]
            labels_batch[j, :] = labels[i%n]
            i += 1
        yield inputs_batch, labels_batch

# with mirrored_strategy.scope():
inputs = keras.Input(shape=(n_rand_perm, None, n_feat))
# lay1 = LSTM(
lay1 = GRU(
# lay1 = Dense(
    num_neurons_in_f,
    # activation="tanh",
    activation="relu",
    # activation="linear",
)
out_lay1 = lay1(inputs[0, ...])
output = tf.math.reduce_mean(
    out_lay1,
    axis=0,
    keepdims=True,
)
rho_lays = []
for i in range(num_layers_in_rho):
    rho_lays.append(
        Dense(
            num_neurons_in_rho,
            # activation="tanh",
            activation="relu",
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

model.load_weights('saved_models/inv_nn_v6d/inv_nn_v6d')
# model.load_weights('saved_models/inv_nn_v6d2/inv_nn_v6d2')
sys.exit()

log_filename = "saved_models/inv_nn_v6d2/inv_nn_v6d2.log"
log_cb = tf.keras.callbacks.CSVLogger(
    log_filename,
    separator=' ',
    append=True,
)

chk_filename = "saved_models/inv_nn_v6d2/inv_nn_v6d2"
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

inputs_test = keras.Input(shape=(None, n_feat))
# lay1_test = Dense(
lay1_test = GRU(
# lay1_test = LSTM(
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
n_rand_perm_test = 10000
for i in tqdm(range(n_valid)):
    n_mem = clu_num[ix_valid[i]]
    ix = np.random.rand(n_mem).argsort()
    tmp_input = X_valid[i][None, ix, :]
    out = model_test.predict(tmp_input)
    eval_out[i, :] = np.mean(out, axis=0)


import numpy as np
import matplotlib.pyplot as plt
l0 = np.loadtxt('saved_models/inv_nn_v6/inv_nn_v6.log', skiprows=1)
l1 = np.loadtxt('saved_models/inv_nn_v6b/inv_nn_v6b.log', skiprows=1)
l2 = np.loadtxt('saved_models/inv_nn_v6c/inv_nn_v6c.log', skiprows=1)
l3 = np.loadtxt('saved_models/inv_nn_v6b3/inv_nn_v6b3.log', skiprows=1)
# l4 = np.loadtxt('saved_models/inv_nn_v6b2.log', skiprows=1)
l4 = np.loadtxt('saved_models/inv_nn_v6c3/inv_nn_v6c3.log', skiprows=1)
l5 = np.loadtxt('saved_models/inv_nn_v6b4/inv_nn_v6b4.log', skiprows=1)
fig, ax = plt.subplots(2, 3, sharey=True, figsize=(15, 10))
ax[0][0].set_yscale('log')
ax[0][0].plot(l0[1658:, 2], alpha=0.6, label='training loss')
ax[0][0].plot(l0[1658:, 4], alpha=0.6, label='validation loss')
ax[0][0].legend()
ax[0][0].set_xlabel('epochs')
ax[0][0].set_ylabel('mean square error')
ax[0][0].set_title("30 x 100 NN, full sample")
ax[0][1].set_yscale('log')
ax[0][1].plot(l1[:, 2], alpha=0.6)
ax[0][1].plot(l1[:, 4], alpha=0.6)
ax[0][1].set_xlabel('epochs')
ax[0][1].set_title("100 x 100 NN, full sample")
ax[0][2].set_yscale('log')
ax[0][2].plot(l2[:, 2], alpha=0.6)
ax[0][2].plot(l2[:, 4], alpha=0.6)
ax[0][2].set_title(r"100 x 100 NN, clusters w/ $\sigma(z_{spec})<0.005$")
ax[0][2].set_xlabel('epochs')
ax[1][0].set_yscale('log')
ax[1][0].plot(l3[:, 2], alpha=0.6, label='training loss')
ax[1][0].plot(l3[:, 4], alpha=0.6, label='validation loss')
ax[1][0].legend()
ax[1][0].set_xlabel('epochs')
ax[1][0].set_ylabel('mean square error')
ax[1][1].set_yscale('log')
ax[0][2].plot(l4[:, 2], alpha=0.6, label='training loss')
ax[0][2].plot(l4[:, 4], alpha=0.6, label='validation loss')
ax[0][2].set_xscale('log')
ax[1][1].legend()
ax[1][1].set_xlabel('epochs')
ax[1][1].set_ylabel('mean square error')
ax[1][2].set_yscale('log')
ax[1][2].plot(l5[:, 2], alpha=0.6, label='training loss')
ax[1][2].plot(l5[:, 4], alpha=0.6, label='validation loss')
ax[1][2].legend()
ax[1][2].set_xlabel('epochs')
ax[1][2].set_ylabel('mean square error')
# ax[0][0].set_ylim(2e-4, 1e-3)
fig.savefig('tmp_loss.pdf')


allY2 = np.zeros((n_clus, n_labs))
for ix, l in enumerate(labs):
    allY2[:, ix] = clu_full_data['Z_LAMBDA'][ix_keep_clu]
Y_train2 = allY2[:max_ix]
Y_valid2 = allY2[max_ix:]

g = (eval_out[:, 0] > 2.)
eval_out[g, 0] = Y_valid2[g, 0] 

mini = min(Y_valid.min(), eval_out.min(), Y_valid2.min())
maxi = max(Y_valid.max(), eval_out.max(), Y_valid2.max())

fig, axs = plt.subplots(2, 2)

axs[0][0].set_title("NN")
axs[0][0].plot(Y_valid[:, 0], eval_out[:, 0], '+')
axs[0][0].set_ylabel('z from NN')
axs[0][0].set_xlim(mini, maxi)
axs[0][0].set_ylim(mini, maxi)

axs[0][1].set_title("redmapper")
axs[0][1].plot(Y_valid[:, 0], Y_valid2[:, 0], '+')
axs[0][1].set_ylabel('z from redmapper')
axs[0][1].set_xlim(mini, maxi)
axs[0][1].set_ylim(mini, maxi)

axs[1][0].hist2d(Y_valid[:, 0], eval_out[:, 0], bins=32, range=[[mini, maxi]]*2)
axs[1][0].set_xlabel('spectro z')
axs[1][0].set_ylabel('z from NN')

axs[1][1].hist2d(Y_valid[:, 0], Y_valid2[:, 0], bins=32, range=[[mini, maxi]]*2)
axs[1][1].set_xlabel('spectro z')
axs[1][1].set_ylabel('z from redmapper')

fig.subplots_adjust(wspace=0.3)
plt.savefig('tmp0.pdf')

dzRED = (Y_valid2[:, 0] - Y_valid[:, 0]) / (1. + Y_valid[:, 0])
dzNN = (eval_out[:, 0] - Y_valid[:, 0]) / (1. + Y_valid[:, 0])
pbRED = np.mean(dzRED)
pbNN = np.mean(dzNN)
adRED = np.abs(dzRED - np.median(dzRED))
adNN = np.abs(dzNN - np.median(dzNN))
madRED = np.median(np.abs(dzRED - np.median(dzRED)))
madNN = np.median(np.abs(dzNN - np.median(dzNN)))
foRED = np.sum(np.abs(dzRED) > 0.05) / len(dzRED)
foNN = np.sum(np.abs(dzNN) > 0.05) / len(dzNN)
print("                                     RED                           NN   ")
print("-----------------------------------------------------------------------------")
print("|  prediction bias         |  %s  |  %s  |" % (pbRED, pbNN))
print("|  mean absolute deviation |   %s  |   %s  |" % (madRED, madNN))
print("|  fraction of outliers    |   %s  |   %s  |" % (foRED, foNN))
print("-----------------------------------------------------------------------------")
