
# b5: learning LAMBDA with janossy Dense + Dense

import os, sys, gc
import numpy as np
from tqdm import tqdm
from astropy.io import fits
from multiprocessing import Pool
from itertools import combinations, permutations
import healpy as hp

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras import layers, Model, regularizers
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
clu_full_data['M500'] = np.zeros(len(hdul1[1].data))
clu_full_data['M500_ERR_LOW'] = np.zeros(len(hdul1[1].data))
clu_full_data['M500_ERR_UPP'] = np.zeros(len(hdul1[1].data))
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
# sys.exit()

hdul3 = fits.open("/home/users/ilic/ML/other_cats/comprass.fits")
ix_red_comp = []
for i in range(len(hdul3[1].data)):
    if hdul3[1].data['REDMAPPER'][i] != '':
        tmp = np.where(hdul3[1].data['REDMAPPER'][i] == hdul1[1].data['NAME'])[0]
        if len(tmp) != 0:
            clu_full_data['M500'][tmp[0]] = hdul3[1].data['M500'][i]
            clu_full_data['M500_ERR_LOW'][tmp[0]] = hdul3[1].data['M500_ERR_LOW'][i]
            clu_full_data['M500_ERR_UPP'][tmp[0]] = hdul3[1].data['M500_ERR_UPP'][i]
            ix_red_comp.append(tmp[0])


# IDs of clusters and corresponding number of members
clu_ids, clu_num = np.unique(mem_full_data['ID'], return_counts=True)
n_clus_tot = len(clu_ids)
n_mem_max = 30
ix_keep_clu = []
for i in range(n_clus_tot):
    if i in ix_red_comp:
        ix_keep_clu.append(i)
np.random.seed(457896)
np.random.shuffle(ix_keep_clu)
np.random.seed()
n_clus = len(ix_keep_clu)

# Which feature to use
feat = ['R', 'P', 'P_FREE', 'THETA_I', 'THETA_R', 'IMAG', 'IMAG_ERR', 'MODEL_MAG_U', 'MODEL_MAGERR_U', 'MODEL_MAG_G', 'MODEL_MAGERR_G', 'MODEL_MAG_R', 'MODEL_MAGERR_R', 'MODEL_MAG_I', 'MODEL_MAGERR_I', 'MODEL_MAG_Z', 'MODEL_MAGERR_Z', 'X', 'Y', 'Z']
n_feat = len(feat)

# Which labels to predicts
labs = ['M500']
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
n_rand_rot = 128
n_rand_perm = 128
num_neurons_in_f = 80
num_layers_in_rho = 1
num_neurons_in_rho = 80

def get_randmat():
    v = np.random.randn(4)
    v /= np.sqrt(np.sum(v**2.))
    w, x, y, z, = v
    mat = np.array([
        [1-2*y**2-2*z**2,     2*x*y-2*z*w,     2*x*z+2*y*w],
        [    2*x*y+2*z*w, 1-2*x**2-2*z**2,     2*y*z-2*x*w],
        [    2*x*z-2*y*w,     2*y*z+2*x*w, 1-2*x**2-2*y**2],
    ])
    return mat

### v1
'''
def generator(inputs, labels, n, batch_size, ixs):
    i = 0
    while True:
        n_mem = clu_num[ixs[i%n]]
        inputs_batch = np.zeros((batch_size, n_rand_perm, n_mem_max * n_feat))
        labels_batch = np.zeros((batch_size, n_labs))
        for j in range(batch_size):
            tmp_input = inputs[i%n]
            rm1 = get_randmat()
            rm2 = get_randmat()
            rm3 = get_randmat()
            new_vecs = hp.rotator.rotateVector(rm1, tmp_input[:, -3:].T)
            new_vecs = hp.rotator.rotateVector(rm2, new_vecs)
            tmp_input[:, -3:] = hp.rotator.rotateVector(rm3, new_vecs).T
            ######
            ix = np.random.rand(n_rand_perm, n_mem).argsort(axis=1)[:, :n_mem_max]
            # inputs_batch[j, :, :, :] = inputs[i%n][ix, :]
            inputs_batch[j, :, :] = tmp_input[ix, :].reshape(n_rand_perm, -1)
            labels_batch[j, :] = labels[i%n]
        i += 1
        yield inputs_batch, labels_batch
'''

### v2
'''
def generator(inputs, labels, n, batch_size, ixs):
    i = 0
    while True:
        inputs_batch = np.zeros((batch_size, n_rand_perm, n_mem_max * n_feat))
        labels_batch = np.zeros((batch_size, n_labs))
        for j in range(batch_size):
            n_mem = clu_num[ixs[i%n]]
            tmp_input = inputs[i%n]
            rm1 = get_randmat()
            rm2 = get_randmat()
            rm3 = get_randmat()
            new_vecs = hp.rotator.rotateVector(rm1, tmp_input[:, -3:].T)
            new_vecs = hp.rotator.rotateVector(rm2, new_vecs)
            tmp_input[:, -3:] = hp.rotator.rotateVector(rm3, new_vecs).T
            ######
            ix = np.random.rand(n_rand_perm, n_mem).argsort(axis=1)[:, :n_mem_max]
            # inputs_batch[j, :, :, :] = inputs[i%n][ix, :]
            inputs_batch[j, :, :] = tmp_input[ix, :].reshape(n_rand_perm, -1)
            labels_batch[j, :] = labels[i%n]
            i += 1
        yield inputs_batch, labels_batch
'''

### v3
def sk(v):
    return np.array([
        [0., -v[2], v[1]],
        [v[2], 0., -v[0]],
        [-v[1], v[0], 0.],
    ])
def generator(inputs, labels, n, batch_size, ixs):
    i = 0
    while True:
        inputs_batch = np.zeros((batch_size, n_rand_perm, n_mem_max * n_feat))
        labels_batch = np.zeros((batch_size, n_labs))
        for j in range(batch_size):
            tmp_input = inputs[i%n]
            v = tmp_input[:, -3:]
            ref_v = np.mean(v, axis=0)
            ref_v /= np.sqrt(np.sum(ref_v**2.))
            end_v = np.random.randn(3)
            end_v /= np.sqrt(np.sum(end_v**2.))
            skv = sk(np.cross(ref_v, end_v))
            c = np.dot(ref_v, end_v)
            mat = np.eye(3) + skv + np.dot(skv, skv) / (1. + c)
            v1 = hp.rotator.rotateVector(mat, v.T).T
            rot_ang = np.random.rand() * 2. * np.pi
            v_par = np.dot(v1, end_v)[:, None] * end_v
            v_ort = v1 - v_par
            w = np.cross(end_v, v_ort)
            nv = np.sqrt(np.sum(v_ort**2., axis=1))
            nw = np.sqrt(np.sum(w**2., axis=1))
            tmp = (
                np.cos(rot_ang) * v_ort
                + np.sin(rot_ang) * w * (nv / nw)[:, None]
            )
            v2 = tmp + v_par
            tmp_input[:, -3:] = v2
            ######
            n_mem = clu_num[ixs[i%n]]
            ix = np.random.rand(n_rand_perm, n_mem).argsort(axis=1)[:, :n_mem_max]
            # inputs_batch[j, :, :, :] = inputs[i%n][ix, :]
            inputs_batch[j, :, :] = tmp_input[ix, :].reshape(n_rand_perm, -1)
            labels_batch[j, :] = labels[i%n]
            i += 1
        yield inputs_batch, labels_batch




# mirrored_strategy = tf.distribute.MirroredStrategy()
# with mirrored_strategy.scope():
if True:
    inputs = keras.Input(shape=(n_rand_perm, n_mem_max * n_feat))
    lay1 = Dense(
        num_neurons_in_f,
        activation="relu", # b2/b3/b4
        # activation="tanh",   # b
        # activation="linear",
        # kernel_regularizer=regularizers.l2(0.001),
    )
    out_lay1 = lay1(inputs)
    # out_lay1 = Dropout(0.2)(lay1(inputs))
    output = tf.math.reduce_mean(
        out_lay1,
        axis=1,
    )
    rho_lays = []
    for i in range(num_layers_in_rho):
        rho_lays.append(
            Dense(
                num_neurons_in_rho,
                activation="relu",
                # activation="tanh",
                # activation="linear",
                kernel_initializer=tf.keras.initializers.GlorotUniform(),
                # kernel_regularizer=regularizers.l2(0.001),
            )
        )
        output = rho_lays[-1](output)
        # output = Dropout(0.2)(rho_lays[-1](output))
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

model.load_weights('saved_models/inn_mass_calc_nn_v3/inn_mass_calc_nn_v3')
sys.exit()

log_filename = "saved_models/inn_mass_calc_nn_v7/inn_mass_calc_nn_v7.log"
log_cb = tf.keras.callbacks.CSVLogger(
    log_filename,
    separator=' ',
    append=True,
)

chk_filename = "saved_models/inn_mass_calc_nn_v7/inn_mass_calc_nn_v7"
chk_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath=chk_filename,
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    save_weights_only=True,
)

chk2_filename = "saved_models/inn_mass_calc_nn_v7/inn_mass_calc_nn_v7_last"
chk2_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath=chk2_filename,
    monitor='val_loss',
    verbose=1,
    save_best_only=False,
    save_weights_only=True,
)

# ANN hyperparamters
nb_epoch=10000

# batch_size = n_rand_rot
# batch_size = 1024
history = model.fit(
    # generator(X_train, Y_train, n_train, batch_size, ix_train),
    # validation_data=generator(X_valid, Y_valid, n_valid, batch_size, ix_valid),
    generator(X_train, Y_train, n_train, n_train, ix_train),
    validation_data=generator(X_valid, Y_valid, n_valid, n_valid, ix_valid),
    # steps_per_epoch=n_train,
    # validation_steps=n_valid,
    steps_per_epoch=50*2,
    validation_steps=50*2,
    epochs=nb_epoch,
    callbacks=[log_cb, chk_cb, chk2_cb],
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
    # kernel_regularizer=regularizers.l2(0.001),
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
            # kernel_regularizer=regularizers.l2(0.001),
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
    # ix = np.random.rand(n_mem, n_rand_perm_test).argsort(axis=0)
    # tmp_input = np.hstack((
    #     X_valid[i].reshape(n_mem, n_feat).T[:,ix].T.reshape(n_rand_perm_test, -1),
    #     np.zeros((n_rand_perm_test, (n_mem_max - n_mem) * n_feat)),
    # ))
    ix = np.random.rand(n_rand_perm_test, n_mem).argsort(axis=1)[:, :n_mem_max]
    tmp_input = X_valid[i][ix, :].reshape(n_rand_perm_test, -1)
    out = model_test.predict(tmp_input)
    eval_out[i, :] = np.mean(out, axis=0)

eval_out_train = np.zeros((n_train, n_labs))
n_rand_perm_test = 10000
for i in tqdm(range(n_train)):
    n_mem = clu_num[ix_train[i]]
    # ix = np.random.rand(n_mem, n_rand_perm_test).argsort(axis=0)
    # tmp_input = np.hstack((
    #     X_train[i].reshape(n_mem, n_feat).T[:,ix].T.reshape(n_rand_perm_test, -1),
    #     np.zeros((n_rand_perm_test, (n_mem_max - n_mem) * n_feat)),
    # ))
    ix = np.random.rand(n_rand_perm_test, n_mem).argsort(axis=1)[:, :n_mem_max]
    tmp_input = X_train[i][ix, :].reshape(n_rand_perm_test, -1)
    out = model_test.predict(tmp_input)
    eval_out_train[i, :] = np.mean(out, axis=0)



plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
errlow = clu_full_data['M500_ERR_LOW'][ix_valid]
errupp = clu_full_data['M500_ERR_UPP'][ix_valid]
plt.hist(
    (eval_out[:, 0] - Y_valid[:, 0]),
    bins=32,
    alpha=0.4
)
plt.xlabel('(M_NN - M_Comprass)')
plt.title('validation set')
plt.subplot(1,2,2)
errlow = clu_full_data['M500_ERR_LOW'][ix_train]
errupp = clu_full_data['M500_ERR_UPP'][ix_train]
plt.hist(
    (eval_out_train[:, 0] - Y_train[:, 0]),
    bins=32,
    alpha=0.4
)
plt.xlabel('(M_NN - M_Comprass)')
plt.title('training set')
plt.savefig('tmp3.pdf')



plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
errlow = clu_full_data['M500_ERR_LOW'][ix_valid]
errupp = clu_full_data['M500_ERR_UPP'][ix_valid]
plt.hist(
    (eval_out[:, 0] - Y_valid[:, 0]) / errlow,
    bins=32,
    alpha=0.4
)
plt.hist(
    (eval_out[:, 0] - Y_valid[:, 0]) / errupp,
    bins=32,
    alpha=0.4
)
plt.xlabel('(M_NN - M_Comprass)/error_Comprass')
plt.title('validation set')
plt.subplot(1,2,2)
errlow = clu_full_data['M500_ERR_LOW'][ix_train]
errupp = clu_full_data['M500_ERR_UPP'][ix_train]
plt.hist(
    (eval_out_train[:, 0] - Y_train[:, 0]) / errlow,
    bins=32,
    alpha=0.4
)
plt.hist(
    (eval_out_train[:, 0] - Y_train[:, 0]) / errupp,
    bins=32,
    alpha=0.4
)
plt.xlabel('(M_NN - M_Comprass)/error_Comprass')
plt.title('training set')
plt.savefig('tmp2.pdf')


mi = min(np.min(Y_valid[:, 0]), np.min(eval_out[:, 0]))
ma = max(np.max(Y_valid[:, 0]), np.max(eval_out[:, 0]))
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
errlow = clu_full_data['M500_ERR_LOW'][ix_valid]
errupp = clu_full_data['M500_ERR_UPP'][ix_valid]
plt.errorbar(Y_valid[:, 0], eval_out[:, 0], xerr=np.array([errlow, errupp]), fmt='+')
plt.xlabel('M500 from Comprass')
plt.ylabel('M500 from NN')
plt.title('validation set')
plt.plot([mi, ma], [mi, ma], color='red')
plt.subplot(1,2,2)
errlow = clu_full_data['M500_ERR_LOW'][ix_train]
errupp = clu_full_data['M500_ERR_UPP'][ix_train]
plt.errorbar(Y_train[:, 0], eval_out_train[:, 0], xerr=np.array([errlow, errupp]), fmt='+')
plt.xlabel('M500 from Comprass')
plt.ylabel('M500 from NN')
plt.title('training set')
plt.plot([mi, ma], [mi, ma], color='red')
plt.savefig('tmp1.pdf')


mi = min(np.min(Y_valid[:, 0]), np.min(eval_out[:, 0]))
ma = max(np.max(Y_valid[:, 0]), np.max(eval_out[:, 0]))
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.hist2d(Y_valid[:, 0], eval_out[:, 0], bins=32, norm=matplotlib.colors.LogNorm(), range=[[mi, ma], [mi, ma]])
plt.xlabel('M500 from Comprass')
plt.ylabel('M500 from NN')
plt.title('validation set')
plt.colorbar()
plt.plot([mi, ma], [mi, ma], color='red')
plt.subplot(1,2,2)
plt.hist2d(Y_train[:, 0], eval_out_train[:, 0], bins=32, norm=matplotlib.colors.LogNorm(), range=[[mi, ma], [mi, ma]])
plt.xlabel('M500 from Comprass')
plt.ylabel('M500 from NN')
plt.colorbar()
plt.title('training set')
plt.plot([mi, ma], [mi, ma], color='red')
plt.savefig('tmp0.pdf')



########################

'''
t0 = np.genfromtxt('saved_models/inn_mass_calc/inn_mass_calc.log',names=True)
t00 = np.genfromtxt('saved_models/inn_mass_calc_v2/inn_mass_calc_v2.log',names=True)
t1 = np.genfromtxt('saved_models/inn_mass_calc_gru/inn_mass_calc_gru.log',names=True)
t2 = np.genfromtxt('saved_models/inn_mass_calc_gru_v2/inn_mass_calc_gru_v2.log',names=True)
plt.subplot(2, 2, 1)
plt.title("Dense(10) + Dense(10)")
plt.plot(t0['loss'], alpha=0.5, label='Training loss')
plt.plot(t0['val_loss'], alpha=0.5, label='Validation loss')
plt.legend()
plt.ylim(0, 4)
plt.subplot(2, 2, 2)
plt.title("Dense(10) + Dropout(0.1) + Dense(10)")
plt.plot(t00['loss'], alpha=0.5, label='Training loss')
plt.plot(t00['val_loss'], alpha=0.5, label='Validation loss')
plt.legend()
plt.ylim(0, 4)
plt.subplot(2, 2, 3)
plt.title("GRU(10) + Dense(10)")
plt.plot(t1['loss'], alpha=0.5, label='Training loss')
plt.plot(t1['val_loss'], alpha=0.5, label='Validation loss')
plt.legend()
plt.ylim(0, 4)
plt.subplot(2, 2, 4)
plt.title("GRU(20) + Dense(20)")
plt.plot(t2['loss'], alpha=0.5, label='Training loss')
plt.plot(t2['val_loss'], alpha=0.5, label='Validation loss')
plt.legend()
plt.ylim(0, 4)
plt.show()
'''

import matplotlib.pyplot as plt
import numpy as np

fig, axs = plt.subplots(1, 2, figsize=(10,5))
t1 = np.genfromtxt('saved_models/inn_mass_calc_gru_v5/inn_mass_calc_gru_v5.log',names=True)
t1b = np.genfromtxt('saved_models/inn_mass_calc_lstm/inn_mass_calc_lstm.log',names=True)
axs[0].plot(t1['loss'], alpha=0.5,label='loss GRU')
axs[0].plot(t1['val_loss'], alpha=0.5,label='val_loss GRU')
axs[0].plot(t1b['loss'], alpha=0.5,label='loss LSTM')
axs[0].plot(t1b['val_loss'], alpha=0.5,label='val_loss LSTM')
axs[0].set_ylim(0, 4.5)
axs[0].set_xlabel("Epochs")
axs[0].legend()

fnames = [
    # 'saved_models/inn_mass_calc_nn/inn_mass_calc_nn.log',
    # 'saved_models/inn_mass_calc_nn_v2/inn_mass_calc_nn_v2.log',
    'saved_models/inn_mass_calc_nn_v3/inn_mass_calc_nn_v3.log',
    'saved_models/inn_mass_calc_nn_v4/inn_mass_calc_nn_v4.log',
    'saved_models/inn_mass_calc_nn_v5/inn_mass_calc_nn_v5.log',
    'saved_models/inn_mass_calc_nn_v6/inn_mass_calc_nn_v6.log',
]
labs = [
    # 'Dense 1',
    # 'Dense 2',
    'Dense 3',
    'Dense 4',
    'Dense 5',
    'Dense 6',
]
cols = plt.cm.tab20(np.linspace(0., 1., 20, endpoint=False))
ct = 0
for fn, la in zip(fnames, labs):
    t = np.genfromtxt(fn, names=True)
    axs[1].plot(t['loss'], alpha=0.5, label='loss ' + la, color=cols[ct])
    ct +=1
    axs[1].plot(t['val_loss'], alpha=0.5, label='val_loss ' + la, color=cols[ct])
    ct +=1
axs[1].set_ylim(0, 4.5)
axs[1].set_xlabel("Epochs")
axs[1].set_xscale("log")
axs[1].legend()
plt.savefig('tmp.pdf')
