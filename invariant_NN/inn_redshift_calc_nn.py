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


frac = 9./10. # fraction of data dedicated to training (first try was 4/5)


# Free parameters
# Define NN model
n_mem_max = 2
n_rand_perm = 128
n_rand_shift = 64
num_layers_in_f = 3
num_neurons_in_f = [128,128,128]
num_layers_in_rho = 0
num_neurons_in_rho = []
act = "tanh"
# drop_f = None
drop_f = 0.5
# drop_rho = None
drop_rho = 0.5
l2reg_f = None
# l2reg_f = 0.01
l2reg_rho = None
# l2reg_rho = 0.01
loss = 'mean_squared_error'
learn_rate = 0.0001
labs = ['Z_SPEC']
suff = ''
# suff = '_8xpermvalid'
in_wgt = None
# in_wgt = 'saved_models/inn_zspec_nmm-10_nperm-128_nshft-64_f0-128_f1-128_r0-128_act-tanh_df-0p5_dr-0p5_l2f-none_l2r-none_loss-mean_squared_error_lr-0p0001/last_weights'


# Path to fits files
pathData="/home/users/ilic/ML/SDSS_fits_data/"

# Read data in fits files
hdul1 = fits.open(pathData+'redmapper_dr8_public_v6.3_catalog.fits')
hdul2 = fits.open(pathData+'redmapper_dr8_public_v6.3_members.fits')

# Create dictionary of cluster data from fits + extra derived quantities
clu_full_data = {}
for n in hdul1[1].data.dtype.names:
    clu_full_data[n] = hdul1[1].data[n]
ras = clu_full_data['RA'] / 180. * np.pi
decs = clu_full_data['DEC'] / 180. * np.pi
clu_full_data['X'] = np.cos(decs) * np.cos(ras)
clu_full_data['Y'] = np.cos(decs) * np.sin(ras)
clu_full_data['Z'] = np.sin(decs)

# Create dictionary of mem gal data from fits + extra derived quantities
mem_full_data = {}
for n in hdul2[1].data.dtype.names:
    mem_full_data[n] = hdul2[1].data[n]
ras = mem_full_data['RA'] / 180. * np.pi
decs = mem_full_data['DEC'] / 180. * np.pi
mem_full_data['X'] = np.cos(decs) * np.cos(ras)
mem_full_data['Y'] = np.cos(decs) * np.sin(ras)
mem_full_data['Z'] = np.sin(decs)

# Print all variable in both dictionary for info
print("Variables in cluster catalog:")
print(clu_full_data.keys())
print("Variables in member catalog:")
print(mem_full_data.keys())
# sys.exit()

# Get IDs of clusters and corresponding number of members
clu_ids, clu_num = np.unique(mem_full_data['ID'], return_counts=True)
n_clus_tot = len(clu_ids)
ix_keep_clu = []
for i in range(n_clus_tot):
    if clu_full_data['Z_SPEC'][i] != -1.:
        ix_keep_clu.append(i) # Keep only cluster with z_spec

# Shuffle the clusters with a fixed seed
np.random.seed(121212)
np.random.shuffle(ix_keep_clu)
np.random.seed()
n_clus = len(ix_keep_clu)

# Choose which mem gal features to use in training
feat = [
    'R',
    # 'P',
    # 'P_FREE',
    'THETA_I',
    'THETA_R',
    'IMAG',
    'IMAG_ERR',
    'MODEL_MAG_U',
    'MODEL_MAGERR_U',
    'MODEL_MAG_G',
    'MODEL_MAGERR_G',
    'MODEL_MAG_R',
    'MODEL_MAGERR_R',
    'MODEL_MAG_I',
    'MODEL_MAGERR_I',
    'MODEL_MAG_Z',
    'MODEL_MAGERR_Z',
    'X',
    'Y',
    'Z',
]
n_feat = len(feat)

# Choose which cluster labels to predict
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

# Build label array
allY = np.zeros((n_clus, n_labs))
for ix, l in enumerate(labs):
    allY[:, ix] = clu_full_data[l][ix_keep_clu]

# Shuffle and split training and validation datasets
max_ix = int(n_clus * frac)
X_train = allX[:max_ix]
X_valid = allX[max_ix:]
Y_train = allY[:max_ix]
Y_valid = allY[max_ix:]
ix_train = ix_keep_clu[:max_ix]
ix_valid = ix_keep_clu[max_ix:]
n_train = len(X_train)
n_valid = len(X_valid)

# Function generating batches
def sk(v):
    return np.array([
        [0., -v[2], v[1]],
        [v[2], 0., -v[0]],
        [-v[1], v[0], 0.],
    ])
def train_generator(inputs, labels, n, batch_size, ixs):
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
def valid_generator(inputs, labels, n, batch_size, ixs):
    inputs_batch = np.zeros((batch_size, n_rand_perm, n_mem_max * n_feat))
    i = 0
    while True:
        labels_batch = np.zeros((batch_size, n_labs))
        for j in range(batch_size):
            tmp_input = inputs[i%n]
            n_mem = clu_num[ixs[i%n]]
            ix = np.random.rand(n_rand_perm, n_mem).argsort(axis=1)[:, :n_mem_max]
            inputs_batch[j, :, :] = tmp_input[ix, :].reshape(n_rand_perm, -1)
            labels_batch[j, :] = labels[i%n]
            i += 1
        yield inputs_batch, labels_batch


# mirrored_strategy = tf.distribute.MirroredStrategy()
# with mirrored_strategy.scope():
if True:
    inputs = keras.Input(shape=(None, n_mem_max * n_feat))
    f_layers = []
    for i in range(num_layers_in_f):
        if l2reg_f is None:
            f_layers.append(
                Dense(
                    num_neurons_in_f[i],
                    activation=act,
                )
            )
        else:
            f_layers.append(
                Dense(
                    num_neurons_in_f[i],
                    activation=act,
                    kernel_regularizer=regularizers.l2(l2reg_f),
                )
            )
        if i == 0:
            if drop_f is None:
                outputs = f_layers[-1](inputs)
            else:
                outputs = Dropout(drop_f)(f_layers[-1](inputs))
        else:
            if drop_f is None:
                outputs = f_layers[-1](outputs)
            else:
                outputs = Dropout(drop_f)(f_layers[-1](outputs))
    outputs = tf.math.reduce_mean(
        outputs,
        axis=1,
    )
    rho_layers = []
    for i in range(num_layers_in_rho):
        if l2reg_rho is None:
            rho_layers.append(
                Dense(
                    num_neurons_in_rho[i],
                    activation=act,
                )
            )
        else:
            rho_layers.append(
                Dense(
                    num_neurons_in_rho[i],
                    activation=act,
                    kernel_regularizer=regularizers.l2(l2reg_rho),
                )
            )
        if drop_rho is None:
            outputs = rho_layers[-1](outputs)
        else:
            outputs = Dropout(drop_rho)(rho_layers[-1](outputs))
    lay3 = Dense(
        n_labs,
        activation="linear",
    )
    outputs = lay3(outputs)
    model = keras.Model(inputs, outputs)
    model.compile(
        loss=loss,
        optimizer=Adam(learning_rate=learn_rate),
        metrics=['accuracy'],
    )
model.summary()

# Output name
fname = "inn"
if labs[0] == "Z_SPEC":
    fname += "_zspec"
elif labs[0] == "log10_Z_SPEC":
    fname += "_log10zspec"
fname += "_nmm-%s" % n_mem_max
fname += "_nperm-%s" % n_rand_perm
fname += "_nshft-%s" % n_rand_shift
for i, n in enumerate(num_neurons_in_f):
    fname += "_f%s-%s" % (i, n)
for i, n in enumerate(num_neurons_in_rho):
    fname += "_r%s-%s" % (i, n)
fname += "_act-%s" % act
fname += "_df-%s" % (drop_f if drop_f is not None else 'none')
fname += "_dr-%s" % (drop_rho if drop_rho is not None else 'none')
fname += "_l2f-%s" % (l2reg_f if l2reg_f is not None else 'none')
fname += "_l2r-%s" % (l2reg_rho if l2reg_rho is not None else 'none')
fname += "_loss-%s" % loss
fname += "_lr-%s" % learn_rate
fname += suff
fname = fname.replace('.', 'p')

# Input weights if asked
if in_wgt is not None:
    model.load_weights(in_wgt)

# Output logs
os.system("mkdir saved_models/%s" % fname)
log_filename = "saved_models/%s/output.log" % fname
log_cb = tf.keras.callbacks.CSVLogger(
    log_filename,
    separator=' ',
    append=True,
)
chk_filename = "saved_models/%s/best_weights" % fname
chk_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath=chk_filename,
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    save_weights_only=True,
)
chk2_filename = "saved_models/%s/last_weights" % fname
chk2_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath=chk2_filename,
    monitor='val_loss',
    verbose=1,
    save_best_only=False,
    save_weights_only=True,
)

nb_epoch=10000
history = model.fit(
    train_generator(X_train, Y_train, n_train, n_train, ix_train),
    validation_data=valid_generator(X_valid, Y_valid, n_valid, n_valid, ix_valid),
    steps_per_epoch=n_rand_shift,
    validation_steps=1,
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
