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

ix = int(sys.argv[1])

input_file = [
    'inn_mass_nmm-30_nperm-128_nshft-64_f0-64_r0-64_act-tanh_df-none_dr-none_l2f-none_l2r-none_loss-mean_squared_error_lr-0p001', #0
    'inn_mass_nmm-30_nperm-128_nshft-64_f0-32_r0-32_act-tanh_df-none_dr-none_l2f-none_l2r-none_loss-mean_squared_error_lr-0p0001', #1
    'inn_mass_nmm-30_nperm-128_nshft-64_f0-128_r0-128_act-tanh_df-none_dr-none_l2f-none_l2r-none_loss-mean_squared_error_lr-0p0001', #2
    'inn_mass_nmm-30_nperm-128_nshft-16_f0-64_r0-64_act-tanh_df-none_dr-none_l2f-none_l2r-none_loss-mean_squared_error_lr-0p0001', #3
    'inn_mass_nmm-30_nperm-128_nshft-64_f0-64_f1-64_r0-64_act-tanh_df-none_dr-none_l2f-none_l2r-none_loss-mean_squared_error_lr-0p0001', #4
    'inn_mass_nmm-30_nperm-128_nshft-64_f0-64_r0-64_act-tanh_df-0p5_dr-0p5_l2f-none_l2r-none_loss-mean_squared_error_lr-0p0001', #5
    'inn_mass_nmm-10_nperm-128_nshft-64_f0-64_r0-64_act-tanh_df-none_dr-none_l2f-none_l2r-none_loss-mean_squared_error_lr-0p0001', #6
    'inn_mass_nmm-30_nperm-128_nshft-64_f0-64_r0-64_act-tanh_df-none_dr-none_l2f-none_l2r-none_loss-mean_squared_error_lr-0p0001', #7
    'inn_log10mass_nmm-30_nperm-128_nshft-64_f0-64_r0-64_act-tanh_df-none_dr-none_l2f-none_l2r-none_loss-mean_squared_error_lr-0p0001', #8
    'inn_mass_nmm-30_nperm-128_nshft-64_f0-64_r0-64_act-tanh_df-none_dr-none_l2f-none_l2r-none_loss-mean_squared_error_lr-0p0001_8xpermvalid', #9
    'inn_mass_nmm-30_nperm-128_nshft-64_f0-32_f1-32_f2-32_r0-32_r1-32_act-tanh_df-0p5_dr-0p5_l2f-none_l2r-none_loss-mean_squared_error_lr-0p0001', #10
    'inn_mass_nmm-10_nperm-128_nshft-64_f0-32_f1-32_f2-32_r0-32_r1-32_act-tanh_df-0p5_dr-0p5_l2f-none_l2r-none_loss-mean_squared_error_lr-0p0001', #11
    'inn_mass_nmm-5_nperm-128_nshft-64_f0-32_f1-32_f2-32_r0-32_r1-32_act-tanh_df-0p5_dr-0p5_l2f-none_l2r-none_loss-mean_squared_error_lr-0p0001', #12
    'inn_mass_nmm-10_nperm-128_nshft-64_f0-64_f1-64_f2-64_r0-64_act-tanh_df-0p5_dr-0p5_l2f-0p001_l2r-0p001_loss-mean_squared_error_lr-0p0001', #13
    'inn_mass_nmm-10_nperm-128_nshft-64_f0-64_f1-64_f2-64_r0-64_act-tanh_df-0p5_dr-0p5_l2f-0p01_l2r-0p01_loss-mean_squared_error_lr-0p0001', #14
    'inn_mass_nmm-30_nperm-128_nshft-64_f0-64_r0-64_act-tanh_df-0p5_dr-0p5_l2f-none_l2r-none_loss-mean_squared_error_lr-0p001', #15
][ix]

spl = input_file.split('_')
if spl[1] == 'mass':
    labs = ['M500']
elif spl[1] == 'log10mass':
    labs = ['log10_M500']
tmp_spl = spl[2].split('-')
n_mem_max = int(tmp_spl[1])
tmp_spl = spl[3].split('-')
n_rand_perm = int(tmp_spl[1])
tmp_spl = spl[4].split('-')
n_rand_shift = int(tmp_spl[1])
i = 5
num_layers_in_f = 0
num_neurons_in_f = []
while spl[i][0] == 'f':
    tmp_spl = spl[i].split('-')
    num_layers_in_f += 1
    num_neurons_in_f.append(int(tmp_spl[1]))
    i += 1
num_layers_in_rho = 0
num_neurons_in_rho = []
while spl[i][0] == 'r':
    tmp_spl = spl[i].split('-')
    num_layers_in_rho += 1
    num_neurons_in_rho.append(int(tmp_spl[1]))
    i += 1
tmp_spl = spl[i].split('-')
act = tmp_spl[1]
i += 1
tmp_spl = spl[i].split('-')
if tmp_spl[1] == 'none':
    drop_f = None
else:
    drop_f = float(tmp_spl[1].replace('p','.'))
i += 1
tmp_spl = spl[i].split('-')
if tmp_spl[1] == 'none':
    drop_rho = None
else:
    drop_rho = float(tmp_spl[1].replace('p','.'))
i += 1
tmp_spl = spl[i].split('-')
if tmp_spl[1] == 'none':
    l2reg_f = None
else:
    l2reg_f = float(tmp_spl[1].replace('p','.'))
i += 1
tmp_spl = spl[i].split('-')
if tmp_spl[1] == 'none':
    l2reg_rho = None
else:
    l2reg_rho = float(tmp_spl[1].replace('p','.'))
loss = 'mean_squared_error'
i += 1
while spl[i][:3] != 'lr-':
    i += 1
tmp_spl = spl[i].split('-')
learn_rate = float(tmp_spl[1].replace('p','.'))

in_wgt = 'saved_models/%s/best_weights' % input_file


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
clu_full_data['M500'] = np.zeros(len(hdul1[1].data))
clu_full_data['log10_M500'] = np.zeros(len(hdul1[1].data))
clu_full_data['M500_ERR_LOW'] = np.zeros(len(hdul1[1].data))
clu_full_data['M500_ERR_UPP'] = np.zeros(len(hdul1[1].data))

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

# Read cluster mass from Comprass catalogue and add to cluster dictionary
hdul3 = fits.open("/home/users/ilic/ML/other_cats/comprass.fits")
ix_red_comp = []
for i in range(len(hdul3[1].data)):
    if hdul3[1].data['REDMAPPER'][i] != '':
        tmp = np.where(hdul3[1].data['REDMAPPER'][i] == hdul1[1].data['NAME'])[0]
        if len(tmp) != 0:
            clu_full_data['M500'][tmp[0]] = hdul3[1].data['M500'][i]
            clu_full_data['log10_M500'][tmp[0]] = np.log10(hdul3[1].data['M500'][i])
            clu_full_data['M500_ERR_LOW'][tmp[0]] = hdul3[1].data['M500_ERR_LOW'][i]
            clu_full_data['M500_ERR_UPP'][tmp[0]] = hdul3[1].data['M500_ERR_UPP'][i]
            ix_red_comp.append(tmp[0])

# Get IDs of clusters and corresponding number of members
clu_ids, clu_num = np.unique(mem_full_data['ID'], return_counts=True)
n_clus_tot = len(clu_ids)
ix_keep_clu = []
for i in range(n_clus_tot):
    if i in ix_red_comp:
        ix_keep_clu.append(i) # Keep only cluster with Comprass mass

# Shuffle the clusters with a fixed seed
np.random.seed(457896)
np.random.shuffle(ix_keep_clu)
np.random.seed()
n_clus = len(ix_keep_clu)

# Choose which mem gal features to use in training
feat = ['R', 'P', 'P_FREE', 'THETA_I', 'THETA_R', 'IMAG', 'IMAG_ERR', 'MODEL_MAG_U', 'MODEL_MAGERR_U', 'MODEL_MAG_G', 'MODEL_MAGERR_G', 'MODEL_MAG_R', 'MODEL_MAGERR_R', 'MODEL_MAG_I', 'MODEL_MAGERR_I', 'MODEL_MAG_Z', 'MODEL_MAGERR_Z', 'X', 'Y', 'Z']
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

# mirrored_strategy = tf.distribute.MirroredStrategy()
# with mirrored_strategy.scope():
if True:
    inputs = keras.Input(shape=(n_mem_max * n_feat))
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
            outputs = f_layers[-1](inputs)
        else:
            outputs = f_layers[-1](outputs)
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
        outputs = rho_layers[-1](outputs)
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
model.load_weights(in_wgt)

eval_out = np.zeros((3, n_valid, n_labs))
n_rand_perm_test = 4096
for i in tqdm(range(n_valid)):
    n_mem = clu_num[ix_valid[i]]
    ix = np.random.rand(n_rand_perm_test, n_mem).argsort(axis=1)[:, :n_mem_max]
    tmp_input = X_valid[i][ix, :].reshape(n_rand_perm_test, -1)
    out = model.predict(tmp_input)
    eval_out[0, i, :] = np.mean(out[:n_rand_perm_test//4, :], axis=0)
    eval_out[1, i, :] = np.mean(out[:n_rand_perm_test//2, :], axis=0)
    eval_out[2, i, :] = np.mean(out, axis=0)

eval_out_train = np.zeros((3, n_train, n_labs))
n_rand_perm_test = 4096
for i in tqdm(range(n_train)):
    n_mem = clu_num[ix_train[i]]
    ix = np.random.rand(n_rand_perm_test, n_mem).argsort(axis=1)[:, :n_mem_max]
    tmp_input = X_train[i][ix, :].reshape(n_rand_perm_test, -1)
    out = model.predict(tmp_input)
    eval_out_train[0, i, :] = np.mean(out[:n_rand_perm_test//4, :], axis=0)
    eval_out_train[1, i, :] = np.mean(out[:n_rand_perm_test//2, :], axis=0)
    eval_out_train[2, i, :] = np.mean(out, axis=0)

np.savez(
    'saved_models/%s/inference' % input_file,
    eval_out=eval_out,
    eval_out_train=eval_out_train,
    Y_train=Y_train,
    Y_valid=Y_valid,
    errY_train=[
        clu_full_data['M500_ERR_LOW'][ix_train],
        clu_full_data['M500_ERR_UPP'][ix_train],
    ],
    errY_valid=[
        clu_full_data['M500_ERR_LOW'][ix_valid],
        clu_full_data['M500_ERR_UPP'][ix_valid],
    ],
)