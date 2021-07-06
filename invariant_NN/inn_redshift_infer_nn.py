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

ix = 0

input_file = [
    'inn_zspec_nmm-10_nperm-128_nshft-64_f0-128_f1-128_r0-128_act-tanh_df-0p5_dr-0p5_l2f-none_l2r-none_loss-mean_squared_error_lr-0p0001', #0
][ix]

spl = input_file.split('_')
if spl[1] == 'zspec':
    labs = ['Z_SPEC']
elif spl[1] == 'log10zspec':
    labs = ['log10_Z_SPEC']
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
frac = 4./5. # fraction of data dedicated to training
# frac = 9./10. # fraction of data dedicated to training
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
)

sys.exit()

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

f = np.load(
    'saved_models/'
    'inn_zspec_nmm-10_nperm-128_nshft-64_f0-128_f1-128_r0-128_act-tanh_df-0p5_dr-0p5_l2f-none_l2r-none_loss-mean_squared_error_lr-0p0001/'
    'inference.npz',
    allow_pickle=True,
)

Y_valid = f['Y_valid']
eval_out = f['eval_out']
Y_train = f['Y_train']
eval_out_train = f['eval_out_train']


plt.figure(figsize=(10,10))

mi = min(np.min(Y_valid[:, 0]), np.min(clu_full_data['Z_LAMBDA'][ix_valid]))
ma = max(np.max(Y_valid[:, 0]), np.max(clu_full_data['Z_LAMBDA'][ix_valid]))
plt.subplot(2,2,3)
h1 = plt.hist2d(Y_valid[:, 0], clu_full_data['Z_LAMBDA'][ix_valid], bins=64, norm=matplotlib.colors.LogNorm(), range=[[mi, ma], [mi, ma]])
plt.xlabel('Z_SPEC from redmapper')
plt.ylabel('Z_LAMBDA from redmapper')
# plt.title('validation set')
plt.colorbar()
plt.plot([mi, ma], [mi, ma], color='red')

mi = min(np.min(Y_train[:, 0]), np.min(clu_full_data['Z_LAMBDA'][ix_train]))
ma = max(np.max(Y_train[:, 0]), np.max(clu_full_data['Z_LAMBDA'][ix_train]))
plt.subplot(2,2,4)
h2 = plt.hist2d(Y_train[:, 0], clu_full_data['Z_LAMBDA'][ix_train], bins=64, norm=matplotlib.colors.LogNorm(), range=[[mi, ma], [mi, ma]])
plt.xlabel('Z_SPEC from redmapper')
plt.ylabel('Z_LAMBDA from redmapper')
plt.colorbar()
# plt.title('training set')
plt.plot([mi, ma], [mi, ma], color='red')

mi = min(np.min(Y_valid[:, 0]), np.min(eval_out[2, :, 0]))
ma = max(np.max(Y_valid[:, 0]), np.max(eval_out[2, :, 0]))
plt.subplot(2,2,1)
plt.hist2d(Y_valid[:, 0], eval_out[2, :, 0], bins=64, norm=matplotlib.colors.LogNorm(), range=[[mi, ma], [mi, ma]], vmax=h1[0].max())
# plt.xlabel('Z_SPEC from redmapper')
plt.ylabel('Z from NN')
plt.title('validation set')
plt.colorbar()
plt.plot([mi, ma], [mi, ma], color='red')

mi = min(np.min(Y_train[:, 0]), np.min(eval_out_train[2, :, 0]))
ma = max(np.max(Y_train[:, 0]), np.max(eval_out_train[2, :, 0]))
plt.subplot(2,2,2)
plt.hist2d(Y_train[:, 0], eval_out_train[2, :, 0], bins=64, norm=matplotlib.colors.LogNorm(), range=[[mi, ma], [mi, ma]], vmax=h2[0].max())
# plt.xlabel('Z_SPEC from redmapper')
plt.ylabel('Z from NN')
plt.colorbar()
plt.title('training set')
plt.plot([mi, ma], [mi, ma], color='red')

plt.tight_layout()
plt.savefig('versus_16.pdf')


plt.figure(figsize=(5,5))
err = clu_full_data['Z_LAMBDA_ERR'][ix_valid]
tp = (eval_out[2, :, 0] - Y_valid[:, 0])
plt.hist(
    tp,
    bins=32,
    alpha=0.5,
    label="from NN"
)
for p, ls in zip([2.5, 16, 50, 84, 97.5], ['-.','--','-','--','-.']):
    plt.axvline(np.percentile(tp, p), color='C0', ls=ls, alpha=0.5)
tp = (clu_full_data['Z_LAMBDA'][ix_valid] - Y_valid[:, 0])
plt.hist(
    tp,
    bins=32,
    alpha=0.5,
    label="from redmapper"
)
for p, ls in zip([2.5, 16, 50, 84, 97.5], ['-.','--','-','--','-.']):
    plt.axvline(np.percentile(tp, p), color='C1', ls=ls, alpha=0.5)
plt.xlabel('(z_inferred - zspec_redmapper)')
# plt.yscale('log')
plt.legend()
plt.tight_layout()
plt.savefig('error_16.pdf')

plt.figure(figsize=(5,5))
err = clu_full_data['Z_LAMBDA_ERR'][ix_valid]
tp = (eval_out[2, :, 0] - Y_valid[:, 0]) / err
plt.hist(
    tp,
    bins=32,
    alpha=0.5,
    label="from NN"
)
for p, ls in zip([2.5, 16, 50, 84, 97.5], ['-.','--','-','--','-.']):
    plt.axvline(np.percentile(tp, p), color='C0', ls=ls, alpha=0.5)
tp = (clu_full_data['Z_LAMBDA'][ix_valid] - Y_valid[:, 0]) / err
plt.hist(
    tp,
    bins=32,
    alpha=0.5,
    label="from redmapper"
)
for p, ls in zip([2.5, 16, 50, 84, 97.5], ['-.','--','-','--','-.']):
    plt.axvline(np.percentile(tp, p), color='C1', ls=ls, alpha=0.5)
plt.xlabel('(z_inferred - zspec_redmapper)/err_zlambda_redmapper')
# plt.yscale('log')
plt.tight_layout()
plt.savefig('rerror_16.pdf')

plt.figure(figsize=(5,5))
mi = min(np.min(Y_valid[:, 0]), np.min(eval_out[2, :, 0]))
ma = max(np.max(Y_valid[:, 0]), np.max(eval_out[2, :, 0]))
h,b1,b2 = np.histogram2d(Y_valid[:, 0], eval_out[2, :, 0], bins=64, range=[[mi, ma], [mi, ma]])
x, xerr, y, yerr, y2, yerr2 = [], [], [], [], [], []
for i in range(len(b1)-1):
    g = (Y_valid[:, 0] >= b1[i]) & (Y_valid[:, 0] <= b1[i+1])
    if g.sum() > 0:
        x.append(0.5 * (b1[i] + b1[i+1]))
        xerr.append(0.5 * (b1[i+1] - b1[i]))
        y.append(np.mean(eval_out[2, :, 0][g]))
        yerr.append(np.std(eval_out[2, :, 0][g]))
        y2.append(np.mean(clu_full_data['Z_LAMBDA'][ix_valid][g]))
        yerr2.append(np.std(clu_full_data['Z_LAMBDA'][ix_valid][g]))
plt.errorbar(x, y, xerr=xerr,yerr=yerr,fmt='.',ms=0,label='from NN',alpha=0.5)
plt.errorbar(x, y2, xerr=xerr,yerr=yerr2,fmt='.',ms=0,label='from redmapper',alpha=0.5)
# plt.errorbar(x, np.array(y)-np.array(x), xerr=xerr,yerr=yerr,fmt='+')
plt.plot([mi, ma], [mi, ma], color='red')
plt.xlabel('z_spec from redmapper')
# plt.ylabel('z from NN')
plt.ylabel('z inferred')
plt.title('validation set')
plt.axis('equal')
plt.legend()
plt.tight_layout()
plt.savefig('error_plot_16.pdf')

#########################

z_nn = Y_valid[:, 0]
z_spec = eval_out[2, :, 0]
z_lambda = clu_full_data['Z_LAMBDA'][ix_valid]

# resid
dz_nn = (z_nn - z_spec) / (1. + z_spec)
dz_lambda = (z_lambda - z_spec) / (1. + z_spec)
# pred bias
mdz_nn = np.mean(dz_nn)
mdz_lambda = np.mean(dz_lambda)
print(mdz_nn, mdz_lambda)
# MAD
MAD_nn = np.median(np.abs(dz_nn - mdz_nn))
MAD_lambda = np.median(np.abs(dz_lambda - mdz_lambda))
print(MAD_nn, MAD_lambda)
# MAD dev
sMAD_nn = 1.4826 * MAD_nn
sMAD_lambda = 1.4826 * MAD_lambda
print(sMAD_nn, sMAD_lambda)
# frac outliers
out_nn = np.abs(dz_nn) > 0.05
out_lambda = np.abs(dz_lambda) > 0.05
print(np.sum(out_nn) / 3139., np.sum(out_lambda) / 3139.)


plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.hist(dz_nn, bins=32, alpha=0.5, label='NN',density=True)
plt.hist(dz_lambda, bins=32, alpha=0.5, label='redMaPPer',density=True)
plt.legend()
plt.xlabel("(z - z_spec) / (1 + z_spec)")
plt.tight_layout()
plt.subplot(1,2,2)
plt.hist(dz_nn, bins=32, alpha=0.5, label='NN',density=True)
plt.hist(dz_lambda, bins=32, alpha=0.5, label='redMaPPer',density=True)
plt.legend()
plt.yscale('log')
plt.xlabel("(z - z_spec) / (1 + z_spec)")
plt.tight_layout()
plt.savefig("residuals.png")
plt.clf()

h, b = np.histogram(z_spec, bins=32)

x, y1, y2 = [], [], []
for i in range(len(h)):
    x.append(b[i])
    x.append(b[i+1])
    g = (z_spec >= b[i]) & (z_spec <= b[i+1])
    y1.append(np.mean(dz_nn[g]))
    y1.append(np.mean(dz_nn[g]))
    y2.append(np.mean(dz_lambda[g]))
    y2.append(np.mean(dz_lambda[g]))
plt.plot(x, y1, label='NN')
plt.plot(x, y2, label='redMaPPer')
plt.xlabel("z_spec")
plt.ylabel("Prediction bias")
plt.axhline(0., ls='--', color='black')
plt.legend()
plt.tight_layout()
plt.savefig("pred_bias.png")
plt.clf()

x, y1, y2 = [], [], []
for i in range(len(h)):
    x.append(b[i])
    x.append(b[i+1])
    g = (z_spec >= b[i]) & (z_spec <= b[i+1])
    mdz_nn = np.mean(dz_nn[g])
    mdz_lambda = np.mean(dz_lambda[g])
    MAD_nn = np.median(np.abs(dz_nn[g] - mdz_nn))
    MAD_lambda = np.median(np.abs(dz_lambda[g] - mdz_lambda))
    y1.append(MAD_nn)
    y1.append(MAD_nn)
    y2.append(MAD_lambda)
    y2.append(MAD_lambda)
plt.plot(x, y1, label='NN')
plt.plot(x, y2, label='redMaPPer')
plt.xlabel("z_spec")
plt.ylabel("MAD")
plt.axhline(0., ls='--', color='black')
plt.legend()
plt.tight_layout()
plt.savefig("MAD.png")
plt.clf()

x, y1, y2 = [], [], []
for i in range(len(h)):
    x.append(b[i])
    x.append(b[i+1])
    g = (z_spec >= b[i]) & (z_spec <= b[i+1])
    g2 = np.abs(dz_nn[g]) > 0.05
    y1.append(100. * g2.sum() / g.sum())
    y1.append(100. * g2.sum() / g.sum())
    g2 = np.abs(dz_lambda[g]) > 0.05
    y2.append(100. * g2.sum() / g.sum())
    y2.append(100. * g2.sum() / g.sum())
plt.semilogy(x, y1, label='NN')
plt.semilogy(x, y2, label='redMaPPer')
plt.xlabel("z_spec")
plt.ylabel("Percentage of outliers")
plt.legend()
plt.tight_layout()
plt.savefig("outliers.png")
plt.clf()
