import os, sys
import numpy as np
from tqdm import tqdm
from astropy.io import fits
from multiprocessing import Pool
from itertools import combinations, permutations
from multiprocessing import Pool
import emcee

# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt

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
n_feat = len(feat)

# Which labels to predicts
labs = ['X', 'Y', 'Z']
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
    g = mem_full_data['ID'] == clu_ids[i]
    n_mem = clu_num[i]
    n_perm = len(perm_dict[n_mem])
    X = np.zeros((n_perm, janossy_k * n_feat))
    for ix, f in enumerate(feat):
        base = mem_full_data[f][g]
        for j in range(n_perm):
            X[j, ix*janossy_k:(ix+1)*janossy_k] = base[perm_dict[n_mem][j]]
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

'''
tab = np.load('/home/users/ilic/ML/ML_clusters_project/invariant_NN/data.npz', allow_pickle=True)
X_train = tab['X_train']
X_valid = tab['X_valid']
Y_train = tab['Y_train']
Y_valid = tab['Y_valid']
'''


n_train = len(X_train)
n_valid = len(X_valid)

def act_id(x):
    return x

def act_relu(x):
    return 0.5 * (np.abs(x) + x)

# Define NN model
num_neurons_in_f = 3
# num_layers_in_rho = 1
# num_neurons_in_rho = 100

n_par = 0
ix_par = [0]
n_par += janossy_k * n_feat * num_neurons_in_f
ix_par += [ix_par[-1] + janossy_k * n_feat * num_neurons_in_f]
n_par += num_neurons_in_f
ix_par += [ix_par[-1] + num_neurons_in_f]
n_par += num_neurons_in_f * n_labs
ix_par += [ix_par[-1] + num_neurons_in_f * n_labs]
n_par += n_labs
ix_par += [ix_par[-1] + n_labs]

def lnlike(p):
    m1 = p[ix_par[0]:ix_par[1]].reshape(janossy_k * n_feat, num_neurons_in_f)
    b1 = p[ix_par[1]:ix_par[2]]
    m2 = p[ix_par[2]:ix_par[3]].reshape(num_neurons_in_f, n_labs)
    b2 = p[ix_par[3]:ix_par[4]]
    res = np.zeros((n_train, n_labs))
    for i in range(n_train):
        # X_train[i] => (n_perm, janossy_k * n_feat)
        # m1         => (janossy_k * n_feat, num_neurons_in_f)
        tmp = np.dot(X_train[i], m1) + b1 # (n_perm, num_neurons_in_f)
        # tmp = act_relu(tmp)
        tmp = np.mean(tmp, axis=0) # (num_neurons_in_f,)
        # m2         => (num_neurons_in_f, n_labs)
        res[i, :] = np.dot(tmp, m2) + b2 # (n_labs,)
    return -0.5 * np.sum((res - Y_train)**2./0.001**2.)


n_steps = 100
n_walkers = 2 * n_par
p_start = np.random.rand(n_walkers, n_par)
p_start = np.loadtxt("last_samp.txt")

backend = emcee.backends.HDFBackend('/home/users/ilic/ML/ML_clusters_project/invariant_NN/chain_emcee.h5')

# ix = backend.get_last_sample().log_prob.argmax()
# p_start = backend.get_last_sample().coords[ix, :] + p_start * 1e-4


pool = Pool(64)
sampler = emcee.EnsembleSampler(
    n_walkers,
    n_par,
    lnlike,
    # moves=emcee.moves.StretchMove(a=ini['stretch']),
    pool=pool,
    backend=backend,
    # blobs_dtype=blobs_dtype,
)

iter_num = 0
while True:
    iter_num += 1
    print("#### Starting iteration %s ####" % iter_num)
    ct = 0
    for result in sampler.sample(p_start, iterations=n_steps):
        ct += 1
        print('Current step : %s of %s' % (ct, n_steps))
    lim = np.percentile(sampler.lnprobability[:, -1], 25)
    g = sampler.lnprobability[:, -1] > lim
    l = g.sum()
    tmp = sampler.chain[g, -1, :][:(n_walkers-l), :]
    tmp =  np.random.rand(tmp.shape[0], tmp.shape[1])
    p_start = np.vstack(
        (sampler.chain[g, -1, :], sampler.chain[g, -1, :][:(n_walkers-l), :] + tmp * np.std(sampler.chain[g, -1, :], axis=0)*0.1)
    )


########################

import emcee
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

f = emcee.backends.HDFBackend('tmp.h5')
plt.semilogy(-f.get_log_prob());plt.show()
